import json
import random
import re
import sys
import zlib
from types import SimpleNamespace

import pytest

import rageval
from rageval import jobs, llm, settings, store, text
from rageval.datagen import freeze, generate, ingest, translate, validate
from rageval.schema import Evidence, Page, Question

EN = "abcdefghijklmnopqrstuvwxyz"
LANG = {"law/a.txt": "en", "law/b.txt": "en", "th/c.txt": "th"}


def words(seed, n=6):
    r = random.Random(seed)
    return " ".join("".join(r.choice(EN) for _ in range(r.randint(4, 8))) for _ in range(n))


def english(seed, paragraphs):
    return "\n\n".join(words(seed * 1000 + i, 80) + "." for i in range(paragraphs))


def thai(paragraphs):
    return "\n\n".join(" ".join(f"ข้อ {i}-{j} ผู้ใดกระทำการตามที่กฎหมายนี้ห้ามต้องระวางโทษตามที่กำหนดไว้" for j in range(12))
                       for i in range(paragraphs))


FILES = {"law/a.txt": english(1, 12), "law/b.txt": english(2, 10), "th/c.txt": thai(12)}
PAGES = {doc: dict(enumerate(text.pseudo_pages(body), 1)) for doc, body in FILES.items()}


def quote(doc, page):
    return PAGES[doc][page][:220]


@pytest.fixture
def material(root):
    for doc, body in FILES.items():
        path = root / "material" / doc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return root / "material"


@pytest.fixture(autouse=True)
def env(monkeypatch):
    for k in ("API_KEY", "URL", "MODEL"):
        monkeypatch.setenv(f"GEN_AI_{k}", "x")


# ---- the fake model ----

class Fake:
    """Stands in for llm.call. Behaviour is steered by tags written into question text."""

    def __init__(self, root):
        self.root = root
        self.tasks: list[str] = []
        self.generate_calls: list[tuple[str, str]] = []
        self.tags: dict[tuple[str, str], list[str]] = {}
        self.follow_tags: list[str] = []
        self.bad: set[tuple[str, str]] = set()
        self.flaky: dict[str, int] = {}  # source id -> translations to spoil
        self.n = 0
        self.last: dict[str, str] = {}
        self.stop_after: int | None = None
        self.lang: dict[str, str] = {}  # doc -> language reply, default LANG
        self.broken: set[str] = set()  # source ids whose translation comes back malformed

    def __call__(self, role, prompt):
        m = re.search(r"^Task: (\w+)", prompt, re.M)
        task = m[1] if m else "closed_book"
        self.tasks.append(task)
        if self.stop_after is not None and len(self.tasks) == self.stop_after:
            for d in (self.root / "datasets").iterdir():
                (d / "job.stop").touch()
        return getattr(self, task)(prompt)

    def closed_book(self, prompt):
        return "an answer"

    def generate(self, prompt):
        doc = re.search(r"^Document name: (.+)$", prompt, re.M)[1]
        qtype = re.search(r"^Question type: (\w+)$", prompt, re.M)[1]
        n = int(re.search(r"^Number of questions: (\d+)$", prompt, re.M)[1])
        self.generate_calls.append((doc, qtype))
        if (doc, qtype) in self.bad:
            return "this is not json"
        slots = {int(a): (int(b), int(c)) for a, b, c in re.findall(r"^Slot (\d+): pages (\d+) to (\d+)$", prompt, re.M)}
        items = []
        for i in range(n):
            tags = self.tags.get((doc, qtype), [])
            items.append(self.item(doc, qtype, i + 1, slots.get(i + 1), tags.pop(0) if tags else ""))
        return json.dumps({"items": items})

    def item(self, doc, qtype, slot, rng, tag):
        self.n += 1
        subject = doc.split("/")[0]
        question = words(5000 + self.n) + (f" {tag}" if tag else "")
        if tag == "DUP":
            question = self.last[subject]
        elif not tag:
            self.last[subject] = question
        pages = sorted(PAGES[doc])
        lo, hi = rng or (1, 1)
        outside = next((p for p in pages if p < lo or p > hi), None)
        page = outside if tag == "SPREAD" and outside else lo
        if tag == "SPREAD" and not outside:
            slot = 99  # no page outside the range: break the slot rule instead
        evidence = [{"page": page, "quote": quote(doc, page)}]
        if qtype == "multi_hop":
            other = next(p for p in pages if p != page)
            evidence.append({"page": other, "quote": quote(doc, other)})
        if tag == "QUOTE":
            evidence = [{"page": page, "quote": "z" * 150}]
        if tag == "STRUCT":
            evidence = []
        if tag == "NULLPAGE":
            evidence = [{"page": None, "quote": "x"}]
        if qtype == "unanswerable":
            return {"slot": slot, "question": question, "gold_evidence": []}
        return {"slot": slot, "question": question, "gold_evidence": evidence}

    def question_check(self, prompt):
        typed, resolved = re.search(r"types it:\n(.*?)\n\nQuestion rewritten.*?:\n(.*?)\n\nEvidence", prompt, re.S).groups()
        quotes = len(re.findall(r"^Quote \d+:", prompt, re.M))
        return json.dumps({
            "question_standalone": typed == resolved or "ISSTANDALONE" in typed,
            "resolved_standalone": "NOTSTANDALONE" not in resolved,
            "answerable": "NOTCOVERED" not in typed,
            "quotes_used": [True] + [False if "UNUSED" in typed else True] * (quotes - 1),
        })

    def answer(self, prompt):
        q = re.search(r"Question:\n(.*?)\n\nQuestion rewritten", prompt, re.S)[1]
        if "BADANSWER" in q:
            return "this is not json"
        return json.dumps({"reference_answer": f"answer to {q[:12]}", "required_facts": [f"fact of {q[:12]}"]})

    def answerable(self, prompt):
        typed, resolved = re.search(r"types it:\n(.*?)\n\nQuestion rewritten.*?:\n(.*?)\n\nGold", prompt, re.S).groups()
        kp = len(re.findall(r"^Key point \d+:", prompt, re.M))
        quotes = len(re.findall(r"^Quote \d+:", prompt, re.M))
        return json.dumps({
            "question_standalone": typed == resolved or "ISSTANDALONE" in typed,
            "resolved_standalone": "NOTSTANDALONE" not in resolved,
            "answer": "a",
            "key_points_covered": [False if "NOTCOVERED" in typed else True] * kp,
            "quotes_used": [True] + [False if "UNUSED" in typed else True] * (quotes - 1),
        })

    def unanswerable(self, prompt):
        q = re.search(r"Question:\n(.*?)\n\nDecide", prompt, re.S)[1]
        return json.dumps({"standalone": "NOTSTANDALONE" not in q, "answer_found": "FOUND" in q, "quote": "",
                           "closest_passage": "closest text"})

    def follow_up(self, prompt):
        doc = re.search(r"^Document name: (.+)$", prompt, re.M)[1]
        counts = {sc: int(n) for sc, n in re.findall(r"^Scenario (\w+): (\d+)$", prompt, re.M)}
        subject = doc.split("/")[0]
        cands = store.read_jsonl(next((self.root / "datasets").glob("*/candidates.jsonl")), Question)
        singles = [q for q in cands if q.subject == subject and not q.is_follow_up and q.answerable]
        mine = [q for q in singles if q.gold_evidence[0].doc == doc]
        theirs = [q for q in singles if q.gold_evidence[0].doc != doc]
        items = []
        for sc, n in counts.items():
            for _ in range(n):
                self.n += 1
                tag = self.follow_tags.pop(0) if self.follow_tags else ""
                first = theirs[0] if sc == "diff_topic" else mine[0]
                used = {e.page for e in first.gold_evidence}
                page = next(iter(used)) if sc == "same" else next(p for p in sorted(PAGES[doc]) if p not in used)
                if tag == "BADSCENARIO":
                    page = next(iter(used))
                body = words(7000 + self.n)
                resolved = f"{body} {tag}".strip()
                typed = f"and then {words(9000 + self.n, 2)} {tag}".strip() if sc == "same" else resolved
                items.append({"scenario": sc, "previous_id": first.id, "question": typed, "resolved_question": resolved,
                              "question_type": "fact",
                              "gold_evidence": [{"page": page, "quote": quote(doc, page)}]})
        return json.dumps({"items": items})

    def translate(self, prompt):
        lang = re.search(r'code "(\w+)"', prompt)[1]
        items = json.loads(prompt.split("Items:\n", 1)[1].split("\n\nReturn JSON")[0])
        out = []
        for it in items:
            spoil = self.flaky.get(it["id"], 0) > 0
            self.flaky[it["id"]] = self.flaky.get(it["id"], 0) - 1
            mark = "BAD " if spoil else ""
            if it["id"] in self.broken:
                out.append({"id": it["id"]})
                continue
            out.append({"id": it["id"], "question": f"{mark}[{lang}] {it['question']}",
                        "resolved_question": f"{mark}[{lang}] {it['resolved_question']}",
                        "reference_answer": f"[{lang}] {it['reference_answer']}",
                        "gold_ans": [f"[{lang}] {a}" for a in it["gold_ans"]]})
        return json.dumps({"items": out})

    def translation_check(self, prompt):
        items = json.loads(prompt.split("Items (source and translation):\n", 1)[1].split("\n\nReturn JSON")[0])
        return json.dumps({"items": [
            {"id": it["id"], "question_same": "BAD" not in it["translation"]["question"],
             "key_points_same": [True] * len(it["source"]["gold_ans"])} for it in items]})


@pytest.fixture
def fake(root, monkeypatch):
    f = Fake(root)
    monkeypatch.setattr(llm, "call", f)
    return f


@pytest.fixture
def fake_judge(monkeypatch):
    mod = SimpleNamespace(answer_check=lambda q, answer, cache_path=None: SimpleNamespace(
        answer_recall=1.0 if "KNOWN" in q.question else 0.0))
    monkeypatch.setitem(sys.modules, "rageval.judge", mod)
    monkeypatch.setattr(rageval, "judge", mod, raising=False)
    return mod


def small_settings(**kw):
    base = dict(questions_per_subject={"fact": 4, "definition": 0, "comparison": 0, "multi_hop": 2, "exception": 0,
                                       "unanswerable": 2},
                follow_ups_per_subject={"same": 1, "similar": 1, "diff_topic": 1}, overgenerate=1.0)
    settings.save(settings.Settings(**{**base, **kw}))


# ---- ingest ----

def test_ingest_txt_pages(material, root):
    pages = ingest.ingest(root / "datasets" / "d")
    assert {p.doc for p in pages} == set(FILES)
    a = [p for p in pages if p.doc == "law/a.txt"]
    assert [p.page for p in a] == list(range(1, len(PAGES["law/a.txt"]) + 1))
    assert a[0].text == PAGES["law/a.txt"][1] and all(p.subject == "law" for p in a)
    assert store.read_jsonl(root / "datasets" / "d" / "pages.jsonl", Page) == pages


def test_ingest_one_subject_keeps_the_others(material, root):
    d = root / "datasets" / "d"
    ingest.ingest(d)
    only = ingest.ingest(d, "th")
    assert {p.subject for p in only} == {"th"}
    assert {p.subject for p in store.read_jsonl(d / "pages.jsonl", Page)} == {"law", "th"}


def test_ingest_file_without_text_names_the_file(material, root):
    (material / "law" / "empty.txt").write_text("  \n\n ", encoding="utf-8")
    with pytest.raises(SystemExit, match="empty.txt"):
        ingest.ingest(root / "datasets" / "d")


def test_ingest_file_over_the_token_limit_names_the_file(material, root):
    small_settings(gen_max_input_tokens=1000)
    (material / "big").mkdir()
    (material / "big" / "big.txt").write_text("ก" * 1500, encoding="utf-8")
    with pytest.raises(SystemExit, match="big.txt"):
        ingest.ingest(root / "datasets" / "d", "big")


# ---- split and page ranges ----

def test_split_is_proportional_with_ties_by_name():
    assert generate.split(10, {"a": 1, "b": 1, "c": 1}) == {"a": 4, "b": 3, "c": 3}
    assert generate.split(7, {"a": 3, "b": 1, "c": 0}) == {"a": 5, "b": 2}
    assert generate.split(0, {"a": 1}) == {} and generate.split(5, {"a": 0}) == {}


def test_a_document_without_eligible_pages_gets_nothing():
    docs = {"s/a.txt": [Page(subject="s", doc="s/a.txt", page=1, text="long " * 60)],
            "s/b.txt": [Page(subject="s", doc="s/b.txt", page=1, text="short")]}
    weights = generate._weights(docs)
    assert weights["s/b.txt"] == 0 and generate.split(4, weights) == {"s/a.txt": 4}


def test_page_ranges_divide_the_eligible_pages():
    assert generate._ranges([1, 2, 3, 4, 5, 6, 7], 3) == [(1, 2), (3, 4), (5, 7)]
    assert generate._ranges([4, 9], 3) == [(4, 4), (4, 4), (9, 9)]


# ---- check_row, difficulty ----

P1, P2, P3 = (words(10 + i, 60) for i in range(3))
PMAP = {("law/a.txt", 1): P1, ("law/a.txt", 2): P2, ("law/b.txt", 1): P3}


def ev(doc="law/a.txt", page=1, body=None):
    return Evidence(doc=doc, page=page, quote=body or PMAP[(doc, page)][:200])


def mk(qid="law-001-en", qtype="fact", **kw):
    base = dict(id=qid, base_id=qid.rsplit("-", 1)[0], subject="law", lang="en", source_lang="en", question=f"{words(zlib.crc32(qid.encode()) % 99999)}?",
                question_type=qtype, answerable=qtype != "unanswerable", status="candidate")
    base.setdefault("resolved_question", kw.get("question") or base["question"])
    if qtype != "unanswerable":
        base.update(gold_ans=["kp"], reference_answer="ref", answer_status="approved", gold_evidence=[ev()])
    return Question(**{**base, **kw})


def test_check_row_passes_each_type():
    assert validate.check_row(mk(), PMAP, []) is None
    assert validate.check_row(mk(qtype="multi_hop", gold_evidence=[ev(page=1), ev(page=2)]), PMAP, []) is None
    assert validate.check_row(mk(qtype="unanswerable"), PMAP, []) is None


def test_check_row_quote_failures():
    assert validate.check_row(mk(gold_evidence=[ev(body="z" * 150)]), PMAP, []).startswith("quote")
    assert validate.check_row(mk(gold_evidence=[ev(body=P1[:60])]), PMAP, []).startswith("quote")  # too short
    assert validate.check_row(mk(gold_evidence=[ev(page=2, body=P1[:200])]), PMAP, []).startswith("quote")  # wrong page
    assert validate.check_row(mk(gold_evidence=[Evidence(doc="law/a.txt", page=9, quote=P1[:200])]), PMAP, []).startswith("quote")


def test_check_row_structure_per_type():
    one_page = mk(qtype="multi_hop", gold_evidence=[ev(page=1), ev(page=1, body=P1[200:400])])
    assert validate.check_row(one_page, PMAP, []).startswith("structure")
    assert validate.check_row(mk(gold_evidence=[]), PMAP, []).startswith("structure")  # answerable, no evidence
    assert validate.check_row(mk(qtype="unanswerable", gold_evidence=[ev()]), PMAP, []).startswith("structure")
    assert validate.check_row(mk(gold_ans=["a"] * 5), PMAP, []).startswith("structure")


def test_check_row_duplicate():
    first = mk("law-001-en", question="what is the rule on late filing of reports?")
    near = mk("law-002-en", question="what is the rule on late filing of reports")
    assert validate.check_row(near, PMAP, [first]).startswith("duplicate")
    assert validate.check_row(mk("law-003-en"), PMAP, [first]) is None


def follow(scenario, page, doc="law/a.txt", **kw):
    typed = "and when did it end?" if scenario == "same" else "a standalone question about something else entirely"
    resolved = "when did the first thing end?" if scenario == "same" else typed
    return mk("law-009-en", question=typed, resolved_question=resolved, previous_id="law-001-en", scenario=scenario,
              gold_evidence=[ev(doc, page)], **kw)


FIRST = mk("law-001-en", question="what is the first thing about the first rule?")


@pytest.mark.parametrize("scenario,good,bad", [
    ("same", ("law/a.txt", 1), ("law/a.txt", 2)),
    ("similar", ("law/a.txt", 2), ("law/a.txt", 1)),
    ("diff_topic", ("law/b.txt", 1), ("law/a.txt", 2)),
])
def test_check_row_scenarios(scenario, good, bad):
    assert validate.check_row(follow(scenario, good[1], good[0]), PMAP, [FIRST]) is None
    reason = validate.check_row(follow(scenario, bad[1], bad[0]), PMAP, [FIRST])
    assert reason.startswith("structure") and scenario in reason
    assert validate.check_row(follow(scenario, bad[1], bad[0]), PMAP, [FIRST], strict_scenario=False) is None


def test_check_row_follow_up_rules():
    same = follow("same", 1)
    assert validate.check_row(same.model_copy(update={"resolved_question": same.question}), PMAP, [FIRST]).startswith("structure")
    similar = follow("similar", 2)
    assert validate.check_row(similar.model_copy(update={"resolved_question": "different"}), PMAP, [FIRST]).startswith("structure")
    assert validate.check_row(follow("same", 1), PMAP, []).startswith("structure")  # first turn not found
    unanswerable = mk("law-009-en", qtype="unanswerable", previous_id="law-001-en", scenario="similar")
    assert validate.check_row(unanswerable, PMAP, [FIRST]).startswith("structure")


def test_difficulty():
    assert validate.difficulty(mk(gold_evidence=[ev(page=1), ev(page=2)])) == "hard"
    assert validate.difficulty(mk(resolved_question=P1[:100])) == "easy"
    assert validate.difficulty(mk(resolved_question="completely unrelated wording here")) == "medium"


# ---- judge_checks ----

def test_judge_checks_removes_unused_evidence_and_sets_closest_passage(fake):
    docs = {"law/a.txt": [Page(subject="law", doc="law/a.txt", page=1, text=P1)]}
    q = mk(question="plain question UNUSED", gold_evidence=[ev(page=1), ev(page=2)])
    out, failed = validate.judge_checks(q, docs)
    assert failed is None and [e.page for e in out.gold_evidence] == [1]
    out, failed = validate.judge_checks(mk(qtype="unanswerable"), docs)
    assert failed is None and out.closest_passage == "closest text"


@pytest.mark.parametrize("tag,check", [("NOTCOVERED", "answerable"), ("NOTSTANDALONE", "standalone")])
def test_judge_checks_failures(fake, tag, check):
    docs = {"law/a.txt": [Page(subject="law", doc="law/a.txt", page=1, text=P1)]}
    assert validate.judge_checks(mk(question=f"plain question {tag}"), docs) == (None, check)
    assert validate.judge_checks(mk(qtype="unanswerable", question="x FOUND"), docs) == (None, "unanswerable")
    assert validate.judge_checks(mk(qtype="unanswerable", question="x NOTSTANDALONE"), docs) == (None, "standalone")


def test_judge_checks_one_unanswerable_call_per_document(fake):
    docs = {d: [Page(subject="law", doc=d, page=1, text=P1)] for d in ("law/a.txt", "law/b.txt")}
    validate.judge_checks(mk(qtype="unanswerable"), docs)
    assert fake.tasks == ["unanswerable", "unanswerable"]


# ---- generate ----

def candidates(root):
    return store.read_jsonl(next((root / "datasets").glob("*/candidates.jsonl")), Question)


def build_of(root):
    return store.read_json(next((root / "datasets").glob("*/build.json")))


def test_generate_end_to_end_counts_every_drop(material, root, fake):
    small_settings()
    fake.tags[("law/a.txt", "fact")] = ["NOTCOVERED", "QUOTE", "STRUCT", "SPREAD"]
    fake.tags[("law/b.txt", "fact")] = ["NOTSTANDALONE", "", "DUP", ""]
    fake.tags[("law/a.txt", "unanswerable")] = ["FOUND"]
    fake.tags[("th/c.txt", "unanswerable")] = ["NOTSTANDALONE"]
    fake.follow_tags = ["ISSTANDALONE", "BADSCENARIO"]
    fake.bad = {("th/c.txt", "multi_hop")}
    dataset = generate.run()

    rows = candidates(root)
    assert dataset in {p.name for p in (root / "datasets").iterdir()}
    assert len({q.id for q in rows}) == len(rows) and all(re.fullmatch(r"(law|th)-\d{3}-en", q.id) for q in rows)
    assert all(q.status == "candidate" and q.hard_negative is None for q in rows)
    assert {q.lang for q in rows} == {"en"} and all(q.base_id == q.id[:-3] and not q.gold_ans for q in rows)
    assert all(q.answer_status == "none" and q.reference_answer is None for q in rows)
    assert all(q.difficulty for q in rows if q.answerable)

    dropped = build_of(root)["dropped"]
    for check in ("quote", "structure", "spread", "answerable", "standalone", "duplicate", "unanswerable", "bad_reply"):
        assert check in dropped, check
    assert dropped["bad_reply"]["multi_hop"] == 3  # the same batch in all three rounds
    assert dropped["unanswerable"] == {"unanswerable": 1} and dropped["duplicate"] == {"fact": 1}
    assert dropped["standalone"] == {"fact": 1, "unanswerable": 1, "same": 1}  # a `same` follow-up that made sense alone
    assert dropped["structure"] == {"fact": 1, "similar": 1}  # missing evidence; a scenario rule broken
    assert dropped["quote"] == {"fact": 1} and dropped["spread"] == {"fact": 1} and dropped["answerable"] == {"fact": 1}

    follow_ups = [q for q in rows if q.is_follow_up]
    assert {q.scenario for q in follow_ups} <= {"same", "similar", "diff_topic"}
    for q in follow_ups:
        first = next(r for r in rows if r.id == q.previous_id)
        assert validate.scenario_ok(q, first) and not first.is_follow_up
        assert (q.question != q.resolved_question) == (q.scenario == "same")
    # a one-document subject cannot have diff_topic follow-ups; the shortfall says so
    shortfall = build_of(root)["shortfall"]
    assert shortfall["th"] == {"multi_hop": 2, "diff_topic": 1} and shortfall["law"] == {}  # refills made up the fact drops
    assert "diff_topic" in {q.scenario for q in follow_ups if q.subject == "law"}


def test_generate_refills_after_drops(material, root, fake):
    small_settings(questions_per_subject={"fact": 4, "definition": 0, "comparison": 0, "multi_hop": 0, "exception": 0,
                                          "unanswerable": 0},
                   follow_ups_per_subject={"same": 0, "similar": 0, "diff_topic": 0})
    fake.tags[("law/a.txt", "fact")] = ["QUOTE"] * 3
    fake.tags[("law/b.txt", "fact")] = ["QUOTE"] * 3
    generate.run("law")
    law = [q for q in candidates(root) if q.subject == "law"]
    assert len(law) == 4
    assert build_of(root)["shortfall"]["law"] == {}
    assert {"single:0", "single:1"} <= {name.rsplit(":", 1)[0] for name in build_of(root)["plan"]}


def test_generate_second_call_makes_no_model_calls(material, root, fake):
    small_settings()
    dataset = generate.run()
    calls, before = len(fake.tasks), candidates(root)
    assert generate.run(dataset_id=dataset) == dataset
    assert len(fake.tasks) == calls and candidates(root) == before


def test_generate_resumes_after_stop_without_repeating_batches(material, root, fake, monkeypatch):
    small_settings()
    baseline = Fake(root)
    monkeypatch.setattr(llm, "call", baseline)
    generate.run()
    full = len(baseline.generate_calls) + len([t for t in baseline.tasks if t == "follow_up"])
    for d in (root / "datasets").iterdir():
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    monkeypatch.setattr(llm, "call", fake)
    fake.stop_after = 6
    with pytest.raises(jobs.Stopped):
        generate.run()
    assert (build_of(root)["done"])  # finished batches were recorded
    fake.stop_after = None
    generate.run()
    total = len(fake.generate_calls) + len([t for t in fake.tasks if t == "follow_up"])
    assert full <= total <= full + 1  # at most the batch that was interrupted is asked again
    ids = [q.id for q in candidates(root)]
    assert len(ids) == len(set(ids))


def test_generate_refuses_a_frozen_dataset(material, root, fake):
    d = root / "datasets" / "x"
    d.mkdir(parents=True)
    (d / "questions.jsonl").write_text("")
    with pytest.raises(SystemExit, match="frozen"):
        generate.run(dataset_id="x")


# ---- translate, freeze ----

def row(qid, qtype="fact", subject="law", **kw):
    return mk(qid, qtype, subject=subject, status="approved", **kw)


def follow_row(qid, first, scenario="similar", **kw):
    text_ = f"{qid} {words(zlib.crc32(qid.encode()) % 77777)}"
    return mk(qid, question=f"and {qid}?" if scenario == "same" else text_, resolved_question=text_, previous_id=first,
              scenario=scenario, status="approved", **kw)


def dataset(root, rows, name="ds"):
    d = root / "datasets" / name
    store.write_jsonl(d / "candidates.jsonl", rows)
    return name


def test_translate_creates_variants_idempotently(root, fake):
    small_settings(extra_langs=["th", "fr"])
    rows = [row("law-001-en"), row("law-002-en", lang="th", source_lang="th"), follow_row("law-003-en", "law-001-en"),
            follow_row("law-004-en", "law-002-en", "same"), row("law-005-en").model_copy(update={"status": "candidate"})]
    ds = dataset(root, rows)
    assert len(translate.missing(ds)) == 7  # th: the three English-source questions; fr: all four approved ones
    first = translate.run(ds)
    out = {q.id: q for q in store.read_jsonl(root / "datasets" / ds / "candidates.jsonl", Question)}
    assert first == {"created": 7, "dropped": 0} and len(out) == len(rows) + 7
    assert "law-001-th" in out and "law-001-fr" in out and "law-002-fr" in out
    assert "law-002-th" not in out  # already Thai
    assert "law-005-th" not in out  # not approved
    v = out["law-003-th"]
    assert v.previous_id == "law-001-th" and v.base_id == "law-003" and v.lang == "th" and v.source_lang == "en"
    assert v.question == v.resolved_question and v.status == "approved" and not v.translation_reviewed
    assert out["law-004-fr"].previous_id == "law-002-fr"
    assert out["law-004-th"].previous_id == "law-002-en"  # its first turn is already Thai
    assert out["law-004-th"].question != out["law-004-th"].resolved_question
    assert v.gold_evidence == out["law-003-en"].gold_evidence
    assert translate.missing(ds) == []
    calls = len(fake.tasks)
    assert translate.run(ds) == {"created": 0, "dropped": 0} and len(fake.tasks) == calls


def test_translate_retries_once_then_drops(root, fake):
    small_settings(extra_langs=["fr"])
    ds = dataset(root, [row("law-001-en"), row("law-002-en"), row("law-003-en"), follow_row("law-004-en", "law-002-en")])
    fake.flaky = {"law-001-en": 1, "law-002-en": 5}
    result = translate.run(ds)
    ids = {q.id for q in store.read_jsonl(root / "datasets" / ds / "candidates.jsonl", Question)}
    assert "law-001-fr" in ids and "law-002-fr" not in ids and "law-003-fr" in ids
    assert "law-004-fr" not in ids  # its first turn has no French variant
    assert result["dropped"] == 2 and result["created"] == 2
    assert translate.missing(ds) == []  # dropped variants are reported, not missing
    # editing the source question gives it a fresh try
    rows = store.read_jsonl(root / "datasets" / ds / "candidates.jsonl", Question)
    rows = [r.model_copy(update={"question": "edited " + r.question}) if r.id == "law-002-en" else r for r in rows]
    store.write_jsonl(root / "datasets" / ds / "candidates.jsonl", rows)
    assert ("law-002", "fr") in translate.missing(ds)


def test_drop_variants():
    first = mk("law-001-en")
    rows = [first, first.model_copy(update={"id": "law-001-th", "lang": "th"}), mk("law-002-en")]
    assert [r.id for r in translate.drop_variants(rows, "law-001")] == ["law-001-en", "law-002-en"]


def test_select_respects_targets_and_keeps_first_turns():
    s = settings.Settings(questions_per_subject={"fact": 2, "definition": 1, "comparison": 0, "multi_hop": 0,
                                                  "exception": 0, "unanswerable": 0},
                          follow_ups_per_subject={"same": 0, "similar": 1, "diff_topic": 0})
    rows = [row("law-001-en"), row("law-002-en"), row("law-003-en"), row("law-004-en", "definition"), row("law-005-en", "definition"),
            row("law-006-en"), follow_row("law-007-en", "law-006-en"), follow_row("law-008-en", "law-001-en"),
            follow_row("law-009-en", "law-010-en"), row("law-010-en").model_copy(update={"status": "candidate"}),
            row("law-011-en").model_copy(update={"status": "rejected"})]
    ids = [q.id for q in freeze.select(rows, s)]
    # similar target is 1: law-007-en wins and brings its first turn law-006-en (counted as a fact), so one more fact fits;
    # law-009-en's first turn is not approved, and rejected or candidate rows never count
    assert ids == ["law-001-en", "law-004-en", "law-006-en", "law-007-en"]


def test_select_takes_variants_of_selected_questions_only():
    s = settings.Settings(questions_per_subject={"fact": 1, "definition": 0, "comparison": 0, "multi_hop": 0,
                                                  "exception": 0, "unanswerable": 0},
                          follow_ups_per_subject={"same": 0, "similar": 1, "diff_topic": 0})
    first, extra, fu = row("law-001-en"), row("law-002-en"), follow_row("law-003-en", "law-001-en")

    def variant(q, prev=None, **kw):
        return q.model_copy(update={"id": f"{q.id[:-3]}-th", "lang": "th", "previous_id": prev or q.previous_id, **kw})

    rows = [first, extra, fu, variant(first), variant(extra), variant(fu, "law-001-th"),
            variant(fu, "law-001-fr", id="law-003-fr"), variant(first, id="law-001-de", status="rejected")]
    # law-002-en is over the fact target, so its variant stays out; law-003-fr points at a first turn that is not frozen
    assert [q.id for q in freeze.select(rows, s)] == ["law-001-en", "law-003-en", "law-001-th", "law-003-th"]


def approved_pair(root):
    rows = [row("law-001-en", question="KNOWN fact one", resolved_question="KNOWN fact one"),
            row("law-002-en"), row("law-003-en", "unanswerable")]
    return dataset(root, rows)


def test_freeze_refuses_while_variants_are_missing(root, fake, fake_judge):
    small_settings(extra_langs=["th"])
    ds = approved_pair(root)
    with pytest.raises(SystemExit, match="translate"):
        freeze.run(ds)
    assert not store.is_frozen(ds)


def test_freeze_flags_closed_book_and_writes_a_stable_manifest(root, fake, fake_judge):
    small_settings(extra_langs=["th"])
    ds = approved_pair(root)
    translate.run(ds)
    m = freeze.run(ds)
    out = {q.id: q for q in store.read_jsonl(root / "datasets" / ds / "questions.jsonl", Question)}
    assert out["law-001-en"].closed_book and not out["law-002-en"].closed_book and not out["law-003-en"].closed_book
    assert out["law-001-th"].closed_book and not out["law-002-th"].closed_book  # variants inherit
    assert m["sha256"] == store.sha256(root / "datasets" / ds / "questions.jsonl")
    assert store.read_json(root / "datasets" / ds / "manifest.json") == m
    assert m["translations"] == 3 and m["translations_reviewed"] == 0 and m["languages"] == ["en", "th"]
    assert m["counts"] == {"law": {"fact": 2, "unanswerable": 1}}
    assert m["settings"]["extra_langs"] == ["th"] and set(m["prompt_versions"]) >= {"generate", "translate"}
    assert m["generator_model"] == "x"
    closed_book_calls = fake.tasks.count("closed_book")
    assert closed_book_calls == 2  # answerable source-language questions only
    with pytest.raises(SystemExit, match="frozen"):
        freeze.run(ds)


def test_freeze_hash_is_stable_across_identical_inputs(root, fake, fake_judge):
    a = freeze.run(approved_pair(root))["sha256"]
    b = freeze.run(dataset(root, store.read_jsonl(root / "datasets" / "ds" / "candidates.jsonl", Question), "ds2"))["sha256"]
    assert a == b


def test_interrupted_freeze_leaves_no_questions_file_and_resumes(root, fake, fake_judge):
    ds = approved_pair(root)
    fake.stop_after = 1
    with pytest.raises(jobs.Stopped):
        freeze.run(ds)
    d = root / "datasets" / ds
    assert not (d / "questions.jsonl").exists() and not (d / "manifest.json").exists() and not store.is_frozen(ds)
    fake.stop_after = None
    freeze.run(ds)
    assert store.is_frozen(ds)
    assert fake.tasks.count("closed_book") == 2  # the question checked before the stop was not asked again


def test_duplicate_as_draft(root, fake, fake_judge):
    ds = approved_pair(root)
    (root / "datasets" / ds / "pages.jsonl").write_text("", encoding="utf-8")
    freeze.run(ds)
    new = freeze.duplicate_as_draft(ds)
    assert new != ds and not store.is_frozen(new)
    frozen = store.read_jsonl(root / "datasets" / ds / "questions.jsonl", Question)
    assert store.read_jsonl(root / "datasets" / new / "candidates.jsonl", Question) == frozen
    assert (root / "datasets" / new / "pages.jsonl").exists()
    with pytest.raises(SystemExit):
        freeze.duplicate_as_draft(new)


def test_generate_plans_again_when_a_subject_is_added(root, monkeypatch):
    """A plan made for one set of documents must not hide documents added to the dataset later."""
    from rageval.datagen import generate
    calls = []
    monkeypatch.setattr(generate, "_batch_single", lambda st, key, subject, *a: (calls.append(subject), generate._commit(st, key, [])))
    monkeypatch.setattr(generate, "_batch_follow", lambda st, key, subject, *a: generate._commit(st, key, []))
    for sub in ("aaa", "bbb"):
        (root / "material" / sub).mkdir(parents=True)
        (root / "material" / sub / "d.txt").write_text("word " * 400, encoding="utf-8")
    ds = generate.run("aaa")
    assert set(calls) == {"aaa"}
    calls.clear()
    assert generate.run("bbb", dataset_id=ds) == ds
    assert set(calls) == {"bbb"}


# ---- review fixes ----

ONE_TYPE = dict(questions_per_subject={"fact": 3, "definition": 0, "comparison": 0, "multi_hop": 0, "exception": 0,
                                       "unanswerable": 0},
                follow_ups_per_subject={"same": 0, "similar": 0, "diff_topic": 0})


def test_one_malformed_item_does_not_discard_the_batch(material, root, fake):
    small_settings(**ONE_TYPE)
    fake.tags[("th/c.txt", "fact")] = ["NULLPAGE"]
    generate.run("th")
    th = candidates(root)
    assert len(th) == 3 and build_of(root)["dropped"]["bad_reply"] == {"fact": 1}  # the other items survived and refilled


def test_one_malformed_translation_fails_only_that_question(root, fake):
    small_settings(extra_langs=["fr"])
    ds = dataset(root, [row("law-001-en"), row("law-002-en")])
    fake.broken = {"law-001-en"}
    assert translate.run(ds) == {"created": 1, "dropped": 1}


def test_crash_while_committing_never_reuses_an_id_or_keeps_half_a_batch(material, root, fake, monkeypatch):
    small_settings(**ONE_TYPE)
    real, seen = store.append_jsonl, []

    def dying(path, row):
        if seen:
            raise RuntimeError("killed")
        seen.append(row.id)
        real(path, row)

    monkeypatch.setattr(store, "append_jsonl", dying)
    with pytest.raises(RuntimeError):
        generate.run("th")
    monkeypatch.setattr(store, "append_jsonl", real)
    assert build_of(root)["pending"]["ids"][0] == seen[0]
    generate.run("th")
    ids = [q.id for q in candidates(root)]
    assert len(ids) == len(set(ids)) == 3 and seen[0] not in ids  # the half-written batch was dropped and redone
    assert "pending" not in build_of(root)


def test_closed_book_with_an_unusable_judge_reply_still_freezes(root, fake, monkeypatch):
    def check(q, answer, cache_path=None):
        if "BADJUDGE" in q.question:
            raise llm.BadReply("no json")
        return SimpleNamespace(answer_recall=1.0)

    mod = SimpleNamespace(answer_check=check)
    monkeypatch.setitem(sys.modules, "rageval.judge", mod)
    monkeypatch.setattr(rageval, "judge", mod, raising=False)
    ds = dataset(root, [row("law-001-en", question="BADJUDGE one", resolved_question="BADJUDGE one"), row("law-002-en")])
    m = freeze.run(ds)
    out = {q.id: q for q in store.read_jsonl(root / "datasets" / ds / "questions.jsonl", Question)}
    assert not out["law-001-en"].closed_book and out["law-002-en"].closed_book and m["closed_book_unchecked"] == 1


def test_translate_refuses_a_frozen_dataset(root, fake, fake_judge):
    small_settings()
    ds = approved_pair(root)
    freeze.run(ds)
    small_settings(extra_langs=["th"])
    before = (root / "datasets" / ds / "candidates.jsonl").read_bytes()
    with pytest.raises(SystemExit, match="frozen"):
        translate.run(ds)
    assert (root / "datasets" / ds / "candidates.jsonl").read_bytes() == before


# ---- stage 2: answers ----

def drafted_dataset(root, *rows):
    return dataset(root, [r.model_copy(update={"gold_ans": [], "reference_answer": None, "answer_status": "none"}) for r in rows])


def test_generate_writes_english_questions_without_answers(material, root, fake):
    small_settings(**ONE_TYPE)
    generate.run("th")  # a Thai document still gets English questions
    rows = candidates(root)
    assert rows and all(q.lang == "en" and q.id == f"{q.base_id}-en" and not q.gold_ans for q in rows)
    assert all(re.fullmatch(r"th-\d{3}-en", q.id) for q in rows)
    assert all("Write every question in English" in llm.render("generate", document="", doc="d", qtype="fact", n="1",
                                                                guidance="", slots="") for _ in [0])


def test_answer_stage_drafts_verifies_and_resumes(root, fake):
    from rageval.datagen import answer
    ds = drafted_dataset(root, row("law-001-en"), row("law-002-en"), row("law-003-en", "unanswerable"),
                         row("law-004-en").model_copy(update={"status": "candidate"}))
    out = answer.run(ds)
    assert out == {"drafted": 2, "failed": 0}
    rows = {q.id: q for q in store.read_jsonl(root / "datasets" / ds / "candidates.jsonl", Question)}
    for qid in ("law-001-en", "law-002-en"):
        assert rows[qid].answer_status == "candidate" and rows[qid].reference_answer and len(rows[qid].gold_ans) == 1
        assert rows[qid].note is None
    assert rows["law-003-en"].answer_status == "none" and rows["law-004-en"].answer_status == "none"
    with pytest.raises(SystemExit, match="No approved question"):
        answer.run(ds)  # nothing left to do; a rerun makes no model calls
    assert fake.tasks.count("answer") == 2


def test_answer_stage_keeps_a_failed_check_as_a_note_and_retries_an_unusable_reply(root, fake):
    from rageval.datagen import answer
    ds = drafted_dataset(root, row("law-001-en", question="NOTCOVERED why", resolved_question="NOTCOVERED why"),
                         row("law-002-en", question="BADANSWER what", resolved_question="BADANSWER what"))
    assert answer.run(ds) == {"drafted": 1, "failed": 1}
    rows = {q.id: q for q in store.read_jsonl(root / "datasets" / ds / "candidates.jsonl", Question)}
    assert rows["law-001-en"].answer_status == "candidate" and "answer:" in rows["law-001-en"].note
    assert rows["law-002-en"].answer_status == "none"  # left for the next start


def test_translation_and_freeze_wait_for_approved_answers(root, fake, fake_judge):
    from rageval.datagen import answer, review
    small_settings(extra_langs=["th"])
    ds = drafted_dataset(root, row("law-001-en"), row("law-002-en"))
    assert translate.missing(ds) == []  # no answers yet, so nothing to translate
    answer.run(ds)
    assert translate.missing(ds) == []  # drafted but not approved
    assert review.set_answer_status(ds, ["law-001-en"], "approved") == 1
    assert translate.missing(ds) == [("law-001", "th")]
    translate.run(ds)
    out = {q.id: q for q in store.read_jsonl(root / "datasets" / ds / "candidates.jsonl", Question)}
    th = out["law-001-th"]
    assert th.reference_answer.startswith("[th] ") and th.answer_status == "approved" and th.base_id == "law-001"
    freeze.run(ds)
    frozen = {q.id for q in store.read_jsonl(root / "datasets" / ds / "questions.jsonl", Question)}
    assert frozen == {"law-001-en", "law-001-th"}  # law-002 has no approved answer


def test_editing_an_approved_answer_needs_approval_again_and_drops_translations(root):
    from rageval.datagen import review
    ds = dataset(root, [row("law-001-en"), row("law-001-th", lang="th").model_copy(update={"base_id": "law-001"})])
    store.write_jsonl(root / "datasets" / ds / "pages.jsonl", [Page(subject="law", doc=d, page=n, text=t) for (d, n), t in PMAP.items()])
    first = review.load(ds)[0]
    assert review.save_row(ds, first.model_copy(update={"reference_answer": "a new answer"})) is None
    rows = review.load(ds)
    assert rows[0].answer_status == "candidate" and [r.id for r in rows] == ["law-001-en"]
    assert review.set_answer_status(ds, ["law-001-en"], "approved") == 1


def test_pdf_text_is_cached_between_runs(root, monkeypatch):
    pdf = root / "material" / "law" / "a.pdf"
    pdf.parent.mkdir(parents=True)
    pdf.write_bytes(b"%PDF-1.4 not a real file")
    calls = []

    class Page_:
        def extract_text(self):
            return "page text"

    class Reader:
        def __init__(self, path):
            calls.append(path)
            self.pages = [Page_(), Page_()]

    monkeypatch.setattr(ingest, "PdfReader", Reader)
    assert ingest._extract(pdf) == ["page text", "page text"]
    assert ingest._extract(pdf) == ["page text", "page text"] and len(calls) == 1  # second read comes from .cache
    pdf.write_bytes(b"%PDF-1.4 changed and longer")
    ingest._extract(pdf)
    assert len(calls) == 2  # a changed file is read again
