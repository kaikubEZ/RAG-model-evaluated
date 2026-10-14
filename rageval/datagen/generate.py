"""`rageval-datagen generate`, stage 1: English questions with their evidence, validated (architecture 3.3 and 3.4). Answers come later."""
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, ValidationError

from . import ingest, validate
from .. import jobs, llm, settings, store, text
from ..schema import QUESTION_TYPES, SCENARIOS, Evidence, Page, Question

SINGLE_ROUNDS = 3  # the first round and up to two refills
FOLLOW_ROUNDS = 2  # the first round and one refill
BASE_LANG = "en"  # the baseline set is English; other languages come from the translate stage


class _Quote(BaseModel):
    page: int
    quote: str


class _Item(BaseModel):
    slot: int | None = None
    question: str
    resolved_question: str | None = None
    question_type: str | None = None
    scenario: str | None = None
    previous_id: str | None = None
    gold_evidence: list[_Quote] = []


class _Reply(BaseModel):
    language: str = "en"  # ignored: every baseline question is English
    items: list[dict]  # validated one by one, so a single malformed item does not discard the batch


@dataclass
class _State:
    dir: Path
    s: settings.Settings
    job: jobs.Job
    build: dict
    cands: list[Question]
    docs: dict[str, dict[str, list[Page]]]  # subject -> doc -> pages
    pmap: dict[tuple[str, int], str]
    drops: list[tuple[str, str]] = field(default_factory=list)  # this batch's drops, counted when it commits


def split(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Share `total` across documents in proportion to `weights`: largest remainder, ties by name."""
    whole = sum(weights.values())
    if total <= 0 or not whole:
        return {}
    out = {d: total * w // whole for d, w in weights.items()}
    by_remainder = sorted(weights, key=lambda d: (-(total * weights[d] % whole), d))
    for d in by_remainder[:total - sum(out.values())]:
        out[d] += 1
    return {d: n for d, n in out.items() if n}


def _weights(docs: dict[str, list[Page]]) -> dict[str, int]:
    """Normalised characters on each document's eligible pages."""
    return {doc: sum(len(text.normalise(p.text)) for p in pages if text.eligible(p.text)) for doc, pages in docs.items()}


def _ranges(eligible: list[int], n: int) -> list[tuple[int, int]]:
    """Divide the eligible pages into n equal ranges; question k draws its first quote from range k."""
    out = []
    for i in range(n):
        lo = i * len(eligible) // n
        out.append((eligible[lo], eligible[max(lo, (i + 1) * len(eligible) // n - 1)]))
    return out


def _guidance(st: _State) -> str:
    g = st.s.generation_guidance.strip()
    return f"Guidance from the user: {g}" if g else ""


def _save(st: _State) -> None:
    store.write_json(st.dir / "build.json", st.build)


def _drop(st: _State, check: str, label: str) -> None:
    st.drops.append((check, label))


def _items(st: _State, reply: _Reply, label: str, limit: int | None = None) -> list[_Item]:
    """The reply's items that are well formed; each malformed one is a bad_reply drop."""
    out = []
    for raw in reply.items[:limit]:
        try:
            out.append(_Item.model_validate(raw))
        except ValidationError:
            _drop(st, "bad_reply", label)
    return out


def _new_id(st: _State, subject: str) -> str:
    n = st.build["next_id"][subject] = st.build["next_id"].get(subject, 0) + 1
    return f"{subject}-{n:03d}"


def _commit(st: _State, key: str, survivors: list[Question]) -> None:
    """Make a finished batch durable.

    The advanced id counter is saved before any candidate is written, together with the ids about to be written,
    so a crash at any point can neither reuse an id nor leave a half-written batch behind (see _recover).
    """
    st.build["pending"] = {"key": key, "ids": [q.id for q in survivors]}
    _save(st)
    for q in survivors:
        store.append_jsonl(st.dir / "candidates.jsonl", q)
    st.cands += survivors
    for check, label in st.drops:
        counts = st.build["dropped"].setdefault(check, {})
        counts[label] = counts.get(label, 0) + 1
    st.drops.clear()
    st.build["done"].append(key)
    del st.build["pending"]
    _save(st)


def _recover(st: _State) -> None:
    """After a crash inside _commit, remove the candidates of the unfinished batch; it will be generated again."""
    pending = st.build.pop("pending", None)
    if pending:
        st.cands = [q for q in st.cands if q.id not in set(pending["ids"])]
        store.write_jsonl(st.dir / "candidates.jsonl", st.cands)
        _save(st)


def _vet(st: _State, q: Question, pool: list[Question], label: str, rng: tuple[int, int] | None) -> Question | None:
    """All checks for one candidate; counts the drop and returns None when it fails."""
    reason = validate.check_row(q, st.pmap, pool)
    check = reason.split(":")[0] if reason else None
    if not check and rng and not rng[0] <= q.gold_evidence[0].page <= rng[1]:
        check = "spread"
    if not check:
        st.job.check_stop()
        try:
            out, check = validate.judge_checks(q, st.docs[q.subject]) if st.s.judge_checks else (q, None)
        except llm.BadReply:
            check = "bad_reply"
        if not check and (validate.check_row(out, st.pmap, pool)):
            check = "structure"  # removing unused evidence broke a rule
    if check:
        _drop(st, check, label)
        return None
    if out.answerable:
        out = out.model_copy(update={"difficulty": validate.difficulty(out)})
    pool.append(out)
    return out


def _batch_single(st: _State, key: str, subject: str, qtype: str, doc: str, n: int) -> None:
    pages = st.docs[subject][doc]
    eligible = [p.page for p in pages if text.eligible(p.text)]
    ranges = [] if qtype == "unanswerable" else _ranges(eligible, n)
    slots = ""
    if ranges:
        slots = (f'- Give each question a "slot" number from 1 to {n}. The first evidence quote of slot k must come '
                 "from the pages listed for slot k:\n" + "\n".join(f"Slot {k}: pages {lo} to {hi}" for k, (lo, hi) in enumerate(ranges, 1)))
    prompt = llm.render("generate", document=ingest.document_text(pages), doc=doc, qtype=qtype, n=str(n),
                        guidance=_guidance(st), slots=slots)
    try:
        reply = llm.ask("gen", prompt, _Reply)
    except llm.BadReply:
        _drop(st, "bad_reply", qtype)
        return _commit(st, key, [])
    pool = [q for q in st.cands if q.subject == subject]
    kept: list[Question] = []
    for item in _items(st, reply, qtype, n):
        rng = None
        if ranges:
            if not item.slot or not 1 <= item.slot <= n:
                _drop(st, "spread", qtype)
                continue
            rng = ranges[item.slot - 1]
        answerable = qtype != "unanswerable"
        base = _new_id(st, subject)
        q = Question(id=f"{base}-{BASE_LANG}", base_id=base, subject=subject, lang=BASE_LANG, source_lang=BASE_LANG,
                     question=item.question, resolved_question=item.question, question_type=qtype, answerable=answerable,
                     gold_evidence=[Evidence(doc=doc, **e.model_dump()) for e in item.gold_evidence] if answerable else [])
        if out := _vet(st, q, pool, qtype, rng):
            kept.append(out)
    _commit(st, key, kept)


def _batch_follow(st: _State, key: str, subject: str, doc: str, counts: dict[str, int]) -> None:
    singles = [q for q in st.cands if q.subject == subject and q.answerable and not q.is_follow_up]
    firsts = [q for q in singles if q.gold_evidence[0].doc == doc]
    pages = lambda q: ", ".join(str(p) for p in sorted({e.page for e in q.gold_evidence}))  # noqa: E731
    prompt = llm.render(
        "follow_up", document=ingest.document_text(st.docs[subject][doc]), doc=doc, guidance=_guidance(st),
        counts="\n".join(f"Scenario {sc}: {n}" for sc, n in counts.items()),
        firsts="\n".join(f"{q.id} | {q.question_type} | {pages(q)} | {q.question}" for q in firsts) or "(none)",
        others="\n".join(f"{q.id} | {q.question}" for q in singles if q not in firsts) or "(none)")
    try:
        reply = llm.ask("gen", prompt, _Reply)
    except llm.BadReply:
        _drop(st, "bad_reply", "follow_up")
        return _commit(st, key, [])
    pool = [q for q in st.cands if q.subject == subject]
    taken = dict.fromkeys(counts, 0)
    kept: list[Question] = []
    for item in _items(st, reply, "follow_up"):
        if item.scenario not in taken or taken[item.scenario] >= counts[item.scenario]:
            continue
        taken[item.scenario] += 1
        base = _new_id(st, subject)
        try:
            q = Question(id=f"{base}-{BASE_LANG}", base_id=base, subject=subject, lang=BASE_LANG, source_lang=BASE_LANG,
                         question=item.question, resolved_question=item.resolved_question or item.question,
                         previous_id=item.previous_id, scenario=item.scenario, question_type=item.question_type,
                         answerable=True, gold_evidence=[Evidence(doc=doc, **e.model_dump()) for e in item.gold_evidence])
        except ValidationError:
            _drop(st, "structure", item.scenario)
            continue
        if out := _vet(st, q, pool, item.scenario, None):
            kept.append(out)
    _commit(st, key, kept)


def _plan_single(st: _State) -> list[list]:
    plan = []
    for subject, docs in sorted(st.docs.items()):
        weights = _weights(docs)
        for qtype in QUESTION_TYPES:
            have = sum(1 for q in st.cands if q.subject == subject and q.question_type == qtype and not q.is_follow_up)
            need = st.s.candidates(st.s.questions_per_subject.get(qtype, 0)) - have
            plan += [[subject, qtype, doc, n] for doc, n in sorted(split(need, weights).items())]
    return plan


def _plan_follow(st: _State) -> list[list]:
    plan = []
    for subject, docs in sorted(st.docs.items()):
        weights = _weights(docs)
        with_questions = {q.gold_evidence[0].doc for q in st.cands
                          if q.subject == subject and q.answerable and not q.is_follow_up}
        usable = {
            "same": {d: w for d, w in weights.items() if d in with_questions},
            "diff_topic": {d: w for d, w in weights.items() if with_questions - {d}},
        }
        usable["similar"] = usable["same"]
        per_doc: dict[str, dict[str, int]] = {}
        for sc in SCENARIOS:
            have = sum(1 for q in st.cands if q.subject == subject and q.scenario == sc)
            need = st.s.candidates(st.s.follow_ups_per_subject.get(sc, 0)) - have
            for doc, n in split(need, usable[sc]).items():
                per_doc.setdefault(doc, {})[sc] = n
        plan += [[subject, doc, counts] for doc, counts in sorted(per_doc.items())]
    return plan


def _stage(st: _State, kind: str, rounds: int, plan_fn, batch_fn) -> None:
    for r in range(rounds):
        # the plan is per set of documents, so adding a subject or a file to the dataset plans new batches
        name = f"{kind}:{r}:{st.build['corpus']}"
        if name not in st.build["plan"]:
            st.build["plan"][name] = plan_fn(st)
            _save(st)
        batches = st.build["plan"][name]
        if not batches:
            continue
        st.job.step(name, len(batches))
        for i, batch in enumerate(batches):
            if f"{name}:{i}" not in st.build["done"]:
                batch_fn(st, f"{name}:{i}", *batch)
            st.job.tick(f"{name}:{i}")


def _shortfall(st: _State) -> dict:
    """Targets minus what survived validation, per subject and type or scenario."""
    out: dict[str, dict[str, int]] = {}
    for subject in st.docs:
        mine = [q for q in st.cands if q.subject == subject]
        wanted = [(t, n, sum(1 for q in mine if q.question_type == t and not q.is_follow_up))
                  for t, n in st.s.questions_per_subject.items()]
        wanted += [(sc, n, sum(1 for q in mine if q.scenario == sc)) for sc, n in st.s.follow_ups_per_subject.items()]
        out[subject] = {name: target - have for name, target, have in wanted if target > have}
    return out


def run(subject: str | None = None, dataset_id: str | None = None) -> str:
    """The whole `rageval generate` job. Returns the dataset id; running it again resumes."""
    s = settings.load()
    dataset_id = dataset_id or store.newest_unfrozen() or store.new_id()
    if store.is_frozen(dataset_id):
        raise SystemExit(f"Dataset {dataset_id} is frozen and cannot be changed. Generate a new dataset instead.")
    d = store.datasets_dir() / dataset_id
    with jobs.Job(d, "generate") as job:
        job.step("ingest", 1)
        pages = ingest.ingest(d, subject)
        job.tick()
        if not pages:
            raise SystemExit("There are no documents under material/. Add files to a subject folder first.")
        docs: dict[str, dict[str, list[Page]]] = {}
        for p in pages:
            docs.setdefault(p.subject, {}).setdefault(p.doc, []).append(p)
        build = store.read_json(d / "build.json")
        for k, v in (("done", []), ("plan", {}), ("dropped", {}), ("next_id", {}), ("langs", {}), ("shortfall", {})):
            build.setdefault(k, v)
        build["corpus"] = hashlib.sha256("|".join(sorted(d for ds in docs.values() for d in ds)).encode()).hexdigest()[:8]
        st = _State(d, s, job, build, store.read_jsonl(d / "candidates.jsonl", Question), docs,
                    {(p.doc, p.page): p.text for p in pages})
        _recover(st)
        _stage(st, "single", SINGLE_ROUNDS, _plan_single, _batch_single)
        _stage(st, "follow", FOLLOW_ROUNDS, _plan_follow, _batch_follow)
        build["shortfall"] = _shortfall(st)
        _save(st)
    return dataset_id
