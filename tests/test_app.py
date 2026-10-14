import json
import random
import time
from pathlib import Path

import pytest
from streamlit.proto.TextInput_pb2 import TextInput
from streamlit.testing.v1 import AppTest

from rageval import csvio, jobs, settings, store
from rageval.evaluation import dataset
from rageval.datagen import material, review
from rageval.schema import Answer, Evidence, Page, Question, Score

EN = "abcdefghijklmnopqrstuvwxyz"
ROOT_DIR = Path(__file__).resolve().parent.parent / "rageval"
APPS = {"datagen": ROOT_DIR / "datagen" / "app.py", "evaluation": ROOT_DIR / "evaluation" / "app.py"}
PAGES = [("datagen", n) for n in ("Setup", "Material", "Generate", "Review")] + [
    ("evaluation", n) for n in ("Setup", "Run and score", "Results", "Compare", "Labelling")]
ENV_KEYS = ["GEN_AI_API_KEY", "GEN_AI_URL", "GEN_AI_MODEL", "JUDGE_AI_API_KEY", "JUDGE_AI_URL", "JUDGE_AI_MODEL",
            "RAG_API_KEY", "RAG_URL"]


@pytest.fixture(autouse=True)
def no_env(monkeypatch):
    for k in ENV_KEYS:
        monkeypatch.setenv(k, "")  # so that monkeypatch removes anything the Setup page puts in os.environ
        monkeypatch.delenv(k)


def words(seed, n=30):
    r = random.Random(seed)
    return " ".join("".join(r.choice(EN) for _ in range(r.randint(4, 8))) for _ in range(n))


PAGE_TEXT = {1: words(1), 2: words(2), 3: words(3)}


def quote(page):
    return PAGE_TEXT[page][:150]


def question(n, page, **kw):
    qid = f"law-{n:03d}"
    fields = dict(id=f"{qid}-en", base_id=qid, subject="law", lang="en", source_lang="en", question=f"What is {words(100 + n, 4)}?",
                  resolved_question=f"What is {words(100 + n, 4)}?", question_type="fact", answerable=True,
                  gold_ans=[words(200 + n, 5)], reference_answer=words(300 + n, 6), answer_status="approved", gold_evidence=[Evidence(doc="law/a.txt", page=page, quote=quote(page))],
                  status="candidate")
    fields.update(kw)
    if "resolved_question" not in kw:
        fields["resolved_question"] = fields["question"]
    return Question(**fields)


def make_dataset(frozen=False, ds="20260101-000000", rows=None):
    d = store.datasets_dir() / ds
    store.write_jsonl(d / "pages.jsonl", [Page(subject="law", doc="law/a.txt", page=p, text=t) for p, t in PAGE_TEXT.items()])
    rows = rows if rows is not None else [question(1, 1), question(2, 2)]
    store.write_jsonl(d / ("questions.jsonl" if frozen else "candidates.jsonl"), rows)
    if frozen:
        by_id = {q.id: q for q in rows}
        csvio.write(d / "dataset.csv", [csvio.to_row(q, by_id, scope="current", corpus_version="v1", source_hashes={}) for q in rows])
        store.write_json(d / "manifest.json", {"dataset_id": ds, "sha256": "abc123"})
    return ds


def make_run(ds, run_id="20260102-000000-first"):
    rd = store.runs_dir() / run_id
    store.write_json(rd / "manifest.json", {
        "run_id": run_id, "dataset_id": ds, "dataset_sha256": dataset.sha(ds), "settings": settings.Settings().model_dump(),
        "rag_url": "", "adapter": "http_json", "label": "first", "top_k": 5, "matcher": {"window": 20, "coverage": 0.8},
        "seed": 0, "started": "2026-01-02T00:00:00+00:00"})
    qs = dataset.load(ds)
    store.write_jsonl(rd / "answers.jsonl", [Answer(id=q.id, answer="an answer", contexts=[{"text": quote(1)}]) for q in qs])
    store.write_jsonl(rd / "scores.jsonl", [
        Score(id=q.id, base_id=q.base_id, recall_at_k=1.0, reciprocal_rank=1.0, context_count=1, context_chars=150,
              refused=False, answer_lang="en", language_match=1, answer_recall=1.0, contradiction=0,
              key_points=[{"point": q.gold_ans[0], "verdict": "supported"}], faithfulness=1.0,
              claims=[{"claim": "a claim", "supported": True}], judge_model="m") for q in qs])
    return run_id


# ---------------------------------------------------------------- material

def test_upload_name_is_reduced_to_a_base_name(root):
    material.add_subject("law")
    for given, expected in (("..\\..\\evil.pdf", "evil.pdf"), ("a/b/notes.md", "notes.md"), ("my file.txt", "my file.txt"), ("C:\\x\\01-กฎหมาย.pdf", "01-กฎหมาย.pdf"), ("a<b>?.md", "a_b__.md")):
        assert material.save_upload("law", given, b"x") == root / "material" / "law" / expected
    assert sorted(p.name for p in (root / "material").rglob("*") if p.is_file()) == ["01-กฎหมาย.pdf", "a_b__.md", "evil.pdf", "my file.txt", "notes.md"]


def test_upload_refuses_other_types_and_unknown_subjects(root):
    material.add_subject("law")
    with pytest.raises(ValueError, match=r"\.pdf, \.txt, \.md"):
        material.save_upload("law", "run.exe", b"x")
    with pytest.raises(ValueError):
        material.save_upload("law", "..", b"x")
    with pytest.raises(ValueError):
        material.save_upload("nope", "a.txt", b"x")
    with pytest.raises(ValueError):
        material.add_subject("  ..  ")


def test_subjects_list_page_counts(root):
    material.add_subject("law")
    material.save_upload("law", "a.txt", "\n\n".join(words(i, 400) for i in range(3)).encode())
    material.save_upload("law", "tiny.txt", b"short")
    (sub,) = material.subjects()
    files = {f["name"]: f for f in sub["files"]}
    assert files["a.txt"]["pages"] >= 2 and files["a.txt"]["eligible"] == files["a.txt"]["pages"] and files["a.txt"]["fits"]
    assert (files["tiny.txt"]["pages"], files["tiny.txt"]["eligible"]) == (1, 0)
    settings.save(settings.Settings(gen_max_input_tokens=1000))
    assert not {f["name"]: f for f in material.subjects()[0]["files"]}["a.txt"]["fits"]


def test_delete_stays_inside_the_subject(root):
    material.add_subject("law")
    material.add_subject("other")
    material.save_upload("other", "keep.txt", b"x")
    material.save_upload("law", "a.txt", b"x")
    for bad in ("..\\other\\keep.txt", "../other/keep.txt", "missing.txt"):
        with pytest.raises(ValueError):
            material.delete_file("law", bad)
    material.delete_file("law", "a.txt")
    assert [p.name for p in (root / "material" / "other").iterdir()] == ["keep.txt"]
    assert not (root / "material" / "law" / "a.txt").exists()


def test_synced_folder_detection(root, monkeypatch):
    assert material.synced_folder() is None
    for part, label in (("OneDrive", "OneDrive"), ("OneDrive - Acme", "OneDrive"), ("Dropbox", "Dropbox"),
                        ("Google Drive", "Google Drive"), ("iCloudDrive", "iCloud")):
        monkeypatch.setattr(store, "ROOT", root / part / "project")
        assert material.synced_folder() == label


# ---------------------------------------------------------------- review

def test_valid_edit_is_saved_and_difficulty_recomputed(root):
    ds = make_dataset()
    q = review.load(ds)[0]
    assert review.save_row(ds, q.model_copy(update={"question": "A brand new wording?", "resolved_question": "A brand new wording?",
                                                     "status": "approved"})) is None
    saved = review.load(ds)[0]
    assert (saved.question, saved.status, saved.note, saved.difficulty) == ("A brand new wording?", "approved", None, "medium")
    assert len(review.load(ds)) == 2


def test_quote_off_the_page_stays_a_candidate_and_cannot_be_approved(root):
    ds = make_dataset()
    q = review.load(ds)[0]
    bad = q.model_copy(update={"status": "approved", "gold_evidence": [Evidence(doc="law/a.txt", page=1, quote=quote(2))]})
    reason = review.save_row(ds, bad)
    assert reason.startswith("quote:")
    saved = review.load(ds)[0]
    assert (saved.status, saved.note) == ("candidate", reason)
    assert review.set_status(ds, [saved.id], "approved") == 0
    assert review.load(ds)[0].status == "candidate"
    assert review.save_row(ds, saved.model_copy(update={"gold_evidence": q.gold_evidence, "status": "approved"})) is None
    assert review.load(ds)[0].note is None


def test_new_row_gets_a_fresh_id(root):
    ds = make_dataset()
    new = question(9, 3, id="new", base_id="new")
    assert review.save_row(ds, new) is None
    ids = [r.id for r in review.load(ds)]
    assert ids == ["law-001-en", "law-002-en", "law-003-en"]
    assert review.load(ds)[2].base_id == "law-003"
    review.save_row(ds, question(8, 2, id="new", base_id="new"))
    assert review.load(ds)[3].id == "law-004-en"


def test_editing_an_approved_source_question_removes_its_variants(root):
    ds = make_dataset(rows=[question(1, 1, status="approved"), question(2, 2, status="approved"),
                            question(1, 1, id="law-001-th", lang="th", status="approved"),
                            question(2, 2, id="law-002-th", lang="th", status="approved", base_id="law-002")])
    rows = review.load(ds)
    rows[2] = rows[2].model_copy(update={"base_id": "law-001"})
    store.write_jsonl(store.datasets_dir() / ds / "candidates.jsonl", rows)
    first = review.load(ds)[0]
    review.save_row(ds, first.model_copy(update={"gold_ans": ["changed key point"]}))
    assert [r.id for r in review.load(ds)] == ["law-001-en", "law-002-en", "law-002-th"]
    review.save_row(ds, review.load(ds)[0].model_copy(update={"difficulty": "hard"}))  # not a content change
    assert len(review.load(ds)) == 3


def test_editing_a_variant_rewrites_only_the_variant_and_review_is_marked(root):
    ds = make_dataset(rows=[question(1, 1, status="approved"),
                            question(1, 1, id="law-001-th", lang="th", status="approved", base_id="law-001")])
    variant = review.load(ds)[1]
    review.save_row(ds, variant.model_copy(update={"question": "ใหม่?", "resolved_question": "ใหม่?"}))
    rows = review.load(ds)
    assert (rows[0].status, rows[1].question, rows[1].difficulty) == ("approved", "ใหม่?", None)
    review.mark_reviewed(ds, "law-001-th")
    assert review.load(ds)[1].translation_reviewed
    with pytest.raises(ValueError):
        review.mark_reviewed(ds, "law-001-en")


def test_scenario_override_is_counted_once_per_row(root):
    ds = make_dataset(rows=[question(1, 1, status="approved"),
                            question(2, 2, question="and then?", resolved_question="What follows? Tell me.",
                                     previous_id="law-001-en", scenario="same")])
    follow = review.load(ds)[1]  # "same" but sharing no page with the first turn: disagrees with the page rule
    assert review.save_row(ds, follow) is None
    review.save_row(ds, review.load(ds)[1])
    assert store.read_json(store.datasets_dir() / ds / "build.json")["scenario_overrides"] == 1
    other = question(3, 3, id="law-003-en", question="and also?", resolved_question="What else? Tell me.", previous_id="law-001-en", scenario="same")
    review.save_row(ds, other.model_copy(update={"id": "x"}))
    build = store.read_json(store.datasets_dir() / ds / "build.json")
    assert build["scenario_overrides"] == 2 and len(build["scenario_override_ids"]) == 2


def test_bulk_approve_is_counted_and_failed_rows_are_skipped(root):
    ds = make_dataset(rows=[question(1, 1), question(2, 2), question(3, 3, note="structure: broken")])
    assert review.set_status(ds, ["law-001-en", "law-002-en", "law-003-en"], "approved", bulk=True) == 2
    assert review.set_status(ds, ["law-001-en"], "approved", bulk=True) == 0
    assert review.set_status(ds, ["law-003-en"], "approved") == 0
    assert store.read_json(store.datasets_dir() / ds / "build.json")["bulk_approved"] == 2
    assert review.summary(ds)["bulk_approved"] == 2


def test_rejecting_a_source_rejects_its_variants(root):
    ds = make_dataset(rows=[question(1, 1, status="approved"), question(2, 2, status="approved"),
                            question(1, 1, id="law-001-th", lang="th", status="approved", base_id="law-001")])
    assert review.set_status(ds, ["law-001-en"], "rejected") == 2
    assert [r.status for r in review.load(ds)] == ["rejected", "approved", "rejected"]


def test_saving_is_refused_while_a_job_runs_and_when_frozen(root):
    ds = make_dataset()
    q = review.load(ds)[0]
    d = store.datasets_dir() / ds
    store.write_json(d / "job.lock", {"pid": 1, "job": "generate", "heartbeat": time.time()})
    for call in (lambda: review.save_row(ds, q), lambda: review.set_status(ds, [q.id], "approved"),
                 lambda: review.mark_reviewed(ds, q.id)):
        with pytest.raises(ValueError, match="job is running"):
            call()
    (d / "job.lock").unlink()
    frozen = make_dataset(frozen=True, ds="20260102-000000")
    with pytest.raises(ValueError, match="frozen"):
        review.save_row(frozen, review.load(frozen)[0])
    with pytest.raises(ValueError, match="frozen"):
        review.set_status(frozen, ["law-001-en"], "rejected")
    assert len(review.load(frozen)) == 2


def test_summary_counts_and_shortfall(root):
    ds = make_dataset(rows=[question(1, 1, status="approved"), question(2, 2)])
    sm = review.summary(ds)
    assert (sm["total"], sm["approved"], sm["by_status"], sm["by_type"]) == (2, 1, {"approved": 1, "candidate": 1}, {"fact": 2})
    assert sm["shortfall"]["law"]["fact"] == 11 and sm["shortfall"]["law"]["unanswerable"] == 10


# ---------------------------------------------------------------- app

def page(name, app=None, **session):
    app = app or ("evaluation" if name in ("Run and score", "Results", "Compare", "Labelling") else "datagen")
    at = AppTest.from_file(str(APPS[app]), default_timeout=60)
    at.session_state["page"] = name
    for k, v in session.items():
        at.session_state[k] = v
    return at.run()


def state(root, kind):
    if kind == "empty":
        return
    material.add_subject("law")
    material.save_upload("law", "a.txt", ("\n\n".join(PAGE_TEXT.values())).encode())
    if kind == "draft":
        make_dataset()
        return
    ds = make_dataset(frozen=True)
    if kind == "scored":
        make_run(ds)


@pytest.mark.parametrize("kind", ["empty", "draft", "frozen", "scored"])
@pytest.mark.parametrize("app,name", PAGES)
def test_every_page_renders(root, kind, app, name):
    state(root, kind)
    at = page(name, app)
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]


def test_empty_project_says_what_to_do_next(root):
    assert "Create one above" in page("Material").info[0].value
    assert "Generate page" in page("Review").info[0].value
    assert "No dataset.csv" in page("Run and score").info[0].value
    assert "No scored run" in page("Results").info[0].value


def test_review_form_opens_for_a_new_question_and_saves_it(root):
    state(root, "draft")
    at = page("Review", rv_new=True)
    assert not at.exception, [e.value for e in at.exception]
    fk = "rv|20260101-000000|new"
    at.session_state[f"{fk}|question"] = "What is new?"
    at.session_state[f"{fk}|resolved"] = "What is new?"
    at.session_state[f"{fk}|kp0"] = "a new fact"
    at.session_state[f"{fk}|ev0|quote"] = quote(3)
    at.session_state[f"{fk}|ev0|page"] = 3
    at.run()
    next(b for b in at.button if b.key == f"{fk}|save").click().run()
    assert not at.exception, [e.value for e in at.exception]
    rows = review.load("20260101-000000")
    assert [r.id for r in rows][-1] == "law-003-en" and rows[-1].question == "What is new?" and rows[-1].note is None


def test_results_show_the_scored_run_and_compare_refuses_other_datasets(root):
    state(root, "scored")
    at = page("Results")
    assert not at.exception and not at.error
    assert any("Headline" in s.value for s in at.subheader)
    second = make_dataset(frozen=True, ds="20260103-000000")
    make_run(second, "20260104-000000-second")
    mismatch = store.runs_dir() / "20260104-000000-second" / "manifest.json"
    store.write_json(mismatch, {**store.read_json(mismatch), "dataset_sha256": "different"})
    at = page("Compare")
    assert not at.exception
    assert "different datasets" in at.error[0].value


def test_setup_saves_env_and_settings(root):
    at = page("Setup")
    assert not at.exception
    by_label = {w.label: w for w in at.text_input}
    by_label["GEN_AI_URL"].input("https://api.example.test/v1")
    by_label["GEN_AI_API_KEY"].input("secret-key")
    next(b for b in at.button if b.label == "Save").click().run()
    assert not at.exception and not at.error
    env = (root / ".env").read_text(encoding="utf-8")
    assert "GEN_AI_URL" in env and "secret-key" in env
    assert [w.proto.type for w in at.text_input if w.label == "GEN_AI_API_KEY"] == [TextInput.PASSWORD]
    assert settings.load().top_k == 5


def test_translate_step_takes_any_language_codes_the_user_types(root, monkeypatch):
    from rageval.datagen import translate
    assert translate.parse_langs("th, ZH  ja;th") == ["th", "zh", "ja"]
    for bad in ("swedish", "t", "en", "th, 12"):
        with pytest.raises(ValueError):
            translate.parse_langs(bad)
    state(root, "draft")
    monkeypatch.setenv("GEN_AI_URL", "https://api.example.test/v1")
    monkeypatch.setenv("GEN_AI_API_KEY", "k")
    monkeypatch.setenv("GEN_AI_MODEL", "m")
    started = []
    monkeypatch.setattr(jobs, "start_detached", lambda module, args: started.append(args))
    at = page("Review")
    assert not at.exception, [e.value for e in at.exception]
    box = next(w for w in at.text_input if w.label.startswith("Translate into"))
    box.input("th, zh").run()
    next(c for c in at.checkbox if c.key == "privacy_review").check().run()
    next(b for b in at.button if b.label == "Translate").click().run()
    assert not at.exception and not at.error
    assert settings.load().extra_langs == ["th", "zh"] and started == [["translate", "--yes"]]
    box = next(w for w in at.text_input if w.label.startswith("Translate into"))
    box.input("swedish").run()
    next(b for b in at.button if b.label == "Translate").click().run()
    assert "Not a language code" in at.error[0].value and settings.load().extra_langs == ["th", "zh"]


def test_generate_page_lists_new_dataset_and_old_drafts(root):
    state(root, "draft")
    make_dataset(ds="20260103-000000")
    make_dataset(frozen=True, ds="20260104-000000")
    at = page("Generate")
    assert not at.exception, [e.value for e in at.exception]
    box = next(b for b in at.selectbox if b.label == "Dataset to generate into")
    assert box.options[0] == "New dataset" and len(box.options) == 3  # the frozen dataset is not offered
    assert box.value == "20260103-000000"  # the newest unfinished draft is the default
    assert "2 questions so far" in box.format_func(box.value)
    box.select("New dataset").run()
    assert any(w.label.startswith("Name for the new dataset") for w in at.text_input)


def test_review_table_shows_the_answer_and_evidence(root):
    state(root, "draft")
    ds = "20260101-000000"
    rows = review.load(ds)
    store.write_jsonl(store.datasets_dir() / ds / "candidates.jsonl", [
        rows[0].model_copy(update={"reference_answer": "THE REFERENCE ANSWER", "answer_status": "candidate"}), rows[1]])
    at = page("Review")
    assert not at.exception, [e.value for e in at.exception]
    df = at.dataframe[0].value
    assert list(df.columns[:len(csvio.COLUMNS)]) == csvio.COLUMNS  # the contract's columns, in its order
    first = df.iloc[0]
    assert first["case_id"] == "law-001-en" and first["reference_answer"] == "THE REFERENCE ANSWER"
    facts, groups = json.loads(first["required_facts_json"]), json.loads(first["evidence_json"])
    assert facts[0]["fact_id"] == "f1" and groups[0]["alternatives"][0]["document_id"] == "law/a.txt"
    assert first["answer status"] == "candidate"
