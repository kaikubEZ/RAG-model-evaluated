"""The hand-off between data generation and evaluation: one CSV row per test case (docs/dataset-contract.md).

Blank cells mean unknown or unlabelled; "[]" means an explicitly empty list. Columns the generator cannot fill yet
stay blank. Fields of `Question` that have no contract column travel in `provenance_json`, so a CSV written here reads
back as the same questions.
"""
import csv
import io
import json
from pathlib import Path

from . import store
from .schema import QUESTION_TYPES, Evidence, Question

COLUMNS = ["case_id", "course_id", "scope", "question", "history_json", "corpus_version", "answerability",
           "reference_answer", "required_facts_json", "evidence_json", "retrieval_qrels_json", "relevance_coverage",
           "answer_rubric_json", "label_status", "language", "split", "question_family_id", "tags_json",
           "provenance_json", "expected_labels_json"]
REQUIRED = ["case_id", "course_id", "scope", "question"]
NO_ANSWER = "The course material does not answer this question; the tutor should say so."


def _j(value) -> str:
    return json.dumps(value, ensure_ascii=False)


def to_row(q: Question, by_id: dict[str, Question], *, scope: str, corpus_version: str,
           source_hashes: dict[str, str | None], generator_model: str = "") -> dict[str, str]:
    prev = by_id.get(q.previous_id) if q.previous_id else None
    history = [{"role": "user", "content": prev.question},
               {"role": "assistant", "content": "\n".join(prev.gold_ans)}] if prev else []
    evidence = [{"fact_id": None, "alternatives": [{
        "document_id": e.doc, "source_hash": source_hashes.get(e.doc),
        "spans": [{"page": e.page, "start": None, "end": None}], "quote": e.quote}]} for e in q.gold_evidence]
    reviewed = q.status == "approved" and (not q.is_translated or q.translation_reviewed)
    return {
        "case_id": q.id, "course_id": q.subject, "scope": scope, "question": q.question,
        "history_json": _j(history), "corpus_version": corpus_version,
        "answerability": "answerable" if q.answerable else "unanswerable",
        "reference_answer": "\n".join(q.gold_ans) if q.answerable else NO_ANSWER,
        "required_facts_json": _j([{"fact_id": f"f{i}", "text": t} for i, t in enumerate(q.gold_ans, 1)]),
        "evidence_json": _j(evidence),
        "retrieval_qrels_json": "", "relevance_coverage": "", "answer_rubric_json": "",
        "label_status": "human_reviewed" if reviewed else "automatically_validated",
        "language": q.lang, "split": "", "question_family_id": q.base_id,
        "tags_json": _j([t for t in (q.question_type, q.scenario) if t]),
        "provenance_json": _j({
            "generator_model": generator_model, "base_id": q.base_id, "source_lang": q.source_lang,
            "resolved_question": q.resolved_question, "previous_id": q.previous_id, "scenario": q.scenario,
            "question_type": q.question_type, "difficulty": q.difficulty, "closed_book": q.closed_book,
            "translation_reviewed": q.translation_reviewed}),
        "expected_labels_json": "",
    }


def write(path: Path, rows: list[dict[str, str]]) -> None:
    buf = io.StringIO()
    w = csv.DictWriter(buf, COLUMNS, lineterminator="\n")  # the csv writer escapes quotes, commas and newlines
    w.writeheader()
    w.writerows(rows)
    store.write_text_atomic(path, buf.getvalue())


def _loads(cell: str, default):
    return json.loads(cell) if cell.strip() else default


def read(path: Path) -> list[Question]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"{path.name} lacks required column(s): {', '.join(missing)}.")
        rows = list(reader)
    out = []
    for n, r in enumerate(rows, 2):  # line 1 is the header
        try:
            out.append(_question(r))
        except (ValueError, KeyError, TypeError) as e:
            raise SystemExit(f"{path.name} line {n} (case {r.get('case_id')!r}): {type(e).__name__}: {e}")
    ids = [q.id for q in out]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{path.name} has duplicate case_id values.")
    return out


def _question(r: dict[str, str]) -> Question:
    prov = _loads(r.get("provenance_json", ""), {})
    answerable = r.get("answerability", "") != "unanswerable"
    tags = _loads(r.get("tags_json", ""), [])
    facts = _loads(r.get("required_facts_json", ""), [])
    evidence = [Evidence(doc=a["document_id"], page=((a.get("spans") or [{}])[0].get("page") or 0), quote=a["quote"])
                for g in _loads(r.get("evidence_json", ""), []) for a in g.get("alternatives", [])]
    lang = r.get("language", "")
    return Question(
        id=r["case_id"], base_id=prov.get("base_id") or r.get("question_family_id") or r["case_id"],
        subject=r["course_id"], lang=lang, source_lang=prov.get("source_lang") or lang,
        question=r["question"], resolved_question=prov.get("resolved_question") or r["question"],
        previous_id=prov.get("previous_id"), scenario=prov.get("scenario"),
        question_type=prov.get("question_type") or next((t for t in tags if t in QUESTION_TYPES), None)
        or ("fact" if answerable else "unanswerable"),
        difficulty=prov.get("difficulty"), answerable=answerable,
        gold_ans=[f["text"] for f in facts], gold_evidence=evidence,
        closed_book=bool(prov.get("closed_book")), translation_reviewed=bool(prov.get("translation_reviewed")),
        status="approved")
