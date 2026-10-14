"""The hand-off between data generation and evaluation: one CSV row per test case (docs/dataset-contract.md).

Blank cells mean unknown or unlabelled; "[]" means an explicitly empty list. Columns the generator cannot fill yet
stay blank. Fields of `Question` that have no contract column travel in `provenance_json`, so a CSV written here reads
back as the same questions.
"""
import csv
import io
import json
import re
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


def locate(page_text: str, quote: str) -> tuple[int, int] | None:
    """Zero-based start and exclusive end of the quote in the page text, ignoring differences in whitespace."""
    i = page_text.find(quote)
    if i >= 0:
        return i, i + len(quote)
    words = quote.split()
    m = re.search(r"\s+".join(re.escape(w) for w in words), page_text) if words else None
    return m.span() if m else None


def _evidence(q: Question, source_hashes: dict[str, str | None], pages: dict | None) -> list[dict]:
    """One group per required fact with the quotes that support it; quotes tied to no fact get a group of their own."""
    def alt(e) -> dict:
        span = locate(pages.get((e.doc, e.page), ""), e.quote) if pages else None
        return {"document_id": e.doc, "source_hash": source_hashes.get(e.doc), "quote": e.quote,
                "spans": [{"page": e.page, "start": span[0] if span else None, "end": span[1] if span else None}]}
    groups = []
    for i in range(1, len(q.gold_ans) + 1):
        alts = [alt(e) for e in q.gold_evidence if f"f{i}" in e.facts]
        if alts:
            groups.append({"fact_id": f"f{i}", "alternatives": alts})
    return groups + [{"fact_id": None, "alternatives": [alt(e)]} for e in q.gold_evidence if not e.facts]


def to_row(q: Question, by_id: dict[str, Question], *, scope: str, corpus_version: str,
           source_hashes: dict[str, str | None], generator_model: str = "",
           pages: dict[tuple[str, int], str] | None = None) -> dict[str, str]:
    prev = by_id.get(q.previous_id) if q.previous_id else None
    history = [{"role": "user", "content": prev.question},
               {"role": "assistant", "content": "\n".join(prev.gold_ans)}] if prev else []
    evidence = _evidence(q, source_hashes, pages)
    reviewed = q.status == "approved" and (not q.is_translated or q.translation_reviewed)
    return {
        "case_id": q.id, "course_id": q.subject, "scope": scope, "question": q.question,
        "history_json": _j(history), "corpus_version": corpus_version,
        "answerability": "answerable" if q.answerable else "unanswerable",
        "reference_answer": (q.reference_answer or "\n".join(q.gold_ans)) if q.answerable else NO_ANSWER,
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
    merged: dict[tuple, Evidence] = {}  # a quote that supports several facts appears once, with all their ids
    for g in _loads(r.get("evidence_json", ""), []):
        for a in g.get("alternatives", []):
            page = (a.get("spans") or [{}])[0].get("page") or 0
            e = merged.setdefault((a["document_id"], page, a["quote"]), Evidence(doc=a["document_id"], page=page, quote=a["quote"]))
            if g.get("fact_id") and g["fact_id"] not in e.facts:
                e.facts.append(g["fact_id"])
    evidence = list(merged.values())
    lang = r.get("language", "")
    return Question(
        id=r["case_id"], base_id=prov.get("base_id") or r.get("question_family_id") or r["case_id"],
        subject=r["course_id"], lang=lang, source_lang=prov.get("source_lang") or lang,
        question=r["question"], resolved_question=prov.get("resolved_question") or r["question"],
        previous_id=prov.get("previous_id"), scenario=prov.get("scenario"),
        question_type=prov.get("question_type") or next((t for t in tags if t in QUESTION_TYPES), None)
        or ("fact" if answerable else "unanswerable"),
        difficulty=prov.get("difficulty"), answerable=answerable,
        reference_answer=(r.get("reference_answer") or None) if answerable else None, answer_status="approved" if answerable else "none",
        gold_ans=[f["text"] for f in facts], gold_evidence=evidence,
        closed_book=bool(prov.get("closed_book")), translation_reviewed=bool(prov.get("translation_reviewed")),
        status="approved")
