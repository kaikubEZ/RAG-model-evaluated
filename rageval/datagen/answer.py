"""`rageval-datagen answer`, stage 2: draft each approved question's reference answer and required facts, then verify them."""
from pydantic import BaseModel

from . import validate
from .. import jobs, llm, settings, store
from ..schema import Question


class _Fact(BaseModel):
    text: str
    quotes: list[int] = []


class _Drafted(BaseModel):
    reference_answer: str
    required_facts: list[_Fact | str]


def todo(rows: list[Question]) -> list[Question]:
    """Approved English questions that still have no drafted answer."""
    return [q for q in rows if q.status == "approved" and q.answerable and not q.is_translated and q.answer_status == "none"]


def draft(q: Question) -> Question:
    """One question's answer, drafted from its quotes and checked against them. A failed check keeps the draft with a note."""
    quotes = "\n".join(f"Quote {i}: {' '.join(e.quote.split())}" for i, e in enumerate(q.gold_evidence, 1))
    r = llm.ask("gen", llm.render("answer", question=q.question, resolved_question=q.resolved_question, quotes=quotes), _Drafted)
    facts = [f if isinstance(f, _Fact) else _Fact(text=f) for f in r.required_facts]
    facts = [_Fact(text=f.text.strip(), quotes=f.quotes) for f in facts if f.text.strip()]
    n = len(q.gold_evidence)
    linked = []  # a quote supports the facts that name it; a fact that names no valid quote is tied to all of them
    for j in range(1, n + 1):
        linked.append([f"f{i}" for i, f in enumerate(facts, 1) if j in f.quotes or not any(1 <= k <= n for k in f.quotes)])
    evidence = [e.model_copy(update={"facts": ids}) for e, ids in zip(q.gold_evidence, linked)]
    q = q.model_copy(update={"reference_answer": r.reference_answer.strip(), "gold_ans": [f.text for f in facts],
                             "gold_evidence": evidence, "answer_status": "candidate"})
    reason = None if 1 <= len(q.gold_ans) <= 4 else "answer: one to four required facts are needed"
    try:
        reason = reason or (validate.answer_supported(q) if settings.load().judge_checks else None)
    except llm.BadReply:
        reason = "answer: the check could not be completed"
    return q.model_copy(update={"note": reason})


def run(dataset_id: str) -> dict:
    """Draft the missing answers. Running it again only fills gaps, so it also resumes after Stop."""
    d = store.datasets_dir() / dataset_id
    if store.is_frozen(dataset_id):
        raise SystemExit(f"Dataset {dataset_id} is frozen and cannot be changed. Duplicate it as a draft first.")
    rows = store.read_jsonl(d / "candidates.jsonl", Question)
    pending = todo(rows)
    if not pending:
        raise SystemExit("No approved question is waiting for an answer. Approve questions on the Review page first.")
    drafted = failed = 0
    with jobs.Job(d, "answer") as job:
        job.step("answer", len(pending))
        for q in pending:
            try:
                new = draft(q)
            except llm.BadReply:  # stays without an answer; the next start tries it again
                failed += 1
            else:
                rows = [new if r.id == q.id else r for r in rows]
                store.write_jsonl(d / "candidates.jsonl", rows)
                drafted += 1
            job.tick(q.id)
    return {"drafted": drafted, "failed": failed}
