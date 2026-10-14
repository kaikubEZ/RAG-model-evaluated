"""Code metrics from the contexts a target returned (architecture 3.9 and 3.10)."""
from .. import text
from ..schema import Answer, Question


def retrieval(q: Question, contexts: list[str] | None, top_k: int) -> dict:
    """Recall@k, reciprocal rank and context size; None where not measurable."""
    out = {"recall_at_k": None, "reciprocal_rank": None, "context_count": None, "context_chars": None}
    if contexts is None:
        return out
    out["context_count"], out["context_chars"] = len(contexts), sum(len(c) for c in contexts)
    if q.answerable and q.gold_evidence:
        found = [text.found_at(e.quote, contexts) for e in q.gold_evidence]
        hits = [k for k in found if k is not None]
        out["recall_at_k"] = sum(k <= top_k for k in hits) / len(found)
        out["reciprocal_rank"] = 1 / min(hits) if hits else 0.0
    return out


def judge_contexts(q: Question, answers: dict[str, Answer]) -> list[str]:
    """What the faithfulness judge sees: this turn's contexts, then the first turn's for a follow-up."""
    out: list[str] = []
    for i in (q.id, q.previous_id):
        if i in answers and answers[i].contexts:
            out += [c.text for c in answers[i].contexts]
    return out
