"""Candidate checks (architecture 3.4): code checks first, then the judge checks."""
from pydantic import BaseModel

from . import ingest
from .. import llm, text
from ..schema import Page, Question


class _Answerable(BaseModel):
    question_standalone: bool
    resolved_standalone: bool
    answer: str = ""
    key_points_covered: list[bool]
    quotes_used: list[bool] = []


class _QuestionCheck(BaseModel):
    question_standalone: bool
    resolved_standalone: bool
    answerable: bool
    quotes_used: list[bool] = []


class _Unanswerable(BaseModel):
    standalone: bool
    answer_found: bool
    quote: str = ""
    closest_passage: str = ""


def scenario_ok(q: Question, first: Question) -> bool:
    """The scenario rule of architecture 3.4, on where the evidence sits."""
    mine = {(e.doc, e.page) for e in q.gold_evidence}
    theirs = {(e.doc, e.page) for e in first.gold_evidence}
    same_doc = bool({d for d, _ in mine} & {d for d, _ in theirs})
    if q.scenario == "same":
        return bool(mine & theirs)
    if q.scenario == "similar":
        return same_doc and not mine & theirs
    return not same_doc  # diff_topic


def _structure(q: Question, others: list[Question], strict_scenario: bool) -> str | None:
    if q.answerable != (q.question_type != "unanswerable"):
        return "answerable does not match the question type"
    if q.answerable:
        if q.answer_status != "none" and not 1 <= len(q.gold_ans) <= 4:
            return "an answer needs one to four required facts"
        if not q.gold_evidence:
            return "an answerable question needs at least one evidence quote"
    elif q.gold_ans or q.gold_evidence or q.is_follow_up:
        return "an unanswerable question has no key points or evidence and cannot be a follow-up"
    if q.question_type == "multi_hop":
        if len({e.doc for e in q.gold_evidence}) != 1 or len({e.page for e in q.gold_evidence}) < 2:
            return "multi_hop needs evidence on at least two pages of one document"
    if q.is_follow_up != (q.scenario is not None):
        return "previous_id and scenario go together"
    if not q.is_follow_up:
        return None
    first = next((o for o in others if o.id == q.previous_id), None)
    if first is None or first.is_follow_up or not first.answerable:
        return f"first turn {q.previous_id} is missing, is itself a follow-up, or is unanswerable"
    if (q.question == q.resolved_question) == (q.scenario == "same"):
        return "a same follow-up needs a question that differs from its resolved form; other scenarios need them equal"
    if strict_scenario and not scenario_ok(q, first):
        return f"the evidence pages do not fit scenario {q.scenario}"
    return None


def check_row(q: Question, pages: dict[tuple[str, int], str], others: list[Question],
              strict_scenario: bool = True) -> str | None:
    """Code-only checks (quote, structure, duplicate). None when the row passes, else the reason.

    The reason starts with the check name. `others` are the other questions of the same subject.
    """
    for e in q.gold_evidence:
        page = pages.get((e.doc, e.page))
        if page is None:
            return f"quote: {e.doc} has no page {e.page}"
        if not text.quote_length_ok(e.quote):
            return f"quote: a quote on page {e.page} is not {text.QUOTE_MIN} to {text.QUOTE_MAX} characters"
        if not text.quote_on_page(e.quote, page):
            return f"quote: a quote does not appear on {e.doc} page {e.page}"
    reason = _structure(q, others, strict_scenario)
    if reason:
        return f"structure: {reason}"
    for o in others:
        if (o.base_id != q.base_id and o.lang == q.lang and o.status != "rejected"
                and text.jaccard(q.resolved_question, o.resolved_question) >= text.DUPLICATE_JACCARD):
            return f"duplicate: too close to {o.id}"
    return None


def difficulty(q: Question) -> str:
    if len({(e.doc, e.page) for e in q.gold_evidence}) >= 2:
        return "hard"
    if text.overlap(q.resolved_question, [e.quote for e in q.gold_evidence]) >= text.EASY_OVERLAP:
        return "easy"
    return "medium"


def _numbered(label: str, items: list[str]) -> str:
    return "\n".join(f"{label} {i}: {' '.join(s.split())}" for i, s in enumerate(items, 1))


def judge_checks(q: Question, pages_by_doc: dict[str, list[Page]]) -> tuple[Question | None, str | None]:
    """The model checks for one candidate. (question, None) on pass, (None, check name) on failure.

    May raise llm.BadReply; the caller counts that as a dropped candidate.
    """
    if not q.answerable:
        passage = ""
        for pages in pages_by_doc.values():
            r = llm.ask("judge", llm.render("unanswerable", document=ingest.document_text(pages), question=q.question),
                        _Unanswerable)
            if r.answer_found:
                return None, "unanswerable"
            if not r.standalone:
                return None, "standalone"
            passage = passage or r.closest_passage
        return q.model_copy(update={"closest_passage": passage or None}), None
    if not q.gold_ans:  # stage 1: the question alone, against its quotes
        c = llm.ask("judge", llm.render("question_check", question=q.question, resolved_question=q.resolved_question,
                                        quotes=_numbered("Quote", [e.quote for e in q.gold_evidence])), _QuestionCheck)
        if not c.resolved_standalone or c.question_standalone != (q.scenario != "same"):
            return None, "standalone"
        if not c.answerable:
            return None, "answerable"
        used = [e for e, u in zip(q.gold_evidence, c.quotes_used) if u] if len(c.quotes_used) == len(q.gold_evidence) else []
        return q.model_copy(update={"gold_evidence": used or q.gold_evidence}), None
    r = llm.ask("judge", llm.render(
        "answerable", question=q.question, resolved_question=q.resolved_question,
        key_points=_numbered("Key point", q.gold_ans), quotes=_numbered("Quote", [e.quote for e in q.gold_evidence])),
        _Answerable)
    if not r.resolved_standalone or r.question_standalone != (q.scenario != "same"):
        return None, "standalone"
    if len(r.key_points_covered) != len(q.gold_ans) or not all(r.key_points_covered):
        return None, "answerable"
    used = [e for e, u in zip(q.gold_evidence, r.quotes_used) if u] if len(r.quotes_used) == len(q.gold_evidence) else []
    return q.model_copy(update={"gold_evidence": used or q.gold_evidence}), None


def answer_supported(q: Question) -> str | None:
    """Stage 2 check: do the quotes alone support every required fact, and is the question still standalone?

    None on pass, else the reason. May raise llm.BadReply.
    """
    r = llm.ask("judge", llm.render(
        "answerable", question=q.question, resolved_question=q.resolved_question,
        key_points=_numbered("Key point", q.gold_ans), quotes=_numbered("Quote", [e.quote for e in q.gold_evidence])),
        _Answerable)
    if len(r.key_points_covered) != len(q.gold_ans) or not all(r.key_points_covered):
        return "answer: the quotes do not support every required fact"
    return None
