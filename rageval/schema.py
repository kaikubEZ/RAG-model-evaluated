"""Row models for every file the tool reads or writes (docs/PRD.md section 7, docs/architecture.md section 4)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict

QUESTION_TYPES = ("fact", "definition", "comparison", "multi_hop", "exception", "unanswerable")
SCENARIOS = ("same", "similar", "diff_topic")

QuestionType = Literal["fact", "definition", "comparison", "multi_hop", "exception", "unanswerable"]
Scenario = Literal["same", "similar", "diff_topic"]
Difficulty = Literal["easy", "medium", "hard"]
Status = Literal["candidate", "approved", "rejected"]
AnswerStatus = Literal["none", "candidate", "approved"]
Verdict = Literal["supported", "contradicted", "absent"]


class Row(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Page(Row):
    subject: str
    doc: str  # path relative to material/, forward slashes, e.g. "law/02-roman-law.pdf"
    page: int  # 1-based
    text: str


class Evidence(Row):
    doc: str
    page: int
    quote: str
    facts: list[str] = []  # ids (f1, f2...) of the required facts this quote supports


class Question(Row):
    id: str  # base_id + "-" + lang, for example med-001-en; its translations are med-001-th and so on
    base_id: str
    subject: str
    lang: str
    source_lang: str
    question: str
    resolved_question: str
    previous_id: str | None = None
    scenario: Scenario | None = None
    question_type: QuestionType
    difficulty: Difficulty | None = None
    answerable: bool
    gold_ans: list[str] = []  # the required facts, drafted by the answer stage
    reference_answer: str | None = None
    answer_status: AnswerStatus = "none"  # none until the answer stage drafts it; approved by the reviewer
    gold_evidence: list[Evidence] = []
    hard_negative: Evidence | None = None  # deferred in v1: always None
    closed_book: bool = False
    translation_reviewed: bool = False
    status: Status = "candidate"
    # review-only fields, kept in candidates.jsonl, harmless in questions.jsonl
    closest_passage: str | None = None
    note: str | None = None  # why a row failed a check on save, shown to the reviewer

    @property
    def is_follow_up(self) -> bool:
        return self.previous_id is not None

    @property
    def answer_ready(self) -> bool:
        """An unanswerable question needs no answer; any other needs an approved one before it is translated or frozen."""
        return not self.answerable or self.answer_status == "approved"

    @property
    def is_translated(self) -> bool:
        return self.lang != self.source_lang


class Context(Row):
    model_config = ConfigDict(extra="ignore")
    text: str


class Answer(Row):
    id: str
    answer: str | None = None
    contexts: list[Context] | None = None
    error: str | None = None
    latency_ms: int | None = None


class KeyPoint(Row):
    point: str
    verdict: Verdict


class Claim(Row):
    claim: str
    supported: bool


class Score(Row):
    id: str
    base_id: str
    error: str | None = None  # endpoint error
    judge_error: str | None = None
    recall_at_k: float | None = None
    reciprocal_rank: float | None = None
    context_count: int | None = None
    context_chars: int | None = None
    refused: bool | None = None
    answer_lang: str | None = None
    language_match: int | None = None
    answer_recall: float | None = None
    contradiction: int | None = None
    key_points: list[KeyPoint] | None = None
    faithfulness: float | None = None
    claims: list[Claim] | None = None
    judge_model: str | None = None
    judge_prompt_versions: dict[str, str] = {}
