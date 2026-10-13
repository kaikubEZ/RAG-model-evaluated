"""The two judge calls (architecture 3.10) with a reply cache, so re-scoring gives the same scores."""
import hashlib
import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel

from . import llm
from .schema import Claim, KeyPoint, Question


class AnswerCheck(BaseModel):
    refused: bool
    answer_lang: str
    key_points: list[KeyPoint]

    @property
    def answer_recall(self) -> float:
        if self.refused or not self.key_points:
            return 0.0
        return sum(k.verdict == "supported" for k in self.key_points) / len(self.key_points)

    @property
    def contradiction(self) -> int:
        return int(not self.refused and any(k.verdict == "contradicted" for k in self.key_points))


class _Claims(BaseModel):
    claims: list[Claim]


def read_json_lines(path: Path) -> list[dict]:
    """Rows of a jsonl file; a line cut short by a killed process is skipped."""
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    return rows


def append_json_line(path: Path, row: dict) -> None:
    """Append one row, starting a fresh line if the file ends in a partial one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        line = "\n" + line
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)


def _cached(name: str, prompt: str, shape: type[BaseModel], cache_path: Path | None, refresh: bool,
            wrong: Callable[[BaseModel], str | None] = lambda reply: None):
    """Ask the judge, or reuse the stored reply for the same model, prompt version and prompt.

    `wrong` names what is wrong with a reply of the right shape; such a reply is asked for once more
    and is never stored.
    """
    key = hashlib.sha256(f"{llm.model_name('judge')}\n{llm.version(name)}\n{prompt}".encode()).hexdigest()
    if cache_path is not None and not refresh:
        hits = {row["key"]: row["reply"] for row in read_json_lines(cache_path)}
        if key in hits and not wrong(shape.model_validate(hits[key])):
            return shape.model_validate(hits[key])
    for _ in range(2):
        reply = llm.ask("judge", prompt, shape)
        problem = wrong(reply)
        if not problem:
            break
    else:
        raise llm.BadReply(problem)
    if cache_path is not None:
        append_json_line(cache_path, {"key": key, "reply": reply.model_dump()})
    return reply


def _inert(s: str) -> str:
    """Target-supplied text goes inside <answer> and [context] markers; stop it from closing them."""
    return s.replace("<", "‹").replace("[", "❲")


def answer_check(q: Question, answer: str, cache_path: Path | None = None, refresh: bool = False) -> AnswerCheck:
    """Call A. refresh ignores the stored reply and stores the new one."""
    points = q.gold_ans if q.answerable else []
    prompt = llm.render("answer_check", lang=q.lang, question=q.resolved_question, key_points=points, answer=_inert(answer))
    return _cached("answer_check", prompt, AnswerCheck, cache_path, refresh,
                   lambda r: None if len(r.key_points) == len(points)
                   else f"the judge returned {len(r.key_points)} key points for {len(points)}")


def faithfulness(answer: str, contexts: list[str], cache_path: Path | None = None,
                 refresh: bool = False) -> list[Claim]:
    """Call B: the claims in the answer and whether the contexts support each."""
    shown = "\n".join(f"[context {i}]\n{_inert(c)}\n[end of context {i}]" for i, c in enumerate(contexts, 1))
    prompt = llm.render("faithfulness", answer=_inert(answer), contexts=shown)
    return _cached("faithfulness", prompt, _Claims, cache_path, refresh).claims
