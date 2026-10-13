"""The `rageval score` job: code metrics and judge calls for every stored answer (architecture 3.10)."""
from openai import APIStatusError

from . import judge, jobs, llm, metrics, report, run as run_mod, store
from .schema import Answer, Question, Score


def _judged(q: Question, ans: Answer, answers: dict[str, Answer], cache, refresh: bool) -> dict:
    """Judge fields of a score row; any unusable judge reply turns the whole question into a judge_error."""
    try:
        check = judge.answer_check(q, ans.answer or "", cache, refresh)
        out = {
            "refused": check.refused, "answer_lang": check.answer_lang,
            "language_match": int(check.answer_lang.strip().lower() == q.lang.lower()),
        }
        if not q.answerable:
            return out
        out |= {"answer_recall": check.answer_recall, "contradiction": check.contradiction,
                "key_points": check.key_points}
        contexts = metrics.judge_contexts(q, answers)
        if not check.refused and ans.contexts:
            claims = judge.faithfulness(ans.answer or "", contexts, cache, refresh)
            out |= {"claims": claims, "faithfulness": sum(c.supported for c in claims) / len(claims) if claims else None}
        return out
    except llm.BadReply as e:
        return {"judge_error": str(e)}
    except APIStatusError as e:
        if 400 <= e.status_code < 500 and e.status_code != 429:  # this question cannot be judged (too long, filtered)
            return {"judge_error": f"the judge provider refused the request (HTTP {e.status_code})"}
        raise  # 429 and 5xx were already retried by the SDK: stop, and a restart resumes


def _decisions(s: Score) -> list:
    """Every verdict the judge gave for a row, in a comparable order."""
    return [s.refused, *(k.verdict for k in s.key_points or []), *(c.supported for c in s.claims or [])]


def _flip_rate(old: dict[str, Score], new: list[Score]) -> float | None:
    """Share of judge verdicts that differ from the previous scores; a changed list length counts all as changed."""
    total = changed = 0
    for s in new:
        o = old.get(s.id)
        if o is None or o.judge_error or s.judge_error or o.error or s.error:
            continue
        a, b = _decisions(o), _decisions(s)
        total += max(len(a), len(b))
        changed += sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    return changed / total if total else None


def run(run_id: str, refresh_judge: bool = False) -> dict:
    """Score a run, write scores.jsonl and report.md, and return report.summary(run_id)."""
    rd = store.runs_dir() / run_id
    manifest = store.read_json(rd / "manifest.json")
    if not manifest:
        raise SystemExit(f"There is no run {run_id}. Check the id.")
    run_mod.check_dataset(manifest["dataset_id"], manifest["dataset_sha256"])
    questions = store.read_jsonl(store.datasets_dir() / manifest["dataset_id"] / "questions.jsonl", Question)
    answers = run_mod.latest_answers(run_id)
    old = {s.id: s for s in store.read_jsonl(rd / "scores.jsonl", Score)}
    todo = [q for q in questions if q.id in answers]
    rows: list[Score] = []
    with jobs.Job(rd, "score") as job:
        job.step("score", len(todo))
        for q in todo:
            ans = answers[q.id]
            if ans.error:
                row = Score(id=q.id, base_id=q.base_id, error=ans.error)
            else:
                fields = metrics.retrieval(q, None if ans.contexts is None else [c.text for c in ans.contexts],
                                           manifest["top_k"])
                fields |= _judged(q, ans, answers, rd / "judge_cache.jsonl", refresh_judge)
                if "judge_error" not in fields:
                    fields |= {"judge_model": llm.model_name("judge"),
                               "judge_prompt_versions": {n: llm.version(n) for n in ("answer_check", "faithfulness")}}
                row = Score(id=q.id, base_id=q.base_id, **fields)
            rows.append(row)
            job.tick(q.id)
    store.write_jsonl(rd / "scores.jsonl", rows)
    report.write(run_id)
    result = report.summary(run_id)
    if refresh_judge:
        result["flip_rate"] = _flip_rate(old, rows)
    return result
