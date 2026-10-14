"""The `rageval run` job: ask every frozen question and store the raw answers (architecture 3.8)."""
import time
from datetime import datetime, timezone

from . import adapters, dataset
from .. import jobs, settings, store, text
from ..schema import Answer, Context, Question

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 1.0
MAX_CONSECUTIVE_ERRORS = 10
SEED = 0


def latest_answers(run_id: str) -> dict[str, Answer]:
    """The latest row per question id; that is the one scored."""
    return {a.id: a for a in store.read_jsonl(store.runs_dir() / run_id / "answers.jsonl", Answer)}


def _ask(ask, q: Question, history: list[dict]) -> Answer:
    """One question with retries; a permanent failure stops the run."""
    error = ""
    for attempt in range(MAX_ATTEMPTS):
        started = time.monotonic()
        try:
            reply = ask(q.question, history)
        except adapters.Transient as e:
            error = str(e)
            if attempt < MAX_ATTEMPTS - 1:
                time.sleep(BACKOFF_SECONDS * 2 ** attempt)
            continue
        except adapters.BadReply as e:
            return Answer(id=q.id, error=str(e))
        except adapters.Permanent as e:
            raise SystemExit(f"The endpoint refused the request ({e}). Check RAG_URL and RAG_API_KEY in .env, "
                             "then start the run again to resume.")
        except jobs.Stopped:
            raise
        except Exception as e:  # e.g. a dropped connection mid-reply, or a bug in a custom adapter
            return Answer(id=q.id, error=f"{type(e).__name__}: {e}")
        contexts = reply.get("contexts")
        return Answer(id=q.id, answer=reply["answer"], latency_ms=round((time.monotonic() - started) * 1000),
                      contexts=None if contexts is None else [Context(text=c["text"]) for c in contexts])
    return Answer(id=q.id, error=f"gave up after {MAX_ATTEMPTS} attempts: {error}")


def check_dataset(dataset_id: str, expected_sha: str) -> None:
    """Refuse to run or score on a questions file that no longer matches the hash recorded when it was frozen."""
    if dataset.sha(dataset_id) != expected_sha:
        raise SystemExit(f"The questions of dataset {dataset_id} have changed since the run started (hash mismatch). "
                         "Restore the original dataset.csv, or start a new run.")


def run(dataset_id: str | None = None, label: str = "", run_id: str | None = None) -> str:
    """Start a run on a frozen dataset, or resume `run_id`. Returns the run id."""
    if run_id:
        manifest = store.read_json(store.runs_dir() / run_id / "manifest.json")
        if not manifest:
            raise SystemExit(f"There is no run {run_id}. Check the id, or start a new run with a dataset.")
        dataset_id, s = manifest["dataset_id"], settings.Settings.model_validate(manifest["settings"])
        if s.rag_adapter == "http_json" and adapters.public_url() != manifest["rag_url"]:
            raise SystemExit(f"RAG_URL is now {adapters.public_url() or 'empty'} but this run used {manifest['rag_url']}. "
                             "Set RAG_URL back to the original address to resume, or start a new run.")
    else:
        if not dataset_id:
            raise SystemExit("Name the dataset to run on.")
        s, run_id, manifest = settings.load(), store.new_id(label), None
    if not dataset.exists(dataset_id):
        raise SystemExit(f"There is no datasets/{dataset_id}/dataset.csv. Export one with `rageval-datagen freeze`.")
    dataset_sha = dataset.expected_sha(dataset_id)
    check_dataset(dataset_id, manifest["dataset_sha256"] if manifest else dataset_sha)
    questions = dataset.load(dataset_id)
    by_id = {q.id: q for q in questions}
    rd = store.runs_dir() / run_id
    if manifest is None:
        store.write_json(rd / "manifest.json", {
            "run_id": run_id, "dataset_id": dataset_id, "dataset_sha256": dataset_sha,
            "settings": s.model_dump(), "rag_url": adapters.public_url(), "adapter": s.rag_adapter, "label": label,
            "top_k": s.top_k, "matcher": {"window": text.WINDOW, "coverage": text.COVERAGE}, "seed": SEED,
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    if not s.supports_history:
        print("Follow-up questions are skipped: supports_history is off in settings.json.", flush=True)
    ask = adapters.get(s.rag_adapter)
    got = latest_answers(run_id)
    wanted = [q for q in questions if not q.is_follow_up] + [q for q in questions if q.is_follow_up and s.supports_history]
    todo = [q for q in wanted if q.id not in got or got[q.id].error]
    in_a_row = 0
    with jobs.Job(rd, "run") as job:
        job.step("ask", len(todo))
        for q in todo:
            history: list[dict] = []
            skipped = False
            if q.is_follow_up:
                first, first_q = got.get(q.previous_id), by_id.get(q.previous_id)
                if first is None or first.error or first_q is None:
                    row, skipped = Answer(id=q.id, error="first turn failed"), True
                else:
                    history = [{"question": first_q.question, "answer": first.answer}]
            if not skipped:
                row = _ask(ask, q, history)
                in_a_row = in_a_row + 1 if row.error else 0
            got[q.id] = row
            store.append_jsonl(rd / "answers.jsonl", row)
            if in_a_row >= MAX_CONSECUTIVE_ERRORS:
                raise SystemExit(f"Stopped after {MAX_CONSECUTIVE_ERRORS} errors in a row; last: {row.error}. "
                                 "Check that the endpoint is up, then start the run again to resume.")
            job.tick(q.id)
    return run_id
