"""`rageval freeze`: pick the approved questions, flag closed-book ones, write the frozen dataset (architecture 3.6)."""
import hashlib
import shutil
from datetime import datetime, timezone

from . import jobs, llm, settings, store, translate
from .schema import SCENARIOS, Question
from .settings import Settings


def select(rows: list[Question], s: Settings) -> list[Question]:
    """Approved rows to freeze, in candidate order.

    Per subject: follow-ups up to the target per scenario, then their first turns even beyond a type's target,
    then other single questions until each type reaches its target; plus the approved variants of what was taken.
    """
    chosen: set[str] = set()
    for subject in dict.fromkeys(r.subject for r in rows):
        mine = [r for r in rows if r.subject == subject and r.status == "approved" and not r.is_translated]
        by_id = {r.id: r for r in mine}
        taken = dict.fromkeys(SCENARIOS, 0)
        picked: dict[str, Question] = {}
        for q in mine:
            if q.is_follow_up and q.previous_id in by_id and taken[q.scenario] < s.follow_ups_per_subject.get(q.scenario, 0):
                taken[q.scenario] += 1
                picked[q.id], picked[q.previous_id] = q, by_id[q.previous_id]
        per_type: dict[str, int] = {}
        for q in picked.values():
            if not q.is_follow_up:
                per_type[q.question_type] = per_type.get(q.question_type, 0) + 1
        for q in mine:
            if not q.is_follow_up and q.id not in picked and per_type.get(q.question_type, 0) < s.questions_per_subject.get(q.question_type, 0):
                per_type[q.question_type] = per_type.get(q.question_type, 0) + 1
                picked[q.id] = q
        chosen |= picked.keys()
    variants = [r for r in rows if r.is_translated and r.status == "approved" and r.base_id in chosen]
    chosen |= {v.id for v in variants if not v.is_follow_up}
    chosen |= {v.id for v in variants if v.is_follow_up and v.previous_id in chosen}
    return [r for r in rows if r.id in chosen]


def _counts(rows: list[Question]) -> dict[str, dict[str, int]]:
    """Source-language rows per subject: single questions by type, follow-ups by scenario."""
    out: dict[str, dict[str, int]] = {}
    for q in rows:
        if not q.is_translated:
            name = q.scenario if q.is_follow_up else q.question_type
            out.setdefault(q.subject, {})[name] = out.get(q.subject, {}).get(name, 0) + 1
    return out


def run(dataset_id: str) -> dict:
    """Freeze the dataset. Writes questions.jsonl last, so an interrupted freeze leaves nothing half-frozen."""
    d = store.datasets_dir() / dataset_id
    if store.is_frozen(dataset_id):
        raise SystemExit(f"Dataset {dataset_id} is already frozen. Use 'duplicate as draft' to edit a copy.")
    gaps = translate.missing(dataset_id)
    if gaps:
        raise SystemExit(f"{len(gaps)} translations are missing (first: {gaps[0][0]} in {gaps[0][1]}). Run translate first.")
    s = settings.load()
    chosen = select(store.read_jsonl(d / "candidates.jsonl", Question), s)
    if not chosen:
        raise SystemExit("No question is approved yet. Approve questions on the Review page, then freeze.")
    with jobs.Job(d, "freeze") as job:
        from . import judge
        build = store.read_json(d / "build.json")
        flags = build.setdefault("closed_book", {})
        todo = [q for q in chosen if q.answerable and not q.is_translated and q.id not in flags]
        job.step("closed-book check", len(todo))
        for q in todo:
            answer = llm.call("gen", "Answer the question below from your general knowledge, in a few sentences. "
                                     f"You are given no documents.\n\nQuestion: {q.resolved_question}")
            try:
                flags[q.id] = judge.answer_check(q, answer).answer_recall >= 1.0
            except llm.BadReply:  # an unusable judge reply must not make the dataset impossible to freeze
                flags[q.id] = False
                build.setdefault("closed_book_unchecked", []).append(q.id)
            store.write_json(d / "build.json", build)
            job.tick(q.id)
        final = [q.model_copy(update={"closed_book": flags.get(q.base_id, False)}) for q in chosen]
        body = "".join(q.model_dump_json() + "\n" for q in final)
        counts = _counts(final)
        targets = {**s.questions_per_subject, **s.follow_ups_per_subject}
        translated = [q for q in final if q.is_translated]
        manifest = {
            "dataset_id": dataset_id,
            "sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "settings": s.model_dump(),
            "generator_model": llm.model_name("gen"),
            "prompt_versions": {p.stem: llm.version(p.stem) for p in sorted(llm.PROMPTS.glob("*.txt"))},
            "counts": counts,
            "shortfall": build.get("shortfall", {}),  # at generation
            "frozen_shortfall": {sub: {n: t - c.get(n, 0) for n, t in targets.items() if t > c.get(n, 0)}
                                 for sub, c in counts.items()},
            "dropped": build.get("dropped", {}),
            "languages": list(dict.fromkeys(q.lang for q in final)),
            "bulk_approved": build.get("bulk_approved", 0),
            "scenario_overrides": build.get("scenario_overrides", 0),
            "closed_book_unchecked": sum(q.id in build.get("closed_book_unchecked", []) for q in chosen),
            "translations": len(translated),
            "translations_reviewed": sum(q.translation_reviewed for q in translated),
            "created": datetime.now(timezone.utc).date().isoformat(),
        }
        store.write_json(d / "manifest.json", manifest)
        store.write_text_atomic(d / "questions.jsonl", body)
    return manifest


def duplicate_as_draft(dataset_id: str) -> str:
    """Copy a frozen dataset's questions into a new, editable dataset and return its id."""
    src = store.datasets_dir() / dataset_id
    if not store.is_frozen(dataset_id):
        raise SystemExit(f"Dataset {dataset_id} is not frozen, so it can be edited directly.")
    new_id = store.new_id("draft")
    dst = store.datasets_dir() / new_id
    store.write_jsonl(dst / "candidates.jsonl", store.read_jsonl(src / "questions.jsonl", Question))
    if (src / "pages.jsonl").exists():
        shutil.copyfile(src / "pages.jsonl", dst / "pages.jsonl")
    return new_id
