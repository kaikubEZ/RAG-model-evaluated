"""The logic behind the Review page: load, save, approve, summarise (architecture 3.6). No Streamlit here."""
from pathlib import Path

from . import jobs, settings, store, translate, validate
from .schema import Page, Question


def _dir(dataset_id: str) -> Path:
    return store.datasets_dir() / dataset_id


def load(dataset_id: str) -> list[Question]:
    name = "questions.jsonl" if store.is_frozen(dataset_id) else "candidates.jsonl"
    return store.read_jsonl(_dir(dataset_id) / name, Question)


def pages(dataset_id: str) -> dict[tuple[str, int], str]:
    return {(p.doc, p.page): p.text for p in store.read_jsonl(_dir(dataset_id) / "pages.jsonl", Page)}


def _writable(dataset_id: str) -> Path:
    d = _dir(dataset_id)
    if store.is_frozen(dataset_id):
        raise ValueError(f"Dataset {dataset_id} is frozen and cannot be edited. Use 'duplicate as draft' to edit a copy.")
    if jobs.is_running(d):
        raise ValueError("A job is running on this dataset. Wait for it to finish, or stop it, before editing.")
    return d


def _fresh_id(build: dict, rows: list[Question], subject: str) -> str:
    taken = {r.id for r in rows}
    n = build.setdefault("next_id", {}).get(subject, 0)
    while True:
        n += 1
        if f"{subject}-{n:04d}" not in taken:
            break
    build["next_id"][subject] = n
    return f"{subject}-{n:04d}"


def save_row(dataset_id: str, q: Question) -> str | None:
    """Save one edited or new row. Returns the reason when a check failed (the row is then kept as a candidate)."""
    d = _writable(dataset_id)
    rows = store.read_jsonl(d / "candidates.jsonl", Question)
    build = store.read_json(d / "build.json")
    old = next((r for r in rows if r.id == q.id), None)
    if old is None:
        new_id = _fresh_id(build, rows, q.subject)
        q = q.model_copy(update={"id": new_id, "base_id": new_id})
    others = [r for r in rows if r.subject == q.subject and r.id != q.id]
    reason = validate.check_row(q, pages(dataset_id), others, strict_scenario=False)
    update: dict = {"note": reason, "status": "candidate" if reason else q.status}
    if not q.is_translated:
        update["difficulty"] = validate.difficulty(q) if q.answerable else None
    q = q.model_copy(update=update)
    first = next((r for r in others if r.id == q.previous_id), None)
    if first is not None and not validate.scenario_ok(q, first):
        ids = build.setdefault("scenario_override_ids", [])
        if q.id not in ids:
            ids.append(q.id)
            build["scenario_overrides"] = build.get("scenario_overrides", 0) + 1
    if old is None:
        rows.append(q)
    else:
        rows = [q if r.id == q.id else r for r in rows]
        if (old.status == "approved" and not old.is_translated
                and (old.question, old.resolved_question, old.gold_ans) != (q.question, q.resolved_question, q.gold_ans)):
            rows = translate.drop_variants(rows, q.base_id)
    store.write_jsonl(d / "candidates.jsonl", rows)
    store.write_json(d / "build.json", build)
    return reason


def set_status(dataset_id: str, ids: list[str], status: str, bulk: bool = False) -> int:
    """Approve or reject rows. Rows with a failed check cannot be approved. Returns how many rows changed."""
    d = _writable(dataset_id)
    rows = store.read_jsonl(d / "candidates.jsonl", Question)
    wanted = set(ids)
    if status == "rejected":  # a rejected source question takes its variants with it
        wanted |= {r.id for r in rows if r.is_translated and r.base_id in wanted}
    out, changed = [], 0
    for r in rows:
        if r.id in wanted and r.status != status and not (status == "approved" and r.note):
            r = r.model_copy(update={"status": status})
            changed += 1
        out.append(r)
    store.write_jsonl(d / "candidates.jsonl", out)
    if bulk and changed:
        build = store.read_json(d / "build.json")
        build["bulk_approved"] = build.get("bulk_approved", 0) + changed
        store.write_json(d / "build.json", build)
    return changed


def mark_reviewed(dataset_id: str, variant_id: str) -> None:
    d = _writable(dataset_id)
    rows = store.read_jsonl(d / "candidates.jsonl", Question)
    if not any(r.id == variant_id and r.is_translated for r in rows):
        raise ValueError(f"{variant_id} is not a translated variant.")
    store.write_jsonl(d / "candidates.jsonl", [
        r.model_copy(update={"translation_reviewed": True}) if r.id == variant_id else r for r in rows])


def summary(dataset_id: str) -> dict:
    """Counts by subject, type, scenario, language and status; approved source questions against the targets."""
    rows = load(dataset_id)
    s = settings.load()

    def count(key) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            if (k := key(r)) is not None:
                out[k] = out.get(k, 0) + 1
        return out

    approved = [r for r in rows if r.status == "approved" and not r.is_translated]
    targets = {**s.questions_per_subject, **s.follow_ups_per_subject}
    shortfall = {}
    for subject in sorted({r.subject for r in rows}):
        have: dict[str, int] = {}
        for r in approved:
            if r.subject == subject:
                name = r.scenario if r.is_follow_up else r.question_type
                have[name] = have.get(name, 0) + 1
        shortfall[subject] = {n: t - have.get(n, 0) for n, t in targets.items() if t > have.get(n, 0)}
    build = store.read_json(_dir(dataset_id) / "build.json")
    return {
        "total": len(rows), "by_subject": count(lambda r: r.subject), "by_type": count(lambda r: r.question_type),
        "by_scenario": count(lambda r: r.scenario), "by_language": count(lambda r: r.lang),
        "by_status": count(lambda r: r.status), "approved": len(approved), "targets": targets, "shortfall": shortfall,
        "bulk_approved": build.get("bulk_approved", 0), "scenario_overrides": build.get("scenario_overrides", 0),
    }
