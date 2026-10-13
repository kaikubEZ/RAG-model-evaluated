"""`rageval translate`: variants of approved questions in the extra test languages (architecture 3.5)."""
import hashlib
import json

from pydantic import BaseModel, ValidationError

from . import jobs, llm, settings, store
from .schema import Question

BATCH = 10


class _Translated(BaseModel):
    id: str
    question: str
    resolved_question: str
    gold_ans: list[str]


class _Checked(BaseModel):
    id: str
    question_same: bool
    key_points_same: list[bool]


class _Reply(BaseModel):
    items: list[dict]  # validated one by one, so a single malformed item only fails that question


def _valid(reply: _Reply, shape: type[BaseModel]) -> dict:
    out = {}
    for raw in reply.items:
        try:
            item = shape.model_validate(raw)
        except ValidationError:
            continue
        out[item.id] = item
    return out


def _fingerprint(q: Question) -> str:
    """What a dropped variant was dropped for; an edited source question gets a fresh try."""
    return hashlib.sha256(json.dumps([q.question, q.resolved_question, q.gold_ans]).encode()).hexdigest()[:12]


def _todo(rows: list[Question], build: dict, lang: str) -> list[Question]:
    """Approved source-language questions that still need a variant in `lang`; first turns before follow-ups."""
    have = {r.id for r in rows}
    approved = {r.id: r for r in rows if r.status == "approved" and not r.is_translated}
    dropped = build.get("translation_dropped", {})

    def gone(q: Question) -> bool:
        return dropped.get(f"{q.base_id}.{lang}") == _fingerprint(q)

    out = []
    for q in approved.values():
        if q.source_lang == lang or f"{q.base_id}.{lang}" in have or gone(q):
            continue
        if q.is_follow_up:
            first = approved.get(q.previous_id)
            if first is None or (first.source_lang != lang and gone(first)):
                continue
        out.append(q)
    return sorted(out, key=lambda q: q.is_follow_up)


def missing(dataset_id: str) -> list[tuple[str, str]]:
    """(base_id, lang) pairs that still need a variant. Variants reported as dropped are not missing."""
    d = store.datasets_dir() / dataset_id
    rows = store.read_jsonl(d / "candidates.jsonl", Question)
    build = store.read_json(d / "build.json")
    return [(q.base_id, lang) for lang in settings.load().extra_langs for q in _todo(rows, build, lang)]


def drop_variants(rows: list[Question], base_id: str) -> list[Question]:
    """Remove the translated variants of a question, used when its source is edited."""
    return [r for r in rows if not (r.base_id == base_id and r.is_translated)]


def _translate(batch: list[Question], lang: str) -> dict[str, _Translated]:
    """One translate call and one meaning check; only the items that pass are returned."""
    items = [{"id": q.id, "question": q.question, "resolved_question": q.resolved_question, "gold_ans": q.gold_ans}
             for q in batch]
    try:
        translated = _valid(llm.ask("gen", llm.render("translate", lang=lang, items=items), _Reply), _Translated)
        pairs = [{"id": q.id, "source": item, "translation": translated[q.id].model_dump()}
                 for q, item in zip(batch, items) if q.id in translated]
        checks = _valid(llm.ask("judge", llm.render("translation_check", items=pairs), _Reply), _Checked)
    except llm.BadReply:
        return {}
    ok = {}
    for q in batch:
        t, c = translated.get(q.id), checks.get(q.id)
        if t and c and c.question_same and len(t.gold_ans) == len(q.gold_ans) == len(c.key_points_same) and all(c.key_points_same):
            ok[q.id] = t
    return ok


def run(dataset_id: str) -> dict:
    """Add the missing variants. Existing variants are kept, so running it again only fills gaps."""
    d = store.datasets_dir() / dataset_id
    if store.is_frozen(dataset_id):
        raise SystemExit(f"Dataset {dataset_id} is frozen and cannot be changed. Duplicate it as a draft to add languages.")
    # candidates.jsonl is read once and rewritten after every batch; the job lock keeps the Review page read-only meanwhile
    rows = store.read_jsonl(d / "candidates.jsonl", Question)
    created = dropped = 0
    with jobs.Job(d, "translate") as job:
        build = store.read_json(d / "build.json")
        for lang in settings.load().extra_langs:
            todo = _todo(rows, build, lang)
            job.step(f"translate to {lang}", (len(todo) + BATCH - 1) // BATCH)
            for start in range(0, len(todo), BATCH):
                batch = todo[start:start + BATCH]
                done = _translate(batch, lang)
                if len(done) < len(batch):  # one more try for the ones that failed
                    done |= _translate([q for q in batch if q.id not in done], lang)
                for q in batch:
                    t = done.get(q.id)
                    previous = None
                    if q.is_follow_up:
                        first = next(r for r in rows if r.id == q.previous_id)
                        previous = first.id if first.source_lang == lang else f"{first.base_id}.{lang}"
                    if t is None or (previous and previous not in {r.id for r in rows}):
                        build.setdefault("translation_dropped", {})[f"{q.base_id}.{lang}"] = _fingerprint(q)
                        dropped += 1
                        continue
                    same = q.question == q.resolved_question
                    rows.append(q.model_copy(update={
                        "id": f"{q.base_id}.{lang}", "lang": lang, "question": t.question,
                        "resolved_question": t.question if same else t.resolved_question, "gold_ans": t.gold_ans,
                        "previous_id": previous, "status": "approved", "translation_reviewed": False,
                        "closed_book": False, "closest_passage": None, "note": None}))
                    created += 1
                store.write_jsonl(d / "candidates.jsonl", rows)
                store.write_json(d / "build.json", build)
                job.tick(lang)
    return {"created": created, "dropped": dropped}
