"""`rageval-datagen translate`, stage 3: variants of approved questions in the extra test languages (architecture 3.5)."""
import hashlib
import json
import re

from pydantic import BaseModel, ValidationError

from .. import jobs, llm, settings, store
from ..schema import Question

BATCH = 10
BASE_LANG = "en"
# the writing system of a target language: its translation must contain at least one of these characters
SCRIPTS = {"th": "\u0e00-\u0e7f", "zh": "\u4e00-\u9fff", "ja": "\u3040-\u30ff\u4e00-\u9fff", "ko": "\uac00-\ud7af",
           "ru": "\u0400-\u04ff", "uk": "\u0400-\u04ff", "ar": "\u0600-\u06ff", "hi": "\u0900-\u097f",
           "he": "\u0590-\u05ff", "el": "\u0370-\u03ff"}
REMINDER = ("Your previous reply left some of these items in English. Translate every word of \"question\", "
            "\"resolved_question\" and \"reference_answer\" into the target language; copying the English text is wrong.")


def is_translated(source: str, out: str, lang: str) -> bool:
    """A translation must differ from its source and, for a language with its own script, be written in it."""
    if not source.strip():
        return True
    if out.strip() == source.strip():
        return False
    script = SCRIPTS.get(lang.lower().split("-")[0])
    return bool(re.search(f"[{script}]", out)) if script else True


def _variant_ok(src: Question, v) -> bool:
    """`v` is a translated Question or _Translated; the question, its resolved form and the answer must really be translated."""
    lang = getattr(v, "lang", None)
    return all(is_translated(a, b or "", lang) for a, b in (
        (src.question, v.question), (src.resolved_question, v.resolved_question), (src.reference_answer or "", v.reference_answer or "")))


def parse_langs(raw: str) -> list[str]:
    """Language codes typed by the user, "th, zh" or "th zh", as a clean list; raises ValueError on a bad code."""
    codes = list(dict.fromkeys(c.lower() for c in re.split(r"[,\s;]+", raw) if c))
    bad = [c for c in codes if not re.fullmatch(r"[a-z]{2,3}(-[a-z0-9]{2,8})?", c)]
    if bad:
        raise ValueError(f"Not a language code: {', '.join(bad)}. Use codes such as th, zh, ja, fr.")
    if BASE_LANG in codes:
        raise ValueError(f"{BASE_LANG} is the baseline language of the questions; leave it out.")
    return codes


class _Translated(BaseModel):
    id: str
    question: str
    resolved_question: str
    reference_answer: str
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
    return hashlib.sha256(json.dumps([q.question, q.resolved_question, q.reference_answer, q.gold_ans]).encode()).hexdigest()[:12]


def _todo(rows: list[Question], build: dict, lang: str) -> list[Question]:
    """Approved source-language questions with approved answers that still need a variant in `lang`; first turns first."""
    have = {r.id for r in rows}
    approved = {r.id: r for r in rows if r.status == "approved" and r.answer_ready and not r.is_translated}
    dropped = build.get("translation_dropped", {})

    def gone(q: Question) -> bool:
        return dropped.get(f"{q.base_id}-{lang}") == _fingerprint(q)

    out = []
    for q in approved.values():
        if q.source_lang == lang or f"{q.base_id}-{lang}" in have or gone(q):
            continue
        if q.is_follow_up:
            first = approved.get(q.previous_id)
            if first is None or (first.source_lang != lang and gone(first)):
                continue
        out.append(q)
    return sorted(out, key=lambda q: q.is_follow_up)


def untranslated(rows: list[Question]) -> set[str]:
    """Ids of variants nobody has reviewed whose question or answer is still the source text."""
    by_id = {r.id: r for r in rows}
    out = set()
    for r in rows:
        src = by_id.get(f"{r.base_id}-{r.source_lang}")
        if r.is_translated and not r.translation_reviewed and src and not _variant_ok(src, r):
            out.add(r.id)
    return out


def _clean(rows: list[Question]) -> list[Question]:
    bad = untranslated(rows)
    bad |= {r.id for r in rows if r.previous_id in bad}  # a follow-up points at its first turn's variant
    return [r for r in rows if r.id not in bad]


def missing(dataset_id: str) -> list[tuple[str, str]]:
    """(base_id, lang) pairs that still need a variant. Variants reported as dropped are not missing."""
    d = store.datasets_dir() / dataset_id
    rows = _clean(store.read_jsonl(d / "candidates.jsonl", Question))
    build = store.read_json(d / "build.json")
    return [(q.base_id, lang) for lang in settings.load().extra_langs for q in _todo(rows, build, lang)]


def drop_variants(rows: list[Question], base_id: str) -> list[Question]:
    """Remove the translated variants of a question, used when its source is edited."""
    return [r for r in rows if not (r.base_id == base_id and r.is_translated)]


def _translate(batch: list[Question], lang: str, retry: bool = False) -> dict[str, _Translated]:
    """One translate call and the checks; only the items that pass are returned."""
    items = [{"id": q.id, "question": q.question, "resolved_question": q.resolved_question,
              "reference_answer": q.reference_answer or "", "gold_ans": q.gold_ans}
             for q in batch]
    try:
        translated = _valid(llm.ask("gen", llm.render("translate", lang=lang, items=items, reminder=REMINDER if retry else ""), _Reply), _Translated)
        pairs = [{"id": q.id, "source": item, "translation": translated[q.id].model_dump()}
                 for q, item in zip(batch, items) if q.id in translated]
        checks = (_valid(llm.ask("judge", llm.render("translation_check", items=pairs), _Reply), _Checked)
                  if settings.load().judge_checks else None)
    except llm.BadReply:
        return {}
    ok = {}
    for q in batch:
        t = translated.get(q.id)
        if t and not all(is_translated(a, b, lang) for a, b in (
                (q.question, t.question), (q.resolved_question, t.resolved_question), (q.reference_answer or "", t.reference_answer))):
            continue  # still English: counts as failed, so it is asked again and then reported
        if checks is None:  # judge off: the translation is kept for the reviewer to read
            if t and len(t.gold_ans) == len(q.gold_ans):
                ok[q.id] = t
            continue
        c = checks.get(q.id)
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
        before = len(rows)
        rows = _clean(rows)  # variants that were left in English by an earlier run are made again
        redone = before - len(rows)
        build.pop("translation_dropped", None)  # pressing Translate tries every missing variant again
        for lang in settings.load().extra_langs:
            todo = _todo(rows, build, lang)
            job.step(f"translate to {lang}", (len(todo) + BATCH - 1) // BATCH)
            for start in range(0, len(todo), BATCH):
                batch = todo[start:start + BATCH]
                done = _translate(batch, lang)
                if len(done) < len(batch):  # one more try for the ones that failed
                    done |= _translate([q for q in batch if q.id not in done], lang, retry=True)
                for q in batch:
                    t = done.get(q.id)
                    previous = None
                    if q.is_follow_up:
                        first = next(r for r in rows if r.id == q.previous_id)
                        previous = first.id if first.source_lang == lang else f"{first.base_id}-{lang}"
                    if t is None or (previous and previous not in {r.id for r in rows}):
                        build.setdefault("translation_dropped", {})[f"{q.base_id}-{lang}"] = _fingerprint(q)
                        dropped += 1
                        continue
                    same = q.question == q.resolved_question
                    rows.append(q.model_copy(update={
                        "id": f"{q.base_id}-{lang}", "lang": lang, "question": t.question,
                        "resolved_question": t.question if same else t.resolved_question, "gold_ans": t.gold_ans,
                        "reference_answer": t.reference_answer if q.answerable else None,
                        "answer_status": "approved" if q.answerable else "none",
                        "previous_id": previous, "status": "approved", "translation_reviewed": False,
                        "closed_book": False, "closest_passage": None, "note": None}))
                    created += 1
                store.write_jsonl(d / "candidates.jsonl", rows)
                store.write_json(d / "build.json", build)
                job.tick(lang)
    return {"created": created, "dropped": dropped, "redone": redone}
