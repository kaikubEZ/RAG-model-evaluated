"""Human labels for the judge and the evidence matcher (architecture 3.12)."""
import hashlib
import random
from pathlib import Path

from . import llm, metrics, run as run_mod, stats, store, text
from .judge import append_json_line as _append, read_json_lines as _read
from .schema import Question, Score

# decision type -> (judge call whose prompt it belongs to, allowed verdict values)
TYPES = {
    "key_point": ("answer_check", ("supported", "contradicted", "absent")),
    "claim": ("faithfulness", ("supported", "unsupported")),
    "refusal": ("answer_check", ("refused", "not_refused")),
}
MIN_TYPE_LABELS = 30
MIN_VALUE_LABELS = 10
MIN_OVERALL, MIN_TYPE = 0.85, 0.80
MIN_MATCHER_LABELS = 50


def judge_file(call: str, model: str | None = None, version: str | None = None) -> Path:
    """Labels belong to a judge model and the prompt version of one call; default to the ones in use now."""
    name = f"{store.safe_name(model or llm.model_name('judge'))}__{call}__{version or llm.version(call)}.jsonl"
    return store.labels_dir() / "judge" / name


def _pair(kind: str, qid: str, answer: str) -> str:
    """One decision type never draws the same question-and-answer pair twice."""
    return f"{kind}|{qid}|{hashlib.sha256(answer.encode()).hexdigest()[:16]}"


def _pick(pool: list[dict], n: int, rng: random.Random, used: set[str]) -> list[dict]:
    """Up to n items, taking the judge's verdict values in turn so the sample is as balanced as the pool allows."""
    by_verdict: dict[str, list[dict]] = {}
    for item in rng.sample(pool, len(pool)):
        by_verdict.setdefault(item["judge_verdict"], []).append(item)
    out: list[dict] = []
    while len(out) < n and any(by_verdict.values()):
        for items in by_verdict.values():
            while items and items[-1]["pair"] in used:
                items.pop()
            if items and len(out) < n:
                item = items.pop()
                used.add(item["pair"])
                out.append(item)
    return out


def sample_judge(run_ids: list[str], lang: str, per_type: int = 50, seed: int = 0) -> list[dict]:
    """Unlabelled judge decisions for a person to label. "judge_verdict" is hidden from the person."""
    done = {r["pair"] for call in ("answer_check", "faithfulness") for r in _read(judge_file(call))}
    pools: dict[str, list[dict]] = {t: [] for t in TYPES}
    current = {"model": llm.model_name("judge"), **{c: llm.version(c) for c in ("answer_check", "faithfulness")}}
    for run_id in run_ids:
        manifest = store.read_json(store.runs_dir() / run_id / "manifest.json")
        by_id = {q.id: q for q in store.read_jsonl(
            store.datasets_dir() / manifest["dataset_id"] / "questions.jsonl", Question)}
        answers = run_mod.latest_answers(run_id)
        for s in store.read_jsonl(store.runs_dir() / run_id / "scores.jsonl", Score):
            q = by_id.get(s.id)
            if q is None or q.lang != lang or s.error or s.judge_error or s.refused is None:
                continue
            if s.judge_model != current["model"]:  # decisions of another judge are not ours to label
                continue
            fresh = {c: s.judge_prompt_versions.get(c) == current[c] for c in ("answer_check", "faithfulness")}
            ans = answers[s.id].answer or ""
            base = {"run_id": run_id, "id": s.id, "lang": lang, "question": q.resolved_question, "answer": ans}
            if fresh["answer_check"]:
                pools["refusal"].append({**base, "type": "refusal", "call": "answer_check",
                                         "pair": _pair("refusal", s.id, ans),
                                         "judge_verdict": "refused" if s.refused else "not_refused"})
                pools["key_point"] += [
                    {**base, "type": "key_point", "call": "answer_check", "pair": _pair("key_point", s.id, ans),
                     "text": k.point, "judge_verdict": k.verdict} for k in s.key_points or []]
            contexts = metrics.judge_contexts(q, answers)
            pools["claim"] += [
                {**base, "type": "claim", "call": "faithfulness", "pair": _pair("claim", s.id, ans), "text": c.claim,
                 "contexts": contexts, "cross_lang": s.answer_lang != q.source_lang,
                 "judge_verdict": "supported" if c.supported else "unsupported"}
                for c in (s.claims or []) if fresh["faithfulness"]]
    rng = random.Random(seed)
    out = []
    for kind, pool in pools.items():
        used = set(done)
        if kind == "claim":
            got = _pick([i for i in pool if i["cross_lang"]], per_type // 2, rng, used)  # answer and contexts differ
            got += _pick([i for i in pool if not i["cross_lang"]], per_type - len(got), rng, used)
            out += got + _pick(pool, per_type - len(got), rng, used)
        else:
            out += _pick(pool, per_type, rng, used)
    return out


def save_judge_label(item: dict, human_verdict: str) -> None:
    allowed = TYPES[item["type"]][1]
    if human_verdict not in allowed:
        raise ValueError(f"A {item['type']} label must be one of: {', '.join(allowed)}.")
    _append(judge_file(item["call"]), {**item, "human_verdict": human_verdict})


def _kappa(pairs: list[tuple[str, str]]) -> float | None:
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pe = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in {x for p in pairs for x in p})
    return None if pe == 1 else (po - pe) / (1 - pe)


def judge_agreement(lang: str, model: str | None = None, versions: dict[str, str] | None = None) -> dict:
    """Agreement between the judge and the person per decision type, and whether the judge is validated for lang.

    `model` and `versions` ({call: prompt version}) name the judge whose labels count; default: the ones in use now.
    """
    versions = versions or {}
    rows = [r for call in ("answer_check", "faithfulness") for r in _read(judge_file(call, model, versions.get(call)))
            if r["lang"] == lang]
    types, agree_all = {}, 0
    for kind, (_, values) in TYPES.items():
        mine = [r for r in rows if r["type"] == kind]
        n = len(mine)
        agree = sum(r["judge_verdict"] == r["human_verdict"] for r in mine)
        agree_all += agree
        entry = {"n": n, "agreement": None, "low": None, "high": None, "kappa": None, "validated": False,
                 "values": {v: {"n": sum(r["judge_verdict"] == v for r in mine),
                                "few": sum(r["judge_verdict"] == v for r in mine) < MIN_VALUE_LABELS} for v in values}}
        if n:
            entry["agreement"], entry["low"], entry["high"] = stats.wilson(agree, n)
            entry["kappa"] = _kappa([(r["judge_verdict"], r["human_verdict"]) for r in mine])
            entry["validated"] = n >= MIN_TYPE_LABELS and agree / n >= MIN_TYPE
        types[kind] = entry
    labelled = [t for t in types.values() if t["n"] >= MIN_TYPE_LABELS]
    overall = agree_all / len(rows) if rows else None
    return {
        "types": types, "overall": overall,
        "not_validated": [k for k, t in types.items() if t["n"] < MIN_TYPE_LABELS],
        "validated": bool(labelled) and overall >= MIN_OVERALL and all(t["agreement"] >= MIN_TYPE for t in labelled),
    }


def _matcher_file() -> Path:
    return store.labels_dir() / "matcher.jsonl"


def sample_matcher(run_id: str, n: int = 100, seed: int = 0) -> list[dict]:
    """Quote and top-k contexts for a person to label retrieved or not. "matcher_verdict" is hidden."""
    manifest = store.read_json(store.runs_dir() / run_id / "manifest.json")
    top_k = manifest["top_k"]
    by_id = {q.id: q for q in store.read_jsonl(store.datasets_dir() / manifest["dataset_id"] / "questions.jsonl", Question)}
    done = {(r["run_id"], r["id"], r["quote"]) for r in _read(_matcher_file())}
    pool = []
    for a in run_mod.latest_answers(run_id).values():
        q = by_id.get(a.id)
        if q is None or a.error or a.contexts is None or not q.answerable:
            continue
        shown = [c.text for c in a.contexts[:top_k]]
        for e in q.gold_evidence:
            if (run_id, a.id, e.quote) not in done:
                pool.append({"run_id": run_id, "id": a.id, "quote": e.quote, "contexts": shown,
                             "matcher_verdict": text.found_at(e.quote, shown) is not None})
    pool = random.Random(seed).sample(pool, len(pool))
    yes = [i for i in pool if i["matcher_verdict"]]
    no = [i for i in pool if not i["matcher_verdict"]]
    yes_n = min(len(yes), n // 2)  # half each, the larger side filling what the other lacks
    no_n = min(len(no), n - yes_n)
    return yes[:min(len(yes), n - no_n)] + no[:no_n]


def save_matcher_label(item: dict, retrieved: bool) -> None:
    _append(_matcher_file(), {**item, "retrieved": retrieved, "window": text.WINDOW, "coverage": text.COVERAGE})


def matcher_rates() -> dict:
    """How often the matcher misses a retrieved quote or accepts one that was not, for the constants in use."""
    rows = [r for r in _read(_matcher_file()) if (r["window"], r["coverage"]) == (text.WINDOW, text.COVERAGE)]
    yes = [r for r in rows if r["retrieved"]]
    no = [r for r in rows if not r["retrieved"]]
    return {
        "n_retrieved": len(yes), "n_not_retrieved": len(no),
        "miss_rate": sum(not r["matcher_verdict"] for r in yes) / len(yes) if yes else None,
        "false_accept_rate": sum(r["matcher_verdict"] for r in no) / len(no) if no else None,
        "calibrated": len(yes) >= MIN_MATCHER_LABELS and len(no) >= MIN_MATCHER_LABELS,
    }
