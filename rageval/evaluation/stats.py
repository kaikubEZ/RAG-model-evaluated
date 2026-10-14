"""Groups, per-question values, intervals and paired comparison (architecture 3.11)."""
import math

import numpy as np

from ..schema import SCENARIOS, Question, Score

METRICS: dict[str, dict] = {
    "recall_at_k": {"binary": False, "lower_is_better": False},
    "reciprocal_rank": {"binary": False, "lower_is_better": False},
    "answer_recall": {"binary": False, "lower_is_better": False},
    "contradiction": {"binary": True, "lower_is_better": True},
    "faithfulness": {"binary": False, "lower_is_better": False},
    "refusal_accuracy": {"binary": True, "lower_is_better": False},
    "false_refusal": {"binary": True, "lower_is_better": True},
    "language_match": {"binary": True, "lower_is_better": False},
}
RESAMPLES = 2000
NO_DIFFERENCE = "no detectable difference"
NOT_MEASURED_IN_BOTH = "not measured in both runs"


def groups(questions: list[Question]) -> dict[str, list[Question]]:
    """Scores are never pooled across these groups. Only non-empty groups are returned."""
    out = {"headline": [q for q in questions if not q.is_follow_up and not q.is_translated]}
    for sc in SCENARIOS:
        out[f"scenario={sc}"] = [q for q in questions if q.is_follow_up and not q.is_translated and q.scenario == sc]
    for d in sorted({f"{q.lang}-{q.source_lang}" for q in questions if q.is_translated}):
        out[f"direction={d}"] = [q for q in questions if q.is_translated and f"{q.lang}-{q.source_lang}" == d]
    return {k: v for k, v in out.items() if v}


def unit(q: Question, by_id: dict[str, Question]) -> str:
    """Resampling unit: the question's base_id, or its first turn's base_id for a follow-up."""
    return by_id.get(q.previous_id, q).base_id if q.is_follow_up else q.base_id


def value(metric: str, q: Question, s: Score) -> float | None:
    """Per-question value in the headline convention.

    An endpoint error scores 0 on the higher-is-better metrics that apply to the question. It is None for the
    lower-is-better ones and for language match: a failed call is not a contradiction, a refusal or a language.
    A judge error is None for every judged metric.
    """
    if metric in ("recall_at_k", "reciprocal_rank") and not q.answerable:
        return None
    if s.error:
        applies = q.answerable if metric in ("recall_at_k", "reciprocal_rank", "answer_recall", "faithfulness") \
            else not q.answerable and metric == "refusal_accuracy"
        return 0.0 if applies else None
    if metric in ("recall_at_k", "reciprocal_rank"):
        return getattr(s, metric)
    if s.judge_error:
        return None
    if metric == "refusal_accuracy":
        return None if q.answerable or s.refused is None else float(s.refused)
    if metric == "false_refusal":
        return None if not q.answerable or s.refused is None else float(s.refused)
    if metric == "language_match":
        return s.language_match
    if not q.answerable:
        return None
    return getattr(s, metric)


def values(metric: str, rows: list[tuple[Question, Score]]) -> dict[str, float]:
    """Per-question values by id; empty when no answered question has one.

    Without that rule an errored question would score 0 on a metric the run never measured,
    for example recall when the endpoint returns no contexts.
    """
    out = {q.id: v for q, s in rows if (v := value(metric, q, s)) is not None}
    return out if any(not s.error and q.id in out for q, s in rows) else {}


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float, float]:
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, centre - half), min(1.0, centre + half)


def _bootstrap(values: list[float], units: list[str], seed: int) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean, resampling whole units."""
    names = sorted(set(units))
    idx = np.array([names.index(u) for u in units])
    sums = np.bincount(idx, weights=np.array(values, dtype=float), minlength=len(names))
    counts = np.bincount(idx, minlength=len(names))
    draw = np.random.default_rng(seed).integers(0, len(names), size=(RESAMPLES, len(names)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(np.mean(values)), float(low), float(high)


def interval(metric: str, values: list[float], units: list[str], seed: int = 0) -> tuple[float, float, float]:
    """(mean, low, high): Wilson for yes/no metrics, percentile bootstrap by unit otherwise."""
    if METRICS[metric]["binary"]:
        return wilson(sum(values), len(values))
    return _bootstrap(values, units, seed)


def compare(a: dict[str, float], b: dict[str, float], units: dict[str, str], seed: int = 0) -> dict:
    """Paired B minus A over the ids in both. The verdict is sign-neutral: higher, lower or no detectable difference."""
    ids = sorted(a.keys() & b.keys())
    if not ids:
        return {"n": 0, "units": 0, "diff": None, "low": None, "high": None, "detectable": None, "verdict": NOT_MEASURED_IN_BOTH}
    diffs = [b[i] - a[i] for i in ids]
    us = [units[i] for i in ids]
    mean, low, high = _bootstrap(diffs, us, seed)
    names = sorted(set(us))
    idx = np.array([names.index(u) for u in us])
    unit_means = np.bincount(idx, weights=np.array(diffs), minlength=len(names)) / np.bincount(idx, minlength=len(names))
    sd = float(unit_means.std(ddof=1)) if len(names) > 1 else 0.0
    return {
        "n": len(ids), "units": len(names), "diff": mean, "low": low, "high": high,
        "detectable": 2.8 * sd / math.sqrt(len(names)) if sd > 1e-12 else None,
        "verdict": NO_DIFFERENCE if low <= 0 <= high else ("higher" if mean > 0 else "lower"),
    }
