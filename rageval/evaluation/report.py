"""What the report and the app's Results page show, as plain data, plus report.md and run comparison (3.13)."""
from pathlib import Path

from . import adapters, dataset, labels, stats
from .. import store
from .run import SEED
from ..schema import Question, Score

MIN_CELL = 20
TOO_SMALL = "n too small"
NOT_MEASURED = "not measured"
# the decision type whose label validation each judged metric depends on
FED_BY = {"answer_recall": "key_point", "contradiction": "key_point", "faithfulness": "claim",
          "refusal_accuracy": "refusal", "false_refusal": "refusal", "language_match": "refusal"}


def _load(run_id: str) -> tuple[dict, list[Question], dict[str, Score]]:
    rd = store.runs_dir() / run_id
    manifest = store.read_json(rd / "manifest.json")
    if not manifest:
        raise SystemExit(f"There is no run {run_id}. Check the id.")
    scores = {s.id: s for s in store.read_jsonl(rd / "scores.jsonl", Score)}
    if not scores:
        raise SystemExit(f"Run {run_id} has no scores yet. Score it first.")
    questions = dataset.load(manifest["dataset_id"])
    return manifest, questions, scores


def _agg(metric: str, rows: list[tuple[Question, Score]], by_id: dict[str, Question]) -> dict | None:
    got = stats.values(metric, rows)
    pairs = [(got[q.id], stats.unit(q, by_id)) for q, _ in rows if q.id in got]
    if not pairs:
        return None
    values, units = zip(*pairs)
    mean, low, high = stats.interval(metric, list(values), list(units), SEED)
    return {"mean": mean, "low": low, "high": high, "n": len(values)}


def _metrics(rows: list[tuple[Question, Score]], by_id: dict[str, Question]) -> dict:
    """Per metric, with endpoint errors as zero ("all") and with errored questions left out ("answered")."""
    answered = [r for r in rows if not r[1].error]
    return {m: {"all": _agg(m, rows, by_id), "answered": _agg(m, answered, by_id)} for m in stats.METRICS}


def _context(rows: list[tuple[Question, Score]]) -> dict | None:
    seen = [s for _, s in rows if s.context_count is not None]
    if not seen:
        return None
    return {"n": len(seen), "mean_count": sum(s.context_count for s in seen) / len(seen),
            "mean_chars": sum(s.context_chars for s in seen) / len(seen)}


def summary(run_id: str) -> dict:
    manifest, questions, scores = _load(run_id)
    by_id = {q.id: q for q in questions}
    scored = [q for q in questions if q.id in scores]
    judge_model = next((s.judge_model for s in scores.values() if s.judge_model), None)
    judge_versions = next((s.judge_prompt_versions for s in scores.values() if s.judge_prompt_versions), {})
    langs = sorted({q.lang for q in scored})
    judge = {lang: labels.judge_agreement(lang, judge_model, judge_versions) for lang in langs} if judge_model else {}

    def not_validated(qs: list[Question]) -> list[str]:
        gl = {q.lang for q in qs}
        return sorted(m for m, t in FED_BY.items() if any(
            not judge[lang]["validated"] or not judge[lang]["types"][t]["validated"] for lang in gl)) if judge else []

    out_groups = {}
    for name, qs in stats.groups(scored).items():
        rows = [(q, scores[q.id]) for q in qs]
        entry = {"n": len(qs), "metrics": _metrics(rows, by_id), "context": _context(rows),
                 "judge_not_validated": not_validated(qs)}
        if name.startswith("scenario="):
            firsts = {q.previous_id: by_id[q.previous_id] for q in qs if q.previous_id in scores}
            entry["first_turns"] = _metrics([(q, scores[i]) for i, q in firsts.items()], by_id)
        if name.startswith("direction="):
            entry["translations_not_reviewed"] = {"not_reviewed": sum(not q.translation_reviewed for q in qs), "total": len(qs)}
        out_groups[name] = entry

    headline = [(q, scores[q.id]) for q in stats.groups(scored).get("headline", [])]
    breakdowns = {}
    for field in ("subject", "question_type", "difficulty"):
        cells = {}
        def label(q):
            return str(getattr(q, field) or "unanswerable (no difficulty)")
        for v in sorted({label(q) for q, _ in headline}):
            part = [r for r in headline if label(r[0]) == v]
            cells[v] = {"n": len(part), "metrics": _metrics(part, by_id) if len(part) >= MIN_CELL else TOO_SMALL}
        breakdowns[field] = cells

    has_contexts = any(s.context_count is not None for s in scores.values())
    measured = [m for ok, m in ((has_contexts, "retrieval metrics and faithfulness"),
                                (judge_model, "judged metrics (answer recall, contradiction, refusals, language match)")) if ok]
    not_measured = ["hard-negative rate (deferred in v1)"]
    if not has_contexts:
        not_measured.append("retrieval metrics and faithfulness (the endpoint returned no contexts)")
    if not judge_model:
        not_measured.append("judged metrics (no judge scores in this run)")
    skipped = [q for q in questions if q.is_follow_up and q.id not in scores]
    if skipped:
        not_measured.append(f"{len(skipped)} follow-up questions (" + (
            "not answered yet" if manifest["settings"]["supports_history"]
            else "not asked: the endpoint's history support is not confirmed") + ")")
    pending = [q for q in questions if not q.is_follow_up and q.id not in scores]
    if pending:
        not_measured.append(f"{len(pending)} questions not answered yet")
    return {
        "run_id": run_id, "dataset_id": manifest["dataset_id"], "dataset_sha256": manifest["dataset_sha256"],
        "label": manifest["label"], "started": manifest["started"],
        "measured": measured, "not_measured": not_measured,
        "groups": out_groups, "breakdowns": breakdowns,
        "closed_book": {"flagged": sum(q.closed_book for q, _ in headline),
                        "without": _metrics([r for r in headline if not r[0].closed_book], by_id)},
        "errors": [{"id": s.id, "error": s.error} for s in scores.values() if s.error],
        "judge_errors": [{"id": s.id, "error": s.judge_error} for s in scores.values() if s.judge_error],
        "judge": {"model": judge_model, "prompt_versions": judge_versions, "by_lang": judge},
        "matcher": labels.matcher_rates(),
        "settings": manifest["settings"],
        "constants": {"window": manifest["matcher"]["window"], "coverage": manifest["matcher"]["coverage"],
                      "seed": manifest["seed"], "top_k": manifest["top_k"], "adapter": manifest["adapter"],
                      "rag_url": adapters.public_url(manifest["rag_url"]), "min_cell": MIN_CELL},
    }


def _fmt(m: dict | None) -> str:
    return NOT_MEASURED if m is None else f"{m['mean']:.3f} [{m['low']:.3f}, {m['high']:.3f}] n={m['n']}"


def _table(metrics: dict, not_validated: list[str] = ()) -> list[str]:
    lines = ["| Metric | Errors count as failures | Errored questions left out |", "|---|---|---|"]
    for m, v in metrics.items():
        name = m + (" (lower is better)" if stats.METRICS[m]["lower_is_better"] else "") + (" †" if m in not_validated else "")
        lines.append(f"| {name} | {_fmt(v['all'])} | {_fmt(v['answered'])} |")
    return lines


def _render(s: dict) -> str:
    g = s["groups"]
    L = [f"# Report for run {s['run_id']}", "",
         f"Dataset {s['dataset_id']} (sha256 {s['dataset_sha256'][:12]}), label \"{s['label']}\", started {s['started']}.", "",
         "## What was measured", ""]
    L += [f"- Measured: {m}" for m in s["measured"]] + [f"- Not measured: {m}" for m in s["not_measured"]]
    L += ["", "## Headline scores", "", "Single questions asked in their document's own language. Mean [95% interval] and "
          "the number of questions behind it. † judge not validated for this metric (see Validation).", ""]
    L += _table(g["headline"]["metrics"], g["headline"]["judge_not_validated"]) if "headline" in g else ["No headline questions were scored."]
    L += ["", "## Validation", ""]
    j = s["judge"]
    if j["model"]:
        L.append(f"Judge model {j['model']}, prompt versions {j['prompt_versions']}. Agreement is on a balanced sample, "
                 "which is lower than agreement on a typical run.")
        for lang, a in j["by_lang"].items():
            L.append(f"- {lang}: {'validated' if a['validated'] else 'judge not validated'}"
                     + (f"; overall agreement {a['overall']:.3f}" if a["overall"] is not None else "; no labels yet")
                     + (f"; fewer than 30 labels: {', '.join(a['not_validated'])}" if a["not_validated"] else ""))
            for kind, t in a["types"].items():
                if t["n"]:
                    L.append(f"  - {kind}: n={t['n']}, agreement {t['agreement']:.3f} [{t['low']:.3f}, {t['high']:.3f}]"
                             + (f", kappa {t['kappa']:.3f}" if t["kappa"] is not None else ""))
    else:
        L.append("No judged scores in this run.")
    mt = s["matcher"]
    L.append(f"Evidence matcher (window {s['constants']['window']}, coverage {s['constants']['coverage']}): "
             + ("calibrated" if mt["calibrated"] else "matcher not calibrated")
             + f"; {mt['n_retrieved']} retrieved and {mt['n_not_retrieved']} not-retrieved labels"
             + (f"; miss rate {mt['miss_rate']:.3f}" if mt["miss_rate"] is not None else "")
             + (f"; false-accept rate {mt['false_accept_rate']:.3f}" if mt["false_accept_rate"] is not None else "") + ".")
    L += ["", "## Translation review", ""]
    dirs = {k: v for k, v in g.items() if k.startswith("direction=")}
    L += [f"- {k}: translations not reviewed by a person: {v['translations_not_reviewed']['not_reviewed']} of "
          f"{v['translations_not_reviewed']['total']}" for k, v in dirs.items()] or ["No translated questions."]
    L += ["", "## Context size", ""]
    L += [f"- {k}: {v['context']['mean_count']:.1f} contexts, {v['context']['mean_chars']:.0f} characters on average "
          f"(n={v['context']['n']})" if v["context"] else f"- {k}: no contexts returned" for k, v in g.items()]
    L += ["", "## Headline breakdowns", "", f"A cell with fewer than {MIN_CELL} questions is shown as \"{TOO_SMALL}\".", ""]
    for field, cells in s["breakdowns"].items():
        L += [f"### By {field}", ""]
        for v, cell in cells.items():
            if cell["metrics"] == TOO_SMALL:
                L.append(f"- {v}: {TOO_SMALL} (n={cell['n']})")
            else:
                L += [f"**{v}** (n={cell['n']})", ""] + _table(cell["metrics"]) + [""]
        L.append("")
    L += ["## Follow-ups by scenario", "", "Retrieval metrics on `same` follow-ups assume the target retrieves again; a target "
          "that answers correctly from the conversation gets no retrieval credit.", ""]
    scen = {k: v for k, v in g.items() if k.startswith("scenario=")}
    for k, v in scen.items():
        L += [f"### {k} (n={v['n']})", ""] + _table(v["metrics"], v["judge_not_validated"]) + ["", "First turns of these follow-ups:", ""]
        L += _table(v["first_turns"]) + [""]
    L += [] if scen else ["No follow-ups were scored.", ""]
    L += ["## Translated directions", ""]
    for k, v in dirs.items():
        t = v["translations_not_reviewed"]
        L += [f"### {k} (n={v['n']})", "", f"Translations not reviewed by a person: {t['not_reviewed']} of {t['total']}.", ""]
        L += _table(v["metrics"], v["judge_not_validated"]) + [""]
    L += [] if dirs else ["No translated questions.", ""]
    L += ["## Closed-book questions", "", f"{s['closed_book']['flagged']} headline questions are flagged closed-book (the "
          "generator model answered them without the documents). Headline scores without them:", ""]
    L += _table(s["closed_book"]["without"]) + [""]
    L += ["## Errors", "", f"Endpoint errors: {len(s['errors'])}. Judge errors: {len(s['judge_errors'])} (left out of judged metrics)."]
    L += [f"- {e['id']}: {e['error']}" for e in s["errors"] + s["judge_errors"]]
    c = s["constants"]
    L += ["", "## Settings and constants", "", f"- Adapter {c['adapter']}, endpoint {c['rag_url'] or 'n/a'}, top k {c['top_k']}",
          f"- Matcher window {c['window']}, coverage {c['coverage']}; bootstrap seed {c['seed']}",
          f"- Settings: {s['settings']}", "",
          "## What these scores mean", "",
          "These scores describe these documents and this question set, not other documents. Intervals ignore clustering "
          "within documents and judge noise, so they are somewhat too narrow."]
    return "\n".join(L) + "\n"


def write(run_id: str) -> Path:
    path = store.runs_dir() / run_id / "report.md"
    store.write_text_atomic(path, _render(summary(run_id)))
    return path


def compare(run_a: str, run_b: str, primary: str = "answer_recall", only: str | None = None) -> dict:
    """Paired comparison of two runs on one group; verdicts are "better", "worse" or "no detectable difference"."""
    ma, questions, sa = _load(run_a)
    mb, _, sb = _load(run_b)
    if ma["dataset_sha256"] != mb["dataset_sha256"]:
        raise SystemExit("These runs used different datasets (the hashes differ), so they cannot be compared. "
                         "Pick two runs on the same frozen dataset.")
    if primary not in stats.METRICS:
        raise SystemExit(f"Unknown primary metric \"{primary}\". Allowed values: {', '.join(stats.METRICS)}.")
    by_id = {q.id: q for q in questions}
    allowed = stats.groups([q for q in questions if q.id in sa and q.id in sb])
    group = only or "headline"
    if group not in allowed:
        raise SystemExit(f"No questions in group \"{group}\" were scored in both runs. Groups available: {', '.join(allowed) or 'none'}.")
    result = {"run_a": run_a, "run_b": run_b, "group": group, "primary": primary, "metrics": {}}
    for m, info in stats.METRICS.items():
        va = stats.values(m, [(q, sa[q.id]) for q in allowed[group]])
        vb = stats.values(m, [(q, sb[q.id]) for q in allowed[group]])
        res = stats.compare(va, vb, {q.id: stats.unit(q, by_id) for q in allowed[group]}, SEED)
        if res["verdict"] in ("higher", "lower"):
            res["verdict"] = "better" if (res["verdict"] == "higher") != info["lower_is_better"] else "worse"
        result["metrics"][m] = {**res, "role": "primary" if m == primary else "exploratory"}
    return result


def compare_markdown(result: dict) -> str:
    p = result["metrics"][result["primary"]]
    L = [f"# Comparison of {result['run_a']} (A) and {result['run_b']} (B)", "",
         f"Group: {result['group']}. Differences are B minus A, with a 95% paired bootstrap interval. Endpoint errors count as failures.", "",
         f"**Primary metric {result['primary']}: {p['verdict']}**" if p["diff"] is None
         else f"**Primary metric {result['primary']}: B is {p['verdict']} than A**" if p["verdict"] in ("better", "worse")
         else f"**Primary metric {result['primary']}: {p['verdict']}**"
         + (f" (smallest difference this comparison could have detected: {p['detectable']:.3f})" if p["detectable"] is not None
            else " (smallest detectable difference: not estimable)"),
         "", "| Metric | Role | n | Difference | 95% interval | Smallest detectable | Verdict |", "|---|---|---|---|---|---|---|"]
    for m, r in result["metrics"].items():
        if r["diff"] is None:
            L.append(f"| {m} | {r['role']} | 0 | {NOT_MEASURED} | | | |")
            continue
        det = "not estimable" if r["detectable"] is None else f"{r['detectable']:.3f}"
        L.append(f"| {m} | {r['role']} | {r['n']} | {r['diff']:+.3f} | [{r['low']:+.3f}, {r['high']:+.3f}] | {det} | {r['verdict']} |")
    L += ["", "Exploratory metrics are not corrected for multiple tests. \"No detectable difference\" does not mean "
          "no difference: read it with the smallest detectable difference."]
    return "\n".join(L) + "\n"
