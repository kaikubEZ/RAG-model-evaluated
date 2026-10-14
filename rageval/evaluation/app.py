"""Feature 2, auto evaluation: Setup, Run and score, Results, Compare, Labelling, as numbered steps."""
import pandas as pd
import streamlit as st

from rageval import jobs, llm, settings, store
from rageval.evaluation import adapters, dataset, labels, report, stats
from rageval.evaluation import run as run_mod
from rageval.schema import Score
from rageval.ui import (STOP_NOTE, attempt, env_form, job_status, newest_run, privacy, ss, start_job,
                        test)

MODULE = "rageval.evaluation.cli"
ENV_KEYS = [("JUDGE_AI_API_KEY", True), ("JUDGE_AI_URL", False), ("JUDGE_AI_MODEL", False),
            ("RAG_API_KEY", True), ("RAG_URL", False)]
MATCHER_RULE = ('Label "retrieved" when you can see the whole quoted statement in these contexts taken together, '
                'ignoring line breaks and stray characters. Half of a quote is "not retrieved".')


def _endpoint(s) -> dict:
    adapter = st.selectbox("Endpoint adapter", list(adapters.ADAPTERS), index=list(adapters.ADAPTERS).index(s.rag_adapter)
                           if s.rag_adapter in adapters.ADAPTERS else 0)
    history = st.checkbox("My endpoint accepts conversation history (the history field)", value=s.supports_history)
    top_k = st.number_input("Top k for retrieval metrics", min_value=1, value=s.top_k, step=1)
    return {"rag_adapter": adapter, "supports_history": history, "top_k": int(top_k)}


def setup_page() -> None:
    st.caption("Step 1 of 5. Tell the program which model judges the answers and where your RAG endpoint is. You only do this once.")
    env_form(ENV_KEYS, _endpoint)
    st.subheader("Test connection")
    st.caption("Each test sends one tiny request.")
    if st.button("Test judge", key="test_judge"):
        test(lambda: llm.call("judge", "Reply with the word ok."))
    if st.button("Test target", key="test_target"):
        test(lambda: adapters.get(settings.load().rag_adapter)("ping", []))


def fmt(m: dict | None) -> str:
    return "not measured" if m is None else f"{m['mean']:.3f} [{m['low']:.3f}, {m['high']:.3f}] n={m['n']}"


def metric_df(metrics: dict, flagged=()) -> pd.DataFrame:
    return pd.DataFrame([{"metric": m + (" (judge not validated)" if m in flagged else ""),
                          "errors count as failures": fmt(v["all"]), "errored questions left out": fmt(v["answered"])}
                         for m, v in metrics.items()])


# ---------------------------------------------------------------- Run and score

def runs_table() -> list[dict]:
    out = []
    for rid in store.list_ids(store.runs_dir()):
        m = store.read_json(store.runs_dir() / rid / "manifest.json")
        if not m:
            continue
        qs = dataset.load(m["dataset_id"])
        want = sum(1 for q in qs if not q.is_follow_up or m["settings"]["supports_history"])
        got = run_mod.latest_answers(rid)
        out.append({"run": rid, "dataset": m["dataset_id"], "label": m["label"], "answered": f"{len(got)} of {want}",
                    "errors": sum(1 for a in got.values() if a.error),
                    "scored": bool(store.read_jsonl(store.runs_dir() / rid / "scores.jsonl", Score)),
                    "running": jobs.is_running(store.runs_dir() / rid)})
    return out


def run_page() -> None:
    st.caption("Step 2 of 5. Pick a dataset.csv, ask every question to your RAG endpoint, then score the answers.")
    frozen = dataset.ids()[::-1]
    if not frozen:
        st.info("No dataset.csv yet. Freeze one with rageval-datagen, or copy a datasets/<id>/dataset.csv here.")
        return
    s = settings.load()
    if not s.supports_history:
        st.info("Follow-up questions will be skipped and reported as not measured, because \"my endpoint accepts "
                "conversation history\" is off on the Setup page.")
    ds = st.selectbox("Dataset", frozen)
    label = st.text_input("Label for this run (for example rerank-on)")
    latest = newest_run()
    busy = bool(latest) and jobs.is_running(latest)
    if st.button("Start run", disabled=busy):
        start_job(MODULE, ["run", "--dataset", ds, "--label", label])
    st.caption(STOP_NOTE)
    job_status("@run")

    ok, table = attempt(runs_table)
    if not ok:
        return
    if not table:
        st.info("No runs yet.")
        return
    st.subheader("Runs")
    st.dataframe(pd.DataFrame(table[::-1]), hide_index=True, width="stretch")
    rid = st.selectbox("Run", [r["run"] for r in table][::-1])
    rd = store.runs_dir() / rid
    if latest is None or rd != latest:
        job_status(str(rd))
    running = jobs.is_running(rd)
    sent = privacy("privacy_score", ("judge",))
    a, b, c = st.columns(3)
    if a.button("Resume", disabled=running):
        start_job(MODULE, ["run", "--resume", rid])
    if b.button("Score", disabled=running or not sent):
        start_job(MODULE, ["score", "--run", rid, "--yes"])
    if c.button("Re-score, ignoring stored judge replies", disabled=running or not sent):
        start_job(MODULE, ["score", "--run", rid, "--yes", "--refresh-judge"])


# ---------------------------------------------------------------- Results

def scored_runs() -> list[str]:
    return [r["run"] for r in runs_table() if r["scored"]][::-1]


def results_page() -> None:
    st.caption("Step 3 of 5. Read the scores of a finished run.")
    ids = scored_runs()
    if not ids:
        st.info("No scored run yet. Start and score a run on the Run and score page.")
        return
    rid = st.selectbox("Run", ids)
    ok, sm = attempt(report.summary, rid)
    if not ok:
        return
    st.write(f"Dataset {sm['dataset_id']}, label \"{sm['label']}\", started {sm['started']}.")
    st.subheader("What was measured")
    for m in sm["measured"]:
        st.write(f"Measured: {m}")
    for m in sm["not_measured"]:
        st.warning(f"Not measured: {m}")
    st.subheader("Headline scores")
    head = sm["groups"].get("headline")
    if head:
        st.caption("Mean [95% interval] and n, with endpoint errors counted as failures and with errored questions left out.")
        st.dataframe(metric_df(head["metrics"], head["judge_not_validated"]), hide_index=True, width="stretch")
    else:
        st.info("No headline questions were scored.")
    st.subheader("Validation")
    j = sm["judge"]
    if j["model"]:
        for lang, a in j["by_lang"].items():
            (st.success if a["validated"] else st.warning)(
                f"{lang}: judge {'validated' if a['validated'] else 'not validated'}"
                + (f", overall agreement {a['overall']:.3f}" if a["overall"] is not None else ", no labels yet")
                + (f". Fewer than 30 labels: {', '.join(a['not_validated'])}." if a["not_validated"] else "."))
    else:
        st.info("This run has no judged scores.")
    mt = sm["matcher"]
    (st.success if mt["calibrated"] else st.warning)(
        f"Evidence matcher {'calibrated' if mt['calibrated'] else 'not calibrated'}: {mt['n_retrieved']} retrieved and "
        f"{mt['n_not_retrieved']} not-retrieved labels.")
    st.subheader("Groups")
    for name, g in sm["groups"].items():
        with st.expander(f"{name} (n={g['n']})"):
            st.dataframe(metric_df(g["metrics"], g["judge_not_validated"]), hide_index=True, width="stretch")
            if g["context"]:
                st.caption(f"Contexts returned: {g['context']['mean_count']:.1f} on average, "
                           f"{g['context']['mean_chars']:.0f} characters (n={g['context']['n']}).")
            if "first_turns" in g:
                st.write("First turns of these follow-ups")
                st.dataframe(metric_df(g["first_turns"]), hide_index=True, width="stretch")
            if "translations_not_reviewed" in g:
                t = g["translations_not_reviewed"]
                st.caption(f"Translations not reviewed by a person: {t['not_reviewed']} of {t['total']}.")
    st.subheader("Headline breakdowns")
    for field, cells in sm["breakdowns"].items():
        for value, cell in cells.items():
            with st.expander(f"{field} = {value} (n={cell['n']})"):
                if isinstance(cell["metrics"], str):
                    st.write(f"{cell['metrics']}: fewer than {sm['constants']['min_cell']} questions.")
                else:
                    st.dataframe(metric_df(cell["metrics"]), hide_index=True, width="stretch")
    st.subheader("Errors")
    errors = sm["errors"] + sm["judge_errors"]
    st.write(f"Endpoint errors: {len(sm['errors'])}. Judge errors: {len(sm['judge_errors'])}.")
    if errors:
        st.dataframe(pd.DataFrame(errors), hide_index=True, width="stretch")
    with st.expander("Settings and constants"):
        st.json({"settings": sm["settings"], "constants": sm["constants"]})
    path = store.runs_dir() / rid / "report.md"
    if path.exists():
        st.download_button("Download report.md", path.read_text(encoding="utf-8"), file_name="report.md", mime="text/markdown")
    else:
        st.caption("report.md has not been written for this run. Score it again to write it.")
    question_table(rid)


def question_table(rid: str) -> None:
    rd = store.runs_dir() / rid
    m = store.read_json(rd / "manifest.json")
    questions = {q.id: q for q in dataset.load(m["dataset_id"])}
    answers = run_mod.latest_answers(rid)
    scores = {s.id: s for s in store.read_jsonl(rd / "scores.jsonl", Score)}
    st.subheader("Questions")
    st.dataframe(pd.DataFrame([{
        "id": i, "question": questions[i].resolved_question if i in questions else "", "answer": (answers[i].answer or "") if i in answers else "",
        **{k: getattr(s, k) for k in ("recall_at_k", "reciprocal_rank", "answer_recall", "contradiction", "faithfulness", "refused",
                                      "error", "judge_error")}} for i, s in scores.items()]), hide_index=True, width="stretch")
    if not scores:
        return
    one = st.selectbox("Look at one question", list(scores), format_func=lambda i: f"{i}: {questions[i].resolved_question[:80] if i in questions else ''}")
    s, a, q = scores[one], answers.get(one), questions.get(one)
    if q:
        st.write(f"Question: {q.question}")
        st.write("Key points: " + "; ".join(q.gold_ans))
    if a:
        st.write(f"Answer: {a.answer or a.error}")
    if s.key_points:
        st.write("Key-point verdicts")
        st.dataframe(pd.DataFrame([k.model_dump() for k in s.key_points]), hide_index=True, width="stretch")
    if s.claims:
        st.write("Claims in the answer")
        st.dataframe(pd.DataFrame([c.model_dump() for c in s.claims]), hide_index=True, width="stretch")
    if a and a.contexts:
        for n, c in enumerate(a.contexts, 1):
            with st.expander(f"Returned context {n}"):
                st.text(c.text)
    elif a and a.contexts is None:
        st.caption("The endpoint returned no contexts.")


# ---------------------------------------------------------------- Compare

def compare_page() -> None:
    st.caption("Step 4 of 5. Compare two scored runs on the same dataset to see whether a change helped.")
    ids = scored_runs()
    if len(ids) < 2:
        st.info("Comparing needs two scored runs on the same frozen dataset.")
        return
    a, b = st.columns(2)
    run_a, run_b = a.selectbox("Run A", ids, index=1), b.selectbox("Run B", ids, index=0)
    primary = st.selectbox("Primary metric", list(stats.METRICS), index=list(stats.METRICS).index("answer_recall"))
    m = store.read_json(store.runs_dir() / run_a / "manifest.json")
    qs = dataset.load(m["dataset_id"])
    group = st.selectbox("Group", list(dict.fromkeys(["headline", *stats.groups(qs)])))
    ok, result = attempt(report.compare, run_a, run_b, primary, group)
    if not ok:
        return
    st.markdown(report.compare_markdown(result))
    st.dataframe(pd.DataFrame([{"metric": k, "role": r["role"], "n": r["n"], "difference": r["diff"], "low": r["low"],
                                "high": r["high"], "smallest detectable": r["detectable"], "verdict": r["verdict"]}
                               for k, r in result["metrics"].items()]), hide_index=True, width="stretch")


# ---------------------------------------------------------------- Labelling

def labelling_page() -> None:
    st.caption("Step 5 of 5. Optional. Check the judge against your own labels so you know how far to trust its scores.")
    judge_tab, matcher_tab = st.tabs(["Judge decisions", "Matcher pairs"])
    with judge_tab:
        judge_labelling()
    with matcher_tab:
        matcher_labelling()


def run_langs(run_ids: list[str]) -> list[str]:
    langs: set[str] = set()
    for rid in run_ids:
        m = store.read_json(store.runs_dir() / rid / "manifest.json")
        langs |= {q.lang for q in dataset.load(m["dataset_id"])}
    return sorted(langs)


def judge_labelling() -> None:
    ids = scored_runs()
    if not ids:
        st.info("No scored run yet. Label judge decisions after a run has been scored.")
        return
    try:
        llm.model_name("judge")
    except SystemExit as e:
        st.warning(f"{e} Labels are stored per judge model, so the judge must be set up first.")
        return
    runs = st.multiselect("Runs to draw decisions from", ids, default=ids[:1], key="jl_runs")
    langs = run_langs(runs)
    if not langs:
        return
    lang = st.selectbox("Language", langs, key="jl_lang")
    if st.button("Load a sample to label", key="jl_load"):
        ok, items = attempt(labels.sample_judge, runs, lang)
        if ok:
            ss["jl_items"], ss["jl_i"] = items, 0
    items, i = ss.get("jl_items", []), ss.get("jl_i", 0)
    if i < len(items):
        item = items[i]
        st.write(f"Decision {i + 1} of {len(items)} ({item['type'].replace('_', ' ')})")
        st.write(f"Question: {item['question']}")
        st.text_area("Answer being judged", item["answer"], disabled=True, key=f"jl_ans{i}")
        if "text" in item:
            st.write(("Key point: " if item["type"] == "key_point" else "Claim: ") + item["text"])
        if item["type"] == "key_point":
            st.caption("Does the answer support, contradict, or say nothing about this key point?")
        elif item["type"] == "claim":
            st.caption("Do the contexts below support this claim?")
            for n, c in enumerate(item["contexts"], 1):
                with st.expander(f"Context {n}"):
                    st.text(c)
        else:
            st.caption("Is this answer a refusal (it says it cannot answer and gives no answer)?")
        cols = st.columns(len(labels.TYPES[item["type"]][1]))
        for col, value in zip(cols, labels.TYPES[item["type"]][1]):
            if col.button(value.replace("_", " "), key=f"jl_{i}_{value}"):
                if attempt(labels.save_judge_label, item, value)[0]:
                    ss["jl_i"] = i + 1
                    st.rerun()
    elif items:
        st.success("Sample finished. Load another sample to continue.")
    st.subheader("Agreement")
    ok, ag = attempt(labels.judge_agreement, lang)
    if ok:
        st.dataframe(pd.DataFrame([{"decision type": k, "labels": t["n"], "agreement": t["agreement"], "low": t["low"], "high": t["high"],
                                    "kappa": t["kappa"], "validated": t["validated"]} for k, t in ag["types"].items()]),
                     hide_index=True, width="stretch")
        st.write(f"Overall agreement: {ag['overall'] if ag['overall'] is not None else 'no labels yet'}. "
                 f"Judge {'validated' if ag['validated'] else 'not validated'} for {lang}.")
        st.caption("Agreement is measured on a balanced sample, so it is lower than on a typical run.")


def matcher_labelling() -> None:
    ids = scored_runs()
    if not ids:
        st.info("No scored run yet.")
        return
    rid = st.selectbox("Run", ids, key="ml_run")
    st.info(MATCHER_RULE)
    if st.button("Load a sample to label", key="ml_load"):
        ok, items = attempt(labels.sample_matcher, rid)
        if ok:
            ss["ml_items"], ss["ml_i"] = items, 0
    items, i = ss.get("ml_items", []), ss.get("ml_i", 0)
    if i < len(items):
        item = items[i]
        st.write(f"Pair {i + 1} of {len(items)}")
        st.text_area("Quote", item["quote"], disabled=True, key=f"ml_q{i}")
        for n, c in enumerate(item["contexts"], 1):
            with st.expander(f"Context {n}", expanded=True):
                st.text(c)
        a, b = st.columns(2)
        for col, name, value in ((a, "retrieved", True), (b, "not retrieved", False)):
            if col.button(name, key=f"ml_{i}_{value}"):
                if attempt(labels.save_matcher_label, item, value)[0]:
                    ss["ml_i"] = i + 1
                    st.rerun()
    elif items:
        st.success("Sample finished. Load another sample to continue.")
    st.subheader("Matcher rates")
    st.json(labels.matcher_rates())


STEPS = ["1. Setup", "2. Run and score", "3. Results", "4. Compare", "5. Labelling"]
PAGES = dict(zip(STEPS, [setup_page, run_page, results_page, compare_page, labelling_page]))
