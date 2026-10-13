"""The Streamlit app: the eight pages of PRD FR-11 (architecture 3.14). Pick a page in the sidebar."""
import os
import re
from pathlib import Path
from urllib.parse import urlparse

import dotenv
import pandas as pd
import streamlit as st

from rageval import (adapters, freeze, jobs, labels, llm, material, report, review, settings, stats, store,
                     translate)
from rageval import run as run_mod
from rageval.schema import QUESTION_TYPES, SCENARIOS, Question, Score

ss = st.session_state
ENV_KEYS = [("GEN_AI_API_KEY", True), ("GEN_AI_URL", False), ("GEN_AI_MODEL", False),
            ("JUDGE_AI_API_KEY", True), ("JUDGE_AI_URL", False), ("JUDGE_AI_MODEL", False),
            ("RAG_API_KEY", True), ("RAG_URL", False)]
COMMON_LANGS = ["en", "th", "zh", "ja", "ko", "es", "fr", "de", "vi", "id", "pt", "ru", "ar", "hi", "it", "ms", "tr"]
MATCHER_RULE = ('Label "retrieved" when you can see the whole quoted statement in these contexts taken together, '
                'ignoring line breaks and stray characters. Half of a quote is "not retrieved".')
STOP_NOTE = ("Closing this tab does not stop a job. Press Stop to stop it. After a stop or a crash, press Start again "
             "and it continues where it stopped.")


def attempt(fn, *args, **kwargs):
    """Call into the tool; show the message of a user-facing error instead of a traceback. Returns (ok, value)."""
    try:
        return True, fn(*args, **kwargs)
    except (SystemExit, ValueError) as e:
        st.error(str(e) or type(e).__name__)
        return False, None


def flash(kind: str, message: str) -> None:
    ss["flash"] = (kind, message)


def show_flash() -> None:
    if "flash" in ss:
        kind, message = ss.pop("flash")
        getattr(st, kind)(message)


def purge(prefix: str) -> None:
    for k in [k for k in ss if str(k).startswith(prefix)]:
        del ss[k]


def fmt(m: dict | None) -> str:
    return "not measured" if m is None else f"{m['mean']:.3f} [{m['low']:.3f}, {m['high']:.3f}] n={m['n']}"


def metric_df(metrics: dict, flagged=()) -> pd.DataFrame:
    return pd.DataFrame([{"metric": m + (" (judge not validated)" if m in flagged else ""),
                          "errors count as failures": fmt(v["all"]), "errored questions left out": fmt(v["answered"])}
                         for m, v in metrics.items()])


def newest_run() -> Path | None:
    ids = store.list_ids(store.runs_dir())
    return store.runs_dir() / ids[-1] if ids else None


@st.fragment(run_every=2)
def job_status(target: str) -> None:
    """Progress of the job in a directory; "@dataset" is the newest unfrozen dataset, "@run" the newest run."""
    if target == "@dataset":
        newest = store.newest_unfrozen()
        d = store.datasets_dir() / newest if newest else None
    elif target == "@run":
        d = newest_run()
    else:
        d = Path(target)
    if d is None:
        st.caption("Nothing has run yet.")
        return
    running = jobs.is_running(d)
    if ss.setdefault(f"was_running|{d}", running) != running:
        ss[f"was_running|{d}"] = running
        st.rerun()  # unlock or lock the page's buttons
    p = jobs.progress(d)
    if not p:
        st.caption(f"No job has run in {d.name} yet.")
        return
    label = f"{d.name}: {p['job']}, {p['step']} {p['done']}/{p['total']}"
    st.progress(min(p["done"] / p["total"], 1.0) if p["total"] else 0.0, text=f"{label} {p['message']}".strip())
    if p["message"].startswith("failed"):
        st.error(f"{p['message']}. Details are in {store.ROOT / 'rageval-job.log'}.")
    if running and st.button("Stop", key=f"stop|{d}"):
        jobs.request_stop(d)
        st.info("Stop requested. The job stops after the call in hand.")


def start_job(args: list[str]) -> None:
    ok, _ = attempt(jobs.start_detached, args)
    if ok:
        flash("success", f"Started rageval {' '.join(args)}. {STOP_NOTE}")
        st.rerun()


def privacy(key: str) -> bool:
    """Say where document text goes (PRD section 9); the tick must be set before a model job can start."""
    try:
        hosts = sorted({urlparse(llm.config(r)["url"]).netloc for r in ("gen", "judge")})
    except SystemExit as e:
        st.warning(f"{e} Nothing can be sent until the model settings are saved.")
        return False
    return st.checkbox(f"I understand that text from my documents will be sent to: {', '.join(hosts)}", key=key)


# ---------------------------------------------------------------- Setup

def setup_page() -> None:
    env_path = store.ROOT / ".env"
    current = dotenv.dotenv_values(env_path) if env_path.exists() else {}
    s = settings.load()
    st.caption("Keys are saved to .env in the project folder. A blank key box keeps the saved key.")
    with st.form("setup"):
        values = {}
        for key, secret in ENV_KEYS:
            if secret:
                values[key] = st.text_input(key, type="password", placeholder="saved" if current.get(key) else "not set")
            else:
                values[key] = st.text_input(key, value=current.get(key) or "")
        st.caption("Blank JUDGE_AI values fall back to the GEN_AI values.")
        adapter = st.selectbox("Endpoint adapter", list(adapters.ADAPTERS), index=list(adapters.ADAPTERS).index(s.rag_adapter)
                               if s.rag_adapter in adapters.ADAPTERS else 0)
        history = st.checkbox("My endpoint accepts conversation history (the history field)", value=s.supports_history)
        top_k = st.number_input("Top k for retrieval metrics", min_value=1, value=s.top_k, step=1)
        saved = st.form_submit_button("Save")
    if saved:
        def save():
            store.ROOT.mkdir(parents=True, exist_ok=True)
            for key, secret in ENV_KEYS:
                if values[key] or not secret:
                    dotenv.set_key(str(env_path), key, values[key])
                    os.environ[key] = values[key]
            settings.save(settings.Settings.model_validate(
                {**s.model_dump(), "rag_adapter": adapter, "supports_history": history, "top_k": int(top_k)}))
        if attempt(save)[0]:
            st.success("Saved.")
    st.subheader("Test connection")
    st.caption("Each test sends one tiny request.")
    for name, role in (("generator", "gen"), ("judge", "judge")):
        if st.button(f"Test {name}", key=f"test_{role}"):
            test(lambda r=role: llm.call(r, "Reply with the word ok."))
    if st.button("Test target", key="test_target"):
        test(lambda: adapters.get(settings.load().rag_adapter)("ping", []))


def test(call) -> None:
    try:
        call()
        st.success("Connected.")
    except (SystemExit, Exception) as e:  # any failure of a connection test is the answer being asked for
        st.error(str(e) or type(e).__name__)


# ---------------------------------------------------------------- Material

def material_page() -> None:
    if (host := material.synced_folder()):
        st.warning(f"This project folder is inside {host}. The {host} client uploads material/ to the cloud "
                   "regardless of .gitignore. Keep private documents in a folder that is not synced.")
    with st.form("new_subject", clear_on_submit=True):
        name = st.text_input("New subject")
        if st.form_submit_button("Create subject") and attempt(material.add_subject, name)[0]:
            st.rerun()
    ok, subjects = attempt(material.subjects)
    if not ok:
        return
    if not subjects:
        st.info("No subjects yet. Create one above (for example the name of a topic), then upload its documents.")
    for sub in subjects:
        with st.expander(sub["name"], expanded=True):
            if sub["files"]:
                st.dataframe(pd.DataFrame(sub["files"]).rename(columns={"eligible": "eligible pages", "tokens": "estimated tokens"}),
                             hide_index=True, width="stretch")
                for f in sub["files"]:
                    if f["error"]:
                        st.error(f["error"])
                    elif not f["fits"]:
                        st.warning(f"{f['name']} is too large for the generator. Split it, or raise the largest document size on the Generate page.")
            else:
                st.caption("No files yet.")
            files = st.file_uploader("Upload files (.pdf, .txt, .md)", accept_multiple_files=True, key=f"up|{sub['name']}")
            if files and st.button("Save uploaded files", key=f"save|{sub['name']}"):
                done = [f.name for f in files if attempt(material.save_upload, sub["name"], f.name, f.getvalue())[0]]
                if done:
                    flash("success", f"Saved {len(done)} file(s) to {sub['name']}.")
                    st.rerun()
            if sub["files"]:
                gone = st.selectbox("File to delete", [f["name"] for f in sub["files"]], key=f"del|{sub['name']}")
                sure = st.checkbox(f"Yes, delete {gone}", key=f"sure|{sub['name']}")
                if st.button("Delete file", key=f"delbtn|{sub['name']}", disabled=not sure) and attempt(material.delete_file, sub["name"], gone)[0]:
                    st.rerun()


# ---------------------------------------------------------------- Generate

def generate_page() -> None:
    s = settings.load()
    langs = sorted(set(COMMON_LANGS) | set(s.extra_langs))
    with st.form("generate_settings"):
        st.write("Questions per subject, per type (0 switches a type off)")
        per_type = {t: st.number_input(t, min_value=0, value=s.questions_per_subject.get(t, 0), step=1) for t in QUESTION_TYPES}
        st.write("Follow-ups per subject, per scenario")
        per_sc = {sc: st.number_input(sc, min_value=0, value=s.follow_ups_per_subject.get(sc, 0), step=1) for sc in SCENARIOS}
        over = st.number_input("Over-generation factor", min_value=1.0, max_value=5.0, value=float(s.overgenerate), step=0.1)
        guidance = st.text_area("Generation guidance (optional)", value=s.generation_guidance)
        picked = st.multiselect("Extra test languages", langs, default=s.extra_langs)
        other = st.text_input("Other language codes, separated by commas (ISO 639-1, for example sv, pl)")
        biggest = st.number_input("Largest document for the generator, in tokens", min_value=1000, value=s.gen_max_input_tokens, step=1000)
        saved = st.form_submit_button("Save settings")
    if saved:
        def save():
            extra = [c.strip().lower() for c in other.split(",") if c.strip()]
            bad = [c for c in extra if not re.fullmatch(r"[a-z]{2,3}", c)]
            if bad:
                raise ValueError(f"Not a language code: {', '.join(bad)}. Use two-letter codes such as sv.")
            settings.save(settings.Settings.model_validate({
                **s.model_dump(), "questions_per_subject": {t: int(n) for t, n in per_type.items()},
                "follow_ups_per_subject": {sc: int(n) for sc, n in per_sc.items()}, "overgenerate": over,
                "generation_guidance": guidance, "extra_langs": list(dict.fromkeys(picked + extra)),
                "gen_max_input_tokens": int(biggest)}))
        if attempt(save)[0]:
            st.rerun()
    s = settings.load()
    st.subheader("What starting will do")
    ok, subjects = attempt(material.subjects)
    if not ok:
        return
    if not subjects or not any(sub["files"] for sub in subjects):
        st.info("There is no material yet. Add a subject and files on the Material page first.")
    target = sum(s.questions_per_subject.values()) + sum(s.follow_ups_per_subject.values())
    cands = sum(s.candidates(n) for n in [*s.questions_per_subject.values(), *s.follow_ups_per_subject.values()])
    active = sum(1 for n in s.questions_per_subject.values() if n) + (1 if any(s.follow_ups_per_subject.values()) else 0)
    if subjects:
        st.dataframe(pd.DataFrame([{"subject": sub["name"], "documents": len(sub["files"]), "target questions": target,
                                    "candidates to generate": cands} for sub in subjects]), hide_index=True, width="stretch")
        docs = sum(len(sub["files"]) for sub in subjects)
        st.caption(f"Roughly {docs * active} generator calls and {len(subjects) * cands} judge calls for a first pass; "
                   "refills and translations add more.")
    for lang in s.extra_langs:
        st.warning(f"{lang}: every question is also translated to {lang}. Its judged scores carry a warning until 150 "
                   f"judge decisions are labelled by someone who reads {lang}.")
    d = store.newest_unfrozen()
    running = bool(d) and jobs.is_running(store.datasets_dir() / d)
    if d:
        st.caption(f"Starting continues the unfinished dataset {d}.")
    sent = privacy("privacy_generate")
    has_files = any(sub["files"] for sub in subjects)
    if running:
        st.info("Generation is running.")
    if st.button("Start generating", disabled=running or not sent or not has_files):
        start_job(["generate", "--yes"])
    st.caption(STOP_NOTE)
    job_status("@dataset")


# ---------------------------------------------------------------- Review

def review_page() -> None:
    ids = store.list_ids(store.datasets_dir())[::-1]
    if not ids:
        st.info("No dataset yet. Generate one on the Generate page.")
        return
    if "goto_dataset" in ss:
        ss["review_ds"] = ss.pop("goto_dataset")
    newest = store.newest_unfrozen()
    ds = st.selectbox("Dataset", ids, index=ids.index(newest) if newest in ids else 0, key="review_ds",
                      format_func=lambda i: i + (" (frozen, read-only)" if store.is_frozen(i) else ""))
    d = store.datasets_dir() / ds
    frozen, running = store.is_frozen(ds), jobs.is_running(d)
    readonly = frozen or running
    if running:
        st.info("A job is running on this dataset, so the page is read-only until it finishes.")
    job_status(str(d))
    ok, rows = attempt(review.load, ds)
    if not ok:
        return
    pmap = review.pages(ds)
    by_id = {r.id: r for r in rows}

    cols = st.columns(5)
    pick = [cols[i].selectbox(label, ["all", *sorted({str(getattr(r, field) or "") for r in rows} - {""})], key=f"flt|{ds}|{label}")
            for i, (label, field) in enumerate([("Subject", "subject"), ("Type", "question_type"), ("Scenario", "scenario"),
                                                 ("Language", "lang"), ("Status", "status")])]
    fields = ["subject", "question_type", "scenario", "lang", "status"]
    shown = [r for r in rows if all(p == "all" or str(getattr(r, f) or "") == p for p, f in zip(pick, fields))]
    event = st.dataframe(
        pd.DataFrame([{"id": r.id, "subject": r.subject, "type": r.question_type, "scenario": r.scenario or "",
                       "lang": r.lang, "status": r.status, "check failed": bool(r.note), "question": r.question[:120]}
                      for r in shown]),
        hide_index=True, on_select="rerun", selection_mode="single-row", key=f"table|{ds}", width="stretch")
    chosen = [shown[i].id for i in event.selection.rows if i < len(shown)]
    sel = chosen[0] if chosen else None
    if sel != ss.get("rv_last"):
        ss["rv_last"], ss["rv_new"] = sel, False
    if st.button("New question", disabled=readonly):
        ss["rv_new"] = True
    q0 = None if ss.get("rv_new") else by_id.get(sel)
    if q0 or ss.get("rv_new"):
        review_form(ds, q0, rows, by_id, pmap, readonly)
    else:
        if "rv_fk" in ss:  # widget values of a form that is no longer shown are gone, so its saved state must go too
            purge(ss.pop("rv_fk"))
        st.caption("Select a row to edit it, or press New question.")

    st.subheader("Actions")
    a, b = st.columns(2)
    if a.button("Approve selected", disabled=readonly or not q0):
        n = attempt(review.set_status, ds, [q0.id], "approved")[1]
        flash("success" if n else "warning", "Approved." if n else "Not approved: it has a failed check or is already approved.")
        st.rerun()
    if b.button("Reject selected", disabled=readonly or not q0):
        n = attempt(review.set_status, ds, [q0.id], "rejected")[1]
        flash("success", f"Rejected {n} row(s).")
        st.rerun()
    sure = st.checkbox(f"Yes, approve all {len(shown)} rows in the current filter that have no failed check", key="bulk_sure")
    if st.button("Bulk approve filtered", disabled=readonly or not sure):
        n = attempt(review.set_status, ds, [r.id for r in shown], "approved", True)[1]
        flash("success", f"Approved {n} row(s) in bulk. This count is recorded in the manifest.")
        st.rerun()

    st.subheader("Translate and freeze")
    s = settings.load()
    gaps = [] if frozen else attempt(translate.missing, ds)[1] or []
    st.caption(f"Extra test languages: {', '.join(s.extra_langs) or 'none'}. Translations missing: {len(gaps)}.")
    sent = privacy("privacy_review")
    t, f, dup = st.columns(3)
    if t.button("Translate", disabled=readonly or not sent or not s.extra_langs):
        start_job(["translate", "--yes"])
    if f.button("Freeze", disabled=readonly or not sent):
        start_job(["freeze", "--yes"])
    if dup.button("Duplicate as draft", disabled=not frozen):
        ok, new = attempt(freeze.duplicate_as_draft, ds)
        if ok:
            ss["goto_dataset"] = new
            flash("success", f"Created draft {new}. It is selected now.")
            st.rerun()
    st.caption(STOP_NOTE)

    st.subheader("Counts")
    ok, sm = attempt(review.summary, ds)
    if ok:
        st.write(f"{sm['total']} rows, {sm['approved']} approved source questions. Approved in bulk: {sm['bulk_approved']}. "
                 f"Scenario labels corrected by the reviewer: {sm['scenario_overrides']}.")
        for title, key in (("Subject", "by_subject"), ("Type", "by_type"), ("Scenario", "by_scenario"),
                           ("Language", "by_language"), ("Status", "by_status")):
            if sm[key]:
                st.write(f"{title}: " + ", ".join(f"{k} {v}" for k, v in sm[key].items()))
        short = {sub: need for sub, need in sm["shortfall"].items() if need}
        if short:
            st.warning("Approved so far against the targets, still needed: " + "; ".join(
                f"{sub}: " + ", ".join(f"{n} {k}" for k, n in need.items()) for sub, need in short.items()))
        elif rows:
            st.success("Every target is met by approved questions.")


def review_form(ds: str, q0: Question | None, rows: list[Question], by_id: dict, pmap: dict, readonly: bool) -> None:
    fk = f"rv|{ds}|{q0.id if q0 else 'new'}"
    if ss.get("rv_fk") not in (None, fk):
        purge(ss["rv_fk"])
    ss["rv_fk"] = fk
    st.subheader("Edit question" if q0 else "New question")
    if q0 is None and not rows and not pmap:
        st.info("This dataset has no pages, so a hand-written question cannot be checked yet.")
    if f"{fk}|init" not in ss:
        ss[f"{fk}|init"] = True
        ss[f"{fk}|question"] = q0.question if q0 else ""
        ss[f"{fk}|resolved"] = q0.resolved_question if q0 else ""
        ss[f"{fk}|kp"] = list(range(len(q0.gold_ans))) if q0 else [0]
        for i, p in enumerate(q0.gold_ans if q0 else [""]):
            ss[f"{fk}|kp{i}"] = p
        ss[f"{fk}|ev"] = list(range(len(q0.gold_evidence))) if q0 else [0]
        for i, e in enumerate(q0.gold_evidence if q0 else []):
            ss[f"{fk}|ev{i}|doc"], ss[f"{fk}|ev{i}|page"], ss[f"{fk}|ev{i}|quote"] = e.doc, e.page, e.quote
        ss[f"{fk}|next"] = 100
    if q0:
        if q0.previous_id and q0.previous_id in by_id:
            st.info(f"First-turn question: {by_id[q0.previous_id].question}")
        if q0.is_translated and q0.base_id in by_id:
            st.info(f"Source question ({by_id[q0.base_id].lang}): {by_id[q0.base_id].question}")
        if q0.closest_passage:
            st.caption(f"Closest passage the judge found: {q0.closest_passage}")
        if q0.note:
            st.error(f"A check failed on the last save: {q0.note}. The question stays a candidate until it passes.")
    question = st.text_area("Question", key=f"{fk}|question", disabled=readonly)
    resolved = st.text_area("Resolved question (the same question made self-contained)", key=f"{fk}|resolved", disabled=readonly)

    st.write("Gold key points")
    for i in list(ss[f"{fk}|kp"]):
        c1, c2 = st.columns([8, 1])
        c1.text_area(f"Key point {i}", key=f"{fk}|kp{i}", label_visibility="collapsed", disabled=readonly)
        if c2.button("Remove", key=f"{fk}|kprm{i}", disabled=readonly):
            ss[f"{fk}|kp"].remove(i)
            st.rerun()
    if st.button("Add key point", key=f"{fk}|kpadd", disabled=readonly):
        ss[f"{fk}|kp"].append(ss[f"{fk}|next"])
        ss[f"{fk}|kp{ss[f'{fk}|next']}"] = ""
        ss[f"{fk}|next"] += 1
        st.rerun()

    subject = q0.subject if q0 else st.selectbox(
        "Subject", sorted({r.subject for r in rows} | {d.split("/")[0] for d, _ in pmap} | set(store.subjects())), key=f"{fk}|subject")
    docs = sorted(d for d, _ in pmap if d.startswith(f"{subject}/"))
    docs = sorted(set(docs))
    st.write("Evidence")
    for i in list(ss[f"{fk}|ev"]):
        left, right = st.columns(2)
        with left:
            if docs:
                cur = ss.get(f"{fk}|ev{i}|doc")
                st.selectbox("Document", docs if cur in (None, *docs) else [cur, *docs], key=f"{fk}|ev{i}|doc", disabled=readonly)
            else:
                st.text_input("Document", key=f"{fk}|ev{i}|doc", disabled=readonly)
            st.number_input("Page", min_value=1, step=1, key=f"{fk}|ev{i}|page", disabled=readonly)
            st.text_area("Quote", key=f"{fk}|ev{i}|quote", disabled=readonly)
            if st.button("Remove evidence", key=f"{fk}|evrm{i}", disabled=readonly):
                ss[f"{fk}|ev"].remove(i)
                st.rerun()
        with right:
            page = pmap.get((ss.get(f"{fk}|ev{i}|doc"), int(ss.get(f"{fk}|ev{i}|page") or 0)))
            st.text_area("Text of the cited page", value=page if page is not None else "(no such page)", height=250,
                         disabled=True, key=f"{fk}|ev{i}|view|{hash(page)}")
    if st.button("Add evidence", key=f"{fk}|evadd", disabled=readonly):
        n = ss[f"{fk}|next"]
        ss[f"{fk}|ev"].append(n)
        ss[f"{fk}|ev{n}|doc"], ss[f"{fk}|ev{n}|page"], ss[f"{fk}|ev{n}|quote"] = (docs[0] if docs else ""), 1, ""
        ss[f"{fk}|next"] += 1
        st.rerun()

    c1, c2, c3 = st.columns(3)
    qtype = c1.selectbox("Type", QUESTION_TYPES, index=QUESTION_TYPES.index(q0.question_type) if q0 else 0,
                         key=f"{fk}|type", disabled=readonly)
    scenarios = ["(single question)", *SCENARIOS]
    scenario = c2.selectbox("Scenario", scenarios, index=scenarios.index(q0.scenario) if q0 and q0.scenario else 0,
                            key=f"{fk}|scenario", disabled=readonly)
    status = c3.selectbox("Status", ["candidate", "approved", "rejected"],
                          index=["candidate", "approved", "rejected"].index(q0.status) if q0 else 0,
                          key=f"{fk}|status", disabled=readonly)
    previous, lang = (q0.previous_id, q0.lang) if q0 else (None, "en")
    if q0 is None:
        mine = [r for r in rows if r.subject == subject]
        lang = st.text_input("Language of the question", value=mine[0].source_lang if mine else "en", key=f"{fk}|lang")
        firsts = ["(none)", *[r.id for r in mine if r.answerable and not r.is_follow_up]]
        previous = st.selectbox("First-turn question, for a follow-up", firsts, key=f"{fk}|prev")
        previous = None if previous == "(none)" else previous
    if st.button("Save", key=f"{fk}|save", disabled=readonly):
        points = [ss[f"{fk}|kp{i}"].strip() for i in ss[f"{fk}|kp"] if ss[f"{fk}|kp{i}"].strip()]
        evidence = [{"doc": ss[f"{fk}|ev{i}|doc"], "page": int(ss[f"{fk}|ev{i}|page"]), "quote": ss[f"{fk}|ev{i}|quote"].strip()}
                    for i in ss[f"{fk}|ev"] if ss[f"{fk}|ev{i}|quote"].strip()]
        answerable = qtype != "unanswerable"
        base = q0.model_dump() if q0 else {"id": "new", "base_id": "new", "subject": subject, "lang": lang, "source_lang": lang}
        ok, q = attempt(Question.model_validate, {
            **base, "question": question.strip(), "resolved_question": resolved.strip() or question.strip(),
            "previous_id": previous, "scenario": None if scenario == scenarios[0] else scenario, "question_type": qtype,
            "answerable": answerable, "gold_ans": points if answerable else [], "gold_evidence": evidence if answerable else [],
            "status": status})
        if ok:
            ok, reason = attempt(review.save_row, ds, q)
            if ok:
                purge(fk)
                ss["rv_new"] = False
                flash("warning" if reason else "success", f"Saved as a candidate: {reason}" if reason else "Saved.")
                st.rerun()
    if q0 and q0.is_translated and not q0.translation_reviewed and st.button("Mark translation reviewed", key=f"{fk}|rev", disabled=readonly):
        if attempt(review.mark_reviewed, ds, q0.id)[0]:
            st.rerun()


# ---------------------------------------------------------------- Run and score

def runs_table() -> list[dict]:
    out = []
    for rid in store.list_ids(store.runs_dir()):
        m = store.read_json(store.runs_dir() / rid / "manifest.json")
        if not m:
            continue
        qs = store.read_jsonl(store.datasets_dir() / m["dataset_id"] / "questions.jsonl", Question)
        want = sum(1 for q in qs if not q.is_follow_up or m["settings"]["supports_history"])
        got = run_mod.latest_answers(rid)
        out.append({"run": rid, "dataset": m["dataset_id"], "label": m["label"], "answered": f"{len(got)} of {want}",
                    "errors": sum(1 for a in got.values() if a.error),
                    "scored": bool(store.read_jsonl(store.runs_dir() / rid / "scores.jsonl", Score)),
                    "running": jobs.is_running(store.runs_dir() / rid)})
    return out


def run_page() -> None:
    frozen = [i for i in store.list_ids(store.datasets_dir())[::-1] if store.is_frozen(i)]
    if not frozen:
        st.info("No frozen dataset yet. Review and freeze one on the Review page, then come back to run it.")
        return
    s = settings.load()
    if not s.supports_history:
        st.info("Follow-up questions will be skipped and reported as not measured, because \"my endpoint accepts "
                "conversation history\" is off on the Setup page.")
    ds = st.selectbox("Frozen dataset", frozen)
    label = st.text_input("Label for this run (for example rerank-on)")
    latest = newest_run()
    busy = bool(latest) and jobs.is_running(latest)
    if st.button("Start run", disabled=busy):
        start_job(["run", "--dataset", ds, "--label", label])
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
    sent = privacy("privacy_score")
    a, b, c = st.columns(3)
    if a.button("Resume", disabled=running):
        start_job(["run", "--resume", rid])
    if b.button("Score", disabled=running or not sent):
        start_job(["score", "--run", rid, "--yes"])
    if c.button("Re-score, ignoring stored judge replies", disabled=running or not sent):
        start_job(["score", "--run", rid, "--yes", "--refresh-judge"])


# ---------------------------------------------------------------- Results

def scored_runs() -> list[str]:
    return [r["run"] for r in runs_table() if r["scored"]][::-1]


def results_page() -> None:
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
    questions = {q.id: q for q in store.read_jsonl(store.datasets_dir() / m["dataset_id"] / "questions.jsonl", Question)}
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
    ids = scored_runs()
    if len(ids) < 2:
        st.info("Comparing needs two scored runs on the same frozen dataset.")
        return
    a, b = st.columns(2)
    run_a, run_b = a.selectbox("Run A", ids, index=1), b.selectbox("Run B", ids, index=0)
    primary = st.selectbox("Primary metric", list(stats.METRICS), index=list(stats.METRICS).index("answer_recall"))
    m = store.read_json(store.runs_dir() / run_a / "manifest.json")
    qs = store.read_jsonl(store.datasets_dir() / m["dataset_id"] / "questions.jsonl", Question)
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
    judge_tab, matcher_tab = st.tabs(["Judge decisions", "Matcher pairs"])
    with judge_tab:
        judge_labelling()
    with matcher_tab:
        matcher_labelling()


def run_langs(run_ids: list[str]) -> list[str]:
    langs: set[str] = set()
    for rid in run_ids:
        m = store.read_json(store.runs_dir() / rid / "manifest.json")
        langs |= {q.lang for q in store.read_jsonl(store.datasets_dir() / m["dataset_id"] / "questions.jsonl", Question)}
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


PAGES = {"Setup": setup_page, "Material": material_page, "Generate": generate_page, "Review": review_page,
         "Run and score": run_page, "Results": results_page, "Compare": compare_page, "Labelling": labelling_page}

st.set_page_config(page_title="rageval", layout="wide")
page = st.sidebar.radio("Page", list(PAGES), key="page")
st.title(page)
show_flash()
try:
    PAGES[page]()
except (SystemExit, ValueError) as e:  # anything a page did not catch itself
    st.error(str(e) or type(e).__name__)
