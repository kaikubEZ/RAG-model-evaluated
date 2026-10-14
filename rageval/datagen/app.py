"""The data-generation app: Setup, Material, Generate, Review. Pick a page in the sidebar."""
import re

import pandas as pd
import streamlit as st

from rageval import jobs, llm, settings, store
from rageval.datagen import freeze, material, review, translate
from rageval.schema import QUESTION_TYPES, SCENARIOS, Question
from rageval.ui import (STOP_NOTE, attempt, env_form, flash, job_status, privacy, purge, run_app, ss, start_job,
                        test)

MODULE = "rageval.datagen.cli"
ENV_KEYS = [("GEN_AI_API_KEY", True), ("GEN_AI_URL", False), ("GEN_AI_MODEL", False),
            ("JUDGE_AI_API_KEY", True), ("JUDGE_AI_URL", False), ("JUDGE_AI_MODEL", False)]
COMMON_LANGS = ["en", "th", "zh", "ja", "ko", "es", "fr", "de", "vi", "id", "pt", "ru", "ar", "hi", "it", "ms", "tr"]
SCOPES = ["current", "published"]


def setup_page() -> None:
    env_form(ENV_KEYS, lambda s: {"scope": st.selectbox(
        "AI-TUTOR scope of the questions (current = not deployed, published = deployed)", SCOPES,
        index=SCOPES.index(s.scope))})
    st.subheader("Test connection")
    st.caption("Each test sends one tiny request.")
    for name, role in (("generator", "gen"), ("judge", "judge")):
        if st.button(f"Test {name}", key=f"test_{role}"):
            test(lambda r=role: llm.call(r, "Reply with the word ok."))


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
        start_job(MODULE, ["generate", "--yes"])
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
        start_job(MODULE, ["translate", "--yes"])
    if f.button("Freeze", disabled=readonly or not sent):
        start_job(MODULE, ["freeze", "--yes"])
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


run_app({"Setup": setup_page, "Material": material_page, "Generate": generate_page, "Review": review_page},
        "rageval datagen")
