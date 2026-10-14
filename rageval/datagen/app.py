"""The data-generation app: Setup, Material, Generate, Review. Pick a page in the sidebar."""

import pandas as pd
import streamlit as st

from rageval import csvio, jobs, llm, settings, store
from rageval.datagen import answer, freeze, material, review, translate
from rageval.schema import QUESTION_TYPES, SCENARIOS, Question
from rageval.ui import (STOP_NOTE, attempt, env_form, flash, job_status, privacy, purge, run_app, ss, start_job,
                        test)

MODULE = "rageval.datagen.cli"
ENV_KEYS = [("GEN_AI_API_KEY", True), ("GEN_AI_URL", False), ("GEN_AI_MODEL", False),
            ("JUDGE_AI_API_KEY", True), ("JUDGE_AI_URL", False), ("JUDGE_AI_MODEL", False)]
SCOPES = ["current", "published"]
NEW_DATASET = "New dataset"


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
    with st.form("generate_settings"):
        st.caption("Step 1 writes English questions with their evidence. Step 2 (answers) and step 3 (translation) "
                   "start from the Review page once you have approved questions.")
        st.write("Questions per subject, per type (0 switches a type off)")
        per_type = {t: st.number_input(t, min_value=0, value=s.questions_per_subject.get(t, 0), step=1) for t in QUESTION_TYPES}
        st.write("Follow-ups per subject, per scenario")
        per_sc = {sc: st.number_input(sc, min_value=0, value=s.follow_ups_per_subject.get(sc, 0), step=1) for sc in SCENARIOS}
        over = st.number_input("Over-generation factor", min_value=1.0, max_value=5.0, value=float(s.overgenerate), step=0.1)
        guidance = st.text_area("Generation guidance (optional)", value=s.generation_guidance)
        judge = st.checkbox("Use AI judge checks (on questions, answers and translations; more accurate, much more expensive)",
                            value=s.judge_checks)
        biggest = st.number_input("Largest document for the generator, in tokens", min_value=1000, value=s.gen_max_input_tokens, step=1000)
        saved = st.form_submit_button("Save settings")
    if saved:
        def save():
            settings.save(settings.Settings.model_validate({
                **s.model_dump(), "questions_per_subject": {t: int(n) for t, n in per_type.items()},
                "follow_ups_per_subject": {sc: int(n) for sc, n in per_sc.items()}, "overgenerate": over,
                "generation_guidance": guidance, "judge_checks": judge, "gen_max_input_tokens": int(biggest)}))
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
    drafts = [i for i in store.list_ids(store.datasets_dir())[::-1] if not store.is_frozen(i)]
    if "goto_gen" in ss:
        ss["gen_ds"] = ss.pop("goto_gen")
    if ss.get("gen_ds") not in (NEW_DATASET, *drafts):
        ss["gen_ds"] = drafts[0] if drafts else NEW_DATASET

    def describe(i: str) -> str:
        if i == NEW_DATASET:
            return NEW_DATASET
        n = len(store.read_jsonl(store.datasets_dir() / i / "candidates.jsonl", Question))
        return f"{i}  ({n} questions so far)"

    choice = st.selectbox("Dataset to generate into", [NEW_DATASET, *drafts], key="gen_ds", format_func=describe,
                          help="Pick an unfinished dataset to continue it (up to the targets above), or start a new one.")
    name = st.text_input("Name for the new dataset (optional)", key="gen_name") if choice == NEW_DATASET else ""
    d = None if choice == NEW_DATASET else store.datasets_dir() / choice
    running = bool(d) and jobs.is_running(d)
    sent = privacy("privacy_generate")
    has_files = any(sub["files"] for sub in subjects)
    if running:
        st.info("Generation is running.")
    if st.button("Start generating English questions", disabled=running or not sent or not has_files):
        ds = choice if d else store.new_id(name.strip())
        (store.datasets_dir() / ds).mkdir(parents=True, exist_ok=True)
        ss["goto_gen"] = ss["goto_dataset"] = ds  # the page and the Review page follow the dataset just started
        start_job(MODULE, ["generate", "--yes", "--dataset", ds])
    st.caption(STOP_NOTE)
    if d:
        job_status(str(d))


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
    # the columns of the dataset.csv contract, in the contract's order, then the review-only state
    table = pd.DataFrame(review.contract_rows(ds, shown), columns=csvio.COLUMNS)
    table["review status"] = [r.status for r in shown]
    table["answer status"] = [r.answer_status if r.answerable else "n/a" for r in shown]
    table["check failed"] = [r.note or "" for r in shown]
    # text cannot wrap in a table, so each long column is as wide as its longest cell and the table scrolls sideways
    wide = {c: st.column_config.TextColumn(width=int(min(max(table[c].str.len().max() * 7 + 24, 120), 3000)))
            for c in ("question", "reference_answer", "required_facts_json", "evidence_json", "provenance_json",
                      "history_json", "check failed") if len(table)}
    wide["case_id"] = st.column_config.TextColumn(pinned=True)
    event = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row", key=f"table|{ds}",
                         width="stretch", column_config=wide)
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

    st.subheader("Step 2: English answers")
    todo = [r for r in rows if answer.todo([r])] if not frozen else []
    st.caption(f"Approved questions waiting for an answer: {len(todo)}. Drafted answers are checked against the quotes; "
               "open a row to read, edit and approve its answer.")
    sent = privacy("privacy_answer")
    x, y, z = st.columns(3)
    if x.button("Generate answers", disabled=readonly or not sent or not todo):
        start_job(MODULE, ["answer", "--yes"])
    drafted = [r.id for r in shown if r.answerable and not r.is_translated and r.answer_status == "candidate"]
    again = x.checkbox(f"Yes, discard the {len(drafted)} drafted (not approved) answers in the filter", key="redo_sure")
    if x.button("Discard drafted answers to redo them", disabled=readonly or not again or not drafted):
        n = attempt(review.clear_answers, ds, drafted)[1]
        flash("success", f"Discarded {n} drafted answer(s). Press Generate answers to write them again.")
        st.rerun()
    can = bool(q0) and q0.answerable and not q0.is_translated and q0.answer_status != "none"
    if y.button("Approve answer of selected", disabled=readonly or not can):
        n = attempt(review.set_answer_status, ds, [q0.id], "approved")[1]
        flash("success" if n else "warning", "Answer approved." if n else "Not approved: it has a failed check or is already approved.")
        st.rerun()
    ready = [r.id for r in shown if r.answerable and not r.is_translated and r.answer_status == "candidate" and not r.note]
    sure_a = z.checkbox(f"Yes, approve {len(ready)} answers in the filter that have no failed check", key="bulk_ans_sure")
    if z.button("Bulk approve answers", disabled=readonly or not sure_a or not ready):
        n = attempt(review.set_answer_status, ds, ready, "approved")[1]
        flash("success", f"Approved {n} answer(s).")
        st.rerun()

    st.subheader("Step 3: translate, then freeze")
    s = settings.load()
    gaps = [] if frozen else attempt(translate.missing, ds)[1] or []
    if "tr_langs" not in ss:
        ss["tr_langs"] = ", ".join(s.extra_langs)
    typed = st.text_input("Translate into (language codes, separated by commas, for example th, zh)", key="tr_langs",
                          disabled=readonly, help="Any code you type is passed to the model as the target language. "
                          "English is the baseline and is left out.")
    st.caption(f"Languages saved for this project: {', '.join(s.extra_langs) or 'none'}. Translations missing: {len(gaps)}. "
               "Only questions with approved answers are translated. Judged scores in a language carry a warning until "
               "150 judge decisions are labelled by someone who reads it.")
    sent = privacy("privacy_review")
    t, f, dup = st.columns(3)
    if t.button("Translate", disabled=readonly or not sent or not typed.strip()):
        def save_langs():
            settings.save(settings.Settings.model_validate({**s.model_dump(), "extra_langs": translate.parse_langs(typed)}))
        if attempt(save_langs)[0]:
            start_job(MODULE, ["translate", "--yes"])
    present = sorted({r.lang for r in rows if r.is_translated})
    if present and not frozen:
        with st.expander("Discard translations to make them again"):
            gone = st.multiselect("Languages to discard", present, default=present, key="discard_langs")
            n = sum(1 for r in rows if r.is_translated and r.lang in gone)
            sure_d = st.checkbox(f"Yes, delete {n} translated rows, including any I marked as reviewed", key="discard_sure")
            if st.button("Discard translations", disabled=readonly or not sure_d or not gone):
                got = attempt(review.discard_translations, ds, gone)[1]
                flash("success", f"Discarded {got} translated row(s). Press Translate to make them again.")
                st.rerun()
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
        ss[f"{fk}|refans"] = (q0.reference_answer or "") if q0 else ""
        ss[f"{fk}|kp"] = list(range(len(q0.gold_ans))) if q0 else [0]
        for i, p in enumerate(q0.gold_ans if q0 else [""]):
            ss[f"{fk}|kp{i}"] = p
        ss[f"{fk}|ev"] = list(range(len(q0.gold_evidence))) if q0 else [0]
        for i, e in enumerate(q0.gold_evidence if q0 else []):
            ss[f"{fk}|ev{i}|doc"], ss[f"{fk}|ev{i}|page"], ss[f"{fk}|ev{i}|quote"] = e.doc, e.page, e.quote
            ss[f"{fk}|ev{i}|facts"] = list(e.facts)
        ss[f"{fk}|next"] = 100
    if q0:
        if q0.previous_id and q0.previous_id in by_id:
            st.info(f"First-turn question: {by_id[q0.previous_id].question}")
        source = by_id.get(f"{q0.base_id}-{q0.source_lang}")
        if q0.is_translated and source:
            st.info(f"Source question ({source.lang}): {source.question}")
        if q0.closest_passage:
            st.caption(f"Closest passage the judge found: {q0.closest_passage}")
        if q0.note:
            st.error(f"A check failed on the last save: {q0.note}. The question stays a candidate until it passes.")
    question = st.text_area("Question", key=f"{fk}|question", disabled=readonly)
    resolved = st.text_area("Resolved question (the same question made self-contained)", key=f"{fk}|resolved", disabled=readonly)

    st.text_area("Reference answer (English; drafted by step 2)", key=f"{fk}|refans", disabled=readonly)
    if q0 and q0.answerable:
        st.caption(f"Answer status: {q0.answer_status}")
    st.write("Required facts (short key phrases a correct answer must contain)")
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
            options = [f"f{k}" for k in range(1, len(ss[f"{fk}|kp"]) + 1)]
            ss[f"{fk}|ev{i}|facts"] = [f for f in ss.get(f"{fk}|ev{i}|facts", []) if f in options]
            st.multiselect("Supports required facts", options, key=f"{fk}|ev{i}|facts", disabled=readonly)
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
        evidence = [{"doc": ss[f"{fk}|ev{i}|doc"], "page": int(ss[f"{fk}|ev{i}|page"]), "quote": ss[f"{fk}|ev{i}|quote"].strip(),
                     "facts": [f for f in ss.get(f"{fk}|ev{i}|facts", []) if int(f[1:]) <= len(points)]}
                    for i in ss[f"{fk}|ev"] if ss[f"{fk}|ev{i}|quote"].strip()]
        answerable = qtype != "unanswerable"
        base = q0.model_dump() if q0 else {"id": "new", "base_id": "new", "subject": subject, "lang": lang, "source_lang": lang}
        ok, q = attempt(Question.model_validate, {
            **base, "question": question.strip(), "resolved_question": resolved.strip() or question.strip(),
            "previous_id": previous, "scenario": None if scenario == scenarios[0] else scenario, "question_type": qtype,
            "answerable": answerable, "gold_ans": points if answerable else [], "gold_evidence": evidence if answerable else [],
            "reference_answer": ss[f"{fk}|refans"].strip() or None if answerable else None,
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
