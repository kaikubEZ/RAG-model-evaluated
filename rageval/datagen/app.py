"""The data-generation app, one numbered page per step: Setup, Material, Questions, Answers, Translate, Export."""

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


# ---------------------------------------------------------------- 1. Setup

def setup_page() -> None:
    step_intro(1, "Tell the program which AI model writes the questions. You only do this once.")
    env_form(ENV_KEYS, lambda s: {"scope": st.selectbox(
        "AI-TUTOR scope of the questions (current = not deployed, published = deployed)", SCOPES,
        index=SCOPES.index(s.scope))})
    st.subheader("Test connection")
    st.caption("Each test sends one tiny request.")
    for name, role in (("generator", "gen"), ("judge", "judge")):
        if st.button(f"Test {name}", key=f"test_{role}"):
            test(lambda r=role: llm.call(r, "Reply with the word ok."))


# ---------------------------------------------------------------- 2. Material

def material_page() -> None:
    step_intro(2, "Add the course documents the questions will be written from (PDF with a text layer, .txt or .md).")
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
                        st.warning(f"{f['name']} is too large for the generator. Split it, or raise the largest document size under Targets and options in step 3.")
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

# ---------------------------------------------------------------- shared pieces of the numbered steps

STEPS = ["1. Setup", "2. Material", "3. Questions", "4. Answers", "5. Translate", "6. Export"]


def step_intro(n: int, text: str) -> None:
    st.caption(f"Step {n} of {len(STEPS)}. {text}")


def next_step(name: str) -> None:
    st.divider()
    if st.button(f"Next: {name} →", key=f"next|{name}", type="primary"):
        ss["goto_page"] = name
        st.rerun()


def dataset_bar(allow_new: bool = False) -> str | None:
    """The dataset every step works on. The choice is kept while the user moves between steps."""
    ids = store.list_ids(store.datasets_dir())[::-1]
    options = ([NEW_DATASET] if allow_new else []) + ids
    if not options:
        st.info("There is no dataset yet. Create one in step 3.")
        return None
    if "goto_ds" in ss:
        ss["ds"] = ss.pop("goto_ds")
    if ss.get("ds") not in options:
        newest = store.newest_unfrozen()
        ss["ds"] = newest if newest in options else options[0]

    def describe(i: str) -> str:
        if i == NEW_DATASET:
            return "+ New dataset"
        n = len(review.load(i))
        return f"{i}  ({n} rows{', frozen' if store.is_frozen(i) else ''})"

    return st.selectbox("Dataset", options, key="ds", format_func=describe)


def stage_counts(rows: list[Question]) -> dict:
    """What is done at each step, for the progress strip."""
    src = [r for r in rows if not r.is_translated]
    approved = [r for r in src if r.status == "approved"]
    need = [r for r in approved if r.answerable]
    ready = [r for r in approved if r.answer_ready]
    langs = sorted({r.lang for r in rows if r.is_translated})
    return {"questions": (len(approved), len(src)), "answers": (sum(r.answer_status == "approved" for r in need), len(need)),
            "ready": len(ready), "translations": {lang: sum(1 for r in rows if r.lang == lang and r.is_translated) for lang in langs}}


def progress_strip(ds: str, rows: list[Question]) -> None:
    c = stage_counts(rows)
    cols = st.columns(4)
    cols[0].metric("3 Questions approved", f"{c['questions'][0]} / {c['questions'][1]}")
    cols[1].metric("4 Answers approved", f"{c['answers'][0]} / {c['answers'][1]}")
    cols[2].metric("5 Translated", ", ".join(f"{k} {v}" for k, v in c["translations"].items()) or "none")
    cols[3].metric("6 Export", "frozen" if store.is_frozen(ds) else "not yet")


def open_dataset(allow_new: bool = False):
    """The dataset bar plus what every step needs from it. None when there is nothing to show."""
    ds = dataset_bar(allow_new)
    if ds is None or ds == NEW_DATASET:
        return ds, None, None, True
    d = store.datasets_dir() / ds
    readonly = store.is_frozen(ds) or jobs.is_running(d)
    if store.is_frozen(ds):
        st.info("This dataset is frozen, so it is read-only. Use Duplicate as draft on step 6 to change it.")
    elif jobs.is_running(d):
        st.info("A job is running on this dataset, so the page is read-only until it finishes.")
    ok, rows = attempt(review.load, ds)
    if not ok:
        return None, None, None, True
    progress_strip(ds, rows)
    return ds, d, rows, readonly


def show_table(ds: str, shown: list[Question], key: str) -> str | None:
    """The rows as the columns of dataset.csv (in the contract's order), plus review state. Returns the selected id."""
    table = pd.DataFrame(review.contract_rows(ds, shown), columns=csvio.COLUMNS)
    table["review status"] = [r.status for r in shown]
    table["answer status"] = [r.answer_status if r.answerable else "n/a" for r in shown]
    table["check failed"] = [r.note or "" for r in shown]
    # text cannot wrap in a table, so each long column is as wide as its longest cell and the table scrolls sideways
    wide = {c: st.column_config.TextColumn(width=int(min(max(table[c].str.len().max() * 7 + 24, 120), 3000)))
            for c in ("question", "reference_answer", "required_facts_json", "evidence_json", "provenance_json",
                      "history_json", "check failed") if len(table)}
    wide["case_id"] = st.column_config.TextColumn(pinned=True)
    event = st.dataframe(table, hide_index=True, on_select="rerun", selection_mode="single-row", key=f"table|{ds}|{key}",
                         width="stretch", column_config=wide)
    chosen = [shown[i].id for i in event.selection.rows if i < len(shown)]
    sel = chosen[0] if chosen else None
    if sel != ss.get("rv_last"):
        ss["rv_last"], ss["rv_new"] = sel, False
    return sel


def filters(ds: str, rows: list[Question], specs: list[tuple[str, str]]) -> list[Question]:
    cols = st.columns(len(specs))
    picks = [cols[i].selectbox(label, ["all", *sorted({str(getattr(r, field) or "") for r in rows} - {""})],
                               key=f"flt|{ds}|{label}") for i, (label, field) in enumerate(specs)]
    return [r for r in rows if all(p == "all" or str(getattr(r, f) or "") == p for p, (_, f) in zip(picks, specs))]


def edit_selected(ds: str, sel: str | None, rows: list[Question], readonly: bool, allow_new: bool = False):
    """The edit form for the selected row (or a new question). Returns the selected row."""
    by_id = {r.id: r for r in rows}
    if allow_new and st.button("Write a new question by hand", disabled=readonly):
        ss["rv_new"] = True
    q0 = None if ss.get("rv_new") else by_id.get(sel)
    if q0 or ss.get("rv_new"):
        with st.expander("Edit the selected row", expanded=True):
            review_form(ds, q0, rows, by_id, review.pages(ds), readonly)
    else:
        if "rv_fk" in ss:  # widget values of a form that is no longer shown are gone, so its saved state must go too
            purge(ss.pop("rv_fk"))
        st.caption("Click a row in the table to read it in full and edit it.")
    return q0


# ---------------------------------------------------------------- 3. Questions

def questions_page() -> None:
    step_intro(3, "Write English questions from your material, then approve the good ones. Questions are checked "
                  "against the document text, so every quote is real.")
    ds, d, rows, readonly = open_dataset(allow_new=True)
    if ds is None:
        return
    s = settings.load()
    with st.expander("How many questions, and other options"):
        with st.form("generate_settings"):
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
    ok, subjects = attempt(material.subjects)
    if not ok:
        return
    has_files = any(sub["files"] for sub in subjects)
    if not has_files:
        st.info("There is no material yet. Add a subject and files in step 2 first.")
    cands = sum(s.candidates(n) for n in [*s.questions_per_subject.values(), *s.follow_ups_per_subject.values()])
    st.caption(f"Starting writes about {cands} candidate questions for each of {len(subjects)} subject(s), up to your targets. "
               "Starting again on the same dataset only fills what is missing; raise the targets to get more.")
    name = st.text_input("Name for the new dataset (optional)", key="gen_name") if ds == NEW_DATASET else ""
    sent = privacy("privacy_generate")
    if st.button("Start writing English questions", disabled=readonly and ds != NEW_DATASET or not sent or not has_files, type="primary"):
        target = ds if d else store.new_id(name.strip())
        (store.datasets_dir() / target).mkdir(parents=True, exist_ok=True)
        ss["goto_ds"] = target  # the dataset just started stays selected on every step
        start_job(MODULE, ["generate", "--yes", "--dataset", target])
    st.caption(STOP_NOTE)
    if not d:
        return
    job_status(str(d))

    st.subheader("Review the questions")
    shown = filters(ds, rows, [("Subject", "subject"), ("Type", "question_type"), ("Scenario", "scenario"), ("Status", "status")])
    shown = [r for r in shown if not r.is_translated]
    sel = show_table(ds, shown, "questions")
    q0 = edit_selected(ds, sel, rows, readonly, allow_new=True)
    a, b = st.columns(2)
    if a.button("Approve selected question", disabled=readonly or not q0):
        n = attempt(review.set_status, ds, [q0.id], "approved")[1]
        flash("success" if n else "warning", "Approved." if n else "Not approved: it has a failed check or is already approved.")
        st.rerun()
    if b.button("Reject selected question", disabled=readonly or not q0):
        n = attempt(review.set_status, ds, [q0.id], "rejected")[1]
        flash("success", f"Rejected {n} row(s).")
        st.rerun()
    sure = st.checkbox(f"Yes, approve all {len(shown)} questions in the current filter that have no failed check", key="bulk_sure")
    if st.button("Approve everything in the filter", disabled=readonly or not sure):
        n = attempt(review.set_status, ds, [r.id for r in shown], "approved", True)[1]
        flash("success", f"Approved {n} question(s) in bulk. This count is recorded in the manifest.")
        st.rerun()
    ok, sm = attempt(review.summary, ds)
    if ok:
        with st.expander("Counts and what is still short of your targets"):
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
    next_step(STEPS[3])


# ---------------------------------------------------------------- 4. Answers

def answers_page() -> None:
    step_intro(4, "For each approved question, write the English reference answer and its required facts (short key phrases), "
                  "then approve them. Unanswerable questions need no answer.")
    ds, d, rows, readonly = open_dataset()
    if not ds or rows is None:
        return
    job_status(str(d))
    approved = [r for r in rows if r.status == "approved" and not r.is_translated]
    need = [r for r in approved if r.answerable]
    if not approved:
        st.info("No question is approved yet. Approve some in step 3 first.")
        return
    waiting = answer.todo(rows)
    st.write(f"{len(need)} approved questions need an answer: {sum(r.answer_status == 'approved' for r in need)} approved, "
             f"{sum(r.answer_status == 'candidate' for r in need)} written and waiting for you, {len(waiting)} not written yet. "
             f"{len(approved) - len(need)} unanswerable question(s) need none.")
    sent = privacy("privacy_answer")
    if st.button("Write the missing answers", disabled=readonly or not sent or not waiting, type="primary"):
        start_job(MODULE, ["answer", "--yes"])
    st.caption(STOP_NOTE)

    shown = filters(ds, need, [("Subject", "subject"), ("Type", "question_type"), ("Answer", "answer_status")])
    sel = show_table(ds, shown, "answers")
    q0 = edit_selected(ds, sel, rows, readonly)
    can = bool(q0) and q0.answerable and not q0.is_translated and q0.answer_status != "none"
    if st.button("Approve the answer of the selected row", disabled=readonly or not can):
        n = attempt(review.set_answer_status, ds, [q0.id], "approved")[1]
        flash("success" if n else "warning", "Answer approved." if n else "Not approved: it has a failed check or is already approved.")
        st.rerun()
    ready = [r.id for r in shown if r.answer_status == "candidate" and not r.note]
    sure_a = st.checkbox(f"Yes, approve the {len(ready)} written answers in the filter that have no failed check", key="bulk_ans_sure")
    if st.button("Approve all written answers in the filter", disabled=readonly or not sure_a or not ready):
        n = attempt(review.set_answer_status, ds, ready, "approved")[1]
        flash("success", f"Approved {n} answer(s).")
        st.rerun()
    drafted = [r.id for r in shown if r.answer_status == "candidate"]
    with st.expander("Not happy with the written answers? Write them again"):
        again = st.checkbox(f"Yes, discard the {len(drafted)} written (not approved) answers in the filter", key="redo_sure")
        if st.button("Discard written answers", disabled=readonly or not again or not drafted):
            n = attempt(review.clear_answers, ds, drafted)[1]
            flash("success", f"Discarded {n} written answer(s). Press Write the missing answers to make them again.")
            st.rerun()
    next_step(STEPS[4])


# ---------------------------------------------------------------- 5. Translate

def translate_page() -> None:
    step_intro(5, "Translate the questions and answers into other languages. This step is optional: "
                  "skip to step 6 if you only need English.")
    ds, d, rows, readonly = open_dataset()
    if not ds or rows is None:
        return
    job_status(str(d))
    s = settings.load()
    ready = stage_counts(rows)["ready"]
    if not ready:
        st.info("No question has an approved answer yet, so there is nothing to translate. Finish step 4 first.")
        return
    if "tr_langs" not in ss:
        ss["tr_langs"] = ", ".join(s.extra_langs)
    typed = st.text_input("Translate into (language codes, separated by commas, for example th, zh)", key="tr_langs",
                          disabled=readonly, help="Any code you type is passed to the model as the target language. "
                          "English is the baseline and is left out. Use zh for Chinese.")
    gaps = [] if store.is_frozen(ds) else attempt(translate.missing, ds)[1] or []
    st.caption(f"{ready} questions have approved answers and will be translated. Translations still missing for the saved "
               f"languages ({', '.join(s.extra_langs) or 'none'}): {len(gaps)}. Judged scores in a language carry a warning until "
               "150 judge decisions are labelled by someone who reads it.")
    sent = privacy("privacy_translate")
    if st.button("Translate", disabled=readonly or not sent or not typed.strip(), type="primary"):
        def save_langs():
            settings.save(settings.Settings.model_validate({**s.model_dump(), "extra_langs": translate.parse_langs(typed)}))
        if attempt(save_langs)[0]:
            start_job(MODULE, ["translate", "--yes"])
    st.caption(STOP_NOTE)
    present = sorted({r.lang for r in rows if r.is_translated})
    if present and not readonly:
        with st.expander("Discard translations to make them again"):
            gone = st.multiselect("Languages to discard", present, default=present, key="discard_langs")
            n = sum(1 for r in rows if r.is_translated and r.lang in gone)
            sure_d = st.checkbox(f"Yes, delete {n} translated rows, including any I marked as reviewed", key="discard_sure")
            if st.button("Discard translations", disabled=not sure_d or not gone):
                got = attempt(review.discard_translations, ds, gone)[1]
                flash("success", f"Discarded {got} translated row(s). Press Translate to make them again.")
                st.rerun()
    translated = [r for r in rows if r.is_translated]
    if translated:
        st.subheader("Read the translations")
        shown = filters(ds, translated, [("Language", "lang"), ("Subject", "subject"), ("Type", "question_type")])
        sel = show_table(ds, shown, "translations")
        edit_selected(ds, sel, rows, readonly)
    next_step(STEPS[5])


# ---------------------------------------------------------------- 6. Export

def export_page() -> None:
    step_intro(6, "Freeze the dataset and download dataset.csv. A frozen dataset cannot change, so evaluations stay comparable.")
    ds, d, rows, readonly = open_dataset()
    if not ds or rows is None:
        return
    job_status(str(d))
    s = settings.load()
    c = stage_counts(rows)
    frozen = store.is_frozen(ds)
    gaps = [] if frozen else attempt(translate.missing, ds)[1] or []
    left_out = sum(1 for r in rows if r.status == "approved" and not r.is_translated and not r.answer_ready)
    a, b, c2 = st.columns(3)
    a.metric("Questions that will be exported", c["ready"])
    b.metric("Left out: approved but no approved answer", left_out)
    c2.metric("Translations still missing", len(gaps))
    if left_out:
        st.warning(f"{left_out} approved question(s) have no approved answer and will not be exported. Finish them in step 4.")
    if gaps:
        st.warning("Some translations are missing, so freezing is refused. Run step 5 again, or remove the language from the "
                   "box there.")
    if not frozen:
        sent = privacy("privacy_export") if s.judge_checks else True
        if st.button("Freeze and write dataset.csv", disabled=readonly or not sent or not c["ready"] or bool(gaps), type="primary"):
            start_job(MODULE, ["freeze", "--yes"])
        st.caption(STOP_NOTE)
    else:
        csv_path = d / "dataset.csv"
        st.success(f"Frozen. The file is {csv_path}.")
        if csv_path.exists():
            st.download_button("Download dataset.csv", csv_path.read_bytes(), file_name=f"{ds}.csv", mime="text/csv", type="primary")
        manifest = store.read_json(d / "manifest.json")
        if manifest:
            with st.expander("Manifest"):
                st.json(manifest)
        if st.button("Duplicate as draft to change it"):
            ok, new = attempt(freeze.duplicate_as_draft, ds)
            if ok:
                ss["goto_ds"] = new
                flash("success", f"Created draft {new}. It is selected now.")
                st.rerun()
    ok, sm = attempt(review.summary, ds)
    if ok and rows:
        with st.expander("Counts"):
            for title, key in (("Subject", "by_subject"), ("Type", "by_type"), ("Scenario", "by_scenario"),
                               ("Language", "by_language"), ("Status", "by_status")):
                if sm[key]:
                    st.write(f"{title}: " + ", ".join(f"{k} {v}" for k, v in sm[key].items()))


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


run_app(dict(zip(STEPS, [setup_page, material_page, questions_page, answers_page, translate_page, export_page])),
        "rageval datagen")
