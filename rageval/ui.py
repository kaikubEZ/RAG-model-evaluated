"""Streamlit helpers shared by the data-generation app and the evaluation app."""
import os
from pathlib import Path
from urllib.parse import urlparse

import dotenv
import streamlit as st

from rageval import jobs, llm, settings, store

ss = st.session_state
STOP_NOTE = "Closing this tab does not stop the job. Press Stop to stop it; starting again continues where it left off."
ROLE_NAMES = {"gen": "writing model", "judge": "checking model"}
KEY_HELP = {"GEN_AI_API_KEY": "Secret key of the model that writes the questions and answers.",
            "GEN_AI_URL": "Address of any OpenAI-compatible API, for example https://api.openai.com/v1.",
            "GEN_AI_MODEL": "Model name, for example gpt-4o-mini.",
            "JUDGE_AI_API_KEY": "Secret key of the model that checks questions and scores answers.",
            "JUDGE_AI_URL": "Address of the checking model's API.",
            "JUDGE_AI_MODEL": "Name of the checking model.",
            "RAG_API_KEY": "Key your RAG endpoint needs, if any.",
            "RAG_URL": "Address of the RAG endpoint you want to evaluate."}


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
        st.caption("No job has run yet.")
        return
    running = jobs.is_running(d)
    if ss.setdefault(f"was_running|{d}", running) != running:
        ss[f"was_running|{d}"] = running
        st.rerun()  # unlock or lock the page's buttons
    p = jobs.progress(d)
    if not p:
        st.caption(f"No job has run in {d.name} yet.")
        return
    label = f"{p['job']}: {p['step']} {p['done']}/{p['total']}"
    st.progress(min(p["done"] / p["total"], 1.0) if p["total"] else 0.0, text=f"{label} {p['message']}".strip())
    used = jobs.usage(d)
    if used:
        cols = st.columns(len(used) * 3)
        for i, (role, u) in enumerate(sorted(used.items())):
            who = ROLE_NAMES.get(role, role)
            cols[3 * i].metric(f"{who}: requests", f"{u['calls']:,}")
            cols[3 * i + 1].metric(f"{who}: tokens read", f"{u['prompt']:,}")
            cols[3 * i + 2].metric(f"{who}: tokens written", f"{u['completion']:,}")
    with st.expander("Technical log (last 200 lines)"):
        st.code(jobs.log_tail() or "(empty)", language=None)
    if p["message"].startswith("failed"):
        st.error(f"The job failed: {p['message']}. Check the settings on the Setup page, then press Start again. "
                 f"Technical details are in {store.ROOT / 'rageval-job.log'}.")
    if running:
        st.caption(STOP_NOTE)
        if st.button("Stop", key=f"stop|{d}"):
            jobs.request_stop(d)
            st.info("Stop requested. The job stops after the request it is working on.")

def start_job(module: str, args: list[str]) -> None:
    ok, _ = attempt(jobs.start_detached, module, args)
    if ok:
        flash("success", "Job started. Progress is shown on this page.")
        st.rerun()

def privacy(key: str, roles: tuple[str, ...] = ("gen", "judge")) -> bool:
    """Say where document text goes (PRD section 9); the tick must be set before a model job can start.
    Returns whether it is ticked; callers show `NEED_TICK` while it is not."""
    try:
        hosts = sorted({urlparse(llm.config(r)["url"]).netloc for r in roles})
    except SystemExit as e:
        st.warning(f"{e} Save the model settings on the Setup page first.")
        return False
    return st.checkbox(f"I agree that text from my documents is sent to: {', '.join(hosts)}", key=key)


# ---------------------------------------------------------------- Setup

def test(call) -> None:
    try:
        call()
        st.success("Connected. The settings work.")
    except (SystemExit, Exception) as e:  # any failure of a connection test is the answer being asked for
        st.error(str(e) or type(e).__name__)


def env_form(keys: list[tuple[str, bool]], extra=None) -> None:
    """Save API keys to .env; `extra(s)` adds widgets to the form and returns the settings changes to save."""
    env_path = store.ROOT / ".env"
    current = dotenv.dotenv_values(env_path) if env_path.exists() else {}
    s = settings.load()
    st.caption("Saved in the .env file in the project folder. Leave a key box empty to keep the saved key.")
    with st.form("setup"):
        values = {}
        for key, secret in keys:
            if secret:
                values[key] = st.text_input(key, type="password", placeholder="saved" if current.get(key) else "not set", help=KEY_HELP.get(key))
            else:
                values[key] = st.text_input(key, value=current.get(key) or "", help=KEY_HELP.get(key))
        if any(k.startswith("GEN_") for k, _ in keys):
            st.caption("Leave the JUDGE_AI boxes empty to use the same model as GEN_AI.")
        changes = extra(s) if extra else {}
        saved = st.form_submit_button("Save")
    if saved:
        def save():
            store.ROOT.mkdir(parents=True, exist_ok=True)
            for key, secret in keys:
                if values[key] or not secret:
                    dotenv.set_key(str(env_path), key, values[key])
                    os.environ[key] = values[key]
            settings.save(settings.Settings.model_validate({**s.model_dump(), **changes}))
        if attempt(save)[0]:
            st.success("Saved.")


NEED_TICK = "Tick the box above to continue."


def why_disabled(*reasons: tuple[bool, str]) -> None:
    """Say under a greyed-out button why it is greyed out (the first reason that applies)."""
    for applies, text in reasons:
        if applies:
            st.caption(text)
            return


def next_step(name: str) -> None:
    """A button that moves the sidebar to the page called `name`."""
    st.divider()
    if st.button(f"Next: {name} \u2192", key=f"next|{name}", type="primary"):
        ss["goto_page"] = name
        st.rerun()


def run_app(pages: dict, key: str = "page") -> None:
    """Draw the sidebar list of pages (one list per feature, kept in session state under `key`) and the chosen page."""
    if "goto_page" in ss:  # set by a Next button; the radio is not drawn yet, so its value can still be changed
        ss[key] = ss.pop("goto_page")
    page = st.sidebar.radio("Step", list(pages), key=key)
    st.title(page)
    show_flash()
    try:
        pages[page]()
    except (SystemExit, ValueError) as e:  # anything a page did not catch itself
        st.error(str(e) or type(e).__name__)
