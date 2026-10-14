"""Streamlit helpers shared by the data-generation app and the evaluation app."""
import os
from pathlib import Path
from urllib.parse import urlparse

import dotenv
import streamlit as st

from rageval import jobs, llm, settings, store

ss = st.session_state
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

def start_job(module: str, args: list[str]) -> None:
    ok, _ = attempt(jobs.start_detached, module, args)
    if ok:
        flash("success", f"Started {module} {' '.join(args)}. {STOP_NOTE}")
        st.rerun()

def privacy(key: str, roles: tuple[str, ...] = ("gen", "judge")) -> bool:
    """Say where document text goes (PRD section 9); the tick must be set before a model job can start."""
    try:
        hosts = sorted({urlparse(llm.config(r)["url"]).netloc for r in roles})
    except SystemExit as e:
        st.warning(f"{e} Nothing can be sent until the model settings are saved.")
        return False
    return st.checkbox(f"I understand that text from my documents will be sent to: {', '.join(hosts)}", key=key)


# ---------------------------------------------------------------- Setup

def test(call) -> None:
    try:
        call()
        st.success("Connected.")
    except (SystemExit, Exception) as e:  # any failure of a connection test is the answer being asked for
        st.error(str(e) or type(e).__name__)


def env_form(keys: list[tuple[str, bool]], extra=None) -> None:
    """Save API keys to .env; `extra(s)` adds widgets to the form and returns the settings changes to save."""
    env_path = store.ROOT / ".env"
    current = dotenv.dotenv_values(env_path) if env_path.exists() else {}
    s = settings.load()
    st.caption("Keys are saved to .env in the project folder. A blank key box keeps the saved key.")
    with st.form("setup"):
        values = {}
        for key, secret in keys:
            if secret:
                values[key] = st.text_input(key, type="password", placeholder="saved" if current.get(key) else "not set")
            else:
                values[key] = st.text_input(key, value=current.get(key) or "")
        st.caption("Blank JUDGE_AI values fall back to the GEN_AI values.")
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


def run_app(pages: dict, title: str) -> None:
    st.set_page_config(page_title=title, layout="wide")
    page = st.sidebar.radio("Page", list(pages), key="page")
    st.title(page)
    show_flash()
    try:
        pages[page]()
    except (SystemExit, ValueError) as e:  # anything a page did not catch itself
        st.error(str(e) or type(e).__name__)
