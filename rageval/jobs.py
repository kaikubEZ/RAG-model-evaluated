"""Long-job control: lock with heartbeat, progress file, stop signal (architecture 3.14)."""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from . import store

HEARTBEAT_SECONDS = 10
STALE_SECONDS = 60


class Busy(Exception):
    """Another live job holds this directory."""


class Stopped(Exception):
    """The user asked the job to stop."""


def _lock_age(directory: Path) -> float | None:
    """Seconds since the lock's last heartbeat, or None if there is no readable lock."""
    try:
        return time.time() - json.loads((directory / "job.lock").read_text(encoding="utf-8"))["heartbeat"]
    except (OSError, ValueError, KeyError):
        return None


def is_running(directory: Path) -> bool:
    age = _lock_age(directory)
    return age is not None and age < STALE_SECONDS


def progress(directory: Path) -> dict:
    try:
        return json.loads((directory / "progress.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def request_stop(directory: Path) -> None:
    (directory / "job.stop").touch()


class Job:
    """Context manager for one job in a dataset or run directory.

    with Job(directory, "generate") as job:
        job.step("ingest", total=9)
        job.tick("file.pdf")     # raises Stopped if the user pressed Stop
    """

    def __init__(self, directory: Path, name: str):
        self.dir, self.name = directory, name
        self._done = self._total = 0
        self._step = ""
        self._alive = threading.Event()

    def __enter__(self) -> "Job":
        self.dir.mkdir(parents=True, exist_ok=True)
        if is_running(self.dir):
            raise Busy(f"A job is already running in {self.dir.name}. Stop it first, or wait for it to finish.")
        (self.dir / "job.stop").unlink(missing_ok=True)
        self._beat()
        self._alive.set()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        self._write("started")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._alive.clear()
        if exc_type is None:
            state = "finished"
        elif exc_type is Stopped:
            state = "stopped"
        else:
            state = f"failed: {exc}"
        self._write(state)
        (self.dir / "job.lock").unlink(missing_ok=True)
        (self.dir / "job.stop").unlink(missing_ok=True)
        return False

    def _beat(self) -> None:
        store.write_json(self.dir / "job.lock", {"pid": os.getpid(), "job": self.name, "heartbeat": time.time()})

    def _heartbeat(self) -> None:
        # background thread, so a long model call does not let the lock go stale
        while self._alive.is_set():
            time.sleep(HEARTBEAT_SECONDS)
            if self._alive.is_set():
                self._beat()

    def _write(self, message: str) -> None:
        store.write_json(self.dir / "progress.json", {
            "job": self.name, "step": self._step, "done": self._done, "total": self._total,
            "message": message, "time": time.time(),
        })

    def check_stop(self) -> None:
        if (self.dir / "job.stop").exists():
            raise Stopped()

    def step(self, name: str, total: int) -> None:
        self.check_stop()
        self._step, self._done, self._total = name, 0, total
        self._write("")
        print(f"[{self.name}] {name}: {total}", flush=True)

    def tick(self, message: str = "") -> None:
        self._done += 1
        self._write(message)
        self.check_stop()


def start_detached(module: str, args: list[str]) -> None:
    """Start `python -m <module> <args>` so that it outlives the app page that launched it."""
    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    log = (store.ROOT / "rageval-job.log").open("a", encoding="utf-8")
    subprocess.Popen([sys.executable, "-m", module, *args], cwd=store.ROOT, stdout=log, stderr=log,
                     stdin=subprocess.DEVNULL, **kwargs)
