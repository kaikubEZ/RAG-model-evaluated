"""Paths and plain-file IO. Everything the tool keeps is a file under the project root."""
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

ROOT = Path(os.environ.get("RAGEVAL_ROOT", ".")).resolve()
SUPPORTED = (".pdf", ".txt", ".md")
T = TypeVar("T", bound=BaseModel)


def material_dir() -> Path:
    return ROOT / "material"


def datasets_dir() -> Path:
    return ROOT / "datasets"


def runs_dir() -> Path:
    return ROOT / "runs"


def labels_dir() -> Path:
    return ROOT / "labels"


def safe_name(s: str) -> str:
    """File-safe on Windows: model names contain / and :."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)


def new_id(label: str = "") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{safe_name(label)}" if label else stamp


def subjects() -> list[str]:
    d = material_dir()
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def documents(subject: str) -> list[Path]:
    d = material_dir() / subject
    return sorted(p for p in d.iterdir() if p.suffix.lower() in SUPPORTED)


def list_ids(base: Path) -> list[str]:
    return sorted(p.name for p in base.iterdir() if p.is_dir()) if base.exists() else []


def is_frozen(dataset_id: str) -> bool:
    return (datasets_dir() / dataset_id / "questions.jsonl").exists()


def newest_unfrozen() -> str | None:
    ids = [i for i in list_ids(datasets_dir()) if not is_frozen(i)]
    return ids[-1] if ids else None


def write_text_atomic(path: Path, text: str) -> None:
    """Write to a temp file and move it into place; on Windows a file being read cannot be replaced, so retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    os.replace(tmp, path)


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:  # iterating splits on line feeds only, as the files are written
        lines = [ln for ln in f if ln.strip()]
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(model.model_validate_json(line))
        except ValueError:
            if i < len(lines) - 1:
                raise
            # a process killed mid-append leaves a partial last line; the row it was writing is simply redone
    return rows


def write_jsonl(path: Path, rows: list[BaseModel]) -> None:
    write_text_atomic(path, "".join(r.model_dump_json() + "\n" for r in rows))


def append_jsonl(path: Path, row: BaseModel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            # a process killed mid-append left a partial last line; cut it off so it never ends up mid-file
            with path.open("r+b") as f:
                f.truncate(data.rfind(b"\n") + 1)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(row.model_dump_json() + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: dict) -> None:
    write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
