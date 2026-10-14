"""The logic behind the Material page: subjects, files, uploads (architecture 3.14). No Streamlit here."""
from functools import lru_cache
from pathlib import Path

from . import ingest
from .. import settings, store, text

SYNCED = (("onedrive", "OneDrive"), ("dropbox", "Dropbox"), ("google drive", "Google Drive"),
          ("googledrive", "Google Drive"), ("icloud", "iCloud"), ("mobile documents", "iCloud"))


def _clean(name: str) -> str:
    """A file or folder name: no path parts, no characters Windows forbids. Thai and other scripts are kept."""
    base = name.replace(chr(92), "/").rsplit("/", 1)[-1]
    return "".join("_" if c in '<>:"|?*' or ord(c) < 32 else c for c in base).strip(" .")


@lru_cache(maxsize=256)
def _measure(path: str, mtime: float, size: int) -> tuple[int, int, int]:
    """(pages, eligible pages, estimated tokens); the cache key changes when the file does."""
    texts = ingest._extract(Path(path))
    return len(texts), sum(text.eligible(t) for t in texts), text.estimate_tokens("\n".join(texts))


def subjects() -> list[dict]:
    limit = settings.load().gen_max_input_tokens
    out = []
    for name in store.subjects():
        files = []
        for p in store.documents(name):
            st = p.stat()
            try:
                n, eligible, tokens = _measure(str(p), st.st_mtime, st.st_size)
                files.append({"name": p.name, "pages": n, "eligible": eligible, "tokens": tokens,
                              "fits": tokens <= limit, "error": None})
            except SystemExit as e:
                files.append({"name": p.name, "pages": 0, "eligible": 0, "tokens": 0, "fits": False, "error": str(e)})
        out.append({"name": name, "files": files})
    return out


def add_subject(name: str) -> None:
    name = _clean(name)
    if not name:
        raise ValueError("Give the subject a name made of letters or digits.")
    (store.material_dir() / name).mkdir(parents=True, exist_ok=True)


def _folder(subject: str) -> Path:
    if subject not in store.subjects():
        raise ValueError(f"There is no subject {subject}. Create it first.")
    return store.material_dir() / subject


def save_upload(subject: str, filename: str, data: bytes) -> Path:
    folder = _folder(subject)
    name = _clean(filename)
    if Path(name).suffix.lower() not in store.SUPPORTED:
        raise ValueError(f"{filename} cannot be used. Allowed file types: {', '.join(store.SUPPORTED)}.")
    path = folder / name
    path.write_bytes(data)
    return path


def delete_file(subject: str, filename: str) -> None:
    path = (_folder(subject) / filename).resolve()
    if path.parent != _folder(subject).resolve() or not path.is_file():
        raise ValueError(f"{filename} is not a file in {subject}.")
    path.unlink()


def synced_folder() -> str | None:
    for part in store.ROOT.parts:
        for prefix, label in SYNCED:
            if part.lower().startswith(prefix):
                return label
    return None
