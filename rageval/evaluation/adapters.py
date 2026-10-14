"""Target endpoint adapters (architecture 3.7). An adapter is one function: ask(question, history) -> dict."""
import importlib.util
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

from .. import store

TIMEOUT_SECONDS = 60
MAX_CHARS = 20_000  # per answer and per context
MAX_CONTEXTS = 50
MAX_BODY = 8_000_000  # bytes read from one reply
REFERENCE = Path(__file__).resolve().parent.parent / "tests" / "reference_rag.py"


class Permanent(Exception):
    """Configuration problem (401, 403, 404): retrying cannot help."""


class Transient(Exception):
    """Rate limit, server error or timeout: worth another attempt."""


class BadReply(Exception):
    """The endpoint answered, but not in the agreed shape."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def rag_url() -> str:
    load_dotenv(store.ROOT / ".env")
    return os.environ.get("RAG_URL", "")


def public_url(url: str | None = None) -> str:
    """The endpoint address without credentials or query: safe to write to a manifest or a report."""
    parts = urllib.parse.urlsplit(rag_url() if url is None else url)
    if not parts.hostname:
        return ""
    host = f"[{parts.hostname}]" if ":" in parts.hostname else parts.hostname
    return f"{parts.scheme}://{host}{f':{parts.port}' if parts.port else ''}{parts.path}"


def http_json(question: str, history: list[dict]) -> dict:
    """POST the contract in PRD FR-6 to RAG_URL."""
    url = rag_url()
    if not url:
        raise SystemExit("RAG_URL is not set. Add it to .env (see .env.example) or use the app's Setup page.")
    headers = {"Content-Type": "application/json"}
    if os.environ.get("RAG_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['RAG_API_KEY']}"
    body = json.dumps({"question": question, "history": history}).encode("utf-8")
    try:
        with _opener.open(urllib.request.Request(url, body, headers, method="POST"), timeout=TIMEOUT_SECONDS) as r:
            raw = r.read(MAX_BODY + 1)
    except urllib.error.HTTPError as e:
        if 300 <= e.code < 400:  # following it would re-send the API key to wherever it points
            raise BadReply(f"the endpoint redirected (HTTP {e.code}); put the final address in RAG_URL") from e
        if e.code in (401, 403, 404):
            raise Permanent(f"HTTP {e.code}") from e
        if e.code == 429 or e.code >= 500:
            raise Transient(f"HTTP {e.code}") from e
        raise BadReply(f"HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        raise Transient(f"{type(e).__name__}: {e}") from e
    if len(raw) > MAX_BODY:
        raise BadReply(f"the reply is larger than the limit of {MAX_BODY} bytes")
    try:
        data = json.loads(raw)
    except ValueError as e:
        raise BadReply("the reply is not JSON") from e
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        raise BadReply('the reply has no "answer" text')
    if len(data["answer"]) > MAX_CHARS:
        raise BadReply(f"the answer is longer than the limit of {MAX_CHARS} characters")
    contexts = data.get("contexts")
    if contexts is not None:
        if not (isinstance(contexts, list) and all(isinstance(c, dict) and isinstance(c.get("text"), str)
                                                    for c in contexts)):
            raise BadReply('"contexts" must be a list of {"text": ...}')
        if len(contexts) > MAX_CONTEXTS:
            raise BadReply(f"the reply has more than the limit of {MAX_CONTEXTS} contexts")
        if any(len(c["text"]) > MAX_CHARS for c in contexts):
            raise BadReply(f"a context is longer than the limit of {MAX_CHARS} characters")
        contexts = [{"text": c["text"]} for c in contexts]
    return {"answer": data["answer"], "contexts": contexts}


def reference(question: str, history: list[dict]) -> dict:
    """The reference target in tests/reference_rag.py, loaded by path (it is test code, not a package)."""
    module = sys.modules.get("reference_rag")
    if module is None:
        if not REFERENCE.exists():
            raise SystemExit(f"The reference target {REFERENCE} is not there; this adapter needs the tests folder.")
        spec = importlib.util.spec_from_file_location("reference_rag", REFERENCE)
        module = importlib.util.module_from_spec(spec)
        sys.modules["reference_rag"] = module
        spec.loader.exec_module(module)
    return module.ask(question, history)


ADAPTERS = {"http_json": http_json, "reference": reference}


def get(name: str) -> Callable[[str, list[dict]], dict]:
    if name not in ADAPTERS:
        raise SystemExit(f'Unknown adapter "{name}". Allowed values: {", ".join(ADAPTERS)}. Fix rag_adapter in settings.json.')
    return ADAPTERS[name]
