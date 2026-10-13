"""One OpenAI-compatible client for the generator and the judge (architecture section 5)."""
import hashlib
import json
import os
from pathlib import Path
from string import Template
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, TypeAdapter, ValidationError

from . import store

PROMPTS = Path(__file__).parent / "prompts"
T = TypeVar("T")
_clients: dict[str, object] = {}


class BadReply(Exception):
    """The model's reply was not the JSON we asked for, twice."""


def config(role: str) -> dict[str, str]:
    """role is "gen" or "judge"; each unset judge value falls back to the generator value."""
    load_dotenv(store.ROOT / ".env")
    judge_url = os.environ.get("JUDGE_AI_URL")
    if role == "judge" and judge_url and judge_url != os.environ.get("GEN_AI_URL") and not os.environ.get("JUDGE_AI_API_KEY"):
        # never send the generator's key to a different host
        raise SystemExit("JUDGE_AI_URL is set but JUDGE_AI_API_KEY is not. Add the judge's key to .env, "
                         "or remove JUDGE_AI_URL to use the generator for judging.")
    out = {}
    for key in ("API_KEY", "URL", "MODEL"):
        value = os.environ.get(f"GEN_AI_{key}", "")
        if role == "judge":
            value = os.environ.get(f"JUDGE_AI_{key}") or value
        if not value:
            raise SystemExit(f"GEN_AI_{key} is not set. Add it to .env (see .env.example) or use the app's Setup page.")
        out[key.lower()] = value
    return out


def model_name(role: str) -> str:
    return config(role)["model"]


def call(role: str, prompt: str) -> str:
    """Send one prompt, return the reply text. Tests replace this function."""
    cfg = config(role)
    if role not in _clients:
        from openai import OpenAI
        _clients[role] = OpenAI(api_key=cfg["api_key"], base_url=cfg["url"], max_retries=5, timeout=300)
    reply = _clients[role].chat.completions.create(
        model=cfg["model"], temperature=0, messages=[{"role": "user", "content": prompt}])
    return reply.choices[0].message.content or ""


def first_json(text: str):
    """The first JSON value in a reply, ignoring any prose or code fence around it."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text[i:])[0]
            except json.JSONDecodeError:
                continue
    raise ValueError("no JSON found in the reply")


def ask(role: str, prompt: str, shape: type[T]) -> T:
    """Call the model and validate its JSON reply against `shape` (a pydantic model or e.g. list[Model]).

    Retries once on a malformed reply, then raises BadReply.
    """
    adapter = TypeAdapter(shape)
    error = None
    for _ in range(2):
        try:
            return adapter.validate_python(first_json(call(role, prompt)))
        except (ValueError, ValidationError) as e:
            error = e
    raise BadReply(str(error))


def render(name: str, **variables: object) -> str:
    """Fill prompts/<name>.txt. Placeholders are $name, so JSON braces in a prompt need no escaping."""
    text = (PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
    return Template(text).substitute({k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=1)
                                      for k, v in variables.items()})


def version(name: str) -> str:
    """A prompt's version is the first six characters of its content hash."""
    return hashlib.sha256((PROMPTS / f"{name}.txt").read_bytes()).hexdigest()[:6]
