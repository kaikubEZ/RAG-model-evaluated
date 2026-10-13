"""settings.json: everything that shapes a dataset or a run (PRD FR-14). Secrets stay in .env."""
import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import store


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions_per_subject: dict[str, int] = {
        "fact": 12, "definition": 12, "comparison": 12, "multi_hop": 12, "exception": 12, "unanswerable": 10,
    }
    follow_ups_per_subject: dict[str, int] = {"same": 7, "similar": 7, "diff_topic": 7}
    overgenerate: float = Field(1.3, ge=1.0, le=5.0)
    extra_langs: list[str] = []
    generation_guidance: str = ""
    top_k: int = Field(5, ge=1)
    rag_adapter: str = "http_json"
    supports_history: bool = False
    gen_max_input_tokens: int = Field(150_000, ge=1000)

    def candidates(self, target: int) -> int:
        """How many candidates to generate for a target count."""
        return math.ceil(round(target * self.overgenerate, 6))


def path() -> Path:
    return store.ROOT / "settings.json"


def load() -> Settings:
    p = path()
    if not p.exists():
        return Settings()
    try:
        return Settings.model_validate(json.loads(p.read_text(encoding="utf-8")))
    except (ValidationError, json.JSONDecodeError) as e:
        raise SystemExit(f"settings.json is invalid: {e}\nFix the key named above, or delete the file to use defaults.")


def save(s: Settings) -> None:
    store.write_text_atomic(path(), json.dumps(s.model_dump(), indent=2, ensure_ascii=False))
