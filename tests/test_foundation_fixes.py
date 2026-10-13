"""Regression tests for the foundation findings of the code review."""
import random

import pytest

from rageval import llm, store, text
from rageval.schema import Page

EN = "abcdefghijklmnopqrstuvwxyz"


def rnd(n, seed):
    r = random.Random(seed)
    return "".join(r.choice(EN) for _ in range(n))


def test_found_at_matches_coverage_for_every_prefix():
    q = rnd(150, 8)
    contexts = [rnd(200, 9), q[:70] + rnd(50, 10), rnd(200, 11), q[70:]]
    expected = next(k for k in range(1, 5)
                    if text.coverage(text.normalise(q), [text.normalise(c) for c in contexts[:k]]) >= text.COVERAGE)
    assert text.found_at(q, contexts) == expected == 4


def test_partial_last_line_is_skipped_and_next_append_starts_fresh(root):
    p = root / "answers.jsonl"
    row = Page(subject="s", doc="s/a.txt", page=1, text="t")
    store.append_jsonl(p, row)
    with p.open("a", encoding="utf-8") as f:
        f.write('{"subject": "s", "do')  # killed mid-append
    assert store.read_jsonl(p, Page) == [row]
    store.append_jsonl(p, row.model_copy(update={"page": 2}))
    assert [r.page for r in store.read_jsonl(p, Page)] == [1, 2]


def test_broken_line_in_the_middle_still_raises(root):
    p = root / "x.jsonl"
    good = Page(subject="s", doc="d", page=1, text="t").model_dump_json()
    p.write_text('{"bad' + chr(10) + good + chr(10), encoding="utf-8")
    with pytest.raises(ValueError):
        store.read_jsonl(p, Page)


def test_text_with_unusual_line_separators_stays_one_row(root):
    p = root / "pages.jsonl"
    row = Page(subject="s", doc="d", page=1, text="a b\x0cc\x85d")
    store.write_jsonl(p, [row])
    assert store.read_jsonl(p, Page) == [row]


def test_generator_key_is_not_sent_to_a_different_judge_host(root, monkeypatch):
    for k, v in (("GEN_AI_API_KEY", "k"), ("GEN_AI_URL", "https://gen.example/v1"), ("GEN_AI_MODEL", "m"),
                 ("JUDGE_AI_URL", "https://judge.example/v1")):
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("JUDGE_AI_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_AI_MODEL", raising=False)
    with pytest.raises(SystemExit) as e:
        llm.config("judge")
    assert "JUDGE_AI_API_KEY" in str(e.value)
    monkeypatch.setenv("JUDGE_AI_API_KEY", "jk")
    assert llm.config("judge") == {"api_key": "jk", "url": "https://judge.example/v1", "model": "m"}
