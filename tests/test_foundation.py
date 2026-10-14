import json
import random
import threading
import time

import pytest

from rageval import jobs, llm, settings, store, text
from rageval.schema import Page

EN = "abcdefghijklmnopqrstuvwxyz"
TH = "".join(chr(c) for c in range(0x0E01, 0x0E2F))


def rnd(alphabet, n, seed):
    r = random.Random(seed)
    return "".join(r.choice(alphabet) for _ in range(n))


# ---- normalisation ----

def test_normalise_drops_noise():
    assert text.normalise("Exam-\nple​ ­Text\x00!") == "exampletext"


def test_normalise_thai_mark_order():
    a = "ที่"  # consonant, upper vowel, tone mark
    b = "ท่ี"  # same, marks swapped
    assert text.normalise(a) == text.normalise(b)


def test_normalise_thai_erratic_spaces():
    assert text.normalise("ก ฎ\nหมาย") == text.normalise("กฎหมาย")


# ---- matcher ----

@pytest.mark.parametrize("alphabet", [EN, TH])
def test_quote_found_whole_split_and_with_one_changed_character(alphabet):
    q = rnd(alphabet, text.QUOTE_MIN, 1)
    pad = rnd(alphabet, 300, 2)
    assert text.coverage(q, [pad + q + pad]) == 1.0
    for p in range(1, len(q)):
        assert text.coverage(q, [pad + q[:p], q[p:] + pad]) >= text.COVERAGE, f"split at {p}"
        changed = q[:p] + "#" + q[p + 1:]
        assert text.coverage(q, [pad + changed + pad]) >= text.COVERAGE, f"change at {p}"


@pytest.mark.parametrize("alphabet", [EN, TH])
def test_halves_in_reverse_order_and_unrelated_text(alphabet):
    q = rnd(alphabet, 120, 3)
    assert text.coverage(q, [q[60:], q[:60]]) == 1.0
    assert text.coverage(q, [rnd(alphabet, 2000, 4)]) < text.COVERAGE


def test_found_at_counts_contexts_together():
    q = rnd(EN, 120, 5)
    assert text.found_at(q, ["nothing here", q[:60], q[60:]]) == 3
    assert text.found_at(q, [q]) == 1
    assert text.found_at(q, ["nothing", "here"]) is None


def test_quote_on_page_accepts_one_restored_character():
    page = rnd(EN, 400, 6)
    quote = page[100:220]
    assert text.quote_on_page(quote, page)
    assert text.quote_on_page(quote[:60] + "Z" + quote[61:], page)
    assert not text.quote_on_page(rnd(EN, 120, 7), page)


def test_quote_length_and_eligible_page():
    assert not text.quote_length_ok("x" * 99)
    assert text.quote_length_ok("x" * 100) and text.quote_length_ok("x " * 300)
    assert not text.quote_length_ok("x" * 301)
    assert text.eligible("x" * 100) and not text.eligible("x . " * 99)


def test_token_estimate_is_script_aware():
    assert text.estimate_tokens("a" * 300) == 100
    assert text.estimate_tokens("ก" * 300) == 300


def test_pseudo_pages_cut_at_paragraphs():
    pages = text.pseudo_pages("\n\n".join("p" * 1000 for _ in range(7)))
    assert len(pages) > 1 and all(len(p) <= 3100 for p in pages)
    assert "".join(pages).count("p") == 7000


# ---- store ----

def test_safe_name_and_jsonl_roundtrip(root):
    assert store.safe_name("openai/gpt-4o:free") == "openai_gpt-4o_free"
    rows = [Page(subject="s", doc="s/a.txt", page=1, text="กฎหมาย")]
    store.write_jsonl(root / "x" / "pages.jsonl", rows)
    assert store.read_jsonl(root / "x" / "pages.jsonl", Page) == rows


def test_atomic_write_while_file_is_open_for_reading(root):
    p = root / "progress.json"
    store.write_json(p, {"n": 1})

    def reader():  # a page polling progress holds the file briefly
        with p.open(encoding="utf-8"):
            time.sleep(0.3)

    t = threading.Thread(target=reader)
    t.start()
    time.sleep(0.05)
    store.write_json(p, {"n": 2})
    t.join()
    assert store.read_json(p) == {"n": 2}


# ---- settings ----

def test_settings_defaults_and_candidates(root):
    s = settings.load()
    assert sum(s.questions_per_subject.values()) == 70
    assert [s.candidates(n) for n in (12, 10, 7, 0)] == [16, 13, 10, 0]
    assert s.supports_history is False and s.extra_langs == []


def test_settings_invalid_value_names_the_key(root):
    (root / "settings.json").write_text(json.dumps({"top_k": 0}), encoding="utf-8")
    with pytest.raises(SystemExit) as e:
        settings.load()
    assert "top_k" in str(e.value)


# ---- jobs ----

def test_job_lock_progress_and_stop(root):
    d = root / "datasets" / "x"
    with jobs.Job(d, "generate") as job:
        assert jobs.is_running(d)
        with pytest.raises(jobs.Busy):
            jobs.Job(d, "generate").__enter__()
        job.step("ingest", total=2)
        job.tick("a")
        assert jobs.progress(d)["done"] == 1
        jobs.request_stop(d)
        with pytest.raises(jobs.Stopped):
            job.tick("b")
    assert not jobs.is_running(d)
    assert not (d / "job.stop").exists()


def test_stale_lock_is_taken_over(root):
    d = root / "runs" / "x"
    store.write_json(d / "job.lock", {"pid": 1, "job": "run", "heartbeat": time.time() - jobs.STALE_SECONDS - 1})
    assert not jobs.is_running(d)
    with jobs.Job(d, "run"):
        assert jobs.is_running(d)


def test_job_records_stopped_and_failed(root):
    d = root / "runs" / "y"
    with pytest.raises(jobs.Stopped):
        with jobs.Job(d, "run"):
            raise jobs.Stopped()
    assert jobs.progress(d)["message"] == "stopped"
    with pytest.raises(ValueError):
        with jobs.Job(d, "run"):
            raise ValueError("boom")
    assert jobs.progress(d)["message"] == "failed: boom"


# ---- llm ----

def test_first_json_ignores_prose_and_fences():
    assert llm.first_json('Here you go:\n```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert llm.first_json('[{"x": 1}] trailing') == [{"x": 1}]
    with pytest.raises(ValueError):
        llm.first_json("no json")


def test_ask_retries_once_then_raises(monkeypatch):
    replies = iter(["not json", '{"subject": "s", "doc": "d", "page": 1, "text": "t"}'])
    monkeypatch.setattr(llm, "call", lambda role, prompt: next(replies))
    assert llm.ask("gen", "p", Page).page == 1
    monkeypatch.setattr(llm, "call", lambda role, prompt: "still not json")
    with pytest.raises(llm.BadReply):
        llm.ask("gen", "p", Page)


class _Client:
    """Stands in for the OpenAI client; `delay` is how long one request takes."""
    def __init__(self, delay=0.0):
        self.delay = delay
        self.chat = self
        self.completions = self

    def create(self, **kw):
        import time
        from types import SimpleNamespace as NS
        time.sleep(self.delay)
        return NS(choices=[NS(message=NS(content="hi"))], usage=NS(prompt_tokens=7, completion_tokens=3))


def _models(monkeypatch, delay):
    for k in ("API_KEY", "URL", "MODEL"):
        monkeypatch.setenv(f"GEN_AI_{k}", "x")
    monkeypatch.setattr(llm, "_clients", {"gen": _Client(delay)})


def test_model_calls_inside_a_job_are_counted(root, monkeypatch):
    _models(monkeypatch, 0)
    d = root / "datasets" / "d"
    with jobs.Job(d, "generate"):
        assert llm.call("gen", "p") == "hi" and llm.call("gen", "p") == "hi"
    assert jobs.usage(d) == {"gen": {"calls": 2, "prompt": 14, "completion": 6}}
    with jobs.Job(d, "generate"):  # a restart adds to the total
        llm.call("gen", "p")
    assert jobs.usage(d)["gen"]["calls"] == 3


def test_stop_abandons_a_model_call_in_flight(root, monkeypatch):
    import threading
    import time
    _models(monkeypatch, 30)  # a request that would take 30 s
    d = root / "datasets" / "d"
    threading.Timer(0.3, jobs.request_stop, [d]).start()
    t0 = time.monotonic()
    with pytest.raises(jobs.Stopped):
        with jobs.Job(d, "generate"):
            llm.call("gen", "p")
    assert time.monotonic() - t0 < 5 and not jobs.is_running(d)
