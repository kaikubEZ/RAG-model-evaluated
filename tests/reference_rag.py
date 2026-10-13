"""Reference RAG target (architecture 3.8): plain on purpose, used to test and calibrate the tool. Test code only.

ask(question, history) reads material/ under rageval.store.ROOT. REFERENCE_MODE is one of
normal, random, answer_only, no_history, no_translation.
"""
import math
import os
import random
import zlib
from collections import Counter

from rageval import llm, settings, store, text

PIECE, OVERLAP = 700, 100
MODES = ("normal", "random", "answer_only", "no_history", "no_translation")
_indexes: dict = {}


def _extract(path) -> list[str]:
    if path.suffix.lower() == ".pdf":
        import pypdfium2
        pdf = pypdfium2.PdfDocument(str(path))
        try:
            return [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
        finally:
            pdf.close()
    return [path.read_text(encoding="utf-8")]


def _pieces(raw: str) -> list[str]:
    step = PIECE - OVERLAP
    return [raw[i:i + PIECE] for i in range(0, max(len(raw) - OVERLAP, 1), step)]


def _grams(s: str) -> list[str]:
    n = text.normalise(s)
    return [n[i:i + 3] for i in range(len(n) - 2)]


def _index() -> dict:
    """Pieces, trigram counts and document frequencies, built once per process and project root."""
    key = str(store.ROOT)
    if key not in _indexes:
        pieces: list[str] = []
        for subject in store.subjects():
            for doc in store.documents(subject):
                for page in _extract(doc):
                    pieces += [p for p in _pieces(page) if text.normalise(p)]
        counts = [Counter(_grams(p)) for p in pieces]
        df = Counter(g for c in counts for g in c)
        langs = {"Thai" if any("฀" <= ch <= "๿" for ch in p) else "English" for p in pieces}
        _indexes[key] = {"pieces": pieces, "counts": counts, "df": df, "langs": sorted(langs),
                         "avg": sum(sum(c.values()) for c in counts) / max(len(counts), 1)}
    return _indexes[key]


def _bm25(query: str, idx: dict, k: int, k1: float = 1.5, b: float = 0.75) -> list[str]:
    n = len(idx["pieces"])
    terms = set(_grams(query))
    scored = []
    for i, c in enumerate(idx["counts"]):
        length = sum(c.values())
        score = sum(math.log(1 + (n - idx["df"][g] + 0.5) / (idx["df"][g] + 0.5))
                    * c[g] * (k1 + 1) / (c[g] + k1 * (1 - b + b * length / idx["avg"]))
                    for g in terms if g in c)
        scored.append((score, i))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [idx["pieces"][i] for s, i in scored[:k] if s > 0]


def _queries(question: str, idx: dict, translate: bool) -> str:
    if not translate:
        return question
    reply = llm.call("gen", "Translate this question into each of these languages: " + ", ".join(idx["langs"])
                     + ". Reply with one translation per line and nothing else.\n\n" + question)
    return "\n".join([question, *(line for line in reply.splitlines() if line.strip())])


def ask(question: str, history: list[dict]) -> dict:
    mode = os.environ.get("REFERENCE_MODE", "normal")
    if mode not in MODES:
        raise SystemExit(f"REFERENCE_MODE must be one of: {', '.join(MODES)}.")
    if mode == "no_history":
        history = []
    idx = _index()
    k = settings.load().top_k
    if mode == "random":
        found = random.Random(zlib.crc32(question.encode())).sample(idx["pieces"], min(k, len(idx["pieces"])))
    else:
        query = _queries(" ".join([*(h["question"] for h in history), question]), idx, mode != "no_translation")
        found = _bm25(query, idx, k)
    conversation = "".join(f"Earlier question: {h['question']}\nEarlier answer: {h['answer']}\n\n" for h in history)
    passages = "\n\n".join(f"[passage {i}]\n{p}" for i, p in enumerate(found, 1))
    answer = llm.call("gen", "Answer the question using only the passages. If they do not contain the answer, "
                      f"say so.\n\n{passages}\n\n{conversation}Question: {question}")
    return {"answer": answer.strip(), "contexts": None if mode == "answer_only" else [{"text": p} for p in found]}
