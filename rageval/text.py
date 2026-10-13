"""Text comparison shared by validation and metrics (architecture 3.2 and 3.9)."""
import unicodedata

WINDOW = 20  # matcher window, characters
COVERAGE = 0.8  # share of a quote that must be covered to count as found
QUOTE_MIN, QUOTE_MAX = 100, 300  # normalised characters; QUOTE_MIN is 5 x WINDOW on purpose
DUPLICATE_JACCARD = 0.8
EASY_OVERLAP = 0.5

_THAI_MARKS = {0x0E31, *range(0x0E34, 0x0E3B), *range(0x0E47, 0x0E4F)}


def normalise(s: str) -> str:
    """NFKC, lowercase, drop whitespace / format / control / punctuation, sort runs of Thai marks."""
    s = unicodedata.normalize("NFKC", s).lower()
    out: list[str] = []
    marks: list[str] = []
    for c in s:
        cat = unicodedata.category(c)
        if c.isspace() or cat in ("Cf", "Cc") or cat.startswith("P"):
            continue
        if ord(c) in _THAI_MARKS:
            marks.append(c)
            continue
        if marks:
            out += sorted(marks)
            marks = []
        out.append(c)
    return "".join(out + sorted(marks))


def trigrams(s: str) -> set[str]:
    n = normalise(s)
    return {n[i:i + 3] for i in range(len(n) - 2)}


def jaccard(a: str, b: str) -> float:
    ta, tb = trigrams(a), trigrams(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0


def overlap(question: str, quotes: list[str]) -> float:
    """Share of the question's trigrams that occur in the quotes (easy vs medium)."""
    tq = trigrams(question)
    tx = set().union(*(trigrams(q) for q in quotes)) if quotes else set()
    return len(tq & tx) / len(tq) if tq else 0.0


def coverage(quote: str, contexts: list[str], w: int = WINDOW) -> float:
    """Share of the quote's characters lying in a w-character window that occurs in any context.

    Both sides must already be normalised.
    """
    return _covered(quote, {c[i:i + w] for c in contexts for i in range(len(c) - w + 1)}, w)


def _covered(quote: str, grams: set[str], w: int) -> float:
    if not quote:
        return 0.0
    marked = [False] * len(quote)
    for i in range(len(quote) - w + 1):
        if quote[i:i + w] in grams:
            marked[i:i + w] = [True] * w
    return sum(marked) / len(quote)


def found_at(quote: str, contexts: list[str]) -> int | None:
    """Smallest k such that the first k contexts together cover the quote; raw text in, None if never."""
    q = normalise(quote)
    grams: set[str] = set()  # grown one context at a time, so a target returning hundreds of contexts stays cheap
    for k, context in enumerate(contexts, 1):
        c = normalise(context)
        grams.update(c[i:i + WINDOW] for i in range(len(c) - WINDOW + 1))
        if _covered(q, grams, WINDOW) >= COVERAGE:
            return k
    return None


def quote_on_page(quote: str, page_text: str) -> bool:
    """Quote check: verbatim after normalisation, or covered at the matcher threshold."""
    q, p = normalise(quote), normalise(page_text)
    return bool(q) and (q in p or coverage(q, [p]) >= COVERAGE)


def quote_length_ok(quote: str) -> bool:
    return QUOTE_MIN <= len(normalise(quote)) <= QUOTE_MAX


def eligible(page_text: str) -> bool:
    """A page that can hold an evidence quote."""
    return len(normalise(page_text)) >= QUOTE_MIN


def estimate_tokens(s: str) -> int:
    """One token per non-Latin-script character, one per three Latin-script characters."""
    latin = sum(1 for c in s if ord(c) < 0x250)
    return (len(s) - latin) + latin // 3


def pseudo_pages(s: str, size: int = 3000) -> list[str]:
    """Cut a text file into pieces of about `size` characters at paragraph breaks."""
    pages: list[str] = []
    cur = ""
    for para in s.split("\n\n"):
        if cur and len(cur) + len(para) > size:
            pages.append(cur)
            cur = ""
        cur = f"{cur}\n\n{para}" if cur else para
    if cur.strip():
        pages.append(cur)
    return pages or [s]
