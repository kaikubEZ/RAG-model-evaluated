"""Text per page from material/<subject>/* (architecture 3.2)."""
import logging
from pathlib import Path

from pypdf import PdfReader

from .. import settings, store, text
from ..schema import Page


def _extract(path: Path) -> list[str]:
    if path.suffix.lower() == ".pdf":
        try:
            return [p.extract_text() or "" for p in PdfReader(path).pages]
        except Exception as e:  # pypdf raises many kinds of errors for damaged files
            raise SystemExit(f"{path.name} could not be read as a PDF ({e}). Replace the file or remove it from material/.")
    return text.pseudo_pages(path.read_text(encoding="utf-8-sig", errors="replace"))


def document_text(pages: list[Page]) -> str:
    """One document as a prompt block, with a page marker before each page."""
    return "\n\n".join(f"[[page {p.page}]]\n{p.text}" for p in pages)


def ingest(dataset_dir: Path, subject: str | None = None) -> list[Page]:
    """Read material/<subject>/* (all subjects if None) and write dataset_dir/pages.jsonl.

    With one subject, the pages of other subjects already in the file are kept.
    """
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    limit = settings.load().gen_max_input_tokens
    pages: list[Page] = []
    for sub in [subject] if subject else store.subjects():
        if sub not in store.subjects():
            raise SystemExit(f"There is no folder material/{sub}. Create it and add files, or pick another subject.")
        for path in store.documents(sub):
            doc = f"{sub}/{path.name}"
            texts = _extract(path)
            if not any(t.strip() for t in texts):
                raise SystemExit(f"{doc} has no text that can be read (a scanned PDF?). Replace it with a PDF that has a "
                                 "text layer, or remove it from material/.")
            tokens = text.estimate_tokens("\n".join(texts))
            if tokens > limit:
                raise SystemExit(f"{doc} is about {tokens:,} tokens, more than the {limit:,} the generator can read. "
                                 "Split the file in two, or raise gen_max_input_tokens if your model allows it.")
            pages += [Page(subject=sub, doc=doc, page=i, text=t) for i, t in enumerate(texts, 1)]
    target = dataset_dir / "pages.jsonl"
    kept = [p for p in store.read_jsonl(target, Page) if subject and p.subject != subject]
    store.write_jsonl(target, kept + pages)
    return pages
