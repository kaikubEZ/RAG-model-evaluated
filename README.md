# Universal RAG Evaluator — Open Source

An open-source universal framework to automatically evaluate Retrieval-Augmented Generation (RAG) endpoints against any content, using AI-generated questions.

---

## What it does

1. **Ingest your material** — upload or point to your source content (PDFs with a text layer, `.txt`, `.md`). Each document must fit in the question-generation model's input.
2. **Configure an AI generator** — set your AI API (for question generation) via `.env`.
3. **Generate and review test questions** — the framework calls the configured AI to generate questions from your material; you review and approve them before they are used.
4. **Point to a target API** — specify the RAG endpoint you want to test.
5. **Automatic RAG evaluation** — the framework runs the endpoint, compares results against the source material, and produces evaluation metrics automatically.

---

## Project goals

- Open source and universal: works with any RAG endpoint, any source material.
- End-to-end automation: from ingestion to evaluation report.
- Configurable AI backend: bring your own LLM API for question generation.

---

## Quick start

Needs Python 3.11 or newer.

```
python -m venv .venv
.venv\Scriptsctivate          (Windows; on macOS/Linux: source .venv/bin/activate)
pip install -e .[test]
rageval-datagen app     # build and freeze a dataset (writes datasets/<id>/dataset.csv)
rageval-eval app        # run the endpoint on a dataset.csv and score it
```

The data-generation app opens in your browser. Its sidebar is the workflow, one numbered page per step:

1. **Setup**: API key, address and model for question generation (any OpenAI-compatible API).
2. **Material**: create a subject and upload PDFs (with a text layer), `.txt` or `.md` files.
3. **Questions**: write English questions from the material, then approve the good ones.
4. **Answers**: write the English reference answer and required facts for each approved question, then approve them.
5. **Translate** (optional): type any language codes (`th, zh`) to translate questions and answers.
6. **Export**: freeze the dataset and download `dataset.csv`.

Every page says what to do, shows how far each step has got, and ends with a **Next** button. The evaluation app (`rageval-eval app`) is a separate, optional part: it runs a RAG endpoint on a `dataset.csv` and scores it.

The program is two parts joined by the CSV in `docs/dataset-contract.md`. Data generation, in order: `rageval-datagen generate` (English questions), `answer` (English answers), `translate`, `freeze`. Evaluation: `rageval-eval run`, `score`, `compare`.

Your endpoint must accept `POST {"question": "...", "history": [...]}` and return `{"answer": "...", "contexts": [{"text": "..."}]}`; `contexts` is optional but needed for retrieval scores. Other shapes need a small adapter function in `rageval/evaluation/adapters.py`.

Text from your documents is sent to the model provider you configure. `material/`, `datasets/`, `runs/`, `labels/`, `settings.json` and `.env` are gitignored.

---

## Status

Built from `docs/PRD.md` and `docs/architecture.md`. The code passes its tests with a stand-in model, and has run end to end on sample PDFs with that stand-in. It has **not yet been run against a real model or a real RAG endpoint**, so question quality, judge agreement and matcher accuracy (success criteria 2, 3 and 8 in the PRD) are unmeasured. See section 13 of the architecture for what was and was not verified.

Run the tests with `pytest`.

---

## License

See `LICENSE` for details.
