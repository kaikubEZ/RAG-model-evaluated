# rageval: test your RAG system with questions made from your own documents

rageval writes test questions from your course documents, asks them to your RAG endpoint, and scores the answers. It works with any OpenAI-compatible AI model and any RAG endpoint.

**You need:** Python 3.11 or newer, an API key for an OpenAI-compatible model, and (for scoring) a RAG endpoint.

---

## Start in 3 steps

**1. Install**

Windows:
```
python -m venv .venv
.venv\Scripts\activate
pip install -e .[test]
```
macOS / Linux:
```
python -m venv .venv
source .venv/bin/activate
pip install -e .[test]
```

**2. Open the app**
```
rageval app
```
Your browser opens the app. Everything else happens there, one numbered step at a time. Each page explains what to do and ends with a **Next** button.

**3. Follow the steps** (below).

---

## Feature 1: make a dataset from your documents

| Step | What you do |
|---|---|
| 1. Setup | Enter the API key, address and model of the AI that writes the questions. Press **Test** to check them. |
| 2. Material | Create a subject (a course or topic) and upload its files: PDF with selectable text, `.txt` or `.md`. |
| 3. Questions | The AI writes English questions. You read them and approve the good ones. |
| 4. Answers | The AI writes an answer and the required facts for each approved question. You approve them. |
| 5. Translate | Optional. Type language codes such as `th, zh` to translate. |
| 6. Export | Finish the dataset and download `dataset.csv`. |

## Feature 2: score your RAG endpoint (optional)

Switch to it at the top of the sidebar. You can use the `dataset.csv` from Feature 1, or upload your own.

| Step | What you do |
|---|---|
| 1. Setup | Enter the checking model and your RAG endpoint address. Press **Test**. |
| 2. Run and score | Choose a dataset, press **Start run**, then **Score the answers**. |
| 3. Results | Read the scores. Download `report.md`. |
| 4. Compare | Compare two runs to see whether a change helped. |
| 5. Labelling | Optional. Check the checking model against your own judgement. |

Your RAG endpoint must accept `POST {"question": "...", "history": [...]}` and return `{"answer": "...", "contexts": [{"text": "..."}]}`. `contexts` is optional but needed for the evidence scores. Other formats need a small function in `rageval/evaluation/adapters.py`.

---

## Good to know

- **Privacy:** text from your documents is sent to the AI provider you configure. The app asks you to tick a box before it does this.
- **Jobs keep running** if you close the browser tab. Press **Stop** to stop one; starting again continues where it left off.
- **Settings:** keys go in `.env` (see `.env.example`), other options in `settings.json`. Both are saved by the Setup pages, so you do not have to edit them by hand.
- **Private files:** `material/`, `datasets/`, `runs/`, `labels/`, `settings.json` and `.env` are gitignored.
- **Command line:** every step can also run without the app: `rageval-datagen generate | answer | translate | freeze` and `rageval-eval run | score | compare`.
- **Tests:** run `pytest`.

## Status

The code passes its tests with a stand-in model and has run end to end on sample PDFs. It has **not yet been run against a real model or a real RAG endpoint**, so question quality, checking-model agreement and evidence-matching accuracy are unmeasured (see `docs/PRD.md`, success criteria 2, 3 and 8, and section 13 of `docs/architecture.md`).

## More documents

- `docs/PRD.md`: what the program must do
- `docs/architecture.md`: how it is built
- `docs/dataset-contract.md`: the `dataset.csv` format

## License

See `LICENSE`.
