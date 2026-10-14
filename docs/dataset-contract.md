# Dataset contract: data generation → evaluation

`rageval-datagen freeze` writes `datasets/<id>/dataset.csv` (UTF-8, one row per test case). `rageval-eval` reads only that file, so the two halves can run on different machines. `manifest.json` beside it records `csv_sha256`; a run refuses a CSV that no longer matches it. A CSV with no manifest (written by hand or by another tool) is accepted and hashed as is.

Required to load: `case_id`, `course_id`, `scope`, `question`. Blank cell = unknown, `[]` = explicitly empty. Code: `rageval/csvio.py`.

| Column | Filled by datagen today | Notes |
|---|---|---|
| case_id, course_id, scope, question | yes | `course_id` is the material subject name; `scope` comes from settings (`current` / `published`) |
| history_json | follow-ups | previous question and its **reference** answer; the evaluator still sends the system's own earlier answer |
| corpus_version | yes | sha256 of the dataset's `pages.jsonl` |
| answerability | yes | `answerable` / `unanswerable`; `needs_clarification` is not generated yet |
| reference_answer, required_facts_json | yes | one fact per key point, ids `f1…` |
| evidence_json | partly | one group per quote, `fact_id` null (generator does not link quote to fact); spans carry `page` only, `start`/`end` null |
| label_status | yes | `human_reviewed` for approved rows (translations only once reviewed), else `automatically_validated` |
| language, question_family_id | yes | family = `base_id`, so translations share it |
| tags_json, provenance_json | yes | provenance also carries the fields the scorer needs (`base_id`, `previous_id`, `scenario`, `question_type`, …) |
| retrieval_qrels_json, relevance_coverage, answer_rubric_json, split, expected_labels_json | no | blank |

## Not done yet (supervisor requirements still open)

1. Link each evidence quote to its fact; block-local zero-based offsets with exclusive end, `page: null` where unreliable.
2. Frozen corpus manifest (document membership, revisions, GCS generations).
3. `split` assigned by question family; coverage report (Thai/English/mixed, paraphrase, follow-up, ambiguity).
4. Instructor audit workflow for `automatically_validated` rows; corpus-level check for unanswerable cases.
5. Evaluation: canonical passage units and qrels, Precision/Recall/F1@K, abstention and clarification metrics, rubric scoring, stage-wise retrieval export, latency/token/cost capture, pre-registered thresholds, breakdown by label status.
