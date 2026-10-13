# Product Requirements Document — Universal RAG Evaluator

Status: draft for build, revision 6, 2026-10-13. Supersedes the earlier PRD and `flaw.md`. Background research is in `research/`. Section 14 lists what each revision changed.

## 1. Overview

An open-source tool that builds a question set from the user's own documents, asks those questions to a RAG endpoint, and scores the answers with stated uncertainty.

The user gets three things:

1. A reviewed, frozen question set with gold answers and gold evidence.
2. Scores for one RAG endpoint on that set.
3. A statistical comparison of two runs on the same set (for example rerank on versus rerank off).

## 2. Problem

Teams building RAG systems over private documents have no test set for those documents, and public benchmarks do not cover them. Writing questions by hand is slow, and unvalidated LLM scoring gives numbers nobody can defend.

## 3. Users

- Developers tuning a RAG system over their own documents.
- Teams choosing between two RAG configurations or vendors.
- People who are not developers: the whole workflow must be usable from the app, with their own material, without editing files by hand.

## 4. Scope

### What v1 tests

| Situation in real use | Covered by |
|---|---|
| A single question of a given kind | Six question types (FR-2) |
| A question the documents cannot answer | `unanswerable` type |
| A follow-up question in a conversation, on the same, a similar, or a different topic | Two-turn follow-ups (FR-12) |
| A question in a different language from the documents | Test languages (FR-13) |

### In scope for v1

- Source material: PDF with a text layer, `.txt`, `.md`, organised as `material/<subject>/`. Users add their own; the material in the repository is mock data for development.
- Each document must fit in the generator model's input (checked before any call). There is no limit on documents per subject, but cost grows with every document.
- Any language the generator and judge models handle, for documents and for questions.
- One endpoint evaluated per run; two runs compared after the fact.
- A Streamlit app for the whole workflow, and the same steps as commands.

### Non-goals for v1

"Every possible real use" is not reachable; these are left out deliberately and listed so nobody assumes they are tested.

- **Hard negatives**, deferred by the owner on 2026-10-13. v1 does not generate them and does not compute the hard-negative rate; `hard_negative` stays in the schema as `null`. Mentions of them elsewhere in this document describe the later design.

- Conversations longer than two turns.
- Vague questions where the right response is to ask the user what they mean.
- Misspelled or noisy questions.
- Questions that need facts from two different documents.
- Scanned PDFs (OCR), images, tables as images, audio, video.
- Documents larger than the generator's input.
- Driving the target's internals. The tool never switches rerank on or off; the user runs two configurations.
- BLEU and ROUGE. They reward word overlap, not correctness.
- Correcting scores for judge error (prediction-powered inference, as in ARES). v1 reports judge agreement beside the scores.
- Authentication, multi-user access, hosted deployment.

## 5. Workflow

```
material/ ─> generate ─> validate ─> human review ─> translate ─> frozen dataset
                                                                       │
                                    RAG endpoint <── run ──────────────┤
                                                        │              │
                                                     score ──> report  │
                                                        │              │
                                          compare (two runs on the same dataset)
```

1. User adds documents under a subject and enters keys and settings.
2. `generate` produces candidate questions and follow-ups per subject and filters them automatically.
3. User reviews, edits and approves.
4. `translate` writes the approved questions in any extra test languages; the user may review those too, then freezes the set.
5. `run` sends the frozen questions to the endpoint and stores raw answers.
6. `score` computes metrics and writes a report.
7. `compare` tests the difference between two runs.

Every step is a button in the app and a command.

## 6. Functional requirements

### FR-1 Ingestion

- Extract text per page from each file under `material/<subject>/`. A `.txt` or `.md` file is cut into page-sized pieces.
- Stop with a clear message for a file with no extractable text, or one too large for the generator.
- The page is the unit of evidence. The tool does no chunking of its own, because the target system's chunks are unknown.

### FR-2 Question generation

- All counts are settings the user can change (FR-14). Defaults per subject: 70 single questions and 21 follow-ups (FR-12), so 273 for three subjects, each in its document's own language.
- The tool generates more candidates than the target (default 1.3 times, rounded up) so that validation and review rejections do not leave the set short.
- Question types with default targets and candidates per subject:

| Type | Example | Target | Candidates |
|---|---|---|---|
| `fact` | Is UDP reliable? | 12 | 16 |
| `definition` | What is UDP? | 12 | 16 |
| `comparison` | How do UDP and TCP differ? | 12 | 16 |
| `multi_hop` | Why is UDP preferred in video games? | 12 | 16 |
| `exception` | When should UDP not be used? | 12 | 16 |
| `unanswerable` | A plausible in-domain question the material does not answer | 10 | 13 |
| Total | | 70 | 93 |

- Questions are first written in the document's language. Other languages come from FR-13.
- `multi_hop` questions must have evidence on at least two different pages of one document.
- Questions are spread across each document: every question is assigned a page range to draw its evidence from.
- Where the material allows, a question carries a hard negative: a passage on the same topic with shared keywords that does not answer it.
- The user can add free-text guidance for the generator, for example "focus on chapters 3 to 5" or "write as a first-year student would ask".
- **Difficulty is computed by code, not chosen by the model**:

| Level | Rule |
|---|---|
| `hard` | Evidence on two or more pages |
| `easy` | Evidence on one page, and the question shares most of its wording with the evidence quotes |
| `medium` | Evidence on one page, and the question is worded differently from the quotes |

**How many questions is enough.** There is no universal number; it depends on how small a difference the user needs to see. The report prints the smallest difference each comparison could have detected. As a rough guide, to be replaced by measured figures in phase 3:

| Distinct questions compared | Smallest detectable difference in a 0-to-1 score |
|---|---|
| 70 (one subject) | about 0.12 |
| 250 (default set) | about 0.06 |
| 1,000 | about 0.03 |

Halving the detectable difference takes four times the questions. Adding a language does not add distinct questions. A breakdown cell needs at least 20 questions to be shown.

### FR-3 Automatic validation

Each candidate must pass every check that applies or it is dropped. Missing candidates are regenerated for at most two more rounds; any remaining shortfall is reported, not hidden, together with how many candidates each check dropped.

1. **Quote check (code)**: every evidence and hard-negative quote appears verbatim on the cited page after text normalisation, and is 100 to 300 characters long, counted after normalisation (roughly one to three sentences).
2. **Structure (code)**: type rules from FR-2 and scenario rules from FR-12.
3. **Duplicates (code)**: near-duplicate questions within a subject are removed.
4. **Answerable from the quotes (judge)**: given the question, the gold key points and only the evidence quotes, not the pages, the judge's answer covers every gold key point. A quote the judge did not need is removed. This makes each evidence item necessary, which the retrieval metrics assume.
5. **Standalone (judge)**: the question makes sense without seeing the document. For a follow-up, its self-contained form is checked; and a `same` follow-up as typed must not make sense alone, or it would not test the conversation.
6. **Unanswerable (judge)**: for `unanswerable` candidates, each document in the subject is checked. If the judge claims an answer for any document, the candidate is dropped. The judge must give a supporting quote, which code verifies and the reviewer can see; a claim that cannot be verified still drops the candidate, since keeping a question that may be answerable would penalise a correct system.
7. **Closed-book flag (generator model, no context)**: if the model answers an answerable question fully with no documents, the question is kept and flagged `closed_book`; reports show scores with and without flagged questions. This runs at freeze, on the questions kept, because it only sets a flag.

### FR-4 Human review gate

- Mandatory. No source-language question enters a frozen dataset without `approved` status.
- The reviewer sees question, gold answers and evidence quotes side by side. For unanswerable questions the reviewer sees the closest passage the judge found. For a follow-up the reviewer sees the first-turn question above it.
- **Everything is editable**: question text, gold answers, evidence quotes, hard negative, type, scenario. The page shows the text of the cited page beside each quote. The reviewer can also add a question by hand. On save, the code checks (quote, structure, duplicate) run again and a row that fails cannot be approved.
- Per question: approve or reject. Bulk approve exists for convenience, and the manifest records how many questions were approved in bulk.
- Translated variants (FR-13) are produced after the source questions are approved, and are shown beside their source question for editing. Editing an approved source question afterwards discards its translations, which are then redone. Because a reviewer may not read every test language, reviewing them is optional; the manifest records how many were reviewed by a person.

### FR-5 Frozen dataset

- Freezing takes approved questions, up to the targets per subject, always including the first-turn question of every follow-up it takes, and writes them with a manifest containing a content hash, the settings used, generator model, prompt versions, counts and shortfall.
- A frozen dataset is never modified. "Duplicate as draft" copies it into a new, editable dataset.
- Adding or changing material means generating a new dataset.
- Every run records the dataset hash; `compare` refuses runs with different hashes.

### FR-6 Target endpoint contract

The tool talks to the target through an adapter. The default adapter sends:

```
POST {RAG_URL}
Authorization: Bearer {RAG_API_KEY}
{
  "question": "...",
  "history": [{"question": "first-turn question", "answer": "the target's own answer to it"}]
}
```

`history` is empty for a single question. The tool expects:

```json
{
  "answer": "text",
  "contexts": [{"text": "retrieved passage"}]
}
```

- `answer` is required.
- `contexts` is optional. When present it must be in final rank order, best first. Other fields are ignored.
- Endpoints with another shape are supported by writing a small adapter function (see architecture).

| Endpoint capability | Metrics available |
|---|---|
| Returns `answer` only | Answer recall, contradiction rate, refusal metrics, language match |
| Also returns `contexts` | The above plus Recall@k, MRR, hard-negative rate, faithfulness |
| Accepts `history` (the user confirms this in Setup) | Follow-ups are asked; otherwise they are skipped |

The report states what was measured. Missing metrics are shown as "not measured", never as zero.

### FR-7 Metrics

All scores are in [0, 1]. Matching and judging rules are in the architecture.

| Metric | Applies to | Definition | By |
|---|---|---|---|
| Recall@k | Answerable, contexts returned | Gold evidence items found in the top k contexts ÷ gold evidence items | Code |
| MRR | Same | 1 ÷ the smallest k at which any gold evidence item is found; 0 if none | Code |
| Hard-negative rate | Questions with a hard negative whose gold evidence was found | Share where the hard negative is found at a better rank than the gold evidence | Code |
| Answer recall | Answerable | Gold key points supported by the answer ÷ gold key points | Judge |
| Contradiction rate | Answerable | Share of questions where the answer contradicts at least one gold key point | Judge |
| Faithfulness | Answerable, non-refused answers, contexts returned | Claims in the answer supported by the returned contexts ÷ claims in the answer | Judge |
| Refusal accuracy | Unanswerable | Share of unanswerable questions the system declined to answer | Judge |
| False refusal rate | Answerable | Share of answerable questions the system declined | Judge |
| Language match | All answered questions | Share of answers written in the language of the question | Judge |

- Lower is better for hard-negative rate, contradiction rate and false refusal rate.
- An evidence item is "found in the top k" when the top k contexts together contain it; a quote split across two retrieved chunks still counts. This keeps the metric from depending on the target's chunk size.
- Evidence is always matched in the document's language, whatever language the question is in. That is what tests cross-language retrieval.
- Recall is reported beside the mean number and length of contexts returned, because a system that returns more text finds more evidence.
- Hard-negative rate is reported with its n and only where gold was found, so a system that retrieves nothing does not score well on it.

### FR-8 Judge

- Configured separately; each unset judge value falls back to the matching generator value.
- Runs at temperature 0 with versioned prompts committed to the repo. The rubric for every judge decision is written out in the architecture.
- The report names the judge model and prompt version beside every judged metric.
- **Validation** belongs to a judge model plus prompt version, not to a run:
  - The user labels 150 decisions per test language without seeing the judge's verdict: 50 each of key-point verdicts (supported, contradicted, absent), claim verdicts and refusal verdicts.
  - The sample is balanced across the judge's verdict values, drawing on several runs where one run has too few of a kind.
  - The report shows, for each decision type, the number of labels and the agreement with its 95% interval, and Cohen's kappa. It is agreement on a balanced sample, which is a harder test than agreement on a typical run.
  - The judge counts as validated for a language when overall agreement is at least 85% and no decision type is below 80%. A decision type with fewer than 30 labels is not validated. Metrics fed by an unvalidated language or decision type carry a visible "judge not validated" warning.
- Recommended: a judge model different from the one inside the target system.

### FR-9 Reporting

- **Scores are reported per group and never pooled across groups**, because a pooled average would change with how many languages and follow-ups the user chose:
  - the **headline**: single questions asked in their document's own language;
  - follow-ups, by scenario, beside the scores of their first turns;
  - each translated direction, named by question language and document language (for example "English questions, Thai documents"), with the line "translations not reviewed by a person: n of m".
- Every aggregate score has a 95% confidence interval and the number of questions behind it.
- **Endpoint errors count as failures**: a question the endpoint failed on scores 0 in the headline numbers. A second column shows scores excluding errored questions. Judge errors are excluded and counted.
- Headline breakdowns by subject, question type and difficulty. A cell with fewer than 20 questions is shown as "n too small". Groups themselves are always shown, with their n and interval.
- Scores are a statement about these documents and this question set, not about other documents; the report says so.
- Output: `report.md` plus machine-readable per-question scores.

### FR-10 Run comparison

- `compare` takes two runs on the same frozen dataset. It compares the headline group by default, or one named group (a scenario, or a translated direction).
- The user names one **primary metric** (default answer recall). Its verdict is the answer to "is B better than A". Other metrics are shown as exploratory, without correction for multiple tests.
- For each metric: mean paired difference, 95% paired-bootstrap interval, and the **smallest difference this comparison could have detected**.
- In a comparison, endpoint errors count as failures.
- Verdict: "better", "worse", or "no detectable difference" when the interval includes zero (inclusive). The last is always printed with the detectable size, so it is not read as "no difference".
- Verdicts are at system level only. Per-question differences are listed for inspection and labelled as not reliable individually.
- This is how "does rerank help" is answered: run once with rerank, once without, compare.

### FR-11 App

A Streamlit app that covers the whole workflow, so a user never needs the command line or a text editor. See `research/gui-templates.md`.

| Page | The user can |
|---|---|
| Setup | Enter API keys, model names and the endpoint address; test each connection |
| Material | Create subjects, upload and remove files, see page counts and whether each file fits |
| Generate | Change every generation setting (FR-14), start generation, watch progress |
| Review | Edit, add, approve and reject questions; start translation; freeze; duplicate a frozen dataset as a draft |
| Run and score | Pick a dataset, label the run, start it, watch progress |
| Results | Read headline scores, breakdowns and individual questions |
| Compare | Pick two runs and a primary metric |
| Labelling | Label judge decisions and matcher pairs |

- Long jobs keep running if the browser tab is closed, can be stopped with a button, and resume where they stopped.
- The privacy confirmation (section 9) appears in the app before the first call.
- The Review page ships first, with the dataset phase, because it is the review gate.

### FR-12 Follow-up questions

Tests what happens when a user asks a second question in the same conversation.

- A follow-up is asked after a first-turn question from the dataset, with that question and the target's own answer to it passed as `history`.
- Three scenarios, defined so code can check them:

| Scenario | Meaning | Rule | What it tests |
|---|---|---|---|
| `same` | Still about the same thing, and worded so it cannot be understood without the first turn ("and when did it end?") | Shares at least one evidence page with the first turn | Using the conversation to understand the question |
| `similar` | A related topic in the same document | Same document, no shared evidence page | Retrieving again instead of reusing old context |
| `diff_topic` | An unrelated topic | A different document in the subject | Not carrying stale context into a new topic |

- The rules place a follow-up by where its evidence sits, which usually but not always matches topic. The reviewer can correct a scenario label, and corrections are counted.
- Default 7 per scenario per subject (21), generated as 10 candidates each. With defaults, per-scenario results are indicative only; raise the count to study one scenario.
- Each follow-up stores the question as the user would type it and a self-contained form of the same question, which validation and the judge use.
- Both turns are scored: the first turn as an ordinary question, the follow-up with the same metrics. Reports show follow-ups by scenario beside single questions.
- A follow-up whose first-turn question is rejected in review is dropped at freeze.
- If the endpoint does not accept `history`, follow-ups are skipped and reported as not measured. The setting is off until the user confirms it, so that an endpoint which ignores history is not silently marked down.
- Retrieval scores on `same` follow-ups assume the target retrieves again; a target that answers correctly from the conversation gets no retrieval credit, and the report says so.

### FR-13 Test languages

Tests questions asked in a language other than the documents', for example an English question over Thai material.

- Every question always exists in its document's own language. The user adds any number of extra test languages; the default is none.
- For each extra language, every approved question and its gold answers are translated, after review of the source-language questions. Evidence quotes are never translated: the documents are in their own language, and retrieving them from a question in another language is the thing being tested.
- Each translation is checked by the judge for meaning: the translated question must ask the same thing, and each translated key point must state the same fact. A failed translation is redone once, then dropped and reported.
- All language versions of a question share one base id, so languages can be compared on exactly the same questions.
- Each extra language adds a variant of every question not already in that language. For the mock material (two English subjects, one Thai), adding English adds 91 questions and adding Thai adds 182.
- Each extra language also needs 150 judge decisions labelled by someone who reads it; until then its judged scores carry the "judge not validated" warning.
- Follow-ups are translated too; a follow-up in a language is asked after its first turn in that same language.

### FR-14 Settings

Everything that shapes a dataset or a run is a setting with a default, editable in the app and stored in one file. API keys are kept separately.

| Setting | Default |
|---|---|
| Questions per subject, per type | 12, 12, 12, 12, 12, 10 |
| Follow-ups per subject, per scenario | 7, 7, 7 |
| Over-generation factor | 1.3 |
| Extra test languages | none |
| Generation guidance (free text) | empty |
| Top k for retrieval metrics | 5 |
| Endpoint adapter; whether it accepts history | `http_json`; no, until confirmed |
| Largest document for the generator, in tokens | 150,000 |

- A setting of zero switches that kind of question off.
- The settings used are copied into each dataset and run manifest.
- Prompts are plain text files; a changed prompt is detected by its hash and recorded as a new prompt version.

## 7. Data schema

### Question (one JSON object per line)

```json
{
  "id": "law-0007.en",
  "base_id": "law-0007",
  "subject": "law",
  "lang": "en",
  "source_lang": "th",
  "question": "And when did it stop applying?",
  "resolved_question": "When did the Law of the Twelve Tables stop applying?",
  "previous_id": "law-0003.en",
  "scenario": "same",
  "question_type": "fact",
  "difficulty": "easy",
  "answerable": true,
  "gold_ans": ["key point one"],
  "gold_evidence": [
    {"doc": "law/02-roman-law.pdf", "page": 4, "quote": "verbatim text in the source language"}
  ],
  "hard_negative": {"doc": "law/02-roman-law.pdf", "page": 9, "quote": "verbatim text"},
  "closed_book": false,
  "translation_reviewed": false,
  "status": "approved"
}
```

| Field | Rule |
|---|---|
| `id`, `base_id` | `base_id` identifies the question; `id` adds the language |
| `lang`, `source_lang` | ISO 639-1. Equal for a source-language question |
| `question` | As the user would type it |
| `resolved_question` | Self-contained form; equal to `question` except for `same` follow-ups |
| `previous_id`, `scenario` | Set only on follow-ups; `scenario` is `same`, `similar` or `diff_topic`. A follow-up may be of any answerable type |
| `gold_ans` | One to four key points in `lang`, each a single checkable statement. Empty for unanswerable |
| `gold_evidence` | At least one item for answerable questions, every item necessary; empty for unanswerable. Each quote one contiguous span of 100 to 300 normalised characters, in `source_lang` |
| `hard_negative` | One item or `null` |
| `question_type` | `fact`, `definition`, `comparison`, `multi_hop`, `exception`, `unanswerable` |
| `difficulty` | `easy`, `medium`, `hard`; computed on the source-language question |
| `translation_reviewed` | Whether a person reviewed this translated variant |
| `status` | `candidate`, `approved`, `rejected` |

Changes from the original field list:

| Original | Now | Why |
|---|---|---|
| `gold_chunk` (chunk ids) + `evidence_quote` | `gold_evidence` (doc, page, quote) | Chunk ids from our own splitter cannot be matched against another system's chunks; a quoted span can |
| `hard_negative_ref` | `hard_negative` (doc, page, quote) | Same reason |
| `reasoning_required` | removed | Derivable from `question_type` and the number of evidence pages |
| `scenario` | kept, defined in FR-12 | Now describes how a follow-up relates to the first turn |

## 8. Configuration

Secrets in `.env`:

| Variable | Required | Purpose |
|---|---|---|
| `GEN_AI_API_KEY`, `GEN_AI_URL`, `GEN_AI_MODEL` | yes | Question generation and translation (OpenAI-compatible API) |
| `JUDGE_AI_API_KEY`, `JUDGE_AI_URL`, `JUDGE_AI_MODEL` | no | Judge; each defaults to the generator value |
| `RAG_API_KEY`, `RAG_URL` | for the default adapter | Target endpoint |

Everything else is in `settings.json` (FR-14), written by the app's Setup and Generate pages or by hand. Without the file, defaults apply.

Matching thresholds and the random seed are constants in code, recorded in every run manifest and report, so that two runs cannot silently differ in them.

## 9. Non-functional requirements

- **Open source**: MIT (`LICENSE`), with permissively licensed dependencies only. Prompts are committed so results can be reproduced.
- **Easy to use**: a person who has never used a command line can go from their own files to a report in the app. Every error message says what to do next.
- **Reproducible**: the same frozen dataset, stored answers and judge give the same scores. Judge replies are stored and reused on re-scoring, since a model at temperature 0 can still vary.
- **Resumable**: an interrupted job continues where it stopped; questions that errored are retried.
- **Privacy**: document text is sent to the generator and judge providers. The tool names the host and asks for confirmation before the first call, because material may be under usage terms. `material/`, datasets, runs, labels, `settings.json` and `.env` are gitignored. Git rules do not stop a cloud-sync folder (OneDrive, Dropbox) from uploading them; the tool warns when the project sits in one.
- **Cost, in calls, for the mock material with default settings** (9 documents, 369 candidates, one language):
  - Dataset build: 63 generation calls, about 450 validation calls, and about 490 closed-book calls at freeze, so about 1,000 with nothing dropped and roughly 1,100 to 1,300 with refills.
  - About 20 to 30 of those calls per document carry the whole document as input. Input tokens therefore run to tens of times the corpus size; provider prompt caching cuts this substantially where available.
  - Each extra test language: about 20 to 40 more calls at build time for the mock material.
  - Each run: 273 endpoint calls and up to 516 judge calls, plus the same again in proportion for translated variants.
  - Token cost and wall-clock time depend on the models chosen and are not estimated; a dataset build should be expected to take an hour or more.

## 10. Success criteria

Tested against the mock material and the reference RAG target that ships in `tests/` (see architecture).

1. **Dataset**: the tool produces a frozen dataset with at least 60 approved single questions and 15 follow-ups per subject, and reports any shortfall by type and scenario.
2. **Matcher**: on hand-labelled quote-and-contexts pairs from the reference target, which extracts text with a different library from the tool, at least 50 retrieved and 50 not retrieved, the evidence matcher misses at most 10% of retrieved quotes and wrongly accepts at most 5%.
3. **Judge**: on 150 blind labels per language from reference-target answers, overall agreement at least 85% and no decision type below 80%.
4. **Reproducible**: re-scoring stored answers gives identical scores. With stored judge replies ignored, at most 5% of judge verdicts change.
5. **No false alarm**: two separate runs of the same reference target compare as "no detectable difference" on the primary metric. A 95% test fails by chance one time in twenty, so one repeat is allowed.
6. **Sensitive to retrieval**: with the reference target switched to contexts drawn at random from the whole corpus, Recall@5 is below 0.05 and `compare` reports "worse" on Recall@5 and on answer recall.
7. **Sensitive to conversation**: on a dataset built with 20 `same` follow-ups per subject, comparing the reference target with and without history reports "worse" for answer recall on the `same` group and "no detectable difference" on the headline group.
8. **Languages**: a dataset with Thai and English as extra languages builds, runs and reports each translated direction separately; the reference target, which translates queries, finds evidence across languages (Recall@5 above 0.2 in each direction); a reader of both languages finds at least 27 of 30 sampled translations faithful. This is a smoke test, not a measurement of translation quality.
9. **Usable**: a person who was not involved in building the tool goes from a folder of their own files to a report using only the app and its on-screen text.

Tracked but not a pass/fail criterion: the share of validated candidates the reviewer approves without edits. A low share means the generator prompt or the validation needs work.

## 11. Milestones

| Phase | Delivers | Done when |
|---|---|---|
| 1. Dataset | Text-extraction check on the mock PDFs, review-form trial, ingestion, generation of single questions, validation, Review page, freeze | Criterion 1, single questions |
| 2. Run and score | Adapters, reference target, run, metrics, judge with its prompts in final form, confidence intervals, report, Labelling page, matcher calibration | Criterion 2; criterion 4 first half; criterion 6 for Recall@5 |
| 3. Trust | Judge labelling and agreement, `compare` | Criteria 3, 4, 5, 6 |
| 4. Coverage | Follow-ups, test languages | Criterion 1 in full; criteria 7 and 8 |
| 5. App | Setup, Material, Generate, Run and score, Results, Compare pages | Criterion 9 |

Each phase is usable on its own, and **the end of phase 3 is the first release worth using**: a tool proven on single questions in the documents' own language. The schema, the endpoint contract and the judge prompts carry the follow-up and language parts from the start, so phase 4 adds no breaking change and does not void judge labels.

## 12. Threats to validity

Limits of the method that users should know. Sources are summarised in `research/papers.md`; only abstracts were checked for most.

| Threat | Evidence | What v1 does | What remains |
|---|---|---|---|
| Synthetic questions differ from real user questions | Synthetic benchmarks rank retriever configurations reliably but not consistently for generators (arXiv 2508.11758); common generation tools produce unbalanced data (arXiv 2411.19710) | Fixed type quotas, follow-ups, human review, free-text guidance | Scores for comparing generator models deserve less trust than scores for comparing retrieval setups |
| Translated questions are not native questions | Translated benchmarks carry simpler syntax and word choice (arXiv 2502.17163, abstract not re-checked) | Meaning check on every translation; optional human review, counted and shown in the report | Cross-language scores may be easier or harder than real questions from native speakers. By default the model that translates also checks the translation |
| Scenario labels are positional | — | Page-based rules, reviewer corrections counted | "Same topic" is approximated by "same page"; documents that overlap in topic can produce `diff_topic` follow-ups that are not really a change of topic |
| Judge errors | Automatic scoring agrees with humans at system level; per-topic agreement needs further work (arXiv 2504.15068). ARES uses a few hundred human labels to correct its judge (arXiv 2311.09476) | Blind judge validation per language; verdicts at system level only | No correction of judge bias. 150 labels give an interval of several points either side. The judge can omit a claim when listing claims, which labelling does not catch |
| Dataset-build judge calls are not label-validated | — | Every source-language question passes human review | A wrongly labelled unanswerable question penalises a correct system. Unreviewed translations rest on the judge alone |
| Model already knows the answer | Benchmarks note models may have seen the data (arXiv 2504.17137, limitations section) | `closed_book` flag and split reporting | Uses the generator model as a proxy for the target's model |
| Tuning to the test | Scores inflate when evaluation internals are known (arXiv 2601.13227) | The question set is private to the user | A user who tunes on one frozen set should confirm on a fresh one |
| Judge favours text from its own model | Self-preference is reported, with disputed size (arXiv 2410.21819, 2601.22548) | Separate judge configuration | Not enforced. By default the judge is also the question writer, a separate risk these papers do not measure |
| Small samples | Default 21 follow-ups per scenario and 30 unanswerable across three subjects | Intervals, minimum cell size, detectable difference printed, counts adjustable | With defaults, per-scenario and per-type results are indicative only; raise the counts to study one in depth |
| Questions are not independent | Same document; language versions of one question; a follow-up and its first turn | Groups are reported separately; a follow-up is resampled with its first turn | Clustering within a document is ignored, so intervals are somewhat too narrow |
| Follow-ups depend on the first answer | The history contains the target's own first-turn answer | Both turns scored; first-turn score shown beside the follow-up | A bad first answer can cause a bad follow-up, so follow-up scores are not independent of first-turn quality |
| Evidence matching | A target with a different text extractor may return text that differs from our quote | Matching tolerates splits and small differences; error rates measured against a reference target that uses a different extractor (criterion 2) | One reference target; a particular real target may do worse, and one that translates or rewrites its contexts cannot be matched |
| Verbose answers | An answer that copies its contexts scores well on recall and faithfulness | Contradiction rate; context count and length reported | No answer-precision metric in v1 |
| Text extraction | Thai and multi-column PDFs can extract poorly | Extraction is checked on the mock PDFs first; empty files stop the job | Garbled but non-empty text is caught only at human review |

## 13. Decisions from the product owner (2026-10-13)

1. **`scenario`** describes how a follow-up question relates to the question before it; both turns are checked. Implemented as FR-12.
2. **Languages**: questions can be asked in any chosen languages over material in any language, with gold answers translated and the number of test languages editable in the app. Implemented as FR-13. This replaces the single `QUESTION_LANG` override of revision 2 and restores translation, now of gold answers as well as questions.
3. **Counts**: no fixed requirement; everything adjustable. Implemented as FR-14, with guidance in FR-2.
4. **Material in the repository is mock data.** End users add their own, edit anything, and use the tool through the app. Implemented as FR-4, FR-11 and criterion 9.

Still open:

- Whether translated variants should require human review when a reviewer who reads the language is available. Currently optional and counted.
- Resolved 2026-10-13: hard negatives are deferred (see non-goals).
- The defaults in FR-14 are starting values, not derived from data.

## 14. Revision history

**Built 2026-10-13.** All five phases are implemented; see architecture section 13 for differences from the design and for what has and has not been verified. No success criterion has yet been measured with a real model.

**Revision 6** — owner deferred hard negatives; extraction check on the mock PDFs recorded in the architecture (slides with little text per page; an occasional lost Thai tone mark), leading to eligible pages, a proportional split of questions across documents, and a tolerant quote check.

**Revision 5** — after the reviewer's pass on revision 3: `same` follow-ups must depend on the conversation; scores reported per group and never pooled, with `compare` on a named group; extra languages added to the always-present source language; history support off until confirmed; freeze keeps the first turn of every follow-up; closed-book check moved to freeze (about 170 fewer calls); judge prompts fixed in phase 2 and labels kept per prompt; reference target translates queries; job lock, progress and stop rules; review as table plus form; scenario labels correctable; cloud-sync warning.

**Revision 4** — after the reviewer's second pass ("ready after minor fixes"): matcher restated as an exact window lookup; quotes lengthened to 100–300 normalised characters so one stray character or chunk boundary cannot cause a miss; Thai mark order normalised; reference target uses a different PDF library and matcher labels are defined; unanswerable check drops on any claimed answer; judge sample rules, three-way key-point labels and a per-type gate (the kappa gate was redundant on a balanced sample); faithfulness limited to answerable questions; text files split into pages; per-check drop counts; Labelling page moved to phase 2.

**Revision 3** — product-owner decisions in section 13: follow-up questions and `scenario`; test languages with translated gold answers; all counts as settings; fully editable review; the app covers the whole workflow; settings moved from `.env` to `settings.json`; language-match metric; phases 4 and 5.

**Revision 2** — after an independent review found the first draft not ready to build from:

| Area | Change |
|---|---|
| Evidence matching | Scored against the top k contexts together, on matching runs of text, with measured error rates |
| Validation | Answerability checked on quotes, not pages; unnecessary evidence removed; unanswerable checked per document with a verified quote |
| Dataset size | Over-generation; criterion lowered to 60 per subject; per-subject page limit replaced by a per-document token check |
| Judge validation | Blind labels, balanced sample, kappa gate, labels tied to judge and prompt version |
| Testing | A reference RAG target added; success criteria rewritten to be measurable |
| Metrics | Errors count as failures; contradiction rate added; hard-negative rate conditioned on gold found; difficulty computed by code |
| Statistics | Proportion intervals fixed; detectable difference printed; one primary metric |
| Dependencies | PDF library changed to a permissive licence |
| Schema | Context `doc_id` and `score` removed as unused |
