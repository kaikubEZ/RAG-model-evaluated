# Architecture — Universal RAG Evaluator

Status: draft for build, revision 5, 2026-10-13. Implements `docs/PRD.md`; requirement numbers (FR-n) refer to it.

## 1. Shape

A Python package with a command-line entry point and a Streamlit app. Both call the same functions and read and write the same plain files. There is no server and no database.

```
material/<subject>/*              settings.json   .env
        │ ingest
        ▼
datasets/<id>/pages.jsonl
        │ generate + validate            (generator model, judge model)
        ▼
datasets/<id>/candidates.jsonl
        │ review (person)  →  translate (generator, judge)  →  freeze
        ▼
datasets/<id>/questions.jsonl + manifest.json      ← frozen, hashed
        │ run                            (target endpoint via adapter)
        ▼
runs/<id>/answers.jsonl
        │ score                          (code + judge model)
        ▼
runs/<id>/scores.jsonl + report.md
        │ compare (two runs, same dataset hash)
        ▼
comparison report
```

Each step reads the previous step's file, so any step can be re-run alone. Re-scoring with a different judge does not call the endpoint again.

## 2. Commands

Every command is also a button in the app (3.14).

| Command | Does | Writes |
|---|---|---|
| `rageval generate [--subject S] [--yes]` | Ingest, generate, validate (FR-1 to FR-3, FR-12). Creates a dataset, or resumes the newest unfrozen one | `pages.jsonl`, `candidates.jsonl` |
| `rageval review` | Opens the app on the Review page (FR-4) | `candidates.jsonl` |
| `rageval translate` | Translates approved questions into the extra test languages (FR-13) | `candidates.jsonl` |
| `rageval freeze` | Runs the closed-book check on the questions being frozen, then freezes the dataset (FR-5) | `questions.jsonl`, `manifest.json` |
| `rageval run --dataset ID [--label TEXT]` | Asks every question (FR-6) | `answers.jsonl`, run `manifest.json` |
| `rageval score --run ID [--refresh-judge]` | Computes metrics and report (FR-7 to FR-9) | `scores.jsonl`, `judge_cache.jsonl`, `report.md` |
| `rageval compare RUN_A RUN_B [--primary METRIC] [--only KEY=VALUE]` | Paired comparison (FR-10), on the headline group or on one named group: `scenario=same`, `scenario=similar`, `scenario=diff_topic`, or `direction=<question language>-<document language>` such as `direction=en-th` | comparison report |
| `rageval app` | Opens the app (FR-11) | settings, labels |

- Dataset and run ids are the UTC start time, `YYYYMMDD-HHMMSS`, with the label appended when given.
- Each long command is a job with a lock, progress file and stop signal (3.14).

## 3. Components

### 3.1 Settings (`settings.py`)

- `settings.json` in the project root holds everything in PRD FR-14. Missing file or missing keys mean defaults, which live in one `pydantic` model.

```json
{
  "questions_per_subject": {"fact": 12, "definition": 12, "comparison": 12, "multi_hop": 12, "exception": 12, "unanswerable": 10},
  "follow_ups_per_subject": {"same": 7, "similar": 7, "diff_topic": 7},
  "overgenerate": 1.3,
  "extra_langs": [],
  "generation_guidance": "",
  "top_k": 5,
  "rag_adapter": "http_json",
  "supports_history": false,
  "gen_max_input_tokens": 150000
}
```

- `extra_langs` holds ISO 639-1 codes. Every question always exists in its document's own language; each code here adds a translated variant of every question not already in that language.
- `supports_history` is off until the user confirms it on the Setup page, because an endpoint that silently ignores `history` would score badly on follow-ups with no warning. While it is off, follow-ups are generated but not asked, and every command says so.
- Secrets stay in `.env`. The app's Setup page writes both files.
- The settings in force are copied into every dataset and run manifest.

### 3.2 Ingestion (`ingest.py`)

- PDF pages through `pypdf` (BSD licence); `.txt` and `.md` as pseudo-pages (below).
- Output row: `{subject, doc, page, text}`.
- A file yielding no text stops the job with the file name.
- **Token pre-check**: tokens are estimated as one per character outside the Latin script plus one per three Latin-script characters. A document whose estimate exceeds `gen_max_input_tokens` stops the job. The judge receives whole documents too (unanswerable check), so the setting must be no larger than the smaller of the two models' inputs.
- **Long text files**: a `.txt` or `.md` file is cut into pseudo-pages of about 3,000 characters at paragraph breaks, so that page-based rules (`multi_hop`, page spread, scenarios) work for it.
- **First task of phase 1**: extract the nine mock PDFs and have a person read one page of each, the Thai ones especially. If `pypdf` output is unusable, switch to `pypdfium2` (also permissively licensed) and record the decision here.

**Normalisation**, used everywhere text is compared:

1. Unicode NFKC.
2. Lowercase.
3. Remove all whitespace. PDF extraction inserts erratic spaces and line breaks, especially in Thai.
4. Remove format characters (Unicode category Cf, which covers zero-width space and soft hyphen).
5. Remove punctuation (categories starting with P). This also joins words hyphenated at line breaks.
6. Within each run of consecutive Thai combining marks (U+0E31, U+0E34 to U+0E3A, U+0E47 to U+0E4E), sort by code point. Extractors emit an upper vowel and a tone mark in either order, and NFKC does not make the two orders equal.

Other scripts with stacked marks may need the same treatment; none is handled in v1.

### 3.3 Generation (`generate.py`)

**Single questions**

- **Counts**: per subject and type, candidates = target × `overgenerate`, rounded up. Each type's count is split evenly across the subject's documents; any remainder goes one each to documents in file-name order. A count of zero skips the type.
- **One call per document and type.** The prompt contains the whole document with page markers, the user's guidance text if any, and asks for that document's share of questions.
- **Page spread**: the document's pages are divided into as many equal ranges as questions requested, and question i must take its first evidence quote from range i. Code checks this.
- `multi_hop` questions cite quotes from at least two pages of the same document.
- `unanswerable` prompts ask for plausible in-domain questions the document does not answer.
- Questions are written in the document's language. The first call per document also returns its ISO 639-1 code, stored as `source_lang`.

**Follow-ups** (FR-12), generated after single questions have been validated

- **One call per document.** The prompt contains the document, the validated single questions from that document, and the question texts from the subject's other documents.
- It asks for that document's share of each scenario. Every follow-up names its first-turn question (`previous_id`) and gives both `question` and `resolved_question`.
  - `same`: `question` must lean on the conversation ("it", "that rule", "and after that?") so that it cannot be understood alone; `resolved_question` spells it out.
  - `similar` and `diff_topic`: `question` is self-contained and `resolved_question` is identical to it.
- The generator chooses any answerable question type for a follow-up. Follow-ups count toward scenario targets, not type targets.
- The follow-up's evidence always comes from the document in the prompt. For `same` and `similar` the first turn is one of that document's questions; for `diff_topic` it is one from another document.
- A subject with a single document cannot have `diff_topic` follow-ups; the shortfall is reported.
- Only answerable first turns are used, and follow-ups are themselves answerable.

**Output parsing**: every prompt asks for JSON only. Code takes the first JSON value in the reply and validates it with `pydantic` models of the PRD schema. No provider-specific JSON mode is relied on. A reply that fails is retried once, then the batch is reported as failed.

The model does not set `difficulty`; see 3.4.

Rubric given to the generator:

| Term | Definition |
|---|---|
| Key point | One statement that can be judged true or false on its own. "TCP is connection-oriented" is one; "TCP is connection-oriented and reliable" is two |
| Evidence quote | One contiguous verbatim span, 100 to 300 characters after normalisation (roughly one to three sentences), that states the key point, not merely one that appears near it |
| Hard negative | A verbatim span on the same topic, sharing at least one key term with the question, that does not answer it. For "minimum GPA to keep the merit scholarship", the GPA rule for a different scholarship |
| Unanswerable | A question a reader of this subject could reasonably ask, on a topic the document discusses, whose answer the document does not give |
| Resolved question | The follow-up rewritten so that it can be understood with no conversation before it, asking exactly the same thing |

### 3.4 Validation (`validate.py`)

In this order, cheapest first (FR-3):

| Check | How | Fails when |
|---|---|---|
| Quote | Code: normalised quote is a substring of the normalised cited page; normalised length 100 to 300 | Any evidence or hard-negative quote fails |
| Structure | Code: type rules, page-spread rule, scenario rules | e.g. `multi_hop` with evidence on one page; `same` follow-up sharing no evidence page with its first turn |
| Duplicate | Code: character-trigram Jaccard similarity ≥ 0.8 between `resolved_question` and an earlier one in the subject | Near-duplicate |
| Answerable + standalone | Judge, one call. Input: `question`, `resolved_question`, the gold key points, and the evidence quotes with no other document text. The judge is told to answer from the quotes alone, not from its own knowledge. Output: `{question_standalone, resolved_standalone, answer, key_points_covered: [bool], quotes_used: [bool]}` | `resolved_question` not standalone; a key point not covered; a `same` follow-up whose `question` is standalone (it would not test the conversation); any other question whose `question` is not standalone |
| Unanswerable | Judge, one call per document in the subject. Output: `{standalone, answer_found, quote, closest_passage}` | `answer_found` is true for any document |

Scenario rules, on evidence pages:

| Scenario | Rule |
|---|---|
| `same` | Same document as the first turn, at least one evidence page in common |
| `similar` | Same document, no evidence page in common |
| `diff_topic` | Different document |

These rules are a repeatable stand-in for "topic": they describe where the evidence sits, not what it is about. A topic that runs over several pages can make a true same-topic follow-up come out `similar`, and two documents on one topic come out `diff_topic`. The generator must obey the rules; the reviewer may then change a `scenario` label by hand, and the manifest counts such overrides.

After the answerable check:

- Evidence items marked unused are removed. If that leaves a `multi_hop` question with one page, or breaks a scenario rule, it fails.
- **Difficulty is computed**: `hard` if evidence spans two or more pages; otherwise `easy` if at least 50% of the normalised character trigrams of `resolved_question` occur in its evidence quotes, else `medium`.

For the unanswerable check, the judge's quote is tested as a substring of the document only (at least 20 normalised characters; the 100 to 300 rule is for evidence). If the judge says an answer exists, the candidate is dropped whether or not its quote can be verified, because keeping a question that may be answerable would penalise a correct system. `closest_passage` is stored for the reviewer.

Every dropped candidate is counted by check, type and scenario; the counts go into the manifest so that a high drop rate can be traced to the check that caused it.

**Closed-book flag.** Run by `freeze`, only on the answerable questions being frozen, because it merely sets a flag and would otherwise be most of the validation cost. The generator model answers `resolved_question` with no context; the answer check (3.10, Call A) scores it; `closed_book` is true when answer recall is 1.0. Translated variants inherit the flag from their source question and are not checked again.

Refill: missing candidates are regenerated for up to two more rounds. The final shortfall per subject, type and scenario is printed and stored in the manifest.

### 3.5 Translation (`translate.py`)

Runs on approved source-language questions (FR-13).

- For each code in `extra_langs`, and each approved question whose `source_lang` differs from it: translate `question`, `resolved_question` and `gold_ans`. Evidence and hard negative are copied unchanged.
- A question whose source language equals the requested code needs no translation; it already is that variant.
- Translation goes to the generator model in batches of 10 questions.
- **Meaning check**, judge, batches of 10: given source and translation, return for each `{question_same: bool, key_points_same: [bool]}`. Any false value sends that question for one more translation; a second failure drops the variant and reports it.
- A variant gets `id` = `base_id` + `.` + language code, `status: approved`, `translation_reviewed: false`. A follow-up variant's `previous_id` points at its first turn in the same language.
- Editing an approved source question afterwards deletes its variants; `translate` recreates what is missing. `freeze` refuses while variants are missing, unless the failure was reported as dropped.

### 3.6 Review and freeze (`app.py` Review page, `freeze.py`)

- **Table plus form.** A read-only table of `candidates.jsonl` with filters by subject, type, scenario, language and status, and a bulk-approve button for the filtered rows. Selecting a row opens a form under the table:
  - text areas for `question` and `resolved_question`;
  - one text area per gold key point, with add and remove;
  - one group of fields per evidence item (document, page, quote), with the text of the cited page shown beside it so a quote can be copied from it;
  - the same for the hard negative; selectors for type, scenario and status.
- "New question" opens the same form empty.
- Shown for context: closest passage for unanswerable questions; the first-turn question for follow-ups; the source-language question for translated variants.
- The page is read-only while a job holds the dataset's lock (3.14).
- Editing lists inside grid cells was rejected: a quote can contain any separator character, and edits on a filtered grid are easy to save to the wrong row.
- **Spike, first task of phase 1** alongside the extraction check: half a day to confirm that row selection plus a form works acceptably in Streamlit for a few hundred rows.
- **On save**, quote, structure and duplicate checks run on the row, and difficulty is recomputed. A failing row keeps `candidate` status and shows the reason. A `scenario` that disagrees with the page rule is allowed with a warning and counted as an override. The judge checks are not re-run on hand edits; the reviewer's approval stands in for them.
- Marking a translated variant reviewed sets `translation_reviewed`.
- **Freeze** selects approved source-language rows per subject:
  1. follow-ups in candidate order, up to the target per scenario;
  2. the first-turn questions of those follow-ups, even where that exceeds a type's target, so that no approved follow-up is lost for want of its first turn;
  3. other single questions in candidate order, until each type reaches its target.
  It adds the approved variants of the selected rows, runs the closed-book check (3.4) as a resumable job, and only when that has finished writes `questions.jsonl` and a manifest, so an interrupted freeze leaves no half-frozen dataset:

```json
{
  "dataset_id": "20261013-091500",
  "sha256": "hash of questions.jsonl",
  "settings": {},
  "generator_model": "...",
  "prompt_versions": {"generate": "3f9a1c", "translate": "b07e44"},
  "counts": {"law": {"fact": 12, "same": 7}},
  "shortfall": {"law": {"multi_hop": 2}},
  "dropped": {"answerable": {"multi_hop": 9}, "quote": {"fact": 1}},
  "languages": ["th", "en"],
  "bulk_approved": 0,
  "scenario_overrides": 0,
  "translations": 91,
  "translations_reviewed": 0,
  "created": "2026-10-13"
}
```

- A frozen dataset directory is never written again. **Duplicate as draft** copies its questions into a new dataset's `candidates.jsonl`.

### 3.7 Adapters (`adapters.py`)

An adapter is one function:

```python
def ask(question: str, history: list[dict]) -> dict:
    """history: [{"question": str, "answer": str}], empty for a single question.
    Return {"answer": str, "contexts": [{"text": str}] | None}"""
```

- `http_json` (default) implements the contract in PRD FR-6.
- Other endpoint shapes are handled by adding a function to `adapters.py` and naming it in `rag_adapter`. No plugin system. An endpoint that keeps conversation state on its own server instead of accepting history is not supported in v1.
- `contexts` is `None` when the endpoint does not return them, which disables retrieval metrics and faithfulness for that run.
- With `supports_history` false (the default until confirmed), follow-ups are not asked and are reported as not measured.

### 3.8 Reference target and run

**Reference target** (`tests/reference_rag.py`): a deliberately plain RAG system, used to test and calibrate the tool. It is test code, not part of the product.

- Extracts PDF text with `pypdfium2`, a different library from the tool's, so that its contexts differ from the tool's quotes the way a real target's would. This is a test-only dependency.
- Splits raw page text into 700-character pieces with 100 overlap, so chunk boundaries fall inside quotes.
- Retrieval: BM25 over character trigrams, written in a few dozen lines with no extra dependency; returns the top `top_k`. For a follow-up, the first-turn question is prepended to the query.
- Query translation: one generator call translates the question into each language found in the corpus, and retrieval uses the original and the translations together. Without this a lexical retriever finds nothing across languages, and there would be no cross-language answers to validate the judge on.
- Answer: one generator-model call over the retrieved pieces and the history.
- Exposed as adapter `reference`, which needs no `RAG_URL` or `RAG_API_KEY`. The test-only environment variable `REFERENCE_MODE` selects `normal`, `random` (pieces drawn at random from the whole corpus), `answer_only` (no contexts returned), `no_history` (ignores history) or `no_translation` (skips query translation).

Limit: it is one simple design. Its scores say nothing about how good cross-language retrieval can be; it exists to exercise the tool.

**Run** (`run.py`)

- Asks questions sequentially: all single questions first, then follow-ups, appending to `answers.jsonl`: `{id, answer, contexts, error, latency_ms}`.
- A follow-up is sent with history built from its first-turn question and the answer stored for it in this run. If the first turn errored, the follow-up is recorded as an error without being asked.
- Resume: questions with a successful row are skipped; questions whose latest row has an error are asked again. The latest row per id is the one scored.
- Run manifest: dataset id and hash, settings, `RAG_URL`, label, matcher constants, seed, start time.

### 3.9 Evidence matching (`metrics.py`)

All on normalised text, always in the document's language. Constants: window 20 characters, coverage 0.8.

For an evidence quote and the first k contexts:

1. Collect every 20-character substring of the k contexts into one set.
2. Slide a 20-character window along the quote. Wherever the window's text is in the set, mark the 20 quote characters it covers.
3. The quote is **found in the top k** when at least 80% of its characters are marked.

```python
def coverage(quote: str, contexts: list[str], w: int = 20) -> float:
    grams = {c[i:i + w] for c in contexts for i in range(len(c) - w + 1)}
    marked = [False] * len(quote)
    for i in range(len(quote) - w + 1):
        if quote[i:i + w] in grams:
            marked[i:i + w] = [True] * w
    return sum(marked) / len(quote)
```

`found_at` for a quote is the smallest k at which it is found, or none. It is computed over all contexts the target returns, so it can exceed `top_k`; Recall@k counts only `found_at` ≤ `top_k`.

Why this rule:

- Marking across contexts together means a quote cut by a chunk boundary is still found when both halves are retrieved.
- Requiring windows of 20 characters, rather than counting shared trigrams, keeps unrelated text on the same topic from adding up to a match.
- Characters, not words, so the rule works for Thai.
- **Why quotes are at least 100 characters**: one differing character, or one chunk boundary with no overlap, can leave up to 20 quote characters unmarked. With a quote of at least five windows that is at most 20%, so the quote is still found. Shorter quotes would fail on a single stray character.
- A plain window lookup is used, not `difflib.SequenceMatcher`: its matching blocks are one ordered alignment and miss stretches that appear in a different order across contexts.

A target that translates its contexts before returning them cannot be matched; its retrieval metrics will read near zero and the report's context sample will show why.

**Calibration** (phase 2): quote-and-contexts pairs from the reference target are hand-labelled on the Labelling page, at least 50 "retrieved" and at least 50 "not retrieved", the latter including contexts from the right document that miss the quote. The label is "retrieved" when a reader can see the whole quoted statement in the top-k contexts taken together, ignoring line breaks and stray characters; half of a quote is "not retrieved". The miss rate and false-accept rate are measured against PRD criterion 2 and written into this document and every report. If the constants change, calibration is repeated. Until calibration has been done for the constants in use, retrieval metrics carry a "matcher not calibrated" warning.

### 3.10 Metrics (`metrics.py`, `judge.py`)

Code metrics, per question:

| Metric | Computation |
|---|---|
| Recall@k | evidence quotes with `found_at` ≤ k ÷ evidence quotes |
| Reciprocal rank | 1 ÷ the smallest `found_at` among the evidence quotes, else 0 |
| Hard negative above gold | Only when at least one evidence quote is found in the returned contexts: 1 if the hard negative's `found_at` is strictly smaller than the smallest gold `found_at`, else 0 (a tie is 0). Otherwise `null` |
| Context count, context length | Number of contexts returned and their total characters |

Judge metrics use two calls per question, at temperature 0.

**Call A — answer check.** Input: `resolved_question`, gold key points, the answer, the question's language. Output:

```json
{"refused": false, "answer_lang": "en", "key_points": [{"point": "...", "verdict": "supported"}]}
```

- `verdict` is `supported`, `contradicted` or `absent`.
- Answer recall = supported ÷ key points. A refusal scores 0.
- Contradiction = 1 if any key point is `contradicted`.
- Language match = 1 if `answer_lang` equals the question's `lang`.
- For unanswerable questions only `refused` and `answer_lang` are used.

**Call B — faithfulness.** Answerable questions only. Skipped when there are no contexts or the answer is a refusal. Input: the answer and the returned contexts; for a follow-up, also the contexts returned for its first turn, so that a fact correctly carried over from the first turn is not counted as unsupported. Output:

```json
{"claims": [{"claim": "...", "supported": true}]}
```

- Faithfulness = supported claims ÷ claims; `null` when the answer has no factual claims.
- Answer and contexts may be in different languages; the rubric says language does not matter for support.

**Judge rubric.** These definitions go into the prompts word for word, with the examples.

| Decision | Rule | Examples |
|---|---|---|
| Key point `supported` | The answer states the same fact, in any wording or language. Extra correct detail does not matter | Key point "UDP does not guarantee delivery". Answer "UDP may drop packets without notice" → supported. Answer "UDP is a transport protocol" → absent |
| Key point `contradicted` | The answer states something that cannot be true if the key point is true | Key point "UDP does not guarantee delivery". Answer "UDP retransmits lost packets" → contradicted |
| Key point `absent` | Neither of the above, including vague answers that do not commit | Answer "UDP has trade-offs" → absent |
| `refused` | The answer says it cannot answer from the available information and gives no answer to the question asked. An answer that hedges but still answers is not a refusal | "The documents do not cover this." → refused. "I'm not certain, but it is probably 3.0." → not refused |
| `answer_lang` | The language most of the answer's own sentences are written in, ignoring quoted source text and technical terms | An English answer quoting one Thai sentence → `en` |
| Claim | One factual statement the answer makes about the subject. Greetings, hedges and statements about the documents themselves are not claims | "TCP uses a three-way handshake, which adds latency" → two claims |
| Claim `supported` | The contexts state it, in any language, or it follows directly from one context. True facts that are not in the contexts are unsupported | Context mentions the handshake only → first claim supported, second unsupported |
| Standalone | A person who has not seen the document or any earlier turn knows what is being asked | "What does the author conclude in section 3?" → not standalone |
| Translation same | The translated question would be answered by exactly the same facts; a translated key point states the same fact, with names and numbers intact | "at least 2.75" rendered as "more than 2.75" → not the same |

Prompts are text files in `rageval/prompts/`, one per call. A prompt's version is the first six characters of its content hash, written into every manifest and score row, so an edited prompt is recorded automatically. **The Call A and Call B prompts are written in their final form in phase 2**, including `answer_lang` and the any-language wording, because changing a judge prompt voids the labels collected for it.

**Judge cache.** Every judge reply is stored in `judge_cache.jsonl`, keyed by a hash of judge model, prompt version and input. Re-scoring reads the cache, which is what makes scores reproducible. `--refresh-judge` ignores the cache and reports the share of verdicts that changed.

### 3.11 Statistics (`stats.py`)

- **Groups.** Scores are never pooled across kinds of question:
  - the **headline** is single questions asked in their document's own language;
  - follow-ups are reported separately, by scenario;
  - each translated direction is reported separately, named by question language and document language (for example "en question, th documents").
  Pooling them would give an average that moves with how many languages and follow-ups the user chose.
- **Aggregate** = mean of per-question scores over the questions in a group that the metric applies to. Questions with an endpoint error score 0 in the headline; a second set of numbers excludes them. Questions with a judge error are excluded from judged metrics and counted.
- **Intervals, 95%**:
  - Metrics that are yes or no per question (refusal accuracy, false refusal rate, contradiction rate, hard-negative rate, language match): Wilson interval. A bootstrap would give a zero-width interval when every question agrees. Within a group each question appears once, so Wilson applies.
  - Other metrics: percentile bootstrap, 2,000 resamples, fixed seed.
  - **Resampling unit**: a question's `base_id`; for a follow-up, the `base_id` of its first turn. This keeps a follow-up with its first turn, and language versions together, wherever a comparison spans them.
- **Comparison**: on the headline group by default, or on the group named with `--only`. Endpoint errors count as 0. For questions present in both runs, bootstrap the mean of per-question differences (B minus A) over resampling units. The verdict is "no detectable difference" when the interval includes zero, endpoints included.
- **Detectable difference**: 2.8 × the standard deviation of the per-unit mean differences ÷ √(number of units) (80% power at the 5% level), printed for every metric in every comparison. When every difference is zero it is printed as "not estimable".
- Breakdown cells with fewer than 20 questions print "n too small". The minimum applies to breakdown cells only; a group (the headline, a scenario, a translated direction) is always shown, with its n and interval.
- Not modelled: questions from the same document are correlated, and judge noise is not propagated. Both make the intervals somewhat too narrow; the report says so.

### 3.12 Labelling (`app.py`, Labelling page)

Two labelling tasks share one page.

**Judge decisions** (FR-8):

- Sampled from scored runs that used the current judge model and prompt version.
- Target 150 per question language, 50 per decision type, balanced across the judge's verdict values as far as available:
  - key point: `supported`, `contradicted`, `absent`;
  - claim: supported or not;
  - refusal: refused or not.
- The page shows the inputs the judge saw and asks the person for the same verdict the judge had to give (three values for a key point). The judge's verdict is hidden.
- **Filling a short stratum**: one run rarely has enough of the rarer verdicts (refusals, contradictions). Decisions may be drawn from any number of runs, including reference-target runs in `random` mode, which produce refusals and unsupported claims. The same question-and-answer pair is never sampled twice. A decision type that still has fewer than 30 labels is reported as not validated, and any verdict value with fewer than 10 labels is flagged beside the agreement figure, since agreement is easy to reach when one value dominates.
- Where runs contain answers and contexts in different languages, half of the claim labels are drawn from those.
- Stored in `labels/judge/<judge_model>__<call>__<prompt_version>.jsonl`, one file per judge call: key-point and refusal labels belong to Call A's prompt, claim labels to Call B's. Validation carries across runs, and editing one prompt voids only its own labels. Model names and run labels are made file-safe by replacing every character outside letters, digits, dot, dash and underscore with an underscore.
- The report shows, per language and per decision type, the number of labels and the agreement with a Wilson interval, plus Cohen's kappa, and applies the gate in PRD FR-8. The figure is labelled "agreement on a balanced sample": it is lower than agreement on a typical run, where most verdicts are the easy majority kind.

**Matcher pairs** (3.9): quote, top-k contexts, label "retrieved" or "not retrieved". Stored in `labels/matcher.jsonl`.

### 3.13 Report (`report.py`)

`report.md` contains, in order: what was measured and what was not; headline metrics with intervals and n, with and without errored questions; judge and matcher validation status; for each translated direction, "translations not reviewed by a person: n of m"; context count and length; headline breakdowns by subject, type and difficulty; follow-ups by scenario, beside the scores of their first turns, with a note that retrieval metrics on `same` follow-ups assume the target retrieves again and do not credit a target that answers from the conversation; each translated direction; scores with and without `closed_book` questions; error list; settings and constants; the statement that scores describe these documents and this question set.

### 3.14 App (`app.py`)

One Streamlit app with the eight pages of PRD FR-11.

- **Setup**: form over `.env` and the endpoint settings. "Test connection" sends one small request to the generator, the judge and the target, and shows the error text on failure. It asks whether the endpoint accepts conversation history, which sets `supports_history`.
- **Material**: lists `material/<subject>/`; creates subject folders; `st.file_uploader` saves into them; delete with confirmation. Shows page count and the token pre-check per file.
- **Generate**: form over `settings.json` — a number per question type and per scenario, the over-generation factor, guidance text, and extra test languages as a multi-select of common languages plus a box to type any other code. Shows the resulting question count and call estimate before starting, and for each extra language the reminder that its judged scores carry a warning until 150 decisions have been labelled by someone who reads it.
- **Review**: 3.6, with buttons for translate, freeze and duplicate as draft.
- **Run and score**, **Results**, **Compare**, **Labelling**: as named. Results and Compare render the numbers `report.py` and `stats.py` compute; they do not recompute metrics.

**Long jobs.** A button starts the matching `rageval` command as a separate, detached process (same Python interpreter) and the page polls its progress. The job therefore survives a closed tab or a page rerun.

- **Lock**: `job.lock` in the dataset or run directory holds the process id and a heartbeat time, refreshed every 10 seconds by a background thread so that a long model call does not let it lapse. One job at a time per directory. A lock whose process is gone, or whose heartbeat is older than 60 seconds, is stale and is taken over, so an interrupted job is continued by pressing the button again.
- **Progress**: `progress.json` (step, done, total, last message) is written to a temporary file and moved into place, retrying briefly if the move is refused, because on Windows a file being read cannot be replaced.
- **Stop**: a Stop button creates `job.stop`; the job checks for it between model calls, finishes the call in hand, and exits. Everything done so far is kept, and starting again resumes.
- While a job holds a dataset's lock, the Review page for that dataset is read-only, since both write `candidates.jsonl`.

**Privacy confirmation.** The Generate page shows the host that document text will be sent to and requires a tick before the first job; the command line asks the same unless `--yes` is given.

## 4. Stored data

Question rows follow PRD section 7. One row of `scores.jsonl`:

```json
{
  "id": "law-0007.en",
  "base_id": "law-0007",
  "error": null,
  "recall_at_k": 0.5,
  "reciprocal_rank": 0.33,
  "hard_negative_above_gold": 1,
  "context_count": 5,
  "context_chars": 3400,
  "refused": false,
  "answer_lang": "en",
  "language_match": 1,
  "answer_recall": 0.5,
  "contradiction": 0,
  "key_points": [{"point": "...", "verdict": "supported"}],
  "faithfulness": 0.8,
  "claims": [{"claim": "...", "supported": true}],
  "judge_model": "...",
  "judge_prompt_version": "9c2d10"
}
```

A metric that does not apply or was not measured is `null`, never 0. All rows read from or written to disk are validated with `pydantic` models.

## 5. Model access

- Generator and judge are called through the OpenAI-compatible chat API (`openai` SDK with `base_url`), which covers OpenAI, OpenRouter and most local servers with one client.
- A provider without a compatible API needs a gateway in front of it; native SDKs for other providers are not built in.
- Calls are sequential. Where the provider supports prompt caching, the document is placed first in the prompt so repeated whole-document calls reuse it.

## 6. Error handling

Every message names what failed and what the user should do next.

| Situation | Behaviour |
|---|---|
| Missing required `.env` value | Stop at start-up, naming the variable; the app sends the user to Setup |
| Invalid `settings.json` | Stop, naming the key and the allowed values |
| Generator or judge: rate limit, 5xx, timeout | SDK retries with backoff; after that the job stops, and starting it again resumes |
| Generator or judge: reply fails validation | Retry once; then drop the candidate, or mark the question `judge_error` |
| Endpoint: 401, 403, 404 | Stop the run immediately; this is a configuration problem |
| Endpoint: 429, 5xx, timeout (60 s) | Three attempts with backoff; then record `error` for the question and continue |
| Endpoint: 10 consecutive errors | Stop the run |
| Endpoint reply missing `answer` | Record `error` for the question |
| `compare` on different dataset hashes | Refuse |
| A second job on the same dataset or run | Refuse, naming the running job, unless its lock is stale |

## 7. Security and privacy

- Secrets only in `.env`, which is gitignored; `.env.example` is committed with placeholders. The Setup page shows key fields masked.
- `material/`, `datasets/`, `runs/`, `labels/` and `settings.json` are gitignored.
- The app is meant to run on the user's own machine. It has no login, so it must not be exposed on a network; it binds to localhost.
- **Synced folders**: `.gitignore` does not stop OneDrive, Dropbox or similar from uploading `material/`, `datasets/` and `runs/`. For material under usage terms the project should live outside a synced folder; the Material page says so when it detects one. Sync clients can also briefly lock files the tool is writing.
- Uploaded files are saved only under `material/<subject>/`, with the file name reduced to its base name and the extension checked against the supported list.
- Answers and contexts from the target are placed in judge prompts inside delimiters and never executed. A target could still try to steer the judge through its answer text; blind judge validation is the check on that.

## 8. Layout

```
rageval/
├── cli.py
├── settings.py
├── ingest.py
├── generate.py
├── validate.py
├── translate.py
├── freeze.py
├── adapters.py
├── run.py
├── judge.py
├── metrics.py
├── stats.py
├── report.py
├── schema.py         ← pydantic models
├── llm.py            ← one OpenAI-compatible client
├── app.py            ← Streamlit
└── prompts/
tests/
└── reference_rag.py
docs/
research/
.env                  ← gitignored
settings.json         ← gitignored
material/             ← gitignored
datasets/             ← gitignored
runs/                 ← gitignored
labels/               ← gitignored
```

Dependencies, all permissively licensed: `openai`, `pypdf`, `pydantic`, `python-dotenv`, `numpy`, `streamlit`; `pytest` and `pypdfium2` for tests. Licences are to be confirmed from each package's metadata when the dependency file is written.

## 9. Testing

| What | How |
|---|---|
| Settings | Defaults with no file; a zero count skips a type; an invalid value is rejected with the key named |
| Normalisation | Zero-width space, soft hyphen, line-break hyphenation, Thai text with erratic spaces, Thai upper vowel and tone mark in both orders |
| Quote check, evidence matching | English and Thai, at the minimum quote length of 100: quote inside one context; quote split across two contexts at every position; one differing character at every position; halves found in reverse order across contexts; same-topic text that must not match |
| Structure and scenario rules | One passing and one failing case per type and per scenario; a `same` follow-up with a standalone `question` is rejected |
| Jobs | A stale lock is taken over and a live one is not; the stop file ends a job between calls and a restart resumes; progress is replaced while another process holds it open |
| Recall, reciprocal rank, hard-negative logic | Hand-computed expected values, including "nothing retrieved" giving `null` for hard negative |
| Intervals and comparison | All-equal yes/no scores give a non-zero-width Wilson interval; identical runs give a difference of exactly zero and "no detectable difference"; a clearly shifted run excludes zero; a follow-up is resampled with its first turn; groups are never pooled |
| Translation bookkeeping | Variant ids and `previous_id`; editing a source question removes its variants; freeze refuses while variants are missing |
| Freeze | Hash is stable; a frozen directory is not rewritten; freeze stops at the targets; the first turn of every selected follow-up is included |
| Run and resume | Local fake HTTP server from the standard library for `http_json`; history is sent for follow-ups; kill mid-run and resume; an errored question is retried |
| Review save | An edited quote that is not on the page is refused; a hand-added row is checked |
| Upload | A file name with path parts is reduced to its base name; an unsupported extension is refused |
| End to end | Reference target in each mode, with generator and judge replaced by recorded replies |
| Judge prompts | Not unit tested; covered by blind judge validation |
| PRD success criteria | Run by hand against the mock material with real models, results recorded in `docs/` |

## 10. Build order

Follows the PRD milestones.

1. **Dataset**: extraction check, review-form spike, `settings`, `ingest`, `schema`, `llm`, `generate` (single questions), `validate`, Review page, `freeze`.
2. **Run and score**: `adapters`, reference target, `run`, `metrics`, `judge` with final prompts, `stats`, `report`, Labelling page (matcher pairs), matcher calibration.
3. **Trust**: judge labelling on the same page, judge agreement, `compare`.
4. **Coverage**: follow-up generation and scenario rules, history in `run`, `translate`, query translation in the reference target, reporting of follow-ups and translated directions.
5. **App**: Setup, Material, Generate, Run and score, Results, Compare pages; job launching and progress.

The schema, the adapter signature, the judge prompts and the resampling unit are built in their final form in phases 1 and 2, so phase 4 adds rows and fields, not changes, and does not void judge labels. The end of phase 3 is the first release worth using: a tool proven on single questions in the documents' own language.

## 11. Decisions

| Decision | Reason |
|---|---|
| Evidence is page plus verbatim quote, not chunk ids | The target's chunking is unknown; a quote can be matched against any system's contexts |
| Matching on shared 20-character windows across the top k together | Independent of the target's chunk boundaries; fewer accidental matches than counting trigrams |
| No chunking in the tool | Whole documents fit in the generator's input at v1 sizes, and chunk settings would only affect our side |
| Plain files, no database | A few hundred rows per dataset; files are diffable and resumable |
| Rerank tested by comparing two runs | The tool cannot see inside the target, and a paired comparison on a frozen set is the statistically sound form |
| Write in the source language, review, then translate question and gold answers | The reviewer checks questions against evidence in one language; every language asks the same questions, so languages can be compared pairwise |
| Evidence never translated | The documents stay in their language; cross-language retrieval is the thing under test |
| Follow-ups carry the target's own first answer as history | That is what a real conversation contains |
| Scenario defined by evidence pages, reviewer may override | Code can check it and it is repeatable; a person corrects the cases where position and topic disagree |
| Scores reported per group, never pooled | A pooled average would change with the number of languages and follow-ups chosen |
| Closed-book check at freeze | It only sets a flag, so it is run on the questions kept, not on every candidate |
| Review as table plus form | Lists and long quotes do not edit safely inside grid cells |
| Difficulty computed, not generated | A rule code can check is consistent across models |
| Settings in one JSON file, secrets in `.env` | The app can read and write both with the standard library; secrets stay out of manifests |
| App starts commands as separate processes | Streamlit reruns the page script on every interaction; a job inside it would be interrupted |
| Judge replies cached | Needed for reproducible scores and cheap re-reporting |
| Judge labels tied to model and prompt version | Validation is a property of the judge, not of one run |
| Adapter is a function, not a plugin system | One file is enough until there are many adapters |
| `pypdf` rather than PyMuPDF | PyMuPDF is AGPL, which would bind anyone distributing this MIT tool |
| Streamlit | The review table, forms, upload and result pages are built-in widgets; see `research/gui-templates.md` |

## 12. Known limits

- Whole-document prompting caps document size and makes dataset builds slow and input-heavy (PRD section 9). Larger documents need a windowed generator, which is not designed here.
- Matcher constants, the Thai mark handling, the duplicate threshold and the easy/medium overlap threshold are starting values; only the matcher is calibrated in v1.
- The reference target is one extractor and one chunking; matcher error on a particular real target may differ. Being lexical, it says nothing about cross-language retrieval quality.
- Whether an evidence quote is necessary rests on the judge's own report of which quotes it used.
- Hand-edited questions skip the judge checks.
- Scenario labels describe where evidence sits, not topic; they are only as good as the page rule plus the reviewer's corrections.
- Retrieval metrics on `same` follow-ups do not credit a target that correctly answers from the conversation without retrieving again.
- Translations a person has not reviewed rest on the judge's meaning check alone.
- The closed-book check uses the generator model, not the target's model.
- No questions spanning two documents; no conversations beyond two turns.
- Sequential calls throughout; one job at a time per dataset or run.
- Garbled PDF text that is not empty passes ingestion and is caught only at human review.
- Intervals ignore clustering within documents and judge noise.
