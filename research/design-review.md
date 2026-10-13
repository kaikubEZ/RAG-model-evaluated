# Design review — Universal RAG Evaluator

Review of `docs/PRD.md`, `docs/architecture.md`, `flaw.md` and `.env.example` on 2026-10-13. Paper references are in [papers.md](papers.md).

## Blocking design flaws

1. **No scoring stage.** The pipeline goes from answer generation to rerank check to output (`architecture.md` Stages 6–8). Metrics are named (`PRD.md` 4.5) but never defined, and `.env` has no judge model.
2. **Gold chunks cannot be matched to a black-box endpoint.** The tool chunks material at 500 characters and stores its own chunk IDs in `gold_chunk`. The target RAG chunks differently, so the IDs are meaningless to it.
3. **Rerank check is impossible as designed.** Stage 7 receives only a question and answer, but claims to compare retrieval before and after rerank. No request/response contract is defined, and "any endpoint" conflicts with needing ranked contexts.
4. **Contamination check is misconceived.** It flags questions similar to the source material, which they should be. Real contamination is answering from memory; a no-context baseline detects it (MIRAGE).
5. **Translation loses the gold answers.** Only `question` and `lang` are translated, so target-language answers are scored against English gold answers. Translated sets also carry translationese (MEMERAG).
6. **Multi-hop cannot come from single 500-character chunks.** It needs chunk grouping or a knowledge graph (RAGAS). Unanswerable questions are absent (Know Your RAG, UAEval4RAG).
7. **No judge validation, no statistics.** ARES calibrates with 150–300 human labels and reports confidence intervals. TREC 2024 found automatic scoring weaker per question, which is where this design emits "rerank improved/degraded". 210 questions over 5 types × 3 difficulties × 3 scenarios leaves a few items per cell.
8. **No automatic filtering of generated questions** before human review (answerable from gold evidence, standalone, not duplicated).

## Document quality problems

- Flaw 3's mitigations are HTTP retry advice, unrelated to the flaw.
- `material/` has three subjects (`comEn`, `law`, `med`), so "210, 70 per subject" most likely means 3 × 70, not "~70 subjects".
- The schema appears three times with invalid JSON (`["...]`); `gold_chunk` is floats in one place and `[md1, p2]` in another; `scenario` is undefined.
- Contamination check is "default: true" and also "opt-in". Review is "each question" in the PRD and "approve batch" in the architecture. `.env.example` says it is gitignored.
- Section 8.3 appears twice; "Open questions — resolved"; references to `hermes` and "blocks unauthenticated runs"; `architecture.md.bak` in `docs/`.
- `MULTI_HOP_REQUIRED` with `HOP_COUNT=1` default; `m` in the docs versus `DEFAULT_QUESTION_COUNT` in `.env.example`.

## Worth keeping

Human review gate, `evidence_quote`, hard negatives, the question-type and difficulty taxonomy, and the note on metric provenance.

## Add to the PRD

1. **Endpoint contract**: question in; answer out; optional ranked contexts (text, document id, score). Retrieval and rerank metrics run only when contexts are returned.
2. **Metric definitions** with formula, inputs, who computes it, and range. Suggested starting set:
   - retrieval recall@k and MRR against gold evidence
   - answer correctness against gold answers at claim level
   - faithfulness of the answer to retrieved context
   - correct refusal on unanswerable questions
3. **Judge configuration and validation**: a `JUDGE_AI_*` block in `.env`, and agreement measured on about 150 human-labelled items before scores are trusted.
4. **Product success criteria**, for example judge–human agreement above a threshold and repeat runs within the confidence interval.
5. **Non-goals and limits**: supported formats, corpus size and languages for v1.
6. **Unanswerable questions** as a type, at roughly 10–15% of the set.
7. **Language policy**: generate in the source language; if translating, translate gold answers too and review in the tested language.
8. **Cost and runtime estimate** per run.
9. **Reporting**: confidence intervals on every score, breakdowns by subject and type, and a minimum cell size.

## Add to the architecture

1. **Scoring stage** between answer generation and output.
2. **Question-validation stage** before human review.
3. **Evidence as spans**: document id plus offsets or quote, with a defined matching rule.
4. **Multi-hop generation method**: how chunks are grouped across pages or documents.
5. **No-context baseline run**, replacing the contamination check.
6. **Dataset versioning**: frozen, hashed question set; results record dataset version, judge model, prompt version and endpoint config.
7. **Endpoint adapter layer** with one or two reference adapters.
8. **Results data model**, separate from the question schema.
9. **Rerank comparison as a two-run mode** with a paired test on the same frozen set.

## Remove or defer

- GUI, metrics dashboard and idempotency-key material until a command-line version works on one subject.
- PRD Section 9 and `flaw.md` in their current form; replace with a short, accurately cited "threats to validity" section.
- Duplicate schema copies.

## First three things to fix

Endpoint contract, metric definitions, scoring stage. Everything else depends on them.
