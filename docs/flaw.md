# Flaws in RAG Model Evaluation — Research-Backed Assessment

This file documents flaws identified through a survey of recent RAG evaluation research (2023–2026, arXiv papers). Each flaw maps to a specific vulnerability in the evaluation process documented in PRD.md Section 9.

## Flaw #1: LLM Judge Circularity & Prompt Leakage
- **Source**: arxiv 2601.13227 — "Insider Knowledge: How Much Can RAG Systems Gain from Evaluation Secrets?"
- **Impact**: Evaluation metrics (faithfulness/citation) could be gamed if question-generation prompts or judge models are known to the system developer; scores may inflate without actual quality improvement
- **Mitigations**:
  - Document prompt templates as internal artifacts not committed to repo
  - Use held-out judge models for final validation
  - Randomize evaluation prompts across runs
  - Cross-validate with ≥2 independent evaluation frameworks (e.g., RAGAS + ARES or RAGChecker)

## Flaw #2: Metric Coverage Gap
- **Source**: Multiple survey papers (RAGAS, ARES, RAGChecker, MIRAGE)
- **Impact**: PRD tracks only 2 of 5+ evaluation dimensions (accuracy/relevance + faithfulness/citation); missing: context relevance, answer relevance (distinct from faithfulness), citation precision/recall, hallucination detection at token level
- **Mitigations**:
  - Add `context_relevance` metric (measures retrieved context focus, irrelevant info)
  - Add `answer_relevance` metric (distinct from faithfulness; measures if answer addresses actual question)
  - Add `citation_precision` / `citation_recall` for source attribution
  - Benchmark metrics against human judgment on a small validation set before release

## Flaw #3: Evaluation-Induced Optimization / "Gaming"
- **Source**: RAGAS hand-written prompts; ARES beats RAGAS by 59.3 and 14.4 percentage points on average
- **Impact**: Metrics assumed stable/objective but are prompt-sensitive; systems can be optimized to score well without actual quality improvement
- **Mitigations**:
  - Use SDK-default `max_retries=2` with jitter; avoid second retry layer
  - Apply idempotency keys on mutating calls; persist LLM response before triggering side effects
  - For batch/pipeline: bump `max_retries=5` but monitor quota; disable SDK retries behind custom gateway (LiteLLM/router) and let router handle retries/fallbacks centrally
  - Log error envelope: HTTP status code, provider's error type, `Retry-After` header, request idempotency key

## Flaw #4: Chunking → Evaluation Pipeline
- **Source**: RAGChecker; chunking quality bottlenecks retrieval across multiple surveys
- **Impact**: PRD records chunking assumption `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)` but doesn't make it configurable; poor chunking propagates errors through entire pipeline (question gen → answer gen → evaluation) without user awareness
- **Mitigations**:
  - Add `CHUNK_SIZE` / `CHUNK_OVERLAP` to `.env` schema with per-document-type guidance:
    - Dense, fact-heavy content (FAQs, API docs): 200–400 chars, 10–20% overlap
    - Narrative documents (reports, contracts): 800–1500 chars, 10–20% overlap
    - Default/unknown: 500 chars, 50 overlap (10%)
  - Document that chunking is a tunable parameter, not a fixed setting
  - Provide migration path: users on fixed-size chunking should re-chunk and re-generate questions

## Flaw #5: Data Contamination / Static Evaluation Pools
- **Source**: MIRAGE benchmark paper
- **Impact**: Evaluation question set may overlap with input material or publicly available datasets the target RAG model saw during training, producing artificially high scores
- **Mitigations**:
  - Implement semantic overlap check between generated questions and input material (embed both, compute similarity; if > threshold, diversify questions)
  - Document that evaluation pools are user-responsibility; recommend generating questions from held-out material splits
  - Add `EVAL_CONTAMINATION_CHECK` flag to `.env` (default: true) that, when enabled, surfaces estimated overlap percentage in the report

## Flaw #6: Single-Hop Task Focus in Most Benchmarks
- **Source**: MIRAGE benchmark paper — restricted to single-hop QA
- **Impact**: PRD's question_type system includes "Multi-hop reasoning" but the pipeline doesn't enforce multi-hop constraints during answer generation or evaluation; before/after rerank check may not adequately test multi-hop scenarios
- **Mitigations**:
  - Add `MULTI_HOP_REQUIRED` flag to question schema; when true, answer must reference ≥2 chunks with distinct positions in `gold_chunk`
  - Test before/after rerank separately per hop; report per-hop faithfulness scores
  - Add `HOP_COUNT` field to dataset schema (default: 1; user-configurable)
  - For multi-hop questions, surface "missing chunk" diagnostics if answer references fewer chunks than required

## Flaw #7: No Anti-"Gaming" Measures Documented
- **Impact**: System has no documented guardrails against the above vulnerabilities; a developer could unknowingly optimize for metrics that are fragile or easily gamed
- **Mitigations** (from Section 9.1 of PRD):
  - Review Section 9.1 and confirm which mitigations are applicable
  - Add `CHUNK_SIZE` and `CHUNK_OVERLAP` to `.env` if not using defaults
  - Decide whether to enable `EVAL_CONTAMINATION_CHECK` (true/false)
  - If multi-hop questions are needed, configure `MULTI_HOP_REQUIRED` and `HOP_COUNT`
  - Choose at least two evaluation frameworks for cross-validation (document in setup notes)
  - Document prompt templates as internal artifacts; never commit to repo
  - Plan re-evaluation reminder (e.g., calendar note every 3 months)

## How to Use This File
- Review before each planning cycle to confirm applicable mitigations
- Cross-reference with PRD.md Section 9 ("Research-Backed Flaws & Mitigations")
- Update when new research publications identify additional flaws
- Record mitigation status in your project's setup checklist

## Confidence & Recency
- All flaws sourced from arXiv papers published 2023–2026
- Research cutoff: mid-2026; newer papers may identify additional flaws
- Mitigations represent industry-best practices as of document generation date
- Not a substitute for ongoing literature review; new papers may supersede or refine these findings