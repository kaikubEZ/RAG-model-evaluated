# Related work — how others evaluate RAG

Verification level is marked per entry: **[text]** paper text read, **[abstract]** abstract page read, **[summary]** search-result summary only.

## Evaluation frameworks and metrics

### RAGAS — arXiv 2309.15217 [summary]
- Reference-free metrics scored by an LLM: faithfulness, answer relevance, context relevance.
- Its test set generator builds a knowledge graph over the documents and synthesises single-hop and multi-hop queries, each in specific and abstract variants, with optional user personas to vary tone and detail.
- Docs: https://docs.ragas.io/en/latest/concepts/test_data_generation/rag/
- **Implication**: multi-hop questions come from linking several document nodes, not from prompting on one chunk.

### ARES — arXiv 2311.09476 [text]
- Scores context relevance, answer faithfulness and answer relevance.
- Generates synthetic training data to fine-tune lightweight judge models.
- Uses a human-annotated validation set with prediction-powered inference (PPI) to correct judge errors and produce confidence intervals (95% in the experiments). About 150 annotations is the stated minimum; the main experiments used 300.
- Reports beating RAGAS by 59.3 points on context relevance and 14.4 on answer relevance. These measure how accurately the judge classifies query–passage–answer triples. Section 5.1 gives 59.9 for context relevance; the paper does not explain the difference from the introduction's 59.3.
- **Implication**: a judge needs a small human-labelled set to calibrate it, and scores should carry confidence intervals.

### RAGChecker — arXiv 2408.08067 [summary]
- Claim-level entailment checking: claims are extracted from the response and from the ground-truth answer and checked against other texts.
- Separate diagnostic metrics for the retriever and the generator.
- Authors built a human-judgment dataset and report better correlation with humans than other metrics.
- Evaluated 8 RAG systems; one finding is a trade-off between better retrieval and added noise.
- **Implication**: claim-level precision and recall against gold answers is a concrete definition of "answer correctness", and retriever and generator should be scored separately.

### TREC 2024 RAG Track, AutoNuggetizer — arXiv 2504.15068, 2411.09607 [summary]
- Nugget-based evaluation: extract the facts an answer should contain, then check which ones each answer covers.
- Fully automatic nugget scoring agrees strongly with human-based variants at the run (system) level across 301 topics; agreement is weaker per topic.
- Automating only nugget assignment while keeping manual nugget creation gives better agreement, especially per topic.
- LLMs assign labels more strictly than human assessors.
- **Implication**: automatic scores are trustworthy for ranking systems, much less so for per-question verdicts.

### MIRAGE — arXiv 2504.17137 (Findings of NAACL 2025) [text]
- 7,560 queries with a 37,800-chunk retrieval pool, built from five single-hop Wikipedia QA datasets (PopQA, NQ, TriviaQA, IfQA, DROP). Chunks are 330 tokens.
- Each query has at least one positive chunk plus similar negative chunks, so retriever and LLM can be evaluated separately.
- Four adaptability metrics from comparing answers with no context, oracle context and mixed context: noise vulnerability, context acceptability, context insensitivity, context misinterpretation.
- Stated limitations: restricted to single-hop QA; models may have seen parts of the data in training.
- **Implication**: running the same question with no context and with oracle context separates what the model knows from what retrieval contributes. This is the right way to detect contamination.

### RAGEval — arXiv 2408.01262 (ACL 2025) [summary]
- Schema-based pipeline that generates documents, questions, answers and references for vertical domains.
- Metrics: completeness, hallucination, irrelevance; LLM scoring reported as consistent with human evaluation.
- Code: OpenBMB/RAGEval.

## Synthetic question generation

### Can we Evaluate RAGs with Synthetic Data? — arXiv 2508.11758 [summary]
- Synthetic QA benchmarks rank RAG systems reliably when the retriever configuration varies, and not consistently when the generator varies.
- Attributed to task mismatch between synthetic and human benchmarks and stylistic bias favouring some generators.
- Notes that RGB, Ragas and ARES all generate QA pairs by prompting an LLM on specific corpus chunks.

### Know Your RAG — arXiv 2411.19710 (COLING 2025) [summary]
- Public QA datasets can lead to non-optimal system design, and common generation tools produce unbalanced data.
- Taxonomy of question–context pairs: `fact_single`, `summary`, `reasoning`, `unanswerable`.

### DataMorgana — arXiv 2501.12789 [summary]
- Standard synthetic generation produces questions that are individually good but lack the variety of real traffic.
- Lets the user configure question and user categories and control their distribution.
- Used to build the SIGIR 2025 LiveRAG benchmark (arXiv 2511.14531): 895 questions with item-response-theory difficulty estimates.

### UAEval4RAG — arXiv 2412.12300 (ACL 2025) [summary]
- Existing frameworks focus on answerable queries and ignore whether a system correctly declines unanswerable ones.
- Six categories of unanswerable request (including underspecified, false-presupposition, nonsensical), synthesised automatically for any knowledge base.
- Metrics: unanswered ratio, acceptable ratio.
- No single configuration was best on both answerable and unanswerable requests.

## Judge reliability

### Insider Knowledge — arXiv 2601.13227 [abstract]
- Near-perfect scores can be achieved when evaluation elements (prompt templates, gold nuggets) are leaked or predictable.
- Recommends blind evaluation settings and methodological diversity.

### Self-preference bias — arXiv 2410.21819, 2506.02592, 2601.22548 [summary]
- GPT-4 shows measurable self-preference; one explanation is that judges favour low-perplexity text.
- A later paper argues part of the measured bias is a confound, with about half of earlier findings staying significant under a stricter baseline.
- RAGalyst (arXiv 2511.04502) found minimal bias from which model generated the questions.
- **Implication**: evidence is mixed. Using a judge different from the target's generator is cheap insurance, not a critical requirement.

## Multilingual

### MEMERAG — arXiv 2502.17163 [summary]
- Translated benchmarks carry translationese (simpler syntax and word choice) and may not reflect native users.
- Built from MIRACL's native-language questions with native expert annotators, in English, German, Spanish, French and Hindi.
- **Implication**: generate questions in the source language where possible; treat translated sets as a complement.

## Statistics

### Adding Error Bars to Evals — arXiv 2411.00640 [summary]
- Treats eval questions as a sample from a larger population; gives formulas for confidence intervals, comparing two models, and planning sample size.
- Covers clustered standard errors and variance reduction through paired comparison (per a third-party description; not checked in the paper).
- **Implication**: compare two configurations on the same question set with a paired test, and report intervals.
