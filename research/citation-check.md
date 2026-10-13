# Citation check — PRD Section 9 and flaw.md

Each "research-backed flaw" checked against its cited source on 2026-10-13.

| # | Claim in the docs | What the source says | Result |
|---|---|---|---|
| 1 | arXiv 2601.13227 shows LLM-judge circularity and prompt leakage | Paper is real: "Insider Knowledge: How Much Can RAG Systems Gain from Evaluation Secrets?" (Dietz et al.). Scores inflate to near-perfect when prompts or gold nuggets leak. | Citation correct. Mitigation is off: the paper recommends blind evaluation and methodological diversity. "Never commit prompt templates" conflicts with an open-source, reproducible tool. |
| 2 | RAGAS, ARES, RAGChecker, MIRAGE show a metric coverage gap | These frameworks do cover context relevance, answer relevance and faithfulness. | Broadly fair, but cited as "multiple survey papers"; none of these is a survey. |
| 3 | "ARES 59.3 pp higher" shows metrics can be gamed | 59.3 and 14.4 points are ARES's judge accuracy advantage over RAGAS on context relevance and answer relevance. | Misused. It is evidence that judges differ in accuracy, not of gaming. The listed mitigations (`max_retries`, idempotency keys, retry gateways) are unrelated to the flaw. |
| 4 | RAGChecker shows chunking bottlenecks retrieval | RAGChecker reports a trade-off between retrieval improvement and noise. No chunking claim was found in what was checked. | Unverified. The specific chunk-size guidance (200–400, 800–1500 characters) has no source given. |
| 5 | MIRAGE identifies data contamination and static evaluation pools | MIRAGE says models may have seen parts of the dataset in pre-training. It presents its fixed pool as a feature, not a flaw. | Half wrong. The contamination is about model training data, not question-to-source similarity, so the proposed embedding-similarity check does not address it. |
| 6 | MIRAGE shows most benchmarks focus on single-hop | The limitations section says MIRAGE itself is restricted to single-hop QA. | Misattributed. It is a limitation of that benchmark, not a finding about the field. |
| 7 | No anti-gaming measures documented | No source. | Circular: the mitigation is "add this section". |

## Other notes

- `flaw.md` says all flaws are "sourced from arXiv papers published 2023–2026". Flaw 7 has no source and flaws 3–6 misread theirs.
- ARES's introduction says 59.3; its Section 5.1 gives 59.9 for the same comparison.
