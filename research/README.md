# Research notes — RAG evaluation

Research done on 2026-10-13 while reviewing `docs/PRD.md`, `docs/architecture.md` and `flaw.md`.

| File | Contents |
|---|---|
| [papers.md](papers.md) | What each related paper does and what it implies for this project |
| [citation-check.md](citation-check.md) | Each citation in PRD Section 9 / `flaw.md` checked against the source |
| [design-review.md](design-review.md) | Review findings on the original design and what to add |
| [gui-templates.md](gui-templates.md) | GUI framework options and recommendation |

## How much was verified

- **Read from the paper text**: ARES (2311.09476) and MIRAGE (2504.17137).
- **Read from the abstract page**: Insider Knowledge (2601.13227).
- **From search-result summaries only**: all other papers. Their claims here are accurate to those summaries but the full papers were not read; check the paper before quoting a number.

## Main conclusions

1. Every established framework defines its metrics precisely and validates its judge against human labels. The original design (now `docs/flaw.md` and `docs/architecture.md.bak`) did neither.
2. Synthetic question sets are reliable for comparing retrievers and less reliable for comparing generators, so generated questions need filtering and human spot checks.
3. Retrieval and rerank metrics need the target endpoint to return its ranked contexts. A pure question-in, answer-out endpoint can only be scored on the answer.
4. Unanswerable questions and native-language questions are standard parts of recent benchmarks and are missing here.
5. Several citations in PRD Section 9 are misapplied; see [citation-check.md](citation-check.md).
