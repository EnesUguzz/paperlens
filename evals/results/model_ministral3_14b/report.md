# Run `model_ministral3_14b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.extra_body` | {} | None |
| `llm.extra_body.reasoning_effort` | None | none |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.419 | 0.419 | +0.000 |
| answer_f1.abstractive | 0.225 | 0.225 | +0.000 |
| answer_f1.extractive | 0.473 | 0.473 | +0.000 |
| answer_f1.gold.abstractive | 0.302 | 0.302 | +0.000 |
| answer_f1.gold.extractive | 0.464 | 0.464 | +0.000 |
| answer_f1.gold.unanswerable | 0.411 | 0.411 | +0.000 |
| answer_f1.gold.yes_no | 0.464 | 0.464 | +0.000 |
| answer_f1.unanswerable | 0.458 | 0.458 | +0.000 |
| answer_f1.yes_no | 0.486 | 0.486 | +0.000 |
| answer_words_mean | 15.492 | 15.492 | +0.000 |
| completion_tokens_mean | 51.800 | 51.800 | +0.000 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.297 | 0.297 | +0.000 |
| evidence_f1_ceiling | 0.524 | 0.524 | +0.000 |
| false_refusal_rate | 0.042 | 0.042 | +0.000 |
| gold_recall | 0.494 | 0.494 | +0.000 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 1151.187 | 1042.795 | +108.392 |
| latency_p95_ms | 2052.496 | 1850.713 | +201.783 |
| n.abstractive | 55.000 | 55.000 | +0.000 |
| n.extractive | 144.000 | 144.000 | +0.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 24.000 | 24.000 | +0.000 |
| n.yes_no | 27.000 | 27.000 | +0.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2577.000 | 2577.000 | +0.000 |
| prompt_tokens_mean | 2228.868 | 2228.868 | +0.000 |
| refusal_recall | 0.270 | 0.270 | +0.000 |
| retrieval.doc_hit_rate | 1.000 | 1.000 | +0.000 |
| retrieval.mrr | 0.552 | 0.552 | +0.000 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.409 | 0.409 | +0.000 |
| retrieval.ndcg_at_3 | 0.492 | 0.492 | +0.000 |
| retrieval.ndcg_at_5 | 0.527 | 0.527 | +0.000 |
| retrieval.recall_at_1 | 0.335 | 0.335 | +0.000 |
| retrieval.recall_at_3 | 0.603 | 0.603 | +0.000 |
| retrieval.recall_at_5 | 0.703 | 0.703 | +0.000 |

## Biggest regressions

None.
