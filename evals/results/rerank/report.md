# Run `rerank`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `chunking.semantic_breakpoint_percentile` | None | 90.0 |
| `chunking.semantic_min_words` | None | 50 |
| `data.align_min_score` | None | 85.0 |
| `data.align_min_words` | None | 8 |
| `data.align_score_margin` | None | 5.0 |
| `data.pdf_download_delay_s` | None | 3.0 |
| `data.pdf_url_template` | None | https://arxiv.org/pdf/{paper_id} |
| `generation.cite_unit` | None | chunk |
| `judge.extra_body.reasoning_effort` | None | none |
| `judge.model` | None | qwen3:14b |
| `judge.prompt_template` | None | prompts/judge.md |
| `meta.llm_digest` | not recorded | 4760c35aeb9d |
| `meta.server_version` | not recorded | 0.40.6 |
| `query.hyde_prompt` | None | prompts/hyde.md |
| `query.include_original` | None | True |
| `query.multi_query_prompt` | None | prompts/multi_query.md |
| `query.n_queries` | None | 3 |
| `query.transform` | None | none |
| `rerank.batch_size` | None | 16 |
| `rerank.device` | None | cuda |
| `rerank.enabled` | None | True |
| `rerank.max_length` | None | 512 |
| `rerank.model_name` | None | BAAI/bge-reranker-v2-m3 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.387 | 0.419 | -0.032 |
| answer_f1.abstractive | 0.237 | 0.225 | +0.012 |
| answer_f1.extractive | 0.451 | 0.473 | -0.022 |
| answer_f1.gold.abstractive | 0.294 | 0.302 | -0.009 |
| answer_f1.gold.extractive | 0.442 | 0.464 | -0.023 |
| answer_f1.gold.unanswerable | 0.374 | 0.411 | -0.037 |
| answer_f1.gold.yes_no | 0.333 | 0.464 | -0.130 |
| answer_f1.unanswerable | 0.458 | 0.458 | +0.000 |
| answer_f1.yes_no | 0.321 | 0.486 | -0.165 |
| answer_words_mean | 14.540 | 15.492 | -0.952 |
| completion_tokens_mean | 50.776 | 51.800 | -1.024 |
| context_words_mean | 997.604 | 998.848 | -1.244 |
| data.questions_dropped | 0.000 | - | - |
| data.unmatched_evidence_rate | 0.046 | - | - |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.285 | 0.297 | -0.012 |
| evidence_f1_ceiling | 0.526 | 0.524 | +0.002 |
| false_refusal_rate | 0.042 | 0.042 | +0.000 |
| gold_recall | 0.454 | 0.494 | -0.040 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 2256.972 | 1042.795 | +1214.177 |
| latency_p95_ms | 2897.249 | 1850.713 | +1046.535 |
| n.abstractive | 59.000 | 55.000 | +4.000 |
| n.extractive | 139.000 | 144.000 | -5.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 24.000 | 24.000 | +0.000 |
| n.yes_no | 28.000 | 27.000 | +1.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2635.000 | 2577.000 | +58.000 |
| prompt_tokens_mean | 2209.392 | 2228.868 | -19.476 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.270 | 0.270 | +0.000 |
| retrieval.doc_hit_rate | 1.000 | 1.000 | +0.000 |
| retrieval.mrr | 0.501 | 0.552 | -0.051 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.330 | 0.409 | -0.078 |
| retrieval.ndcg_at_3 | 0.441 | 0.492 | -0.051 |
| retrieval.ndcg_at_5 | 0.506 | 0.527 | -0.021 |
| retrieval.recall_at_1 | 0.279 | 0.335 | -0.056 |
| retrieval.recall_at_3 | 0.556 | 0.603 | -0.047 |
| retrieval.recall_at_5 | 0.731 | 0.703 | +0.028 |

## Biggest regressions

- **Did they use a crowdsourcing platform for annotations?** (`0ba3ea93eef5660a79ea3c26c6a270eac32dfa4c`, answer_f1 1.00 -> 0.00)
  - gold: No | Unanswerable
  - predicted: The manually annotated sub-corpus contains sense labels of 106 different words: 50 nouns, 48 verbs and 8 adjectives.
  - top context: [Classification] features, which represent the occurrence of a certain word in context, we have INLINEFORM0 and: INLINEFORM1 Which could be rewritten as: INLINEFORM0 The expression has been formulated as a product of two factors: INLINEFORM0 , independent from observed features and corresponding to 

- **What language are the reviews in?** (`2236386729105f5cf42f73cc055ce3acdea2d452`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | English
  - predicted: TripAdvisor website
  - top context: [Attention-based models for aspect-based sentiment analysis] In these papers, several adaptations of the end-to-end memory network (MemN2N) BIBREF18 are used to predict the polarity of a review regarding a given aspect. In that configuration, the review is encoded into the memory cells and the contr

- **Which real-world datasets are used?** (`3941401a182a3d6234894a5c8a75d48c6116c45c`, answer_f1 1.00 -> 0.00)
  - gold: Tweets related to CyberAttack and tweets related to PoliticianDeath | cyber security (CyberAttack), death of politicians (PoliticianDeath)
  - predicted: multiple real-world datasets
  - top context: [Abstract] Microblogging platforms such as Twitter are increasingly being used in event detection. Existing approaches mainly use machine learning models and rely on event-related keywords to collect the data for model training. These approaches make strong assumptions on the distribution of the rel

- **Do they use attention?** (`45e9533586199bde19313cd43b3d0ecadcaf7a33`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: attention
  - top context: [Model] layer, and also with the output of the first contextual encoding layer as the input of its second encoding layer. Thus, we obtain the final representation of the contextual encoding layer by a concatenation of the outputs of two BiLSTM: INLINEFORM0 for questions and INLINEFORM1 for passages.

- **How many different characters were in dataset?** (`808f0ad46ca4eb4ea5492f9e14ca043fe1e206cc`, answer_f1 1.00 -> 0.00)
  - gold: 45,821 characters | 45,821 characters
  - predicted: 45,820
  - top context: [Methodology ::: Character Community Module (CCM)] employ a two-level connection representation by ranking all characters against each other in terms of their correlation with $c_t$. For the first level, the set $S^{FL}$ is the top 10% (4582) most highly correlated characters with $c_t$ out of the 4
