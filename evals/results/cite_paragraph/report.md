# Run `cite_paragraph`

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
| `generation.cite_unit` | None | paragraph |
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
| `rerank.enabled` | None | False |
| `rerank.max_length` | None | 512 |
| `rerank.model_name` | None | BAAI/bge-reranker-v2-m3 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.421 | 0.419 | +0.002 |
| answer_f1.abstractive | 0.242 | 0.225 | +0.017 |
| answer_f1.extractive | 0.485 | 0.473 | +0.011 |
| answer_f1.gold.abstractive | 0.325 | 0.302 | +0.023 |
| answer_f1.gold.extractive | 0.475 | 0.464 | +0.010 |
| answer_f1.gold.unanswerable | 0.408 | 0.411 | -0.003 |
| answer_f1.gold.yes_no | 0.380 | 0.464 | -0.083 |
| answer_f1.unanswerable | 0.500 | 0.458 | +0.042 |
| answer_f1.yes_no | 0.375 | 0.486 | -0.111 |
| answer_words_mean | 15.116 | 15.492 | -0.376 |
| completion_tokens_mean | 52.444 | 51.800 | +0.644 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| data.questions_dropped | 0.000 | 0.000 | +0.000 |
| data.unmatched_evidence_rate | 0.046 | 0.046 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.455 | 0.297 | +0.158 |
| evidence_f1_ceiling | 0.787 | 0.524 | +0.263 |
| false_refusal_rate | 0.047 | 0.042 | +0.005 |
| gold_recall | 0.496 | 0.494 | +0.002 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| judge.accuracy | 0.524 | 0.500 | +0.024 |
| judge.accuracy.gold.abstractive | 0.362 | 0.310 | +0.052 |
| judge.accuracy.gold.extractive | 0.626 | 0.595 | +0.031 |
| judge.accuracy.gold.unanswerable | 0.514 | 0.486 | +0.027 |
| judge.accuracy.gold.yes_no | 0.375 | 0.458 | -0.083 |
| judge.agreement_on_high_f1 | 0.950 | 1.000 | -0.050 |
| judge.error_rate | 0.000 | 0.000 | +0.000 |
| judge.llm_calls | 179.000 | 176.000 | +3.000 |
| latency_p50_ms | 1086.235 | 1042.795 | +43.440 |
| latency_p95_ms | 1808.953 | 1850.713 | -41.761 |
| n.abstractive | 55.000 | 55.000 | +0.000 |
| n.extractive | 144.000 | 144.000 | +0.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 24.000 | 24.000 | +0.000 |
| n.yes_no | 27.000 | 27.000 | +0.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2662.000 | 2577.000 | +85.000 |
| prompt_tokens_mean | 2265.024 | 2228.868 | +36.156 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.297 | 0.270 | +0.027 |
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

- **did they compare with other extractive summarization methods?** (`c2cb6c4500d9e02fc9a1bdffd22c3df69655189f`, answer_f1 1.00 -> 0.00)
  - gold: No
  - predicted: Unanswerable
  - top context: [Introduction] aims are critical steps in ultimately developing methods that can automate discharge summary creation. In the broader field of summarization, automization was meant to standardize output while also saving time and effort. Pioneering strategies in summarization started by extracting "s

- **Do the authors report results only on English data?** (`c49ee6ac4dc812ff84d255886fd5aff794f53c39`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Unanswerable
  - predicted: For evaluating the effectiveness of Rouge variants and our metric (Sera), we use this benchmark, which consists of 20 topics each with a **biomedical journal article** and 4 gold human written summaries.
  - top context: [Summarization Evaluation by Relevance Analysis (Sera)] keep the noun phrases as informative concepts in the summary and eliminate all other terms; and (iii) Keywords (KW): We only keep the keywords and key phrases in the summary. For extracting the keywords and keyphrases (with length of up to 3 te

- **Are the automatically constructed datasets subject to quality control?** (`e97186c51d4af490dba6faaf833d269c8256426c`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: vulnerable to annotation artifacts, which we carefully control for
  - top context: [Discussion and Conclusion] cheaper alternative to recent attempts to build architectures that explicitly incorporate structured knowledge BIBREF55; we see many areas where our inoculation strategy could be improved for such purposes, including having more complex loss functions that manage old and 

- **what is the source of the data?** (`30af1926559079f59b0df055da76a3a34df8336f`, answer_f1 0.67 -> 0.00)
  - gold: Android application
  - predicted: crowdsourcing
  - top context: [Introduction] and final version of the database was released at the beginning of 2019. In BIBREF4, the running project and its data collection scenarios were described, alongside with some preliminary results and statistics. In this paper, we announce the final and cleaned-up version of the databas

- **What is the size of the parallel corpus used to train the model constraints?** (`9cf070d6671ee4a6353f79a165aa648309e01295`, answer_f1 1.00 -> 0.36)
  - gold: 1500 sentences | 1500 sentences
  - predicted: A part of the ILCI English-Hindi Tourism parallel corpus (1500 sentences)
  - top context: [Experiments and Results] were converted to MSTParser format from ConLL format for training. A part of the ILCI English-Hindi Tourism parallel corpus (1500 sentences) was used for training the classifiers. This corpus was POS-tagged using the Stanford POS Tagger BIBREF5 for English and using the Hin
