# Run `multi_query`

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
| `query.transform` | None | multi_query |
| `rerank.batch_size` | None | 16 |
| `rerank.device` | None | cuda |
| `rerank.enabled` | None | False |
| `rerank.max_length` | None | 512 |
| `rerank.model_name` | None | BAAI/bge-reranker-v2-m3 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.427 | 0.419 | +0.008 |
| answer_f1.abstractive | 0.263 | 0.225 | +0.038 |
| answer_f1.extractive | 0.478 | 0.473 | +0.004 |
| answer_f1.gold.abstractive | 0.343 | 0.302 | +0.041 |
| answer_f1.gold.extractive | 0.461 | 0.464 | -0.003 |
| answer_f1.gold.unanswerable | 0.417 | 0.411 | +0.006 |
| answer_f1.gold.yes_no | 0.458 | 0.464 | -0.005 |
| answer_f1.unanswerable | 0.429 | 0.458 | -0.030 |
| answer_f1.yes_no | 0.483 | 0.486 | -0.003 |
| answer_words_mean | 15.232 | 15.492 | -0.260 |
| completion_tokens_mean | 127.824 | 51.800 | +76.024 |
| context_words_mean | 999.880 | 998.848 | +1.032 |
| data.questions_dropped | 0.000 | - | - |
| data.unmatched_evidence_rate | 0.046 | - | - |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.271 | 0.297 | -0.026 |
| evidence_f1_ceiling | 0.505 | 0.524 | -0.019 |
| false_refusal_rate | 0.033 | 0.042 | -0.009 |
| gold_recall | 0.497 | 0.494 | +0.003 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 2224.180 | 1042.795 | +1181.385 |
| latency_p95_ms | 2917.041 | 1850.713 | +1066.328 |
| n.abstractive | 55.000 | 55.000 | +0.000 |
| n.extractive | 145.000 | 144.000 | +1.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 21.000 | 24.000 | -3.000 |
| n.yes_no | 29.000 | 27.000 | +2.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 3294.000 | 2577.000 | +717.000 |
| prompt_tokens_mean | 2861.144 | 2228.868 | +632.276 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.216 | 0.270 | -0.054 |
| retrieval.doc_hit_rate | 1.000 | 1.000 | +0.000 |
| retrieval.mrr | 0.478 | 0.552 | -0.074 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.326 | 0.409 | -0.083 |
| retrieval.ndcg_at_3 | 0.409 | 0.492 | -0.083 |
| retrieval.ndcg_at_5 | 0.468 | 0.527 | -0.059 |
| retrieval.recall_at_1 | 0.265 | 0.335 | -0.070 |
| retrieval.recall_at_3 | 0.522 | 0.603 | -0.081 |
| retrieval.recall_at_5 | 0.672 | 0.703 | -0.031 |

## Biggest regressions

- **What is the size of bioScope corpus?** (`45be665a4504f0c7f458cf3f75a95d5a75eefd42`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: BF: 2670 sentences, BA: 11871 sentences
  - top context: [Introduction] the speculation, while the words affected by the speculation are in the scope of that cue. This task was the CoNLL-2010 Shared Task (BIBREF0), which had 3 different subtasks. Task 1B was speculation cue detection on the BioScope Corpus, Task 1W was weasel identification from Wikipedia

- **Do they use attention?** (`45e9533586199bde19313cd43b3d0ecadcaf7a33`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: Unanswerable
  - top context: [Abstract] This paper presents an extension of the Stochastic Answer Network (SAN), one of the state-of-the-art machine reading comprehension models, to be able to judge whether a question is unanswerable or not. The extended SAN contains two components: a span detector and a binary classifier for j

- **Do the authors report results only on English data?** (`c49ee6ac4dc812ff84d255886fd5aff794f53c39`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Unanswerable
  - predicted: The only scientific summarization benchmark is from **TAC 2014 summarization track**
  - top context: [Abstract] Evaluation of text summarization approaches have been mostly based on metrics that measure similarities of system generated summaries with a set of human written gold-standard summaries. The most widely used metric in summarization evaluation has been the ROUGE family. ROUGE solely relies

- **What is the size of SFU Review corpus?** (`e86130c5b9ab28f0ec539c2bed1b1ae9efb99b7d`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: SFU Review Corpus (SFU)
  - top context: [Conclusion and Future Scope] SFU on the scope resolution task over the single dataset training approach. We thus significantly advance the state-of-the-art for speculation detection and scope resolution. On the negation scope resolution task, we applied the XLNet and RoBERTa and obtained a gain of 

- **what evaluation metrics are reported?** (`d9412dda3279729e95fcb35cbed09e61577a896e`, answer_f1 1.00 -> 0.04)
  - gold: precision, recall, F1 and accuracy | precision, recall, F1 , accuracy 
  - predicted: P/R/F1 are reported for positive samples and calculated as the mean score over 10-time experiments. Acc is defined as the proportion of test samples classified correctly, equal to micro-precision. MaxFreq refers to always predicting the most frequent label, i.e. support in our dataset.
  - top context: [Introduction] original documents have special typographical characteristics indicating the discourse structure. We also take into account law articles and their corresponding juridical interpretations. We also implement and evaluate previous methods on our dataset, which prove to be strong baseline
