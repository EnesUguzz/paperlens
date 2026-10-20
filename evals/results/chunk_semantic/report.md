# Run `chunk_semantic`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `chunking.semantic_breakpoint_percentile` | None | 90.0 |
| `chunking.semantic_min_words` | None | 50 |
| `chunking.strategy` | fixed | semantic |
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
| `rerank.enabled` | None | False |
| `rerank.max_length` | None | 512 |
| `rerank.model_name` | None | BAAI/bge-reranker-v2-m3 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.427 | 0.419 | +0.009 |
| answer_f1.abstractive | 0.234 | 0.225 | +0.009 |
| answer_f1.extractive | 0.503 | 0.473 | +0.030 |
| answer_f1.gold.abstractive | 0.314 | 0.302 | +0.011 |
| answer_f1.gold.extractive | 0.494 | 0.464 | +0.030 |
| answer_f1.gold.unanswerable | 0.374 | 0.411 | -0.037 |
| answer_f1.gold.yes_no | 0.420 | 0.464 | -0.044 |
| answer_f1.unanswerable | 0.462 | 0.458 | +0.003 |
| answer_f1.yes_no | 0.426 | 0.486 | -0.060 |
| answer_words_mean | 14.240 | 15.492 | -1.252 |
| completion_tokens_mean | 49.168 | 51.800 | -2.632 |
| context_words_mean | 643.400 | 998.848 | -355.448 |
| data.questions_dropped | 0.000 | 0.000 | +0.000 |
| data.unmatched_evidence_rate | 0.046 | 0.046 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.413 | 0.297 | +0.116 |
| evidence_f1_ceiling | 0.658 | 0.524 | +0.134 |
| false_refusal_rate | 0.066 | 0.042 | +0.023 |
| gold_recall | 0.496 | 0.494 | +0.002 |
| index.num_chunks | 9177.000 | 6676.000 | +2501.000 |
| judge.accuracy | 0.548 | 0.500 | +0.048 |
| judge.accuracy.gold.abstractive | 0.448 | 0.310 | +0.138 |
| judge.accuracy.gold.extractive | 0.634 | 0.595 | +0.038 |
| judge.accuracy.gold.unanswerable | 0.486 | 0.486 | +0.000 |
| judge.accuracy.gold.yes_no | 0.417 | 0.458 | -0.042 |
| judge.agreement_on_high_f1 | 1.000 | 1.000 | +0.000 |
| judge.error_rate | 0.000 | 0.000 | +0.000 |
| judge.llm_calls | 167.000 | 176.000 | -9.000 |
| latency_p50_ms | 936.240 | 1042.795 | -106.555 |
| latency_p95_ms | 1500.882 | 1850.713 | -349.831 |
| n.abstractive | 59.000 | 55.000 | +4.000 |
| n.extractive | 139.000 | 144.000 | -5.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 26.000 | 24.000 | +2.000 |
| n.yes_no | 26.000 | 27.000 | -1.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2232.000 | 2577.000 | -345.000 |
| prompt_tokens_mean | 1724.292 | 2228.868 | -504.576 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.297 | 0.270 | +0.027 |
| retrieval.doc_hit_rate | 1.000 | 1.000 | +0.000 |
| retrieval.mrr | 0.506 | 0.552 | -0.046 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.361 | 0.409 | -0.048 |
| retrieval.ndcg_at_3 | 0.439 | 0.492 | -0.053 |
| retrieval.ndcg_at_5 | 0.492 | 0.527 | -0.035 |
| retrieval.recall_at_1 | 0.296 | 0.335 | -0.039 |
| retrieval.recall_at_3 | 0.527 | 0.603 | -0.076 |
| retrieval.recall_at_5 | 0.667 | 0.703 | -0.036 |

## Biggest regressions

- **What is the size of bioScope corpus?** (`45be665a4504f0c7f458cf3f75a95d5a75eefd42`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: BF: 2670 sentences, BA: 11871 sentences
  - top context: [Introduction] This task was the CoNLL-2010 Shared Task (BIBREF0), which had 3 different subtasks. Task 1B was speculation cue detection on the BioScope Corpus, Task 1W was weasel identification from Wikipedia articles, and Task 2 was speculation scope resolution from the BioScope Corpus. For each t

- **Is fine-tuning required to incorporate these embeddings into existing models?** (`b06512c17d99f9339ffdab12cedbc63501ff527e`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: We can just use Dolores embeddings as a drop-in replacement. We just initialize the corresponding embedding layer with Dolores embeddings.
  - top context: [Dolores: Learner] Given a downstream model, Dolores learns a weighted linear combination of the components of $R_t$ to yield a single vector for use in the embedding layer of the downstream model. In particular $$\texttt {\textsc {Dolores}}_t = [ x_t , \sum _{i=1}^{L} \lambda _i h_{t,i} ],$$   (Eq.

- **what evaluation metrics are reported?** (`d9412dda3279729e95fcb35cbed09e61577a896e`, answer_f1 1.00 -> 0.00)
  - gold: precision, recall, F1 and accuracy | precision, recall, F1 , accuracy 
  - predicted: P/R/F1, Acc, MaxFreq
  - top context: [Judgment Prediction] Automatic judgment prediction has been studied for decades. At the very first stage of judgment prediction studies, researchers focus on mathematical and statistical analysis of existing cases, without any conclusions or methodologies on how to predict them BIBREF9 , BIBREF10 ,

- **what dataset was used?** (`e48e750743aef36529fbea4328b8253dbe928b4d`, answer_f1 1.00 -> 0.00)
  - gold: WASSA-2017 Shared Task on Emotion Intensity
  - predicted: dev data set BIBREF19, train and dev sets
  - top context: [Parameter Optimization] In order to find the optimal parameter values for the EmoInt system, an extensive grid search was performed through the scikit-Learn framework over all subsets of the training set (shuffled), using stratified 10-fold cross validation and optimizing the Pearson's Correlation 

- **What is the size of SFU Review corpus?** (`e86130c5b9ab28f0ec539c2bed1b1ae9efb99b7d`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: SFU Review Corpus (SFU)
  - top context: [Introduction] This task was the CoNLL-2010 Shared Task (BIBREF0), which had 3 different subtasks. Task 1B was speculation cue detection on the BioScope Corpus, Task 1W was weasel identification from Wikipedia articles, and Task 2 was speculation scope resolution from the BioScope Corpus. For each t
