# Run `m1b_section_best`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `chunking.chunk_overlap` | 40 | 0 |
| `chunking.semantic_breakpoint_percentile` | None | 90.0 |
| `chunking.semantic_min_words` | None | 50 |
| `chunking.strategy` | fixed | section |
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
| answer_f1 | 0.440 | 0.419 | +0.022 |
| answer_f1.abstractive | 0.247 | 0.225 | +0.022 |
| answer_f1.extractive | 0.506 | 0.473 | +0.033 |
| answer_f1.gold.abstractive | 0.325 | 0.302 | +0.022 |
| answer_f1.gold.extractive | 0.503 | 0.464 | +0.039 |
| answer_f1.gold.unanswerable | 0.440 | 0.411 | +0.029 |
| answer_f1.gold.yes_no | 0.380 | 0.464 | -0.084 |
| answer_f1.unanswerable | 0.500 | 0.458 | +0.042 |
| answer_f1.yes_no | 0.412 | 0.486 | -0.075 |
| answer_words_mean | 14.192 | 15.492 | -1.300 |
| completion_tokens_mean | 49.788 | 51.800 | -2.012 |
| context_words_mean | 689.316 | 998.848 | -309.532 |
| data.questions_dropped | 0.000 | 0.000 | +0.000 |
| data.unmatched_evidence_rate | 0.046 | 0.046 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.491 | 0.297 | +0.194 |
| evidence_f1_ceiling | 0.770 | 0.524 | +0.246 |
| false_refusal_rate | 0.061 | 0.042 | +0.019 |
| gold_recall | 0.507 | 0.494 | +0.012 |
| index.num_chunks | 8360.000 | 6676.000 | +1684.000 |
| latency_p50_ms | 963.592 | 1042.795 | -79.203 |
| latency_p95_ms | 1642.370 | 1850.713 | -208.343 |
| n.abstractive | 53.000 | 55.000 | -2.000 |
| n.extractive | 142.000 | 144.000 | -2.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 28.000 | 24.000 | +4.000 |
| n.yes_no | 27.000 | 27.000 | +0.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2417.000 | 2577.000 | -160.000 |
| prompt_tokens_mean | 1788.992 | 2228.868 | -439.876 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.351 | 0.270 | +0.081 |
| retrieval.doc_hit_rate | 1.000 | 1.000 | +0.000 |
| retrieval.mrr | 0.484 | 0.552 | -0.068 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.322 | 0.409 | -0.087 |
| retrieval.ndcg_at_3 | 0.423 | 0.492 | -0.069 |
| retrieval.ndcg_at_5 | 0.474 | 0.527 | -0.053 |
| retrieval.recall_at_1 | 0.264 | 0.335 | -0.070 |
| retrieval.recall_at_3 | 0.524 | 0.603 | -0.079 |
| retrieval.recall_at_5 | 0.665 | 0.703 | -0.038 |

## Biggest regressions

- **What is the size of bioScope corpus?** (`45be665a4504f0c7f458cf3f75a95d5a75eefd42`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: BF: 2670 sentences, BA: 11871 sentences
  - top context: [Introduction] This task was the CoNLL-2010 Shared Task (BIBREF0), which had 3 different subtasks. Task 1B was speculation cue detection on the BioScope Corpus, Task 1W was weasel identification from Wikipedia articles, and Task 2 was speculation scope resolution from the BioScope Corpus. For each t

- **Is fine-tuning required to incorporate these embeddings into existing models?** (`b06512c17d99f9339ffdab12cedbc63501ff527e`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: The only requirement is that the model accepts as input, an embedding layer (for entities and relations). If a model fulfills this requirement [...] we can just use Dolores embeddings as a drop-in replacement.
  - top context: [Introduction] Knowledge Embeddings learned using Dolores can easily be used as input representations for predictive models on knowledge graphs. More importantly, when existing predictive models use input representations for entities and relations, we can easily replace those representations with Do

- **did they compare with other extractive summarization methods?** (`c2cb6c4500d9e02fc9a1bdffd22c3df69655189f`, answer_f1 1.00 -> 0.00)
  - gold: No
  - predicted: Unanswerable
  - top context: [Related Work] In the broader field of summarization, automization was meant to standardize output while also saving time and effort. Pioneering strategies in summarization started by extracting "significant" sentences in the whole corpus to build an abstract where "significant" sentences were defin

- **what evaluation metrics are reported?** (`d9412dda3279729e95fcb35cbed09e61577a896e`, answer_f1 1.00 -> 0.00)
  - gold: precision, recall, F1 and accuracy | precision, recall, F1 , accuracy 
  - predicted: P/R/F1, Acc, MaxFreq
  - top context: [Case Study] We visualize the heat maps of attention results. As shown in Fig. FIGREF47 , deeper background color represents larger attention score. The attention score is calculated with Eq. ( EQREF15 ). We take the average of the resulting INLINEFORM0 attention matrix over the time dimension to ob

- **what dataset was used?** (`e48e750743aef36529fbea4328b8253dbe928b4d`, answer_f1 1.00 -> 0.00)
  - gold: WASSA-2017 Shared Task on Emotion Intensity
  - predicted: dev data set BIBREF19
  - top context: [Parameter Optimization] In order to find the optimal parameter values for the EmoInt system, an extensive grid search was performed through the scikit-Learn framework over all subsets of the training set (shuffled), using stratified 10-fold cross validation and optimizing the Pearson's Correlation 
