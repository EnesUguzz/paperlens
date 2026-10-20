# Run `model_qwen3_5_9b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.extra_body` | {} | None |
| `llm.extra_body.reasoning_effort` | None | none |
| `llm.model` | ministral-3:14b | qwen3.5:9b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.381 | 0.419 | -0.038 |
| answer_f1.abstractive | 0.223 | 0.225 | -0.002 |
| answer_f1.extractive | 0.467 | 0.473 | -0.006 |
| answer_f1.gold.abstractive | 0.295 | 0.302 | -0.008 |
| answer_f1.gold.extractive | 0.449 | 0.464 | -0.015 |
| answer_f1.gold.unanswerable | 0.504 | 0.411 | +0.093 |
| answer_f1.gold.yes_no | 0.025 | 0.464 | -0.439 |
| answer_f1.unanswerable | 0.593 | 0.458 | +0.134 |
| answer_f1.yes_no | 0.062 | 0.486 | -0.425 |
| answer_words_mean | 10.800 | 15.492 | -4.692 |
| completion_tokens_mean | 37.748 | 51.800 | -14.052 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.306 | 0.297 | +0.009 |
| evidence_f1_ceiling | 0.524 | 0.524 | +0.000 |
| false_refusal_rate | 0.061 | 0.042 | +0.019 |
| gold_recall | 0.427 | 0.494 | -0.067 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| judge.accuracy | 0.460 | 0.480 | -0.020 |
| judge.accuracy.gold.abstractive | 0.328 | 0.310 | +0.017 |
| judge.accuracy.gold.extractive | 0.611 | 0.595 | +0.015 |
| judge.accuracy.gold.unanswerable | 0.432 | 0.351 | +0.081 |
| judge.accuracy.gold.yes_no | 0.000 | 0.458 | -0.458 |
| judge.agreement_on_high_f1 | 0.947 | 1.000 | -0.053 |
| judge.error_rate | 0.000 | 0.000 | +0.000 |
| judge.llm_calls | 155.000 | 163.000 | -8.000 |
| latency_p50_ms | 677.352 | 1042.795 | -365.443 |
| latency_p95_ms | 970.125 | 1850.713 | -880.589 |
| n.abstractive | 59.000 | 55.000 | +4.000 |
| n.extractive | 138.000 | 144.000 | -6.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 27.000 | 24.000 | +3.000 |
| n.yes_no | 26.000 | 27.000 | -1.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2025.000 | 2577.000 | -552.000 |
| prompt_tokens_mean | 1674.264 | 2228.868 | -554.604 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.405 | 0.270 | +0.135 |
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

- **Do they use attention?** (`45e9533586199bde19313cd43b3d0ecadcaf7a33`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: The attention function BIBREF11 is used to compute the similarity score between passages and questions as: INLINEFORM2
  - top context: [Model] layer, and also with the output of the first contextual encoding layer as the input of its second encoding layer. Thus, we obtain the final representation of the contextual encoding layer by a concatenation of the outputs of two BiLSTM: INLINEFORM0 for questions and INLINEFORM1 for passages.

- **Do they experiment with the toolkits?** (`5f2bade0881c719ab026bc2e2962e2ada96cdb25`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: Specifically, we evaluate popular or state-of-the-art models on standard benchmark data sets.
  - top context: [Abstract] We present GluonCV and GluonNLP, the deep learning toolkits for computer vision and natural language processing based on Apache MXNet (incubating). These toolkits provide state-of-the-art pre-trained models, training scripts, and training logs, to facilitate rapid prototyping and promote 

- **Did they pefrorm any cross-lingual vs single language evaluation?** (`612c3675b6c55b60ae6d24265ed8e20f62cb117e`, answer_f1 1.00 -> 0.00)
  - gold: Yes
  - predicted: To test whether horizontal or vertical consistency is more helpful, we train and evaluate M-Bert on a dataset variant where all questions are in their English version.
  - top context: [Baseline Experiments ::: How Important is Consistent Language?] the questions are consistently presented in one language, regardless of the comment. To test whether horizontal or vertical consistency is more helpful, we train and evaluate M-Bert on a dataset variant where all questions are in their

- **Do they report results only on English datasets?** (`8bb0011ad1d63996d5650770f3be18abdd9f7fc6`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Yes
  - predicted: No, they report results on English datasets but also mention plans to use larger knowledge bases like ConceptNet and Freebase in the future.
  - top context: [Data Enrichment Method] WordNet-based data enrichment method, which is aimed at extracting inter-word semantic connections from each passage-question pair in our MRC dataset. The extraction is performed in a controllable manner, and the extracted results are provided as general knowledge to our MRC

- **Is fine-tuning required to incorporate these embeddings into existing models?** (`b06512c17d99f9339ffdab12cedbc63501ff527e`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: The only requirement is that the model accepts as input, an embedding layer (for entities and relations).
  - top context: [Dolores: Learner] incorporate our learned Dolores embeddings into existing predictive models on knowledge graphs. The only requirement is that the model accepts as input, an embedding layer (for entities and relations). If a model fulfills this requirement (which a large number of neural models on 
