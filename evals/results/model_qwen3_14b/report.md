# Run `model_qwen3_14b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.extra_body` | {} | None |
| `llm.extra_body.reasoning_effort` | None | none |
| `llm.model` | ministral-3:14b | qwen3:14b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.397 | 0.419 | -0.022 |
| answer_f1.abstractive | 0.210 | 0.225 | -0.015 |
| answer_f1.extractive | 0.472 | 0.473 | -0.001 |
| answer_f1.gold.abstractive | 0.284 | 0.302 | -0.019 |
| answer_f1.gold.extractive | 0.458 | 0.464 | -0.006 |
| answer_f1.gold.unanswerable | 0.503 | 0.411 | +0.092 |
| answer_f1.gold.yes_no | 0.172 | 0.464 | -0.292 |
| answer_f1.unanswerable | 0.560 | 0.458 | +0.102 |
| answer_f1.yes_no | 0.254 | 0.486 | -0.232 |
| completion_tokens_mean | 38.796 | 51.800 | -13.004 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.288 | 0.297 | -0.009 |
| evidence_f1_ceiling | 0.524 | 0.524 | +0.000 |
| false_refusal_rate | 0.108 | 0.042 | +0.066 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 1097.182 | 1042.795 | +54.388 |
| latency_p95_ms | 2549.161 | 1850.713 | +698.447 |
| n.abstractive | 57.000 | 55.000 | +2.000 |
| n.extractive | 140.000 | 144.000 | -4.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 25.000 | 24.000 | +1.000 |
| n.yes_no | 28.000 | 27.000 | +1.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2042.000 | 2577.000 | -535.000 |
| prompt_tokens_mean | 1676.652 | 2228.868 | -552.216 |
| refusal_recall | 0.351 | 0.270 | +0.081 |
| retrieval.mrr | 0.552 | 0.552 | +0.000 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.409 | 0.409 | +0.000 |
| retrieval.ndcg_at_3 | 0.492 | 0.492 | +0.000 |
| retrieval.ndcg_at_5 | 0.527 | 0.527 | +0.000 |
| retrieval.recall_at_1 | 0.335 | 0.335 | +0.000 |
| retrieval.recall_at_3 | 0.603 | 0.603 | +0.000 |
| retrieval.recall_at_5 | 0.703 | 0.703 | +0.000 |

## Biggest regressions

- **Do they experiment with the toolkits?** (`5f2bade0881c719ab026bc2e2962e2ada96cdb25`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: we evaluate popular or state-of-the-art models on standard benchmark data sets.
  - top context: [Abstract] We present GluonCV and GluonNLP, the deep learning toolkits for computer vision and natural language processing based on Apache MXNet (incubating). These toolkits provide state-of-the-art pre-trained models, training scripts, and training logs, to facilitate rapid prototyping and promote 

- **Did they pefrorm any cross-lingual vs single language evaluation?** (`612c3675b6c55b60ae6d24265ed8e20f62cb117e`, answer_f1 1.00 -> 0.00)
  - gold: Yes
  - predicted: the questions are consistently presented in one language, regardless of the comment. To test whether horizontal or vertical consistency is more helpful, we train and evaluate M-Bert on a dataset variant where all questions are in their English version.
  - top context: [Baseline Experiments ::: How Important is Consistent Language?] the questions are consistently presented in one language, regardless of the comment. To test whether horizontal or vertical consistency is more helpful, we train and evaluate M-Bert on a dataset variant where all questions are in their

- **Do they inspect their model to see if their model learned to associate image parts with words related to entities?** (`7cd22ca9e107d2b13a7cc94252aaa9007976b338`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: For interpretability, visual tags (label output of InceptionNet) are presented instead of actual feature vectors used.
  - top context: [Figures and Tables] embeddings, character embeddings, and visual features are merged with modality attention. Bi-LSTM/CRF takes as input each token and produces an entity label. Table 1: NER performance on the SnapCaptions dataset with varying modalities (W: word, C: char, V: visual). We report pre

- **Is fine-tuning required to incorporate these embeddings into existing models?** (`b06512c17d99f9339ffdab12cedbc63501ff527e`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: the only requirement is that the model accepts as input, an embedding layer (for entities and relations). If a model fulfills this requirement (which a large number of neural models on knowledge graphs do), we can just use Dolores embeddings as a drop-in replacement.
  - top context: [Dolores: Learner] incorporate our learned Dolores embeddings into existing predictive models on knowledge graphs. The only requirement is that the model accepts as input, an embedding layer (for entities and relations). If a model fulfills this requirement (which a large number of neural models on 

- **Does the model have attention?** (`b9d168da5321a7d7b812c52bb102a05210fe45bd`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: attention layer size is 100
  - top context: [Baseline] few details regarding model size and training schedule: the number of LSTM layers is one; embedding size, LSTM layer size and attention layer size is 100; models are trained for 20 epochs; on every epoch, training data is subsampled at a rate of 0.3; LSTM dropout is applied at a rate 0.3;
