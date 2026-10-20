# Run `model_gemma4_12b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.extra_body` | {} | None |
| `llm.extra_body.reasoning_effort` | None | none |
| `llm.model` | ministral-3:14b | gemma4:12b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.421 | 0.419 | +0.002 |
| answer_f1.abstractive | 0.214 | 0.225 | -0.011 |
| answer_f1.extractive | 0.486 | 0.473 | +0.013 |
| answer_f1.gold.abstractive | 0.296 | 0.302 | -0.006 |
| answer_f1.gold.extractive | 0.469 | 0.464 | +0.004 |
| answer_f1.gold.unanswerable | 0.662 | 0.411 | +0.251 |
| answer_f1.gold.yes_no | 0.092 | 0.464 | -0.372 |
| answer_f1.unanswerable | 0.759 | 0.458 | +0.300 |
| answer_f1.yes_no | 0.128 | 0.486 | -0.358 |
| answer_words_mean | 11.288 | 15.492 | -4.204 |
| completion_tokens_mean | 41.548 | 51.800 | -10.252 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| data.questions_dropped | 0.000 | 0.000 | +0.000 |
| data.unmatched_evidence_rate | 0.046 | 0.046 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.316 | 0.297 | +0.019 |
| evidence_f1_ceiling | 0.524 | 0.524 | +0.000 |
| false_refusal_rate | 0.113 | 0.042 | +0.070 |
| gold_recall | 0.467 | 0.494 | -0.028 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| judge.accuracy | 0.520 | 0.500 | +0.020 |
| judge.accuracy.gold.abstractive | 0.345 | 0.310 | +0.034 |
| judge.accuracy.gold.extractive | 0.611 | 0.595 | +0.015 |
| judge.accuracy.gold.unanswerable | 0.757 | 0.486 | +0.270 |
| judge.accuracy.gold.yes_no | 0.083 | 0.458 | -0.375 |
| judge.agreement_on_high_f1 | 0.913 | 1.000 | -0.087 |
| judge.error_rate | 0.000 | 0.000 | +0.000 |
| judge.llm_calls | 157.000 | 176.000 | -19.000 |
| latency_p50_ms | 1071.553 | 1042.795 | +28.759 |
| latency_p95_ms | 1288.133 | 1850.713 | -562.581 |
| n.abstractive | 56.000 | 55.000 | +1.000 |
| n.extractive | 140.000 | 144.000 | -4.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 29.000 | 24.000 | +5.000 |
| n.yes_no | 25.000 | 27.000 | -2.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2008.000 | 2577.000 | -569.000 |
| prompt_tokens_mean | 1670.444 | 2228.868 | -558.424 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.568 | 0.270 | +0.297 |
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

- **Have they made any attempt to correct MRC gold standards according to their findings? ** (`5c88d601e8fca96bffebfa9ef22331ecf31c6d75`, answer_f1 1.00 -> 0.00)
  - gold: Yes | No
  - predicted: The authors state that studying how to introduce semantics-altering grammatical modifiers into gold standards is a future research goal.
  - top context: [Framework for MRC Gold Standard Analysis ::: Dimensions of Interest ::: Complexity] full catalogue of features, their description, detailed annotation guideline as well as illustrating examples can be found in Appendix . We select contemporary MRC benchmarks to represent all four commonly used prob

- **Did they pefrorm any cross-lingual vs single language evaluation?** (`612c3675b6c55b60ae6d24265ed8e20f62cb117e`, answer_f1 1.00 -> 0.00)
  - gold: Yes
  - predicted: we train and evaluate M-Bert on a dataset variant where all questions are in their English version
  - top context: [Baseline Experiments ::: How Important is Consistent Language?] the questions are consistently presented in one language, regardless of the comment. To test whether horizontal or vertical consistency is more helpful, we train and evaluate M-Bert on a dataset variant where all questions are in their

- **Do they inspect their model to see if their model learned to associate image parts with words related to entities?** (`7cd22ca9e107d2b13a7cc94252aaa9007976b338`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: The authors analyze how the modality attention module amplifies relevant visual contexts and attenuates irrelevant signals to improve entity recognition.
  - top context: [Figures and Tables] embeddings, character embeddings, and visual features are merged with modality attention. Bi-LSTM/CRF takes as input each token and produces an entity label. Table 1: NER performance on the SnapCaptions dataset with varying modalities (W: word, C: char, V: visual). We report pre

- **Does the model have attention?** (`b9d168da5321a7d7b812c52bb102a05210fe45bd`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: attention layer size is 100
  - top context: [Baseline] few details regarding model size and training schedule: the number of LSTM layers is one; embedding size, LSTM layer size and attention layer size is 100; models are trained for 20 epochs; on every epoch, training data is subsampled at a rate of 0.3; LSTM dropout is applied at a rate 0.3;

- **did they compare with other extractive summarization methods?** (`c2cb6c4500d9e02fc9a1bdffd22c3df69655189f`, answer_f1 1.00 -> 0.00)
  - gold: No
  - predicted: Unanswerable
  - top context: [Introduction] aims are critical steps in ultimately developing methods that can automate discharge summary creation. In the broader field of summarization, automization was meant to standardize output while also saving time and effort. Pioneering strategies in summarization started by extracting "s
