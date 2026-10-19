# Run `model_gemma4_12b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.model` | qwen3:8b | gemma4:12b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.421 | 0.403 | +0.018 |
| answer_f1.abstractive | 0.214 | 0.239 | -0.025 |
| answer_f1.extractive | 0.486 | 0.461 | +0.025 |
| answer_f1.unanswerable | 0.759 | 0.318 | +0.440 |
| answer_f1.yes_no | 0.128 | 0.517 | -0.389 |
| completion_tokens_mean | 41.548 | 43.948 | -2.400 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.316 | 0.265 | +0.051 |
| false_refusal_rate | 0.113 | 0.047 | +0.066 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 1071.553 | 684.335 | +387.218 |
| latency_p95_ms | 1288.133 | 972.827 | +315.306 |
| n.abstractive | 56.000 | 58.000 | -2.000 |
| n.extractive | 140.000 | 141.000 | -1.000 |
| n.unanswerable | 29.000 | 22.000 | +7.000 |
| n.yes_no | 25.000 | 29.000 | -4.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2008.000 | 2042.000 | -34.000 |
| prompt_tokens_mean | 1670.444 | 1676.652 | -6.208 |
| refusal_recall | 0.568 | 0.162 | +0.405 |
| retrieval.mrr | 0.552 | 0.552 | +0.000 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.409 | 0.409 | +0.000 |
| retrieval.ndcg_at_3 | 0.560 | 0.560 | +0.000 |
| retrieval.ndcg_at_5 | 0.649 | 0.649 | +0.000 |
| retrieval.recall_at_1 | 0.335 | 0.335 | +0.000 |
| retrieval.recall_at_3 | 0.603 | 0.603 | +0.000 |
| retrieval.recall_at_5 | 0.703 | 0.703 | +0.000 |

## Biggest regressions

- **Did they used dataset from another domain for evaluation?** (`1dac4bc5af239024566fcb0f43bb9ff1c248ecec`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: LCSTS dataset
  - top context: [Results] Pointer-Gen+ARL-SEN achieve comparable fluency performance to Pointer-Gen and Pointer-Gen+RL-ROUGE. Test set headlines achieve the best performance among all models, but the difference is not statistically significant. Also, we observe that fine-tuning on sensational headlines will hurt th

- **Do the authors mention any downside of lemmatizing input before training ELMo?** (`5aa12b4063d6182a71870c98e4e1815ff3dc8a72`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: Lemmatization can lead to decreased scores for atypical non-changeable words.
  - top context: [Introduction] if the field of applying deep learning to NLP tasks can be summarised as follows: as minimal pre-processing as possible. It is widely believed that lemmatization or other text input normalisation is not necessary. Advanced neural architectures based on character input (CNNs, BPE, etc)

- **Have they made any attempt to correct MRC gold standards according to their findings? ** (`5c88d601e8fca96bffebfa9ef22331ecf31c6d75`, answer_f1 1.00 -> 0.00)
  - gold: Yes | No
  - predicted: The authors state that studying how to introduce semantics-altering grammatical modifiers into gold standards is a future research goal.
  - top context: [Framework for MRC Gold Standard Analysis ::: Dimensions of Interest ::: Complexity] full catalogue of features, their description, detailed annotation guideline as well as illustrating examples can be found in Appendix . We select contemporary MRC benchmarks to represent all four commonly used prob

- **Did they pefrorm any cross-lingual vs single language evaluation?** (`612c3675b6c55b60ae6d24265ed8e20f62cb117e`, answer_f1 1.00 -> 0.00)
  - gold: Yes
  - predicted: we train and evaluate M-Bert on a dataset variant where all questions are in their English version
  - top context: [Baseline Experiments ::: How Important is Consistent Language?] the questions are consistently presented in one language, regardless of the comment. To test whether horizontal or vertical consistency is more helpful, we train and evaluate M-Bert on a dataset variant where all questions are in their

- **Does the model have attention?** (`b9d168da5321a7d7b812c52bb102a05210fe45bd`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: attention layer size is 100
  - top context: [Baseline] few details regarding model size and training schedule: the number of LSTM layers is one; embedding size, LSTM layer size and attention layer size is 100; models are trained for 20 epochs; on every epoch, training data is subsampled at a rate of 0.3; LSTM dropout is applied at a rate 0.3;
