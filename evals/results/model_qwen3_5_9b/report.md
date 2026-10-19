# Run `model_qwen3_5_9b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.model` | qwen3:8b | qwen3.5:9b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.381 | 0.403 | -0.023 |
| answer_f1.abstractive | 0.223 | 0.239 | -0.016 |
| answer_f1.extractive | 0.467 | 0.461 | +0.006 |
| answer_f1.unanswerable | 0.593 | 0.318 | +0.274 |
| answer_f1.yes_no | 0.062 | 0.517 | -0.456 |
| completion_tokens_mean | 37.748 | 43.948 | -6.200 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.306 | 0.265 | +0.041 |
| false_refusal_rate | 0.061 | 0.047 | +0.014 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 677.352 | 684.335 | -6.984 |
| latency_p95_ms | 970.125 | 972.827 | -2.702 |
| n.abstractive | 59.000 | 58.000 | +1.000 |
| n.extractive | 138.000 | 141.000 | -3.000 |
| n.unanswerable | 27.000 | 22.000 | +5.000 |
| n.yes_no | 26.000 | 29.000 | -3.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2025.000 | 2042.000 | -17.000 |
| prompt_tokens_mean | 1674.264 | 1676.652 | -2.388 |
| refusal_recall | 0.405 | 0.162 | +0.243 |
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

- **Do they use attention?** (`45e9533586199bde19313cd43b3d0ecadcaf7a33`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: The attention function BIBREF11 is used to compute the similarity score between passages and questions as: INLINEFORM2
  - top context: [Model] layer, and also with the output of the first contextual encoding layer as the input of its second encoding layer. Thus, we obtain the final representation of the contextual encoding layer by a concatenation of the outputs of two BiLSTM: INLINEFORM0 for questions and INLINEFORM1 for passages.

- **Do the authors mention any downside of lemmatizing input before training ELMo?** (`5aa12b4063d6182a71870c98e4e1815ff3dc8a72`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: The authors mention that lemmatization can lead to decreased scores for atypical non-changeable words.
  - top context: [Introduction] if the field of applying deep learning to NLP tasks can be summarised as follows: as minimal pre-processing as possible. It is widely believed that lemmatization or other text input normalisation is not necessary. Advanced neural architectures based on character input (CNNs, BPE, etc)

- **Do they experiment with the toolkits?** (`5f2bade0881c719ab026bc2e2962e2ada96cdb25`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: Specifically, we evaluate popular or state-of-the-art models on standard benchmark data sets.
  - top context: [Abstract] We present GluonCV and GluonNLP, the deep learning toolkits for computer vision and natural language processing based on Apache MXNet (incubating). These toolkits provide state-of-the-art pre-trained models, training scripts, and training logs, to facilitate rapid prototyping and promote 

- **Did they pefrorm any cross-lingual vs single language evaluation?** (`612c3675b6c55b60ae6d24265ed8e20f62cb117e`, answer_f1 1.00 -> 0.00)
  - gold: Yes
  - predicted: To test whether horizontal or vertical consistency is more helpful, we train and evaluate M-Bert on a dataset variant where all questions are in their English version.
  - top context: [Baseline Experiments ::: How Important is Consistent Language?] the questions are consistently presented in one language, regardless of the comment. To test whether horizontal or vertical consistency is more helpful, we train and evaluate M-Bert on a dataset variant where all questions are in their
