# Run `model_qwen3_14b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.model` | qwen3:8b | qwen3:14b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.397 | 0.403 | -0.006 |
| answer_f1.abstractive | 0.210 | 0.239 | -0.029 |
| answer_f1.extractive | 0.472 | 0.461 | +0.011 |
| answer_f1.unanswerable | 0.560 | 0.318 | +0.242 |
| answer_f1.yes_no | 0.254 | 0.517 | -0.263 |
| completion_tokens_mean | 38.796 | 43.948 | -5.152 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.288 | 0.265 | +0.023 |
| false_refusal_rate | 0.108 | 0.047 | +0.061 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 1097.182 | 684.335 | +412.847 |
| latency_p95_ms | 2549.161 | 972.827 | +1576.334 |
| n.abstractive | 57.000 | 58.000 | -1.000 |
| n.extractive | 140.000 | 141.000 | -1.000 |
| n.unanswerable | 25.000 | 22.000 | +3.000 |
| n.yes_no | 28.000 | 29.000 | -1.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2042.000 | 2042.000 | +0.000 |
| prompt_tokens_mean | 1676.652 | 1676.652 | +0.000 |
| refusal_recall | 0.351 | 0.162 | +0.189 |
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
  - predicted: test split of LCSTS dataset
  - top context: [Results] Pointer-Gen+ARL-SEN achieve comparable fluency performance to Pointer-Gen and Pointer-Gen+RL-ROUGE. Test set headlines achieve the best performance among all models, but the difference is not statistically significant. Also, we observe that fine-tuning on sensational headlines will hurt th

- **Do the authors mention any downside of lemmatizing input before training ELMo?** (`5aa12b4063d6182a71870c98e4e1815ff3dc8a72`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: lemmatization, and this leads to the decreased score.
  - top context: [Introduction] if the field of applying deep learning to NLP tasks can be summarised as follows: as minimal pre-processing as possible. It is widely believed that lemmatization or other text input normalisation is not necessary. Advanced neural architectures based on character input (CNNs, BPE, etc)

- **Do they experiment with the toolkits?** (`5f2bade0881c719ab026bc2e2962e2ada96cdb25`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: we evaluate popular or state-of-the-art models on standard benchmark data sets.
  - top context: [Abstract] We present GluonCV and GluonNLP, the deep learning toolkits for computer vision and natural language processing based on Apache MXNet (incubating). These toolkits provide state-of-the-art pre-trained models, training scripts, and training logs, to facilitate rapid prototyping and promote 

- **Did they pefrorm any cross-lingual vs single language evaluation?** (`612c3675b6c55b60ae6d24265ed8e20f62cb117e`, answer_f1 1.00 -> 0.00)
  - gold: Yes
  - predicted: the questions are consistently presented in one language, regardless of the comment. To test whether horizontal or vertical consistency is more helpful, we train and evaluate M-Bert on a dataset variant where all questions are in their English version.
  - top context: [Baseline Experiments ::: How Important is Consistent Language?] the questions are consistently presented in one language, regardless of the comment. To test whether horizontal or vertical consistency is more helpful, we train and evaluate M-Bert on a dataset variant where all questions are in their

- **Is fine-tuning required to incorporate these embeddings into existing models?** (`b06512c17d99f9339ffdab12cedbc63501ff527e`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: the only requirement is that the model accepts as input, an embedding layer (for entities and relations). If a model fulfills this requirement (which a large number of neural models on knowledge graphs do), we can just use Dolores embeddings as a drop-in replacement.
  - top context: [Dolores: Learner] incorporate our learned Dolores embeddings into existing predictive models on knowledge graphs. The only requirement is that the model accepts as input, an embedding layer (for entities and relations). If a model fulfills this requirement (which a large number of neural models on 
