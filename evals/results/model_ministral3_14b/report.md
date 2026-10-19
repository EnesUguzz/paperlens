# Run `model_ministral3_14b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.model` | qwen3:8b | ministral-3:14b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.419 | 0.403 | +0.015 |
| answer_f1.abstractive | 0.225 | 0.239 | -0.013 |
| answer_f1.extractive | 0.473 | 0.461 | +0.012 |
| answer_f1.unanswerable | 0.458 | 0.318 | +0.140 |
| answer_f1.yes_no | 0.486 | 0.517 | -0.031 |
| completion_tokens_mean | 51.800 | 43.948 | +7.852 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.297 | 0.265 | +0.032 |
| false_refusal_rate | 0.042 | 0.047 | -0.005 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 1151.187 | 684.335 | +466.852 |
| latency_p95_ms | 2052.496 | 972.827 | +1079.669 |
| n.abstractive | 55.000 | 58.000 | -3.000 |
| n.extractive | 144.000 | 141.000 | +3.000 |
| n.unanswerable | 24.000 | 22.000 | +2.000 |
| n.yes_no | 27.000 | 29.000 | -2.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2577.000 | 2042.000 | +535.000 |
| prompt_tokens_mean | 2228.868 | 1676.652 | +552.216 |
| refusal_recall | 0.270 | 0.162 | +0.108 |
| retrieval.mrr | 0.552 | 0.552 | +0.000 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.409 | 0.409 | +0.000 |
| retrieval.ndcg_at_3 | 0.560 | 0.560 | +0.000 |
| retrieval.ndcg_at_5 | 0.649 | 0.649 | +0.000 |
| retrieval.recall_at_1 | 0.335 | 0.335 | +0.000 |
| retrieval.recall_at_3 | 0.603 | 0.603 | +0.000 |
| retrieval.recall_at_5 | 0.703 | 0.703 | +0.000 |

## Biggest regressions

- **What evaluation metric is used?** (`0ab3df10f0b7203e859e9b62ffa7d6d79ffbbe50`, answer_f1 1.00 -> 0.00)
  - gold: average classification accuracy | average classification accuracy, execution performance
  - predicted: classifying a piece of text by its specific language label
  - top context: [Methodology] winning language is at least one more than the number of words considered to be from the language scored in second place. The stacked classifier is tested against three public LID implementations BIBREF17, BIBREF23, BIBREF8. The LID implementation described in BIBREF17 is available on 

- **Did they used dataset from another domain for evaluation?** (`1dac4bc5af239024566fcb0f43bb9ff1c248ecec`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: Yes
  - top context: [Results] Pointer-Gen+ARL-SEN achieve comparable fluency performance to Pointer-Gen and Pointer-Gen+RL-ROUGE. Test set headlines achieve the best performance among all models, but the difference is not statistically significant. Also, we observe that fine-tuning on sensational headlines will hurt th

- **Do the authors mention any downside of lemmatizing input before training ELMo?** (`5aa12b4063d6182a71870c98e4e1815ff3dc8a72`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: `lemmatization brings along both advantages and disadvantages for WSD with ELMo`
  - top context: [Introduction] if the field of applying deep learning to NLP tasks can be summarised as follows: as minimal pre-processing as possible. It is widely believed that lemmatization or other text input normalisation is not necessary. Advanced neural architectures based on character input (CNNs, BPE, etc)

- **Are the answers double (and not triple) annotated?** (`c70bafc35e27be9d1efae60596bc0dd390c124c0`, answer_f1 1.00 -> 0.00)
  - gold: Yes
  - predicted: every question in the test set is answered by at least two additional experts
  - top context: [Data Collection ::: Analysis] mutually exclusive. Every question is answered by at least one expert. In addition, in order to estimate annotation reliability and provide for better evaluation, every question in the test set is answered by at least two additional experts. Table TABREF14 describes th

- **Does their framework automatically optimize for hyperparameters?** (`de0b650022ad8693465242ded169313419eed7d9`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: No
  - top context: [Experimental Settings] BLEU BIBREF14 is conducted on a development set to decide the early-stopping condition. This evaluation script has also being used to choose the model archiving the best BLEU on the development set instead of the maximal loglikelihood between the translations and target sente
