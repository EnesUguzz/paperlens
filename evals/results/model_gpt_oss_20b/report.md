# Run `model_gpt_oss_20b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.extra_body.reasoning_effort` | none | low |
| `llm.model` | qwen3:8b | gpt-oss:20b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.421 | 0.403 | +0.017 |
| answer_f1.abstractive | 0.184 | 0.239 | -0.054 |
| answer_f1.extractive | 0.457 | 0.461 | -0.004 |
| answer_f1.unanswerable | 0.783 | 0.318 | +0.464 |
| answer_f1.yes_no | 0.448 | 0.517 | -0.069 |
| completion_tokens_mean | 116.168 | 43.948 | +72.220 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.307 | 0.265 | +0.043 |
| false_refusal_rate | 0.160 | 0.047 | +0.113 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 2159.621 | 684.335 | +1475.286 |
| latency_p95_ms | 3081.897 | 972.827 | +2109.070 |
| n.abstractive | 60.000 | 58.000 | +2.000 |
| n.extractive | 138.000 | 141.000 | -3.000 |
| n.unanswerable | 23.000 | 22.000 | +1.000 |
| n.yes_no | 29.000 | 29.000 | +0.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 3596.000 | 2042.000 | +1554.000 |
| prompt_tokens_mean | 1699.700 | 1676.652 | +23.048 |
| refusal_recall | 0.459 | 0.162 | +0.297 |
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
  - predicted: Yes
  - top context: [Results] Pointer-Gen+ARL-SEN achieve comparable fluency performance to Pointer-Gen and Pointer-Gen+RL-ROUGE. Test set headlines achieve the best performance among all models, but the difference is not statistically significant. Also, we observe that fine-tuning on sensational headlines will hurt th

- **Do the authors mention any downside of lemmatizing input before training ELMo?** (`5aa12b4063d6182a71870c98e4e1815ff3dc8a72`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: decreased score
  - top context: [Introduction] if the field of applying deep learning to NLP tasks can be summarised as follows: as minimal pre-processing as possible. It is widely believed that lemmatization or other text input normalisation is not necessary. Advanced neural architectures based on character input (CNNs, BPE, etc)

- **Have they made any attempt to correct MRC gold standards according to their findings? ** (`5c88d601e8fca96bffebfa9ef22331ecf31c6d75`, answer_f1 1.00 -> 0.00)
  - gold: Yes | No
  - predicted: Unanswerable
  - top context: [Framework for MRC Gold Standard Analysis ::: Dimensions of Interest ::: Complexity] full catalogue of features, their description, detailed annotation guideline as well as illustrating examples can be found in Appendix . We select contemporary MRC benchmarks to represent all four commonly used prob

- **did they compare with other extractive summarization methods?** (`c2cb6c4500d9e02fc9a1bdffd22c3df69655189f`, answer_f1 1.00 -> 0.00)
  - gold: No
  - predicted: Unanswerable
  - top context: [Introduction] aims are critical steps in ultimately developing methods that can automate discharge summary creation. In the broader field of summarization, automization was meant to standardize output while also saving time and effort. Pioneering strategies in summarization started by extracting "s

- **Are the answers double (and not triple) annotated?** (`c70bafc35e27be9d1efae60596bc0dd390c124c0`, answer_f1 1.00 -> 0.00)
  - gold: Yes
  - predicted: Unanswerable
  - top context: [Data Collection ::: Analysis] mutually exclusive. Every question is answered by at least one expert. In addition, in order to estimate annotation reliability and provide for better evaluation, every question in the test set is answered by at least two additional experts. Table TABREF14 describes th
