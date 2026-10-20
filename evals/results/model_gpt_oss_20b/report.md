# Run `model_gpt_oss_20b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.extra_body` | {} | None |
| `llm.extra_body.reasoning_effort` | None | low |
| `llm.model` | ministral-3:14b | gpt-oss:20b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.421 | 0.419 | +0.002 |
| answer_f1.abstractive | 0.184 | 0.225 | -0.041 |
| answer_f1.extractive | 0.457 | 0.473 | -0.016 |
| answer_f1.gold.abstractive | 0.214 | 0.302 | -0.089 |
| answer_f1.gold.extractive | 0.460 | 0.464 | -0.004 |
| answer_f1.gold.unanswerable | 0.635 | 0.411 | +0.224 |
| answer_f1.gold.yes_no | 0.375 | 0.464 | -0.089 |
| answer_f1.unanswerable | 0.783 | 0.458 | +0.324 |
| answer_f1.yes_no | 0.448 | 0.486 | -0.038 |
| answer_words_mean | 8.240 | 15.492 | -7.252 |
| completion_tokens_mean | 116.168 | 51.800 | +64.368 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.307 | 0.297 | +0.011 |
| evidence_f1_ceiling | 0.524 | 0.524 | +0.000 |
| false_refusal_rate | 0.160 | 0.042 | +0.117 |
| gold_recall | 0.448 | 0.494 | -0.047 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 2159.621 | 1042.795 | +1116.826 |
| latency_p95_ms | 3081.897 | 1850.713 | +1231.184 |
| n.abstractive | 60.000 | 55.000 | +5.000 |
| n.extractive | 138.000 | 144.000 | -6.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 23.000 | 24.000 | -1.000 |
| n.yes_no | 29.000 | 27.000 | +2.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 3596.000 | 2577.000 | +1019.000 |
| prompt_tokens_mean | 1699.700 | 2228.868 | -529.168 |
| refusal_recall | 0.459 | 0.270 | +0.189 |
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

- **What dataset does this approach achieve state of the art results on?** (`1ed49a8c07ef0ac15cfa6b7decbde6604decbd5b`, answer_f1 1.00 -> 0.00)
  - gold: the English-German dataset
  - predicted: Multi30K
  - top context: [Acknowledgments] Fund Institutional Links Grant, ID 352343575) projects. We also thank the annotators for their valuable help. Figure 1: Examples of lexical and gender ambiguity, and inaccurate English description where post-edits (PE) required the image to correct human translation from English (E

- **Have they made any attempt to correct MRC gold standards according to their findings? ** (`5c88d601e8fca96bffebfa9ef22331ecf31c6d75`, answer_f1 1.00 -> 0.00)
  - gold: Yes | No
  - predicted: Unanswerable
  - top context: [Framework for MRC Gold Standard Analysis ::: Dimensions of Interest ::: Complexity] full catalogue of features, their description, detailed annotation guideline as well as illustrating examples can be found in Appendix . We select contemporary MRC benchmarks to represent all four commonly used prob

- **Do they inspect their model to see if their model learned to associate image parts with words related to entities?** (`7cd22ca9e107d2b13a7cc94252aaa9007976b338`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: Unanswerable
  - top context: [Figures and Tables] embeddings, character embeddings, and visual features are merged with modality attention. Bi-LSTM/CRF takes as input each token and produces an entity label. Table 1: NER performance on the SnapCaptions dataset with varying modalities (W: word, C: char, V: visual). We report pre

- **did they compare with other extractive summarization methods?** (`c2cb6c4500d9e02fc9a1bdffd22c3df69655189f`, answer_f1 1.00 -> 0.00)
  - gold: No
  - predicted: Unanswerable
  - top context: [Introduction] aims are critical steps in ultimately developing methods that can automate discharge summary creation. In the broader field of summarization, automization was meant to standardize output while also saving time and effort. Pioneering strategies in summarization started by extracting "s

- **Are the automatically constructed datasets subject to quality control?** (`e97186c51d4af490dba6faaf833d269c8256426c`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: Yes
  - top context: [Discussion and Conclusion] cheaper alternative to recent attempts to build architectures that explicitly incorporate structured knowledge BIBREF55; we see many areas where our inoculation strategy could be improved for such purposes, including having more complex loss functions that manage old and 
