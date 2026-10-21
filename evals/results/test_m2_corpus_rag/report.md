# Run `test_m2_corpus_rag`

## Changed parameters vs `test_m1b_best`

| Parameter | Reference | This run |
|---|---|---|
| `conversations.path` | evals/conversations/m2.jsonl | evals/conversations/m2_test.jsonl |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.245 | 0.458 | -0.213 |
| answer_f1.abstractive | 0.095 | 0.168 | -0.072 |
| answer_f1.extractive | 0.260 | 0.551 | -0.291 |
| answer_f1.gold.abstractive | 0.165 | 0.317 | -0.152 |
| answer_f1.gold.extractive | 0.234 | 0.504 | -0.270 |
| answer_f1.gold.unanswerable | 0.285 | 0.557 | -0.272 |
| answer_f1.gold.yes_no | 0.407 | 0.370 | +0.037 |
| answer_f1.no_clarify | 0.245 | - | - |
| answer_f1.turn.first | 0.186 | - | - |
| answer_f1.turn.same | 0.294 | - | - |
| answer_f1.turn.switch_signaled | 0.283 | - | - |
| answer_f1.turn.switch_unsignaled | 0.226 | - | - |
| answer_f1.unanswerable | 0.345 | 0.630 | -0.285 |
| answer_f1.yes_no | 0.379 | 0.357 | +0.022 |
| answer_words_mean | 12.972 | 12.692 | +0.280 |
| completion_tokens_mean | 50.212 | 47.960 | +2.252 |
| context_words_mean | 652.204 | 692.988 | -40.784 |
| data.questions_dropped | 0.000 | 0.000 | +0.000 |
| data.unmatched_evidence_rate | 0.039 | 0.039 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.203 | 0.530 | -0.327 |
| evidence_f1.no_clarify | 0.203 | - | - |
| evidence_f1_ceiling | 0.379 | 0.741 | -0.362 |
| false_refusal_rate | 0.138 | 0.088 | +0.051 |
| gold_recall | 0.297 | 0.523 | -0.226 |
| index.num_chunks | 12142.000 | 12142.000 | +0.000 |
| latency_p50_ms | 947.724 | 986.683 | -38.959 |
| latency_p95_ms | 2038.718 | 1929.805 | +108.914 |
| llm_calls_mean | 1.000 | - | - |
| n.abstractive | 58.000 | 52.000 | +6.000 |
| n.extractive | 134.000 | 143.000 | -9.000 |
| n.gold.abstractive | 51.000 | 51.000 | +0.000 |
| n.gold.extractive | 139.000 | 139.000 | +0.000 |
| n.gold.unanswerable | 33.000 | 33.000 | +0.000 |
| n.gold.yes_no | 27.000 | 27.000 | +0.000 |
| n.no_clarify | 250.000 | - | - |
| n.turn.first | 65.000 | - | - |
| n.turn.same | 40.000 | - | - |
| n.turn.switch_signaled | 83.000 | - | - |
| n.turn.switch_unsignaled | 62.000 | - | - |
| n.unanswerable | 29.000 | 27.000 | +2.000 |
| n.yes_no | 29.000 | 28.000 | +1.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2428.000 | 2332.000 | +96.000 |
| prompt_tokens_mean | 1736.800 | 1785.672 | -48.872 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.273 | 0.515 | -0.242 |
| retrieval.doc_hit_rate | 0.376 | 1.000 | -0.624 |
| retrieval.mrr | 0.146 | 0.503 | -0.356 |
| retrieval.n | 226.000 | 226.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.084 | 0.376 | -0.292 |
| retrieval.ndcg_at_3 | 0.140 | 0.475 | -0.335 |
| retrieval.ndcg_at_5 | 0.163 | 0.510 | -0.346 |
| retrieval.recall_at_1 | 0.077 | 0.345 | -0.268 |
| retrieval.recall_at_3 | 0.179 | 0.569 | -0.391 |
| retrieval.recall_at_5 | 0.236 | 0.660 | -0.424 |
| route.clarify_rate | 0.000 | - | - |
| route.paper_accuracy | 0.472 | - | - |
| route.paper_accuracy.first | 0.477 | - | - |
| route.paper_accuracy.same | 0.350 | - | - |
| route.paper_accuracy.switch_signaled | 0.747 | - | - |
| route.paper_accuracy.switch_unsignaled | 0.177 | - | - |
| route.signaled_catch | 0.747 | - | - |
| route.signaled_catch.t0 | 0.882 | - | - |
| route.signaled_catch.t1 | 0.500 | - | - |
| route.signaled_catch.t2 | 0.778 | - | - |
| route.unsignaled_catch | 0.177 | - | - |
| route.wrong_paper_rate | 0.444 | - | - |

## Biggest regressions

- **Do they report results only on English data?** (`2f142cd11731d29d0c3fa426e26ef80d997862e0`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Unanswerable | Unanswerable
  - predicted: No
  - top context: [Experiments] We present results of our two models both on parsing (discriminative and generative) and as a language model (generative only) in English and Chinese.

- **How much is the gap between pretraining on SQuAD and not pretraining on SQuAD?** (`0f9678e11079ee9ea1a1ce693f017177dd495ee5`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable | Unanswerable
  - predicted: almost all percentage changes are within a narrow 2% band of no-change (i.e., 100%)
  - top context: [Evaluation ::: Pre-trained Models and Weak Supervision] For each training set, we calculate the relative test quality change (percentage change in F1 or accuracy) of with-BERT over without-BERT. In Figure FIGREF24b, almost all percentage changes are within a narrow 2% band of no-change (i.e., 100%)

- **What is the baseline?** (`c25014b7e57bb2949138d64d49f356d69838bc25`, answer_f1 1.00 -> 0.00)
  - gold: rule-based unigram matching baseline | a rule-based unigram matching baseline, which takes the entry with highest unigram overlap with the query string to be the top match
  - predicted: Unanswerable
  - top context: [Abstract] In this paper, we present a real-world conversational AI system to search for and book hotels through text messaging. Our architecture consists of a frame-based dialogue management system, which calls machine learning models for intent classification, named entity recognition, and informa

- **What is the size of the dataset?** (`35f48b8f73728fbdeb271b170804190b5448485a`, answer_f1 1.00 -> 0.00)
  - gold: 829 instances | 819
  - predicted: Unanswerable
  - top context: [Experiments ::: Datasets] Four different language pairs have been selected for the experiments. The datasets' size varies from tens of thousands to millions of sentences to test the regularizers' ability to improve translation over a range of low-resource and high-resource language pairs. De-En: Th

- **Do they compare to other methods?** (`5181aefb8a7272b4c83a1f7cb61f864ead6a1f1f`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: Yes
  - top context: [Experiments ::: Sentiment Classification ::: Comparisons of different attribute representation and injection methods] To conduct a fair comparison among the different methods described in Section SECREF2, we compare these methods when applied to our base model using the development set of the datas
