# Run `test_m2_agent`

## Changed parameters vs `test_m1b_best`

| Parameter | Reference | This run |
|---|---|---|
| `conversations.path` | evals/conversations/m2.jsonl | evals/conversations/m2_test.jsonl |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.363 | 0.458 | -0.095 |
| answer_f1.abstractive | 0.092 | 0.168 | -0.075 |
| answer_f1.extractive | 0.407 | 0.551 | -0.145 |
| answer_f1.gold.abstractive | 0.162 | 0.317 | -0.155 |
| answer_f1.gold.extractive | 0.370 | 0.504 | -0.134 |
| answer_f1.gold.unanswerable | 0.637 | 0.557 | +0.080 |
| answer_f1.gold.yes_no | 0.373 | 0.370 | +0.003 |
| answer_f1.no_clarify | 0.346 | - | - |
| answer_f1.turn.first | 0.410 | - | - |
| answer_f1.turn.same | 0.457 | - | - |
| answer_f1.turn.switch_signaled | 0.399 | - | - |
| answer_f1.turn.switch_unsignaled | 0.204 | - | - |
| answer_f1.unanswerable | 0.690 | 0.630 | +0.060 |
| answer_f1.yes_no | 0.385 | 0.357 | +0.027 |
| answer_words_mean | 12.204 | 12.692 | -0.488 |
| completion_tokens_mean | 46.844 | 47.960 | -1.116 |
| condense.changed_rate | 0.032 | - | - |
| context_words_mean | 699.600 | 692.988 | +6.612 |
| data.questions_dropped | 0.000 | 0.000 | +0.000 |
| data.unmatched_evidence_rate | 0.039 | 0.039 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.395 | 0.530 | -0.135 |
| evidence_f1.no_clarify | 0.359 | - | - |
| evidence_f1_ceiling | 0.547 | 0.741 | -0.194 |
| false_refusal_rate | 0.171 | 0.088 | +0.083 |
| gold_recall | 0.415 | 0.523 | -0.108 |
| index.num_chunks | 12142.000 | 12142.000 | +0.000 |
| latency_p50_ms | 1019.289 | 986.683 | +32.606 |
| latency_p95_ms | 2006.031 | 1929.805 | +76.227 |
| llm_calls_mean | 1.040 | - | - |
| n.abstractive | 59.000 | 52.000 | +7.000 |
| n.extractive | 136.000 | 143.000 | -7.000 |
| n.gold.abstractive | 51.000 | 51.000 | +0.000 |
| n.gold.extractive | 139.000 | 139.000 | +0.000 |
| n.gold.unanswerable | 33.000 | 33.000 | +0.000 |
| n.gold.yes_no | 27.000 | 27.000 | +0.000 |
| n.no_clarify | 224.000 | - | - |
| n.turn.first | 65.000 | - | - |
| n.turn.same | 40.000 | - | - |
| n.turn.switch_signaled | 83.000 | - | - |
| n.turn.switch_unsignaled | 62.000 | - | - |
| n.unanswerable | 29.000 | 27.000 | +2.000 |
| n.yes_no | 26.000 | 28.000 | -2.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 3065.000 | 2332.000 | +733.000 |
| prompt_tokens_mean | 1831.276 | 1785.672 | +45.604 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.606 | 0.515 | +0.091 |
| retrieval.doc_hit_rate | 0.748 | 1.000 | -0.252 |
| retrieval.mrr | 0.290 | 0.503 | -0.212 |
| retrieval.n | 226.000 | 226.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.181 | 0.376 | -0.195 |
| retrieval.ndcg_at_3 | 0.277 | 0.475 | -0.198 |
| retrieval.ndcg_at_5 | 0.317 | 0.510 | -0.192 |
| retrieval.recall_at_1 | 0.167 | 0.345 | -0.179 |
| retrieval.recall_at_3 | 0.351 | 0.569 | -0.219 |
| retrieval.recall_at_5 | 0.449 | 0.660 | -0.212 |
| route.clarify_rate | 0.104 | - | - |
| route.paper_accuracy | 0.748 | - | - |
| route.paper_accuracy.first | 0.969 | - | - |
| route.paper_accuracy.same | 0.825 | - | - |
| route.paper_accuracy.switch_signaled | 0.988 | - | - |
| route.paper_accuracy.switch_unsignaled | 0.145 | - | - |
| route.signaled_catch | 0.988 | - | - |
| route.signaled_catch.t0 | 1.000 | - | - |
| route.signaled_catch.t1 | 0.955 | - | - |
| route.signaled_catch.t2 | 1.000 | - | - |
| route.unsignaled_catch | 0.145 | - | - |
| route.wrong_paper_rate | 0.160 | - | - |

## Biggest regressions

- **What is the baseline?** (`c25014b7e57bb2949138d64d49f356d69838bc25`, answer_f1 1.00 -> 0.00)
  - gold: rule-based unigram matching baseline | a rule-based unigram matching baseline, which takes the entry with highest unigram overlap with the query string to be the top match
  - predicted: Unanswerable
  - top context: [Abstract] In this paper, we present a real-world conversational AI system to search for and book hotels through text messaging. Our architecture consists of a frame-based dialogue management system, which calls machine learning models for intent classification, named entity recognition, and informa

- **Is the proposed system compared to existing systems?** (`b8bbdc3987bb456739544426c6037c78ede01b77`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: Unanswerable
  - top context: [Conclusion and Future Work] We have presented a spoken conversational question answering system, in French. The DS orchestrates different QA systems and returns the response with the higher confidence score. The system contains modules specifically designed for dealing with common spoken conversati

- **What language is used for the experiments?** (`fef5b65263c81299acc350a101dabaf5a8cb9c6e`, answer_f1 1.00 -> 0.00)
  - gold: English | English | English
  - predicted: Unanswerable
  - top context: [Introduction] This paper proposes a two-stage framework of phonetic-and-semantic embedding for spoken words. Stage 1 performs phonetic embedding but with speaker characteristics disentangled using separate phonetic and speaker encoders and a speaker discriminator. Stage 2 then performs semantic emb

- **Do they compare against a system that does not use streaming text, but has the entire text at disposal?** (`ea6edf45f094586caf4684463287254d44b00e95`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: No
  - top context: [Abstract] approximation of the two models can be used when longer texts need to be generated. Experiments with three ontologies confirm that the proposed models lead to more compact texts, compared to pipeline systems, with no deterioration or with improvements in the perceived quality of the gener

- **Are experiments conducted on multiple datasets?** (`55bafa0f7394163f4afd1d73340aac94c2d9f36c`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: Unanswerable
  - top context: [Performance on held-out event types] To assess the robustness of the veridicality classifier when applied to new types of events, we compared its performance when trained on all events vs. holding out one category for testing. Table TABREF37 shows the comparison: the second and third columns give F
