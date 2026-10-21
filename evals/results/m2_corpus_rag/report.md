# Run `m2_corpus_rag`

## Changed parameters vs `m1b_best`

| Parameter | Reference | This run |
|---|---|---|
| `agent.clarify.max_candidates` | None | 3 |
| `agent.condense.enabled` | None | True |
| `agent.condense.history_turns` | None | 3 |
| `agent.condense.prompt_template` | None | prompts/condense.md |
| `agent.route.min_lead` | None | 0.3 |
| `agent.route.title_match_min` | None | 90.0 |
| `agent.route.top_k` | None | 10 |
| `agent.tracing` | None | True |
| `conversations.papers_per_conversation` | None | 3 |
| `conversations.path` | None | evals/conversations/m2.jsonl |
| `conversations.return_prob` | None | 0.5 |
| `conversations.seed` | None | 13 |
| `conversations.signal_prob` | None | 0.5 |
| `conversations.signal_templates` | None | ['In the paper "{title}", {question}', 'In {short_title}, {question}', 'About the {short_title} paper: {question}'] |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.216 | 0.440 | -0.225 |
| answer_f1.abstractive | 0.155 | 0.247 | -0.092 |
| answer_f1.extractive | 0.227 | 0.506 | -0.279 |
| answer_f1.gold.abstractive | 0.201 | 0.325 | -0.123 |
| answer_f1.gold.extractive | 0.221 | 0.503 | -0.282 |
| answer_f1.gold.unanswerable | 0.198 | 0.440 | -0.242 |
| answer_f1.gold.yes_no | 0.250 | 0.380 | -0.130 |
| answer_f1.no_clarify | 0.216 | - | - |
| answer_f1.turn.first | 0.200 | - | - |
| answer_f1.turn.same | 0.236 | - | - |
| answer_f1.turn.switch_signaled | 0.304 | - | - |
| answer_f1.turn.switch_unsignaled | 0.131 | - | - |
| answer_f1.unanswerable | 0.292 | 0.500 | -0.208 |
| answer_f1.yes_no | 0.222 | 0.412 | -0.189 |
| answer_words_mean | 16.468 | 14.192 | +2.276 |
| completion_tokens_mean | 56.224 | 49.788 | +6.436 |
| context_words_mean | 670.872 | 689.316 | -18.444 |
| data.questions_dropped | 0.000 | 0.000 | +0.000 |
| data.unmatched_evidence_rate | 0.046 | 0.046 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.181 | 0.491 | -0.311 |
| evidence_f1.no_clarify | 0.181 | - | - |
| evidence_f1_ceiling | 0.402 | 0.770 | -0.368 |
| false_refusal_rate | 0.080 | 0.061 | +0.019 |
| gold_recall | 0.255 | 0.507 | -0.251 |
| index.num_chunks | 8360.000 | 8360.000 | +0.000 |
| latency_p50_ms | 988.826 | 963.592 | +25.234 |
| latency_p95_ms | 1862.520 | 1642.370 | +220.149 |
| llm_calls_mean | 1.000 | - | - |
| n.abstractive | 59.000 | 53.000 | +6.000 |
| n.extractive | 140.000 | 142.000 | -2.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.no_clarify | 250.000 | - | - |
| n.turn.first | 57.000 | - | - |
| n.turn.same | 54.000 | - | - |
| n.turn.switch_signaled | 67.000 | - | - |
| n.turn.switch_unsignaled | 72.000 | - | - |
| n.unanswerable | 24.000 | 28.000 | -4.000 |
| n.yes_no | 27.000 | 27.000 | +0.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2285.000 | 2417.000 | -132.000 |
| prompt_tokens_mean | 1759.548 | 1788.992 | -29.444 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.162 | 0.351 | -0.189 |
| retrieval.doc_hit_rate | 0.366 | 1.000 | -0.634 |
| retrieval.mrr | 0.151 | 0.484 | -0.333 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.083 | 0.322 | -0.239 |
| retrieval.ndcg_at_3 | 0.129 | 0.423 | -0.294 |
| retrieval.ndcg_at_5 | 0.160 | 0.474 | -0.314 |
| retrieval.recall_at_1 | 0.074 | 0.264 | -0.191 |
| retrieval.recall_at_3 | 0.167 | 0.524 | -0.357 |
| retrieval.recall_at_5 | 0.242 | 0.665 | -0.423 |
| route.clarify_rate | 0.000 | - | - |
| route.paper_accuracy | 0.456 | - | - |
| route.paper_accuracy.first | 0.596 | - | - |
| route.paper_accuracy.same | 0.259 | - | - |
| route.paper_accuracy.switch_signaled | 0.687 | - | - |
| route.paper_accuracy.switch_unsignaled | 0.278 | - | - |
| route.signaled_catch | 0.687 | - | - |
| route.signaled_catch.t0 | 0.808 | - | - |
| route.signaled_catch.t1 | 0.700 | - | - |
| route.signaled_catch.t2 | 0.524 | - | - |
| route.unsignaled_catch | 0.278 | - | - |
| route.wrong_paper_rate | 0.492 | - | - |

## Biggest regressions

- **Did they use a crowdsourcing platform for annotations?** (`0ba3ea93eef5660a79ea3c26c6a270eac32dfa4c`, answer_f1 1.00 -> 0.00)
  - gold: No | Unanswerable
  - predicted: Yes
  - top context: [Annotation] The final annotations can be collected using a crowdsourcing platform, a smaller number of highly-trained annotators, or a group of experts. Which type of annotator to use should be informed by the complexity and specificity of the concept. For more complex concepts, highly-trained or e

- **What evaluation metrics were used for the summarization task?** (`aa6d956c2860f58fc9baea74c353c9d985b05605`, answer_f1 1.00 -> 0.00)
  - gold: ROUGE BIBREF22 unigram score | ROUGE
  - predicted: Pyramid evaluation framework, ROUGE-2
  - top context: [Annotations] In the TAC 2014 summarization track, Rouge was suggested as the evaluation metric for summarization and no human assessment was provided for the topics. Therefore, to study the effectiveness of the evaluation metrics, we use the semi-manual Pyramid evaluation framework BIBREF7 , BIBREF

- **Is WordNet useful for taxonomic reasoning for this task?** (`75ff6e425ce304a35f18c0230c0d13d3913a31a9`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Yes
  - predicted: WordNet is used to measure model competence in taxonomic knowledge in different settings (including hypernymy, hyponymy, and synonymy detection)
  - top context: [Introduction] While our methodology is amenable to any knowledge source and set of models/benchmark tasks, we focus on probing state-of-the-art transformer models BIBREF7, BIBREF9 in the domain of science MCQA. For sources of expert knowledge, we use WordNet, a comprehensive lexical ontology, and o

- **Are the automatically constructed datasets subject to quality control?** (`e97186c51d4af490dba6faaf833d269c8256426c`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: automatically constructing probes to be vulnerable to annotation artifacts, which we carefully control for
  - top context: [Discussion and Conclusion] The main appeal of using automatically generate datasets is the ability to systematically manipulate and control the complexity of target questions, which allows for more controlled experimentation and new forms of evaluation. Despite the positive results described above,

- **what are the previous state of the art?** (`db72a78a7102b5f0e75a4d9e1a06a3c2e7aabb21`, answer_f1 1.00 -> 0.00)
  - gold: Farasa, RDI | Farasa BIBREF31, MADAMIRA BIBREF29, RDI (Rashwan et al., 2015), MIT (Belinkow and Glass, 2015), Microsoft ATKS BIBREF28
  - predicted: ELMo BIBREF1, GPT BIBREF2, Bidirectional Encoder Representations from Transformers (Bert; BIBREF0), ER-CNN (extended ranking CNN), R-RNN (ranking RNN), i-vector+svm system, TDNN+stat-pool system (x-vector system)
  - top context: [Introduction] Language model pretraining has advanced the state of the art in many NLP tasks ranging from sentiment analysis, to question answering, natural language inference, named entity recognition, and textual similarity. State-of-the-art pretrained models include ELMo BIBREF1, GPT BIBREF2, an
