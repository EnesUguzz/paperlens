# Run `m2_agent`

## Changed parameters vs `m1b_best`

| Parameter | Reference | This run |
|---|---|---|
| `agent.clarify.max_candidates` | None | 3 |
| `agent.condense.enabled` | None | True |
| `agent.condense.history_turns` | None | 3 |
| `agent.condense.prompt_template` | None | prompts/condense.md |
| `agent.condense.triggers` | None | ['\\bit\\b', '\\bits\\b', '\\bthem\\b', '\\bthat\\b', '\\bthose\\b', '\\bthere\\b', '\\bsame\\b', '\\b(former|latter)\\b', '^\\s*(and|what about|how about)\\b'] |
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
| answer_f1 | 0.290 | 0.440 | -0.151 |
| answer_f1.abstractive | 0.190 | 0.247 | -0.058 |
| answer_f1.extractive | 0.285 | 0.506 | -0.221 |
| answer_f1.gold.abstractive | 0.244 | 0.325 | -0.081 |
| answer_f1.gold.extractive | 0.274 | 0.503 | -0.229 |
| answer_f1.gold.unanswerable | 0.388 | 0.440 | -0.052 |
| answer_f1.gold.yes_no | 0.333 | 0.380 | -0.046 |
| answer_f1.no_clarify | 0.279 | - | - |
| answer_f1.turn.first | 0.396 | - | - |
| answer_f1.turn.same | 0.264 | - | - |
| answer_f1.turn.switch_signaled | 0.391 | - | - |
| answer_f1.turn.switch_unsignaled | 0.131 | - | - |
| answer_f1.unanswerable | 0.520 | 0.500 | +0.020 |
| answer_f1.yes_no | 0.308 | 0.412 | -0.104 |
| answer_words_mean | 12.312 | 14.192 | -1.880 |
| completion_tokens_mean | 47.188 | 49.788 | -2.600 |
| condense.changed_rate | 0.012 | - | - |
| context_words_mean | 706.364 | 689.316 | +17.048 |
| data.questions_dropped | 0.000 | 0.000 | +0.000 |
| data.unmatched_evidence_rate | 0.046 | 0.046 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.293 | 0.491 | -0.198 |
| evidence_f1.no_clarify | 0.282 | - | - |
| evidence_f1_ceiling | 0.503 | 0.770 | -0.267 |
| false_refusal_rate | 0.155 | 0.061 | +0.094 |
| gold_recall | 0.328 | 0.507 | -0.178 |
| index.num_chunks | 8360.000 | 8360.000 | +0.000 |
| latency_p50_ms | 1016.924 | 963.592 | +53.332 |
| latency_p95_ms | 1749.370 | 1642.370 | +107.000 |
| llm_calls_mean | 1.024 | - | - |
| n.abstractive | 56.000 | 53.000 | +3.000 |
| n.extractive | 143.000 | 142.000 | +1.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.no_clarify | 231.000 | - | - |
| n.turn.first | 57.000 | - | - |
| n.turn.same | 54.000 | - | - |
| n.turn.switch_signaled | 67.000 | - | - |
| n.turn.switch_unsignaled | 72.000 | - | - |
| n.unanswerable | 25.000 | 28.000 | -3.000 |
| n.yes_no | 26.000 | 27.000 | -1.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 3005.000 | 2417.000 | +588.000 |
| prompt_tokens_mean | 1828.208 | 1788.992 | +39.216 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.324 | 0.351 | -0.027 |
| retrieval.doc_hit_rate | 0.676 | 1.000 | -0.324 |
| retrieval.mrr | 0.237 | 0.484 | -0.247 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.139 | 0.322 | -0.183 |
| retrieval.ndcg_at_3 | 0.210 | 0.423 | -0.213 |
| retrieval.ndcg_at_5 | 0.252 | 0.474 | -0.222 |
| retrieval.recall_at_1 | 0.117 | 0.264 | -0.148 |
| retrieval.recall_at_3 | 0.267 | 0.524 | -0.256 |
| retrieval.recall_at_5 | 0.374 | 0.665 | -0.292 |
| route.clarify_rate | 0.076 | - | - |
| route.paper_accuracy | 0.676 | - | - |
| route.paper_accuracy.first | 0.982 | - | - |
| route.paper_accuracy.same | 0.630 | - | - |
| route.paper_accuracy.switch_signaled | 1.000 | - | - |
| route.paper_accuracy.switch_unsignaled | 0.167 | - | - |
| route.signaled_catch | 1.000 | - | - |
| route.signaled_catch.t0 | 1.000 | - | - |
| route.signaled_catch.t1 | 1.000 | - | - |
| route.signaled_catch.t2 | 1.000 | - | - |
| route.unsignaled_catch | 0.167 | - | - |
| route.wrong_paper_rate | 0.220 | - | - |

## Biggest regressions

- **Did they use a crowdsourcing platform for annotations?** (`0ba3ea93eef5660a79ea3c26c6a270eac32dfa4c`, answer_f1 1.00 -> 0.00)
  - gold: No | Unanswerable
  - predicted: a competition was set up on Codalab
  - top context: [Participating Systems ::: Loud Whisper] Loud Whisper base their approach on BIBREF9 which is a graph-based sense clustering model. They process the data set to receive bigrams, create a co-occurence graph representation and after clustering assess the type of change per word by comparing the result

- **Is WordNet useful for taxonomic reasoning for this task?** (`75ff6e425ce304a35f18c0230c0d13d3913a31a9`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Yes
  - predicted: WordNet is used to measure model competence in taxonomic knowledge in different settings (including hypernymy, hyponymy, and synonymy detection)
  - top context: [Introduction] While our methodology is amenable to any knowledge source and set of models/benchmark tasks, we focus on probing state-of-the-art transformer models BIBREF7, BIBREF9 in the domain of science MCQA. For sources of expert knowledge, we use WordNet, a comprehensive lexical ontology, and o

- **Are the automatically constructed datasets subject to quality control?** (`e97186c51d4af490dba6faaf833d269c8256426c`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: automatically generate datasets [...] is much harder to validate the quality of such data at such a scale and such varying levels of complexity
  - top context: [Discussion and Conclusion] The main appeal of using automatically generate datasets is the ability to systematically manipulate and control the complexity of target questions, which allows for more controlled experimentation and new forms of evaluation. Despite the positive results described above,

- **What evaluations methods do they take?** (`71e4ba4e87e6596aeca187127c0d088df6570c57`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: Phoneme Error Rate (PER), Word Error Rate (WER), Word Error Rate 100 (WER 100)
  - top context: [Evaluation] We use the following three evaluation metrics: Phoneme Error Rate (PER) is the Levenshtein distance between the predicted phoneme sequences and the gold standard phoneme sequences, divided by the length of the gold standard phoneme sequences. Word Error Rate (WER) is the percentage of w

- **Which dataset do they train their models on?** (`30eacb4595014c9c0e5ee9669103d003cfdfe1e5`, answer_f1 1.00 -> 0.00)
  - gold: relation classification dataset of the SemEval 2010 task 8 | SemEval 2010 task 8 BIBREF8
  - predicted: editor tags, Amazon search terms
  - top context: [Dataset] Our dataset contains two sources of data, one to generate tag recommendations and another one to evaluate tag recommendations. HGV GmbH has collected all data sources and we provide the dataset statistics in Table TABREF3 . Data used to generate recommendations. We employ two sources of e-
