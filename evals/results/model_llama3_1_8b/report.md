# Run `model_llama3_1_8b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.extra_body` | {} | None |
| `llm.extra_body.reasoning_effort` | None | none |
| `llm.model` | ministral-3:14b | llama3.1:8b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.326 | 0.419 | -0.093 |
| answer_f1.abstractive | 0.267 | 0.225 | +0.042 |
| answer_f1.extractive | 0.441 | 0.473 | -0.032 |
| answer_f1.gold.abstractive | 0.302 | 0.302 | -0.001 |
| answer_f1.gold.extractive | 0.438 | 0.464 | -0.026 |
| answer_f1.gold.unanswerable | 0.145 | 0.411 | -0.266 |
| answer_f1.gold.yes_no | 0.052 | 0.464 | -0.411 |
| answer_f1.unanswerable | 0.091 | 0.458 | -0.367 |
| answer_f1.yes_no | 0.046 | 0.486 | -0.440 |
| answer_words_mean | 10.412 | 15.492 | -5.080 |
| completion_tokens_mean | 38.088 | 51.800 | -13.712 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.244 | 0.297 | -0.052 |
| evidence_f1_ceiling | 0.524 | 0.524 | +0.000 |
| false_refusal_rate | 0.023 | 0.042 | -0.019 |
| gold_recall | 0.364 | 0.494 | -0.131 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| judge.accuracy | 0.404 | 0.480 | -0.076 |
| judge.accuracy.gold.abstractive | 0.397 | 0.310 | +0.086 |
| judge.accuracy.gold.extractive | 0.557 | 0.595 | -0.038 |
| judge.accuracy.gold.unanswerable | 0.108 | 0.351 | -0.243 |
| judge.accuracy.gold.yes_no | 0.042 | 0.458 | -0.417 |
| judge.agreement_on_high_f1 | 0.950 | 1.000 | -0.050 |
| judge.error_rate | 0.000 | 0.000 | +0.000 |
| judge.llm_calls | 174.000 | 163.000 | +11.000 |
| latency_p50_ms | 615.905 | 1042.795 | -426.890 |
| latency_p95_ms | 892.121 | 1850.713 | -958.592 |
| n.abstractive | 60.000 | 55.000 | +5.000 |
| n.extractive | 141.000 | 144.000 | -3.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 22.000 | 24.000 | -2.000 |
| n.yes_no | 27.000 | 27.000 | +0.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 1989.000 | 2577.000 | -588.000 |
| prompt_tokens_mean | 1641.536 | 2228.868 | -587.332 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.054 | 0.270 | -0.216 |
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

- **Do the errors of the model reflect linguistic similarity between different L1s?** (`007b13f05d234d37966d1aa7d85b5fd78564ff45`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable
  - predicted: The errors of the model do not necessarily reflect linguistic similarity between different L1s.
  - top context: [Introduction] includes texts produced by adults and children, as well as a spoken subset. The aforementioned Portuguese learner corpora contain very useful data for research, particularly for Native Language Identification (NLI), a task that has received much attention in recent years. NLI is the t

- **Did they use a crowdsourcing platform for annotations?** (`0ba3ea93eef5660a79ea3c26c6a270eac32dfa4c`, answer_f1 1.00 -> 0.00)
  - gold: No | Unanswerable
  - predicted: The morphosyntactic annotation was performed automatically.
  - top context: [Evaluation] within this study. The senses used for annotation are coarse-grained – with one sense covering a range of related usages. Each word has between two and four senses. To employ the method described in previous section, the NCP senses have been manually mapped to fine-grained plWordNet syn

- **What languages pairs are used in machine translation?** (`2f901dab6b757e12763b23ae8b37ae2e517a2271`, answer_f1 1.00 -> 0.00)
  - gold: German–English | German–English
  - predicted: German-English
  - top context: [Language Modeling] PTB. For both LSTM implementations, running time was dominated by the RNN computations, even with the highly optimized cuDNN implementation. For the QRNN implementation, however, the “RNN” layers are no longer the bottleneck. Indeed, there are diminishing returns from further opt

- **What is the size of bioScope corpus?** (`45be665a4504f0c7f458cf3f75a95d5a75eefd42`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: BioScope Corpus has sub corpora of BioScope Abstracts (BA) and BioScope Full Papers (BF), and also uses the SFU Review Corpus (SFU).
  - top context: [Introduction] the speculation, while the words affected by the speculation are in the scope of that cue. This task was the CoNLL-2010 Shared Task (BIBREF0), which had 3 different subtasks. Task 1B was speculation cue detection on the BioScope Corpus, Task 1W was weasel identification from Wikipedia

- **Do they use attention?** (`45e9533586199bde19313cd43b3d0ecadcaf7a33`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: Unanswerable
  - top context: [Model] layer, and also with the output of the first contextual encoding layer as the input of its second encoding layer. Thus, we obtain the final representation of the contextual encoding layer by a concatenation of the outputs of two BiLSTM: INLINEFORM0 for questions and INLINEFORM1 for passages.
