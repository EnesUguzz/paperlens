# Run `model_llama3_1_8b`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `llm.model` | qwen3:8b | llama3.1:8b |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.326 | 0.403 | -0.077 |
| answer_f1.abstractive | 0.267 | 0.239 | +0.028 |
| answer_f1.extractive | 0.441 | 0.461 | -0.020 |
| answer_f1.unanswerable | 0.091 | 0.318 | -0.227 |
| answer_f1.yes_no | 0.046 | 0.517 | -0.471 |
| completion_tokens_mean | 38.088 | 43.948 | -5.860 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.244 | 0.265 | -0.020 |
| false_refusal_rate | 0.023 | 0.047 | -0.023 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 615.905 | 684.335 | -68.431 |
| latency_p95_ms | 892.121 | 972.827 | -80.706 |
| n.abstractive | 60.000 | 58.000 | +2.000 |
| n.extractive | 141.000 | 141.000 | +0.000 |
| n.unanswerable | 22.000 | 22.000 | +0.000 |
| n.yes_no | 27.000 | 29.000 | -2.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 1989.000 | 2042.000 | -53.000 |
| prompt_tokens_mean | 1641.536 | 1676.652 | -35.116 |
| refusal_recall | 0.054 | 0.162 | -0.108 |
| retrieval.mrr | 0.552 | 0.552 | +0.000 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.409 | 0.409 | +0.000 |
| retrieval.ndcg_at_3 | 0.560 | 0.560 | +0.000 |
| retrieval.ndcg_at_5 | 0.649 | 0.649 | +0.000 |
| retrieval.recall_at_1 | 0.335 | 0.335 | +0.000 |
| retrieval.recall_at_3 | 0.603 | 0.603 | +0.000 |
| retrieval.recall_at_5 | 0.703 | 0.703 | +0.000 |

## Biggest regressions

- **Did they use a crowdsourcing platform for annotations?** (`0ba3ea93eef5660a79ea3c26c6a270eac32dfa4c`, answer_f1 1.00 -> 0.00)
  - gold: No | Unanswerable
  - predicted: The morphosyntactic annotation was performed automatically.
  - top context: [Evaluation] within this study. The senses used for annotation are coarse-grained – with one sense covering a range of related usages. Each word has between two and four senses. To employ the method described in previous section, the NCP senses have been manually mapped to fine-grained plWordNet syn

- **Did they used dataset from another domain for evaluation?** (`1dac4bc5af239024566fcb0f43bb9ff1c248ecec`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: Tencent News
  - top context: [Results] Pointer-Gen+ARL-SEN achieve comparable fluency performance to Pointer-Gen and Pointer-Gen+RL-ROUGE. Test set headlines achieve the best performance among all models, but the difference is not statistically significant. Also, we observe that fine-tuning on sensational headlines will hurt th

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
