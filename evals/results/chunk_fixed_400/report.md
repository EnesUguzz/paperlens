# Run `chunk_fixed_400`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `chunking.chunk_overlap` | 40 | 80 |
| `chunking.chunk_size` | 200 | 400 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.423 | 0.419 | +0.005 |
| answer_f1.abstractive | 0.247 | 0.225 | +0.021 |
| answer_f1.extractive | 0.497 | 0.473 | +0.023 |
| answer_f1.unanswerable | 0.273 | 0.458 | -0.186 |
| answer_f1.yes_no | 0.517 | 0.486 | +0.031 |
| completion_tokens_mean | 55.216 | 51.800 | +3.416 |
| context_words_mean | 1928.788 | 998.848 | +929.940 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.222 | 0.297 | -0.075 |
| false_refusal_rate | 0.038 | 0.042 | -0.005 |
| index.num_chunks | 3367.000 | 6676.000 | -3309.000 |
| latency_p50_ms | 1409.095 | 1042.795 | +366.300 |
| latency_p95_ms | 2242.930 | 1850.713 | +392.217 |
| n.abstractive | 56.000 | 55.000 | +1.000 |
| n.extractive | 143.000 | 144.000 | -1.000 |
| n.unanswerable | 22.000 | 24.000 | -2.000 |
| n.yes_no | 29.000 | 27.000 | +2.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 4330.000 | 2577.000 | +1753.000 |
| prompt_tokens_mean | 3529.112 | 2228.868 | +1300.244 |
| refusal_recall | 0.135 | 0.270 | -0.135 |
| retrieval.mrr | 0.622 | 0.552 | +0.070 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.461 | 0.409 | +0.052 |
| retrieval.ndcg_at_3 | 0.647 | 0.560 | +0.087 |
| retrieval.ndcg_at_5 | 0.751 | 0.649 | +0.101 |
| retrieval.recall_at_1 | 0.381 | 0.335 | +0.046 |
| retrieval.recall_at_3 | 0.676 | 0.603 | +0.074 |
| retrieval.recall_at_5 | 0.856 | 0.703 | +0.153 |

## Biggest regressions

- **What is the size of bioScope corpus?** (`45be665a4504f0c7f458cf3f75a95d5a75eefd42`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: BF: 2670 sentences, BA: 11871 sentences
  - top context: [Introduction] the speculation, while the words affected by the speculation are in the scope of that cue. This task was the CoNLL-2010 Shared Task (BIBREF0), which had 3 different subtasks. Task 1B was speculation cue detection on the BioScope Corpus, Task 1W was weasel identification from Wikipedia

- **Do they use attention?** (`45e9533586199bde19313cd43b3d0ecadcaf7a33`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: attention
  - top context: [Model] word embeddings , 16-dim part-of-speech (POS) tagging embeddings, 8-dim named-entity embeddings and 4-dim hard-rule features. Note that we use small embedding size of POS and NER to reduce model size and they mainly serve the role of coarse-grained word clusters. Additionally, we use questio

- **Is fine-tuning required to incorporate these embeddings into existing models?** (`b06512c17d99f9339ffdab12cedbc63501ff527e`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: We just initialize the corresponding embedding layer with Dolores embeddings.
  - top context: [Dolores: Learner] context-independent representations learned by our learner. Note that our learner (which is an $L$ -layer Bi-Directional LSTM) computes a set of $2L + 1$ representations for each entity-relation pair which we denote by: $ R_t = [ x_t, \overrightarrow{h_{t,i}}, \overleftarrow{h_{t,

- **Do they report results only on English data?** (`c7eb71683f53ab7acffd691a36cad6edc7f5522e`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Unanswerable
  - predicted: a snapshot of English Wikipedia measuring 8GB
  - top context: [Quantitative Evaluation for Interpretability] randomly generated embedding. Results are calculated for the parameters INLINEFORM0 and INLINEFORM1 . Our proposed method significantly improves the interpretability for all INLINEFORM2 compared to the original GloVe approach. Our proposed method is sec

- **What is the size of SFU Review corpus?** (`e86130c5b9ab28f0ec539c2bed1b1ae9efb99b7d`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: SFU Review Corpus (SFU)
  - top context: [Introduction] the speculation, while the words affected by the speculation are in the scope of that cue. This task was the CoNLL-2010 Shared Task (BIBREF0), which had 3 different subtasks. Task 1B was speculation cue detection on the BioScope Corpus, Task 1W was weasel identification from Wikipedia
