# Run `topk_10`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `retrieval.top_k` | 5 | 10 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.456 | 0.419 | +0.037 |
| answer_f1.abstractive | 0.294 | 0.225 | +0.069 |
| answer_f1.extractive | 0.518 | 0.473 | +0.045 |
| answer_f1.unanswerable | 0.348 | 0.458 | -0.111 |
| answer_f1.yes_no | 0.559 | 0.486 | +0.072 |
| completion_tokens_mean | 57.324 | 51.800 | +5.524 |
| context_words_mean | 1983.680 | 998.848 | +984.832 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.303 | 0.297 | +0.007 |
| false_refusal_rate | 0.033 | 0.042 | -0.009 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 1465.709 | 1042.795 | +422.914 |
| latency_p95_ms | 2420.295 | 1850.713 | +569.582 |
| n.abstractive | 57.000 | 55.000 | +2.000 |
| n.extractive | 143.000 | 144.000 | -1.000 |
| n.unanswerable | 23.000 | 24.000 | -1.000 |
| n.yes_no | 27.000 | 27.000 | +0.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 4560.000 | 2577.000 | +1983.000 |
| prompt_tokens_mean | 3666.200 | 2228.868 | +1437.332 |
| refusal_recall | 0.189 | 0.270 | -0.081 |
| retrieval.mrr | 0.577 | 0.552 | +0.025 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.409 | 0.409 | +0.000 |
| retrieval.ndcg_at_10 | 0.790 | - | - |
| retrieval.ndcg_at_3 | 0.560 | 0.560 | +0.000 |
| retrieval.ndcg_at_5 | 0.649 | 0.649 | +0.000 |
| retrieval.recall_at_1 | 0.335 | 0.335 | +0.000 |
| retrieval.recall_at_10 | 0.915 | - | - |
| retrieval.recall_at_3 | 0.603 | 0.603 | +0.000 |
| retrieval.recall_at_5 | 0.703 | 0.703 | +0.000 |

## Biggest regressions

- **What language are the reviews in?** (`2236386729105f5cf42f73cc055ce3acdea2d452`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | English
  - predicted: TripAdvisor website
  - top context: [ReviewQA dataset] These questions are divided into 8 groups, regarding the competency required to be answered. In this section, we describe each task and the process followed to generate this dataset. We used a set of reviews extracted from the TripAdvisor website and originally proposed in BIBREF1

- **What is the size of bioScope corpus?** (`45be665a4504f0c7f458cf3f75a95d5a75eefd42`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: BF: 2670 sentences, BA: 11871 sentences
  - top context: [Introduction] the speculation, while the words affected by the speculation are in the scope of that cue. This task was the CoNLL-2010 Shared Task (BIBREF0), which had 3 different subtasks. Task 1B was speculation cue detection on the BioScope Corpus, Task 1W was weasel identification from Wikipedia

- **Do they use attention?** (`45e9533586199bde19313cd43b3d0ecadcaf7a33`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: attention
  - top context: [Model] layer, and also with the output of the first contextual encoding layer as the input of its second encoding layer. Thus, we obtain the final representation of the contextual encoding layer by a concatenation of the outputs of two BiLSTM: INLINEFORM0 for questions and INLINEFORM1 for passages.

- **What is the size of the parallel corpus used to train the model constraints?** (`9cf070d6671ee4a6353f79a165aa648309e01295`, answer_f1 1.00 -> 0.36)
  - gold: 1500 sentences | 1500 sentences
  - predicted: A part of the ILCI English-Hindi Tourism parallel corpus (1500 sentences)
  - top context: [Experiments and Results] were converted to MSTParser format from ConLL format for training. A part of the ILCI English-Hindi Tourism parallel corpus (1500 sentences) was used for training the classifiers. This corpus was POS-tagged using the Stanford POS Tagger BIBREF5 for English and using the Hin

- **Which matrix factorization methods do they use?** (`f7ed3b9ed469ed34f46acde86b8a066c52ecf430`, answer_f1 0.83 -> 0.29)
  - gold: weighted factorization of a word-context co-occurrence matrix  | The LexVec BIBREF7
  - predicted: matrix factorization to learn subword and word representations
  - top context: [Related Work] work). Closely related are models that use morphological segmentation in learning word representations BIBREF20 , BIBREF21 , BIBREF22 , BIBREF23 , BIBREF24 , BIBREF25 . Our model also uses n-grams and morphological segmentation, but it performs explicit matrix factorization to learn s
