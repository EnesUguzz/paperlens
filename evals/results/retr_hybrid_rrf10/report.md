# Run `retr_hybrid_rrf10`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `retrieval.method` | dense | hybrid |
| `retrieval.rrf_k` | 60 | 10 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.424 | 0.419 | +0.005 |
| answer_f1.abstractive | 0.232 | 0.225 | +0.007 |
| answer_f1.extractive | 0.472 | 0.473 | -0.002 |
| answer_f1.gold.abstractive | 0.304 | 0.302 | +0.001 |
| answer_f1.gold.extractive | 0.460 | 0.464 | -0.005 |
| answer_f1.gold.unanswerable | 0.438 | 0.411 | +0.027 |
| answer_f1.gold.yes_no | 0.500 | 0.464 | +0.036 |
| answer_f1.unanswerable | 0.500 | 0.458 | +0.042 |
| answer_f1.yes_no | 0.519 | 0.486 | +0.032 |
| completion_tokens_mean | 53.048 | 51.800 | +1.248 |
| context_words_mean | 1001.932 | 998.848 | +3.084 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.310 | 0.297 | +0.013 |
| evidence_f1_ceiling | 0.535 | 0.524 | +0.011 |
| false_refusal_rate | 0.038 | 0.042 | -0.005 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 1063.629 | 1042.795 | +20.835 |
| latency_p95_ms | 1805.329 | 1850.713 | -45.384 |
| n.abstractive | 58.000 | 55.000 | +3.000 |
| n.extractive | 141.000 | 144.000 | -3.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 24.000 | 24.000 | +0.000 |
| n.yes_no | 27.000 | 27.000 | +0.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2526.000 | 2577.000 | -51.000 |
| prompt_tokens_mean | 2219.456 | 2228.868 | -9.412 |
| refusal_recall | 0.297 | 0.270 | +0.027 |
| retrieval.mrr | 0.521 | 0.552 | -0.031 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.365 | 0.409 | -0.043 |
| retrieval.ndcg_at_3 | 0.467 | 0.492 | -0.025 |
| retrieval.ndcg_at_5 | 0.514 | 0.527 | -0.013 |
| retrieval.recall_at_1 | 0.305 | 0.335 | -0.029 |
| retrieval.recall_at_3 | 0.597 | 0.603 | -0.006 |
| retrieval.recall_at_5 | 0.727 | 0.703 | +0.024 |

## Biggest regressions

- **Do they use attention?** (`45e9533586199bde19313cd43b3d0ecadcaf7a33`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: attention
  - top context: [Model] layer, and also with the output of the first contextual encoding layer as the input of its second encoding layer. Thus, we obtain the final representation of the contextual encoding layer by a concatenation of the outputs of two BiLSTM: INLINEFORM0 for questions and INLINEFORM1 for passages.

- **What do they mean by a person's industry?** (`935d6a6187e6a0c9c0da8e53a42697f853f5c248`, answer_f1 1.00 -> 0.00)
  - gold: the aggregate of enterprises in a particular field | the aggregate of enterprises in a particular field
  - predicted: a user's **industry**
  - top context: [Abstract] Automatic profiling of social media users is an important task for supporting a multitude of downstream applications. While a number of studies have used social media content to extract and study collective social attributes, there is a lack of substantial research that addresses the dete

- **what dataset was used?** (`e48e750743aef36529fbea4328b8253dbe928b4d`, answer_f1 1.00 -> 0.00)
  - gold: WASSA-2017 Shared Task on Emotion Intensity
  - predicted: train and dev sets were merged to perform 10-fold cross validation
  - top context: [Regression] remaining dataset. The predictions are averaged across all the folds to generalize the solution and prevent over-fitting. As described in Section SECREF6 , different combinations of feature extractors were used. After performing feature extraction, the data was then passed to various re

- **what are the previous state of the art?** (`db72a78a7102b5f0e75a4d9e1a06a3c2e7aabb21`, answer_f1 0.97 -> 0.07)
  - gold: Farasa, RDI | Farasa BIBREF31, MADAMIRA BIBREF29, RDI (Rashwan et al., 2015), MIT (Belinkow and Glass, 2015), Microsoft ATKS BIBREF28
  - predicted: BIBREF14 reported a 14% word error rate (WER) while BIBREF22 achieved a 4.1% diacritic error rate (DER) on the Quran (CA). BIBREF23 reported a 9% (DER) without CE, and 28% DER with CE. BIBREF24 achieved an accuracy of 7.33% WER without CE and 23.61% WER with CE. BIBREF25 achieved 5.5% DER and 18% WER on words without CE. Farasa and the RDI diacritizer achieve CEERs of 10.7% and 14.4% respectively for MSA.
  - top context: [Conclusion and Future Work] insure greater consistency. For future work, we want to explore the effectiveness of augmenting our CW model with POS tagging information and a bigram language model. Further, we plan to create a multi reference diacritization test set to handle words that have multiple 

- **What are the two large-scale datasets used?** (`ae90c5567746fe25af2fcea0cc5f355751e05c71`, answer_f1 0.75 -> 0.00)
  - gold: US dataset, Italian dataset | US dataset, Italian dataset
  - predicted: diffusion cascades of news shared respectively in the **United States** and **Italy**
  - top context: [Experiments ::: Classification performances] of the political bias. We further corroborated this result with additional classification experiments, that show similar performances, in which we excluded from the training/test set two specific sources (one at a time and both at the same time) that out
