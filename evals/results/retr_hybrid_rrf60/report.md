# Run `retr_hybrid_rrf60`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `retrieval.method` | dense | hybrid |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.420 | 0.419 | +0.002 |
| answer_f1.abstractive | 0.232 | 0.225 | +0.007 |
| answer_f1.extractive | 0.479 | 0.473 | +0.006 |
| answer_f1.gold.abstractive | 0.306 | 0.302 | +0.003 |
| answer_f1.gold.extractive | 0.462 | 0.464 | -0.002 |
| answer_f1.gold.unanswerable | 0.426 | 0.411 | +0.015 |
| answer_f1.gold.yes_no | 0.458 | 0.464 | -0.005 |
| answer_f1.unanswerable | 0.478 | 0.458 | +0.020 |
| answer_f1.yes_no | 0.464 | 0.486 | -0.022 |
| completion_tokens_mean | 54.268 | 51.800 | +2.468 |
| context_words_mean | 1000.916 | 998.848 | +2.068 |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.304 | 0.297 | +0.008 |
| evidence_f1_ceiling | 0.534 | 0.524 | +0.010 |
| false_refusal_rate | 0.033 | 0.042 | -0.009 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 1067.918 | 1042.795 | +25.123 |
| latency_p95_ms | 1868.099 | 1850.713 | +17.385 |
| n.abstractive | 58.000 | 55.000 | +3.000 |
| n.extractive | 141.000 | 144.000 | -3.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 23.000 | 24.000 | -1.000 |
| n.yes_no | 28.000 | 27.000 | +1.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2526.000 | 2577.000 | -51.000 |
| prompt_tokens_mean | 2217.120 | 2228.868 | -11.748 |
| refusal_recall | 0.270 | 0.270 | +0.000 |
| retrieval.mrr | 0.524 | 0.552 | -0.029 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.370 | 0.409 | -0.039 |
| retrieval.ndcg_at_3 | 0.464 | 0.492 | -0.027 |
| retrieval.ndcg_at_5 | 0.515 | 0.527 | -0.012 |
| retrieval.recall_at_1 | 0.313 | 0.335 | -0.022 |
| retrieval.recall_at_3 | 0.593 | 0.603 | -0.009 |
| retrieval.recall_at_5 | 0.728 | 0.703 | +0.025 |

## Biggest regressions

- **Do they use attention?** (`45e9533586199bde19313cd43b3d0ecadcaf7a33`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: attention
  - top context: [Model] layer, and also with the output of the first contextual encoding layer as the input of its second encoding layer. Thus, we obtain the final representation of the contextual encoding layer by a concatenation of the outputs of two BiLSTM: INLINEFORM0 for questions and INLINEFORM1 for passages.

- **What do they mean by a person's industry?** (`935d6a6187e6a0c9c0da8e53a42697f853f5c248`, answer_f1 1.00 -> 0.00)
  - gold: the aggregate of enterprises in a particular field | the aggregate of enterprises in a particular field
  - predicted: a user's **industry**
  - top context: [Abstract] Automatic profiling of social media users is an important task for supporting a multitude of downstream applications. While a number of studies have used social media content to extract and study collective social attributes, there is a lack of substantial research that addresses the dete

- **Do the authors report results only on English data?** (`c49ee6ac4dc812ff84d255886fd5aff794f53c39`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Unanswerable
  - predicted: the only scientific summarization benchmark is from TAC 2014 summarization track, which consists of 20 topics each with a biomedical journal article and 4 gold human written summaries
  - top context: [Summarization Evaluation by Relevance Analysis (Sera)] keep the noun phrases as informative concepts in the summary and eliminate all other terms; and (iii) Keywords (KW): We only keep the keywords and key phrases in the summary. For extracting the keywords and keyphrases (with length of up to 3 te

- **what dataset was used?** (`e48e750743aef36529fbea4328b8253dbe928b4d`, answer_f1 1.00 -> 0.00)
  - gold: WASSA-2017 Shared Task on Emotion Intensity
  - predicted: train and dev sets were merged to perform 10-fold cross validation
  - top context: [Regression] remaining dataset. The predictions are averaged across all the folds to generalize the solution and prevent over-fitting. As described in Section SECREF6 , different combinations of feature extractors were used. After performing feature extraction, the data was then passed to various re

- **what are the previous state of the art?** (`db72a78a7102b5f0e75a4d9e1a06a3c2e7aabb21`, answer_f1 0.97 -> 0.29)
  - gold: Farasa, RDI | Farasa BIBREF31, MADAMIRA BIBREF29, RDI (Rashwan et al., 2015), MIT (Belinkow and Glass, 2015), Microsoft ATKS BIBREF28
  - predicted: BIBREF14 (14% WER), BIBREF22 (4.1% DER on the Quran (CA)), BIBREF23 (9% DER without CE, 28% DER with CE), BIBREF24 (7.33% WER without CE, 23.61% WER with CE), BIBREF25 (5.5% DER and 18% WER on words without CE), Farasa (10.7% CEER for MSA), RDI diacritizer (14.4% CEER for MSA), Microsoft Arabic Toolkit Services (ATKS), MIT (Belinkow and Glass, 2015), MADAMIRA BIBREF29
  - top context: [Background] hidden Markov model (HMM) BIBREF14, BIBREF22 with an input character sequence, the model attempts to find the best state sequence given previous observations. BIBREF14 reported a 14% word error rate (WER) while BIBREF22 achieved a 4.1% diacritic error rate (DER) on the Quran (CA). BIBRE
