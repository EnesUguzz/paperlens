# Run `hyde`

## Changed parameters vs `baseline`

| Parameter | Reference | This run |
|---|---|---|
| `chunking.semantic_breakpoint_percentile` | None | 90.0 |
| `chunking.semantic_min_words` | None | 50 |
| `data.align_min_score` | None | 85.0 |
| `data.align_min_words` | None | 8 |
| `data.align_score_margin` | None | 5.0 |
| `data.pdf_download_delay_s` | None | 3.0 |
| `data.pdf_url_template` | None | https://arxiv.org/pdf/{paper_id} |
| `generation.cite_unit` | None | chunk |
| `judge.extra_body.reasoning_effort` | None | none |
| `judge.model` | None | qwen3:14b |
| `judge.prompt_template` | None | prompts/judge.md |
| `meta.llm_digest` | not recorded | 4760c35aeb9d |
| `meta.server_version` | not recorded | 0.40.6 |
| `query.hyde_prompt` | None | prompts/hyde.md |
| `query.include_original` | None | True |
| `query.multi_query_prompt` | None | prompts/multi_query.md |
| `query.n_queries` | None | 3 |
| `query.transform` | None | hyde |
| `rerank.batch_size` | None | 16 |
| `rerank.device` | None | cuda |
| `rerank.enabled` | None | False |
| `rerank.max_length` | None | 512 |
| `rerank.model_name` | None | BAAI/bge-reranker-v2-m3 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.414 | 0.419 | -0.005 |
| answer_f1.abstractive | 0.226 | 0.225 | +0.001 |
| answer_f1.extractive | 0.468 | 0.473 | -0.005 |
| answer_f1.gold.abstractive | 0.306 | 0.302 | +0.004 |
| answer_f1.gold.extractive | 0.455 | 0.464 | -0.010 |
| answer_f1.gold.unanswerable | 0.381 | 0.411 | -0.029 |
| answer_f1.gold.yes_no | 0.500 | 0.464 | +0.036 |
| answer_f1.unanswerable | 0.423 | 0.458 | -0.035 |
| answer_f1.yes_no | 0.500 | 0.486 | +0.014 |
| answer_words_mean | 17.000 | 15.492 | +1.508 |
| completion_tokens_mean | 218.048 | 51.800 | +166.248 |
| context_words_mean | 1001.964 | 998.848 | +3.116 |
| data.questions_dropped | 0.000 | - | - |
| data.unmatched_evidence_rate | 0.046 | - | - |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.273 | 0.297 | -0.024 |
| evidence_f1_ceiling | 0.529 | 0.524 | +0.005 |
| false_refusal_rate | 0.042 | 0.042 | +0.000 |
| gold_recall | 0.491 | 0.494 | -0.004 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 3121.668 | 1042.795 | +2078.874 |
| latency_p95_ms | 4124.130 | 1850.713 | +2273.417 |
| n.abstractive | 55.000 | 55.000 | +0.000 |
| n.extractive | 143.000 | 144.000 | -1.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 26.000 | 24.000 | +2.000 |
| n.yes_no | 26.000 | 27.000 | -1.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 3220.000 | 2577.000 | +643.000 |
| prompt_tokens_mean | 2862.308 | 2228.868 | +633.440 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.270 | 0.270 | +0.000 |
| retrieval.doc_hit_rate | 1.000 | 1.000 | +0.000 |
| retrieval.mrr | 0.490 | 0.552 | -0.062 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.317 | 0.409 | -0.091 |
| retrieval.ndcg_at_3 | 0.434 | 0.492 | -0.058 |
| retrieval.ndcg_at_5 | 0.489 | 0.527 | -0.038 |
| retrieval.recall_at_1 | 0.263 | 0.335 | -0.071 |
| retrieval.recall_at_3 | 0.560 | 0.603 | -0.042 |
| retrieval.recall_at_5 | 0.713 | 0.703 | +0.010 |

## Biggest regressions

- **What language are the reviews in?** (`2236386729105f5cf42f73cc055ce3acdea2d452`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | English
  - predicted: TripAdvisor website
  - top context: [ReviewQA dataset] These questions are divided into 8 groups, regarding the competency required to be answered. In this section, we describe each task and the process followed to generate this dataset. We used a set of reviews extracted from the TripAdvisor website and originally proposed in BIBREF1

- **Does the model have attention?** (`b9d168da5321a7d7b812c52bb102a05210fe45bd`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Yes
  - predicted: The attention matrix indicates that this character should be based on the final state of the encoder
  - top context: [Figures and Tables] main objective of predicting the word form made. The magenta square marks the auxiliary objective of predicting the MSD tag V;PST;V.PTCP;PASS. Figure 1: Schematic representation of our approach. The focus here is on the prediction of the final character, e, of the word form made

- **What is the size of SFU Review corpus?** (`e86130c5b9ab28f0ec539c2bed1b1ae9efb99b7d`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Unanswerable
  - predicted: SFU Review Corpus (SFU)
  - top context: [Conclusion and Future Scope] SFU on the scope resolution task over the single dataset training approach. We thus significantly advance the state-of-the-art for speculation detection and scope resolution. On the negation scope resolution task, we applied the XLNet and RoBERTa and obtained a gain of 

- **what are the previous state of the art?** (`db72a78a7102b5f0e75a4d9e1a06a3c2e7aabb21`, answer_f1 0.97 -> 0.07)
  - gold: Farasa, RDI | Farasa BIBREF31, MADAMIRA BIBREF29, RDI (Rashwan et al., 2015), MIT (Belinkow and Glass, 2015), Microsoft ATKS BIBREF28
  - predicted: Farasa and the RDI diacritizer, which are trained on the same dataset and achieve CEERs of **10.7% and 14.4% respectively** for MSA; **BIBREF14 reported a 14% word error rate (WER)**, **BIBREF22 achieved a 4.1% diacritic error rate (DER) on the Quran (CA)**, **BIBREF23 reported a 9% (DER) without CE, and 28% DER with CE**, **BIBREF24 achieved an accuracy of 7.33% WER without CE and 23.61% WER with CE**, **BIBREF25 achieved 5.5% DER and 18% WER on words without CE**
  - top context: [Conclusion and Future Work] may be required to generate some of these features, and such tools may not be readily available for other language varieties, such as dialectal Arabic. However, we showed the efficacy of different varieties of features, such as surface level-features, and they can help i

- **how is diversity measured?** (`249c805ee6f2ebe4dbc972126b3d82fb09fa3556`, answer_f1 0.83 -> 0.10)
  - gold: average dissimilarity of all pairs of tags in the list of recommended tags |  the average dissimilarity of all pairs of tags in the list of recommended tags
  - predicted: Figure FIGREF16 shows the diversity of the tag recommendation approaches
  - top context: [Introduction] on measuring the accuracy of tag recommendations (e.g., BIBREF2 ). However, the authors of BIBREF6 suggest also to use beyond-accuracy metrics such as diversity to evaluate the quality of tag recommendations. In our work, we measure recommendation diversity in addition to recommendati
