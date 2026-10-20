# Run `source_docling`

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
| `data.text_source` | qasper_text | pdf_docling |
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
| `query.transform` | None | none |
| `rerank.batch_size` | None | 16 |
| `rerank.device` | None | cuda |
| `rerank.enabled` | None | False |
| `rerank.max_length` | None | 512 |
| `rerank.model_name` | None | BAAI/bge-reranker-v2-m3 |

## Metrics

| Metric | This run | Reference | Delta |
|---|---|---|---|
| answer_f1 | 0.404 | 0.419 | -0.015 |
| answer_f1.abstractive | 0.271 | 0.225 | +0.046 |
| answer_f1.extractive | 0.454 | 0.473 | -0.020 |
| answer_f1.gold.abstractive | 0.338 | 0.302 | +0.036 |
| answer_f1.gold.extractive | 0.444 | 0.464 | -0.020 |
| answer_f1.gold.unanswerable | 0.354 | 0.411 | -0.056 |
| answer_f1.gold.yes_no | 0.421 | 0.464 | -0.042 |
| answer_f1.unanswerable | 0.348 | 0.458 | -0.111 |
| answer_f1.yes_no | 0.468 | 0.486 | -0.018 |
| answer_words_mean | 16.752 | 15.492 | +1.260 |
| completion_tokens_mean | 59.760 | 51.800 | +7.960 |
| context_words_mean | 1001.040 | 998.848 | +2.192 |
| data.evidence_aligned_rate | 0.904 | - | - |
| data.evidence_split_rate | 0.040 | - | - |
| data.papers_missing | 1.000 | - | - |
| data.questions_dropped | 0.000 | - | - |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.253 | 0.297 | -0.044 |
| evidence_f1_ceiling | 0.484 | 0.524 | -0.040 |
| false_refusal_rate | 0.028 | 0.042 | -0.014 |
| gold_recall | 0.478 | 0.494 | -0.016 |
| index.num_chunks | 7694.000 | 6676.000 | +1018.000 |
| latency_p50_ms | 1358.393 | 1042.795 | +315.599 |
| latency_p95_ms | 3338.709 | 1850.713 | +1487.996 |
| n.abstractive | 57.000 | 55.000 | +2.000 |
| n.extractive | 142.000 | 144.000 | -2.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 23.000 | 24.000 | -1.000 |
| n.yes_no | 28.000 | 27.000 | +1.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 6548.000 | 2577.000 | +3971.000 |
| prompt_tokens_mean | 2413.480 | 2228.868 | +184.612 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.189 | 0.270 | -0.081 |
| retrieval.doc_hit_rate | 1.000 | 1.000 | +0.000 |
| retrieval.mrr | 0.525 | 0.552 | -0.027 |
| retrieval.n | 230.000 | 230.000 | +0.000 |
| retrieval.ndcg_at_1 | 0.387 | 0.409 | -0.022 |
| retrieval.ndcg_at_3 | 0.452 | 0.492 | -0.040 |
| retrieval.ndcg_at_5 | 0.491 | 0.527 | -0.036 |
| retrieval.recall_at_1 | 0.303 | 0.335 | -0.032 |
| retrieval.recall_at_3 | 0.550 | 0.603 | -0.053 |
| retrieval.recall_at_5 | 0.660 | 0.703 | -0.043 |

## Biggest regressions

- **What language are the reviews in?** (`2236386729105f5cf42f73cc055ce3acdea2d452`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | English
  - predicted: TripAdvisor
  - top context: [3.1 Original data] 7 aspects, among value, room, location, cleanliness, check-in/front desk, service, business service, can possibly be rated in a review. Figure 1 displays a review extracted from this dataset. Objective: Starting with the original corpus, we aim at building a machine reading task 

- **What languages pairs are used in machine translation?** (`2f901dab6b757e12763b23ae8b37ae2e517a2271`, answer_f1 1.00 -> 0.00)
  - gold: German–English | German–English
  - predicted: German-English
  - top context: [3.3 CHARACTER-LEVEL NEURAL MACHINE TRANSLATION] described in 2.1 on a challenging neural machine translation task, IWSLT German-English spoken-domain translation, applying fully character-level segmentation. This dataset consists of 209,772 sentence pairs of parallel training data from transcribed 

- **What was the baseline?** (`6f024d4c952fbb53b6f8cf27020659aa4d9a6590`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable
  - predicted: 57.57 42.98 74.64 84.91 5.48
  - top context: [2. Models] R@1 R@5 R@10 Mean Baseline 57.57 42.98 74.64 84.91 5.48 MN 59.24 44.64 76.48 86.41 5.14 MN-Wt 59.54 44.98 77.10 86.38 5.03 LF-RCNN 61.94 48.08 79.04 88.23 4.61 MN-RCNN 62.99 49.07 80.13 88.74 4.45 MN-RCNN-Wt 63.11 49.29 80.10 89.09 4.43 We used Pytorch 2 [4] for implementation. 3 In our 

- **Do they report results only on English datasets?** (`8bb0011ad1d63996d5650770f3be18abdd9f7fc6`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | Yes
  - predicted: No
  - top context: [6 Analysis] and "waterside" makes no sense given the context "the bank manager is walking along the waterside". It is the knowledge aided attention mechanisms that enable KAR to ignore such distracting inter-word semantic connections so that only the important ones are used. In this paper, we innov

- **Is fine-tuning required to incorporate these embeddings into existing models?** (`b06512c17d99f9339ffdab12cedbc63501ff527e`, answer_f1 1.00 -> 0.00)
  - gold: No | No
  - predicted: we can just use DOLORES embeddings as a drop-in replacement. We just initialize the corresponding embedding layer with DOLORES embeddings.
  - top context: [3.4 DOLORES: Learner] this requirement (which a large number of neural models on knowledge graphs do), we can just use DOLORES embeddings as a drop-in replacement. We just initialize the corresponding embedding layer with DOLORES embeddings. In our evaluation below, we show how to improve several s
