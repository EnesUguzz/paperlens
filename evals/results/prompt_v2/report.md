# Run `prompt_v2`

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
| `generation.prompt_template` | prompts/baseline_answer.md | prompts/answer_v2.md |
| `judge.extra_body.reasoning_effort` | None | none |
| `judge.model` | None | qwen3:14b |
| `judge.prompt_template` | None | prompts/judge.md |
| `meta.llm_digest` | not recorded | 4760c35aeb9d |
| `meta.prompt_sha256` | b9e0dd367503dd55 | 083eae4105f7d69e |
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
| answer_f1 | 0.421 | 0.419 | +0.003 |
| answer_f1.abstractive | 0.230 | 0.225 | +0.005 |
| answer_f1.extractive | 0.471 | 0.473 | -0.002 |
| answer_f1.gold.abstractive | 0.305 | 0.302 | +0.003 |
| answer_f1.gold.extractive | 0.463 | 0.464 | -0.002 |
| answer_f1.gold.unanswerable | 0.353 | 0.411 | -0.058 |
| answer_f1.gold.yes_no | 0.583 | 0.464 | +0.120 |
| answer_f1.unanswerable | 0.375 | 0.458 | -0.083 |
| answer_f1.yes_no | 0.586 | 0.486 | +0.100 |
| answer_words_mean | 12.104 | 15.492 | -3.388 |
| completion_tokens_mean | 47.704 | 51.800 | -4.096 |
| context_words_mean | 998.848 | 998.848 | +0.000 |
| data.questions_dropped | 0.000 | - | - |
| data.unmatched_evidence_rate | 0.046 | - | - |
| error_rate | 0.000 | 0.000 | +0.000 |
| evidence_f1 | 0.292 | 0.297 | -0.005 |
| evidence_f1_ceiling | 0.524 | 0.524 | +0.000 |
| false_refusal_rate | 0.038 | 0.042 | -0.005 |
| gold_recall | 0.473 | 0.494 | -0.021 |
| index.num_chunks | 6676.000 | 6676.000 | +0.000 |
| latency_p50_ms | 1050.009 | 1042.795 | +7.214 |
| latency_p95_ms | 1624.046 | 1850.713 | -226.667 |
| n.abstractive | 56.000 | 55.000 | +1.000 |
| n.extractive | 141.000 | 144.000 | -3.000 |
| n.gold.abstractive | 58.000 | 58.000 | +0.000 |
| n.gold.extractive | 131.000 | 131.000 | +0.000 |
| n.gold.unanswerable | 37.000 | 37.000 | +0.000 |
| n.gold.yes_no | 24.000 | 24.000 | +0.000 |
| n.unanswerable | 24.000 | 24.000 | +0.000 |
| n.yes_no | 29.000 | 27.000 | +2.000 |
| n_questions | 250.000 | 250.000 | +0.000 |
| prompt_tokens_max | 2637.000 | 2577.000 | +60.000 |
| prompt_tokens_mean | 2288.868 | 2228.868 | +60.000 |
| query_error_rate | 0.000 | 0.000 | +0.000 |
| refusal_recall | 0.216 | 0.270 | -0.054 |
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

- **What language are the reviews in?** (`2236386729105f5cf42f73cc055ce3acdea2d452`, answer_f1 1.00 -> 0.00)
  - gold: Unanswerable | English
  - predicted: TripAdvisor website
  - top context: [ReviewQA dataset] These questions are divided into 8 groups, regarding the competency required to be answered. In this section, we describe each task and the process followed to generate this dataset. We used a set of reviews extracted from the TripAdvisor website and originally proposed in BIBREF1

- **Do the authors report results only on English data?** (`c49ee6ac4dc812ff84d255886fd5aff794f53c39`, answer_f1 1.00 -> 0.00)
  - gold: Yes | Unanswerable
  - predicted: TAC 2014 scientific summarization dataset
  - top context: [Summarization Evaluation by Relevance Analysis (Sera)] keep the noun phrases as informative concepts in the summary and eliminate all other terms; and (iii) Keywords (KW): We only keep the keywords and key phrases in the summary. For extracting the keywords and keyphrases (with length of up to 3 te

- **what is the source of the data?** (`30af1926559079f59b0df055da76a3a34df8336f`, answer_f1 0.67 -> 0.00)
  - gold: Android application
  - predicted: crowdsourcing
  - top context: [Introduction] and final version of the database was released at the beginning of 2019. In BIBREF4, the running project and its data collection scenarios were described, alongside with some preliminary results and statistics. In this paper, we announce the final and cleaned-up version of the databas

- **What discourse relations does it work best/worst for?** (`f17ca24b135f9fe6bb25dc5084b13e1637ec7744`, answer_f1 0.67 -> 0.00)
  - gold: explicit discourse relations | Best: Expansion (Exp). Worst: Comparison (Comp).
  - predicted: Unanswerable
  - top context: [Introduction] paragraph-level discourse relation prediction model achieves improved performance on PDTB for both implicit discourse relation classification and explicit discourse relation classification. Since the PDTB BIBREF7 corpus was created, a surge of studies BIBREF8 , BIBREF3 , BIBREF9 , BIB

- **What is the size of the parallel corpus used to train the model constraints?** (`9cf070d6671ee4a6353f79a165aa648309e01295`, answer_f1 1.00 -> 0.36)
  - gold: 1500 sentences | 1500 sentences
  - predicted: A part of the ILCI English-Hindi Tourism parallel corpus (1500 sentences)
  - top context: [Experiments and Results] were converted to MSTParser format from ConLL format for training. A part of the ILCI English-Hindi Tourism parallel corpus (1500 sentences) was used for training the classifiers. This corpus was POS-tagged using the Stanford POS Tagger BIBREF5 for English and using the Hin
