# Ablation summary

| run | chunking.chunk_overlap | chunking.chunk_size | chunking.strategy | llm.extra_body | llm.extra_body.reasoning_effort | llm.model | retrieval.method | retrieval.rrf_k | retrieval.top_k | answer_f1 | evidence_f1 | retrieval.recall_at_5 | refusal_recall | false_refusal_rate | error_rate | latency_p95_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 40 | 200 | fixed | {} | - | ministral-3:14b | dense | 60 | 5 | 0.419 | 0.297 | 0.703 | 0.270 | 0.042 | 0.000 | 1850.713 |
| chunk_fixed_100 | 20 | 100 | fixed | {} | - | ministral-3:14b | dense | 60 | 5 | 0.417 | 0.343 | 0.673 | 0.378 | 0.070 | 0.000 | 1514.128 |
| chunk_fixed_400 | 80 | 400 | fixed | {} | - | ministral-3:14b | dense | 60 | 5 | 0.423 | 0.222 | 0.856 | 0.135 | 0.038 | 0.000 | 2242.930 |
| chunk_section_200 | 0 | 200 | section | {} | - | ministral-3:14b | dense | 60 | 5 | 0.442 | 0.425 | 0.665 | 0.243 | 0.061 | 0.000 | 1451.214 |
| model_gemma4_12b | 40 | 200 | fixed | - | none | gemma4:12b | dense | 60 | 5 | 0.421 | 0.316 | 0.703 | 0.568 | 0.113 | 0.000 | 1288.133 |
| model_gpt_oss_20b | 40 | 200 | fixed | - | low | gpt-oss:20b | dense | 60 | 5 | 0.421 | 0.307 | 0.703 | 0.459 | 0.160 | 0.000 | 3081.897 |
| model_llama3_1_8b | 40 | 200 | fixed | - | none | llama3.1:8b | dense | 60 | 5 | 0.326 | 0.244 | 0.703 | 0.054 | 0.023 | 0.000 | 892.121 |
| model_ministral3_14b | 40 | 200 | fixed | - | none | ministral-3:14b | dense | 60 | 5 | 0.419 | 0.297 | 0.703 | 0.270 | 0.042 | 0.000 | 2052.496 |
| model_phi4_14b | 40 | 200 | fixed | - | none | phi4:14b | dense | 60 | 5 | 0.399 | 0.287 | 0.703 | 0.297 | 0.080 | 0.000 | 2365.049 |
| model_qwen3_14b | 40 | 200 | fixed | - | none | qwen3:14b | dense | 60 | 5 | 0.397 | 0.288 | 0.703 | 0.351 | 0.108 | 0.000 | 2549.161 |
| model_qwen3_5_9b | 40 | 200 | fixed | - | none | qwen3.5:9b | dense | 60 | 5 | 0.381 | 0.306 | 0.703 | 0.405 | 0.061 | 0.000 | 970.125 |
| model_qwen3_8b | 40 | 200 | fixed | - | none | qwen3:8b | dense | 60 | 5 | 0.403 | 0.265 | 0.703 | 0.162 | 0.047 | 0.000 | 972.827 |
| retr_bm25 | 40 | 200 | fixed | {} | - | ministral-3:14b | bm25 | 60 | 5 | 0.370 | 0.270 | 0.606 | 0.270 | 0.061 | 0.000 | 1766.440 |
| retr_hybrid_rrf10 | 40 | 200 | fixed | {} | - | ministral-3:14b | hybrid | 10 | 5 | 0.424 | 0.310 | 0.727 | 0.297 | 0.038 | 0.000 | 1805.329 |
| retr_hybrid_rrf60 | 40 | 200 | fixed | {} | - | ministral-3:14b | hybrid | 60 | 5 | 0.420 | 0.304 | 0.728 | 0.270 | 0.033 | 0.000 | 1868.099 |
| topk_10 | 40 | 200 | fixed | {} | - | ministral-3:14b | dense | 60 | 10 | 0.456 | 0.303 | 0.703 | 0.189 | 0.033 | 0.000 | 2420.295 |
| topk_3 | 40 | 200 | fixed | {} | - | ministral-3:14b | dense | 60 | 3 | 0.403 | 0.300 | - | 0.405 | 0.075 | 0.000 | 1521.302 |
