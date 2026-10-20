# Ablation summary

| run | chunking.chunk_overlap | chunking.chunk_size | chunking.strategy | llm.extra_body | llm.extra_body.reasoning_effort | llm.model | retrieval.method | retrieval.rrf_k | retrieval.top_k | answer_f1 | gold_recall | answer_words_mean | evidence_f1 | evidence_f1_ceiling | retrieval.recall_at_5 | refusal_recall | false_refusal_rate | error_rate | latency_p95_ms | Δ answer_f1 vs baseline [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 40 | 200 | fixed | {} | - | ministral-3:14b | dense | 60 | 5 | 0.419 | 0.494 | 15.492 | 0.297 | 0.524 | 0.703 | 0.270 | 0.042 | 0.000 | 1850.713 | - |
| chunk_fixed_100 | 20 | 100 | fixed | {} | - | ministral-3:14b | dense | 60 | 5 | 0.417 | 0.479 | 13.704 | 0.343 | 0.621 | 0.673 | 0.378 | 0.070 | 0.000 | 1514.128 | -0.002 [-0.040, +0.032] |
| chunk_fixed_400 | 80 | 400 | fixed | {} | - | ministral-3:14b | dense | 60 | 5 | 0.423 | 0.509 | 17.604 | 0.222 | 0.466 | 0.856 | 0.135 | 0.038 | 0.000 | 2242.930 | +0.005 [-0.028, +0.039] |
| chunk_section_200 | 0 | 200 | section | {} | - | ministral-3:14b | dense | 60 | 5 | 0.442 | 0.512 | 14.036 | 0.425 | 0.664 | 0.665 | 0.243 | 0.061 | 0.000 | 1451.214 | +0.024 [-0.013, +0.058] |
| model_gemma4_12b | 40 | 200 | fixed | - | none | gemma4:12b | dense | 60 | 5 | 0.421 | 0.467 | 11.288 | 0.316 | 0.524 | 0.703 | 0.568 | 0.113 | 0.000 | 1288.133 | +0.002 [-0.039, +0.045] |
| model_gpt_oss_20b | 40 | 200 | fixed | - | low | gpt-oss:20b | dense | 60 | 5 | 0.421 | 0.448 | 8.240 | 0.307 | 0.524 | 0.703 | 0.459 | 0.160 | 0.000 | 3081.897 | +0.002 [-0.040, +0.046] |
| model_llama3_1_8b | 40 | 200 | fixed | - | none | llama3.1:8b | dense | 60 | 5 | 0.326 | 0.364 | 10.412 | 0.244 | 0.524 | 0.703 | 0.054 | 0.023 | 0.000 | 892.121 | -0.093 [-0.140, -0.048] |
| model_ministral3_14b | 40 | 200 | fixed | - | none | ministral-3:14b | dense | 60 | 5 | 0.419 | 0.494 | 15.492 | 0.297 | 0.524 | 0.703 | 0.270 | 0.042 | 0.000 | 2052.496 | +0.000 [+0.000, +0.000] |
| model_phi4_14b | 40 | 200 | fixed | - | none | phi4:14b | dense | 60 | 5 | 0.399 | 0.490 | 17.864 | 0.287 | 0.524 | 0.703 | 0.297 | 0.080 | 0.000 | 2365.049 | -0.019 [-0.057, +0.020] |
| model_qwen3_14b | 40 | 200 | fixed | - | none | qwen3:14b | dense | 60 | 5 | 0.397 | 0.425 | 11.580 | 0.288 | 0.524 | 0.703 | 0.351 | 0.108 | 0.000 | 2549.161 | -0.022 [-0.060, +0.015] |
| model_qwen3_5_9b | 40 | 200 | fixed | - | none | qwen3.5:9b | dense | 60 | 5 | 0.381 | 0.427 | 10.800 | 0.306 | 0.524 | 0.703 | 0.405 | 0.061 | 0.000 | 970.125 | -0.038 [-0.082, +0.005] |
| model_qwen3_8b | 40 | 200 | fixed | - | none | qwen3:8b | dense | 60 | 5 | 0.403 | 0.434 | 10.856 | 0.265 | 0.524 | 0.703 | 0.162 | 0.047 | 0.000 | 972.827 | -0.015 [-0.052, +0.022] |
| retr_bm25 | 40 | 200 | fixed | {} | - | ministral-3:14b | bm25 | 60 | 5 | 0.370 | 0.442 | 15.196 | 0.270 | 0.471 | 0.606 | 0.270 | 0.061 | 0.000 | 1766.440 | -0.049 [-0.090, -0.010] |
| retr_hybrid_rrf10 | 40 | 200 | fixed | {} | - | ministral-3:14b | hybrid | 10 | 5 | 0.424 | 0.503 | 15.832 | 0.310 | 0.535 | 0.727 | 0.297 | 0.038 | 0.000 | 1805.329 | +0.005 [-0.024, +0.036] |
| retr_hybrid_rrf60 | 40 | 200 | fixed | {} | - | ministral-3:14b | hybrid | 60 | 5 | 0.420 | 0.501 | 16.456 | 0.304 | 0.534 | 0.728 | 0.270 | 0.033 | 0.000 | 1868.099 | +0.002 [-0.027, +0.032] |
| topk_10 | 40 | 200 | fixed | {} | - | ministral-3:14b | dense | 60 | 10 | 0.456 | 0.554 | 18.580 | 0.303 | 0.621 | 0.703 | 0.189 | 0.033 | 0.000 | 2420.295 | +0.037 [+0.007, +0.066] |
| topk_3 | 40 | 200 | fixed | {} | - | ministral-3:14b | dense | 60 | 3 | 0.403 | 0.462 | 14.748 | 0.300 | 0.473 | - | 0.405 | 0.075 | 0.000 | 1521.302 | -0.016 [-0.047, +0.015] |

Δ: paired bootstrap (2000 resamples, 95% CI) over the questions both runs answered. With 16 comparisons and no correction, about 0.8 CI(s) would exclude 0 by chance alone.
