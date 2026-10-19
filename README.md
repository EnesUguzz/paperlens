# paperlens

Agentic RAG over scientific papers (QASPER), built to measure every design choice.

> Work in progress — milestone M1a done: baseline RAG + evaluation loop + first ablations.
> Next: PDF parsing with Docling, reranking, query rewriting (M1b), then a LangGraph agent (M2).

**So far:** on 250 QASPER validation questions, a fully local stack (bge-m3 + ministral-3:14b
on one RTX 5080) reaches **0.456 answer F1** (official QASPER metric) with `top_k=10`, up from
0.419 for the baseline — the only change so far whose 95% CI excludes zero.

## How it works

- **Data:** QASPER (1,585 NLP papers, ~5k questions with gold evidence paragraphs). Every
  gold evidence string is mapped to a paragraph id, so retrieval is scored by id, not text.
- **Retrieval:** own code — fixed or section-aware chunking, Qdrant (dense, bge-m3), BM25
  (bm25s), Reciprocal Rank Fusion.
- **Generation:** any OpenAI-compatible server (Ollama now), JSON-schema output with
  cited passage numbers, one repair round for malformed JSON.
- **Evaluation:** Answer-F1 / Evidence-F1 (official QASPER logic), Recall@k / MRR / nDCG,
  refusal behaviour on unanswerable questions, p50/p95 latency, tokens.
- **Experiment loop:** one YAML per experiment (`extends: ../baseline.yaml` + one change).
  Each run writes `evals/results/<run>/report.md` with the changed parameters, metric deltas
  and the questions that got worse, and logs to MLflow.
- **Every parameter documents itself:** what it does, its typical range, what it changes
  in the metrics — see [`docs/CONFIG.md`](docs/CONFIG.md) (generated, enforced by a test).

## Results (validation, 250 questions, seed 13)

### LLM choice ([decision record](docs/decisions/001-llm-choice.md))

Eight local models, same retrieval and prompt. The top four are statistically tied
(paired bootstrap CIs include 0), so the choice was made on refusal behaviour and latency.

| Model | answer F1 | yes/no F1 | unanswerable F1 | false refusal | p95 latency |
|---|---|---|---|---|---|
| gemma4:12b | 0.421 | 0.13 | 0.76 | 11% | 1.3 s |
| gpt-oss:20b | 0.421 | 0.45 | 0.78 | 16% | 3.1 s |
| **ministral-3:14b** (chosen) | 0.419 | 0.49 | 0.46 | 4% | 2.1 s |
| qwen3:8b | 0.403 | 0.52 | 0.32 | 5% | 1.0 s |
| phi4:14b | 0.399 | 0.22 | 0.52 | 8% | 2.4 s |
| qwen3:14b | 0.397 | 0.25 | 0.56 | 11% | 2.5 s |
| qwen3.5:9b | 0.381 | 0.06 | 0.59 | 6% | 1.0 s |
| llama3.1:8b | 0.326 | 0.05 | 0.09 | 2% | 0.9 s |

### Retrieval ablations (ministral-3:14b)

Baseline: fixed 200-word chunks, 40 overlap, dense bge-m3, top_k 5.
Δ = paired difference in answer F1 vs baseline, with 95% bootstrap CI.

| Change | answer F1 | Δ [95% CI] | evidence F1 | Recall@5 | context words | p95 |
|---|---|---|---|---|---|---|
| baseline | 0.419 | | 0.297 | 0.703 | 999 | 1.9 s |
| **top_k 10** | **0.456** | **+0.037 [+0.007, +0.067]** | 0.303 | 0.703 | 1984 | 2.4 s |
| section-aware chunks | 0.442 | +0.024 [-0.010, +0.058] | **0.425** | 0.665 | 689 | 1.5 s |
| hybrid, RRF k=10 | 0.424 | +0.005 [-0.024, +0.036] | 0.310 | 0.727 | 1002 | 1.8 s |
| fixed 400-word chunks | 0.423 | +0.005 [-0.029, +0.039] | 0.222 | **0.856** | 1929 | 2.2 s |
| hybrid, RRF k=60 | 0.420 | +0.002 [-0.027, +0.032] | 0.304 | 0.728 | 1001 | 1.9 s |
| fixed 100-word chunks | 0.417 | -0.002 [-0.036, +0.034] | 0.343 | 0.673 | 511 | 1.5 s |
| top_k 3 | 0.403 | -0.016 [-0.046, +0.015] | 0.300 | - | 600 | 1.5 s |
| BM25 only | 0.370 | **-0.049 [-0.088, -0.010]** | 0.270 | 0.606 | 974 | 1.8 s |

Answer metrics are deterministic (temperature 0: re-running the baseline reproduced F1 to the
last digit); latency is not — the same config measured 1.9 s and 2.1 s p95 on two runs.
Full per-run reports: `evals/results/<run>/report.md`; all runs in one table:
[`evals/results/SUMMARY.md`](evals/results/SUMMARY.md).

### Findings

- **More context helped, more model did not.** top_k 5→10 is the only significant gain
  (+0.037), while doubling model size (qwen3 8B→14B) changed nothing and cost 2.6x latency:
  at ~70% Recall@5 the bottleneck is what reaches the model, not model capacity.
- **Recall is not the goal.** 400-word chunks raise Recall@5 from 0.70 to 0.86, yet answer F1
  does not move and Evidence-F1 drops (0.30→0.22): the evidence is in the prompt but diluted.
- **Structure-aware chunking is the efficient win.** Section-aware chunks lift Evidence-F1
  from 0.30 to 0.43 with 30% fewer context words — the model cites the right paragraph.
- **BM25 alone loses (-0.049, significant); fusing it with dense adds little here** —
  questions are paraphrased, and single-paper retrieval leaves few lexical distractors.
- **Headline F1 hides behaviour.** Models tied on F1 differ wildly on unanswerable questions
  (refusal recall 0.16 to 0.57); qwen3:8b invents answers to 84% of them. This is what the
  M2 groundedness check targets.
- **Data quality:** 4.6% of gold evidence strings (129/2,808) are section headings rather than
  paragraphs and can never be retrieved; all table/figure evidence maps to captions.

## Quickstart (WSL2 / Linux, NVIDIA GPU, Ollama)

    make setup
    make download
    ollama pull ministral-3:14b
    make eval CONFIG=experiments/configs/baseline.yaml

`make ablation` re-runs the retrieval ablations, `make models` the LLM comparison,
`make explain-config CONFIG=<yaml>` shows every parameter an experiment will use.

## Environment

Python 3.14 (see `.python-version`), PyTorch 2.14 with CUDA 13 wheels, Ollama 0.40 with
`OLLAMA_CONTEXT_LENGTH=16384` (Ollama silently truncates longer prompts; the report warns
when a run gets close).
