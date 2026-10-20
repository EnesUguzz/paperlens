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

Eight local models, same retrieval and prompt. Δ = paired difference in answer F1 vs
qwen3:8b with 95% bootstrap CI; per-type columns are answer F1 by the question's majority
gold type (n questions in brackets). The top four are statistically tied, so the choice
was made on behaviour across answer types and on latency.

| Model | answer F1 | Δ vs qwen3:8b [95% CI] | yes/no (24) | unanswerable (37) | false refusal | p95 |
|---|---|---|---|---|---|---|
| gemma4:12b | 0.421 | +0.018 [-0.029, +0.063] | 0.09 | 0.66 | 11% | 1.3 s |
| gpt-oss:20b | 0.421 | +0.017 [-0.025, +0.060] | 0.38 | 0.63 | 16% | 3.1 s |
| **ministral-3:14b** (chosen) | 0.419 | +0.015 [-0.022, +0.052] | 0.46 | 0.41 | 4% | 2.1 s |
| qwen3:8b | 0.403 | | 0.50 | 0.33 | 5% | 1.0 s |
| phi4:14b | 0.399 | -0.004 [-0.047, +0.039] | 0.21 | 0.42 | 8% | 2.4 s |
| qwen3:14b | 0.397 | -0.006 [-0.046, +0.032] | 0.17 | 0.50 | 11% | 2.5 s |
| qwen3.5:9b | 0.381 | -0.023 [-0.071, +0.022] | 0.03 | 0.50 | 6% | 1.0 s |
| llama3.1:8b | 0.326 | **-0.077 [-0.124, -0.034]** | 0.05 | 0.15 | 2% | 0.9 s |

### Retrieval ablations (ministral-3:14b)

Baseline: fixed 200-word chunks, 40 overlap, dense bge-m3, top_k 5. Δ = paired difference
in answer F1 vs baseline, 95% bootstrap CI. Evidence ceiling = the Evidence-F1 a perfect
citer would reach with the best subset of the retrieved chunks (a cited chunk counts all its
paragraphs).

| Change | answer F1 | Δ [95% CI] | evidence F1 / ceiling | Recall@5 | context words | p95 |
|---|---|---|---|---|---|---|
| baseline | 0.419 | | 0.297 / 0.524 | 0.703 | 999 | 1.9 s |
| **top_k 10** | **0.456** | **+0.037 [+0.007, +0.066]** | 0.303 / 0.621 | 0.703 | 1984 | 2.4 s |
| section-aware chunks | 0.442 | +0.024 [-0.013, +0.058] | 0.425 / 0.664 | 0.665 | 689 | 1.5 s |
| hybrid, RRF k=10 | 0.424 | +0.005 [-0.024, +0.036] | 0.310 / 0.535 | 0.727 | 1002 | 1.8 s |
| fixed 400-word chunks | 0.423 | +0.005 [-0.028, +0.039] | 0.222 / 0.466 | **0.856** | 1929 | 2.2 s |
| hybrid, RRF k=60 | 0.420 | +0.002 [-0.027, +0.032] | 0.304 / 0.534 | 0.728 | 1001 | 1.9 s |
| fixed 100-word chunks | 0.417 | -0.002 [-0.040, +0.032] | 0.343 / 0.621 | 0.673 | 511 | 1.5 s |
| top_k 3 | 0.403 | -0.016 [-0.047, +0.015] | 0.300 / 0.473 | - | 600 | 1.5 s |
| BM25 only | 0.370 | **-0.049 [-0.090, -0.010]** | 0.270 / 0.471 | 0.606 | 974 | 1.8 s |

Sixteen comparisons at 95% without correction: about one CI could exclude 0 by chance, so
treat a single "significant" row (top_k 10 included) as a lead to confirm on the full
validation set, not a final result. CIs are produced by `evals/stats.py` and printed in
[`evals/results/SUMMARY.md`](evals/results/SUMMARY.md) (`make summarize`).
Answer metrics are deterministic (temperature 0: re-running the baseline reproduced F1 to the
last digit); latency is not — the same config measured 1.9 s and 2.1 s p95 on two runs.

### Findings

- **More context helped, more model did not.** top_k 5→10 is the only CI that excludes 0
  (+0.037), while doubling model size (qwen3 8B→14B) changed nothing and cost 2.6x latency:
  at ~70% Recall@5 the bottleneck is what reaches the model, not model capacity.
- **Recall is not the goal.** 400-word chunks raise Recall@5 from 0.70 to 0.86, yet answer F1
  does not move.
- **Evidence-F1 measures chunking as much as the model.** Section-aware chunks raise
  Evidence-F1 from 0.30 to 0.43, mostly because the ceiling rises (0.52→0.66): the chunks are
  paragraph-sized, so a correct citation carries less extra text. The model also gets closer to
  that ceiling (0.64 vs 0.57 of it), since overlapping fixed windows repeat the gold paragraph
  in two chunks and citing both only adds noise. Evidence-F1 is only comparable at equal
  chunking; M1b will have the model cite paragraphs instead of chunks.
- **BM25 alone loses (-0.049); fusing it with dense adds little here** — questions are
  paraphrased, and single-paper retrieval leaves few lexical distractors.
- **Headline F1 hides behaviour.** Models tied on F1 differ widely by question type:
  gemma4 handles unanswerable questions best (0.66) but fails yes/no (0.09: it calls
  "the paper does not do X" unanswerable, QASPER labels it "No"); qwen3:8b answers 84% of
  unanswerable questions anyway. That is what the M2 groundedness check targets.
- **Data quality:** 4.6% of gold evidence strings (129/2,808) are section headings rather than
  paragraphs and can never be retrieved; all table/figure evidence maps to captions.

### Reproducibility

Each run folder stores its config (`params.json`, including `meta.*`: prompt and data
hashes, git commit, model digest), every answer (`records.jsonl`) and a report.
`paperlens rescore` recomputes all metrics from those saved answers without calling the LLM
(used to apply the nDCG fix and the gold-type / evidence-ceiling metrics to past runs).

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
