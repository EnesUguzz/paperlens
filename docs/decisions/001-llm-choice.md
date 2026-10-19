# 001 — LLM for M1/M2

- Date: 2026-10-19 (numbers re-scored after the final review: gold-type buckets, repo-generated CIs)
- Status: accepted
- Setup: QASPER validation, 250 questions (seed 13), baseline retrieval (dense bge-m3,
  fixed 200-word chunks, top_k 5), same prompt, temperature 0, Ollama 0.40.6, RTX 5080 16 GB.
- Runs: `experiments/configs/m1a_models/*.yaml`, results in `evals/results/model_*`.

## Constraints that shaped the candidate list

Open weights, runs locally next to bge-m3 in 16 GB VRAM, available in Ollama.
Excluded: Kimi K2, DeepSeek V3, qwen3.6 27B/35B (do not fit or cloud-only);
DeepSeek-R1 distills (thinking cannot be switched off, ~10x eval time).

## Results

Δ = paired bootstrap difference in answer F1 vs qwen3:8b (`evals/stats.py`, 2000 resamples).
Per-type columns: answer F1 bucketed by the question's majority gold type (`answer_f1.gold.*`;
n = questions). The official QASPER per-type bucket (type of the best-matching annotator)
depends on each model's own answers, so it is not used to compare models.

| Model | answer_f1 | Δ vs qwen3:8b [95% CI] | yes/no (24) | unanswerable (37) | refusal recall | false refusal | error | p95 latency |
|---|---|---|---|---|---|---|---|---|
| gemma4:12b | 0.421 | +0.018 [-0.029, +0.063] | 0.09 | 0.66 | 0.57 | 11% | 0 | 1.3 s |
| gpt-oss:20b (reasoning low) | 0.421 | +0.017 [-0.025, +0.060] | 0.38 | 0.63 | 0.46 | 16% | 0 | 3.1 s |
| **ministral-3:14b** | **0.419** | +0.015 [-0.022, +0.052] | 0.46 | 0.41 | 0.27 | 4% | 0 | 2.1 s |
| qwen3:8b (thinking off) | 0.403 | - | 0.50 | 0.33 | 0.16 | 5% | 0 | 1.0 s |
| phi4:14b | 0.399 | -0.004 [-0.047, +0.039] | 0.21 | 0.42 | 0.30 | 8% | 0 | 2.4 s |
| qwen3:14b (thinking off) | 0.397 | -0.006 [-0.046, +0.032] | 0.17 | 0.50 | 0.35 | 11% | 0 | 2.5 s |
| qwen3.5:9b (thinking off) | 0.381 | -0.023 [-0.071, +0.022] | 0.03 | 0.50 | 0.41 | 6% | 0 | 1.0 s |
| llama3.1:8b | 0.326 | -0.077 [-0.124, -0.034] | 0.05 | 0.15 | 0.05 | 2% | 0 | 0.9 s |

The per-type buckets are small (24 and 37 questions); read them as behaviour profiles, not
precise scores.

## Findings

1. **The top four are statistically tied.** Every paired CI against qwen3:8b includes 0;
   250 questions cannot separate differences of ~0.02 F1. Claiming "model X is best" from
   the headline number would be wrong.
2. **Similar totals hide very different behaviour.** gemma4 handles unanswerable questions
   best (0.66) but fails yes/no (0.09): QASPER labels "the paper does not do X" as "No",
   gemma4 calls it unanswerable. qwen3:8b does the opposite and answers 84% of unanswerable
   questions anyway.
3. **Bigger is not better here.** qwen3:14b is no better than qwen3:8b and 2.6x slower:
   with 70% Recall@5, retrieval and prompt are the bottleneck, not model capacity.
4. **Two years of releases are measurable.** llama3.1:8b (2024) trails by 0.077 F1 (the only
   CI that excludes 0), mostly from never refusing and failing yes/no.
5. **Thinking mode matters for cost, not this task.** qwen3 with thinking on spent ~280
   extra tokens per answer; we ran all Qwen models with `reasoning_effort: none`.

## Decision

**ministral-3:14b**, following the criteria order fixed before the runs
(answer_f1 -> error_rate <= 2% -> refusal balance -> latency as tie-breaker):

- answer_f1 is tied among the top four and error_rate is 0 for all, so refusal balance
  decides: ministral is the only top model with no answer-type collapse (gold-type yes/no
  0.46, unanswerable 0.41) and the lowest false-refusal rate (4%).
- gpt-oss is excluded on latency: p95 3.1 s, and M2's graph makes 4-6 LLM calls per question.

**Cost of this choice:** ~2x latency of qwen3:8b (p95 2.1 s vs 1.0 s) for an F1 gain that is
not statistically proven. If M2 latency becomes the bottleneck, qwen3:8b is the fallback;
its weak refusal is exactly what M2's groundedness check is meant to fix.

## Follow-ups

- Prompt experiment (M1b): state QASPER's convention that "not done in the paper" is "No",
  then re-check gemma4/qwen3.5 yes/no scores.
- Final comparison on the full validation set (1005 questions) once M2 is done, to tighten CIs.
