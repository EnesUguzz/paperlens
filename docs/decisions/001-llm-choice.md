# 001 — LLM for M1/M2

- Date: 2026-10-19
- Status: accepted
- Setup: QASPER validation, 250 questions (seed 13), baseline retrieval (dense bge-m3,
  fixed 200-word chunks, top_k 5), same prompt, temperature 0, Ollama 0.40.6, RTX 5080 16 GB.
- Runs: `experiments/configs/m1a_models/*.yaml`, results in `evals/results/model_*`.

## Constraints that shaped the candidate list

Open weights, runs locally next to bge-m3 in 16 GB VRAM, available in Ollama.
Excluded: Kimi K2, DeepSeek V3, qwen3.6 27B/35B (do not fit or cloud-only);
DeepSeek-R1 distills (thinking cannot be switched off, ~10x eval time).

## Results

| Model | answer_f1 | Δ vs qwen3:8b [95% CI, paired bootstrap] | yes/no | unanswerable | false refusal | error | p95 latency |
|---|---|---|---|---|---|---|---|
| gemma4:12b | 0.421 | +0.018 [-0.030, +0.064] | 0.13 | 0.76 | 11% | 0 | 1.3 s |
| gpt-oss:20b (reasoning low) | 0.421 | +0.017 [-0.024, +0.060] | 0.45 | 0.78 | 16% | 0 | 3.1 s |
| **ministral-3:14b** | **0.419** | +0.015 [-0.022, +0.053] | 0.49 | 0.46 | 4% | 0 | 2.1 s |
| qwen3:8b (thinking off) | 0.403 | - | 0.52 | 0.32 | 5% | 0 | 1.0 s |
| phi4:14b | 0.399 | | 0.22 | 0.52 | 8% | 0 | 2.4 s |
| qwen3:14b (thinking off) | 0.397 | -0.006 [-0.045, +0.034] | 0.25 | 0.56 | 11% | 0 | 2.5 s |
| qwen3.5:9b (thinking off) | 0.381 | | 0.06 | 0.59 | 6% | 0 | 1.0 s |
| llama3.1:8b | 0.326 | | 0.05 | 0.09 | 2% | 0 | 0.9 s |

"yes/no" and "unanswerable" are answer_f1 on questions of that gold type.

## Findings

1. **The top four are statistically tied.** Every paired CI against qwen3:8b includes 0;
   250 questions cannot separate differences of ~0.02 F1. Claiming "model X is best" from
   the headline number would be wrong.
2. **Similar totals hide very different behaviour.** gemma4 refuses well on unanswerable
   questions (0.76) but also refuses most yes/no questions whose gold answer is "No"
   (QASPER labels "the paper does not do X" as "No", gemma4 calls it unanswerable).
   qwen3:8b does the opposite: it answers 84% of unanswerable questions anyway.
3. **Bigger is not better here.** qwen3:14b is no better than qwen3:8b and 2.6x slower:
   with 70% Recall@5, retrieval and prompt are the bottleneck, not model capacity.
4. **Two years of releases are measurable.** llama3.1:8b (2024) trails by ~0.08 F1,
   mostly from never refusing and failing yes/no.
5. **Thinking mode matters for cost, not this task.** qwen3 with thinking on spent ~280
   extra tokens per answer; we ran all Qwen models with `reasoning_effort: none`.

## Decision

**ministral-3:14b**, following the criteria order fixed before the runs
(answer_f1 -> error_rate <= 2% -> refusal balance -> latency as tie-breaker):

- answer_f1 is tied among the top four and error_rate is 0 for all, so refusal balance
  decides: ministral is the only top model with no answer-type collapse (yes/no 0.49,
  unanswerable 0.46) and the lowest false-refusal rate (4%).
- gpt-oss is excluded on latency: p95 3.1 s, and M2's graph makes 4-6 LLM calls per question.

**Cost of this choice:** ~2x latency of qwen3:8b (p95 2.1 s vs 1.0 s) for an F1 gain that is
not statistically proven. If M2 latency becomes the bottleneck, qwen3:8b is the fallback;
its weak refusal is exactly what M2's groundedness check is meant to fix.

## Follow-ups

- Prompt experiment (M1b): state QASPER's convention that "not done in the paper" is "No",
  then re-check gemma4/qwen3.5 yes/no scores.
- Final comparison on the full validation set (1005 questions) once M2 is done, to tighten CIs.
