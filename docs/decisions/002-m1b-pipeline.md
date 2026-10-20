# 002 — M1b pipeline (the baseline M2 is compared against)

- Date: 2026-10-21
- Status: accepted
- Setup: QASPER validation, 250 questions (seed 13), ministral-3:14b (decision 001), temperature 0,
  Ollama 0.40.6, RTX 5080 16 GB, judge qwen3:14b.
- Runs: `experiments/configs/m1b/*.yaml` and `experiments/configs/m1b_best.yaml`, results in
  `evals/results/<run>/`, table in `evals/results/SUMMARY.md` (`make m1b` reproduces all of it).

## Results

Δ = paired bootstrap difference in answer F1 vs baseline (2000 resamples, 95% CI).

| Run | answer F1 | Δ [95% CI] | judge acc. | evidence F1 / ceiling | Recall@5 | false refusal | p95 |
|---|---|---|---|---|---|---|---|
| baseline | 0.419 | | 0.480 | 0.297 / 0.524 | 0.703 | 4.2% | 1851 ms |
| **m1b_best** | 0.435 | +0.016 [-0.024, +0.053] | 0.516 | 0.469 / 0.767 | 0.667 | 5.6% | 1538 ms |
| chunk_semantic | 0.427 | +0.009 [-0.030, +0.045] | 0.524 | 0.413 / 0.658 | 0.667 | 6.6% | 1501 ms |
| cite_paragraph | 0.421 | +0.002 [-0.018, +0.023] | 0.500 | 0.455 / 0.787 | 0.703 | 4.7% | 1809 ms |
| prompt_v2 | 0.421 | +0.003 [-0.019, +0.026] | 0.500 | 0.292 / 0.524 | 0.703 | 3.8% | 1624 ms |
| multi_query | 0.427 | +0.008 [-0.027, +0.045] | 0.472 | 0.271 / 0.505 | 0.672 | 3.3% | 2917 ms |
| hyde | 0.414 | -0.005 [-0.034, +0.025] | 0.460 | 0.273 / 0.529 | 0.713 | 4.2% | 4124 ms |
| rerank | 0.387 | -0.032 [-0.069, +0.005] | 0.472 | 0.285 / 0.526 | 0.731 | 4.2% | 2897 ms |
| hybrid_rerank | 0.387 | -0.032 [-0.069, +0.005] | 0.472 | 0.285 / 0.526 | 0.731 | 4.2% | 2905 ms |
| source_docling | 0.404 | -0.015 [-0.057, +0.025] | 0.500 | 0.253 / 0.484 | 0.660 | 2.8% | 3339 ms |

Error rate was 0 in every run.

## Decision

**M2's baseline is `experiments/configs/m1b_best.yaml`:** extends the M1a baseline with
semantic chunking (`chunking.strategy: semantic`) and paragraph-level citations
(`generation.cite_unit: paragraph`).

**This is chosen on Evidence-F1, not on Answer-F1.** No Answer-F1 delta in M1b has a 95% CI
excluding 0, including `m1b_best` (+0.016 [-0.024, +0.053]). With 250 questions we cannot
claim better answers; we claim better-grounded citations at no measurable cost to answers:

- Evidence-F1 0.297 to 0.469, evidence ceiling 0.524 to 0.767.
- judge.accuracy 0.480 to 0.516 (a lead, not a proven gain).
- p95 latency 1851 to 1538 ms (shorter contexts).

**Costs:** Recall@5 0.703 to 0.667 and false refusal 4.2% to 5.6%, both from semantic chunks.

**The combination is sub-additive.** The single changes add +0.116 (semantic) and +0.158
(paragraph citations) Evidence-F1, but together give +0.172. The ceiling is 0.767, below
cite_paragraph alone (0.787). Paragraph citations already remove most of the chunk-overlap
noise that semantic chunking also reduces, so the two partly cover the same gain.

## Rejected options

- **rerank / hybrid_rerank:** Answer-F1 -0.032 (CI [-0.069, +0.005]) and about +1 s p95. The two
  runs are identical because per-paper retrieval takes a 30-candidate pool that covers nearly the
  whole paper, so the reranker only reorders it. Not a verdict on rerankers; retry in M2's
  corpus mode, where the pool is a real subset.
- **multi_query, hyde:** no gain (Evidence-F1 and judge.accuracy lower than baseline) at 1.6x and
  2.2x p95 latency.
- **prompt_v2:** Answer-F1 +0.003, but refusal_recall 0.270 to 0.216 and Evidence-F1 unchanged.
- **source_docling:** parsing the arXiv PDFs with Docling is worse on Answer-F1 (0.404), Evidence-F1
  (0.253), Recall@5 (0.660) and latency (3.3 s p95). 90.4% of gold evidence aligns to a PDF
  paragraph (`data.evidence_aligned_rate` 0.904; 4.0% split across paragraphs) and 1 paper
  could not be downloaded from arXiv (`data.papers_missing` 1). The baseline keeps the
  QASPER-provided text.

## Caveats

- 25 comparisons at 95% without correction: about 1.3 CIs would exclude 0 by chance.
- Evidence-F1 is comparable across runs only at equal chunking and citation unit; the ceiling
  column shows how much of a change is the unit and how much is the model's citing.
- Metrics other than Answer-F1 (Evidence-F1, judge.accuracy) have no CI in SUMMARY.md.

## Follow-ups

- M2 compares the agentic pipeline against `m1b_best`, on the full validation set (1005
  questions) to tighten CIs.
- Re-test reranking and hybrid retrieval in M2's corpus mode.
