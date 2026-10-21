# 002 — M1b pipeline (the baseline M2 is compared against)

- Date: 2026-10-21
- Status: accepted
- Setup: QASPER validation, 250 questions (seed 13), ministral-3:14b (decision 001), temperature 0,
  Ollama 0.40.6, RTX 5080 16 GB, judge qwen3:14b.
- Runs: `experiments/configs/m1b/*.yaml`, `experiments/configs/m1b_best.yaml` and
  `experiments/configs/m1b_semantic_best.yaml`, results in
  `evals/results/<run>/`, table in `evals/results/SUMMARY.md` (`make m1b` reproduces all of it
  after `make prepare-pdfs`, about 1 h of arXiv downloads and Docling parsing, which
  `source_docling` needs; `make m1b` runs `uv sync --extra ml --extra pdf`, which drops the
  notebook dependency group).

## Results

Δ = paired bootstrap difference in answer F1 vs baseline (2000 resamples, 95% CI).

| Run | answer F1 | Δ [95% CI] | judge acc. | evidence F1 / ceiling | Recall@5 | false refusal | p95 |
|---|---|---|---|---|---|---|---|
| baseline | 0.419 | | 0.500 | 0.297 / 0.524 | 0.703 | 4.2% | 1851 ms |
| **m1b_best** | 0.440 | +0.022 [-0.015, +0.056] | 0.568 | 0.491 / 0.770 | 0.665 | 6.1% | 1642 ms |
| m1b_semantic_best | 0.435 | +0.016 [-0.024, +0.053] | 0.540 | 0.469 / 0.767 | 0.667 | 5.6% | 1538 ms |
| chunk_section_200 (M1a) | 0.442 | +0.024 [-0.013, +0.058] | 0.556 | 0.425 / 0.664 | 0.665 | 6.1% | 1451 ms |
| chunk_semantic | 0.427 | +0.009 [-0.030, +0.045] | 0.548 | 0.413 / 0.658 | 0.667 | 6.6% | 1501 ms |
| cite_paragraph | 0.421 | +0.002 [-0.018, +0.023] | 0.524 | 0.455 / 0.787 | 0.703 | 4.7% | 1809 ms |
| prompt_v2 | 0.421 | +0.003 [-0.019, +0.026] | 0.528 | 0.292 / 0.524 | 0.703 | 3.8% | 1624 ms |
| multi_query | 0.427 | +0.008 [-0.027, +0.045] | 0.496 | 0.271 / 0.505 | 0.672 | 3.3% | 2917 ms |
| hyde | 0.414 | -0.005 [-0.034, +0.025] | 0.484 | 0.273 / 0.529 | 0.713 | 4.2% | 4124 ms |
| rerank | 0.387 | -0.032 [-0.069, +0.005] | 0.496 | 0.285 / 0.526 | 0.731 | 4.2% | 2897 ms |
| hybrid_rerank | 0.387 | -0.032 [-0.069, +0.005] | 0.496 | 0.285 / 0.526 | 0.731 | 4.2% | 2905 ms |
| source_docling | 0.404 | -0.015 [-0.057, +0.025] | 0.532 | 0.253 / 0.484 | 0.660 | 2.8% | 3339 ms |

Error rate was 0 in every run.

## Decision

**M2's baseline is `experiments/configs/m1b_best.yaml`:** extends the M1a baseline with
section chunking (`chunking.strategy: section`, paragraph-aligned chunks that never cross a
section, `chunk_overlap: 0`) and paragraph-level citations (`generation.cite_unit: paragraph`).

**This is chosen on Evidence-F1, not on Answer-F1.** No Answer-F1 delta in M1b has a 95% CI
excluding 0, including `m1b_best` (+0.022 [-0.015, +0.056]). With 250 questions we cannot
claim better answers; we claim better-grounded citations at no measurable cost to answers:

- Evidence-F1 0.297 to 0.491 (Δ +0.194 [+0.144, +0.242]), evidence ceiling 0.524 to 0.770.
- judge.accuracy 0.500 to 0.568 (direction only, see Judge validation).
- p95 latency 1851 to 1642 ms.

**Costs:** Recall@5 0.703 to 0.665 and false refusal 4.2% to 6.1%.

**Why section over semantic chunking.** `m1b_semantic_best` (semantic chunks + paragraph
citations) is behind on every headline metric (Answer-F1 0.435, Evidence-F1 0.469, judge 0.540)
and needs a sentence-embedding pass to chunk. Section chunking was already the best M1a
chunker; M1b first combined paragraph citations with semantic chunking and only later with
section chunking.

**The combination is sub-additive.** Paragraph citations alone add +0.158 Evidence-F1, section
chunking +0.128, together +0.194. Both reduce the same noise (citing paragraphs of a chunk that
are not evidence), so the second change has less left to fix.

## Judge validation

`judge.accuracy` is a secondary metric. It was audited on 190 LLM-judged answers (100 random
from baseline and m1b_best, plus 45 questions x 2 answers from both runs, labelled blind to run
and to the judge's verdict) against labels from a stronger model (Claude, rubric: correct iff the
answer gives the key information of at least one reference; a list must be all-correct and cover
more than half of a reference).

- Agreement 82.6%, Cohen's kappa 0.65; errors both ways (15 too lenient: partial lists, topic-only
  answers, unchecked numbers; 18 too strict: paraphrases, abbreviations).
- Repeatable: 340/342 identical verdicts on a second pass; reversing the reference order flips
  13/342 (3.8%).
- A stricter prompt (rules for numbers and lists) and gpt-oss:20b as judge both reach 83.2%,
  i.e. no measurable gain, so `prompts/judge.md` and qwen3:14b stay.
- On the 45 paired questions every judge finds m1b_best better than baseline, but the size
  varies: +0.13 (Claude labels, 95% CI [0.00, +0.27]), +0.18 (qwen3:14b), +0.11 (stricter
  prompt), +0.02 (gpt-oss:20b). Judge deltas are read as direction, not size.

## Rejected options

- **rerank / hybrid_rerank:** Answer-F1 -0.032 (CI [-0.069, +0.005]) and about +1 s p95. The two
  runs are near-identical (6/250 top-5 lists and 2/250 answers differ) because per-paper
  retrieval takes a 30-candidate pool that is the whole paper for 234 of 281 papers (the other
  47 have up to 98 chunks), so the reranker mostly reorders the whole paper. Not a verdict on
  rerankers; retry in M2's corpus mode, where the pool is a real subset.
- **multi_query, hyde:** no gain (Evidence-F1 and judge.accuracy lower than baseline) at 1.6x and
  2.2x p95 latency.
- **prompt_v2:** Answer-F1 +0.003, but refusal_recall 0.270 to 0.216 and Evidence-F1 unchanged.
- **source_docling:** parsing the arXiv PDFs with Docling is worse on Answer-F1 (0.404), Evidence-F1
  (0.253), Recall@5 (0.660) and latency (3.3 s p95). 90.4% of gold evidence aligns to a PDF
  paragraph (`data.evidence_aligned_rate` 0.904; 4.0% split across paragraphs) and 1 paper
  could not be downloaded from arXiv (`data.papers_missing` 1). The baseline keeps the
  QASPER-provided text. Part of the gap may be version drift rather than Docling:
  `arxiv.org/pdf/<id>` serves the latest arXiv version, while QASPER's text comes from one
  specific version.

## Caveats

- 25 comparisons at 95% without correction: about 1.3 CIs would exclude 0 by chance.
- Evidence-F1 is comparable across runs only at equal chunking and citation unit; the ceiling
  column shows how much of a change is the unit and how much is the model's citing.
- Metrics other than Answer-F1 (Evidence-F1, judge.accuracy) have no CI in SUMMARY.md; the
  Evidence-F1 CIs above come from the same paired bootstrap, the judge has none.
- `judge.agreement_on_high_f1` is below the 0.95 sanity bar on several runs, lowest on source_docling (0.870),
  rerank / hybrid_rerank (0.929) and model_qwen3_14b (0.850), over only 14-23 LLM-judged
  high-F1 answers each; judge.accuracy for those runs is less trustworthy.

## Follow-ups

- M2 compares the agentic pipeline against `m1b_best`, on the full validation set (1005
  questions) to tighten CIs.
- Re-test reranking and hybrid retrieval in M2's corpus mode.
- Add a paired-bootstrap CI for judge.accuracy deltas, and grow the paired judge audit before
  claiming a judge-measured size.
