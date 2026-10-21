"""One Markdown table across all runs: the ablation table for the README.

Purpose: Shows only the parameters that differ between runs next to headline metrics,
plus each run's answer-F1 gain over the reference run with a paired-bootstrap 95% CI,
so the table reads as "this change -> this result, and is it more than noise?".
Flow: evals/results/*/ (params.json, metrics.json, records.jsonl) -> summarize() -> SUMMARY.md.
Params: none (the reference run name is a CLI option, default 'baseline').
Try: After `make ablation`, look for rows whose CI excludes 0, then check latency_p95_ms:
the best config is rarely best on both - that trade-off is your README's story.
"""

from __future__ import annotations

from pathlib import Path

from paperlens.evals.report import IGNORED_PARAM_PREFIXES, RunResult, load_run
from paperlens.evals.stats import BOOTSTRAP_RESAMPLES, CONFIDENCE, paired_bootstrap

DEFAULT_REFERENCE = "baseline"
HEADLINE_METRICS = (
    "answer_f1",
    "gold_recall",
    "answer_words_mean",
    "evidence_f1",
    "evidence_f1_ceiling",
    "retrieval.recall_at_5",
    "refusal_recall",
    "false_refusal_rate",
    "error_rate",
    "judge.accuracy",
    "route.paper_accuracy",
    "route.clarify_rate",
    "latency_p95_ms",
)


def _delta(run: RunResult, reference: RunResult | None) -> str:
    if reference is None or run.run_name == reference.run_name:
        return "-"
    ours = {r.question_id: r.answer_f1 for r in run.records}
    theirs = {r.question_id: r.answer_f1 for r in reference.records}
    if not set(ours) & set(theirs):  # e.g. a test-split run next to validation runs
        return "-"
    result = paired_bootstrap(ours, theirs)
    return f"{result.mean_diff:+.3f} [{result.low:+.3f}, {result.high:+.3f}]"


def summarize(results_dir: Path, reference: str = DEFAULT_REFERENCE) -> str:
    runs = [
        r for d in sorted(results_dir.iterdir()) if d.is_dir() and (r := load_run(d)) is not None
    ]
    if not runs:
        return "No runs found.\n"
    ref_run = next((r for r in runs if r.run_name == reference), None)
    keys = sorted({k for r in runs for k in r.params if not k.startswith(IGNORED_PARAM_PREFIXES)})
    varying = [k for k in keys if len({str(r.params.get(k)) for r in runs}) > 1]
    header = ["run", *varying, *HEADLINE_METRICS, f"Δ answer_f1 vs {reference} [95% CI]"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in runs:
        params = [str(r.params.get(k, "-")) for k in varying]
        metrics = [f"{r.metrics[m]:.3f}" if m in r.metrics else "-" for m in HEADLINE_METRICS]
        lines.append("| " + " | ".join([r.run_name, *params, *metrics, _delta(r, ref_run)]) + " |")
    comparisons = sum(1 for r in runs if ref_run is not None and r is not ref_run)
    lines.append("")
    lines.append(
        f"Δ: paired bootstrap ({BOOTSTRAP_RESAMPLES} resamples, {CONFIDENCE:.0%} CI) over the "
        f"questions both runs answered. With {comparisons} comparisons and no correction, "
        f"about {comparisons * (1 - CONFIDENCE):.1f} CI(s) would exclude 0 by chance alone."
    )
    return "\n".join(lines) + "\n"
