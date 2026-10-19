"""One Markdown table across all runs: the ablation table for the README.

Purpose: Shows only the parameters that differ between runs next to headline metrics,
so the table reads as "this change -> this result".
Flow: evals/results/*/ (params.json, metrics.json) -> summarize() -> SUMMARY.md.
Params: none.
Try: After `make ablation`, sort SUMMARY.md mentally by answer_f1 and by latency_p95_ms:
the best config is rarely best on both - that trade-off is your README's story.
"""

from __future__ import annotations

from pathlib import Path

from paperlens.evals.report import IGNORED_PARAM_PREFIXES, load_run

HEADLINE_METRICS = (
    "answer_f1",
    "evidence_f1",
    "retrieval.recall_at_5",
    "refusal_recall",
    "false_refusal_rate",
    "error_rate",
    "latency_p95_ms",
)


def summarize(results_dir: Path) -> str:
    runs = [
        r for d in sorted(results_dir.iterdir()) if d.is_dir() and (r := load_run(d)) is not None
    ]
    if not runs:
        return "No runs found.\n"
    keys = sorted({k for r in runs for k in r.params if not k.startswith(IGNORED_PARAM_PREFIXES)})
    varying = [k for k in keys if len({str(r.params.get(k)) for r in runs}) > 1]
    header = ["run", *varying, *HEADLINE_METRICS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in runs:
        params = [str(r.params.get(k, "-")) for k in varying]
        metrics = [f"{r.metrics[m]:.3f}" if m in r.metrics else "-" for m in HEADLINE_METRICS]
        lines.append("| " + " | ".join([r.run_name, *params, *metrics]) + " |")
    return "\n".join(lines) + "\n"
