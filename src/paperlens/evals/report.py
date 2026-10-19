"""Markdown report per run: what changed vs the reference, how metrics moved, and why.

Purpose: Closes the experiment loop in one file - changed parameters, metric deltas, the
questions that got worse, and warnings (e.g. prompts near the context window).
Flow: RunResult (+ reference RunResult from results_dir) -> render_report() ->
evals/results/<run_name>/report.md. save_run/load_run persist runs as plain JSON.
Params: eval.reference_run, eval.worst_examples, llm.context_window (for the warning).
Try: After any run, read 'Biggest regressions' first: three concrete questions usually
explain a metric drop faster than any table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperlens.evals.records import EvalRecord

IGNORED_PARAM_PREFIXES = ("eval.", "meta.git_")  # git keys change on every commit
TRUNCATION_WARN_RATIO = 0.95
PARAMS_FILE, METRICS_FILE, RECORDS_FILE = "params.json", "metrics.json", "records.jsonl"


@dataclass
class RunResult:
    run_name: str
    params: dict[str, Any]
    metrics: dict[str, float]
    records: list[EvalRecord]


def save_run(run: RunResult, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    params = json.dumps(run.params, indent=2, default=str)
    (run_dir / PARAMS_FILE).write_text(params, encoding="utf-8")
    (run_dir / METRICS_FILE).write_text(json.dumps(run.metrics, indent=2), encoding="utf-8")
    (run_dir / RECORDS_FILE).write_text(
        "".join(r.model_dump_json() + "\n" for r in run.records), encoding="utf-8"
    )


def load_run(run_dir: Path) -> RunResult | None:
    if not (run_dir / METRICS_FILE).exists():
        return None
    lines = (run_dir / RECORDS_FILE).read_text(encoding="utf-8").splitlines()
    return RunResult(
        run_name=run_dir.name,
        params=json.loads((run_dir / PARAMS_FILE).read_text(encoding="utf-8")),
        metrics=json.loads((run_dir / METRICS_FILE).read_text(encoding="utf-8")),
        records=[EvalRecord.model_validate_json(line) for line in lines if line],
    )


def changed_params(
    current: dict[str, Any], reference: dict[str, Any]
) -> list[tuple[str, Any, Any]]:
    keys = sorted(set(current) | set(reference))
    return [
        (k, current.get(k), reference.get(k))
        for k in keys
        if not k.startswith(IGNORED_PARAM_PREFIXES) and current.get(k) != reference.get(k)
    ]


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def _example(r: EvalRecord, before: float | None) -> str:
    score = f"{before:.2f} -> {r.answer_f1:.2f}" if before is not None else f"{r.answer_f1:.2f}"
    return (
        f"- **{r.question}** (`{r.question_id}`, answer_f1 {score})\n"
        f"  - gold: {' | '.join(r.gold_texts)}\n"
        f"  - predicted: {r.prediction}\n"
        f"  - top context: {r.top_context_preview}\n"
    )


def render_report(run: RunResult, reference: RunResult | None, worst_n: int) -> str:
    out = [f"# Run `{run.run_name}`\n"]

    window = run.params.get("llm.context_window")
    peak = run.metrics.get("prompt_tokens_max", 0.0)
    if isinstance(window, int) and peak >= TRUNCATION_WARN_RATIO * window:
        out.append(
            f"> **WARNING:** the largest prompt used {peak:.0f} tokens, at least "
            f"{TRUNCATION_WARN_RATIO:.0%} of llm.context_window={window}. Ollama silently cuts "
            "longer prompts; lower retrieval.top_k / chunk_size or raise OLLAMA_CONTEXT_LENGTH.\n"
        )

    if reference is None:
        out.append("No reference run: showing absolute metrics only.\n")
    else:
        out.append(f"## Changed parameters vs `{reference.run_name}`\n")
        changes = changed_params(run.params, reference.params)
        if changes:
            out.append("| Parameter | Reference | This run |\n|---|---|---|")
            out.extend(f"| `{k}` | {ref} | {cur} |" for k, cur, ref in changes)
            out.append("")
        else:
            out.append("Identical parameters (differences below are noise).\n")

    out.append("## Metrics\n")
    out.append("| Metric | This run | Reference | Delta |\n|---|---|---|---|")
    for name in sorted(run.metrics):
        cur = run.metrics[name]
        ref = reference.metrics.get(name) if reference else None
        delta = "-" if ref is None else f"{cur - ref:+.3f}"
        out.append(f"| {name} | {_fmt(cur)} | {_fmt(ref)} | {delta} |")
    out.append("")

    if reference is not None:
        before = {r.question_id: r.answer_f1 for r in reference.records}
        moved = [
            r
            for r in run.records
            if r.question_id in before and r.answer_f1 < before[r.question_id]
        ]
        moved.sort(key=lambda r: r.answer_f1 - before[r.question_id])
        out.append("## Biggest regressions\n")
        out.extend(_example(r, before[r.question_id]) for r in moved[:worst_n])
        if not moved:
            out.append("None.\n")
    else:
        out.append("## Lowest-scoring questions\n")
        lowest = sorted(run.records, key=lambda r: r.answer_f1)[:worst_n]
        out.extend(_example(r, None) for r in lowest)
    return "\n".join(out)
