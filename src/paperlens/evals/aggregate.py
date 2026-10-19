"""Collapse per-question records into one flat metric dict for a run.

Purpose: The numbers that go to MLflow, the report and the ablation table. Names use
dots (MLflow-safe), e.g. 'answer_f1.extractive', 'retrieval.recall_at_5'.
Flow: list[EvalRecord] -> aggregate() -> metrics.json / MLflow / report.md.
Params: eval.k_values.
Try: Watch latency_p95_ms rather than the mean: one slow repair round per 20 questions
barely moves the mean but is exactly what users feel.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import get_args

import numpy as np

from paperlens.domain import AnswerType
from paperlens.evals.records import EvalRecord

P50, P95 = 50, 95


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _rate(records: Sequence[EvalRecord], hit: Callable[[EvalRecord], bool]) -> float:
    return sum(hit(r) for r in records) / len(records) if records else 0.0


def aggregate(records: Sequence[EvalRecord], k_values: Sequence[int]) -> dict[str, float]:
    if not records:
        return {"n_questions": 0.0}
    metrics: dict[str, float] = {
        "n_questions": float(len(records)),
        "answer_f1": _mean([r.answer_f1 for r in records]),
        "evidence_f1": _mean([r.evidence_f1 for r in records]),
    }
    for answer_type in get_args(AnswerType):
        subset = [r for r in records if r.best_ref_type == answer_type]
        metrics[f"n.{answer_type}"] = float(len(subset))
        if subset:
            metrics[f"answer_f1.{answer_type}"] = _mean([r.answer_f1 for r in subset])

    scored = [r.retrieval for r in records if r.retrieval is not None]
    metrics["retrieval.n"] = float(len(scored))
    keys = [f"recall_at_{k}" for k in k_values] + [f"ndcg_at_{k}" for k in k_values] + ["mrr"]
    for key in keys:
        if scored:
            metrics[f"retrieval.{key}"] = _mean([s[key] for s in scored])

    unanswerable = [r for r in records if r.gold_unanswerable]
    answerable = [r for r in records if not r.gold_unanswerable]
    if unanswerable:
        metrics["refusal_recall"] = _rate(unanswerable, lambda r: r.refused)
    if answerable:
        metrics["false_refusal_rate"] = _rate(answerable, lambda r: r.refused)
    metrics["error_rate"] = _rate(records, lambda r: r.error is not None)

    latencies = [r.latency_ms for r in records]
    metrics["latency_p50_ms"] = float(np.percentile(latencies, P50))
    metrics["latency_p95_ms"] = float(np.percentile(latencies, P95))
    metrics["prompt_tokens_mean"] = _mean([r.prompt_tokens for r in records])
    metrics["prompt_tokens_max"] = float(max(r.prompt_tokens for r in records))
    metrics["completion_tokens_mean"] = _mean([r.completion_tokens for r in records])
    metrics["context_words_mean"] = _mean([r.context_words for r in records])
    return metrics
