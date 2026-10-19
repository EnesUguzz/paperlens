"""Retrieval metrics against gold evidence paragraphs: Recall@k, MRR, nDCG@k.

Purpose: Measures retrieval on its own, before the LLM can hide or add errors. A chunk
counts as relevant if it contains at least one gold paragraph id.
Flow: PipelineResult.retrieved (as lists of paragraph ids) + Question.references ->
best_over_references() -> evals/records.py.
Params: eval.k_values (which cut-offs are reported).
Try: Compare Recall@5 for chunk_size 100 vs 400; bigger chunks cover more paragraphs, so
recall rises "for free" - then look at context_words_mean to see the price.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from paperlens.domain import Reference


def recall_at_k(ranked: Sequence[Sequence[str]], gold: set[str], k: int) -> float:
    covered = {pid for chunk in ranked[:k] for pid in chunk}
    return len(gold & covered) / len(gold)


def mrr(ranked: Sequence[Sequence[str]], gold: set[str]) -> float:
    for rank, chunk in enumerate(ranked, start=1):
        if gold & set(chunk):
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[Sequence[str]], gold: set[str], k: int) -> float:
    """Binary relevance per chunk, but a chunk only counts if it brings a gold paragraph
    not seen in an earlier chunk. Overlapping (fixed) chunks would otherwise credit one
    paragraph several times and push nDCG above 1. At most min(k, |gold|) chunks can be
    relevant, which is exactly the ideal ranking below, so the score stays in [0, 1]."""
    seen: set[str] = set()
    dcg = 0.0
    for rank, chunk in enumerate(ranked[:k], start=1):
        new_gold = (gold & set(chunk)) - seen
        if new_gold:
            dcg += 1.0 / math.log2(rank + 1)
            seen |= new_gold
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(gold)) + 1))
    return dcg / ideal


def best_over_references(
    ranked: Sequence[Sequence[str]], references: Sequence[Reference], k_values: Sequence[int]
) -> dict[str, float] | None:
    golds = [set(r.evidence_ids) for r in references if r.evidence_ids]
    if not golds:
        return None
    out: dict[str, float] = {}
    for k in k_values:
        out[f"recall_at_{k}"] = max(recall_at_k(ranked, g, k) for g in golds)
        out[f"ndcg_at_{k}"] = max(ndcg_at_k(ranked, g, k) for g in golds)
    out["mrr"] = max(mrr(ranked, g) for g in golds)
    return out
