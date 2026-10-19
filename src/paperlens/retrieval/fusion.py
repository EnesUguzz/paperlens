"""Reciprocal Rank Fusion: merge rankings using only ranks, never raw scores.

Purpose: BM25 scores and cosine similarities live on different scales; RRF sidesteps
score calibration by summing 1/(k + rank) over every ranking a chunk appears in.
Flow: [dense ranking, bm25 ranking] -> reciprocal_rank_fusion() -> fused top_n.
Params: retrieval.rrf_k (k), retrieval.top_k (top_n).
Try: rrf_k=10 vs 60: with 10, whatever each retriever ranks first dominates; with 60,
chunks found by both retrievers float up even if neither ranked them first.
"""

from __future__ import annotations

from collections.abc import Sequence

from paperlens.domain import ScoredChunk


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]], k: int, top_n: int
) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    first_seen: dict[str, ScoredChunk] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(chunk_id, item)
    order = sorted(scores, key=lambda cid: -scores[cid])  # stable: ties keep first-seen order
    return [ScoredChunk(chunk=first_seen[cid].chunk, score=scores[cid]) for cid in order[:top_n]]
