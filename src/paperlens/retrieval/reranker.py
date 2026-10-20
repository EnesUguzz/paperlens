"""Reranker interface: reorder a candidate pool and keep the best few.

Purpose: Lets the Retriever accept a second-stage scorer without knowing which model it
is. The real cross-encoder and a test fake arrive in M1b Task 6.
Flow: Retriever pool (candidate_k chunks) -> Reranker.rerank(query, pool, top_n) -> top_n
ScoredChunk, best first.
Params: none yet (the implementation will read reranker.* config).
Try: Read Retriever.retrieve to see where the pool is cut and the hook is called.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from paperlens.domain import ScoredChunk


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]: ...
