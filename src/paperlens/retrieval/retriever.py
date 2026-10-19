"""One entry point for retrieval: dense, BM25 or hybrid, chosen by config.

Purpose: Pipelines call retrieve() and never know which method runs, so switching method
is a one-line YAML change.
Flow: question (+ paper_id in single-paper mode) -> Retriever -> top_k ScoredChunk
-> pipelines/baseline.py.
Params: retrieval.method, retrieval.top_k, retrieval.candidate_k, retrieval.rrf_k.
Try: Run dense, bm25 and hybrid on the same sample and open the per-type breakdown in the
report: bm25 usually wins on extractive questions about named datasets and metrics.
"""

from __future__ import annotations

from typing import Protocol

from paperlens.config.models import RetrievalConfig
from paperlens.domain import ScoredChunk
from paperlens.retrieval.fusion import reciprocal_rank_fusion


class SearchIndex(Protocol):
    def search(self, query: str, k: int, paper_id: str | None = None) -> list[ScoredChunk]: ...


class Retriever:
    def __init__(self, cfg: RetrievalConfig, dense: SearchIndex, sparse: SearchIndex) -> None:
        self._cfg = cfg
        self._dense = dense
        self._sparse = sparse

    def retrieve(self, query: str, paper_id: str | None = None) -> list[ScoredChunk]:
        cfg = self._cfg
        if cfg.method == "dense":
            return self._dense.search(query, cfg.top_k, paper_id)
        if cfg.method == "bm25":
            return self._sparse.search(query, cfg.top_k, paper_id)
        rankings = [
            self._dense.search(query, cfg.candidate_k, paper_id),
            self._sparse.search(query, cfg.candidate_k, paper_id),
        ]
        return reciprocal_rank_fusion(rankings, k=cfg.rrf_k, top_n=cfg.top_k)
