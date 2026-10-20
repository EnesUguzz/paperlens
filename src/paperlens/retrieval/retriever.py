"""One entry point for retrieval: dense, BM25 or hybrid, chosen by config.

Purpose: Pipelines call retrieve() and never know which method runs, so switching method
is a one-line YAML change.
Flow: question (+ paper_id in single-paper mode, + optional QueryPlan) -> search per plan
text -> RRF when there are several rankings -> optional reranker -> top_k ScoredChunk
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
from paperlens.retrieval.query import QueryPlan
from paperlens.retrieval.reranker import Reranker


class SearchIndex(Protocol):
    def search(self, query: str, k: int, paper_id: str | None = None) -> list[ScoredChunk]: ...


class Retriever:
    def __init__(
        self,
        cfg: RetrievalConfig,
        dense: SearchIndex,
        sparse: SearchIndex,
        reranker: Reranker | None = None,
    ) -> None:
        self._cfg = cfg
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker

    def retrieve(
        self, query: str, paper_id: str | None = None, plan: QueryPlan | None = None
    ) -> list[ScoredChunk]:
        cfg = self._cfg
        plan = plan or QueryPlan.identity(query)
        searches: list[tuple[SearchIndex, str]] = []
        if cfg.method in ("dense", "hybrid"):
            searches += [(self._dense, text) for text in plan.dense]
        if cfg.method in ("bm25", "hybrid"):
            searches += [(self._sparse, text) for text in plan.sparse]

        if len(searches) == 1 and self._reranker is None:
            index, text = searches[0]  # M1a path: one ranking, no fusion, nothing to rerank
            return index.search(text, cfg.top_k, paper_id)

        rankings = [index.search(text, cfg.candidate_k, paper_id) for index, text in searches]
        pool_size = cfg.candidate_k if self._reranker else cfg.top_k
        if len(rankings) == 1:
            pool = rankings[0][:pool_size]  # keep the index's own scores
        else:
            pool = reciprocal_rank_fusion(rankings, k=cfg.rrf_k, top_n=pool_size)
        if self._reranker is None:
            return pool
        return self._reranker.rerank(plan.original, pool, cfg.top_k)
