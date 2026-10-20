from collections.abc import Sequence

import pytest

from paperlens.config.models import RetrievalConfig
from paperlens.domain import Chunk, ScoredChunk
from paperlens.retrieval.fusion import reciprocal_rank_fusion
from paperlens.retrieval.query import QueryPlan
from paperlens.retrieval.retriever import Retriever


def sc(cid: str) -> ScoredChunk:
    chunk = Chunk(chunk_id=cid, paper_id="P", section="S", text=cid, paragraph_ids=[cid])
    return ScoredChunk(chunk=chunk, score=1.0)


class StubIndex:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.calls: list[tuple[str, int, str | None]] = []

    def search(self, query: str, k: int, paper_id: str | None = None) -> list[ScoredChunk]:
        self.calls.append((query, k, paper_id))
        return [sc(i) for i in self.ids[:k]]


def test_rrf_order() -> None:
    fused = reciprocal_rank_fusion([[sc("a"), sc("b"), sc("c")], [sc("c"), sc("a")]], k=60, top_n=2)
    assert [f.chunk.chunk_id for f in fused] == ["a", "c"]
    assert fused[0].score == 1 / 61 + 1 / 62


def test_rrf_ties_keep_first_seen_order() -> None:
    fused = reciprocal_rank_fusion([[sc("a")], [sc("b")]], k=60, top_n=5)
    assert [f.chunk.chunk_id for f in fused] == ["a", "b"]


def test_dense_uses_top_k() -> None:
    dense, sparse = StubIndex(["a", "b", "c"]), StubIndex([])
    out = Retriever(RetrievalConfig(method="dense", top_k=2), dense, sparse).retrieve("q", "P")
    assert [o.chunk.chunk_id for o in out] == ["a", "b"]
    assert dense.calls == [("q", 2, "P")] and sparse.calls == []


def test_hybrid_uses_candidate_k_and_survives_empty_bm25() -> None:
    dense, sparse = StubIndex(["a", "b", "c"]), StubIndex([])
    cfg = RetrievalConfig(method="hybrid", top_k=2, candidate_k=3)
    out = Retriever(cfg, dense, sparse).retrieve("q")
    assert [o.chunk.chunk_id for o in out] == ["a", "b"]
    assert dense.calls == [("q", 3, None)] and sparse.calls == [("q", 3, None)]


def test_bm25_only() -> None:
    dense, sparse = StubIndex(["a"]), StubIndex(["z"])
    out = Retriever(RetrievalConfig(method="bm25", top_k=1), dense, sparse).retrieve("q")
    assert [o.chunk.chunk_id for o in out] == ["z"] and dense.calls == []


class QueryIndex:
    """Returns a different ranking per query text, and records every query it saw."""

    def __init__(self, by_query: dict[str, list[str]]) -> None:
        self.by_query = by_query
        self.seen: list[str] = []

    def search(self, query: str, k: int, paper_id: str | None = None) -> list[ScoredChunk]:
        self.seen.append(query)
        return [sc(i) for i in self.by_query.get(query, [])[:k]]


@pytest.mark.parametrize("method", ["dense", "bm25", "hybrid"])
def test_identity_plan_keeps_m1a_results(method: str) -> None:
    cfg = RetrievalConfig(method=method, top_k=2, candidate_k=3)  # type: ignore[arg-type]
    plain = Retriever(cfg, StubIndex(["a", "b", "c"]), StubIndex(["c", "a", "b"]))
    planned = Retriever(cfg, StubIndex(["a", "b", "c"]), StubIndex(["c", "a", "b"]))
    assert plain.retrieve("q", "P") == planned.retrieve("q", "P", QueryPlan.identity("q"))


def test_multiple_dense_queries_are_fused_with_rrf() -> None:
    dense = QueryIndex({"a": ["x", "y", "z"], "b": ["z", "y", "w"]})
    cfg = RetrievalConfig(method="dense", top_k=2, candidate_k=4, rrf_k=60)
    plan = QueryPlan(original="q", dense=["a", "b"], sparse=["q"])
    out = Retriever(cfg, dense, QueryIndex({})).retrieve("q", plan=plan)
    assert [o.chunk.chunk_id for o in out] == ["z", "y"] and len(out) == cfg.top_k
    assert dense.seen == ["a", "b"]


def test_hyde_style_plan_searches_bm25_with_sparse_texts_only() -> None:
    dense, sparse = QueryIndex({"hypo": ["a"]}), QueryIndex({"q": ["b"]})
    plan = QueryPlan(original="q", dense=["hypo"], sparse=["q"])
    cfg = RetrievalConfig(method="hybrid", top_k=2, candidate_k=3)
    Retriever(cfg, dense, sparse).retrieve("q", plan=plan)
    assert dense.seen == ["hypo"] and sparse.seen == ["q"]


class ReverseReranker:
    def __init__(self) -> None:
        self.args: tuple[str, list[str], int] | None = None

    def rerank(
        self, query: str, candidates: Sequence[ScoredChunk], top_n: int
    ) -> list[ScoredChunk]:
        self.args = (query, [c.chunk.chunk_id for c in candidates], top_n)
        return list(reversed(candidates))[:top_n]


def test_reranker_gets_candidate_k_pool_and_original_query() -> None:
    dense = StubIndex(["a", "b", "c"])
    cfg = RetrievalConfig(method="dense", top_k=2, candidate_k=3)
    reranker = ReverseReranker()
    out = Retriever(cfg, dense, StubIndex([]), reranker).retrieve("q", "P")
    assert dense.calls == [("q", 3, "P")]
    assert reranker.args == ("q", ["a", "b", "c"], 2)
    assert [o.chunk.chunk_id for o in out] == ["c", "b"]
