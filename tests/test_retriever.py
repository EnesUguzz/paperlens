from paperlens.config.models import RetrievalConfig
from paperlens.domain import Chunk, ScoredChunk
from paperlens.retrieval.fusion import reciprocal_rank_fusion
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
