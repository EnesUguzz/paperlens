import numpy as np
from qdrant_client import QdrantClient

from paperlens.config.models import EmbeddingConfig
from paperlens.domain import Chunk
from paperlens.index.dense import DenseIndex
from paperlens.index.embedder import HashingEmbedder, build_embedder


def chunk(cid: str, paper: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, paper_id=paper, section="S", text=text, paragraph_ids=[cid])


CHUNKS = [
    chunk("A::c0", "A", "squad dataset evaluation"),
    chunk("A::c1", "A", "model reaches f1 score"),
    chunk("B::c0", "B", "squad leaderboard results"),
]


def make_index() -> DenseIndex:
    index = DenseIndex(QdrantClient(":memory:"), "chunks", HashingEmbedder())
    index.build(CHUNKS)
    return index


def test_hashing_embedder_is_normalised_and_deterministic() -> None:
    emb = HashingEmbedder()
    a = emb.embed_query("Squad dataset!")
    assert np.isclose(np.linalg.norm(a), 1.0)
    assert np.allclose(a, emb.embed_query("squad dataset"))
    assert np.isclose(np.linalg.norm(emb.embed_query("")), 1.0)  # never a zero vector


def test_search_ranks_by_similarity() -> None:
    hits = make_index().search("squad dataset", k=2)
    assert hits[0].chunk.chunk_id == "A::c0"
    assert len(hits) == 2


def test_paper_filter() -> None:
    hits = make_index().search("squad", k=5, paper_id="B")
    assert [h.chunk.chunk_id for h in hits] == ["B::c0"]


def test_rebuild_replaces_collection() -> None:
    index = make_index()
    index.build(CHUNKS)
    assert index.count() == 3


def test_build_embedder_hashing() -> None:
    assert isinstance(build_embedder(EmbeddingConfig(model_name="hashing")), HashingEmbedder)
