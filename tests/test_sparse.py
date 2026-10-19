from paperlens.domain import Chunk
from paperlens.index.sparse import SparseIndex


def chunk(cid: str, paper: str, text: str) -> Chunk:
    return Chunk(chunk_id=cid, paper_id=paper, section="S", text=text, paragraph_ids=[cid])


CHUNKS = [
    chunk("A::c0", "A", "We use the SQuAD dataset"),
    chunk("A::c1", "A", "The model reaches 81 F1"),
    chunk("B::c0", "B", "SQuAD leaderboard results"),
]


def index() -> SparseIndex:
    return SparseIndex(CHUNKS, k1=1.5, b=0.75)


def test_global_search_finds_keyword() -> None:
    ids = {h.chunk.chunk_id for h in index().search("squad", k=5)}
    assert ids == {"A::c0", "B::c0"}


def test_paper_filter() -> None:
    assert [h.chunk.chunk_id for h in index().search("squad", k=5, paper_id="A")] == ["A::c0"]


def test_stopword_only_query_returns_empty() -> None:
    assert index().search("the of and", k=5) == []


def test_unknown_words_return_empty() -> None:
    assert index().search("zebra", k=5) == []


def test_unknown_paper_returns_empty() -> None:
    assert index().search("squad", k=5, paper_id="nope") == []


def test_k_larger_than_corpus() -> None:
    assert len(index().search("squad", k=50, paper_id="A")) == 1
