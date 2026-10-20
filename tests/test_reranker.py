from pathlib import Path

import pytest

from paperlens.config.models import AppConfig, RerankConfig, RetrievalConfig
from paperlens.domain import Chunk, Paper, Question, ScoredChunk
from paperlens.index.embedder import HashingEmbedder
from paperlens.index.store import build_or_load
from paperlens.llm.fake import FakeLLM
from paperlens.pipelines.factory import build_pipeline
from paperlens.retrieval.reranker import LexicalReranker, build_reranker
from paperlens.retrieval.retriever import Retriever
from paperlens.trace import render_trace, trace_question

REPO = Path(__file__).resolve().parents[1]


def sc(cid: str, text: str = "", score: float = 1.0) -> ScoredChunk:
    chunk = Chunk(chunk_id=cid, paper_id="P", section="S", text=text or cid, paragraph_ids=[cid])
    return ScoredChunk(chunk=chunk, score=score)


class StubIndex:
    def __init__(self, ids: list[str], texts: dict[str, str] | None = None) -> None:
        self.ids = ids
        self.texts = texts or {}
        self.calls: list[tuple[str, int, str | None]] = []

    def search(self, query: str, k: int, paper_id: str | None = None) -> list[ScoredChunk]:
        self.calls.append((query, k, paper_id))
        return [sc(i, self.texts.get(i, "")) for i in self.ids[:k]]


def test_disabled_rerank_builds_nothing() -> None:
    assert build_reranker(RerankConfig()) is None


def test_lexical_reranker_reorders_pool() -> None:
    pool = [sc("a", "the weather is nice"), sc("b", "we use the SQuAD dataset")]
    out = LexicalReranker().rerank("which dataset do they use", pool, top_n=1)
    assert [o.chunk.chunk_id for o in out] == ["b"]
    assert out[0].score == 2.0  # shared distinct words: 'dataset' and 'use'


def test_rerank_ties_keep_first_stage_order() -> None:
    pool = [sc("a", "zzz"), sc("b", "yyy"), sc("c", "xxx")]
    out = LexicalReranker().rerank("nothing shared", pool, top_n=3)
    assert [o.chunk.chunk_id for o in out] == ["a", "b", "c"]


def test_retriever_reranks_candidate_pool() -> None:
    ids = [f"c{i}" for i in range(30)]
    dense = StubIndex(ids, {"c17": "the squad dataset"})
    cfg = RetrievalConfig(method="dense", top_k=5, candidate_k=30)
    out = Retriever(cfg, dense, StubIndex([]), LexicalReranker()).retrieve("squad dataset", "P")
    assert dense.calls == [("squad dataset", 30, "P")]
    assert len(out) == 5
    assert out[0].chunk.chunk_id == "c17"


def test_hybrid_pool_is_fused_then_reranked() -> None:
    dense = StubIndex(["a", "b", "c"], {"c": "we use squad"})
    sparse = StubIndex(["b", "a", "d"], {"d": "squad squad"})
    cfg = RetrievalConfig(method="hybrid", top_k=2, candidate_k=3)
    out = Retriever(cfg, dense, sparse, LexicalReranker()).retrieve("squad", "P")
    assert dense.calls == [("squad", 3, "P")] and sparse.calls == [("squad", 3, "P")]
    # pool = RRF over both rankings (4 unique chunks cut to 3); the reranker picks from it
    assert {o.chunk.chunk_id for o in out} <= {"a", "b", "c", "d"}
    assert len(out) == 2
    assert out[0].chunk.chunk_id in {"c", "d"}  # the lexical match beats the RRF winners


def _cfg(tmp_path: Path, rerank_enabled: bool) -> AppConfig:
    return AppConfig(
        data={"data_dir": tmp_path / "data"},
        chunking={"strategy": "section", "chunk_size": 20, "chunk_overlap": 0},
        embedding={"model_name": "hashing"},
        retrieval={"method": "hybrid", "top_k": 2, "candidate_k": 4},
        rerank={"enabled": rerank_enabled, "model_name": "lexical"},
        generation={"prompt_template": REPO / "prompts" / "baseline_answer.md"},
        eval={"k_values": [1, 2]},
    )


def test_build_pipeline_builds_reranker_from_config(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    cfg = _cfg(tmp_path, True)
    bundle = build_or_load(cfg, [sample_paper], HashingEmbedder())
    try:
        pipeline = build_pipeline(cfg, bundle, FakeLLM([]), {sample_paper.paper_id: "T"})
        question = sample_questions[0]
        out = pipeline.retrieve(question, pipeline.plan(question))
        assert 0 < len(out) <= cfg.retrieval.top_k
    finally:
        bundle.dense.close()


def test_trace_shows_pool_and_rerank_only_when_enabled(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    texts = {}
    for enabled in (True, False):
        cfg = _cfg(tmp_path, enabled)
        embedder = HashingEmbedder()
        bundle = build_or_load(cfg, [sample_paper], embedder)
        try:
            trace = trace_question(cfg, sample_questions[0], sample_paper, bundle, embedder, None)
        finally:
            bundle.dense.close()
        texts[enabled] = render_trace(trace)
        if enabled:
            assert trace.first_stage_pool is not None and trace.reranked is not None
            assert len(trace.reranked) <= cfg.retrieval.top_k
            assert len(trace.first_stage_pool) <= cfg.retrieval.candidate_k
    assert "first stage pool" in texts[True] and "after rerank" in texts[True]
    assert "first stage pool" not in texts[False] and "after rerank" not in texts[False]


@pytest.mark.gpu
def test_cross_encoder_prefers_relevant_passage() -> None:
    reranker = build_reranker(RerankConfig(enabled=True))
    assert reranker is not None
    pool = [sc("weather", "The weather is nice."), sc("squad", "We use SQuAD.")]
    out = reranker.rerank("Which dataset?", pool, top_n=2)
    assert [o.chunk.chunk_id for o in out] == ["squad", "weather"]
    assert out[0].score > out[1].score
