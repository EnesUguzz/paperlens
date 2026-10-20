from pathlib import Path

from paperlens.config.models import AppConfig, QueryConfig
from paperlens.domain import Paper, Question
from paperlens.index.embedder import HashingEmbedder
from paperlens.index.store import build_or_load
from paperlens.llm.fake import FakeLLM
from paperlens.trace import render_trace, trace_question

REPO = Path(__file__).resolve().parents[1]


def make_cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data={"data_dir": tmp_path / "data"},
        chunking={"strategy": "section", "chunk_size": 20, "chunk_overlap": 0},
        embedding={"model_name": "hashing"},
        retrieval={"method": "hybrid", "top_k": 2, "candidate_k": 4},
        generation={"prompt_template": REPO / "prompts" / "baseline_answer.md"},
        eval={"k_values": [1, 2, 5]},
    )


def test_trace_collects_every_step(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    cfg = make_cfg(tmp_path)
    embedder = HashingEmbedder()
    bundle = build_or_load(cfg, [sample_paper], embedder)
    question = sample_questions[0]  # "Which dataset do they use?" -> SQuAD
    llm = FakeLLM(['{"answer_type": "extractive", "answer": "SQuAD", "evidence": [1]}'])
    try:
        trace = trace_question(cfg, question, sample_paper, bundle, embedder, llm)
    finally:
        bundle.dense.close()

    assert set(trace.rankings) == {"dense", "bm25", "hybrid"}
    assert all(len(r) <= cfg.retrieval.top_k for r in trace.rankings.values())
    assert trace.query_vector_dim == embedder.dim and len(trace.query_vector_head) == 8
    assert trace.gold_evidence_ids and trace.n_chunks_in_paper > 0
    # the prompt shown is exactly the prompt the LLM received
    assert trace.result is not None and trace.record is not None
    assert llm.calls[0][0][0]["content"] == trace.prompt
    assert question.question in trace.prompt
    assert trace.record.answer_f1 == 1.0

    text = render_trace(trace)
    for heading in ("QUESTION", "CHUNKS", "QUERY EMBEDDING", "RETRIEVAL", "PROMPT", "LLM", "SCORE"):
        assert heading in text
    assert "[gold]" in text  # chunks holding gold evidence are marked


def test_trace_without_llm_stops_after_prompt(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    cfg = make_cfg(tmp_path)
    embedder = HashingEmbedder()
    bundle = build_or_load(cfg, [sample_paper], embedder)
    try:
        trace = trace_question(cfg, sample_questions[0], sample_paper, bundle, embedder, None)
    finally:
        bundle.dense.close()
    assert trace.result is None and trace.record is None
    assert "PROMPT" in render_trace(trace) and "skipped" in render_trace(trace)


def test_run_trace_selects_by_id_or_sample_index(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    import pytest

    from paperlens.trace import run_trace

    cfg = make_cfg(tmp_path)
    data = ([sample_paper], sample_questions)
    by_id = run_trace(
        cfg, question_id="q3", use_llm=False, dataset=data, embedder=HashingEmbedder()
    )
    assert "What F1 do they reach?" in by_id
    by_index = run_trace(cfg, index=0, use_llm=False, dataset=data, embedder=HashingEmbedder())
    assert "QUESTION" in by_index
    with pytest.raises(ValueError, match="nope"):
        run_trace(cfg, question_id="nope", use_llm=False, dataset=data, embedder=HashingEmbedder())
    with pytest.raises(ValueError, match="index"):
        run_trace(cfg, index=99, use_llm=False, dataset=data, embedder=HashingEmbedder())


def hyde_cfg(tmp_path: Path) -> AppConfig:
    cfg = make_cfg(tmp_path)
    return cfg.model_copy(
        update={"query": QueryConfig(transform="hyde", hyde_prompt=REPO / "prompts" / "hyde.md")}
    )


def test_trace_shows_query_plan(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    cfg = hyde_cfg(tmp_path)
    embedder = HashingEmbedder()
    bundle = build_or_load(cfg, [sample_paper], embedder)
    question = sample_questions[0]
    llm = FakeLLM(
        [
            "HYPOTHETICAL PARAGRAPH about SQuAD.",  # the single plan call, reused by the run
            '{"answer_type": "extractive", "answer": "SQuAD", "evidence": [1]}',
        ]
    )
    try:
        trace = trace_question(cfg, question, sample_paper, bundle, embedder, llm)
    finally:
        bundle.dense.close()
    text = render_trace(trace)
    assert text.index("QUERY PLAN") < text.index("QUERY EMBEDDING")
    assert "HYPOTHETICAL PARAGRAPH about SQuAD." in text and question.question in text
    assert len(llm.calls) == 2  # planned once, not twice
    assert trace.result is not None
    assert trace.result.answer.usage.prompt_tokens == 2 * FakeLLM.PROMPT_TOKENS


def test_trace_without_llm_shows_identity_plan_and_error(
    tmp_path: Path, sample_paper: Paper, sample_questions: list[Question]
) -> None:
    cfg = hyde_cfg(tmp_path)
    embedder = HashingEmbedder()
    bundle = build_or_load(cfg, [sample_paper], embedder)
    try:
        trace = trace_question(cfg, sample_questions[0], sample_paper, bundle, embedder, None)
    finally:
        bundle.dense.close()
    text = render_trace(trace)
    assert "QUERY PLAN" in text and "LLM disabled" in text
    assert sample_questions[0].question in text
