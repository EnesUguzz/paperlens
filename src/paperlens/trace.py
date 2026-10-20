"""Follow one question through every RAG step and print what each step produced.

Purpose: A learning and debugging view of the pipeline. Reports show averages over 250
questions; a trace shows the concrete chunks, scores, prompt and answer for one question,
so you can see *why* a metric moved.
Flow: AppConfig + Question -> trace_question() -> Trace (paper stats, query vector,
dense / bm25 / hybrid rankings, the exact prompt, LLM answer, scores) -> render_trace().
Params: retrieval.* (all three methods are shown; retrieval.method is the one that feeds
the prompt), chunking.* and embedding.* (through the index), llm.* (unless --no-llm).
Try: `uv run paperlens trace --index 3 --no-llm` and compare which chunks dense and bm25
rank first; then rerun with a config where chunking.chunk_size is 100.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, get_args

from pydantic import BaseModel

from paperlens.config.models import AppConfig, RetrievalConfig
from paperlens.domain import Paper, PipelineResult, Question, ScoredChunk
from paperlens.evals.records import EvalRecord, build_record
from paperlens.index.embedder import Embedder, build_embedder
from paperlens.index.store import IndexBundle, build_or_load
from paperlens.ingestion.qasper import download_split, load_split, sample_questions
from paperlens.llm.client import LLMClient, LLMResponse, Message, OpenAICompatClient
from paperlens.pipelines.baseline import BaselinePipeline
from paperlens.retrieval.retriever import Retriever

METHODS: tuple[str, ...] = get_args(RetrievalConfig.model_fields["method"].annotation)
VECTOR_HEAD = 8  # how many leading vector components to print
PREVIEW_CHARS = 220  # chunk text shown per ranked chunk


class _NoLLM:
    """Stands in for the client when --no-llm is set; the trace never calls it."""

    def complete(
        self, messages: Sequence[Message], response_format: dict[str, Any] | None = None
    ) -> LLMResponse:
        raise RuntimeError("LLM disabled for this trace")


class Trace(BaseModel):
    question: Question
    paper_title: str
    n_paragraphs: int
    n_chunks_in_paper: int
    gold_evidence_ids: list[str]
    method: str
    query_vector_dim: int
    query_vector_head: list[float]
    rankings: dict[str, list[ScoredChunk]]
    prompt: str
    result: PipelineResult | None
    record: EvalRecord | None


def trace_question(
    cfg: AppConfig,
    question: Question,
    paper: Paper,
    bundle: IndexBundle,
    embedder: Embedder,
    llm: LLMClient | None,
) -> Trace:
    rankings = {
        method: Retriever(
            cfg.retrieval.model_copy(update={"method": method}), bundle.dense, bundle.sparse
        ).retrieve(question.question, paper_id=question.paper_id)
        for method in METHODS
    }
    pipeline = BaselinePipeline(
        Retriever(cfg.retrieval, bundle.dense, bundle.sparse),
        llm or _NoLLM(),
        cfg.generation.prompt_template.read_text(encoding="utf-8"),
        {paper.paper_id: paper.title},
    )
    prompt = pipeline.build_prompt(question, rankings[cfg.retrieval.method])
    result: PipelineResult | None = None
    record: EvalRecord | None = None
    if llm is not None:
        result = pipeline.run(question)
        k_values = [k for k in cfg.eval.k_values if k <= cfg.retrieval.top_k]
        record = build_record(question, result, k_values)
    vector = embedder.embed_query(question.question)
    return Trace(
        question=question,
        paper_title=paper.title,
        n_paragraphs=len(paper.paragraphs),
        n_chunks_in_paper=sum(1 for c in bundle.chunks if c.paper_id == paper.paper_id),
        gold_evidence_ids=sorted({p for r in question.references for p in r.evidence_ids}),
        method=cfg.retrieval.method,
        query_vector_dim=int(vector.shape[0]),
        query_vector_head=[round(float(x), 4) for x in vector[:VECTOR_HEAD]],
        rankings=rankings,
        prompt=prompt,
        result=result,
        record=record,
    )


def run_trace(
    cfg: AppConfig,
    *,
    question_id: str | None = None,
    index: int = 0,
    use_llm: bool = True,
    dataset: tuple[list[Paper], list[Question]] | None = None,
    embedder: Embedder | None = None,
    llm: LLMClient | None = None,
) -> str:
    """Pick a question (by id, or by position in the eval sample) and render its trace."""
    papers, questions = dataset or load_split(download_split(cfg.data, cfg.data.split))
    if question_id is not None:
        found = [q for q in questions if q.question_id == question_id]
        if not found:
            raise ValueError(f"question_id {question_id!r} is not in split {cfg.data.split!r}")
        question = found[0]
    else:
        sample = sample_questions(questions, cfg.data.sample_size, cfg.data.seed)
        if not 0 <= index < len(sample):
            raise ValueError(f"index {index} is outside the sample (0..{len(sample) - 1})")
        question = sample[index]
    paper = next(p for p in papers if p.paper_id == question.paper_id)
    embedder = embedder or build_embedder(cfg.embedding)
    bundle = build_or_load(cfg, papers, embedder)
    try:
        client = (llm or OpenAICompatClient(cfg.llm)) if use_llm else None
        return render_trace(trace_question(cfg, question, paper, bundle, embedder, client))
    finally:
        bundle.dense.close()


def _heading(n: int, title: str) -> str:
    return f"\n{'=' * 78}\n{n}. {title}\n{'=' * 78}"


def _ranking_lines(ranking: list[ScoredChunk], gold: set[str]) -> list[str]:
    if not ranking:
        return ["  (nothing retrieved)"]
    lines = []
    for rank, sc in enumerate(ranking, 1):
        mark = " [gold]" if gold & set(sc.chunk.paragraph_ids) else ""
        lines.append(
            f"  #{rank} score={sc.score:.4f} {sc.chunk.chunk_id} section={sc.chunk.section!r}{mark}"
        )
        lines.append(f"     paragraphs: {', '.join(sc.chunk.paragraph_ids)}")
        text = " ".join(sc.chunk.text.split())
        more = "..." if len(text) > PREVIEW_CHARS else ""
        lines.append(f"     {text[:PREVIEW_CHARS]}{more}")
    return lines


def render_trace(trace: Trace) -> str:
    q = trace.question
    gold = set(trace.gold_evidence_ids)
    out = [_heading(1, "QUESTION")]
    out.append(f"  {q.question}\n  paper: {q.paper_id} - {trace.paper_title}")
    for i, ref in enumerate(q.references, 1):
        out.append(f"  gold answer {i} ({ref.answer_type}): {ref.answer_text}")
    out.append(f"  gold evidence paragraphs: {', '.join(trace.gold_evidence_ids) or '(none)'}")

    out.append(_heading(2, "CHUNKS (offline: chunking + indexing, cached)"))
    out.append(f"  this paper: {trace.n_paragraphs} paragraphs -> {trace.n_chunks_in_paper} chunks")

    out.append(_heading(3, "QUERY EMBEDDING"))
    head = ", ".join(f"{x:+.4f}" for x in trace.query_vector_head)
    out.append(
        f"  {trace.query_vector_dim} numbers, first {len(trace.query_vector_head)}: [{head}, ...]"
    )

    out.append(_heading(4, f"RETRIEVAL (config uses '{trace.method}'; all methods shown)"))
    for method, ranking in trace.rankings.items():
        used = "  <- feeds the prompt" if method == trace.method else ""
        out.append(f"\n  -- {method}{used}")
        out.extend(_ranking_lines(ranking, gold))

    out.append(_heading(5, "PROMPT (exactly what the LLM receives)"))
    out.append(trace.prompt)

    out.append(_heading(6, "LLM ANSWER"))
    if trace.result is None:
        out.append("  skipped (--no-llm)")
    else:
        a = trace.result.answer
        out.append(f"  type: {a.answer_type}\n  answer: {a.text}")
        out.append(f"  cited chunks: {', '.join(a.citations) or '(none)'}")
        out.append(f"  evidence paragraphs: {', '.join(a.evidence_ids) or '(none)'}")
        if a.error:
            out.append(f"  error: {a.error}")
        out.append(
            f"  tokens: {a.usage.prompt_tokens} in / {a.usage.completion_tokens} out, "
            f"{a.usage.latency_ms:.0f} ms"
        )

    out.append(_heading(7, "SCORE"))
    if trace.record is None:
        out.append("  skipped (--no-llm)")
    else:
        r = trace.record
        out.append(f"  answer_f1={r.answer_f1:.3f}  evidence_f1={r.evidence_f1:.3f}")
        if r.retrieval:
            out.append("  " + "  ".join(f"{k}={v:.3f}" for k, v in r.retrieval.items()))
    return "\n".join(out) + "\n"
