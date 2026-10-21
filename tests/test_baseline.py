from pathlib import Path

import pytest

from paperlens.domain import (
    Chunk,
    PipelineResult,
    Question,
    Reference,
    ScoredChunk,
    Segment,
    Usage,
)
from paperlens.llm.client import LLMError
from paperlens.llm.fake import FakeLLM
from paperlens.pipelines.baseline import NO_PASSAGES, BaselinePipeline
from paperlens.retrieval.query import QueryPlan

TEMPLATE = (Path(__file__).resolve().parents[1] / "prompts" / "baseline_answer.md").read_text()
QUESTION = Question(
    question_id="q1",
    paper_id="P",
    question="Which dataset?",
    references=[Reference(answer_type="extractive", answer_text="SQuAD", evidence_ids=["P::s0p1"])],
)


def sc(cid: str, section: str, text: str, pids: list[str], score: float) -> ScoredChunk:
    chunk = Chunk(chunk_id=cid, paper_id="P", section=section, text=text, paragraph_ids=pids)
    return ScoredChunk(chunk=chunk, score=score)


CHUNKS = [
    sc("P::c0", "Intro", "intro text", ["P::s0p0"], 0.9),
    sc("P::c1", "Data", "We use SQuAD", ["P::s0p1", "P::s0p2"], 0.8),
]


class StubRetriever:
    def __init__(self, result: list[ScoredChunk]) -> None:
        self.result = result
        self.plans: list[QueryPlan | None] = []

    def retrieve(
        self, query: str, paper_id: str | None = None, plan: QueryPlan | None = None
    ) -> list[ScoredChunk]:
        self.plans.append(plan)
        return self.result


def run(
    reply: str | Exception, retrieved: list[ScoredChunk] = CHUNKS
) -> tuple[FakeLLM, PipelineResult]:
    llm = FakeLLM([reply])
    pipe = BaselinePipeline(StubRetriever(retrieved), llm, TEMPLATE, {"P": "Tiny Paper"})
    return llm, pipe.run(QUESTION)


def test_answer_with_citations_drops_invalid_indices() -> None:
    llm, result = run('{"answer_type": "extractive", "answer": " SQuAD ", "evidence": [2, 2, 7]}')
    answer = result.answer
    assert answer.text == "SQuAD" and not answer.refused
    assert answer.citations == ["P::c1"]
    assert answer.evidence_ids == ["P::s0p1", "P::s0p2"]
    prompt = llm.calls[0][0][0]["content"]
    assert "Which dataset?" in prompt and "Tiny Paper" in prompt
    assert "[2] (section: Data)" in prompt


def test_unanswerable_is_a_refusal_without_evidence() -> None:
    _, result = run('{"answer_type": "unanswerable", "answer": "n/a", "evidence": [1]}')
    answer = result.answer
    assert answer.refused and answer.text == "Unanswerable" and answer.evidence_ids == []


def test_llm_failure_becomes_counted_error() -> None:
    _, result = run(LLMError("server down"))
    answer = result.answer
    assert answer.refused and answer.error is not None and "LLMError" in answer.error


def test_empty_retrieval_says_so_in_prompt() -> None:
    reply = '{"answer_type": "unanswerable", "answer": "Unanswerable", "evidence": []}'
    llm, _ = run(reply, retrieved=[])
    assert NO_PASSAGES in llm.calls[0][0][0]["content"]


def test_template_without_placeholders_is_rejected() -> None:
    with pytest.raises(ValueError, match="question"):
        BaselinePipeline(StubRetriever([]), FakeLLM([]), "no placeholders", {})


class StubTransformer:
    def plan(self, question: str, title: str) -> QueryPlan:
        return QueryPlan(
            original=question,
            dense=[question, "hypothetical"],
            sparse=[question],
            usage=Usage(prompt_tokens=7, completion_tokens=3, latency_ms=4.0),
            error="LLMError: boom",
        )


def test_pipeline_adds_plan_usage_and_error() -> None:
    llm = FakeLLM(['{"answer_type": "extractive", "answer": "SQuAD", "evidence": [1]}'])
    retriever = StubRetriever(CHUNKS)
    pipe = BaselinePipeline(retriever, llm, TEMPLATE, {"P": "Tiny"}, StubTransformer())
    result = pipe.run(QUESTION)
    assert result.query_error == "LLMError: boom"
    assert result.answer.usage.prompt_tokens == FakeLLM.PROMPT_TOKENS + 7
    assert result.answer.usage.completion_tokens == FakeLLM.COMPLETION_TOKENS + 3
    assert retriever.plans[0] is not None and retriever.plans[0].dense[1] == "hypothetical"


def test_default_pipeline_has_no_query_error_and_normal_usage() -> None:
    _, result = run('{"answer_type": "extractive", "answer": "x", "evidence": [1]}')
    assert result.query_error is None
    assert result.answer.usage.prompt_tokens == FakeLLM.PROMPT_TOKENS


def seg_chunk(cid: str, section: str, pids: list[str]) -> ScoredChunk:
    segments = [Segment(paragraph_id=p, text=f"text of {p}") for p in pids]
    chunk = Chunk(
        chunk_id=cid,
        paper_id="P",
        section=section,
        text=" ".join(s.text for s in segments),
        paragraph_ids=pids,
        segments=segments,
    )
    return ScoredChunk(chunk=chunk, score=0.5)


SEG_CHUNKS = [
    seg_chunk("P::c0", "Intro", ["P::a", "P::b"]),
    seg_chunk("P::c1", "Data", ["P::c", "P::d"]),
]


def test_paragraph_mode_numbers_segments_across_chunks() -> None:
    from paperlens.pipelines.baseline import format_paragraph_context

    context, numbered = format_paragraph_context(SEG_CHUNKS)
    assert [s.paragraph_id for _, s in numbered] == ["P::a", "P::b", "P::c", "P::d"]
    assert "Passage from section: Intro" in context and "Passage from section: Data" in context
    assert "[1] text of P::a" in context and "[4] text of P::d" in context


def test_paragraph_mode_cites_exact_paragraphs() -> None:
    llm = FakeLLM(['{"answer_type": "extractive", "answer": "x", "evidence": [2, 2, 9]}'])
    pipe = BaselinePipeline(
        StubRetriever(SEG_CHUNKS), llm, TEMPLATE, {"P": "T"}, cite_unit="paragraph"
    )
    answer = pipe.run(QUESTION).answer
    assert answer.evidence_ids == ["P::b"]
    assert answer.citations == ["P::c0"]
    assert "[3] text of P::c" in llm.calls[0][0][0]["content"]


class RecordingRetriever(StubRetriever):
    def __init__(self, result: list[ScoredChunk]) -> None:
        super().__init__(result)
        self.paper_ids: list[str | None] = []

    def retrieve(
        self, query: str, paper_id: str | None = None, plan: QueryPlan | None = None
    ) -> list[ScoredChunk]:
        self.paper_ids.append(paper_id)
        return super().retrieve(query, paper_id, plan)


def test_corpus_scope_no_gold_leak() -> None:
    other = Chunk(
        chunk_id="O::c0", paper_id="O", section="Data", text="We use SQuAD", paragraph_ids=[]
    )
    retriever = RecordingRetriever([ScoredChunk(chunk=other, score=0.9)])
    llm = FakeLLM(['{"answer_type": "extractive", "answer": "SQuAD", "evidence": [1]}'])
    titles = {"P": "Gold Title", "O": "Other Title"}
    pipe = BaselinePipeline(retriever, llm, TEMPLATE, titles, scope="corpus")
    pipe.run(QUESTION, text='In "Other Title", which dataset?')
    assert retriever.paper_ids == [None]
    prompt = llm.calls[0][0][0]["content"]
    assert "Other Title" in prompt and "Gold Title" not in prompt
    assert 'In "Other Title", which dataset?' in prompt


def test_generate_matches_run() -> None:
    reply = '{"answer_type": "extractive", "answer": "SQuAD", "evidence": [2]}'
    _, result = run(reply)
    pipe = BaselinePipeline(StubRetriever(CHUNKS), FakeLLM([reply]), TEMPLATE, {"P": "Tiny Paper"})
    answer = pipe.generate(QUESTION, QUESTION.question, "Tiny Paper", CHUNKS)
    assert answer == result.answer
