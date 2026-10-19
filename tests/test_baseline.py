from pathlib import Path

import pytest

from paperlens.domain import Chunk, PipelineResult, Question, Reference, ScoredChunk
from paperlens.llm.client import LLMError
from paperlens.llm.fake import FakeLLM
from paperlens.pipelines.baseline import NO_PASSAGES, BaselinePipeline

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

    def retrieve(self, query: str, paper_id: str | None = None) -> list[ScoredChunk]:
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
