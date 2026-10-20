"""Domain objects shared by every module (papers, questions, chunks, answers).

Purpose: One vocabulary for the whole pipeline, so modules exchange typed objects instead
of dicts. Paragraph ids are the join key between gold evidence and retrieved chunks.
Flow: ingestion creates Paper/Question/Chunk -> index and retrieval pass ScoredChunk ->
pipelines return PipelineResult -> evals score it.
Params: none.
Try: Print a few Question.references to see how often QASPER annotators disagree.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AnswerType = Literal["extractive", "abstractive", "yes_no", "unanswerable"]

# The official QASPER evaluator compares unanswerable predictions against this string.
UNANSWERABLE_TEXT = "Unanswerable"


class Paragraph(BaseModel):
    paragraph_id: str
    section: str
    text: str
    is_caption: bool = False


class Paper(BaseModel):
    paper_id: str
    title: str
    abstract: str
    paragraphs: list[Paragraph]


class Reference(BaseModel):
    """One annotator's answer. A question usually has 1-3 of these."""

    answer_type: AnswerType
    answer_text: str
    evidence_ids: list[str]
    unmatched_evidence: int = 0


class Question(BaseModel):
    question_id: str
    paper_id: str
    question: str
    references: list[Reference]


class Chunk(BaseModel):
    chunk_id: str
    paper_id: str
    section: str
    text: str
    paragraph_ids: list[str]


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
        )


class Answer(BaseModel):
    question_id: str
    answer_type: AnswerType
    text: str
    citations: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    refused: bool = False
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)


class PipelineResult(BaseModel):
    answer: Answer
    retrieved: list[ScoredChunk]
    retrieval_ms: float
    query_error: str | None = None  # query-transform failure (pipeline fell back to the question)
