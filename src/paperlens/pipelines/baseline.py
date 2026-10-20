"""Naive RAG: retrieve once, answer once. The reference every later system must beat.

Purpose: The simplest pipeline that can score on QASPER. M2's agentic graph is judged by
how much it improves on this, at what extra latency.
Flow: Question -> plan() (identity unless a transformer is set) -> Retriever.retrieve() -> numbered context -> prompt -> structured_call()
-> AnswerDraft -> Answer (with citations and evidence paragraph ids) -> evals.
Params: generation.prompt_template, retrieval.* (through the Retriever), llm.* (through
the client).
Try: Edit prompts/baseline_answer.md (e.g. drop the "shortest exact span" rule) and
compare answer_f1.extractive in the report - prompt wording is a parameter too.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, Field

from paperlens.domain import (
    UNANSWERABLE_TEXT,
    Answer,
    AnswerType,
    PipelineResult,
    Question,
    ScoredChunk,
)
from paperlens.llm.client import LLMClient, LLMError
from paperlens.llm.structured import StructuredOutputError, structured_call
from paperlens.retrieval.query import IdentityTransformer, QueryPlan, QueryTransformer

PLACEHOLDERS = ("{{title}}", "{{context}}", "{{question}}")
NO_PASSAGES = "(no passages were retrieved)"


class AnswerDraft(BaseModel):
    """What the LLM must return; passage numbers are 1-based positions in the context."""

    answer_type: AnswerType
    answer: str
    evidence: list[int] = Field(default_factory=list)


class RetrieverLike(Protocol):
    def retrieve(
        self, query: str, paper_id: str | None = None, plan: QueryPlan | None = None
    ) -> list[ScoredChunk]: ...


class Pipeline(Protocol):
    def run(self, question: Question) -> PipelineResult: ...


def format_context(retrieved: Sequence[ScoredChunk]) -> str:
    if not retrieved:
        return NO_PASSAGES
    return "\n\n".join(
        f"[{i}] (section: {sc.chunk.section})\n{sc.chunk.text}" for i, sc in enumerate(retrieved, 1)
    )


def render_prompt(template: str, title: str, context: str, question: str) -> str:
    return (
        template.replace("{{title}}", title)
        .replace("{{context}}", context)
        .replace("{{question}}", question)
    )


class BaselinePipeline:
    def __init__(
        self,
        retriever: RetrieverLike,
        llm: LLMClient,
        template: str,
        titles: Mapping[str, str],
        transformer: QueryTransformer | None = None,
    ) -> None:
        missing = [p for p in PLACEHOLDERS if p not in template]
        if missing:
            raise ValueError(f"prompt template is missing placeholders: {missing}")
        self._retriever = retriever
        self._llm = llm
        self._template = template
        self._titles = titles
        self._transformer = transformer or IdentityTransformer()

    def build_prompt(self, question: Question, retrieved: Sequence[ScoredChunk]) -> str:
        """The exact text sent to the LLM (also shown by `paperlens trace`)."""
        return render_prompt(
            self._template,
            title=self._titles.get(question.paper_id, ""),
            context=format_context(retrieved),
            question=question.question,
        )

    def plan(self, question: Question) -> QueryPlan:
        """The texts to search with (the question itself unless a transformer is set)."""
        title = self._titles.get(question.paper_id, "")
        return self._transformer.plan(question.question, title)

    def retrieve(self, question: Question, plan: QueryPlan) -> list[ScoredChunk]:
        return self._retriever.retrieve(question.question, paper_id=question.paper_id, plan=plan)

    def run(self, question: Question) -> PipelineResult:
        started = time.perf_counter()
        plan = self.plan(question)
        retrieved = self.retrieve(question, plan)
        retrieval_ms = (time.perf_counter() - started) * 1000
        messages = [{"role": "user", "content": self.build_prompt(question, retrieved)}]
        try:
            draft, usage = structured_call(self._llm, messages, AnswerDraft)
        except (LLMError, StructuredOutputError) as err:
            failed = Answer(
                question_id=question.question_id,
                answer_type="unanswerable",
                text=UNANSWERABLE_TEXT,
                refused=True,
                error=f"{type(err).__name__}: {err}",
                usage=plan.usage,
            )
            return PipelineResult(
                answer=failed,
                retrieved=retrieved,
                retrieval_ms=retrieval_ms,
                query_error=plan.error,
            )

        refused = draft.answer_type == "unanswerable"
        cited = (
            []
            if refused
            else [
                retrieved[i - 1] for i in dict.fromkeys(draft.evidence) if 1 <= i <= len(retrieved)
            ]
        )
        answer = Answer(
            question_id=question.question_id,
            answer_type=draft.answer_type,
            text=UNANSWERABLE_TEXT if refused else draft.answer.strip(),
            citations=[sc.chunk.chunk_id for sc in cited],
            evidence_ids=list(dict.fromkeys(pid for sc in cited for pid in sc.chunk.paragraph_ids)),
            refused=refused,
            usage=plan.usage + usage,
        )
        return PipelineResult(
            answer=answer,
            retrieved=retrieved,
            retrieval_ms=retrieval_ms,
            query_error=plan.error,
        )
