"""Naive RAG: retrieve once, answer once. The reference every later system must beat.

Purpose: The simplest pipeline that can score on QASPER. M2's agentic graph is judged by
how much it improves on this, at what extra latency.
Flow: Question -> plan() (identity unless a transformer is set) -> Retriever.retrieve() -> numbered context -> prompt -> structured_call()
-> AnswerDraft -> Answer (with citations and evidence paragraph ids) -> evals.
Params: generation.prompt_template, generation.cite_unit (chunk or paragraph citations), retrieval.* (through the Retriever), llm.* (through
the client).
Try: Edit prompts/baseline_answer.md (e.g. drop the "shortest exact span" rule) and
compare answer_f1.extractive in the report - prompt wording is a parameter too.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from paperlens.domain import (
    UNANSWERABLE_TEXT,
    Answer,
    AnswerType,
    PipelineResult,
    Question,
    ScoredChunk,
    Segment,
    Usage,
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


def format_paragraph_context(
    retrieved: Sequence[ScoredChunk],
) -> tuple[str, list[tuple[ScoredChunk, Segment]]]:
    """Number every paragraph segment across the whole context (1-based), grouped by chunk.

    Returns the context text and the (chunk, segment) behind each number, in order."""
    if not retrieved:
        return NO_PASSAGES, []
    numbered: list[tuple[ScoredChunk, Segment]] = []
    blocks: list[str] = []
    for sc in retrieved:
        lines = [f"Passage from section: {sc.chunk.section}"]
        for segment in sc.chunk.segments:
            numbered.append((sc, segment))
            lines.append(f"[{len(numbered)}] {segment.text}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks), numbered


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
        cite_unit: Literal["chunk", "paragraph"] = "chunk",
        scope: Literal["paper", "corpus"] = "paper",
    ) -> None:
        missing = [p for p in PLACEHOLDERS if p not in template]
        if missing:
            raise ValueError(f"prompt template is missing placeholders: {missing}")
        self._retriever = retriever
        self._llm = llm
        self._template = template
        self._titles = titles
        self._transformer = transformer or IdentityTransformer()
        self._cite_unit = cite_unit
        # 'corpus' (M2 corpus_rag) never reads question.paper_id: it searches every paper
        # and shows the title of the top chunk's paper, so the gold paper cannot leak.
        self._scope = scope

    def _title(self, question: Question, retrieved: Sequence[ScoredChunk] = ()) -> str:
        if self._scope == "paper":
            return self._titles.get(question.paper_id, "")
        return self._titles.get(retrieved[0].chunk.paper_id, "") if retrieved else ""

    def build_prompt(self, question: Question, retrieved: Sequence[ScoredChunk]) -> str:
        """The exact text sent to the LLM (also shown by `paperlens trace`)."""
        return render_prompt(
            self._template,
            title=self._title(question, retrieved),
            context=self._context(retrieved)[0],
            question=question.question,
        )

    def _context(
        self, retrieved: Sequence[ScoredChunk]
    ) -> tuple[str, list[tuple[ScoredChunk, Segment]]]:
        """Context text plus, in paragraph mode, the (chunk, segment) behind each number."""
        if self._cite_unit == "paragraph":
            return format_paragraph_context(retrieved)
        return format_context(retrieved), []

    def plan(self, question: Question, text: str | None = None) -> QueryPlan:
        """The texts to search with (the question itself unless a transformer is set)."""
        return self._transformer.plan(text or question.question, self._title(question))

    def retrieve(
        self, question: Question, plan: QueryPlan, text: str | None = None
    ) -> list[ScoredChunk]:
        paper_id = question.paper_id if self._scope == "paper" else None
        return self._retriever.retrieve(text or question.question, paper_id=paper_id, plan=plan)

    def run(
        self, question: Question, plan: QueryPlan | None = None, text: str | None = None
    ) -> PipelineResult:
        """Answer one question; pass a `plan` you already made to avoid a second transform call.

        `text` is what the user typed when it differs from the dataset question (M2 turns)."""
        text = text or question.question
        plan = self.plan(question, text) if plan is None else plan
        # The timer starts after planning: the transform's latency is already in plan.usage
        # (and so in answer.usage), and EvalRecord adds both, so it must be counted once.
        started = time.perf_counter()
        retrieved = self.retrieve(question, plan, text)
        retrieval_ms = (time.perf_counter() - started) * 1000
        answer = self.generate(
            question, text, self._title(question, retrieved), retrieved, plan.usage
        )
        return PipelineResult(
            answer=answer,
            retrieved=retrieved,
            retrieval_ms=retrieval_ms,
            query_error=plan.error,
        )

    def generate(
        self,
        question: Question,
        text: str,
        title: str,
        retrieved: Sequence[ScoredChunk],
        usage: Usage | None = None,
    ) -> Answer:
        """One structured LLM call over already retrieved chunks (also the M2 agent's last step).

        `usage` is what earlier steps of this turn cost; it is added to the answer's usage."""
        before = usage or Usage()
        context, numbered = self._context(retrieved)
        prompt = render_prompt(self._template, title, context, text)
        messages = [{"role": "user", "content": prompt}]
        try:
            draft, call_usage = structured_call(self._llm, messages, AnswerDraft)
        except (LLMError, StructuredOutputError) as err:
            return Answer(
                question_id=question.question_id,
                answer_type="unanswerable",
                text=UNANSWERABLE_TEXT,
                refused=True,
                error=f"{type(err).__name__}: {err}",
                usage=before,
            )

        refused = draft.answer_type == "unanswerable"
        picks = [] if refused else list(dict.fromkeys(draft.evidence))
        if self._cite_unit == "paragraph":
            hits = [numbered[i - 1] for i in picks if 1 <= i <= len(numbered)]
            citations = list(dict.fromkeys(sc.chunk.chunk_id for sc, _ in hits))
            evidence_ids = list(dict.fromkeys(seg.paragraph_id for _, seg in hits))
        else:
            cited = [retrieved[i - 1] for i in picks if 1 <= i <= len(retrieved)]
            citations = [sc.chunk.chunk_id for sc in cited]
            evidence_ids = list(dict.fromkeys(p for sc in cited for p in sc.chunk.paragraph_ids))
        return Answer(
            question_id=question.question_id,
            answer_type=draft.answer_type,
            text=UNANSWERABLE_TEXT if refused else draft.answer.strip(),
            citations=citations,
            evidence_ids=evidence_ids,
            refused=refused,
            usage=before + call_usage,
        )
