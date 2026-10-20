"""Query plans: what text each retriever searches with, decided before retrieval.

Purpose: Query transforms (multi-query, HyDE) change the search text, not the retrieval
code. A QueryPlan carries the texts per channel plus the LLM cost and any failure, so
the Retriever stays a pure function of the plan.
Flow: question + paper title -> QueryTransformer.plan() -> QueryPlan(dense texts, sparse
texts, usage, error) -> Retriever.retrieve(plan=...) -> pipelines/baseline.py.
Params: none here (IdentityTransformer is the no-op default; real transformers add their
own config).
Try: Print QueryPlan.identity("q") and notice that dense and sparse hold the same text;
that is exactly the M1a behaviour.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from paperlens.domain import Usage


class QueryPlan(BaseModel):
    """Texts to search with. `original` is always the user's question (reranker input)."""

    original: str
    dense: list[str]
    sparse: list[str]
    usage: Usage = Usage()
    error: str | None = None

    @classmethod
    def identity(cls, question: str) -> QueryPlan:
        return cls(original=question, dense=[question], sparse=[question])


class QueryTransformer(Protocol):
    def plan(self, question: str, title: str) -> QueryPlan: ...


class IdentityTransformer:
    """Searches with the question itself; the M1a behaviour."""

    def plan(self, question: str, title: str) -> QueryPlan:
        return QueryPlan.identity(question)
