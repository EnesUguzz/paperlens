"""Query plans: what text each retriever searches with, decided before retrieval.

Purpose: Query transforms (multi-query, HyDE) change the search text, not the retrieval
code. A QueryPlan carries the texts per channel plus the LLM cost and any failure, so
the Retriever stays a pure function of the plan.
Flow: question + paper title -> QueryTransformer.plan() -> QueryPlan(dense texts, sparse
texts, usage, error) -> Retriever.retrieve(plan=...) -> pipelines/baseline.py. An LLM
transformer that fails (timeout, unparseable JSON, empty reply) returns the identity plan
with `error` set, so one bad call costs a question its rewrite, never the whole answer.
Params: query.transform (none / multi_query / hyde), query.n_queries,
query.include_original, query.multi_query_prompt, query.hyde_prompt.
Try: Run `paperlens trace --config experiments/configs/m1b/hyde.yaml --index 16` and read
the QUERY PLAN section: is the hypothetical paragraph plausible, and does it use the
paper's own vocabulary? Then change prompts/hyde.md and compare.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from paperlens.config.models import QueryConfig
from paperlens.domain import Usage
from paperlens.llm.client import LLMClient, LLMError
from paperlens.llm.structured import StructuredOutputError, strip_reasoning, structured_call

MULTI_QUERY_PLACEHOLDERS = ("{{question}}", "{{title}}", "{{n}}")
HYDE_PLACEHOLDERS = ("{{question}}", "{{title}}")


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


class Paraphrases(BaseModel):
    """What the multi-query prompt must return."""

    queries: list[str]


def _check_placeholders(template: str, required: tuple[str, ...]) -> None:
    missing = [p for p in required if p not in template]
    if missing:
        raise ValueError(f"prompt template is missing placeholders: {missing}")


def _fallback(question: str, err: Exception, usage: Usage) -> QueryPlan:
    """The plain question, the failure recorded, and the cost already spent kept."""
    plan = QueryPlan.identity(question)
    return plan.model_copy(update={"usage": usage, "error": f"{type(err).__name__}: {err}"})


class MultiQueryTransformer:
    """Asks the LLM for paraphrases; dense and bm25 both search every text (fused by RRF)."""

    def __init__(
        self, llm: LLMClient, template: str, n_queries: int, include_original: bool
    ) -> None:
        _check_placeholders(template, MULTI_QUERY_PLACEHOLDERS)
        self._llm = llm
        self._template = template
        self._n = n_queries
        self._include_original = include_original

    def plan(self, question: str, title: str) -> QueryPlan:
        prompt = (
            self._template.replace("{{question}}", question)
            .replace("{{title}}", title)
            .replace("{{n}}", str(self._n))
        )
        try:
            parsed, usage = structured_call(
                self._llm, [{"role": "user", "content": prompt}], Paraphrases
            )
        except StructuredOutputError as err:
            return _fallback(question, err, err.usage)
        except LLMError as err:
            return _fallback(question, err, Usage())
        stripped = (q.strip() for q in parsed.queries)
        paraphrases = list(dict.fromkeys(q for q in stripped if q))[: self._n]
        texts = list(
            dict.fromkeys([question, *paraphrases] if self._include_original else paraphrases)
        )
        if not texts:  # the model returned only blanks: search with the question
            texts = [question]
        return QueryPlan(original=question, dense=texts, sparse=list(texts), usage=usage)


class HydeTransformer:
    """Hypothetical Document Embeddings: dense searches with an invented answer paragraph.

    BM25 keeps the plain question: the invented facts would add words the paper never
    uses and pull the keyword ranking toward noise."""

    def __init__(self, llm: LLMClient, template: str, include_original: bool) -> None:
        _check_placeholders(template, HYDE_PLACEHOLDERS)
        self._llm = llm
        self._template = template
        self._include_original = include_original

    def plan(self, question: str, title: str) -> QueryPlan:
        prompt = self._template.replace("{{question}}", question).replace("{{title}}", title)
        try:
            reply = self._llm.complete([{"role": "user", "content": prompt}])
        except LLMError as err:
            return _fallback(question, err, Usage())
        paragraph = strip_reasoning(reply.content)
        if not paragraph:
            return _fallback(question, ValueError("empty hypothetical paragraph"), reply.usage)
        dense = [paragraph, question] if self._include_original else [paragraph]
        return QueryPlan(original=question, dense=dense, sparse=[question], usage=reply.usage)


def build_transformer(cfg: QueryConfig, llm: LLMClient) -> QueryTransformer:
    """The transformer named by query.transform (IdentityTransformer when 'none')."""
    if cfg.transform == "multi_query":
        template = cfg.multi_query_prompt.read_text(encoding="utf-8")
        return MultiQueryTransformer(llm, template, cfg.n_queries, cfg.include_original)
    if cfg.transform == "hyde":
        template = cfg.hyde_prompt.read_text(encoding="utf-8")
        return HydeTransformer(llm, template, cfg.include_original)
    return IdentityTransformer()
