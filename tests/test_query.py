import json
import time
from pathlib import Path

import pytest

from paperlens.config.models import QueryConfig
from paperlens.domain import Answer, PipelineResult, Question, Reference, ScoredChunk, Usage
from paperlens.evals.aggregate import aggregate
from paperlens.evals.records import build_record
from paperlens.llm.client import LLMError
from paperlens.llm.fake import FakeLLM
from paperlens.pipelines.baseline import BaselinePipeline
from paperlens.retrieval.query import (
    HydeTransformer,
    IdentityTransformer,
    MultiQueryTransformer,
    QueryPlan,
    build_transformer,
)

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"
MQ_TEMPLATE = (PROMPTS / "multi_query.md").read_text(encoding="utf-8")
HYDE_TEMPLATE = (PROMPTS / "hyde.md").read_text(encoding="utf-8")


def test_identity_plan_uses_question_for_both_channels() -> None:
    plan = QueryPlan.identity("What is X?")
    assert plan.original == "What is X?"
    assert plan.dense == plan.sparse == ["What is X?"]
    assert plan.usage == Usage() and plan.error is None


def test_identity_transformer_returns_identity_plan() -> None:
    assert IdentityTransformer().plan("q", "title") == QueryPlan.identity("q")


def test_usage_addition_sums_all_fields() -> None:
    total = Usage(prompt_tokens=1, completion_tokens=2, latency_ms=3.0) + Usage(
        prompt_tokens=10, completion_tokens=20, latency_ms=30.0
    )
    assert total == Usage(prompt_tokens=11, completion_tokens=22, latency_ms=33.0)


def _multi(reply: str | Exception, n: int = 3, original: bool = True) -> tuple[FakeLLM, QueryPlan]:
    llm = FakeLLM([reply])
    plan = MultiQueryTransformer(llm, MQ_TEMPLATE, n, original).plan("q", "Title")
    return llm, plan


def test_multi_query_plan_dedupes_and_caps() -> None:
    reply = json.dumps({"queries": ["a", "", "a", "  ", "b", "c", "d"]})
    llm, plan = _multi(reply)
    assert plan.dense == plan.sparse == ["q", "a", "b", "c"]
    assert plan.error is None and plan.original == "q"
    assert plan.usage.prompt_tokens == FakeLLM.PROMPT_TOKENS
    prompt = llm.calls[0][0][0]["content"]
    assert "Title" in prompt and "into 3 different" in prompt


def test_multi_query_without_original() -> None:
    _, plan = _multi('{"queries": ["a", "b"]}', original=False)
    assert "q" not in plan.dense and plan.dense == plan.sparse == ["a", "b"]


def test_multi_query_empty_result_falls_back_to_original() -> None:
    _, plan = _multi('{"queries": ["", " "]}', original=False)
    assert plan.dense == plan.sparse == ["q"]


def test_hyde_searches_dense_with_hypothesis_and_bm25_with_question() -> None:
    llm = FakeLLM(["We evaluate on SQuAD."])
    plan = HydeTransformer(llm, HYDE_TEMPLATE, include_original=True).plan("q", "Title")
    assert plan.dense == ["We evaluate on SQuAD.", "q"]
    assert plan.sparse == ["q"]
    assert plan.usage.completion_tokens == FakeLLM.COMPLETION_TOKENS
    assert llm.calls[0][1] is None  # free text, no JSON schema


def test_hyde_without_original_uses_only_hypothesis_for_dense() -> None:
    plan = HydeTransformer(FakeLLM(["para"]), HYDE_TEMPLATE, include_original=False).plan("q", "T")
    assert plan.dense == ["para"] and plan.sparse == ["q"]


def test_hyde_strips_think_blocks() -> None:
    llm = FakeLLM(["<think>hmm</think>\nThe paragraph."])
    plan = HydeTransformer(llm, HYDE_TEMPLATE, include_original=False).plan("q", "T")
    assert plan.dense == ["The paragraph."]


def test_hyde_empty_reply_falls_back() -> None:
    plan = HydeTransformer(FakeLLM(["<think>x</think>"]), HYDE_TEMPLATE, True).plan("q", "T")
    assert plan.dense == plan.sparse == ["q"] and plan.error is not None


def test_transform_failure_falls_back_to_question() -> None:
    _, plan = _multi(LLMError("timeout"))
    assert plan.dense == plan.sparse == ["q"]
    assert plan.error == "LLMError: timeout"
    hyde = HydeTransformer(FakeLLM([LLMError("timeout")]), HYDE_TEMPLATE, True).plan("q", "T")
    assert hyde.dense == hyde.sparse == ["q"] and hyde.error == "LLMError: timeout"


def test_structured_failure_keeps_spent_usage() -> None:
    llm = FakeLLM(["not json", "still not json"])
    plan = MultiQueryTransformer(llm, MQ_TEMPLATE, 3, True).plan("q", "T")
    assert plan.error is not None and plan.error.startswith("StructuredOutputError")
    assert plan.usage.prompt_tokens == 2 * FakeLLM.PROMPT_TOKENS
    assert plan.dense == ["q"]


def test_missing_placeholder_is_rejected() -> None:
    with pytest.raises(ValueError, match="n"):
        MultiQueryTransformer(FakeLLM([]), "{{question}} {{title}}", 3, True)
    with pytest.raises(ValueError, match="title"):
        HydeTransformer(FakeLLM([]), "{{question}}", True)


def test_build_transformer_dispatches_on_config() -> None:
    llm = FakeLLM([])
    assert isinstance(build_transformer(QueryConfig(), llm), IdentityTransformer)
    assert isinstance(build_transformer(QueryConfig(transform="hyde"), llm), HydeTransformer)
    multi = build_transformer(QueryConfig(transform="multi_query", n_queries=2), llm)
    assert isinstance(multi, MultiQueryTransformer)


QUESTION = Question(
    question_id="q1",
    paper_id="P",
    question="Which dataset?",
    references=[Reference(answer_type="extractive", answer_text="SQuAD", evidence_ids=["P::s0p1"])],
)


def _result(query_error: str | None) -> PipelineResult:
    answer = Answer(question_id="q1", answer_type="extractive", text="SQuAD")
    return PipelineResult(answer=answer, retrieved=[], retrieval_ms=1.0, query_error=query_error)


def test_record_and_aggregate_count_query_errors() -> None:
    records = [
        build_record(QUESTION, _result("LLMError: boom"), [1]),
        build_record(QUESTION, _result(None), [1]),
    ]
    assert records[0].query_error == "LLMError: boom" and records[1].query_error is None
    assert aggregate(records, [1])["query_error_rate"] == 0.5


DELAY_MS = 50.0


class _SlowTransformer:
    def plan(self, question: str, title: str) -> QueryPlan:
        time.sleep(DELAY_MS / 1000)
        plan = QueryPlan.identity(question)
        return plan.model_copy(update={"usage": Usage(latency_ms=DELAY_MS)})


class _EmptyRetriever:
    def retrieve(
        self, query: str, paper_id: str | None = None, plan: QueryPlan | None = None
    ) -> list[ScoredChunk]:
        return []


def test_transform_latency_is_counted_once() -> None:
    template = (PROMPTS / "baseline_answer.md").read_text(encoding="utf-8")
    llm = FakeLLM(['{"answer_type": "unanswerable", "answer": "", "evidence": []}'])
    pipe = BaselinePipeline(_EmptyRetriever(), llm, template, {"P": "T"}, _SlowTransformer())
    result = pipe.run(QUESTION)
    assert result.retrieval_ms < DELAY_MS / 2  # planning is outside the retrieval timer
    record = build_record(QUESTION, result, [1])
    assert record.latency_ms == pytest.approx(result.retrieval_ms + DELAY_MS + FakeLLM.LATENCY_MS)
    assert record.latency_ms < DELAY_MS * 1.5
