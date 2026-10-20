from paperlens.domain import Usage
from paperlens.retrieval.query import IdentityTransformer, QueryPlan


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
