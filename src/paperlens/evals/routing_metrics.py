"""Did the conversation systems answer from the right paper, and at what cost?

Purpose: The M2 numbers that answer_f1 alone hides: paper accuracy per turn type, how
often a named switch was caught (per signal template, so an easy template cannot carry
the score), how often the agent asked, and how often it answered from the wrong paper
without asking. Also answer_f1 on turns without a clarification: the simulated user always
names the right paper, so asking is a free win the second number removes.
Flow: list[EvalRecord] with turn fields -> routing_metrics() -> metrics.json / report /
MLflow (merged into aggregate()'s dict by the runner).
Params: none.
Try: Compare route.paper_accuracy.switch_unsignaled between corpus_rag and the agent: the
agent stays on the active paper, so it loses there by design and wins on 'same' turns.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from paperlens.evals.records import EvalRecord

TURN_TYPES = ("first", "same", "switch_signaled", "switch_unsignaled")


def _rate(records: Sequence[EvalRecord], hit: Callable[[EvalRecord], bool]) -> float:
    return sum(hit(r) for r in records) / len(records) if records else 0.0


def _right_paper(r: EvalRecord) -> bool:
    return r.resolved_paper_id == r.paper_id


def routing_metrics(records: Sequence[EvalRecord]) -> dict[str, float]:
    turns = [r for r in records if r.turn_type is not None]
    if not turns:
        return {}
    m: dict[str, float] = {
        "route.paper_accuracy": _rate(turns, _right_paper),
        "route.clarify_rate": _rate(turns, lambda r: r.clarified),
        "route.wrong_paper_rate": _rate(
            turns, lambda r: not r.clarified and not r.refused and not _right_paper(r)
        ),
    }
    for kind in TURN_TYPES:
        subset = [r for r in turns if r.turn_type == kind]
        m[f"n.turn.{kind}"] = float(len(subset))
        if subset:
            m[f"route.paper_accuracy.{kind}"] = _rate(subset, _right_paper)
            m[f"answer_f1.turn.{kind}"] = sum(r.answer_f1 for r in subset) / len(subset)
    signaled = [r for r in turns if r.turn_type == "switch_signaled"]
    if signaled:
        m["route.signaled_catch"] = _rate(signaled, _right_paper)
        for template in sorted(
            {r.signal_template for r in signaled if r.signal_template is not None}
        ):
            subset = [r for r in signaled if r.signal_template == template]
            m[f"route.signaled_catch.t{template}"] = _rate(subset, _right_paper)
    unsignaled = [r for r in turns if r.turn_type == "switch_unsignaled"]
    if unsignaled:
        m["route.unsignaled_catch"] = _rate(unsignaled, _right_paper)
    plain = [r for r in turns if not r.clarified]
    m["n.no_clarify"] = float(len(plain))
    if plain:
        m["answer_f1.no_clarify"] = sum(r.answer_f1 for r in plain) / len(plain)
        m["evidence_f1.no_clarify"] = sum(r.evidence_f1 for r in plain) / len(plain)
    condensed = [r for r in turns if r.standalone_question is not None]
    if condensed:
        m["condense.changed_rate"] = _rate(
            condensed, lambda r: r.standalone_question != r.shown_question
        )
    calls = [r.llm_calls for r in turns if r.llm_calls is not None]
    if calls:
        m["llm_calls_mean"] = sum(calls) / len(calls)
    return m
