"""Play the synthetic conversations through corpus_rag or the M2 agent, one record per turn.

Purpose: The conversation half of the eval loop. Every turn becomes an ordinary EvalRecord
(same answer/evidence scoring as M1) plus the routing fields, so reports, the summary
table, the judge and paired bootstraps work unchanged across all three systems.
Flow: conversations + questions -> run_conversations(pipeline= | graph=) -> per turn:
corpus_rag: pipeline.run(text) | agent: graph.invoke on the conversation's thread; on a
clarifying question the simulated user replies with the gold paper's title ->
build_record() + turn fields -> list[EvalRecord] -> evals/runner.py.
Params: eval.system (picks pipeline vs graph in the runner), agent.tracing,
eval.mlflow_tracking_uri, eval.experiment_name.
Try: Open records.jsonl of m2_agent and filter decision == "stay" with answer_f1 == 0 and
turn_type == "switch_unsignaled": those are the switches a person would also have missed.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

import structlog
from langgraph.types import Command

from paperlens.config.models import AppConfig
from paperlens.domain import UNANSWERABLE_TEXT, Answer, PipelineResult, Question
from paperlens.evals.conversations import Turn
from paperlens.evals.records import EvalRecord, build_record

log = structlog.get_logger(__name__)


def enable_tracing(cfg: AppConfig) -> bool:
    """Send LangGraph runs to MLflow as traces; False (with a warning) if that fails."""
    try:
        import mlflow

        mlflow.set_tracking_uri(cfg.eval.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.eval.experiment_name)
        mlflow.langchain.autolog()
    except Exception as err:  # tracing is a nice-to-have: never fail an eval over it
        log.warning("mlflow tracing disabled", error=str(err))
        return False
    return True


@contextlib.contextmanager
def _conversation_span(name: str, tracing: bool) -> Iterator[None]:
    if not tracing:
        yield
        return
    import mlflow

    with mlflow.start_span(name=name, span_type="CHAIN"):
        yield


def _failed(question: Question, err: Exception) -> PipelineResult:
    answer = Answer(
        question_id=question.question_id,
        answer_type="unanswerable",
        text=UNANSWERABLE_TEXT,
        refused=True,
        error=f"{type(err).__name__}: {err}",
    )
    return PipelineResult(answer=answer, retrieved=[], retrieval_ms=0.0)


def _agent_turn(
    graph: Any, turn: Turn, question: Question, titles: Mapping[str, str]
) -> tuple[PipelineResult, dict[str, Any]]:
    config = {"configurable": {"thread_id": turn.conversation_id}}
    started = time.perf_counter()
    state = graph.invoke({"question": question, "text": turn.text}, config)
    if "__interrupt__" in state:
        # simulated user: names the paper the question really belongs to
        state = graph.invoke(Command(resume=titles.get(turn.paper_id, "")), config)
    wall_ms = (time.perf_counter() - started) * 1000
    answer: Answer = state["answer"]
    # retrieval_ms carries the non-LLM time, so record latency == wall-clock time of the turn
    result = PipelineResult(
        answer=answer,
        retrieved=state.get("retrieved", []),
        retrieval_ms=max(wall_ms - answer.usage.latency_ms, 0.0),
    )
    fields = {
        "standalone_question": state.get("standalone"),
        "decision": state.get("decision"),
        "resolved_paper_id": state.get("resolved_paper_id"),
        "clarified": bool(state.get("clarified")),
        "llm_calls": state.get("llm_calls"),
    }
    return result, fields


def run_conversations(
    conversations: Sequence[Sequence[Turn]],
    questions: Mapping[str, Question],
    titles: Mapping[str, str],
    k_values: Sequence[int],
    cite_unit: Literal["chunk", "paragraph"],
    *,
    pipeline: Any = None,  # BaselinePipeline(scope="corpus"): run(question, text=...)
    graph: Any = None,
    tracing: bool = False,
    log_every: int = 25,
) -> list[EvalRecord]:
    """Exactly one of `pipeline` (corpus_rag) or `graph` (agent) must be given."""
    if (pipeline is None) == (graph is None):
        raise ValueError("pass exactly one of pipeline= or graph=")
    records: list[EvalRecord] = []
    total = sum(len(c) for c in conversations)
    for conversation in conversations:
        with _conversation_span(conversation[0].conversation_id, tracing):
            for turn in conversation:
                question = questions[turn.question_id]
                fields: dict[str, Any] = {}
                try:
                    if graph is not None:
                        result, fields = _agent_turn(graph, turn, question, titles)
                    else:
                        result = pipeline.run(question, text=turn.text)
                        top = result.retrieved[0].chunk.paper_id if result.retrieved else None
                        fields = {"decision": "vote", "resolved_paper_id": top, "llm_calls": 1}
                except Exception as err:  # one broken turn must not end the eval
                    log.warning("turn failed", turn=turn.question_id, error=str(err))
                    result = _failed(question, err)
                record = build_record(question, result, k_values, cite_unit)
                records.append(
                    record.model_copy(
                        update={
                            "conversation_id": turn.conversation_id,
                            "turn_index": turn.turn_index,
                            "turn_type": turn.turn_type,
                            "signal_template": turn.template,
                            "shown_question": turn.text,
                            **fields,
                        }
                    )
                )
                if len(records) % log_every == 0 or len(records) == total:
                    mean_f1 = sum(r.answer_f1 for r in records) / len(records)
                    log.info(
                        "progress", done=len(records), total=total, answer_f1=round(mean_f1, 3)
                    )
    return records
