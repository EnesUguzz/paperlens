"""One EvalRecord per question: scores plus what was retrieved and cited.

Purpose: Keeps every number traceable to a concrete question, so a metric change in the
report can be explained by opening the questions that moved.
Flow: Question + PipelineResult -> build_record() -> records.jsonl -> aggregate / report.
Params: eval.k_values.
Try: Open records.jsonl, filter answer_f1 == 0 and retrieval.recall_at_5 == 1: those are
generation failures (the evidence was there, the model still missed).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from paperlens.domain import AnswerType, PipelineResult, Question
from paperlens.evals.answer_metrics import is_gold_unanswerable, prediction_text, score_answer
from paperlens.evals.retrieval_metrics import best_over_references

PREVIEW_CHARS = 300


class EvalRecord(BaseModel):
    question_id: str
    paper_id: str
    question: str
    gold_texts: list[str]
    gold_unanswerable: bool
    prediction: str
    predicted_type: AnswerType
    refused: bool
    error: str | None
    answer_f1: float
    evidence_f1: float
    best_ref_type: AnswerType
    retrieval: dict[str, float] | None
    retrieved_chunk_ids: list[str]
    cited_chunk_ids: list[str]
    top_context_preview: str
    context_words: int
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


def build_record(question: Question, result: PipelineResult, k_values: Sequence[int]) -> EvalRecord:
    answer = result.answer
    prediction = prediction_text(answer)
    best = score_answer(prediction, answer.evidence_ids, question.references)
    ranked = [sc.chunk.paragraph_ids for sc in result.retrieved]
    preview = result.retrieved[0].chunk.text[:PREVIEW_CHARS] if result.retrieved else ""
    return EvalRecord(
        question_id=question.question_id,
        paper_id=question.paper_id,
        question=question.question,
        gold_texts=[r.answer_text for r in question.references],
        gold_unanswerable=is_gold_unanswerable(question.references),
        prediction=prediction,
        predicted_type=answer.answer_type,
        refused=answer.refused,
        error=answer.error,
        answer_f1=best.answer_f1,
        evidence_f1=best.evidence_f1,
        best_ref_type=best.ref_type,
        retrieval=best_over_references(ranked, question.references, k_values),
        retrieved_chunk_ids=[sc.chunk.chunk_id for sc in result.retrieved],
        cited_chunk_ids=answer.citations,
        top_context_preview=preview,
        context_words=sum(len(sc.chunk.text.split()) for sc in result.retrieved),
        prompt_tokens=answer.usage.prompt_tokens,
        completion_tokens=answer.usage.completion_tokens,
        latency_ms=result.retrieval_ms + answer.usage.latency_ms,
    )
