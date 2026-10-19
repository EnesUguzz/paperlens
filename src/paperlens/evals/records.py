"""One EvalRecord per question: scores plus what was retrieved and cited.

Purpose: Keeps every number traceable to a concrete question, so a metric change in the
report can be explained by opening the questions that moved.
Flow: Question + PipelineResult -> build_record() -> records.jsonl -> aggregate / report.
Params: eval.k_values.
Try: Open records.jsonl, filter answer_f1 == 0 and retrieval.recall_at_5 == 1: those are
generation failures (the evidence was there, the model still missed). Compare evidence_f1
with evidence_f1_ceiling: the ceiling is what perfect citing could reach with these chunks,
so the gap is the model's citation error and the ceiling itself is the chunking's limit.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from paperlens.domain import AnswerType, PipelineResult, Question, Reference
from paperlens.evals.answer_metrics import (
    gold_type,
    is_gold_unanswerable,
    paragraph_f1,
    prediction_text,
    score_answer,
)
from paperlens.evals.retrieval_metrics import best_over_references

PREVIEW_CHARS = 300


def evidence_ceiling(ranked: Sequence[Sequence[str]], references: Sequence[Reference]) -> float:
    """Best Evidence-F1 reachable by citing exactly the retrieved chunks that hold gold.

    A citation is a whole chunk, so its paragraphs all count as predicted evidence: with
    big chunks even a perfect citer pays a precision penalty. Evidence-F1 is therefore
    only comparable across runs with the same chunking; compare evidence_f1 / ceiling."""
    best = 0.0
    for ref in references:
        gold = set(ref.evidence_ids)
        cited = {pid for chunk in ranked if gold & set(chunk) for pid in chunk}
        best = max(best, paragraph_f1(sorted(cited), ref.evidence_ids))
    return best


class EvalRecord(BaseModel):
    question_id: str
    paper_id: str
    question: str
    gold_texts: list[str]
    gold_unanswerable: bool
    gold_type: AnswerType
    prediction: str
    predicted_type: AnswerType
    refused: bool
    error: str | None
    answer_f1: float
    evidence_f1: float
    evidence_f1_ceiling: float
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
        gold_type=gold_type(question.references),
        prediction=prediction,
        predicted_type=answer.answer_type,
        refused=answer.refused,
        error=answer.error,
        answer_f1=best.answer_f1,
        evidence_f1=best.evidence_f1,
        evidence_f1_ceiling=evidence_ceiling(ranked, question.references),
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
