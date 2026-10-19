import pytest

from paperlens.domain import Answer, AnswerType, Chunk, PipelineResult, Question, ScoredChunk, Usage
from paperlens.evals.aggregate import aggregate
from paperlens.evals.records import build_record

CHUNK = ScoredChunk(
    chunk=Chunk(
        chunk_id="P::c0",
        paper_id="P",
        section="S",
        text="We use the SQuAD dataset",
        paragraph_ids=["1234.5678::s0p1"],
    ),
    score=1.0,
)


def result(
    qid: str,
    answer_type: AnswerType,
    text: str,
    evidence: list[str],
    refused: bool = False,
    prompt_tokens: int = 100,
) -> PipelineResult:
    usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=10, latency_ms=90.0)
    answer = Answer(
        question_id=qid,
        answer_type=answer_type,
        text=text,
        evidence_ids=evidence,
        refused=refused,
        usage=usage,
    )
    return PipelineResult(answer=answer, retrieved=[CHUNK], retrieval_ms=10.0)


def test_build_record_scores_answer_and_retrieval(sample_questions: list[Question]) -> None:
    q1 = sample_questions[0]
    rec = build_record(q1, result("q1", "extractive", "SQuAD", ["1234.5678::s0p1"]), [1])
    assert rec.answer_f1 == 1.0 and rec.evidence_f1 == 1.0
    assert rec.retrieval == {"recall_at_1": 1.0, "ndcg_at_1": 1.0, "mrr": 1.0}
    assert rec.latency_ms == 100.0 and rec.context_words == 5
    assert rec.top_context_preview.startswith("We use")


def test_aggregate(sample_questions: list[Question]) -> None:
    q1, q2, _ = sample_questions
    records = [
        build_record(q1, result("q1", "extractive", "SQuAD", ["1234.5678::s0p1"]), [1]),
        build_record(
            q2,
            result("q2", "unanswerable", "Unanswerable", [], refused=True, prompt_tokens=300),
            [1],
        ),
    ]
    m = aggregate(records, [1])
    assert m["n_questions"] == 2
    assert m["answer_f1"] == 1.0
    assert m["answer_f1.extractive"] == 1.0 and m["n.extractive"] == 1
    assert m["refusal_recall"] == 1.0 and m["false_refusal_rate"] == 0.0
    assert m["retrieval.n"] == 2  # q2 has one annotator with evidence (the caption)
    assert m["prompt_tokens_max"] == 300
    assert m["latency_p50_ms"] == pytest.approx(100.0)


def test_aggregate_empty() -> None:
    assert aggregate([], [1]) == {"n_questions": 0.0}
