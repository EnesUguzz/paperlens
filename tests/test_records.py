import pytest

from paperlens.domain import (
    Answer,
    AnswerType,
    Chunk,
    PipelineResult,
    Question,
    Reference,
    ScoredChunk,
    Usage,
)
from paperlens.evals.aggregate import aggregate
from paperlens.evals.answer_metrics import gold_type
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


WIDE_CHUNK = ScoredChunk(
    chunk=Chunk(
        chunk_id="P::c1",
        paper_id="P",
        section="S",
        text="wide chunk",
        paragraph_ids=["1234.5678::s0p1", "1234.5678::s0p0"],
    ),
    score=1.0,
)


def test_gold_type_is_majority_with_unanswerable_tie_break() -> None:
    def ref(t: AnswerType) -> Reference:
        return Reference(answer_type=t, answer_text="x", evidence_ids=[])

    assert gold_type([ref("extractive"), ref("extractive"), ref("yes_no")]) == "extractive"
    assert gold_type([ref("unanswerable"), ref("yes_no")]) == "unanswerable"


def test_evidence_ceiling_reflects_chunk_granularity(sample_questions: list[Question]) -> None:
    q1 = sample_questions[0]  # gold evidence: s0p1
    exact = build_record(q1, result("q1", "extractive", "SQuAD", ["1234.5678::s0p1"]), [1])
    assert exact.evidence_f1_ceiling == 1.0
    wide = result("q1", "extractive", "SQuAD", ["1234.5678::s0p1", "1234.5678::s0p0"])
    wide = wide.model_copy(update={"retrieved": [WIDE_CHUNK]})
    rec = build_record(q1, wide, [1])
    assert rec.evidence_f1_ceiling == pytest.approx(
        2 / 3
    )  # best possible when citing a 2-paragraph chunk
    assert rec.gold_type == "extractive"


def test_evidence_ceiling_is_the_best_subset_when_gold_chunks_overlap() -> None:
    # gold paragraph "a" sits in two overlapping 5-paragraph chunks (fixed chunking overlap):
    # citing one gives F1 = 2*(1/5)*1 / (1/5 + 1) = 1/3; citing both (9 paragraphs) gives 0.2
    from paperlens.evals.records import evidence_ceiling

    ref = Reference(answer_type="extractive", answer_text="x", evidence_ids=["a"])
    ranked = [["a", "b", "c", "d", "e"], ["e", "f", "g", "h", "a"]]
    assert evidence_ceiling(ranked, [ref]) == pytest.approx(1 / 3)


def test_evidence_ceiling_is_never_below_a_real_citation(sample_questions: list[Question]) -> None:
    # a real, scored citation can only be at or below the ceiling
    from paperlens.evals.records import evidence_ceiling

    ref = Reference(answer_type="extractive", answer_text="x", evidence_ids=["a", "f"])
    ranked = [["a", "b"], ["a", "c", "d"], ["f"], ["z"]]
    assert evidence_ceiling(ranked, [ref]) == pytest.approx(0.8)  # cite [a,b] + [f]


def test_aggregate_reports_gold_type_buckets_and_ceiling(sample_questions: list[Question]) -> None:
    q1, q2, _ = sample_questions
    records = [
        build_record(q1, result("q1", "extractive", "SQuAD", ["1234.5678::s0p1"]), [1]),
        build_record(q2, result("q2", "yes_no", "No", []), [1]),
    ]
    m = aggregate(records, [1])
    # q2's best-matching reference is yes_no, but its gold type (tie) is unanswerable
    assert m["n.yes_no"] == 1 and m["n.gold.unanswerable"] == 1
    assert m["answer_f1.gold.unanswerable"] == 1.0
    assert m["evidence_f1_ceiling"] == pytest.approx(1.0)
