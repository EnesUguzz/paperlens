import pytest

from paperlens.domain import Answer, Reference
from paperlens.evals.answer_metrics import (
    is_gold_unanswerable,
    normalize_answer,
    paragraph_f1,
    prediction_text,
    score_answer,
    token_f1,
    token_recall,
)


def test_normalize_matches_squad_rules() -> None:
    assert normalize_answer("The SQuAD, dataset!") == "squad dataset"


def test_token_f1() -> None:
    assert token_f1("squad dataset", "SQuAD") == pytest.approx(2 / 3)
    assert token_f1("x", "y") == 0.0


def test_paragraph_f1_official_edge_cases() -> None:
    assert paragraph_f1([], []) == 1.0
    assert paragraph_f1(["a"], []) == 0.0
    assert paragraph_f1(["a", "b"], ["a"]) == pytest.approx(2 / 3)


def test_score_answer_takes_best_reference() -> None:
    refs = [
        Reference(answer_type="unanswerable", answer_text="Unanswerable", evidence_ids=[]),
        Reference(answer_type="yes_no", answer_text="No", evidence_ids=["p::f0"]),
    ]
    best = score_answer("No", [], refs)
    assert best.answer_f1 == 1.0
    assert best.ref_type == "yes_no"
    assert best.evidence_f1 == 1.0  # empty vs the unanswerable reference's empty evidence


def test_prediction_text_normalises_types() -> None:
    refused = Answer(question_id="q", answer_type="extractive", text="x", refused=True)
    assert prediction_text(refused) == "Unanswerable"
    yes = Answer(question_id="q", answer_type="yes_no", text="yes, they do")
    assert prediction_text(yes) == "Yes"


def test_gold_unanswerable_on_tie_counts_as_unanswerable() -> None:
    refs = [
        Reference(answer_type="unanswerable", answer_text="Unanswerable", evidence_ids=[]),
        Reference(answer_type="yes_no", answer_text="No", evidence_ids=[]),
    ]
    assert is_gold_unanswerable(refs)
    assert not is_gold_unanswerable(refs[1:])


def test_token_recall() -> None:
    assert token_recall("We use the SQuAD dataset", "SQuAD") == 1.0
    assert token_recall("SQuAD", "SQuAD and TriviaQA") == pytest.approx(1 / 3)
    assert token_recall("anything", "") == 0.0
