"""Parity test: our QASPER Answer-F1 / Evidence-F1 code must equal the official evaluator.

Purpose: Pin evals/answer_metrics.py to allenai's evaluator (vendored in tests/official/) so a
change that drifts from the official scoring is caught on every commit.
Flow: random + hand-picked inputs -> our function vs official function -> exact equality.
Params: none (fixed seeds; the vocabulary below is the only knob).
Try: Add a nasty string to TRICKY whenever a scoring surprise shows up in a real run.
"""

from __future__ import annotations

import random

from conftest import SAMPLE_ROW

# tests/ is on sys.path (pytest prepend import mode), so the vendored module imports as below.
from official import qasper_evaluator as official

from paperlens.domain import Answer, Question
from paperlens.evals.answer_metrics import (
    paragraph_f1,
    prediction_text,
    score_answer,
    token_f1,
)

TRICKY = [
    ("The SQuAD dataset.", "SQuAD"),
    ("a an the", "the"),
    ("", "x"),
    ("x", ""),
    ("F1 of 81.2%", "81.2 F1"),
    ("yes yes yes", "yes"),
    ("Café–naïve", "cafe naive"),  # noqa: RUF001 (en dash is the point of this case)
]

# Words that stress normalisation: articles, punctuation-only tokens, digits, brackets.
VOCAB = [
    "the", "a", "an", "F1", "81.2", ",", "(BERT)", "model", "Model", "dataset", "SQuAD",
    "yes", "no", "of", "on", "test", "set", "we", "use", "BERT", "CNN-LSTM", "5%", "it's",
    "word", "retrieval", "QA", "systems", "-", "x", "Y",
]  # fmt: skip

PARAGRAPH_IDS = [f"p{i}" for i in range(8)]


def test_token_f1_matches_official_on_tricky_pairs() -> None:
    for pred, gold in TRICKY:
        assert token_f1(pred, gold) == official.token_f1_score(pred, gold), (pred, gold)


def test_token_f1_matches_official_on_random_pairs() -> None:
    rng = random.Random(0)
    for _ in range(1000):
        pred = " ".join(rng.choices(VOCAB, k=rng.randint(0, 12)))
        gold = " ".join(rng.choices(VOCAB, k=rng.randint(0, 12)))
        assert token_f1(pred, gold) == official.token_f1_score(pred, gold), (pred, gold)


def test_paragraph_f1_matches_official() -> None:
    rng = random.Random(1)
    for _ in range(300):
        pred = rng.sample(PARAGRAPH_IDS, rng.randint(0, 8))
        gold = rng.sample(PARAGRAPH_IDS, rng.randint(0, 8))
        assert paragraph_f1(pred, gold) == official.paragraph_f1_score(pred, gold), (pred, gold)


def _gold_json_from_row(row: dict) -> dict:
    """Build the evaluator's input the way ~/scratch/official/crosscheck.py does."""
    qas = row["qas"]
    out = []
    for i, qid in enumerate(qas["question_id"]):
        answers = [{"answer": a} for a in qas["answers"][i]["answer"]]
        out.append({"question_id": qid, "answers": answers})
    return {row["id"]: {"qas": out}}


def test_answer_f1_matches_official_end_to_end(sample_questions: list[Question]) -> None:
    gold = official.get_answers_and_evidence(_gold_json_from_row(SAMPLE_ROW), False)
    answers = {
        "q1": Answer(question_id="q1", answer_type="extractive", text="The SQuAD dataset."),
        "q2": Answer(question_id="q2", answer_type="unanswerable", text="", refused=True),
        "q3": Answer(question_id="q3", answer_type="yes_no", text="No, they do not"),
    }
    preds = {
        qid: {"answer": prediction_text(answer), "evidence": []} for qid, answer in answers.items()
    }
    assert preds["q2"]["answer"] == "Unanswerable"
    assert preds["q3"]["answer"] == "No"

    official_f1 = official.evaluate(gold, preds)["Answer F1"]
    per_question = [
        score_answer(preds[q.question_id]["answer"], [], q.references).answer_f1
        for q in sample_questions
    ]
    # Same aggregation as the official evaluator (sum / len); statistics.mean rounds differently.
    assert sum(per_question) / len(per_question) == official_f1
