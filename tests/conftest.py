from typing import Any

import pytest

from paperlens.domain import Paper, Question
from paperlens.ingestion.qasper import parse_paper, parse_questions

EMPTY = ["", "", ""]

SAMPLE_ROW: dict[str, Any] = {
    "id": "1234.5678",
    "title": "A Tiny Paper About Retrieval",
    "abstract": "We study retrieval for question answering.",
    "full_text": {
        "section_name": ["Introduction", "Experiments"],
        "paragraphs": [
            ["Retrieval matters for QA systems.", "We use the SQuAD dataset for evaluation."],
            ["Our model reaches 81.2 F1 on the test set.", ""],
        ],
    },
    "qas": {
        "question": [
            "Which dataset do they use?",
            "Do they release code?",
            "What F1 do they reach?",
        ],
        "question_id": ["q1", "q2", "q3"],
        "nlp_background": EMPTY,
        "topic_background": EMPTY,
        "paper_read": EMPTY,
        "search_query": EMPTY,
        "question_writer": EMPTY,
        "answers": [
            {
                "answer": [
                    {
                        "unanswerable": False,
                        "extractive_spans": ["SQuAD"],
                        "yes_no": None,
                        "free_form_answer": "",
                        "highlighted_evidence": [],
                        "evidence": ["We use the SQuAD dataset for evaluation."],
                    },
                ],
                "annotation_id": ["a1"],
                "worker_id": ["w1"],
            },
            {
                "answer": [
                    {
                        "unanswerable": True,
                        "extractive_spans": [],
                        "yes_no": None,
                        "free_form_answer": "",
                        "evidence": [],
                        "highlighted_evidence": [],
                    },
                    {
                        "unanswerable": False,
                        "extractive_spans": [],
                        "yes_no": False,
                        "free_form_answer": "",
                        "highlighted_evidence": [],
                        "evidence": ["FLOAT SELECTED: Table 1: Main results."],
                    },
                ],
                "annotation_id": ["a2", "a3"],
                "worker_id": ["w2", "w3"],
            },
            {
                "answer": [
                    {
                        "unanswerable": False,
                        "extractive_spans": [],
                        "yes_no": None,
                        "free_form_answer": "81.2 F1",
                        "highlighted_evidence": [],
                        "evidence": [
                            "Our model reaches 81.2 F1 on the test set.",
                            "A sentence that is not in the paper.",
                        ],
                    },
                ],
                "annotation_id": ["a4"],
                "worker_id": ["w4"],
            },
        ],
    },
    "figures_and_tables": {"caption": ["Table 1: Main results."], "file": ["table1.png"]},
}


@pytest.fixture
def sample_row() -> dict[str, Any]:
    return SAMPLE_ROW


@pytest.fixture
def sample_paper() -> Paper:
    return parse_paper(SAMPLE_ROW)


@pytest.fixture
def sample_questions(sample_paper: Paper) -> list[Question]:
    return parse_questions(SAMPLE_ROW, sample_paper)
