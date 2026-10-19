"""QASPER answer metrics: Answer-F1, Evidence-F1 and refusal bookkeeping.

Purpose: Re-implements the official QASPER evaluator's logic (SQuAD-style token F1, max
over annotators; paragraph-set F1 for evidence) so scores are comparable to the paper.
Flow: Answer (pipelines) + Question.references -> score_answer() -> evals/records.py.
Params: none (metrics must not be tunable, or results stop being comparable).
Try: Compare answer_f1 per answer type in a report; extractive questions punish verbose
models, yes/no questions punish models that explain instead of answering.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from paperlens.domain import UNANSWERABLE_TEXT, Answer, AnswerType, Reference

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = set(string.punctuation)


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    no_punct = "".join(ch for ch in lowered if ch not in _PUNCTUATION)
    no_articles = _ARTICLES.sub(" ", no_punct)
    return " ".join(no_articles.split())


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def paragraph_f1(predicted: Sequence[str], gold: Sequence[str]) -> float:
    if not predicted and not gold:
        return 1.0
    same = len(set(predicted) & set(gold))
    if same == 0:
        return 0.0
    precision = same / len(set(predicted))
    recall = same / len(set(gold))
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class BestMatch:
    answer_f1: float
    evidence_f1: float
    ref_type: AnswerType


def score_answer(
    prediction: str, predicted_evidence: Sequence[str], references: Sequence[Reference]
) -> BestMatch:
    """Max over annotators, independently for answer and evidence (official behaviour)."""
    answer_scores = [token_f1(prediction, r.answer_text) for r in references]
    best = max(range(len(references)), key=lambda i: answer_scores[i])
    evidence = max(paragraph_f1(predicted_evidence, r.evidence_ids) for r in references)
    return BestMatch(
        answer_f1=answer_scores[best], evidence_f1=evidence, ref_type=references[best].answer_type
    )


def prediction_text(answer: Answer) -> str:
    if answer.refused or answer.answer_type == "unanswerable":
        return UNANSWERABLE_TEXT
    if answer.answer_type == "yes_no":
        lowered = answer.text.strip().lower()
        if lowered.startswith("yes"):
            return "Yes"
        if lowered.startswith("no"):
            return "No"
    return answer.text


def is_gold_unanswerable(references: Sequence[Reference]) -> bool:
    """At least half of the annotators said unanswerable (ties count as unanswerable)."""
    unanswerable = sum(r.answer_type == "unanswerable" for r in references)
    return unanswerable * 2 >= len(references)
