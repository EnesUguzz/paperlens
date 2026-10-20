"""Align QASPER gold evidence to the paragraphs of a PDF-parsed paper.

Purpose: QASPER's gold evidence is quoted from its own parsed text. A Docling paper has
different paragraph boundaries and ids, so each gold paragraph is fuzzy-matched to the PDF
paragraphs that contain it. Evidence that cannot be matched stays visible as a sentinel id
instead of silently vanishing (which would make a question look like it has no evidence).
Flow: (qasper Paper, pdf Paper, questions) -> align_paragraph() per evidence paragraph ->
align_questions() -> questions whose evidence_ids are PDF paragraph ids + AlignmentStats.
Params: data.align_min_score, data.align_min_words.
Try: Print a few (qasper text, matched PDF text, score) pairs for one paper before trusting
a new threshold; a split paragraph should map to two adjacent PDF ids.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel
from rapidfuzz import fuzz, process

from paperlens.domain import Paper, Paragraph, Question, Reference

UNALIGNED_MARK = "unaligned"  # sentinel id: f"{paper_id}::unaligned{n}"


class AlignmentStats(BaseModel):
    evidence: int = 0  # gold evidence paragraphs seen
    aligned: int = 0  # ... that matched at least one PDF paragraph
    split: int = 0  # ... that matched more than one PDF paragraph


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def scored_matches(
    text: str, candidates: Sequence[Paragraph], min_score: float, min_words: int
) -> list[tuple[str, float]]:
    """(paragraph id, partial_ratio score) of every eligible candidate above min_score,
    in the candidates' document order."""
    eligible = [c for c in candidates if c.is_caption or len(c.text.split()) >= min_words]
    if not eligible:
        return []
    hits = process.extract(
        _normalize(text),
        [_normalize(c.text) for c in eligible],
        scorer=fuzz.partial_ratio,
        score_cutoff=min_score,
        limit=None,
        processor=None,  # already normalised above
    )
    return [
        (eligible[idx].paragraph_id, score) for _, score, idx in sorted(hits, key=lambda h: h[2])
    ]


def align_paragraph(
    text: str, candidates: Sequence[Paragraph], min_score: float, min_words: int
) -> list[str]:
    """Ids of the PDF paragraphs that hold `text` (document order)."""
    return [pid for pid, _ in scored_matches(text, candidates, min_score, min_words)]


def align_questions(
    qasper_paper: Paper,
    pdf_paper: Paper,
    questions: Sequence[Question],
    min_score: float,
    min_words: int,
) -> tuple[list[Question], AlignmentStats]:
    """Rewrite each reference's evidence_ids from QASPER ids to PDF paragraph ids."""
    text_of = {p.paragraph_id: p.text for p in qasper_paper.paragraphs}
    cache: dict[str, list[str]] = {}
    stats = AlignmentStats()
    out: list[Question] = []
    for question in questions:
        references: list[Reference] = []
        for ref in question.references:
            evidence_ids: list[str] = []
            unmatched = ref.unmatched_evidence
            unaligned = 0
            for qid in ref.evidence_ids:
                if qid not in cache:
                    cache[qid] = align_paragraph(
                        text_of[qid], pdf_paper.paragraphs, min_score, min_words
                    )
                matched = cache[qid]
                stats.evidence += 1
                if matched:
                    stats.aligned += 1
                    stats.split += len(matched) > 1
                    evidence_ids.extend(m for m in matched if m not in evidence_ids)
                else:
                    evidence_ids.append(f"{pdf_paper.paper_id}::{UNALIGNED_MARK}{unaligned}")
                    unaligned += 1
                    unmatched += 1
            references.append(
                ref.model_copy(
                    update={"evidence_ids": evidence_ids, "unmatched_evidence": unmatched}
                )
            )
        out.append(question.model_copy(update={"references": references}))
    return out, stats
