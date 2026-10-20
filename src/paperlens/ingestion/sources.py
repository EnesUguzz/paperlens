"""One entry point that loads papers and questions for the configured text source.

Purpose: Every caller (runner, rescore, trace, CLI) gets its data from load_dataset(), so
switching `data.text_source` changes nothing else. For 'pdf_docling' the QASPER questions are
kept and their gold evidence is re-pointed at Docling paragraphs.
Flow: DataConfig -> download_split/load_split -> (qasper_text) as is | (pdf_docling)
load_blocks -> blocks_to_paper -> align_questions -> LoadedData; sample_dataset() then draws
the eval sample from ALL questions and drops those whose paper has no parsed PDF.
Params: data.text_source, data.split, data.sample_size, data.seed, data.align_min_score,
data.align_min_words.
Try: Compare data.evidence_aligned_rate for align_min_score 80 vs 95; if it barely moves the
alignment is robust, if it jumps the threshold is cutting real matches.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel

from paperlens.config.models import DataConfig
from paperlens.domain import Paper, Question
from paperlens.ingestion.align import AlignmentStats, align_questions
from paperlens.ingestion.docling_parser import blocks_to_paper, load_blocks
from paperlens.ingestion.pdf import parsed_path
from paperlens.ingestion.qasper import download_split, load_split, sample_questions

log = structlog.get_logger()


class LoadedData(BaseModel):
    papers: list[Paper]
    questions: list[Question]
    missing_paper_ids: set[str]  # papers without a parsed PDF (their questions are dropped)
    metrics: dict[str, float]


def _rate(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def load_dataset(cfg: DataConfig) -> LoadedData:
    qasper_papers, questions = load_split(download_split(cfg, cfg.split))
    if cfg.text_source == "qasper_text":
        refs = [r for q in questions for r in q.references]
        unmatched = sum(r.unmatched_evidence for r in refs)
        total = unmatched + sum(len(r.evidence_ids) for r in refs)
        return LoadedData(
            papers=qasper_papers,
            questions=questions,
            missing_paper_ids=set(),
            metrics={"data.unmatched_evidence_rate": _rate(unmatched, total)},
        )

    by_paper: dict[str, list[Question]] = {}
    for question in questions:
        by_paper.setdefault(question.paper_id, []).append(question)
    papers: list[Paper] = []
    aligned_questions: dict[str, Question] = {}
    missing: set[str] = set()
    stats = AlignmentStats()
    for qasper_paper in qasper_papers:
        pid = qasper_paper.paper_id
        if not parsed_path(cfg, pid).exists():
            missing.add(pid)
            continue
        pdf_paper = blocks_to_paper(pid, qasper_paper.title, load_blocks(cfg, pid))
        aligned, paper_stats = align_questions(
            qasper_paper,
            pdf_paper,
            by_paper.get(pid, []),
            cfg.align_min_score,
            cfg.align_min_words,
        )
        papers.append(pdf_paper)
        aligned_questions.update({q.question_id: q for q in aligned})
        stats.evidence += paper_stats.evidence
        stats.aligned += paper_stats.aligned
        stats.split += paper_stats.split
    if not papers:
        raise FileNotFoundError("no parsed PDFs found: run `paperlens prepare-pdfs` first")
    metrics = {
        "data.papers_missing": float(len(missing)),
        "data.evidence_aligned_rate": _rate(stats.aligned, stats.evidence),
        "data.evidence_split_rate": _rate(stats.split, stats.evidence),
    }
    log.info("pdf dataset loaded", papers=len(papers), missing=len(missing), **metrics)
    # Questions of missing papers keep their QASPER evidence ids; sample_dataset drops them.
    return LoadedData(
        papers=papers,
        questions=[aligned_questions.get(q.question_id, q) for q in questions],
        missing_paper_ids=missing,
        metrics=metrics,
    )


def sample_dataset(
    loaded: LoadedData, sample_size: int | None, seed: int
) -> tuple[list[Question], int]:
    """Sample from all questions first (same ids as qasper_text), then drop missing papers."""
    sample = sample_questions(loaded.questions, sample_size, seed)
    kept = [q for q in sample if q.paper_id not in loaded.missing_paper_ids]
    return kept, len(sample) - len(kept)


def resolve_dataset(
    cfg: DataConfig, dataset: tuple[list[Paper], list[Question]] | None = None
) -> LoadedData:
    """load_dataset(cfg), unless a ready (papers, questions) pair is injected (tests)."""
    if dataset is None:
        return load_dataset(cfg)
    papers, questions = dataset
    return LoadedData(papers=papers, questions=questions, missing_paper_ids=set(), metrics={})
