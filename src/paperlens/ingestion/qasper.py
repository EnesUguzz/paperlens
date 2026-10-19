"""Download QASPER splits and turn raw rows into Paper and Question objects.

Purpose: The only module that knows QASPER's nested format. It also maps every gold
evidence string to a paragraph id, so retrieval is scored by id instead of fuzzy text.
Flow: HF parquet -> download_split() -> data/raw/qasper/<split>.parquet -> load_split()
-> (papers, questions) -> sample_questions() -> evals/runner.py.
Params: data.split, data.sample_size, data.seed, data.data_dir, data.parquet_url_template.
Try: Sum Reference.unmatched_evidence over a split; it shows how often annotators quoted
text that is not a paragraph (should be near 0 for qasper_text, higher for PDFs in M1b).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import httpx
import pyarrow.parquet as pq

from paperlens.config.models import DataConfig
from paperlens.domain import (
    UNANSWERABLE_TEXT,
    AnswerType,
    Paper,
    Paragraph,
    Question,
    Reference,
)

FLOAT_PREFIX = "FLOAT SELECTED:"  # QASPER marks table/figure evidence with this prefix
ABSTRACT_SECTION = "Abstract"
CAPTION_SECTION = "Figures and Tables"
UNTITLED_SECTION = "Untitled"
DOWNLOAD_TIMEOUT_S = 120.0


def raw_path(cfg: DataConfig, split: str) -> Path:
    return cfg.data_dir / "raw" / "qasper" / f"{split}.parquet"


def download_split(cfg: DataConfig, split: str, client: httpx.Client | None = None) -> Path:
    target = raw_path(cfg, split)
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    http = client or httpx.Client(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_S)
    try:
        with http.stream("GET", cfg.parquet_url_template.format(split=split)) as response:
            response.raise_for_status()
            with tmp.open("wb") as fh:
                for block in response.iter_bytes():
                    fh.write(block)
    finally:
        if client is None:
            http.close()
    tmp.replace(target)  # rename only after a complete download
    return target


def _norm(text: str) -> str:
    return " ".join(text.split())


def parse_paper(row: dict[str, Any]) -> Paper:
    pid = row["id"]
    abstract = row.get("abstract") or ""
    paragraphs: list[Paragraph] = []
    if abstract.strip():
        paragraphs.append(
            Paragraph(paragraph_id=f"{pid}::abs", section=ABSTRACT_SECTION, text=abstract)
        )
    full = row["full_text"]
    for s_idx, (section, paras) in enumerate(
        zip(full["section_name"], full["paragraphs"], strict=True)
    ):
        for p_idx, text in enumerate(paras):
            if text and text.strip():
                paragraphs.append(
                    Paragraph(
                        paragraph_id=f"{pid}::s{s_idx}p{p_idx}",
                        section=section or UNTITLED_SECTION,
                        text=text,
                    )
                )
    for f_idx, caption in enumerate(row["figures_and_tables"]["caption"]):
        if caption and caption.strip():
            paragraphs.append(
                Paragraph(
                    paragraph_id=f"{pid}::f{f_idx}",
                    section=CAPTION_SECTION,
                    text=caption,
                    is_caption=True,
                )
            )
    return Paper(paper_id=pid, title=row["title"], abstract=abstract, paragraphs=paragraphs)


def _reference(annotation: dict[str, Any], index: dict[str, str]) -> Reference | None:
    """Mirror the official QASPER evaluator's precedence of answer types."""
    answer_type: AnswerType
    if annotation["unanswerable"]:
        answer_type, text = "unanswerable", UNANSWERABLE_TEXT
    elif annotation["extractive_spans"]:
        answer_type, text = "extractive", ", ".join(annotation["extractive_spans"])
    elif annotation["free_form_answer"]:
        answer_type, text = "abstractive", annotation["free_form_answer"]
    elif annotation["yes_no"] is not None:
        answer_type, text = "yes_no", "Yes" if annotation["yes_no"] else "No"
    else:
        return None
    evidence_ids: list[str] = []
    unmatched = 0
    for evidence in annotation["evidence"]:
        paragraph_id = index.get(_norm(evidence.removeprefix(FLOAT_PREFIX)))
        if paragraph_id is None:
            unmatched += 1
        elif paragraph_id not in evidence_ids:
            evidence_ids.append(paragraph_id)
    return Reference(
        answer_type=answer_type,
        answer_text=text,
        evidence_ids=evidence_ids,
        unmatched_evidence=unmatched,
    )


def parse_questions(row: dict[str, Any], paper: Paper) -> list[Question]:
    index = {_norm(p.text): p.paragraph_id for p in paper.paragraphs}
    qas = row["qas"]
    questions: list[Question] = []
    for text, qid, answers in zip(qas["question"], qas["question_id"], qas["answers"], strict=True):
        refs = [r for a in answers["answer"] if (r := _reference(a, index)) is not None]
        if refs:
            questions.append(
                Question(question_id=qid, paper_id=paper.paper_id, question=text, references=refs)
            )
    return questions


def load_split(path: Path) -> tuple[list[Paper], list[Question]]:
    papers: list[Paper] = []
    questions: list[Question] = []
    for row in pq.read_table(path).to_pylist():
        paper = parse_paper(row)
        papers.append(paper)
        questions.extend(parse_questions(row, paper))
    return papers, questions


def sample_questions(
    questions: list[Question], sample_size: int | None, seed: int
) -> list[Question]:
    ordered = sorted(questions, key=lambda q: q.question_id)
    if sample_size is None or sample_size >= len(ordered):
        return ordered
    picked = random.Random(seed).sample(ordered, sample_size)
    return sorted(picked, key=lambda q: q.question_id)
