import copy
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from paperlens.config.models import DataConfig
from paperlens.ingestion.docling_parser import _BLOCKS, Block
from paperlens.ingestion.pdf import parsed_path
from paperlens.ingestion.qasper import load_split, raw_path, sample_questions
from paperlens.ingestion.sources import load_dataset, sample_dataset

PID = "1234.5678"
OTHER = "9999.0001"
SPLIT = "validation"


def _two_paper_rows(sample_row: dict[str, Any]) -> list[dict[str, Any]]:
    other = copy.deepcopy(sample_row)
    other["id"] = OTHER
    other["qas"]["question_id"] = [f"o{i}" for i in range(1, 4)]
    return [sample_row, other]


def _write_split(cfg: DataConfig, rows: list[dict[str, Any]]) -> None:
    path = raw_path(cfg, SPLIT)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def _write_parsed(cfg: DataConfig, paper_id: str) -> None:
    blocks = [
        Block(kind="header", text="Experiments"),
        Block(kind="text", text="We use the SQuAD dataset for evaluation."),
        Block(kind="text", text="Our model reaches 81.2 F1 on the test set."),
        Block(kind="caption", text="Table 1: Main results."),
    ]
    path = parsed_path(cfg, paper_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_BLOCKS.dump_json(blocks))


def test_load_dataset_qasper_is_unchanged(tmp_path: Path, sample_row: dict[str, Any]) -> None:
    cfg = DataConfig(data_dir=tmp_path, align_min_words=3)
    _write_split(cfg, [sample_row])
    papers, questions = load_split(raw_path(cfg, SPLIT))
    loaded = load_dataset(cfg)
    assert loaded.papers == papers and loaded.questions == questions
    assert loaded.missing_paper_ids == set()
    assert "data.unmatched_evidence_rate" in loaded.metrics


def test_load_dataset_pdf_aligns_and_reports_missing(
    tmp_path: Path, sample_row: dict[str, Any]
) -> None:
    cfg = DataConfig(data_dir=tmp_path, text_source="pdf_docling", align_min_words=3)
    _write_split(cfg, _two_paper_rows(sample_row))
    _write_parsed(cfg, PID)
    loaded = load_dataset(cfg)
    assert [p.paper_id for p in loaded.papers] == [PID]
    assert loaded.missing_paper_ids == {OTHER}
    assert loaded.metrics["data.papers_missing"] == 1.0
    q1 = next(q for q in loaded.questions if q.question_id == "q1")
    assert q1.references[0].evidence_ids == [f"{PID}::d0"]
    # every QASPER-matched evidence string aligns to its PDF paragraph or caption
    assert loaded.metrics["data.evidence_aligned_rate"] == 1.0
    assert loaded.metrics["data.evidence_split_rate"] == 0.0


def test_load_dataset_pdf_samples_same_question_ids(
    tmp_path: Path, sample_row: dict[str, Any]
) -> None:
    base = DataConfig(data_dir=tmp_path, align_min_words=3)
    _write_split(base, _two_paper_rows(sample_row))
    _write_parsed(base, PID)
    qasper_sample = sample_questions(load_dataset(base).questions, 4, base.seed)
    pdf_cfg = base.model_copy(update={"text_source": "pdf_docling"})
    sample, dropped = sample_dataset(load_dataset(pdf_cfg), 4, base.seed)
    assert {q.question_id for q in sample} <= {q.question_id for q in qasper_sample}
    assert all(q.paper_id == PID for q in sample)
    assert dropped == len(qasper_sample) - len(sample) > 0


def test_load_dataset_pdf_without_parsed_files_explains(
    tmp_path: Path, sample_row: dict[str, Any]
) -> None:
    cfg = DataConfig(data_dir=tmp_path, text_source="pdf_docling")
    _write_split(cfg, [sample_row])
    with pytest.raises(FileNotFoundError, match="prepare-pdfs"):
        load_dataset(cfg)
