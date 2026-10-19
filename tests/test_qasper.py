from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from paperlens.config.models import DataConfig
from paperlens.domain import Paper, Question
from paperlens.ingestion import qasper
from paperlens.ingestion.qasper import download_split, load_split

PID = "1234.5678"


def test_paragraph_ids_and_sections(sample_paper: Paper) -> None:
    ids = [p.paragraph_id for p in sample_paper.paragraphs]
    assert ids == [f"{PID}::abs", f"{PID}::s0p0", f"{PID}::s0p1", f"{PID}::s1p0", f"{PID}::f0"]
    assert sample_paper.paragraphs[-1].is_caption
    assert sample_paper.paragraphs[0].section == "Abstract"


def test_reference_types_and_evidence(sample_questions: list[Question]) -> None:
    q1, q2, q3 = sample_questions
    assert q1.references[0].answer_type == "extractive"
    assert q1.references[0].answer_text == "SQuAD"
    assert q1.references[0].evidence_ids == [f"{PID}::s0p1"]
    assert [r.answer_type for r in q2.references] == ["unanswerable", "yes_no"]
    assert q2.references[0].answer_text == "Unanswerable"
    assert q2.references[1].answer_text == "No"
    assert q2.references[1].evidence_ids == [f"{PID}::f0"]  # FLOAT SELECTED -> caption
    assert q3.references[0].answer_type == "abstractive"
    assert q3.references[0].unmatched_evidence == 1


def test_sampling_is_deterministic(sample_questions: list[Question]) -> None:
    a = [q.question_id for q in qasper.sample_questions(sample_questions, 2, 7)]
    b = [q.question_id for q in qasper.sample_questions(sample_questions, 2, 7)]
    assert a == b and len(a) == 2
    assert len(qasper.sample_questions(sample_questions, None, 7)) == 3


def test_load_split_from_parquet(tmp_path: Path, sample_row: dict[str, Any]) -> None:
    path = tmp_path / "validation.parquet"
    pq.write_table(pa.Table.from_pylist([sample_row]), path)
    papers, questions = load_split(path)
    assert len(papers) == 1 and len(questions) == 3


def test_download_is_cached(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, content=b"PAR1")

    cfg = DataConfig(data_dir=tmp_path)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    first = download_split(cfg, "validation", client=client)
    second = download_split(cfg, "validation", client=client)
    assert first == second and first.read_bytes() == b"PAR1"
    assert len(calls) == 1
    assert calls[0].endswith("/validation/0.parquet")
