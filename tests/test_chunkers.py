from paperlens.config.models import ChunkingConfig
from paperlens.domain import Paper
from paperlens.ingestion.chunkers import chunk_paper

PID = "1234.5678"


def cfg(**kw: object) -> ChunkingConfig:
    base: dict[str, object] = {"include_section_title": False}
    base.update(kw)
    return ChunkingConfig(**base)  # type: ignore[arg-type]


def test_fixed_windows_overlap(sample_paper: Paper) -> None:
    chunks = chunk_paper(sample_paper, cfg(strategy="fixed", chunk_size=10, chunk_overlap=2))
    assert len(chunks) == 4  # 31 words, stride 8 -> starts 0, 8, 16, 24
    assert chunks[0].paragraph_ids == [f"{PID}::abs", f"{PID}::s0p0"]
    assert chunks[0].text.split()[-2:] == chunks[1].text.split()[:2]
    covered = {pid for c in chunks for pid in c.paragraph_ids}
    assert covered == {p.paragraph_id for p in sample_paper.paragraphs}


def test_captions_can_be_excluded(sample_paper: Paper) -> None:
    chunks = chunk_paper(sample_paper, cfg(chunk_size=10, chunk_overlap=2, include_captions=False))
    assert all(f"{PID}::f0" not in c.paragraph_ids for c in chunks)


def test_section_strategy_never_crosses_sections(sample_paper: Paper) -> None:
    chunks = chunk_paper(sample_paper, cfg(strategy="section", chunk_size=20, chunk_overlap=0))
    assert [c.paragraph_ids for c in chunks] == [
        [f"{PID}::abs"],
        [f"{PID}::s0p0", f"{PID}::s0p1"],
        [f"{PID}::s1p0"],
        [f"{PID}::f0"],
    ]
    assert [c.section for c in chunks] == [
        "Abstract",
        "Introduction",
        "Experiments",
        "Figures and Tables",
    ]


def test_section_strategy_splits_oversized_paragraphs(sample_paper: Paper) -> None:
    chunks = chunk_paper(sample_paper, cfg(strategy="section", chunk_size=6, chunk_overlap=0))
    s1p0 = [c for c in chunks if c.paragraph_ids == [f"{PID}::s1p0"]]
    assert len(s1p0) == 2  # 9 words -> 6 + 3
    assert all(len(c.text.split()) <= 6 for c in chunks)


def test_section_title_prefix(sample_paper: Paper) -> None:
    chunks = chunk_paper(
        sample_paper,
        cfg(strategy="section", chunk_size=20, chunk_overlap=0, include_section_title=True),
    )
    assert chunks[1].text.startswith("[Introduction] ")


def test_chunk_ids_are_unique_and_stable(sample_paper: Paper) -> None:
    a = chunk_paper(sample_paper, cfg(chunk_size=10, chunk_overlap=2))
    b = chunk_paper(sample_paper, cfg(chunk_size=10, chunk_overlap=2))
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_empty_paper_gives_no_chunks() -> None:
    empty = Paper(paper_id="x", title="t", abstract="", paragraphs=[])
    assert chunk_paper(empty, cfg(chunk_size=10, chunk_overlap=2)) == []
