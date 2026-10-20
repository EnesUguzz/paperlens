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


def test_fixed_chunk_segments_rebuild_chunk_text(sample_paper: Paper) -> None:
    chunks = chunk_paper(sample_paper, cfg(strategy="fixed", chunk_size=10, chunk_overlap=2))
    for c in chunks:
        assert " ".join(s.text for s in c.segments) == c.text
        assert [s.paragraph_id for s in c.segments] == c.paragraph_ids


def test_segments_exclude_the_section_prefix(sample_paper: Paper) -> None:
    chunks = chunk_paper(
        sample_paper,
        cfg(strategy="fixed", chunk_size=10, chunk_overlap=2, include_section_title=True),
    )
    for c in chunks:
        prefix = f"[{c.section}] "
        assert c.text.startswith(prefix)
        assert " ".join(s.text for s in c.segments) == c.text[len(prefix) :]


def test_section_chunk_segments_are_whole_paragraphs(sample_paper: Paper) -> None:
    chunks = chunk_paper(sample_paper, cfg(strategy="section", chunk_size=20, chunk_overlap=0))
    by_id = {p.paragraph_id: p.text for p in sample_paper.paragraphs}
    for c in chunks:
        assert [s.paragraph_id for s in c.segments] == c.paragraph_ids
        assert all(s.text == by_id[s.paragraph_id] for s in c.segments)


# --- semantic strategy ----------------------------------------------------------------------

RETRIEVAL = [
    f"Retrieval index query ranks documents {w}." for w in ("fast", "well", "often", "now")
]
WEATHER = [f"Weather rain sun cloud wind {w}." for w in ("today", "later", "again", "soon")]


def _paper(*paras: tuple[str, str, str]) -> Paper:
    """paras: (paragraph_id suffix, section, text)."""
    from paperlens.domain import Paragraph

    return Paper(
        paper_id="p",
        title="t",
        abstract="",
        paragraphs=[Paragraph(paragraph_id=f"p::{i}", section=s, text=t) for i, s, t in paras],
    )


def _semantic_cfg(**kw: object) -> ChunkingConfig:
    return cfg(strategy="semantic", chunk_size=200, chunk_overlap=0, semantic_min_words=5, **kw)


def test_semantic_breaks_between_topics() -> None:
    from paperlens.index.embedder import HashingEmbedder

    paper = _paper(("0", "Intro", " ".join(RETRIEVAL + WEATHER)))
    chunks = chunk_paper(paper, _semantic_cfg(), HashingEmbedder())
    assert [c.text for c in chunks] == [" ".join(RETRIEVAL), " ".join(WEATHER)]
    for c in chunks:  # one paragraph -> one segment, same text
        assert [s.text for s in c.segments] == [c.text]
        assert c.paragraph_ids == ["p::0"]


def test_semantic_never_crosses_sections() -> None:
    from paperlens.index.embedder import HashingEmbedder

    # identical topic in two sections: no semantic reason to split, only the section change
    paper = _paper(("0", "A", " ".join(RETRIEVAL[:2])), ("1", "B", " ".join(RETRIEVAL[2:])))
    chunks = chunk_paper(paper, _semantic_cfg(), HashingEmbedder())
    assert [c.section for c in chunks] == ["A", "B"]
    assert [c.paragraph_ids for c in chunks] == [["p::0"], ["p::1"]]


def test_semantic_respects_chunk_size() -> None:
    from paperlens.index.embedder import HashingEmbedder

    paper = _paper(("0", "A", " ".join(RETRIEVAL * 3)))
    chunks = chunk_paper(
        paper,
        cfg(strategy="semantic", chunk_size=12, chunk_overlap=0, semantic_min_words=5),
        HashingEmbedder(),
    )
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 12 for c in chunks)
    long_sentence = " ".join(f"w{i}" for i in range(30)) + "."  # one sentence > chunk_size
    chunks = chunk_paper(
        _paper(("0", "A", long_sentence)),
        cfg(strategy="semantic", chunk_size=12, chunk_overlap=0, semantic_min_words=5),
        HashingEmbedder(),
    )
    assert [len(c.text.split()) for c in chunks] == [12, 12, 6]


def test_semantic_segments_follow_paragraphs() -> None:
    from paperlens.index.embedder import HashingEmbedder

    paper = _paper(("0", "A", " ".join(RETRIEVAL[:2])), ("1", "A", " ".join(RETRIEVAL[2:])))
    chunks = chunk_paper(paper, _semantic_cfg(), HashingEmbedder())
    for c in chunks:
        assert " ".join(s.text for s in c.segments) == c.text
        assert [s.paragraph_id for s in c.segments] == c.paragraph_ids


def test_semantic_without_embedder_raises() -> None:
    import pytest

    with pytest.raises(ValueError, match="embedder"):
        chunk_paper(_paper(("0", "A", RETRIEVAL[0])), _semantic_cfg())
