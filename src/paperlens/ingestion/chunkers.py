"""Cut papers into retrievable chunks while remembering which paragraphs each covers.

Purpose: Chunking is the first lever on retrieval quality. Every Chunk keeps its
paragraph_ids so gold evidence (paragraph ids) can be matched to retrieved chunks.
Flow: Paper (ingestion/qasper.py) -> chunk_paper() -> list[Chunk] -> index/store.py.
Params: chunking.strategy, chunking.chunk_size, chunking.chunk_overlap,
chunking.include_section_title, chunking.include_captions.
Try: Compare 'fixed' vs 'section' at the same chunk_size, then halve chunk_size and watch
Recall@k and Evidence-F1 move in opposite directions.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import groupby

from paperlens.config.models import ChunkingConfig
from paperlens.domain import Chunk, Paper, Paragraph, Segment

# (section, text, paragraph_ids, segments) before ids and prefixes are attached
_Piece = tuple[str, str, list[str], list[Segment]]


def chunk_paper(paper: Paper, cfg: ChunkingConfig) -> list[Chunk]:
    paragraphs = [p for p in paper.paragraphs if cfg.include_captions or not p.is_caption]
    if cfg.strategy == "fixed":
        pieces = _fixed(paragraphs, cfg.chunk_size, cfg.chunk_overlap)
    else:
        pieces = _section(paragraphs, cfg.chunk_size)
    return [
        Chunk(
            chunk_id=f"{paper.paper_id}::c{i}",
            paper_id=paper.paper_id,
            section=section,
            text=f"[{section}] {text}" if cfg.include_section_title else text,
            paragraph_ids=paragraph_ids,
            segments=segments,
        )
        for i, (section, text, paragraph_ids, segments) in enumerate(pieces)
    ]


def _fixed(paragraphs: Sequence[Paragraph], size: int, overlap: int) -> list[_Piece]:
    """Sliding window over all words; the section of a window is that of its first word."""
    words = [(word, p) for p in paragraphs for word in p.text.split()]
    pieces: list[_Piece] = []
    if not words:
        return pieces
    stride = size - overlap
    start = 0
    while True:
        window = words[start : start + size]
        paragraph_ids = list(dict.fromkeys(p.paragraph_id for _, p in window))
        segments: list[Segment] = []  # consecutive words of one paragraph form one segment
        for pid, group in groupby(window, key=lambda wp: wp[1].paragraph_id):
            segments.append(Segment(paragraph_id=pid, text=" ".join(w for w, _ in group)))
        pieces.append(
            (window[0][1].section, " ".join(w for w, _ in window), paragraph_ids, segments)
        )
        if start + size >= len(words):
            return pieces
        start += stride


def _section(paragraphs: Sequence[Paragraph], size: int) -> list[_Piece]:
    """Pack whole paragraphs up to `size` words; never cross a section boundary."""
    pieces: list[_Piece] = []
    buffer: list[Paragraph] = []
    buffer_words = 0

    def flush() -> None:
        if buffer:
            pieces.append(
                (
                    buffer[0].section,
                    " ".join(p.text for p in buffer),
                    [p.paragraph_id for p in buffer],
                    [Segment(paragraph_id=p.paragraph_id, text=p.text) for p in buffer],
                )
            )

    for para in paragraphs:
        words = para.text.split()
        if buffer and (para.section != buffer[0].section or buffer_words + len(words) > size):
            flush()
            buffer, buffer_words = [], 0
        if len(words) > size:  # a single paragraph longer than a chunk: split it on its own
            for start in range(0, len(words), size):
                part = " ".join(words[start : start + size])
                pieces.append(
                    (
                        para.section,
                        part,
                        [para.paragraph_id],
                        [Segment(paragraph_id=para.paragraph_id, text=part)],
                    )
                )
            continue
        buffer.append(para)
        buffer_words += len(words)
    flush()
    return pieces
