"""Cut papers into retrievable chunks while remembering which paragraphs each covers.

Purpose: Chunking is the first lever on retrieval quality. Every Chunk keeps its
paragraph_ids so gold evidence (paragraph ids) can be matched to retrieved chunks.
Flow: Paper (ingestion/qasper.py) -> chunk_paper() -> list[Chunk] -> index/store.py.
Params: chunking.strategy, chunking.chunk_size, chunking.chunk_overlap,
chunking.include_section_title, chunking.include_captions,
chunking.semantic_breakpoint_percentile, chunking.semantic_min_words.
Try: Compare 'fixed' vs 'section' vs 'semantic' at the same chunk_size, then halve chunk_size and watch
Recall@k and Evidence-F1 move in opposite directions.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from itertools import groupby

import numpy as np

from paperlens.config.models import ChunkingConfig
from paperlens.domain import Chunk, Paper, Paragraph, Segment
from paperlens.index.embedder import Embedder

# A sentence ends at . ! ? followed by whitespace and a capital letter, digit, '[' or '('
# (so 'e.g. the' and '0.5 F1' do not split): a cheap heuristic that needs no NLP model.
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")

# (section, text, paragraph_ids, segments) before ids and prefixes are attached
_Piece = tuple[str, str, list[str], list[Segment]]


def chunk_paper(paper: Paper, cfg: ChunkingConfig, embedder: Embedder | None = None) -> list[Chunk]:
    paragraphs = [p for p in paper.paragraphs if cfg.include_captions or not p.is_caption]
    if cfg.strategy == "fixed":
        pieces = _fixed(paragraphs, cfg.chunk_size, cfg.chunk_overlap)
    elif cfg.strategy == "semantic":
        if embedder is None:
            raise ValueError("chunking.strategy='semantic' needs an embedder")
        pieces = _semantic(
            paragraphs,
            cfg.chunk_size,
            cfg.semantic_min_words,
            cfg.semantic_breakpoint_percentile,
            embedder,
        )
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


def _semantic(
    paragraphs: Sequence[Paragraph],
    size: int,
    min_words: int,
    percentile: float,
    embedder: Embedder,
) -> list[_Piece]:
    """Pack sentences; close a chunk at a section change, at `size` words, or where the
    embedding distance to the previous sentence is above the paper's `percentile`."""
    sentences = [
        (p, text)
        for p in paragraphs
        for text in (t.strip() for t in SENTENCE_BOUNDARY.split(p.text))
        if text
    ]
    if not sentences:
        return []
    vectors = embedder.embed_passages([text for _, text in sentences])  # one call per paper
    # distances[i] = cosine distance between sentence i and i+1 (vectors are L2-normalised);
    # NaN across a section change, where the chunk closes anyway.
    distances = [
        1.0 - float(np.dot(vectors[i], vectors[i + 1]))
        if sentences[i][0].section == sentences[i + 1][0].section
        else float("nan")
        for i in range(len(sentences) - 1)
    ]
    valid = [d for d in distances if d == d]
    threshold = float(np.percentile(valid, percentile)) if valid else float("inf")

    pieces: list[_Piece] = []
    buffer: list[tuple[Paragraph, str]] = []
    buffer_words = 0

    def flush() -> None:
        segments: list[Segment] = []  # consecutive sentences of one paragraph form one segment
        for pid, group in groupby(buffer, key=lambda ps: ps[0].paragraph_id):
            segments.append(Segment(paragraph_id=pid, text=" ".join(t for _, t in group)))
        pieces.append(
            (
                buffer[0][0].section,
                " ".join(t for _, t in buffer),
                list(dict.fromkeys(s.paragraph_id for s in segments)),
                segments,
            )
        )

    for i, (para, text) in enumerate(sentences):
        words = text.split()
        if len(words) > size:  # a single sentence longer than a chunk: split it on its own
            if buffer:
                flush()
                buffer, buffer_words = [], 0
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
        if buffer and (
            para.section != buffer[0][0].section
            or buffer_words + len(words) > size
            or (distances[i - 1] > threshold and buffer_words >= min_words)
        ):
            flush()
            buffer, buffer_words = [], 0
        buffer.append((para, text))
        buffer_words += len(words)
    if buffer:
        flush()
    return pieces
