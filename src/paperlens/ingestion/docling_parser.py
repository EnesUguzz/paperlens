"""Parse PDFs with Docling into Paper objects and run the PDF preparation job.

Purpose: Build the 'pdf_docling' text source: PDF -> labelled blocks -> Paper with section
names, caption flags and references removed, so retrieval can be compared with QASPER's own
parsed text. Parsed blocks are cached as JSON so the slow Docling step runs once per paper.
Flow: ids -> pdf.download_pdf() (paced) -> parse_pdf() -> data/parsed/docling/<id>.json;
later blocks_to_paper(id, title, blocks) -> Paper.
Params: data.pdf_url_template, data.pdf_download_delay_s, data.data_dir.
Try: Open one parsed JSON and check that header blocks look like section titles and that no
reference list entries remain; wrong labels here directly shift retrieval scores.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Literal

import httpx
import structlog
from pydantic import BaseModel, TypeAdapter

from paperlens.config.models import DataConfig
from paperlens.domain import Paper, Paragraph
from paperlens.ingestion.pdf import (
    PdfDownloadError,
    build_client,
    download_pdf,
    parsed_path,
    pdf_path,
)
from paperlens.ingestion.qasper import ABSTRACT_SECTION, CAPTION_SECTION, UNTITLED_SECTION

log = structlog.get_logger()

BlockKind = Literal["header", "text", "caption"]

# Docling label -> our block kind; labels not listed here (page_header, page_footer,
# footnote, picture, table, formula, ...) are dropped.
LABEL_TO_KIND: dict[str, BlockKind] = {
    "title": "header",
    "section_header": "header",
    "text": "text",
    "paragraph": "text",
    "list_item": "text",
    "caption": "caption",
}
REFERENCE_HEADERS = ("references", "bibliography")  # text after such a header is dropped
_NUMBERING = re.compile(r"^(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+")  # "7 ", "VII. ", "2.1 "


class Block(BaseModel):
    kind: BlockKind
    text: str


_BLOCKS = TypeAdapter(list[Block])


@dataclass
class PrepareStats:
    downloaded: int = 0
    cached: int = 0
    missing: int = 0
    failed: int = 0
    parsed: int = 0


@cache
def _converter() -> Any:
    """One shared Docling converter; building it loads layout models (slow), so do it once."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # arXiv PDFs carry a text layer, and table bodies are not used, so both are off.
    options = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def parse_pdf(path: Path) -> list[Block]:
    """Run Docling on one PDF and keep only the labels we use. Needs `--extra pdf`."""
    converter = _converter()
    doc = converter.convert(path).document
    blocks: list[Block] = []
    for item, _level in doc.iterate_items():
        raw = getattr(item, "label", "")
        label = getattr(raw, "value", str(raw))
        kind = LABEL_TO_KIND.get(label)
        text = getattr(item, "text", "") or ""
        if kind is not None and text.strip():
            blocks.append(Block(kind=kind, text=" ".join(text.split())))
    return blocks


def _is_reference_header(text: str) -> bool:
    return _NUMBERING.sub("", text.strip()).lower().startswith(REFERENCE_HEADERS)


def blocks_to_paper(paper_id: str, title: str, blocks: Sequence[Block]) -> Paper:
    paragraphs: list[Paragraph] = []
    section = UNTITLED_SECTION
    for block in blocks:
        if block.kind == "header":
            if _is_reference_header(block.text):
                break
            section = ABSTRACT_SECTION if block.text.strip().lower() == "abstract" else block.text
            continue
        is_caption = block.kind == "caption"
        paragraphs.append(
            Paragraph(
                paragraph_id=f"{paper_id}::d{len(paragraphs)}",
                section=CAPTION_SECTION if is_caption else section,
                text=block.text,
                is_caption=is_caption,
            )
        )
    abstract = " ".join(p.text for p in paragraphs if p.section == ABSTRACT_SECTION)
    return Paper(paper_id=paper_id, title=title, abstract=abstract, paragraphs=paragraphs)


def load_blocks(cfg: DataConfig, paper_id: str) -> list[Block]:
    return _BLOCKS.validate_json(parsed_path(cfg, paper_id).read_bytes())


def prepare_pdfs(
    cfg: DataConfig,
    paper_ids: Sequence[str],
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> PrepareStats:
    """Download then parse every paper; reruns skip finished work (resumable)."""
    stats = PrepareStats()
    http = client or build_client()
    available: list[str] = []
    made_request = False
    try:
        for pid in paper_ids:
            if pdf_path(cfg, pid).exists():
                stats.cached += 1
                available.append(pid)
                continue
            if made_request:
                sleep(cfg.pdf_download_delay_s)  # arXiv: pace automated requests
            made_request = True
            try:
                path = download_pdf(cfg, pid, http)
            except (httpx.HTTPError, PdfDownloadError) as err:
                log.warning("pdf_download_failed", paper_id=pid, error=str(err))
                stats.failed += 1
                continue
            if path is None:
                stats.missing += 1
            else:
                stats.downloaded += 1
                available.append(pid)
    finally:
        if client is None:
            http.close()
    for pid in available:
        out = parsed_path(cfg, pid)
        if out.exists():
            continue
        try:
            blocks = parse_pdf(pdf_path(cfg, pid))
        except Exception as err:  # Docling raises many types; one bad PDF must not stop the job
            log.warning("pdf_parse_failed", paper_id=pid, error=str(err))
            stats.failed += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".part")
        tmp.write_bytes(_BLOCKS.dump_json(blocks))
        tmp.replace(out)
        stats.parsed += 1
    return stats
