"""Download arXiv PDFs for QASPER papers (cached on disk, polite to arXiv).

Purpose: Fetch the original PDF of a paper so Docling can parse it (M1b measures how parser
quality changes retrieval). Knows where PDFs and their parsed JSON live on disk.
Flow: paper id -> download_pdf() -> data/raw/pdf/<id>.pdf -> docling_parser.parse_pdf()
-> data/parsed/docling/<id>.json.
Params: data.pdf_url_template, data.pdf_download_delay_s, data.data_dir.
Try: Download one PDF, open it, and compare its first page with QASPER's abstract for the
same id; a missing (404) id means the paper has no PDF under that arXiv identifier.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from paperlens.config.models import DataConfig

# arXiv asks automated clients to identify themselves.
USER_AGENT = "paperlens/0.1 (research; QASPER reproduction)"
PDF_MAGIC = b"%PDF"  # every real PDF starts with this; HTML error pages do not
DOWNLOAD_TIMEOUT_S = 120.0


class PdfDownloadError(RuntimeError):
    """The server answered, but not with a PDF (captcha page, rate-limit page, ...)."""


def pdf_path(cfg: DataConfig, paper_id: str) -> Path:
    return cfg.data_dir / "raw" / "pdf" / f"{paper_id}.pdf"


def parsed_path(cfg: DataConfig, paper_id: str) -> Path:
    return cfg.data_dir / "parsed" / "docling" / f"{paper_id}.json"


def build_client() -> httpx.Client:
    """HTTP client with the descriptive User-Agent arXiv expects."""
    return httpx.Client(
        follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_S, headers={"User-Agent": USER_AGENT}
    )


def download_pdf(cfg: DataConfig, paper_id: str, client: httpx.Client) -> Path | None:
    """Return the cached or freshly downloaded PDF; None when arXiv answers 404."""
    target = pdf_path(cfg, paper_id)
    if target.exists():
        return target
    response = client.get(
        cfg.pdf_url_template.format(paper_id=paper_id), headers={"User-Agent": USER_AGENT}
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    if not response.content.startswith(PDF_MAGIC):
        raise PdfDownloadError(f"{paper_id}: response is not a PDF")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")
    tmp.write_bytes(response.content)
    tmp.replace(target)  # rename only after a complete write
    return target
