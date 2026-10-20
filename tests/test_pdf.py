import httpx
import pytest

from paperlens.config.models import DataConfig
from paperlens.ingestion.pdf import PdfDownloadError, download_pdf, parsed_path, pdf_path

PID = "1234.5678"
PDF_BYTES = b"%PDF-1.5\n fake body"


def _client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler, follow_redirects=True)


def test_download_writes_pdf_once(tmp_path) -> None:
    cfg = DataConfig(data_dir=tmp_path)
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        assert "paperlens" in request.headers["user-agent"]
        return httpx.Response(200, content=PDF_BYTES)

    client = _client(httpx.MockTransport(handler))
    path = download_pdf(cfg, PID, client)
    assert path == pdf_path(cfg, PID) and path.read_bytes() == PDF_BYTES
    assert download_pdf(cfg, PID, client) == path
    assert calls == [f"https://arxiv.org/pdf/{PID}"]  # second call hit the cache
    assert not path.with_suffix(".part").exists()


def test_download_404_returns_none(tmp_path) -> None:
    cfg = DataConfig(data_dir=tmp_path)
    client = _client(httpx.MockTransport(lambda r: httpx.Response(404)))
    assert download_pdf(cfg, PID, client) is None
    assert not pdf_path(cfg, PID).exists()


def test_download_rejects_html_body(tmp_path) -> None:
    cfg = DataConfig(data_dir=tmp_path)
    client = _client(httpx.MockTransport(lambda r: httpx.Response(200, content=b"<html>captcha")))
    with pytest.raises(PdfDownloadError):
        download_pdf(cfg, PID, client)
    assert not pdf_path(cfg, PID).exists()
    assert not pdf_path(cfg, PID).with_suffix(".part").exists()


def test_paths(tmp_path) -> None:
    cfg = DataConfig(data_dir=tmp_path)
    assert pdf_path(cfg, PID) == tmp_path / "raw" / "pdf" / f"{PID}.pdf"
    assert parsed_path(cfg, PID) == tmp_path / "parsed" / "docling" / f"{PID}.json"
