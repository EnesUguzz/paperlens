import httpx

from paperlens.config.models import DataConfig
from paperlens.ingestion import docling_parser
from paperlens.ingestion.docling_parser import Block, blocks_to_paper, prepare_pdfs
from paperlens.ingestion.pdf import parsed_path

PID = "1234.5678"
BLOCKS = [
    Block(kind="header", text="Abstract"),
    Block(kind="text", text="We study retrieval."),
    Block(kind="text", text="It works."),
    Block(kind="header", text="1 Introduction"),
    Block(kind="text", text="Intro text."),
    Block(kind="caption", text="Figure 1: A diagram."),
    Block(kind="header", text="2 Method"),
    Block(kind="text", text="Method text."),
    Block(kind="header", text="References"),
    Block(kind="text", text="[1] Someone. A paper."),
    Block(kind="header", text="Appendix"),
    Block(kind="text", text="After references."),
]


def test_blocks_to_paper_sections_and_ids() -> None:
    paper = blocks_to_paper(PID, "T", BLOCKS)
    ids = [p.paragraph_id for p in paper.paragraphs]
    assert ids == [f"{PID}::d{i}" for i in range(5)]
    assert [p.section for p in paper.paragraphs] == [
        "Abstract",
        "Abstract",
        "1 Introduction",
        "Figures and Tables",
        "2 Method",
    ]
    caption = paper.paragraphs[3]
    assert caption.is_caption and caption.text == "Figure 1: A diagram."
    assert not paper.paragraphs[2].is_caption


def test_blocks_to_paper_drops_references() -> None:
    paper = blocks_to_paper(PID, "T", BLOCKS)
    texts = " ".join(p.text for p in paper.paragraphs)
    assert "Someone" not in texts and "After references" not in texts


def test_blocks_to_paper_fills_abstract() -> None:
    paper = blocks_to_paper(PID, "T", BLOCKS)
    assert paper.title == "T"
    assert paper.abstract == "We study retrieval. It works."


def test_prepare_pdfs_resumes(tmp_path, monkeypatch) -> None:
    cfg = DataConfig(data_dir=tmp_path, pdf_download_delay_s=0.5)
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path.endswith("0000.0000"):
            return httpx.Response(404)
        return httpx.Response(200, content=b"%PDF-1.5 x")

    parsed: list[str] = []

    def fake_parse(path):
        parsed.append(path.stem)
        return BLOCKS[:2]

    monkeypatch.setattr(docling_parser, "parse_pdf", fake_parse)
    sleeps: list[float] = []
    client = httpx.Client(transport=httpx.MockTransport(handler))
    ids = ["1111.1111", "0000.0000", "2222.2222"]

    stats = prepare_pdfs(cfg, ids, client=client, sleep=sleeps.append)
    assert (stats.downloaded, stats.missing, stats.parsed, stats.failed) == (2, 1, 2, 0)
    assert parsed_path(cfg, "1111.1111").exists()
    assert sleeps == [0.5, 0.5]  # between the 3 network requests only

    sleeps.clear()
    parsed.clear()
    stats = prepare_pdfs(cfg, ids, client=client, sleep=sleeps.append)
    assert parsed == []  # already parsed -> skipped
    assert stats.parsed == 0 and stats.cached == 2 and stats.missing == 1
    assert sleeps == []  # the 404 id is retried, but it is the only network request
