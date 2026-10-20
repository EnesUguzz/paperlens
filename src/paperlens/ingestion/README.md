# ingestion

| File | Role |
|---|---|
| `qasper.py` | download HF parquet, parse papers/questions, map gold evidence -> paragraph ids, sample `dev_small` |
| `chunkers.py` | cut a paper into chunks (`fixed` or `section`), each remembering its paragraph ids and its paragraph `segments` (what `generation.cite_unit: paragraph` cites) |
| `pdf.py` | download arXiv PDFs (cached, `.part` + rename, User-Agent, 404 -> None, rejects non-PDF bodies); on-disk paths |
| `docling_parser.py` | Docling PDF -> labelled blocks -> `Paper` (`::d<n>` ids, references dropped); `prepare_pdfs` job (paced, resumable) |

Paragraph ids (`<paper>::s<section>p<paragraph>`, `::abs`, `::f<caption>`; PDF-parsed papers use `::d<n>`) are how retrieval
gets scored: a gold evidence paragraph is "found" if a retrieved chunk contains its id.
Experiments touching this package: chunk size / overlap / strategy, captions on/off, `data.pdf_download_delay_s` (>= 3 s for arXiv).
