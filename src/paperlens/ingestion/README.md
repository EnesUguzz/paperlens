# ingestion

| File | Role |
|---|---|
| `qasper.py` | download HF parquet, parse papers/questions, map gold evidence -> paragraph ids, sample `dev_small` |
| `chunkers.py` | cut a paper into chunks (`fixed` or `section`), each remembering its paragraph ids and its paragraph `segments` (what `generation.cite_unit: paragraph` cites) |
| `pdf.py` | download arXiv PDFs (cached, `.part` + rename, User-Agent, 404 -> None, rejects non-PDF bodies); on-disk paths |
| `docling_parser.py` | Docling PDF -> labelled blocks -> `Paper` (`::d<n>` ids, references dropped); `prepare_pdfs` job (paced, resumable) |
| `align.py` | fuzzy-align QASPER gold evidence paragraphs to PDF paragraphs (rapidfuzz `partial_ratio`); unmatched evidence becomes a `<paper>::unaligned<n>` sentinel id, never an empty list |
| `sources.py` | `load_dataset(cfg.data)`: the single entry point for `data.text_source` (`qasper_text` or `pdf_docling`); `sample_dataset` samples all questions first, then drops papers without a parsed PDF |

Paragraph ids (`<paper>::s<section>p<paragraph>`, `::abs`, `::f<caption>`; PDF-parsed papers use `::d<n>`) are how retrieval
gets scored: a gold evidence paragraph is "found" if a retrieved chunk contains its id.
Experiments touching this package: chunk size / overlap / strategy, captions on/off, `data.pdf_download_delay_s` (>= 3 s for arXiv).

`pdf_docling` evidence: gold evidence is re-pointed from QASPER paragraph ids to PDF paragraph ids by fuzzy matching
(`data.align_min_score`, `data.align_min_words`). A QASPER paragraph split by a page break maps to several PDF ids
(`data.evidence_split_rate`); evidence with no match keeps a `::unaligned<n>` id so it scores as never found
(`data.evidence_aligned_rate` below ~0.8 means alignment loss dominates any parser comparison; inspect examples with
`align.scored_matches` before touching the threshold). Experiments touching this: `data.text_source`, `data.align_*`.
