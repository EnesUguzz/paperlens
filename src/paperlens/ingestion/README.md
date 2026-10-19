# ingestion

| File | Role |
|---|---|
| `qasper.py` | download HF parquet, parse papers/questions, map gold evidence -> paragraph ids, sample `dev_small` |
| `chunkers.py` | cut a paper into chunks (`fixed` or `section`), each remembering its paragraph ids |

Paragraph ids (`<paper>::s<section>p<paragraph>`, `::abs`, `::f<caption>`) are how retrieval
gets scored: a gold evidence paragraph is "found" if a retrieved chunk contains its id.
Experiments touching this package: chunk size / overlap / strategy, captions on/off.
