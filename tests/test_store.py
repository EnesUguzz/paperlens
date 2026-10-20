from pathlib import Path

from paperlens.config.models import AppConfig
from paperlens.domain import Paper
from paperlens.index.embedder import HashingEmbedder
from paperlens.index.store import build_or_load, index_key


def config(tmp_path: Path, **chunking: object) -> AppConfig:
    return AppConfig(
        data={"data_dir": tmp_path},
        embedding={"model_name": "hashing"},
        chunking={"chunk_size": 10, "chunk_overlap": 2, **chunking},
    )


def test_second_call_is_a_cache_hit(tmp_path: Path, sample_paper: Paper) -> None:
    cfg = config(tmp_path)
    first = build_or_load(cfg, [sample_paper], HashingEmbedder())
    first.dense.close()
    second = build_or_load(cfg, [sample_paper], HashingEmbedder())
    assert not first.cache_hit and second.cache_hit
    assert [c.chunk_id for c in second.chunks] == [c.chunk_id for c in first.chunks]
    assert second.dense.count() == len(second.chunks)
    second.dense.close()


def test_interrupted_build_is_rebuilt(tmp_path: Path, sample_paper: Paper) -> None:
    cfg = config(tmp_path)
    first = build_or_load(cfg, [sample_paper], HashingEmbedder())
    first.dense.close()
    # simulate Ctrl-C before the chunks file (the commit marker) was written
    (tmp_path / "index" / index_key(cfg) / "chunks.jsonl").unlink()
    again = build_or_load(cfg, [sample_paper], HashingEmbedder())
    assert not again.cache_hit
    again.dense.close()


def test_key_depends_only_on_index_settings(tmp_path: Path) -> None:
    base = config(tmp_path)
    assert index_key(base) != index_key(config(tmp_path, chunk_size=12))
    retrieval = base.retrieval.model_copy(update={"top_k": 9})
    assert index_key(base) == index_key(base.model_copy(update={"retrieval": retrieval}))


def test_store_rebuilds_cache_without_segments(tmp_path: Path, sample_paper: Paper) -> None:
    import json

    cfg = config(tmp_path)
    first = build_or_load(cfg, [sample_paper], HashingEmbedder())
    first.dense.close()
    chunks_file = tmp_path / "index" / index_key(cfg) / "chunks.jsonl"
    old = []
    for line in chunks_file.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        del row["segments"]  # a cache written before segments existed
        old.append(json.dumps(row))
    chunks_file.write_text("\n".join(old) + "\n", encoding="utf-8")
    again = build_or_load(cfg, [sample_paper], HashingEmbedder())
    assert not again.cache_hit
    assert all(c.segments for c in again.chunks)
    assert [c.chunk_id for c in again.chunks] == [c.chunk_id for c in first.chunks]
    again.dense.close()
