from pathlib import Path

import pytest
from pydantic import ValidationError

from paperlens.config.explain import iter_fields, render_markdown
from paperlens.config.loader import deep_merge, load_config
from paperlens.config.models import AppConfig, ChunkingConfig, RetrievalConfig


def test_defaults_are_valid() -> None:
    cfg = AppConfig()
    assert cfg.retrieval.method == "dense"
    assert cfg.chunking.chunk_size == 200


def test_deep_merge_keeps_unrelated_keys() -> None:
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 1}, {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 1}


def test_empty_mapping_resets_inherited_mapping() -> None:
    merged = deep_merge(
        {"llm": {"extra_body": {"reasoning_effort": "none"}}}, {"llm": {"extra_body": {}}}
    )
    assert merged == {"llm": {"extra_body": {}}}


def test_extends_chain(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("retrieval:\n  top_k: 5\n  method: dense\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nretrieval:\n  method: bm25\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.retrieval.method == "bm25"
    assert cfg.retrieval.top_k == 5


def test_circular_extends_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("retrieval:\n  topk: 5\n")
    with pytest.raises(ValidationError):
        load_config(tmp_path / "bad.yaml")


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size=50, chunk_overlap=50)


def test_candidate_k_must_cover_top_k() -> None:
    with pytest.raises(ValidationError):
        RetrievalConfig(top_k=10, candidate_k=5)


def test_chunks_must_fit_embedder_window() -> None:
    with pytest.raises(ValidationError, match="max_seq_length"):
        AppConfig(chunking={"chunk_size": 800}, embedding={"max_seq_length": 512})


def test_explain_lists_nested_fields() -> None:
    paths = [d.path for d in iter_fields(AppConfig())]
    assert "chunking.chunk_size" in paths
    assert "llm.model" in paths
    table = render_markdown(AppConfig())
    assert "| `retrieval.top_k` | `5` |" in table


M1B_CONFIG_DIR = Path(__file__).resolve().parents[1] / "experiments" / "configs" / "m1b"


def test_m1b_config_dir_is_not_empty() -> None:
    assert sorted(M1B_CONFIG_DIR.glob("*.yaml")), "the parametrized test below would run 0 cases"


@pytest.mark.parametrize("path", sorted(M1B_CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_m1b_configs_load_and_name_their_run(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.eval.run_name == path.stem
    assert cfg.eval.experiment_name == "m1b"
    assert cfg.eval.reference_run == "baseline"


def test_semantic_min_words_must_be_below_chunk_size() -> None:
    with pytest.raises(ValidationError):
        ChunkingConfig(strategy="semantic", chunk_size=50, semantic_min_words=50)


def test_m2_defaults() -> None:
    cfg = AppConfig()
    assert cfg.eval.system == "single_paper"
    assert cfg.agent.route.top_k == 10
    assert cfg.agent.route.min_lead == 0.3
    assert cfg.agent.route.title_match_min == 90.0
    assert cfg.agent.clarify.max_candidates == 3
    assert cfg.agent.condense.enabled and cfg.agent.condense.history_turns == 3
    assert cfg.agent.tracing
    conv = cfg.conversations
    assert (conv.papers_per_conversation, conv.return_prob, conv.signal_prob, conv.seed) == (
        3,
        0.5,
        0.5,
        13,
    )
    assert conv.signal_templates == [
        'In the paper "{title}", {question}',
        "In {short_title}, {question}",
        "About the {short_title} paper: {question}",
    ]
