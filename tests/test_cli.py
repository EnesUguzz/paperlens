from typer.testing import CliRunner

from paperlens.cli import app


def test_explain_config_prints_table() -> None:
    out = CliRunner().invoke(app, ["explain-config"])
    assert out.exit_code == 0
    assert "chunking.chunk_size" in out.output
