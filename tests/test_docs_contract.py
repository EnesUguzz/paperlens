"""Enforces that every module and package explains itself (see spec section 3)."""

import ast
from pathlib import Path

import pytest

from paperlens.config.explain import iter_fields
from paperlens.config.models import AppConfig

SRC = Path(__file__).resolve().parents[1] / "src" / "paperlens"
REQUIRED_MARKERS = ("Purpose:", "Flow:", "Params:", "Try:")
MODULES = sorted(SRC.rglob("*.py"))


@pytest.mark.parametrize("path", MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_module_docstring_explains_itself(path: Path) -> None:
    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    assert doc, f"{path} has no module docstring"
    if path.name == "__init__.py":
        return
    missing = [m for m in REQUIRED_MARKERS if m not in doc]
    assert not missing, f"{path.name} docstring is missing {missing}"


def test_every_subpackage_has_readme() -> None:
    packages = [d for d in SRC.iterdir() if d.is_dir() and (d / "__init__.py").exists()]
    missing = [d.name for d in packages if not (d / "README.md").exists()]
    assert not missing, f"packages without README.md: {missing}"


def test_every_config_field_is_documented() -> None:
    problems = []
    for doc in iter_fields(AppConfig()):
        needed = ["What:", "Effect:"] + (["Typical:"] if doc.is_numeric else [])
        missing = [m for m in needed if m not in doc.description]
        if missing:
            problems.append(f"{doc.path}: missing {missing}")
    assert not problems, "\n".join(problems)
