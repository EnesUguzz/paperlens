"""Render every config field with its documentation as a Markdown table.

Purpose: Backs `paperlens explain-config`, which also writes docs/CONFIG.md, so the
parameter reference is regenerated from code and never edited by hand.
Flow: AppConfig (defaults or a loaded YAML) -> iter_fields() -> render_markdown().
Params: none.
Try: Run it on an ablation YAML to see exactly which values that experiment uses.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, get_args

from pydantic import BaseModel

NUMERIC_TYPES = (int, float)


@dataclass(frozen=True)
class FieldDoc:
    path: str
    value: Any
    annotation: str
    description: str
    is_numeric: bool


def _is_numeric(annotation: Any) -> bool:
    candidates = get_args(annotation) or (annotation,)
    return any(c in NUMERIC_TYPES for c in candidates)


def _type_name(annotation: Any) -> str:
    name = getattr(annotation, "__name__", None)
    return name if isinstance(name, str) else str(annotation).replace("typing.", "")


def iter_fields(model: BaseModel, prefix: str = "") -> Iterator[FieldDoc]:
    for name, info in type(model).model_fields.items():
        value = getattr(model, name)
        path = f"{prefix}{name}"
        if isinstance(value, BaseModel):
            yield from iter_fields(value, prefix=f"{path}.")
            continue
        yield FieldDoc(
            path=path,
            value=value,
            annotation=_type_name(info.annotation),
            description=info.description or "",
            is_numeric=_is_numeric(info.annotation),
        )


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def render_markdown(model: BaseModel) -> str:
    lines = ["| Parameter | Value | Type | Description |", "|---|---|---|---|"]
    for doc in iter_fields(model):
        lines.append(
            f"| `{doc.path}` | `{_cell(str(doc.value))}` | {_cell(doc.annotation)} "
            f"| {_cell(doc.description)} |"
        )
    return "\n".join(lines) + "\n"
