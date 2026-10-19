"""Ask for JSON matching a pydantic schema; clean up and repair the reply once.

Purpose: Small local models wrap JSON in <think> blocks or ```json fences and sometimes
break the schema. This turns those replies into validated objects, or a clear error the
pipeline can count, instead of a crash.
Flow: messages + schema -> LLMClient.complete(response_format=json_schema) ->
strip_reasoning() -> extract_json() -> schema.model_validate_json(); one repair round.
Params: none of its own (inherits llm.*). REPAIR_ATTEMPTS is fixed on purpose so
error_rate stays comparable across runs.
Try: Count how often the repair path fires per model (error_rate and completion_tokens
in the report); a model that needs repairs often is slower than its tokens/s suggests.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ValidationError

from paperlens.domain import Usage
from paperlens.llm.client import LLMClient, Message

REPAIR_ATTEMPTS = 1
REPAIR_PROMPT = (
    "Your previous reply was not valid JSON for the required schema ({error}). "
    "Reply again with only the JSON object."
)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


class StructuredOutputError(ValueError):
    """The model did not produce schema-valid JSON, even after a repair round."""


def strip_reasoning(text: str) -> str:
    return _THINK.sub("", text).strip()


def extract_json(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _add(a: Usage, b: Usage) -> Usage:
    return Usage(
        prompt_tokens=a.prompt_tokens + b.prompt_tokens,
        completion_tokens=a.completion_tokens + b.completion_tokens,
        latency_ms=a.latency_ms + b.latency_ms,
    )


def structured_call[T: BaseModel](
    llm: LLMClient, messages: Sequence[Message], schema: type[T]
) -> tuple[T, Usage]:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
    }
    conversation = list(messages)
    total = Usage()
    last_error = ""
    for _ in range(1 + REPAIR_ATTEMPTS):
        reply = llm.complete(conversation, response_format=response_format)
        total = _add(total, reply.usage)
        try:
            return schema.model_validate_json(extract_json(strip_reasoning(reply.content))), total
        except ValidationError as err:
            last_error = str(err).splitlines()[0]
            conversation = [
                *conversation,
                {"role": "assistant", "content": reply.content},
                {"role": "user", "content": REPAIR_PROMPT.format(error=last_error)},
            ]
    raise StructuredOutputError(last_error)
