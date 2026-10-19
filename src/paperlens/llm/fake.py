"""Scripted LLM for tests: returns queued replies in order and records every call.

Purpose: Lets pipeline and runner tests run without a GPU or a server, deterministically.
Flow: test -> FakeLLM(replies) -> pipeline -> FakeLLM.calls inspected by the test.
Params: none.
Try: Queue a malformed reply first to exercise the repair path in a test.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from paperlens.domain import Usage
from paperlens.llm.client import LLMResponse, Message


class FakeLLM:
    PROMPT_TOKENS = 10
    COMPLETION_TOKENS = 5
    LATENCY_MS = 1.0

    def __init__(self, replies: list[str | Exception]) -> None:
        self._replies = list(replies)
        self.calls: list[tuple[list[Message], dict[str, Any] | None]] = []

    def complete(
        self, messages: Sequence[Message], response_format: dict[str, Any] | None = None
    ) -> LLMResponse:
        self.calls.append((list(messages), response_format))
        if not self._replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return LLMResponse(
            content=reply,
            usage=Usage(
                prompt_tokens=self.PROMPT_TOKENS,
                completion_tokens=self.COMPLETION_TOKENS,
                latency_ms=self.LATENCY_MS,
            ),
        )
