"""OpenAI-compatible chat client with timeouts, retries and token accounting.

Purpose: Every model call goes through here, whether the server is Ollama (now) or vLLM
serving the Project 2 fine-tune (later); only llm.base_url and llm.model change.
Flow: messages -> OpenAICompatClient.complete() -> LLMResponse(content, usage)
-> llm/structured.py -> pipelines.
Params: llm.base_url, llm.api_key, llm.model, llm.temperature, llm.max_tokens,
llm.timeout_s, llm.max_retries, llm.retry_backoff_s, llm.extra_body.
Try: Set llm.timeout_s very low (e.g. 2) and watch error_rate in the report: that is how
an overloaded server shows up in metrics rather than as a crash.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, Protocol

import httpx
import openai
from pydantic import BaseModel
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from paperlens.config.models import LLMConfig
from paperlens.domain import Usage

Message = dict[str, str]

# Errors worth retrying: the server was unreachable, slow or failed. 4xx (bad request)
# is not retried because repeating the same request cannot fix it.
_RETRYABLE = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.RateLimitError,
)


class LLMError(RuntimeError):
    """The model call failed after all retries."""


class LLMResponse(BaseModel):
    content: str
    usage: Usage


class LLMClient(Protocol):
    def complete(
        self, messages: Sequence[Message], response_format: dict[str, Any] | None = None
    ) -> LLMResponse: ...


class OpenAICompatClient:
    def __init__(self, cfg: LLMConfig, http_client: httpx.Client | None = None) -> None:
        self._cfg = cfg
        self._client = openai.OpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout_s,
            max_retries=0,  # retries are ours (tenacity), so they are visible and configurable
            http_client=http_client,
        )

    def complete(
        self, messages: Sequence[Message], response_format: dict[str, Any] | None = None
    ) -> LLMResponse:
        cfg = self._cfg
        kwargs: dict[str, Any] = {
            "model": cfg.model,
            "messages": list(messages),
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format
        if cfg.extra_body:
            kwargs["extra_body"] = cfg.extra_body
        retrying = Retrying(
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential(multiplier=cfg.retry_backoff_s),
            retry=retry_if_exception_type(_RETRYABLE),
            reraise=True,
        )
        started = time.perf_counter()
        try:
            response: Any = retrying(lambda: self._client.chat.completions.create(**kwargs))
        except openai.OpenAIError as err:
            raise LLMError(f"{type(err).__name__}: {err}") from err
        latency_ms = (time.perf_counter() - started) * 1000
        usage = response.usage
        return LLMResponse(
            content=response.choices[0].message.content or "",
            usage=Usage(
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                latency_ms=latency_ms,
            ),
        )
