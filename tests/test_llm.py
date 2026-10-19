import json

import httpx
import pytest
from pydantic import BaseModel

from paperlens.config.models import LLMConfig
from paperlens.llm.client import LLMError, OpenAICompatClient
from paperlens.llm.fake import FakeLLM
from paperlens.llm.structured import (
    StructuredOutputError,
    extract_json,
    strip_reasoning,
    structured_call,
)

MSG = [{"role": "user", "content": "hi"}]


class Out(BaseModel):
    answer: str


def completion(content: str) -> dict[str, object]:
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def test_strip_reasoning_and_fences() -> None:
    raw = '<think>let me think</think>\n```json\n{"answer": "x"}\n```'
    assert extract_json(strip_reasoning(raw)) == '{"answer": "x"}'


def test_structured_happy_path_passes_schema() -> None:
    llm = FakeLLM(['{"answer": "SQuAD"}'])
    out, _ = structured_call(llm, MSG, Out)
    assert out.answer == "SQuAD"
    fmt = llm.calls[0][1]
    assert fmt is not None and fmt["json_schema"]["name"] == "Out"


def test_structured_repairs_once_and_sums_usage() -> None:
    llm = FakeLLM(["not json", '<think>ok</think>{"answer": "y"}'])
    out, usage = structured_call(llm, MSG, Out)
    assert out.answer == "y" and len(llm.calls) == 2
    assert "not valid" in llm.calls[1][0][-1]["content"]
    assert usage.prompt_tokens == 2 * FakeLLM.PROMPT_TOKENS


def test_structured_gives_up_after_repair() -> None:
    with pytest.raises(StructuredOutputError):
        structured_call(FakeLLM(["nope", "still nope"]), MSG, Out)


def test_openai_client_sends_config_and_retries_5xx() -> None:
    bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(500, json={"error": {"message": "boom"}})
        return httpx.Response(200, json=completion('{"answer": "z"}'))

    cfg = LLMConfig(
        model="m", temperature=0.0, max_tokens=64, retry_backoff_s=0.01, extra_body={"think": False}
    )
    http = httpx.Client(transport=httpx.MockTransport(handler))
    resp = OpenAICompatClient(cfg, http_client=http).complete(
        MSG, response_format={"type": "json_object"}
    )
    assert resp.content == '{"answer": "z"}' and resp.usage.prompt_tokens == 3
    assert len(bodies) == 2
    assert bodies[1]["model"] == "m" and bodies[1]["think"] is False
    assert bodies[1]["response_format"] == {"type": "json_object"}


def test_openai_client_raises_llm_error_after_retries() -> None:
    cfg = LLMConfig(max_retries=2, retry_backoff_s=0.01)
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(503)))
    with pytest.raises(LLMError):
        OpenAICompatClient(cfg, http_client=http).complete(MSG)
