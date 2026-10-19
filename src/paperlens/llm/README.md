# llm

| File | Role |
|---|---|
| `client.py` | OpenAI-compatible client (Ollama now, vLLM later): timeout, retries, token/latency |
| `structured.py` | JSON-schema output -> pydantic object; strips `<think>`/fences, one repair round |
| `fake.py` | scripted replies for tests |

Experiments: `llm.model` (biggest lever), `llm.temperature`, `llm.max_tokens`, `llm.extra_body`
(e.g. turning off a model's thinking mode). Check `error_rate` in every report.
