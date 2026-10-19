"""Record what a run actually used beyond the config: prompt text, data, code and model.

Purpose: The config stores the prompt's *path*, not its text; editing the prompt and
re-running would otherwise look like "identical parameters". These meta.* params make a
run reproducible from its own folder and make prompt/model changes visible in reports.
Flow: AppConfig -> collect_meta() -> merged into params.json (keys 'meta.*') ->
evals/report.py diffs them (except git keys, which change on every commit).
Params: generation.prompt_template, data.split / data.data_dir, llm.base_url, llm.model.
Try: Change one word in prompts/baseline_answer.md, re-run, and look at the report's
"Changed parameters" table: meta.prompt_sha256 is listed.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import httpx

from paperlens.config.models import AppConfig
from paperlens.ingestion.qasper import raw_path

HASH_CHARS = 16
DIGEST_CHARS = 12
SERVER_TIMEOUT_S = 2.0
NOT_QUERIED = "not queried"
UNKNOWN = "unknown"


def _sha(path: Path) -> str:
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:HASH_CHARS]


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def _server_info(cfg: AppConfig) -> tuple[str, str]:
    """(model digest, server version) from an Ollama server; 'unknown' for anything else."""
    base = cfg.llm.base_url.rstrip("/").removesuffix("/v1")
    try:
        tags = httpx.get(f"{base}/api/tags", timeout=SERVER_TIMEOUT_S).json()
        version = httpx.get(f"{base}/api/version", timeout=SERVER_TIMEOUT_S).json()
    except (httpx.HTTPError, ValueError):
        return UNKNOWN, UNKNOWN
    digest = next(
        (m.get("digest", "") for m in tags.get("models", []) if m.get("name") == cfg.llm.model),
        "",
    )
    return (digest[:DIGEST_CHARS] or UNKNOWN), str(version.get("version", UNKNOWN))


def collect_meta(cfg: AppConfig, query_server: bool = True) -> dict[str, str]:
    sha = _git("rev-parse", "--short", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    digest, version = _server_info(cfg) if query_server else (NOT_QUERIED, NOT_QUERIED)
    return {
        "meta.prompt_sha256": _sha(cfg.generation.prompt_template),
        "meta.data_sha256": _sha(raw_path(cfg.data, cfg.data.split)),
        "meta.git_sha": sha or UNKNOWN,
        "meta.git_dirty": UNKNOWN if status is None else str(bool(status)),
        "meta.llm_digest": digest,
        "meta.server_version": version,
    }
