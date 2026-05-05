"""
Append-only JSONL audit trail for outbound LLM calls (Groq-compatible API).
Inspired by reproducibility/governance patterns; does not store API keys or raw retrieval text in full unless passed as prompt_material.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = _BACKEND_ROOT / "logs"
DEFAULT_LOG_PATH = LOG_DIR / "llm_calls.jsonl"


def _prompt_hash(prompt_material: str) -> str:
    return hashlib.sha256(prompt_material.encode("utf-8", errors="replace")).hexdigest()


def log_llm_call(
    *,
    stage: str,
    provider: str,
    model: str,
    prompt_material: str,
    input_artifacts: Optional[Sequence[str]] = None,
    output_artifact: Optional[str] = None,
    log_path: Optional[Path] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
    latency_ms: Optional[int] = None,
    route: Optional[str] = None,
    run_id: Optional[str] = None,
    status: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """
    Append one NDJSON record. Failures are swallowed so audit never breaks user flows.

    Required fields match historical schema: stage, timestamp, provider, model,
    prompt_hash, input_artifacts, output_artifact.

    Optional fields are omitted from the JSON object when not provided (except numeric
    zeros are still written when explicitly passed).
    """
    path = log_path or DEFAULT_LOG_PATH
    record: dict[str, Any] = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "prompt_hash": _prompt_hash(prompt_material),
        "input_artifacts": list(input_artifacts) if input_artifacts else [],
        "output_artifact": output_artifact or "",
    }
    optional_keys = (
        ("input_tokens", input_tokens),
        ("output_tokens", output_tokens),
        ("total_tokens", total_tokens),
        ("latency_ms", latency_ms),
        ("route", route),
        ("run_id", run_id),
        ("status", status),
        ("error_message", error_message),
    )
    for key, val in optional_keys:
        if val is not None:
            record[key] = val

    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
