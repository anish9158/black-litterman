"""
Append-only JSONL audit trail for outbound LLM calls (Groq-compatible API).
Inspired by reproducibility/governance patterns; does not store API keys or raw retrieval text in full unless passed as prompt_material.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

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
) -> None:
    """
    Append one NDJSON record. Failures are swallowed so audit never breaks user flows.
    """
    path = log_path or DEFAULT_LOG_PATH
    record = {
        "stage": stage,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "prompt_hash": _prompt_hash(prompt_material),
        "input_artifacts": list(input_artifacts) if input_artifacts else [],
        "output_artifact": output_artifact or "",
    }
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass
