#!/usr/bin/env python3
"""
Validate repository invariants for the Black-Litterman portfolio + RAG app.
Run from repo root: python scripts/validate_portfolio_project.py
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    backend = repo / "backend"
    sys.path.insert(0, str(backend))

    errors: list[str] = []
    warnings: list[str] = []

    # --- Static dashboard payload ---
    try:
        from app.static_results import NOTEBOOK_RESULTS
    except Exception as exc:
        print(f"ERROR: cannot import NOTEBOOK_RESULTS: {exc}", file=sys.stderr)
        return 1

    if not isinstance(NOTEBOOK_RESULTS, dict):
        errors.append("NOTEBOOK_RESULTS must be a dict")

    benchmark = NOTEBOOK_RESULTS.get("benchmark") or {}
    if not benchmark.get("name"):
        errors.append("NOTEBOOK_RESULTS.benchmark.name missing")
    models = NOTEBOOK_RESULTS.get("models") or []
    if not isinstance(models, list) or len(models) < 1:
        errors.append("NOTEBOOK_RESULTS.models must be a non-empty list")
    else:
        for i, m in enumerate(models):
            if not isinstance(m, dict):
                errors.append(f"models[{i}] is not an object")
                continue
            for key in ("name", "annual_return"):
                if key not in m:
                    errors.append(f"models[{i}] missing '{key}'")

    weights = NOTEBOOK_RESULTS.get("weights") or []
    if not isinstance(weights, list):
        errors.append("NOTEBOOK_RESULTS.weights must be a list")
    else:
        for i, w in enumerate(weights[:50]):
            if not isinstance(w, dict) or "ticker" not in w or "weight" not in w:
                errors.append(f"weights[{i}] invalid shape")
                break

    sens = NOTEBOOK_RESULTS.get("sensitivity") or {}
    mult = sens.get("multipliers") or []
    series = sens.get("series") or []
    if not isinstance(mult, list) or len(mult) < 2:
        errors.append("sensitivity.multipliers must be a list with length >= 2")
    if not isinstance(series, list) or len(series) < 1:
        errors.append("sensitivity.series must be non-empty")
    else:
        for i, row in enumerate(series):
            if not isinstance(row, dict) or "ticker" not in row:
                errors.append(f"sensitivity.series[{i}] invalid")
                break
            ww = row.get("weights") or []
            if not isinstance(ww, list) or len(ww) != len(mult):
                errors.append(f"sensitivity.series[{i}].weights length must match multipliers")

    notes = NOTEBOOK_RESULTS.get("notes")
    if notes is None or (isinstance(notes, str) and not notes.strip()):
        warnings.append("NOTEBOOK_RESULTS.notes empty or missing")

    # --- RAG corpus on disk ---
    rag_docs = backend / "rag_docs"
    if not rag_docs.is_dir():
        errors.append(f"Missing directory {rag_docs}")
    else:
        md_files = list(rag_docs.rglob("*.md"))
        nb_files = list(rag_docs.rglob("*.ipynb"))
        if len(md_files) + len(nb_files) < 1:
            warnings.append("No .md/.ipynb under rag_docs (RAG corpus may be empty)")

    # --- FAISS index (optional; rebuilt on demand) ---
    index_faiss = backend / "rag_index" / "index.faiss"
    if not index_faiss.is_file():
        warnings.append(f"No FAISS index at {index_faiss} (will build on first RAG hit)")

    # --- Env example ---
    env_ex = backend / ".env.example"
    if env_ex.is_file():
        txt = env_ex.read_text(encoding="utf-8", errors="ignore")
        if "GROQ_API_KEY" not in txt:
            warnings.append(".env.example should mention GROQ_API_KEY")

    for w in warnings:
        print(f"WARN: {w}", file=sys.stderr)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print("validate: OK — static notebook payload, corpus path, optional index checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
