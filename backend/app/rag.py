"""
RAG chatbot with conversation history, richer document ingestion, and improved prompt.
Faithfully converted from notebook cells 13 and 14.
"""
import json
import os
import re
import textwrap
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import load_settings
from .llm_audit import log_llm_call

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatOpenAI = None  # type: ignore[assignment,misc]
    ChatPromptTemplate = None  # type: ignore[assignment]

BASE_DIR = Path(__file__).resolve().parents[1]
RAG_DOCS_DIR = BASE_DIR / "rag_docs"
RAG_INDEX_DIR = BASE_DIR / "rag_index"

# ---------------------------------------------------------------------------
# Allowed topic keywords — used for lightweight off-topic detection
# ---------------------------------------------------------------------------
_ALLOWED_TOPICS = {
    # Model & methodology
    "black-litterman", "black litterman", "blacklitterman", "bl model",
    "posterior", "prior", "covariance", "shrinkage", "tau", "cvxpy",
    "xgboost", "gradient boost", "backtest", "rolling window", "walk-forward",
    "fama", "french", "smb", "hml", "rmw", "cma", "fama-french",
    "shap", "feature importance", "explainability",
    # Portfolio / finance terms
    "portfolio", "allocation", "weight", "rebalance",
    "sharpe", "drawdown", "volatility", "annual return", "cumulative return",
    "benchmark", "outperform", "alpha", "beta", "risk-adjusted",
    "optimis", "optimiz", "convex",
    "sensitivity", "multiplier", "view", "views", "equilibrium",
    "omega", "pick matrix", "long-only",
    "momentum", "rsi", "sma", "sma20", "sma50", "moving average",
    "pe ratio", "price-to-book", "fundamental",
    "markowitz", "mean-variance", "ledoit", "shrink",
    "multifactor",
    # Indices & stocks
    "nifty", "nse", "bse", "sensex", "nifty50",
    "reliance", "tcs", "infosys", "infy", "hdfc", "hdfcbank",
    "icicibank", "icici", "sbin", "bajaj", "axisbank", "axis bank",
    "wipro", "techm", "tech mahindra", "sunpharma", "maruti", "divislab",
    "cipla", "kotakbank", "kotak", "bhartiartl", "hindunilvr",
    # App features
    "rag", "retrieval", "chatbot", "embedding", "faiss", "vectorstore",
    "groq", "langchain", "sentiment", "report", "pdf",
    "backtest result", "performance metric",
}

_OFF_TOPIC_REPLY = (
    "I can only answer questions about this Black-Litterman portfolio project — "
    "topics such as the model methodology, backtest results, portfolio allocation, "
    "XGBoost views, SHAP feature importances, Fama-French factors, sensitivity analysis, "
    "and the RAG chatbot itself.\n\n"
    "Your question appears to be outside that scope. Please ask something related to "
    "the portfolio optimiser."
)

# Diverse anchors so suggestion mining does not depend on a single query embedding.
_SUGGESTION_ANCHOR_QUERIES = [
    "Black-Litterman posterior optimisation cvxpy portfolio",
    "XGBoost views SHAP feature importance",
    "rolling window backtest Sharpe cumulative returns NIFTY",
    "Fama French five factor SMB HML",
    "sensitivity analysis view multiplier weights",
    "covariance shrinkage tau equilibrium",
]

# Offline-safe starters aligned with NOTEBOOK_OVERVIEW / app docs (no live metrics implied).
_FALLBACK_GROUNDED_STARTER_QUESTIONS = [
    "What steps does the documented Black-Litterman workflow include?",
    "How does the project describe training XGBoost for portfolio views?",
    "What does the documentation say about rolling-window backtests and sensitivity analysis?",
    "What inputs does the workflow derive from market data (e.g. technicals, fundamentals)?",
]

_STOP_WORDS_SUGGESTIONS = frozenset({
    "what", "does", "this", "that", "from", "with", "have", "been", "were", "how", "why",
    "when", "where", "which", "about", "into", "than", "then", "these", "those", "such",
    "some", "each", "other", "there", "their", "they", "them", "will", "would", "could",
    "should", "might", "also", "more", "most", "very", "just", "like", "the", "and", "for",
    "are", "but", "not", "you", "all", "can", "was", "one", "our", "out", "get", "use",
    "any", "may", "way", "who", "its", "now", "did", "say", "says", "being", "here",
})


def _is_off_topic(question: str) -> bool:
    """
    Lightweight topic guard — returns True if the question has no overlap
    with allowed portfolio/finance keywords.
    Does NOT call the LLM; purely string-based to keep it fast and offline.
    """
    q_lower = question.lower()
    return not any(kw in q_lower for kw in _ALLOWED_TOPICS)


def _question_terms_overlap_context(question: str, context: str) -> bool:
    """True if at least one substantive term from the question appears in context (cheap grounding check)."""
    ctx = context.lower()
    words = re.findall(r"[a-z][a-z0-9]{2,}", question.lower())
    for w in words:
        if w in _STOP_WORDS_SUGGESTIONS:
            continue
        if w in ctx:
            return True
    return False


def _filter_grounded_question_candidates(
    candidates: List[str],
    context: str,
    max_n: int = 6,
) -> List[str]:
    """Keep suggestions that pass the topic guard and overlap retrieved context."""
    out: List[str] = []
    for q in candidates:
        q = (q or "").strip()
        if len(q) < 10:
            continue
        if _is_off_topic(q):
            continue
        if not _question_terms_overlap_context(q, context):
            continue
        if q not in out:
            out.append(q)
        if len(out) >= max_n:
            break
    return out


def _gather_chunks_for_suggestions(vstore: FAISS, per_anchor: int = 2) -> List[Document]:
    seen: set[tuple] = set()
    out: List[Document] = []
    for anchor in _SUGGESTION_ANCHOR_QUERIES:
        for doc, _ in vstore.similarity_search_with_score(anchor, k=per_anchor):
            key = (doc.metadata.get("source"), doc.page_content[:120])
            if key in seen:
                continue
            seen.add(key)
            out.append(doc)
    return out


def _format_suggestion_context(docs: List[Document], max_chars: int = 7200) -> str:
    blocks: List[str] = []
    n = 0
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata.get("source", "unknown")
        block = f"[Excerpt {i} | {meta}]\n{doc.page_content.strip()}\n\n"
        if n + len(block) > max_chars:
            break
        blocks.append(block)
        n += len(block)
    return "".join(blocks)


def get_grounded_rag_suggestions(n: int = 6) -> List[str]:
    """
    Starter questions mined from current index: diverse retrieval + LLM (if configured),
    filtered so each suggestion is plausibly on-topic and grounded in retrieved excerpts.
    """
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore(force_rebuild=False)

    docs = _gather_chunks_for_suggestions(_vectorstore)
    context = _format_suggestion_context(docs)
    if not context.strip():
        return _FALLBACK_GROUNDED_STARTER_QUESTIONS[:n]

    api_key = os.getenv("GROQ_TOKEN") or settings.groq_api_key
    fallback_filtered = _filter_grounded_question_candidates(
        _FALLBACK_GROUNDED_STARTER_QUESTIONS, context, max_n=n
    )

    if not (api_key and ChatOpenAI):
        return fallback_filtered or _FALLBACK_GROUNDED_STARTER_QUESTIONS[:n]

    try:
        llm = ChatOpenAI(
            model=settings.groq_model,
            api_key=api_key,
            base_url=settings.llm_base_url,
            temperature=0.25,
        )
        prompt = (
            f"You propose starter questions for a documentation-only RAG chatbot. "
            f"Output at most {n} questions. Each question MUST be fully answerable using "
            "ONLY the EXCERPTS below — not general knowledge. "
            "Do not invent ticker rankings, live performance, or specific numeric backtest results "
            "unless the EXCERPTS explicitly state them. "
            "Prefer methodology, workflow, inputs, and outputs the text actually describes. "
            f"Return ONLY a JSON array of up to {n} strings, no markdown.\n\n"
            f"EXCERPTS:\n{context}"
        )
        t0 = time.perf_counter()
        sug_msg = llm.invoke([("human", prompt)])
        raw = sug_msg.content.strip()
        sug_ms = int(round((time.perf_counter() - t0) * 1000))
        sug_usage = getattr(sug_msg, "response_metadata", {}).get("token_usage", {}) or {}
        log_llm_call(
            stage="rag_starter_suggestions",
            provider="groq",
            model=settings.groq_model,
            prompt_material=prompt,
            input_artifacts=["backend/rag_docs/"],
            output_artifact="",
            route="/rag-suggested-prompts",
            latency_ms=sug_ms,
            input_tokens=sug_usage.get("prompt_tokens"),
            output_tokens=sug_usage.get("completion_tokens"),
            total_tokens=sug_usage.get("total_tokens"),
            status="ok",
        )
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
            if isinstance(parsed, list):
                filtered = _filter_grounded_question_candidates(
                    [str(x) for x in parsed], context, max_n=n
                )
                if filtered:
                    return filtered
    except Exception:
        pass

    return fallback_filtered or _FALLBACK_GROUNDED_STARTER_QUESTIONS[:n]


# System prompt — strict closed-ecosystem guardrails
_SYSTEM_PROMPT = """
You are a closed-ecosystem assistant for the Black-Litterman Portfolio Optimiser project.

STRICT RULES — follow every one of them without exception:
1. Answer ONLY from the retrieved context provided below. Do not use any external knowledge, training data, or information from the internet.
2. ONLY answer questions about: portfolio optimisation, Black-Litterman model, XGBoost views, backtest results, Fama-French factors, SHAP feature importances, sensitivity analysis, NIFTY50 stocks, and the RAG chatbot itself.
3. If the question is unrelated to this project (e.g. weather, sports, cooking, current events, general knowledge), reply with exactly: "I can only answer questions about this Black-Litterman portfolio project."
4. If the answer is not in the retrieved context, say: "This information is not in my knowledge base."
5. Never speculate, hallucinate, or provide information not supported by the retrieved context.
6. Be concise and analytical. Do NOT include a "Sources:" section — sources appear separately in the UI.
7. Format answers for Markdown rendering in the frontend: use **bold** for key terms (not asterisks as plain symbols where bold is intended). Use $inline math$ for short formulas or \\(...\\), and $$...$$ for display equations instead of unreadable escapes.
"""

NOTEBOOK_OVERVIEW = textwrap.dedent("""
Notebook overview:
- Builds a Black-Litterman portfolio workflow for NIFTY assets.
- Downloads market data with yfinance.
- Generates technical, fundamental, and feature-engineering inputs (SMA20, SMA50, RSI, momentum, PE, P/B).
- Trains XGBoost models to form views.
- Uses a Black-Litterman posterior update (full two-inverse formula) and cvxpy optimization.
- Runs rolling-window backtests with covariance shrinkage.
- Computes cumulative returns, annualised metrics, and sensitivity analysis.
- Includes a Fama-French 5-factor decomposition (SMB, HML, RMW, CMA).
""").strip()

settings = load_settings()


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------

def _safe_markdown_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df is None or len(df) == 0:
        return "No rows available."
    try:
        return df.head(max_rows).to_markdown(index=True)
    except Exception:
        return df.head(max_rows).to_string()


def read_file(path: Path) -> List[Document]:
    """
    Parse a file into Documents. Supports .ipynb, .txt, .md, .py,
    .json, .csv, .xlsx/.xls, .pdf from notebook cell 13.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".ipynb":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            docs = []
            for i, c in enumerate(data.get("cells", [])):
                src = "".join(c.get("source", []))
                if src.strip():
                    docs.append(Document(
                        page_content=src,
                        metadata={"source": path.name, "cell": i, "type": c.get("cell_type")},
                    ))
            return docs

        if suffix in {".txt", ".md", ".py", ".html", ".htm"}:
            return [Document(
                page_content=path.read_text(encoding="utf-8", errors="ignore"),
                metadata={"source": path.name},
            )]

        if suffix == ".json":
            raw = path.read_text(encoding="utf-8", errors="ignore")
            return [Document(page_content=raw[:8000], metadata={"source": path.name})]

        if suffix == ".csv":
            df = pd.read_csv(path)
            text = f"CSV file: {path.name}\n\n" + _safe_markdown_table(df)
            return [Document(page_content=text, metadata={"source": path.name})]

        if suffix in {".xlsx", ".xls"}:
            import openpyxl  # noqa: F401  ensure installed
            xls = pd.ExcelFile(path)
            out = []
            for sheet in xls.sheet_names:
                df = pd.read_excel(path, sheet_name=sheet)
                text = f"Excel file: {path.name} | Sheet: {sheet}\n\n" + _safe_markdown_table(df)
                out.append(Document(page_content=text, metadata={"source": path.name, "sheet": sheet}))
            return out

        if suffix == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            pages = []
            for i, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                if txt.strip():
                    pages.append(Document(
                        page_content=txt,
                        metadata={"source": path.name, "page": i + 1},
                    ))
            return pages

    except Exception as exc:
        return [Document(
            page_content=f"Could not parse {path.name}: {exc}",
            metadata={"source": path.name, "error": True},
        )]
    return []


def load_documents() -> List[Document]:
    """
    Load documents from rag_docs/ with automatic category tagging.

    Subdirectory structure is used to set a 'category' metadata field:
      rag_docs/                    → category=notebook
      rag_docs/market_reports/     → category=market_report
      rag_docs/company_filings/    → category=company_filing
      rag_docs/sector_research/    → category=sector_research
      (any other subdir name)      → category=<subdir name>
    """
    RAG_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    docs = [Document(
        page_content=NOTEBOOK_OVERVIEW,
        metadata={"source": "app_overview", "category": "notebook"},
    )]

    for path in sorted(RAG_DOCS_DIR.rglob("*")):
        if not path.is_file():
            continue
        # Determine category from parent directory relative to rag_docs/
        rel = path.relative_to(RAG_DOCS_DIR)
        category = rel.parts[0] if len(rel.parts) > 1 else "notebook"

        file_docs = read_file(path)
        for doc in file_docs:
            doc.metadata.setdefault("category", category)
        docs.extend(file_docs)

    return docs


RAG_EXTENSIONS = frozenset({
    ".md", ".txt", ".py", ".ipynb", ".pdf", ".json", ".csv", ".xlsx", ".xls", ".htm", ".html",
})


def _rag_docs_supersede_index(index_path: Path) -> bool:
    """
    True if any known rag_docs file was modified after the saved FAISS index —
    embeddings should be regenerated so retrieval includes new/edited docs.
    """
    if not index_path.exists():
        return True
    try:
        idx_mtime = index_path.stat().st_mtime
    except OSError:
        return True
    if not RAG_DOCS_DIR.exists():
        return False
    for path in sorted(RAG_DOCS_DIR.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix == "":
            continue
        if suffix not in RAG_EXTENSIONS:
            continue
        try:
            if path.stat().st_mtime > idx_mtime:
                return True
        except OSError:
            continue
    return False


def get_vectorstore(force_rebuild: bool = False):
    global _vectorstore

    def _needs_rebuild() -> bool:
        index_file_local = RAG_INDEX_DIR / "index.faiss"
        # Note: intentional edits to rag_docs/ are detected via mtime; to force rebuild when
        # nothing changed on disk delete backend/rag_index/ or restart with code change.
        return force_rebuild or _rag_docs_supersede_index(index_file_local)

    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_file = RAG_INDEX_DIR / "index.faiss"
    stale = _needs_rebuild()

    if _vectorstore is not None and not stale:
        return _vectorstore

    _vectorstore = None  # invalidate cache whenever we rebuild disk index

    if index_file.exists() and not stale:
        _vectorstore = FAISS.load_local(
            str(RAG_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        return _vectorstore

    docs = load_documents()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150
    ).split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(RAG_INDEX_DIR))
    _vectorstore = store
    return store


_vectorstore = None


# ---------------------------------------------------------------------------
# Conversation helpers (notebook cell 14)
# ---------------------------------------------------------------------------

def _format_docs(docs: List[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = ", ".join(f"{k}={v}" for k, v in doc.metadata.items())
        blocks.append(f"[Chunk {i} | {meta}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _history_to_text(history: List[Dict[str, str]], max_turns: int = 4) -> str:
    recent = history[-max_turns:]
    if not recent:
        return "No prior conversation."
    lines = []
    for turn in recent:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['assistant']}")
    return "\n".join(lines)


def _fallback_answer(question: str, docs: List[Document]) -> str:
    header = "No GROQ_API_KEY detected — retrieval-only response.\n\n"
    bullets = []
    for i, doc in enumerate(docs[:3], start=1):
        snippet = doc.page_content[:700].strip().replace("\n\n", "\n")
        bullets.append(f"Source {i} ({doc.metadata}):\n{snippet}")
    return header + "\n\n".join(bullets)


def _build_rich_sources(docs_with_scores: List[tuple]) -> List[Dict[str, Any]]:
    """
    Convert (Document, distance) pairs from FAISS similarity_search_with_score.

    LangChain's default IndexFlatL2 returns squared L2 distance in embedding space —
    smaller values mean vectors are closer → better semantic match to the query.
    """
    out = []
    for doc, raw in docs_with_scores:
        preview = doc.page_content[:220].strip().replace("\n", " ")
        d = float(raw)
        out.append({
            "source": doc.metadata.get("source", "unknown"),
            "category": doc.metadata.get("category", "notebook"),
            "preview": preview,
            "char_count": len(doc.page_content),
            "score": round(d, 4),
            "l2_distance": round(d, 4),
            "metric": "faiss_embedding_distance",
            "interpretation": "FAISS L2 embedding distance returned by LangChain (lower = closer to the query embedding).",
        })
    return out


def _generate_follow_ups(
    answer: str,
    question: str,
    api_key: str,
    retrieved_context: str,
    audit_route: Optional[str] = None,
) -> List[str]:
    """LLM follow-ups that must be answerable from the same retrieved context; post-filtered for grounding."""
    if not (api_key and ChatOpenAI):
        return []
    ctx = (retrieved_context or "")[:6500]
    if not ctx.strip():
        return []
    try:
        llm = ChatOpenAI(
            model=settings.groq_model,
            api_key=api_key,
            base_url=settings.llm_base_url,
            temperature=0.25,
        )
        prompt = (
            "You propose follow-up questions for a STRICT retrieval-grounded chatbot. "
            "The next turn will retrieve fresh chunks, but each follow-up you propose MUST be answerable "
            "using ONLY the RETRIEVED CONTEXT below (same corpus excerpt the current answer used).\n\n"
            "Rules:\n"
            "1. Output exactly 3 short follow-up questions.\n"
            "2. Each must be answerable from the RETRIEVED CONTEXT alone — not general knowledge.\n"
            "3. Do not ask about specific tickers, metrics, or rankings unless they appear in the RETRIEVED CONTEXT.\n"
            "4. Re-use terminology from the RETRIEVED CONTEXT when possible.\n"
            "5. Return ONLY a JSON array of 3 strings.\n\n"
            f"Original user question: {question}\n\n"
            f"Assistant answer (for phrasing only; ground questions in CONTEXT): {answer[:650]}\n\n"
            f"RETRIEVED CONTEXT:\n{ctx}"
        )
        t0 = time.perf_counter()
        fu_msg = llm.invoke([("human", prompt)])
        raw = fu_msg.content.strip()
        fu_ms = int(round((time.perf_counter() - t0) * 1000))
        fu_usage = getattr(fu_msg, "response_metadata", {}).get("token_usage", {}) or {}
        log_llm_call(
            stage="rag_follow_up_suggestions",
            provider="groq",
            model=settings.groq_model,
            prompt_material=prompt,
            input_artifacts=["backend/rag_docs/"],
            output_artifact="",
            route=audit_route,
            latency_ms=fu_ms,
            input_tokens=fu_usage.get("prompt_tokens"),
            output_tokens=fu_usage.get("completion_tokens"),
            total_tokens=fu_usage.get("total_tokens"),
            status="ok",
        )
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw[start:end])
            if isinstance(parsed, list):
                return _filter_grounded_question_candidates(
                    [str(x) for x in parsed], retrieved_context, max_n=3
                )
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Public ask_rag function
# ---------------------------------------------------------------------------

def ask_rag(
    question: str,
    k: int = 5,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Retrieve context, call the LLM, and return a rich response including:
    answer, sources (with scores), token_usage, latency_ms, follow_up_questions.
    """
    history = history or []

    # Fast off-topic guard — no LLM call needed
    if _is_off_topic(question):
        updated = history + [{"user": question, "assistant": _OFF_TOPIC_REPLY}]
        return {
            "answer": _OFF_TOPIC_REPLY,
            "sources": [],
            "history": updated,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "latency_ms": 0,
            "follow_up_questions": [],
        }

    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore(force_rebuild=False)

    # Use similarity_search_with_score to get relevance scores
    docs_with_scores = _vectorstore.similarity_search_with_score(question, k=k)
    docs = [d for d, _ in docs_with_scores]
    rich_sources = _build_rich_sources(docs_with_scores)
    context = _format_docs(docs)
    history_text = _history_to_text(history)

    api_key = os.getenv("GROQ_TOKEN") or settings.groq_api_key
    token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    t0 = time.time()
    if api_key and ChatOpenAI and ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT.strip()),
            ("human", "Question: {question}\n\nConversation history:\n{history}\n\nRetrieved context:\n{context}"),
        ])
        llm = ChatOpenAI(
            model=settings.groq_model,
            api_key=api_key,
            base_url=settings.llm_base_url,
            temperature=0,
        )
        msg = llm.invoke(
            prompt.format_messages(question=question, context=context, history=history_text)
        )
        answer = msg.content
        audit_text = (
            _SYSTEM_PROMPT.strip()
            + "\nQuestion: "
            + question
            + "\nConversation history:\n"
            + history_text
            + "\nRetrieved context:\n"
            + context
        )
        # Extract token usage from response metadata (Groq returns this)
        usage = getattr(msg, "response_metadata", {}).get("token_usage", {}) or {}
        if usage:
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
    else:
        answer = _fallback_answer(question, docs)
    latency_ms = round((time.time() - t0) * 1000)

    if api_key and ChatOpenAI and ChatPromptTemplate and answer:
        log_llm_call(
            stage="rag_chat",
            provider="groq",
            model=settings.groq_model,
            prompt_material=audit_text,
            input_artifacts=["backend/rag_docs/"],
            output_artifact="",
            route="/ask-rag",
            latency_ms=int(latency_ms),
            input_tokens=usage.get("prompt_tokens") if usage else None,
            output_tokens=usage.get("completion_tokens") if usage else None,
            total_tokens=usage.get("total_tokens") if usage else None,
            status="ok",
        )

    follow_ups = (
        _generate_follow_ups(answer, question, api_key or "", context, audit_route="/ask-rag")
        if api_key
        else []
    )
    updated_history = history + [{"user": question, "assistant": answer}]
    return {
        "answer": answer,
        "sources": rich_sources,
        "history": updated_history,
        "token_usage": token_usage,
        "latency_ms": latency_ms,
        "follow_up_questions": follow_ups,
    }


# ---------------------------------------------------------------------------
# Knowledge Base listing
# ---------------------------------------------------------------------------

def get_knowledge_base_chunks() -> List[Dict[str, Any]]:
    """
    Return metadata for all chunks currently indexed in the vector store.
    Used by the /knowledge-base endpoint.
    """
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    result = []
    for i, chunk in enumerate(chunks):
        preview = chunk.page_content[:200].strip().replace("\n", " ")
        result.append({
            "id": i,
            "source": chunk.metadata.get("source", "unknown"),
            "category": chunk.metadata.get("category", "notebook"),
            "preview": preview,
            "char_count": len(chunk.page_content),
        })
    return result


# ---------------------------------------------------------------------------
# LLM Portfolio Narrative
# ---------------------------------------------------------------------------

def generate_narrative(
    allocation: List[Dict[str, Any]],
    metrics: Dict[str, float],
    tickers: List[str],
    audit_route: Optional[str] = None,
) -> str:
    """
    Use the Groq LLM to produce a concise, human-readable portfolio explanation.
    Returns empty string if no API key is available or if the call fails.
    """
    api_key = os.getenv("GROQ_TOKEN") or settings.groq_api_key
    if not (api_key and ChatOpenAI):
        return ""

    top_holdings = sorted(allocation, key=lambda x: -x.get("weight", 0))[:5]
    alloc_str = ", ".join(
        f"{a['ticker']} ({a.get('weight', 0):.1%})" for a in top_holdings
    )
    benchmark_return = metrics.get("benchmark_annual_return", None)
    port_return = metrics.get("annual_return", None)
    sharpe = metrics.get("sharpe_ratio", None)
    max_dd = metrics.get("max_drawdown", None)

    metric_lines = []
    if port_return is not None:
        metric_lines.append(f"Annual return: {port_return:+.2%}")
    if benchmark_return is not None:
        outperf = (port_return or 0) - benchmark_return
        metric_lines.append(f"Benchmark (NIFTY 50): {benchmark_return:+.2%} (portfolio outperforms by {outperf:+.2%})")
    if sharpe is not None:
        metric_lines.append(f"Sharpe ratio: {sharpe:.3f}")
    if max_dd is not None:
        metric_lines.append(f"Max drawdown: {max_dd:.2%}")

    prompt = (
        "You are a concise quantitative portfolio analyst. In 3-4 sentences explain this "
        "Black-Litterman optimised portfolio to a non-technical investor. Be specific with numbers.\n\n"
        f"Top holdings: {alloc_str}\n"
        + "\n".join(metric_lines)
        + "\n\nExplain: (1) what drove the allocation, (2) what the return and risk numbers mean, "
        "(3) whether it beat the NIFTY 50 benchmark. Keep it under 120 words. No bullet points."
    )

    try:
        llm = ChatOpenAI(
            model=settings.groq_model,
            api_key=api_key,
            base_url=settings.llm_base_url,
            temperature=0.3,
        )
        t0 = time.perf_counter()
        narr_msg = llm.invoke([("human", prompt)])
        content = narr_msg.content
        narr_ms = int(round((time.perf_counter() - t0) * 1000))
        narr_usage = getattr(narr_msg, "response_metadata", {}).get("token_usage", {}) or {}
        log_llm_call(
            stage="portfolio_narrative",
            provider="groq",
            model=settings.groq_model,
            prompt_material=prompt,
            input_artifacts=[],
            output_artifact="",
            route=audit_route,
            latency_ms=narr_ms,
            input_tokens=narr_usage.get("prompt_tokens"),
            output_tokens=narr_usage.get("completion_tokens"),
            total_tokens=narr_usage.get("total_tokens"),
            status="ok",
        )
        return content
    except Exception as exc:
        log_llm_call(
            stage="portfolio_narrative",
            provider="groq",
            model=settings.groq_model,
            prompt_material=prompt,
            input_artifacts=[],
            output_artifact="",
            route=audit_route,
            status="error",
            error_message=str(exc)[:500],
        )
        return ""


# ---------------------------------------------------------------------------
# Streaming RAG (SSE)
# ---------------------------------------------------------------------------

def ask_rag_stream(
    question: str,
    k: int = 5,
    history: Optional[List[Dict[str, str]]] = None,
):
    """
    Generator that yields Server-Sent Events (SSE) for a streaming RAG response.
    Each event is either:
      data: {"token": "..."}\n\n   — partial LLM token
      data: {"done": true, "sources": [...], "history": [...]}\n\n  — final metadata
      data: [DONE]\n\n           — stream terminator
    """
    import json as _json

    history = history or []

    # Fast off-topic guard — no LLM call needed
    if _is_off_topic(question):
        updated = history + [{"user": question, "assistant": _OFF_TOPIC_REPLY}]
        yield f"data: {_json.dumps({'token': _OFF_TOPIC_REPLY})}\n\n"
        yield f"data: {_json.dumps({'done': True, 'sources': [], 'history': updated})}\n\n"
        yield "data: [DONE]\n\n"
        return

    import time as _time
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore(force_rebuild=False)

    docs_with_scores = _vectorstore.similarity_search_with_score(question, k=k)
    docs = [d for d, _ in docs_with_scores]
    rich_sources = _build_rich_sources(docs_with_scores)
    context = _format_docs(docs)
    history_text = _history_to_text(history)

    api_key = os.getenv("GROQ_TOKEN") or settings.groq_api_key

    if not (api_key and ChatOpenAI):
        fallback = _fallback_answer(question, docs)
        yield f"data: {_json.dumps({'token': fallback})}\n\n"
        updated = history + [{"user": question, "assistant": fallback}]
        yield f"data: {_json.dumps({'done': True, 'sources': rich_sources, 'history': updated, 'latency_ms': 0, 'follow_up_questions': []})}\n\n"
        yield "data: [DONE]\n\n"
        return

    prompt_msgs = [
        ("system", _SYSTEM_PROMPT.strip()),
        (
            "human",
            f"Question: {question}\n\nConversation history:\n{history_text}\n\nRetrieved context:\n{context}",
        ),
    ]

    llm = ChatOpenAI(
        model=settings.groq_model,
        api_key=api_key,
        base_url=settings.llm_base_url,
        temperature=0,
        streaming=True,
    )

    full_answer = ""
    t0 = _time.time()
    stream_run_id = str(uuid.uuid4())
    stream_err: Optional[str] = None
    try:
        for chunk in llm.stream(prompt_msgs):
            token = chunk.content
            if token:
                full_answer += token
                yield f"data: {_json.dumps({'token': token})}\n\n"
    except Exception as exc:
        stream_err = str(exc)[:500]
        yield f"data: {_json.dumps({'token': f'[Error: {exc}]'})}\n\n"

    latency_ms = round((_time.time() - t0) * 1000)
    stream_audit_text = (
        _SYSTEM_PROMPT.strip()
        + "\nQuestion: "
        + question
        + "\nConversation history:\n"
        + history_text
        + "\nRetrieved context:\n"
        + context
    )
    # Estimate token counts from character count (≈4 chars per token) — same heuristic as API metadata for stream
    prompt_est = len(stream_audit_text) // 4
    completion_est = len(full_answer) // 4
    token_usage = {
        "prompt_tokens": prompt_est,
        "completion_tokens": completion_est,
        "total_tokens": prompt_est + completion_est,
    }
    log_llm_call(
        stage="rag_chat_stream",
        provider="groq",
        model=settings.groq_model,
        prompt_material=stream_audit_text,
        input_artifacts=["backend/rag_docs/"],
        output_artifact="",
        route="/ask-rag-stream",
        run_id=stream_run_id,
        latency_ms=int(latency_ms),
        input_tokens=prompt_est,
        output_tokens=completion_est,
        total_tokens=prompt_est + completion_est,
        status="error" if stream_err else "ok",
        error_message=stream_err,
    )
    follow_ups = _generate_follow_ups(
        full_answer, question, api_key, context, audit_route="/ask-rag-stream"
    )
    updated = history + [{"user": question, "assistant": full_answer}]
    yield f"data: {_json.dumps({'done': True, 'sources': rich_sources, 'history': updated, 'token_usage': token_usage, 'latency_ms': latency_ms, 'follow_up_questions': follow_ups})}\n\n"
    yield "data: [DONE]\n\n"
