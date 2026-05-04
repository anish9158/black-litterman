"""
RAG chatbot with conversation history, richer document ingestion, and improved prompt.
Faithfully converted from notebook cells 13 and 14.
"""
import json
import os
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import load_settings

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
    "sensitivity", "multiplier", "view", "equilibrium",
    "momentum", "rsi", "sma", "sma20", "sma50", "moving average",
    "pe ratio", "price-to-book", "fundamental",
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


def _is_off_topic(question: str) -> bool:
    """
    Lightweight topic guard — returns True if the question has no overlap
    with allowed portfolio/finance keywords.
    Does NOT call the LLM; purely string-based to keep it fast and offline.
    """
    q_lower = question.lower()
    return not any(kw in q_lower for kw in _ALLOWED_TOPICS)


# System prompt — strict closed-ecosystem guardrails
_SYSTEM_PROMPT = """
You are a closed-ecosystem assistant for the Black-Litterman Portfolio Optimiser project.

STRICT RULES — follow every one of them without exception:
1. Answer ONLY from the retrieved context provided below. Do not use any external knowledge, training data, or information from the internet.
2. ONLY answer questions about: portfolio optimisation, Black-Litterman model, XGBoost views, backtest results, Fama-French factors, SHAP feature importances, sensitivity analysis, NIFTY50 stocks, and the RAG chatbot itself.
3. If the question is unrelated to this project (e.g. weather, sports, cooking, current events, general knowledge), reply with exactly: "I can only answer questions about this Black-Litterman portfolio project."
4. If the answer is not in the retrieved context, say: "This information is not in my knowledge base."
5. Never speculate, hallucinate, or provide information not supported by the retrieved context.
6. Be concise and analytical. Do NOT include a "Sources:" section in your answer — sources are displayed separately in the UI.
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


def get_vectorstore(force_rebuild: bool = False):
    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_file = RAG_INDEX_DIR / "index.faiss"
    if index_file.exists() and not force_rebuild:
        return FAISS.load_local(
            str(RAG_INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
    docs = load_documents()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150
    ).split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(RAG_INDEX_DIR))
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
    """Convert (Document, score) pairs into rich source dicts for the frontend."""
    out = []
    for doc, score in docs_with_scores:
        preview = doc.page_content[:220].strip().replace("\n", " ")
        out.append({
            "source": doc.metadata.get("source", "unknown"),
            "category": doc.metadata.get("category", "notebook"),
            "preview": preview,
            "char_count": len(doc.page_content),
            "score": round(float(score), 4),
        })
    return out


def _generate_follow_ups(answer: str, question: str, api_key: str) -> List[str]:
    """Ask the LLM to generate 3 follow-up questions based on the answer."""
    if not (api_key and ChatOpenAI):
        return []
    try:
        llm = ChatOpenAI(
            model=settings.groq_model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0.4,
        )
        prompt = (
            "Based on this Q&A about a Black-Litterman portfolio optimiser, "
            "generate exactly 3 short follow-up questions a user might ask next. "
            "Return ONLY a JSON array of 3 strings, no other text.\n\n"
            f"Q: {question}\nA: {answer[:500]}"
        )
        raw = llm.invoke([("human", prompt)]).content.strip()
        # Extract JSON array robustly
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start >= 0 and end > start:
            import json as _j
            return _j.loads(raw[start:end])[:3]
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
    import time
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
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
        )
        msg = llm.invoke(
            prompt.format_messages(question=question, context=context, history=history_text)
        )
        answer = msg.content
        # Extract token usage from response metadata (Groq returns this)
        usage = getattr(msg, "response_metadata", {}).get("token_usage", {})
        if usage:
            token_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
    else:
        answer = _fallback_answer(question, docs)
    latency_ms = round((time.time() - t0) * 1000)

    follow_ups = _generate_follow_ups(answer, question, api_key or "") if api_key else []
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
            base_url="https://api.groq.com/openai/v1",
            temperature=0.3,
        )
        return llm.invoke([("human", prompt)]).content
    except Exception:
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
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
        streaming=True,
    )

    full_answer = ""
    t0 = _time.time()
    try:
        for chunk in llm.stream(prompt_msgs):
            token = chunk.content
            if token:
                full_answer += token
                yield f"data: {_json.dumps({'token': token})}\n\n"
    except Exception as exc:
        yield f"data: {_json.dumps({'token': f'[Error: {exc}]'})}\n\n"

    latency_ms = round((_time.time() - t0) * 1000)
    # Estimate token counts from character count (≈4 chars per token)
    prompt_est = len(context) // 4
    completion_est = len(full_answer) // 4
    token_usage = {
        "prompt_tokens": prompt_est,
        "completion_tokens": completion_est,
        "total_tokens": prompt_est + completion_est,
    }
    follow_ups = _generate_follow_ups(full_answer, question, api_key)
    updated = history + [{"user": question, "assistant": full_answer}]
    yield f"data: {_json.dumps({'done': True, 'sources': rich_sources, 'history': updated, 'token_usage': token_usage, 'latency_ms': latency_ms, 'follow_up_questions': follow_ups})}\n\n"
    yield "data: [DONE]\n\n"
