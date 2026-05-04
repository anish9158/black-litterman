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
    "portfolio", "stock", "ticker", "nifty", "nse", "bse", "india", "indian",
    "black-litterman", "black litterman", "blacklitterman",
    "sharpe", "return", "volatility", "drawdown", "benchmark",
    "xgboost", "model", "backtest", "rolling", "window", "train", "predict",
    "view", "prior", "posterior", "covariance", "shrinkage", "optimis", "optimiz",
    "weight", "allocation", "factor", "fama", "french", "smb", "hml", "rmw", "cma",
    "rag", "chatbot", "retrieval", "embedding", "faiss", "langchain", "groq",
    "sentiment", "news", "headline", "shap", "feature", "importance",
    "reliance", "tcs", "infosys", "hdfc", "icici", "sbin", "bajaj", "axis",
    "infy", "wipro", "techm", "sunpharma", "maruti", "divislab", "cipla",
    "sensitivity", "multiplier", "risk", "aversion", "tau", "cvxpy",
    "annual", "cumulative", "momentum", "rsi", "sma", "pe", "price", "book",
    "how", "why", "what", "explain", "describe", "interpret", "compare",
    "metric", "result", "performance", "outperform", "exceed", "beat",
    "report", "pdf", "analysis", "quantitative", "financial", "finance",
    "equity", "asset", "market", "capital", "invest",
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
6. Be concise and analytical. End every answer with a short "Sources:" section listing the retrieved chunks you used.
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


# ---------------------------------------------------------------------------
# Public ask_rag function
# ---------------------------------------------------------------------------

def ask_rag(
    question: str,
    k: int = 5,
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    Retrieve context, optionally pass conversation history to the LLM,
    and return {answer, sources, history}.
    """
    history = history or []

    # Fast off-topic guard — no LLM call needed
    if _is_off_topic(question):
        updated = history + [{"user": question, "assistant": _OFF_TOPIC_REPLY}]
        return {"answer": _OFF_TOPIC_REPLY, "sources": [], "history": updated}

    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore(force_rebuild=False)

    docs = _vectorstore.as_retriever(search_kwargs={"k": k}).invoke(question)
    context = _format_docs(docs)
    sources = [d.metadata for d in docs]
    history_text = _history_to_text(history)

    api_key = os.getenv("GROQ_TOKEN") or settings.groq_api_key
    if api_key and ChatOpenAI and ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT.strip()),
            (
                "human",
                "Question: {question}\n\nConversation history:\n{history}\n\nRetrieved context:\n{context}",
            ),
        ])
        llm = ChatOpenAI(
            model=settings.groq_model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            temperature=0,
        )
        answer = llm.invoke(
            prompt.format_messages(
                question=question,
                context=context,
                history=history_text,
            )
        ).content
    else:
        answer = _fallback_answer(question, docs)

    updated_history = history + [{"user": question, "assistant": answer}]
    return {"answer": answer, "sources": sources, "history": updated_history}


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

    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore(force_rebuild=False)

    docs = _vectorstore.as_retriever(search_kwargs={"k": k}).invoke(question)
    context = _format_docs(docs)
    sources = [d.metadata for d in docs]
    history_text = _history_to_text(history)

    api_key = os.getenv("GROQ_TOKEN") or settings.groq_api_key

    if not (api_key and ChatOpenAI):
        fallback = _fallback_answer(question, docs)
        yield f"data: {_json.dumps({'token': fallback})}\n\n"
        updated = history + [{"user": question, "assistant": fallback}]
        yield f"data: {_json.dumps({'done': True, 'sources': sources, 'history': updated})}\n\n"
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
    try:
        for chunk in llm.stream(prompt_msgs):
            token = chunk.content
            if token:
                full_answer += token
                yield f"data: {_json.dumps({'token': token})}\n\n"
    except Exception as exc:
        yield f"data: {_json.dumps({'token': f'[Error: {exc}]'})}\n\n"

    updated = history + [{"user": question, "assistant": full_answer}]
    yield f"data: {_json.dumps({'done': True, 'sources': sources, 'history': updated})}\n\n"
    yield "data: [DONE]\n\n"
