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

# Grounded portfolio-analysis system prompt from notebook cell 14
_SYSTEM_PROMPT = """
You are a grounded portfolio-analysis assistant for this Black-Litterman notebook.
Answer only from the retrieved context.
If the answer is not supported by the retrieved context, say that clearly.
Prefer concise, analytical answers.
End with a short Sources section listing the retrieved sources you used.
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
    RAG_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    docs = [Document(page_content=NOTEBOOK_OVERVIEW, metadata={"source": "app_overview"})]
    for path in sorted(RAG_DOCS_DIR.rglob("*")):
        if path.is_file():
            docs.extend(read_file(path))
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
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore(force_rebuild=False)

    docs = _vectorstore.as_retriever(search_kwargs={"k": k}).invoke(question)
    context = _format_docs(docs)
    sources = [d.metadata for d in docs]
    history = history or []
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
