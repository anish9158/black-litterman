import json
import os
import textwrap
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import load_settings

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
except Exception:
    ChatOpenAI = None
    ChatPromptTemplate = None

BASE_DIR = Path(__file__).resolve().parents[1]
RAG_DOCS_DIR = BASE_DIR / "rag_docs"
RAG_INDEX_DIR = BASE_DIR / "rag_index"

OVERVIEW = """
This application was converted from a Black-Litterman portfolio notebook with RAG support.
The workflow downloads market prices, computes returns, forms Black-Litterman implied equilibrium returns,
combines those returns with return views, optimizes long-only weights, evaluates metrics, and exposes a chatbot
that answers questions from indexed project documents and the original notebook.
"""

def read_file(path: Path) -> List[Document]:
    try:
        if path.suffix.lower() == ".ipynb":
            data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            return [Document(page_content="".join(c.get("source", [])), metadata={"source": path.name, "cell": i, "type": c.get("cell_type")}) for i, c in enumerate(data.get("cells", [])) if "".join(c.get("source", [])).strip()]
        if path.suffix.lower() in {".txt", ".md", ".py"}:
            return [Document(page_content=path.read_text(encoding="utf-8", errors="ignore"), metadata={"source": path.name})]
    except Exception as exc:
        return [Document(page_content=f"Could not parse {path.name}: {exc}", metadata={"source": path.name, "error": True})]
    return []

def load_documents() -> List[Document]:
    RAG_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    docs = [Document(page_content=textwrap.dedent(OVERVIEW).strip(), metadata={"source": "app_overview"})]
    for path in sorted(RAG_DOCS_DIR.rglob("*")):
        if path.is_file():
            docs.extend(read_file(path))
    return docs

def get_vectorstore(force_rebuild: bool = False):
    RAG_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if (RAG_INDEX_DIR / "index.faiss").exists() and not force_rebuild:
        return FAISS.load_local(str(RAG_INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
    docs = load_documents()
    chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(str(RAG_INDEX_DIR))
    return store

_vectorstore = None
settings = load_settings()

def ask_rag(question: str, k: int = 5) -> Dict[str, Any]:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = get_vectorstore(force_rebuild=False)
    docs = _vectorstore.as_retriever(search_kwargs={"k": k}).invoke(question)
    context = "\n\n".join([f"Source: {d.metadata}\n{d.page_content}" for d in docs])
    sources = [d.metadata for d in docs]
    api_key = os.getenv("GROQ_TOKEN") or settings.groq_api_key
    if api_key and ChatOpenAI and ChatPromptTemplate:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer only from the retrieved context. If unsupported, say so clearly. End with a short Sources section."),
            ("human", "Question: {question}\n\nRetrieved context:\n{context}")
        ])
        llm = ChatOpenAI(model=settings.groq_model, api_key=api_key, base_url="https://api.groq.com/openai/v1", temperature=0)
        answer = llm.invoke(prompt.format_messages(question=question, context=context)).content
    else:
        answer = "No GROQ key is configured, so this is retrieval-only.\n\n" + "\n\n".join([d.page_content[:700] for d in docs[:3]])
    return {"answer": answer, "sources": sources}
