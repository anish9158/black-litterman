# RAG (retrieval-augmented generation) in domain-specific assistants

Stages: ingestion → chunk segmentation preserving semantic coherence → embedding into vector DB (Faiss cosine/L2 setups) → top-k retrieval for user query augmented into LLM context window constrained by prompting.

Closed-domain safeguards: whitelist topics, grounding requirement, abstention clauses, surfaced similarity scores aiding audit.

Trade-offs: chunky documents harm precision; microscopic chunks lose context hybrid retrieval (dense + lexical BM25) sometimes improves factual hit rate relevance reranking layers optional heavier compute latency token budgets.
