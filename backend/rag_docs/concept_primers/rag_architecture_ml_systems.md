# RAG (retrieval-augmented generation) in domain-specific assistants

## Pipeline phases
**(1) Ingestion** — Markdown, notebooks, PDFs, tables, JSON snippets.

**(2) Chunk segmentation** — Split on headings and character windows with overlap so each chunk stays locally coherent without spanning unrelated sections.

**(3) Embedding** — Map chunks to dense vectors (e.g. sentence-transformers / MiniLM class). Faster models trade some semantic nuance for latency.

**(4) Vector index** — **FAISS** (or similar) stores embeddings; distance metric matches the model training (often L2 on normalised vectors ≈ cosine).

**(5) Retrieval** — Encode the user question, fetch top‑k neighbours, optionally merge adjacent chunks.

**(6) Augmented generation** — Build a prompt: system rules + retrieved snippets + conversation; stream tokens back (e.g. SSE) to the client.

## Closed-domain safeguards
Keyword allow-lists or light classifiers gate off-topic questions before costly calls. Prompts insist answers cite only retrieved text. UI shows **sources** and similarity **distance** (or scores) so users can sanity-check grounding.

## Trade-offs and upgrades
Very large chunks dilute retrieval precision; tiny chunks lose definitions and notation. Dense retrieval can miss exact tickers — **hybrid BM25 + vector** retrieval is common. **Reranking** with a cross-encoder improves precision at extra latency. Context windows cap how many chunks fit at once — trim or summarise when budgets are tight.

## Evaluation (short checklist)
Grounded claims only from cited chunks? Retrieval hits the right subsection on labelled questions? Failures logged (wrong section, synonym mismatch) for iterative fixes.
