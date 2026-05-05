# Embeddings, FAISS, and knowledge chunks (this app's RAG path)

## Text → vectors
**Sentence embeddings** map each text chunk into a moderate-dimensional vector preserving semantic neighbourhood structure (cosine similarity or L2 after normalisation regimes).

**sentence-transformers/all-MiniLM-L6-v2** is lightweight and CPU-friendly adequate for explanatory portfolio documentation scale.

## Chunking rationale
Whole documents overwhelm context budgets and dilute specificity. Recursive character splitting near ~ hundreds of characters with overlaps preserves headings adjacency mildly.

Smaller chunks: higher precision retrieval poorer global narrative. Larger chunks bleed irrelevant sentences into top-k retrieval sets.

## FAISS ANN indexing
Approximate neighbourhood search retrieves nearest chunk vectors to encoded user query quickly—latency friendly for SSE streaming chat backends.

Returned **embedding distance scores** denote geometric separation—not calibrated probability correctness.

## Combining retrieval with LLMs
Retriever supplies **evidence excerpts** stitched into constrained prompts so generation stays grounded.**Guardrails** (keyword prefilter plus instruction prompts) constrain domain when internet search disabled.

Improvement horizons: BM25 lexical hybrid merges acronym hits (Ticker symbols) retrieval gaps pure dense vectors gloss over.
