# Design Tradeoffs

## Retrieval

- Used FAISS for fast local vector search.
- Used Maximum Marginal Relevance (MMR) to improve diversity of retrieved passages.

## Embeddings

- Used sentence-transformers/all-MiniLM-L6-v2 for semantic similarity.

## Language Model

- Used Groq Llama 3.3 70B for low-latency inference.

## Limitations

- Works only with supplied PDFs.
- Does not perform live web search.
- Does not process scanned PDFs without OCR.
- Retrieval quality depends on the uploaded documents.