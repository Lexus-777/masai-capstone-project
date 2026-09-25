# Module 3: Support Assistant

## Architecture Walkthrough
The RAG pipeline operates as follows:
1. **Ingestion & Embedding:** 8 policy documents (`doc_01.txt` to `doc_08.txt`) are loaded from `/docs`, embedded locally using `sentence-transformers` (`all-MiniLM-L6-v2`), and indexed into a ChromaDB vector database.
2. **Intent Classification (`classify_intent`):** Queries are routed using keyword heuristics (e.g., "delivery", "refund"). Policy queries route to retrieval, while generic queries route to direct answering.
3. **Retrieval (`retrieve_and_answer`):** Queries classified under `policy_question` execute cosine similarity lookup in ChromaDB to extract top-3 relevant context chunks.
4. **Generation:** Outputs structured Pydantic JSON responses (`answer`, `sources`, `confidence`).
5. **MOCK_LLM Toggle:** Controlled via `MOCK_LLM` environment variable (default `1`). In mock mode, deterministic canned templates deliver grounded context cleanly without external API network calls.

## Local Running Instructions
Run the service with Uvicorn:
```bash
uvicorn main:app --host 0.0.0.0 --port 7860