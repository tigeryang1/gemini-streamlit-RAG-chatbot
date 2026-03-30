# Gemini Streamlit RAG Chatbot

RAG chatbot app built with `Streamlit`, `LangChain`, and the Gemini Developer API, with embeddings, a FAISS index, and document uploads.

## Stack

- `Streamlit` for the chat UI
- `LangChain` for chat message handling
- `langchain-google-genai` for Gemini integration
- Embedding-based RAG flow with Gemini embeddings and FAISS
- Automatic Gemini model fallback on quota and rate-limit failures

## Project Files

- `rag_app.py` - RAG-enabled chat app
- `rag_utils.py` - ingestion, chunking, and vector retrieval helpers
- `knowledge_base/` - sample retrieval documents
- `requirements.txt` - Python dependencies
- `.env.example` - environment variable template

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set your Gemini API key:

```powershell
Copy-Item .env.example .env
```

Optional model fallback chain:

```text
GEMINI_MODEL=Gemini 3.1 Flash Lite
GEMINI_FALLBACK_MODELS=Gemini 3 Flash,Gemini 2.5 Flash,Gemini 2.5 Flash Lite
GEMINI_AVAILABLE_MODELS=Gemini 3.1 Flash Lite,Gemini 3 Flash,Gemini 2.5 Flash,Gemini 2.5 Flash Lite
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
GEMINI_FALLBACK_EMBEDDING_MODELS=gemini-embedding-002
GEMINI_AVAILABLE_EMBEDDING_MODELS=gemini-embedding-001,gemini-embedding-002
```

## Run

RAG chat app:

```powershell
streamlit run rag_app.py
```

## RAG Sample

The sample RAG app:

- loads local `.txt`, `.pdf`, and `.docx` files
- supports additional uploaded documents from the UI
- splits them into chunks with `RecursiveCharacterTextSplitter`
- creates embeddings with Gemini `gemini-embedding-001`
- can fall back to `gemini-embedding-002` if the primary embedding model hits a quota or rate-limit error
- builds an in-memory `FAISS` vector store
- retrieves the most similar chunks for a question
- passes those chunks to Gemini as grounded context
- automatically switches to the next configured Gemini model when the current one hits quota or rate-limit errors
- shows the last retrieved context in the UI

To expand the knowledge base, add more files to `knowledge_base/` or upload them in the sidebar.

## Notes

- Do not commit `.env` files or reuse any Gemini API key that was pasted into chat or source control.
- Gemini free-tier rate limits and available models can change by project and region.
- The app only switches models automatically for quota and rate-limit style failures, not for invalid prompts, auth failures, or unsupported model names.
- The vector index build now has its own embedding fallback chain, separate from the chat-model fallback chain.
- The vector index is in-memory for the current app session and is rebuilt when local or uploaded documents change.
- This project intentionally uses a local FAISS index only. It does not include a remote Google vector-store mode such as Gemini File Search or Vertex AI Vector Search.

## Free-tier Reference

Google's Gemini API docs indicate free-tier access exists and publish model-specific rate limits. The quickstart also shows `GEMINI_API_KEY` environment variable usage and examples with Gemini Flash models.

Sources:

- [Gemini API Quickstart](https://ai.google.dev/gemini-api/docs/quickstart)
- [Gemini API Billing](https://ai.google.dev/gemini-api/docs/billing/)
- [Gemini API Quota](https://ai.google.dev/gemini-api/docs/quota)
