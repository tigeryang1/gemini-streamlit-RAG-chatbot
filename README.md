# Gemini Streamlit RAG Chatbot

Simple chatbot app built with `Streamlit`, `LangChain`, and the Gemini Developer API. This repo also includes a more realistic RAG chatbot with embeddings, a FAISS index, and document uploads.

## Stack

- `Streamlit` for the chat UI
- `LangChain` for chat message handling
- `langchain-google-genai` for Gemini integration
- Embedding-based RAG flow with Gemini embeddings and FAISS

## Project Files

- `app.py` - basic chat app
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

## Run

Basic chat app:

```powershell
streamlit run app.py
```

Sample RAG chat app:

```powershell
streamlit run rag_app.py
```

## RAG Sample

The sample RAG app:

- loads local `.txt`, `.pdf`, and `.docx` files
- supports additional uploaded documents from the UI
- splits them into chunks with `RecursiveCharacterTextSplitter`
- creates embeddings with Gemini `gemini-embedding-001`
- builds an in-memory `FAISS` vector store
- retrieves the most similar chunks for a question
- passes those chunks to Gemini as grounded context
- shows the last retrieved context in the UI

To expand the knowledge base, add more files to `knowledge_base/` or upload them in the sidebar.

## Notes

- Do not commit `.env` files or reuse any Gemini API key that was pasted into chat or source control.
- Gemini free-tier rate limits and available models can change by project and region.
- If `gemini-2.5-flash` is unavailable for your project, switch to `gemini-2.0-flash-lite` or `gemini-2.0-flash`.
- The vector index is in-memory for the current app session and is rebuilt when local or uploaded documents change.

## Free-tier Reference

Google's Gemini API docs indicate free-tier access exists and publish model-specific rate limits. The quickstart also shows `GEMINI_API_KEY` environment variable usage and examples with Gemini Flash models.

Sources:

- [Gemini API Quickstart](https://ai.google.dev/gemini-api/docs/quickstart)
- [Gemini API Billing](https://ai.google.dev/gemini-api/docs/billing/)
- [Gemini API Quota](https://ai.google.dev/gemini-api/docs/quota)
