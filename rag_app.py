from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import dotenv_values, load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from rag_utils import (
    LoadedSource,
    build_documents,
    build_vector_store,
    load_local_sources,
    read_source_bytes,
    retrieve_context,
)


APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"
KNOWLEDGE_DIR = APP_DIR / "knowledge_base"

load_dotenv(dotenv_path=ENV_PATH)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_SYSTEM_PROMPT = (
    "You are a RAG-enabled assistant. Use retrieved context when relevant, cite source file names, "
    "and say clearly when the available documents do not support a confident answer."
)


def mask_secret(value: str | None) -> str:
    if not value:
        return "not set"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def get_api_key_status() -> tuple[str, str | None]:
    env_value = os.environ.get("GEMINI_API_KEY")
    if env_value:
        return "environment", env_value
    file_value = dotenv_values(ENV_PATH).get("GEMINI_API_KEY")
    if file_value:
        return ".env file present but not loaded", file_value
    return "not found", None


def get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Set it in your environment or a .env file.")
    return api_key


def build_llm(model_name: str, temperature: float) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=get_api_key(),
        temperature=temperature,
    )


def init_state() -> None:
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history: List[Tuple[str, str]] = []
    if "rag_system_prompt" not in st.session_state:
        st.session_state.rag_system_prompt = DEFAULT_SYSTEM_PROMPT
    if "rag_last_context" not in st.session_state:
        st.session_state.rag_last_context: List[Document] = []
    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources: list[LoadedSource] = []
    if "rag_vector_store" not in st.session_state:
        st.session_state.rag_vector_store = None
    if "rag_index_signature" not in st.session_state:
        st.session_state.rag_index_signature = ""


def build_messages(
    history: List[Tuple[str, str]],
    system_prompt: str,
    question: str,
    context_docs: list[Document],
) -> List[BaseMessage]:
    context_block = "No relevant context retrieved."
    if context_docs:
        context_block = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
            for doc in context_docs
        )

    messages: List[BaseMessage] = [
        SystemMessage(
            content=(
                f"{system_prompt}\n\nRetrieved context:\n{context_block}\n\n"
                "Base your answer on the retrieved context whenever possible."
            )
        )
    ]
    for role, text in history:
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    messages.append(HumanMessage(content=question))
    return messages


def gather_sources(uploaded_files) -> list[LoadedSource]:
    sources = load_local_sources(KNOWLEDGE_DIR)
    for uploaded in uploaded_files:
        loaded = read_source_bytes(uploaded.name, uploaded.getvalue())
        if loaded:
            sources.append(loaded)
    return sources


def build_index_if_needed(uploaded_files, force_rebuild: bool = False) -> None:
    signature_parts = [f"local:{path.name}:{path.stat().st_mtime_ns}" for path in sorted(KNOWLEDGE_DIR.glob("*")) if path.is_file()]
    signature_parts.extend(
        f"upload:{uploaded.name}:{len(uploaded.getvalue())}"
        for uploaded in uploaded_files
    )
    signature = "|".join(signature_parts)

    if not force_rebuild and st.session_state.rag_vector_store is not None and signature == st.session_state.rag_index_signature:
        return

    sources = gather_sources(uploaded_files)
    documents = build_documents(sources)
    if not documents:
        raise ValueError("No supported documents were found. Add .txt, .pdf, or .docx files.")

    st.session_state.rag_sources = sources
    st.session_state.rag_vector_store = build_vector_store(documents, api_key=get_api_key())
    st.session_state.rag_index_signature = signature


def sidebar():
    st.sidebar.title("RAG Settings")
    model_name = st.sidebar.selectbox(
        "Gemini model",
        options=[
            DEFAULT_MODEL,
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
        ],
        index=0,
    )
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    top_k = st.sidebar.slider("Retrieved chunks", min_value=1, max_value=6, value=4, step=1)
    st.session_state.rag_system_prompt = st.sidebar.text_area(
        "System prompt",
        value=st.session_state.rag_system_prompt,
        height=140,
    )
    uploaded_files = st.sidebar.file_uploader(
        "Upload documents",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Uploaded files are indexed in-memory for this session.",
    )
    rebuild = st.sidebar.button("Rebuild index")
    if st.sidebar.button("Clear chat"):
        st.session_state.rag_chat_history = []
        st.session_state.rag_last_context = []
        st.rerun()

    source, key = get_api_key_status()
    st.sidebar.divider()
    st.sidebar.subheader("Diagnostics")
    st.sidebar.write(f"Key source: `{source}`")
    st.sidebar.write(f"Key detected: `{mask_secret(key)}`")
    st.sidebar.write(f"Local knowledge files: `{len(load_local_sources(KNOWLEDGE_DIR))}`")
    st.sidebar.write(f"Uploaded documents: `{len(uploaded_files) if uploaded_files else 0}`")
    return model_name, temperature, top_k, uploaded_files or [], rebuild


def render_history() -> None:
    for role, text in st.session_state.rag_chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)


def render_context_panel() -> None:
    with st.expander("Last retrieved context", expanded=False):
        if not st.session_state.rag_last_context:
            st.caption("No context retrieved yet.")
            return
        for doc in st.session_state.rag_last_context:
            st.markdown(f"**{doc.metadata.get('source', 'unknown')}**")
            st.caption(f"Chunk {doc.metadata.get('chunk', '?')}")
            st.write(doc.page_content)


def main() -> None:
    st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="📚", layout="centered")
    init_state()

    st.title("Gemini RAG Chatbot")
    st.caption("Streamlit + Gemini embeddings + FAISS + local document retrieval")
    st.write(
        "This version builds an in-memory vector index from local files in `knowledge_base/` "
        "and any uploaded `.txt`, `.pdf`, or `.docx` documents."
    )

    model_name, temperature, top_k, uploaded_files, rebuild = sidebar()
    try:
        build_index_if_needed(uploaded_files, force_rebuild=rebuild)
    except Exception as exc:
        st.error(f"Index build failed: {exc}")

    render_history()
    render_context_panel()

    prompt = st.chat_input("Ask about your indexed documents...")
    if not prompt:
        return

    st.session_state.rag_chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        if st.session_state.rag_vector_store is None:
            raise ValueError("The vector index is not available.")

        retrieved = retrieve_context(st.session_state.rag_vector_store, prompt, top_k=top_k)
        st.session_state.rag_last_context = retrieved
        llm = build_llm(model_name=model_name, temperature=temperature)
        messages = build_messages(
            history=st.session_state.rag_chat_history[:-1],
            system_prompt=st.session_state.rag_system_prompt,
            question=prompt,
            context_docs=retrieved,
        )
        response = llm.invoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        answer = f"Request failed: {exc}"

    st.session_state.rag_chat_history.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)


if __name__ == "__main__":
    main()

