import os

import streamlit as st

from rag_auth import get_api_key_status, mask_secret
from rag_config import KNOWLEDGE_DIR
from rag_llm import get_available_models, parse_model_chain
from rag_utils import load_local_sources, parse_embedding_chain


def sidebar() -> tuple[str, list[str], float, int, list, bool]:
    st.sidebar.title("RAG Settings")
    available_models = get_available_models()
    configured_primary = os.getenv("GEMINI_MODEL", available_models[0])
    if configured_primary not in available_models:
        available_models = [configured_primary, *available_models]
    model_name = st.sidebar.selectbox(
        "Gemini model",
        options=available_models,
        index=available_models.index(configured_primary),
    )
    default_fallbacks = [
        model for model in parse_model_chain(model_name)[1:] if model in available_models
    ]
    fallback_models = st.sidebar.multiselect(
        "Fallback models",
        options=[model for model in available_models if model != model_name],
        default=default_fallbacks,
        help="These models are tried automatically if the primary model hits a quota or rate limit.",
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
    embedding_chain = parse_embedding_chain()
    st.sidebar.divider()
    st.sidebar.subheader("Diagnostics")
    st.sidebar.write(f"Key source: `{source}`")
    st.sidebar.write(f"Key detected: `{mask_secret(key)}`")
    st.sidebar.write(f"Model chain: `{', '.join(parse_model_chain(model_name, fallback_models))}`")
    st.sidebar.write(f"Embedding chain: `{', '.join(embedding_chain)}`")
    st.sidebar.write(f"Local knowledge files: `{len(load_local_sources(KNOWLEDGE_DIR))}`")
    st.sidebar.write(f"Uploaded documents: `{len(uploaded_files) if uploaded_files else 0}`")
    return model_name, fallback_models, temperature, top_k, uploaded_files or [], rebuild


def render_history() -> None:
    for role, text in st.session_state.rag_chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)


def render_context_panel() -> None:
    with st.expander("Last retrieved context", expanded=False):
        if not st.session_state.rag_last_context:
            st.caption("No context retrieved yet.")
        else:
            for doc in st.session_state.rag_last_context:
                st.markdown(f"**{doc.metadata.get('source', 'unknown')}**")
                st.caption(f"Chunk {doc.metadata.get('chunk', '?')}")
                st.write(doc.page_content)

    with st.expander("Last model run", expanded=False):
        if not st.session_state.rag_last_model:
            st.caption("No model invocation yet.")
        else:
            st.write(f"LLM used: `{st.session_state.rag_last_model}`")
            if st.session_state.rag_model_failovers:
                st.write("LLM fallback attempts:")
                for item in st.session_state.rag_model_failovers:
                    st.code(item)

        if st.session_state.rag_last_embedding_model:
            st.write(f"Embedding model used: `{st.session_state.rag_last_embedding_model}`")
            if st.session_state.rag_embedding_failovers:
                st.write("Embedding fallback attempts:")
                for item in st.session_state.rag_embedding_failovers:
                    st.code(item)
