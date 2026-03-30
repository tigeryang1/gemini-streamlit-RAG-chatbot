import streamlit as st
from rag_auth import get_api_key
from rag_config import KNOWLEDGE_DIR
from rag_llm import (
    build_llm,
    get_available_models,
    invoke_with_model_fallback,
    is_model_limit_error,
    normalize_model_name,
    parse_model_chain,
)
from rag_state import build_messages, init_state
from rag_ui import render_context_panel, render_history, sidebar
from rag_utils import (
    build_documents,
    build_vector_store,
    load_local_sources,
    parse_embedding_chain,
    read_source_bytes,
    retrieve_context,
)


def gather_sources(uploaded_files) -> list:
    sources = load_local_sources(KNOWLEDGE_DIR)
    for uploaded in uploaded_files:
        loaded = read_source_bytes(uploaded.name, uploaded.getvalue())
        if loaded:
            sources.append(loaded)
    return sources


def build_index_if_needed(uploaded_files, force_rebuild: bool = False) -> None:
    signature_parts = [
        f"local:{path.name}:{path.stat().st_mtime_ns}"
        for path in sorted(KNOWLEDGE_DIR.glob("*"))
        if path.is_file()
    ]
    signature_parts.extend(
        f"upload:{uploaded.name}:{len(uploaded.getvalue())}" for uploaded in uploaded_files
    )
    signature = "|".join(signature_parts)

    if (
        not force_rebuild
        and st.session_state.rag_vector_store is not None
        and signature == st.session_state.rag_index_signature
    ):
        return

    sources = gather_sources(uploaded_files)
    documents = build_documents(sources)
    if not documents:
        raise ValueError("No supported documents were found. Add .txt, .pdf, or .docx files.")

    st.session_state.rag_sources = sources
    vector_store, embedding_model, embedding_failovers = build_vector_store(
        documents,
        api_key=get_api_key(),
        embedding_chain=parse_embedding_chain(),
    )
    st.session_state.rag_vector_store = vector_store
    st.session_state.rag_last_embedding_model = embedding_model
    st.session_state.rag_embedding_failovers = embedding_failovers
    st.session_state.rag_index_signature = signature

def main() -> None:
    st.set_page_config(page_title="Gemini RAG Chatbot", page_icon="📚", layout="centered")
    init_state()

    st.title("Gemini RAG Chatbot")
    st.caption("Streamlit + Gemini embeddings + FAISS + local document retrieval")
    st.write(
        "This version builds an in-memory vector index from local files in `knowledge_base/` "
        "and any uploaded `.txt`, `.pdf`, or `.docx` documents."
    )

    (
        model_name,
        fallback_models,
        temperature,
        top_k,
        uploaded_files,
        rebuild,
    ) = sidebar()
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
        messages = build_messages(
            history=st.session_state.rag_chat_history[:-1],
            system_prompt=st.session_state.rag_system_prompt,
            question=prompt,
            context_docs=retrieved,
        )
        response, used_model, failovers = invoke_with_model_fallback(
            messages=messages,
            model_chain=parse_model_chain(model_name, fallback_models),
            temperature=temperature,
        )
        st.session_state.rag_last_model = used_model
        st.session_state.rag_model_failovers = failovers
        answer = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        st.session_state.rag_last_model = ""
        st.session_state.rag_model_failovers = []
        answer = f"Request failed: {exc}"

    st.session_state.rag_chat_history.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)


if __name__ == "__main__":
    main()
