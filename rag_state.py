from typing import List, Tuple

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from rag_config import DEFAULT_SYSTEM_PROMPT


def init_state() -> None:
    if "rag_chat_history" not in st.session_state:
        st.session_state.rag_chat_history: List[Tuple[str, str]] = []
    if "rag_system_prompt" not in st.session_state:
        st.session_state.rag_system_prompt = DEFAULT_SYSTEM_PROMPT
    if "rag_last_context" not in st.session_state:
        st.session_state.rag_last_context: List[Document] = []
    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = []
    if "rag_vector_store" not in st.session_state:
        st.session_state.rag_vector_store = None
    if "rag_index_signature" not in st.session_state:
        st.session_state.rag_index_signature = ""
    if "rag_last_model" not in st.session_state:
        st.session_state.rag_last_model = ""
    if "rag_model_failovers" not in st.session_state:
        st.session_state.rag_model_failovers = []
    if "rag_last_embedding_model" not in st.session_state:
        st.session_state.rag_last_embedding_model = ""
    if "rag_embedding_failovers" not in st.session_state:
        st.session_state.rag_embedding_failovers = []


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
