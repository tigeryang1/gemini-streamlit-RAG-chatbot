import os
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import dotenv_values, load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

DEFAULT_MODEL = "gemini-3-flash-preview"
SYSTEM_PROMPT = (
    "You are a concise, helpful AI assistant. "
    "Answer clearly, avoid unnecessary verbosity, and admit uncertainty when needed."
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


def build_llm(model_name: str, temperature: float) -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY. Set it in your environment or a .env file.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Tuple[str, str]] = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = SYSTEM_PROMPT


def to_langchain_messages(history: List[Tuple[str, str]], system_prompt: str) -> List[BaseMessage]:
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    for role, text in history:
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    return messages


def sidebar() -> tuple[str, float]:
    st.sidebar.title("Settings")
    model_name = st.sidebar.selectbox(
        "Gemini model",
        options=[
            DEFAULT_MODEL,
            "gemini-2.5-flash",
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
        ],
        index=0,
        help="Model availability and quotas vary by project and region.",
    )
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    st.session_state.system_prompt = st.sidebar.text_area(
        "System prompt",
        value=st.session_state.system_prompt,
        height=120,
    )
    if st.sidebar.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

    source, key = get_api_key_status()
    st.sidebar.divider()
    st.sidebar.subheader("Diagnostics")
    st.sidebar.write(f"Key source: `{source}`")
    st.sidebar.write(f"Key detected: `{mask_secret(key)}`")
    st.sidebar.caption("Masked for safety. If this says 'not set', the app is not seeing your key.")
    return model_name, temperature


def render_history() -> None:
    for role, text in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)


def main() -> None:
    st.set_page_config(page_title="Gemini Chatbot", page_icon="💬", layout="centered")
    init_state()

    st.title("Gemini Chatbot")
    st.caption("Streamlit + LangChain + Gemini Developer API")

    model_name, temperature = sidebar()
    render_history()

    prompt = st.chat_input("Ask something...")
    if not prompt:
        return

    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        llm = build_llm(model_name=model_name, temperature=temperature)
        messages = to_langchain_messages(
            history=st.session_state.chat_history,
            system_prompt=st.session_state.system_prompt,
        )
        response = llm.invoke(messages)
        answer = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        answer = f"Request failed: {exc}"

    st.session_state.chat_history.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)


if __name__ == "__main__":
    main()
