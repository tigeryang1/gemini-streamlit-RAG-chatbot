import os
from pathlib import Path
from typing import List, Tuple

import streamlit as st
from dotenv import dotenv_values, load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

DEFAULT_MODEL_OPTIONS = [
    "Gemini 3.1 Flash Lite",
    "Gemini 3 Flash",
    "Gemini 2.5 Flash",
    "Gemini 2.5 Flash Lite",
]
SYSTEM_PROMPT = (
    "You are a concise, helpful AI assistant. "
    "Answer clearly, avoid unnecessary verbosity, and admit uncertainty when needed."
)
MODEL_NAME_ALIASES = {
    "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite-preview",
    "Gemini 3 Flash": "gemini-3-flash-preview",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
}


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


def normalize_model_name(model_name: str) -> str:
    return MODEL_NAME_ALIASES.get(model_name, model_name)


def build_llm(model_name: str, temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=normalize_model_name(model_name),
        google_api_key=get_api_key(),
        temperature=temperature,
    )


def get_available_models() -> list[str]:
    env_value = os.getenv("GEMINI_AVAILABLE_MODELS", "")
    configured = [item.strip() for item in env_value.split(",") if item.strip()]
    models = configured or DEFAULT_MODEL_OPTIONS
    return list(dict.fromkeys(models))


def parse_model_chain(primary_model: str, fallback_models: list[str] | None = None) -> list[str]:
    configured = fallback_models or []
    if not configured:
        env_value = os.getenv("GEMINI_FALLBACK_MODELS", "")
        configured = [item.strip() for item in env_value.split(",") if item.strip()]

    chain: list[str] = []
    for candidate in [primary_model, *configured, *get_available_models()]:
        if candidate and candidate not in chain:
            chain.append(candidate)
    return chain


def is_model_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    signals = [
        "429",
        "quota",
        "rate limit",
        "resource exhausted",
        "resource_exhausted",
        "too many requests",
        "exceeded",
        "limit reached",
        "invalid_argument",
        "unexpected model name format",
        "model not found",
        "unsupported model",
    ]
    return any(signal in message for signal in signals)


def invoke_with_model_fallback(messages, model_chain: list[str], temperature: float):
    if not model_chain:
        raise ValueError("Model chain is empty.")

    errors: list[str] = []
    last_exc: Exception | None = None
    for index, model_name in enumerate(model_chain):
        try:
            llm = build_llm(model_name=model_name, temperature=temperature)
            response = llm.invoke(messages)
            return response, model_name, errors
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if index == len(model_chain) - 1 or not is_model_limit_error(exc):
                raise
            errors.append(f"{model_name}: {exc}")

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Model invocation failed without an exception.")


def init_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: List[Tuple[str, str]] = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = SYSTEM_PROMPT
    if "last_model" not in st.session_state:
        st.session_state.last_model = ""
    if "model_failovers" not in st.session_state:
        st.session_state.model_failovers = []


def to_langchain_messages(history: List[Tuple[str, str]], system_prompt: str) -> List[BaseMessage]:
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt)]
    for role, text in history:
        if role == "user":
            messages.append(HumanMessage(content=text))
        else:
            messages.append(AIMessage(content=text))
    return messages


def sidebar() -> tuple[str, list[str], float]:
    st.sidebar.title("Settings")
    available_models = get_available_models()
    configured_primary = os.getenv("GEMINI_MODEL", available_models[0])
    if configured_primary not in available_models:
        available_models = [configured_primary, *available_models]
    model_name = st.sidebar.selectbox(
        "Gemini model",
        options=available_models,
        index=available_models.index(configured_primary),
        help="Model availability and quotas vary by project and region.",
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
    st.sidebar.write(f"Model chain: `{', '.join(parse_model_chain(model_name, fallback_models))}`")
    st.sidebar.caption("Masked for safety. If this says 'not set', the app is not seeing your key.")
    return model_name, fallback_models, temperature


def render_history() -> None:
    for role, text in st.session_state.chat_history:
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(text)


def render_model_panel() -> None:
    with st.expander("Last model run", expanded=False):
        if not st.session_state.last_model:
            st.caption("No model invocation yet.")
            return
        st.write(f"Model used: `{st.session_state.last_model}`")
        if st.session_state.model_failovers:
            st.write("Automatic fallback attempts:")
            for item in st.session_state.model_failovers:
                st.code(item)


def main() -> None:
    st.set_page_config(page_title="Gemini Chatbot", page_icon="💬", layout="centered")
    init_state()

    st.title("Gemini Chatbot")
    st.caption("Streamlit + LangChain + Gemini Developer API")

    model_name, fallback_models, temperature = sidebar()
    render_history()
    render_model_panel()

    prompt = st.chat_input("Ask something...")
    if not prompt:
        return

    st.session_state.chat_history.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        messages = to_langchain_messages(
            history=st.session_state.chat_history,
            system_prompt=st.session_state.system_prompt,
        )
        response, used_model, failovers = invoke_with_model_fallback(
            messages=messages,
            model_chain=parse_model_chain(model_name, fallback_models),
            temperature=temperature,
        )
        st.session_state.last_model = used_model
        st.session_state.model_failovers = failovers
        answer = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        st.session_state.last_model = ""
        st.session_state.model_failovers = []
        answer = f"Request failed: {exc}"

    st.session_state.chat_history.append(("assistant", answer))
    with st.chat_message("assistant"):
        st.markdown(answer)


if __name__ == "__main__":
    main()
