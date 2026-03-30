import os
from pathlib import Path

from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
ENV_PATH = APP_DIR / ".env"
KNOWLEDGE_DIR = APP_DIR / "knowledge_base"

load_dotenv(dotenv_path=ENV_PATH)

DEFAULT_MODEL_OPTIONS = [
    "Gemini 3.1 Flash Lite",
    "Gemini 3 Flash",
    "Gemini 2.5 Flash",
    "Gemini 2.5 Flash Lite",
]
DEFAULT_SYSTEM_PROMPT = (
    "You are a RAG-enabled assistant. Use retrieved context when relevant, cite source file names, "
    "and say clearly when the available documents do not support a confident answer."
)
MODEL_NAME_ALIASES = {
    "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite-preview",
    "Gemini 3 Flash": "gemini-3-flash-preview",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Flash Lite": "gemini-2.5-flash-lite",
}
