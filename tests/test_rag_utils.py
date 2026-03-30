import shutil
from pathlib import Path
from types import SimpleNamespace

from langchain_core.documents import Document

from rag_utils import (
    build_documents,
    build_vector_store,
    get_available_embedding_models,
    is_embedding_limit_error,
    load_local_sources,
    parse_embedding_chain,
    read_source_bytes,
)
from rag_app import (
    get_available_models,
    invoke_with_model_fallback,
    is_model_limit_error,
    normalize_model_name,
    parse_model_chain,
)


def test_read_source_bytes_supports_txt() -> None:
    loaded = read_source_bytes("guide.txt", b"alpha beta gamma")
    assert loaded is not None
    assert loaded.source == "guide.txt"
    assert "alpha" in loaded.text


def test_read_source_bytes_rejects_unknown_suffix() -> None:
    loaded = read_source_bytes("guide.csv", b"a,b,c")
    assert loaded is None


def test_load_local_sources_reads_project_files() -> None:
    local_dir = Path(__file__).resolve().parents[1] / "testdata_local"
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.mkdir()
    try:
        (local_dir / "guide.txt").write_text("alpha beta gamma", encoding="utf-8")
        sources = load_local_sources(local_dir)
        assert len(sources) == 1
        assert sources[0].source == "guide.txt"
    finally:
        shutil.rmtree(local_dir, ignore_errors=True)


def test_build_documents_adds_source_metadata() -> None:
    docs = build_documents(
        [
            type("LoadedSourceStub", (), {"source": "guide.txt", "text": "hello world " * 100})(),
        ]
    )
    assert docs
    assert isinstance(docs[0], Document)
    assert docs[0].metadata["source"] == "guide.txt"


def test_get_available_models_uses_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "GEMINI_AVAILABLE_MODELS",
        "Gemini 3.1 Flash Lite,Gemini 3 Flash,Gemini 2.5 Flash,Gemini 2.5 Flash Lite",
    )
    assert get_available_models() == [
        "Gemini 3.1 Flash Lite",
        "Gemini 3 Flash",
        "Gemini 2.5 Flash",
        "Gemini 2.5 Flash Lite",
    ]


def test_parse_model_chain_deduplicates_primary_and_fallbacks() -> None:
    chain = parse_model_chain(
        "Gemini 3.1 Flash Lite",
        ["Gemini 3 Flash", "Gemini 3.1 Flash Lite", "Gemini 2.5 Flash"],
    )
    assert chain == [
        "Gemini 3.1 Flash Lite",
        "Gemini 3 Flash",
        "Gemini 2.5 Flash",
        "Gemini 2.5 Flash Lite",
    ]


def test_is_model_limit_error_matches_quota_signals() -> None:
    assert is_model_limit_error(Exception("429 RESOURCE_EXHAUSTED: quota exceeded")) is True
    assert is_model_limit_error(Exception("400 INVALID_ARGUMENT: unexpected model name format")) is True
    assert is_model_limit_error(Exception("Invalid API key")) is False


def test_normalize_model_name_maps_known_display_labels() -> None:
    assert normalize_model_name("Gemini 3.1 Flash Lite") == "gemini-3.1-flash-lite-preview"
    assert normalize_model_name("Gemini 3 Flash") == "gemini-3-flash-preview"
    assert normalize_model_name("Gemini 2.5 Flash") == "gemini-2.5-flash"
    assert normalize_model_name("Gemini 2.5 Flash Lite") == "gemini-2.5-flash-lite"


def test_invoke_with_model_fallback_switches_on_limit_error(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_build_llm(model_name: str, temperature: float):
        class FakeLlm:
            def invoke(self, messages):
                attempts.append(model_name)
                if model_name == "Gemini 3.1 Flash Lite":
                    raise Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
                return SimpleNamespace(content=f"response from {model_name}")

        return FakeLlm()

    monkeypatch.setattr("rag_llm.build_llm", fake_build_llm)
    response, used_model, errors = invoke_with_model_fallback(
        messages=["hello"],
        model_chain=["Gemini 3.1 Flash Lite", "Gemini 3 Flash"],
        temperature=0.2,
    )

    assert attempts == ["Gemini 3.1 Flash Lite", "Gemini 3 Flash"]
    assert used_model == "Gemini 3 Flash"
    assert response.content == "response from Gemini 3 Flash"
    assert len(errors) == 1


def test_get_available_embedding_models_uses_env(monkeypatch) -> None:
    monkeypatch.setenv(
        "GEMINI_AVAILABLE_EMBEDDING_MODELS",
        "gemini-embedding-001,gemini-embedding-002",
    )
    assert get_available_embedding_models() == [
        "gemini-embedding-001",
        "gemini-embedding-002",
    ]


def test_parse_embedding_chain_includes_fallback_model() -> None:
    chain = parse_embedding_chain(
        "gemini-embedding-001",
        ["gemini-embedding-002", "gemini-embedding-001"],
    )
    assert chain == ["gemini-embedding-001", "gemini-embedding-002"]


def test_is_embedding_limit_error_matches_quota_signals() -> None:
    assert is_embedding_limit_error(Exception("429 RESOURCE_EXHAUSTED: quota exceeded")) is True
    assert is_embedding_limit_error(Exception("Unsupported embedding model")) is False


def test_build_vector_store_switches_embedding_model_on_limit_error(monkeypatch) -> None:
    docs = [Document(page_content="hello", metadata={"source": "guide.txt", "chunk": 0})]
    attempts: list[str] = []

    class FakeFaiss:
        @staticmethod
        def from_documents(documents, embedding):
            attempts.append(embedding.model)
            if embedding.model == "gemini-embedding-001":
                raise Exception("429 RESOURCE_EXHAUSTED: quota exceeded")
            return {"ok": True, "model": embedding.model}

    class FakeEmbeddings:
        def __init__(self, model_name: str):
            self.model = model_name

    monkeypatch.setattr("rag_utils.FAISS", FakeFaiss)

    def fake_builder(model_name: str, api_key: str):
        return FakeEmbeddings(model_name)

    store, used_model, errors = build_vector_store(
        docs,
        api_key="fake-key",
        embedding_chain=["gemini-embedding-001", "gemini-embedding-002"],
        embedding_builder=fake_builder,
    )

    assert attempts == ["gemini-embedding-001", "gemini-embedding-002"]
    assert used_model == "gemini-embedding-002"
    assert store["model"] == "gemini-embedding-002"
    assert len(errors) == 1
