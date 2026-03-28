import shutil
from pathlib import Path

from langchain_core.documents import Document

from rag_utils import build_documents, load_local_sources, read_source_bytes


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

