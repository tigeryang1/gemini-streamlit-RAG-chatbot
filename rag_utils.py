from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from langchain_core.documents import Document

try:
    from langchain_community.vectorstores import FAISS
except ModuleNotFoundError:  # pragma: no cover - depends on optional install
    FAISS = None

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except ModuleNotFoundError:  # pragma: no cover - depends on optional install
    GoogleGenerativeAIEmbeddings = None

try:
    from pypdf import PdfReader
except ModuleNotFoundError:  # pragma: no cover - depends on optional install
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ModuleNotFoundError:  # pragma: no cover - depends on optional install
    DocxDocument = None

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ModuleNotFoundError:  # pragma: no cover - depends on optional install
    RecursiveCharacterTextSplitter = None


@dataclass
class LoadedSource:
    source: str
    text: str


def _read_txt_bytes(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore").strip()


def _read_pdf_bytes(data: bytes) -> str:
    if PdfReader is None:
        raise ModuleNotFoundError("pypdf is required to ingest PDF documents.")

    reader = PdfReader(BytesIO(data))
    text_parts = []
    for page in reader.pages:
        text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def _read_docx_bytes(data: bytes) -> str:
    if DocxDocument is None:
        raise ModuleNotFoundError("python-docx is required to ingest DOCX documents.")

    doc = DocxDocument(BytesIO(data))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs).strip()


def read_source_bytes(name: str, data: bytes) -> LoadedSource | None:
    suffix = Path(name).suffix.lower()
    if suffix == ".txt":
        text = _read_txt_bytes(data)
    elif suffix == ".pdf":
        text = _read_pdf_bytes(data)
    elif suffix == ".docx":
        text = _read_docx_bytes(data)
    else:
        return None

    if not text:
        return None
    return LoadedSource(source=name, text=text)


def load_local_sources(knowledge_dir: Path) -> list[LoadedSource]:
    sources: list[LoadedSource] = []
    for path in sorted(knowledge_dir.glob("*")):
        if not path.is_file():
            continue
        loaded = read_source_bytes(path.name, path.read_bytes())
        if loaded:
            sources.append(loaded)
    return sources


def build_documents(sources: Iterable[LoadedSource]) -> list[Document]:
    docs: list[Document] = []
    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split = splitter.split_text
    else:
        def split(text: str) -> list[str]:
            chunks: list[str] = []
            start = 0
            while start < len(text):
                end = min(len(text), start + 800)
                chunks.append(text[start:end].strip())
                if end >= len(text):
                    break
                start = max(end - 150, start + 1)
            return [chunk for chunk in chunks if chunk]

    for source in sources:
        for index, chunk in enumerate(split(source.text)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": source.source, "chunk": index},
                )
            )
    return docs


def build_vector_store(documents: list[Document], api_key: str) -> Any:
    if FAISS is None or GoogleGenerativeAIEmbeddings is None:
        raise ModuleNotFoundError(
            "langchain-community and langchain-google-genai are required to build the vector store."
        )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key,
    )
    return FAISS.from_documents(documents, embedding=embeddings)


def retrieve_context(store: Any, query: str, top_k: int = 4) -> list[Document]:
    return store.similarity_search(query, k=top_k)
