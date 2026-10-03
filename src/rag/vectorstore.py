"""Build / load Chroma vector store (persist local)."""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config.settings import settings
from src.rag.embeddings import get_embeddings


def _chroma() -> Chroma:
    return Chroma(
        collection_name=settings.rag.collection_name,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.paths.chroma),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_vectorstore(chunks: list[Document]) -> Chroma:
    """Xóa collection cũ và index lại toàn bộ chunk (id = chunk_id để tái lập được)."""
    store = _chroma()
    store.delete_collection()
    store = _chroma()
    store.add_documents(chunks, ids=[c.metadata["chunk_id"] for c in chunks])
    return store


def load_vectorstore() -> Chroma:
    """Mở vector store đã build; báo lỗi rõ ràng nếu chưa build."""
    store = _chroma()
    if store._collection.count() == 0:  # noqa: SLF001 — Chroma không có API public để đếm
        raise RuntimeError("Vector store rỗng. Chạy trước: uv run python scripts/build_index.py")
    return store
