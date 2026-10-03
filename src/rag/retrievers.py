"""Các retriever cho ablation V0–V4. Hiện có: dense (V0).

Các tầng sau (BM25/hybrid, multi-query, rerank) sẽ thêm vào file này theo đúng thứ tự phase.
"""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever

from config.settings import settings
from src.rag.vectorstore import load_vectorstore


def get_dense_retriever(k: int = settings.rag.top_k) -> BaseRetriever:
    """Dense retriever: cosine similarity trên embedding bge-m3, lấy top-k chunk."""
    return load_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": k})
