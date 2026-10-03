"""Các retriever cho ablation V0–V4. Hiện có: dense (V0), BM25 + hybrid (V1).

Các tầng sau (multi-query, rerank) sẽ thêm vào file này theo đúng thứ tự phase.
Import path kiểm tra theo version đã cài (langchain-classic 1.0.8, langchain-community 0.4.2):
EnsembleRetriever chỉ có ở langchain_classic; BM25Retriever ở langchain_community.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from config.settings import settings
from src.rag.vectorstore import load_vectorstore

_WORD_RE = re.compile(r"\w+")


def bm25_tokenize(text: str) -> list[str]:
    """Tách từ cho BM25: NFC + chữ thường + tách theo \\w+ (bỏ dấu câu).

    Tiếng Việt tách theo âm tiết (khoảng trắng) là đủ cho BM25. Bỏ dấu câu để "NO2," khớp "NO2"
    và "PM2,5" / "PM2.5" cùng thành ["pm2", "5"].
    """
    return _WORD_RE.findall(unicodedata.normalize("NFC", text).lower())


@lru_cache(maxsize=1)
def load_indexed_chunks() -> tuple[Document, ...]:
    """Lấy toàn bộ chunk đã index trong Chroma → BM25 dùng đúng tập chunk với dense."""
    data = load_vectorstore().get(include=["documents", "metadatas"])
    docs = [Document(page_content=t, metadata=m) for t, m in zip(data["documents"], data["metadatas"])]
    return tuple(sorted(docs, key=lambda d: d.metadata["chunk_id"]))  # thứ tự cố định để tái lập


def get_dense_retriever(k: int = settings.rag.top_k) -> BaseRetriever:
    """Dense retriever: cosine similarity trên embedding bge-m3, lấy top-k chunk."""
    return load_vectorstore().as_retriever(search_type="similarity", search_kwargs={"k": k})


def get_bm25_retriever(k: int = settings.rag.top_k) -> BM25Retriever:
    """Sparse retriever BM25 (rank_bm25 Okapi) trên cùng tập chunk với dense."""
    return BM25Retriever.from_documents(list(load_indexed_chunks()), k=k, preprocess_func=bm25_tokenize)


def get_hybrid_retriever(
    k: int = settings.rag.top_k,
    weights: tuple[float, float] = settings.rag.hybrid_weights,
) -> Runnable[str, list[Document]]:
    """Hybrid (V1): BM25 + dense, gộp bằng weighted Reciprocal Rank Fusion (EnsembleRetriever).

    Mỗi retriever con lấy k chunk; hợp của hai danh sách có thể tới 2k chunk → cắt lại còn k
    để số chunk đưa vào prompt bằng V0 (so sánh công bằng).
    """
    ensemble = EnsembleRetriever(
        retrievers=[get_bm25_retriever(k), get_dense_retriever(k)],
        weights=list(weights),  # (bm25, dense)
    )
    return RunnableLambda(lambda q: ensemble.invoke(q)[:k], name="hybrid_retriever")
