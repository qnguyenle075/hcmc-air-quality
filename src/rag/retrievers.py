"""Các retriever cho ablation V0–V4: dense (V0), BM25 + hybrid (V1), multi-query (V2), rerank (V3).

Import path kiểm tra theo version đã cài (langchain-classic 1.0.8, langchain-community 0.4.2):
EnsembleRetriever chỉ có ở langchain_classic; BM25Retriever ở langchain_community.
Reranker dùng thẳng sentence_transformers.CrossEncoder (6.1.0) thay vì HuggingFaceCrossEncoder của
langchain_community để kiểm soát fp16 + max_length (GPU 4 GB), giống embeddings.py.
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from functools import lru_cache

import torch
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda
from sentence_transformers import CrossEncoder

from config.settings import settings
from src.rag.prompts import MULTI_QUERY_PROMPT
from src.rag.vectorstore import load_vectorstore
from src.utils.llm import get_llm

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


def _ensemble(k: int, weights: tuple[float, float]) -> EnsembleRetriever:
    """BM25 + dense gộp bằng weighted RRF; trả về hợp hai danh sách (tối đa 2k chunk)."""
    return EnsembleRetriever(
        retrievers=[get_bm25_retriever(k), get_dense_retriever(k)],
        weights=list(weights),  # (bm25, dense)
    )


def get_hybrid_retriever(
    k: int = settings.rag.top_k,
    weights: tuple[float, float] = settings.rag.hybrid_weights,
) -> Runnable[str, list[Document]]:
    """Hybrid (V1): BM25 + dense, gộp bằng weighted Reciprocal Rank Fusion (EnsembleRetriever).

    Mỗi retriever con lấy k chunk; hợp của hai danh sách có thể tới 2k chunk → cắt lại còn k
    để số chunk đưa vào prompt bằng V0 (so sánh công bằng).
    """
    ensemble = _ensemble(k, weights)
    return RunnableLambda(lambda q: ensemble.invoke(q)[:k], name="hybrid_retriever")


def rrf_fuse(doc_lists: list[list[Document]], c: int = settings.rag.rrf_c) -> list[Document]:
    """Reciprocal Rank Fusion trọng số bằng nhau; khử trùng lặp theo chunk_id, giữ thứ tự điểm giảm dần."""
    scores: dict[str, float] = defaultdict(float)
    first: dict[str, Document] = {}
    for docs in doc_lists:
        for rank, doc in enumerate(docs, start=1):
            cid = doc.metadata["chunk_id"]
            scores[cid] += 1 / (rank + c)
            first.setdefault(cid, doc)
    return [first[cid] for cid in sorted(scores, key=lambda x: -scores[x])]


def parse_queries(text: str, n: int) -> list[str]:
    """Tách output LLM thành tối đa n truy vấn (mỗi dòng 1 câu, bỏ đánh số / gạch đầu dòng)."""
    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip() for line in text.splitlines()]
    return [q for q in lines if q][:n]


def get_multi_query_retriever(
    k: int = settings.rag.top_k,
    n: int = settings.rag.multi_query_n,
    weights: tuple[float, float] = settings.rag.hybrid_weights,
) -> Runnable[str, list[Document]]:
    """Multi-query (V2): câu gốc + n biến thể (LLM sinh) → hybrid cho từng câu → RRF → top k.

    Luôn giữ câu gốc trong danh sách truy vấn để không mất kết quả của V1 khi LLM viết lại lệch ý.
    V3 gọi hàm này với k = rerank_fetch_k để lấy rộng trước khi rerank.
    """
    ensemble = _ensemble(k, weights)
    # reasoning thấp: viết lại câu hỏi là việc đơn giản, tiết kiệm quota Groq
    gen_queries = MULTI_QUERY_PROMPT.partial(n=str(n)) | get_llm(reasoning_effort="low") | StrOutputParser()

    def retrieve(question: str) -> list[Document]:
        queries = [question, *parse_queries(gen_queries.invoke({"question": question}), n)]
        return rrf_fuse([ensemble.invoke(q) for q in queries])[:k]

    return RunnableLambda(retrieve, name="multi_query_retriever")


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    """Singleton cross-encoder bge-reranker-v2-m3 (fp16 trên GPU, tránh nạp lại tốn VRAM)."""
    cfg = settings.rag
    device = cfg.device if (cfg.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model_kwargs = {"torch_dtype": torch.float16} if (device == "cuda" and cfg.use_fp16) else {}
    return CrossEncoder(cfg.reranker_model, device=device, max_length=cfg.rerank_max_length, model_kwargs=model_kwargs)


def rerank(question: str, docs: list[Document], top_n: int = settings.rag.rerank_top_n) -> list[Document]:
    """Chấm (câu hỏi, chunk) bằng cross-encoder, trả top_n chunk điểm giảm dần; điểm ghi vào metadata."""
    if not docs:
        return []
    scores = get_cross_encoder().predict(
        [(question, d.page_content) for d in docs], batch_size=settings.rag.rerank_batch_size
    )
    ranked = sorted(zip(docs, scores), key=lambda x: -float(x[1]))[:top_n]
    # copy Document để không sửa metadata của chunk dùng chung (BM25 cache)
    return [Document(page_content=d.page_content, metadata={**d.metadata, "rerank_score": float(s)}) for d, s in ranked]


def get_rerank_retriever(
    fetch_k: int = settings.rag.rerank_fetch_k,
    top_n: int = settings.rag.rerank_top_n,
) -> Runnable[str, list[Document]]:
    """Rerank (V3): multi-query lấy rộng fetch_k chunk → cross-encoder chấm theo câu hỏi gốc → top_n."""
    candidates = get_multi_query_retriever(k=fetch_k)
    return RunnableLambda(lambda q: rerank(q, candidates.invoke(q), top_n), name="rerank_retriever")
