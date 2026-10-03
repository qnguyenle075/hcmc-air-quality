"""Embedding bge-m3 chạy local trên GPU (fp16), bọc theo interface Embeddings của LangChain."""

from __future__ import annotations

from functools import lru_cache

import torch
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from config.settings import settings


class SentenceTransformerEmbeddings(Embeddings):
    """Embeddings dùng sentence-transformers, chuẩn hóa vector (cosine = dot product).

    Tự viết thay vì dùng HuggingFaceEmbeddings để kiểm soát fp16 và max_seq_length (GPU 4 GB).
    """

    def __init__(
        self,
        model_name: str = settings.rag.embedding_model,
        device: str = settings.rag.device,
        use_fp16: bool = settings.rag.use_fp16,
        max_seq_length: int = settings.rag.max_seq_length,
        batch_size: int = settings.rag.embed_batch_size,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        model_kwargs = {"torch_dtype": torch.float16} if (device == "cuda" and use_fp16) else {}
        self._model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)
        self._model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed danh sách văn bản (chunk)."""
        vecs = self._model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed 1 câu truy vấn."""
        return self.embed_documents([text])[0]


@lru_cache(maxsize=1)
def get_embeddings() -> SentenceTransformerEmbeddings:
    """Singleton embedding model (tránh nạp lại model nhiều lần, tốn VRAM)."""
    return SentenceTransformerEmbeddings()
