"""Test retriever: tokenizer BM25 (thuần) + hybrid V1 (cần Chroma đã build)."""

from __future__ import annotations

import pytest

from config.settings import settings
from src.rag.retrievers import bm25_tokenize


def test_bm25_tokenize_bo_dau_cau_va_chu_thuong() -> None:
    assert bm25_tokenize("Giới hạn NO2, PM2,5 và PM2.5?") == ["giới", "hạn", "no2", "pm2", "5", "và", "pm2", "5"]


def test_bm25_tokenize_chuan_hoa_nfc() -> None:
    # "ạ" dạng tổ hợp (a + dấu nặng) phải trùng dạng dựng sẵn
    assert bm25_tokenize("hạn") == bm25_tokenize("hạn")


@pytest.fixture(scope="module")
def hybrid():
    try:
        from src.rag.retrievers import get_hybrid_retriever

        return get_hybrid_retriever()
    except RuntimeError as e:  # vector store chưa build
        pytest.skip(str(e))


def test_hybrid_tra_dung_top_k_khong_trung(hybrid) -> None:
    docs = hybrid.invoke("Giới hạn PM2,5 trung bình 24 giờ theo QCVN 05:2023")
    ids = [d.metadata["chunk_id"] for d in docs]
    assert len(ids) == settings.rag.top_k
    assert len(set(ids)) == len(ids)
    assert "qcvn_05_2023-004" in ids  # Bảng 1 QCVN
