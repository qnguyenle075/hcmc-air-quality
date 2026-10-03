"""Test retriever: hàm thuần (tokenizer BM25, RRF, parse multi-query, rerank với model giả)
+ hybrid V1 và cross-encoder V3 thật (cần Chroma đã build + model reranker)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from config.settings import settings
from src.rag import retrievers
from src.rag.retrievers import bm25_tokenize, parse_queries, rerank, rrf_fuse


def test_bm25_tokenize_bo_dau_cau_va_chu_thuong() -> None:
    assert bm25_tokenize("Giới hạn NO2, PM2,5 và PM2.5?") == ["giới", "hạn", "no2", "pm2", "5", "và", "pm2", "5"]


def test_bm25_tokenize_chuan_hoa_nfc() -> None:
    # "ạ" dạng tổ hợp (a + dấu nặng) phải trùng dạng dựng sẵn
    assert bm25_tokenize("hạn") == bm25_tokenize("hạn")


def _d(cid: str) -> Document:
    return Document(page_content=cid, metadata={"chunk_id": cid})


def test_rrf_fuse_khu_trung_va_cong_diem() -> None:
    # "b" có mặt ở cả 2 danh sách → điểm cộng dồn cao nhất
    fused = rrf_fuse([[_d("a"), _d("b")], [_d("b"), _d("c")]])
    assert [d.metadata["chunk_id"] for d in fused] == ["b", "a", "c"]


def test_parse_queries_bo_danh_so_va_dong_trong() -> None:
    text = "1. Giới hạn PM2,5\n\n- WHO 24-hour PM2.5\n• câu ba\ncâu bốn"
    assert parse_queries(text, 3) == ["Giới hạn PM2,5", "WHO 24-hour PM2.5", "câu ba"]


class _FakeCrossEncoder:
    """Điểm = độ dài nội dung → biết trước thứ tự."""

    def predict(self, pairs, batch_size: int = 1):
        return [float(len(text)) for _, text in pairs]


def test_rerank_sap_theo_diem_va_cat_top_n(monkeypatch) -> None:
    monkeypatch.setattr(retrievers, "get_cross_encoder", lambda: _FakeCrossEncoder())
    docs = [Document(page_content="x" * n, metadata={"chunk_id": str(n)}) for n in (1, 3, 2, 5)]
    out = rerank("q", docs, top_n=2)
    assert [d.metadata["chunk_id"] for d in out] == ["5", "3"]
    assert out[0].metadata["rerank_score"] == 5.0
    assert "rerank_score" not in docs[3].metadata  # không sửa chunk gốc


def test_rerank_rong() -> None:
    assert rerank("q", []) == []


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


def test_cross_encoder_that_dua_bang_qcvn_len_dau() -> None:
    """Rerank thật (GPU nếu có) trên 20 ứng viên hybrid — không gọi LLM."""
    try:
        from src.rag.retrievers import get_hybrid_retriever

        candidates = get_hybrid_retriever(k=settings.rag.rerank_fetch_k).invoke("PM2,5 trung bình 24 giờ QCVN")
    except RuntimeError as e:
        pytest.skip(str(e))
    question = "Giới hạn PM2,5 trung bình 24 giờ theo QCVN 05:2023 là bao nhiêu?"
    out = rerank(question, candidates)
    ids = [d.metadata["chunk_id"] for d in out]
    assert len(ids) == settings.rag.rerank_top_n
    assert "qcvn_05_2023-004" in ids[:2]  # Bảng 1 QCVN
    scores = [d.metadata["rerank_score"] for d in out]
    assert scores == sorted(scores, reverse=True)
