"""Test tool retrieve_health_guideline với chain giả (không gọi Groq, không tải model).
Integration test chạy chain thật: `pytest -m integration`."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.documents import Document

from src.rag import tool as rag_tool
from src.rag.prompts import NO_INFO_EN, NO_INFO_VI

DOCS = [
    Document(page_content=f"nội dung {i}",
             metadata={"doc_title": f"Tài liệu {i}", "section": f"Mục {i}", "source": f"src{i}", "chunk_id": f"c{i}"})
    for i in range(1, 6)
]


class FakeChain:
    """Chain giả: trả câu trả lời cố định hoặc raise lỗi."""

    def __init__(self, answer: str = "", error: Exception | None = None) -> None:
        self.answer, self.error, self.calls = answer, error, []

    def invoke(self, inp: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(inp)
        if self.error:
            raise self.error
        return {"question": inp["question"], "answer": self.answer, "contexts": DOCS, "sources": []}


@pytest.fixture
def use_chain(monkeypatch: pytest.MonkeyPatch):
    def _use(chain: FakeChain) -> FakeChain:
        monkeypatch.setattr(rag_tool, "_get_chain", lambda variant: chain)
        return chain

    return _use


def test_cited_refs_cac_kieu_trich_dan() -> None:
    assert rag_tool.cited_refs("A [2]. B [1][2]. C 【3†L1-L4】 D【5】") == [2, 1, 3, 5]
    assert rag_tool.cited_refs("không trích") == []


def test_tra_ve_nguon_duoc_trich(use_chain) -> None:
    chain = use_chain(FakeChain("AQI 180 thuộc mức 151 – 200 (Xấu) [3]. Nên ở trong nhà [1]."))
    out = rag_tool.retrieve_health_guideline.invoke({"query": "AQI 180 người hen suyễn nên làm gì?"})
    assert out["found"] is True
    assert [s["ref"] for s in out["sources"]] == [3, 1]
    assert out["sources"][0] == {"ref": 3, "doc_title": "Tài liệu 3", "section": "Mục 3", "source": "src3", "chunk_id": "c3"}
    assert out["variant"] == "v4"
    assert chain.calls == [{"question": "AQI 180 người hen suyễn nên làm gì?"}]


def test_khong_trich_hoac_trich_ngoai_khoang_thi_tra_moi_doan(use_chain) -> None:
    use_chain(FakeChain("Trả lời không trích dẫn [9]."))
    out = rag_tool.retrieve_guideline("câu hỏi")
    assert [s["ref"] for s in out["sources"]] == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("no_info", [NO_INFO_VI, NO_INFO_EN])
def test_tai_lieu_khong_co_thong_tin(use_chain, no_info: str) -> None:
    use_chain(FakeChain(no_info))
    out = rag_tool.retrieve_guideline("Máy lọc không khí loại nào tốt nhất?")
    assert out["found"] is False
    assert out["sources"] == []


def test_loi_chain_tra_tool_error(use_chain) -> None:
    use_chain(FakeChain(error=TimeoutError("hết thời gian")))
    out = rag_tool.retrieve_guideline("câu hỏi")
    assert out["error"] == "api_error"
    assert "hết thời gian" in out["message"]


def test_cau_hoi_rong(use_chain) -> None:
    chain = use_chain(FakeChain("x"))
    assert rag_tool.retrieve_guideline("   ")["error"] == "invalid_input"
    assert chain.calls == []


def test_schema_tool_tu_choi_query_rong() -> None:
    with pytest.raises(Exception):
        rag_tool.retrieve_health_guideline.invoke({"query": ""})


@pytest.mark.integration
def test_chain_that_tra_loi_co_nguon() -> None:
    out = rag_tool.retrieve_guideline("Giới hạn PM2,5 trung bình 24 giờ theo QCVN 05:2023 là bao nhiêu?")
    assert "error" not in out, out
    assert out["found"] is True
    assert any(s["source"] == "qcvn_05_2023" for s in out["sources"])
