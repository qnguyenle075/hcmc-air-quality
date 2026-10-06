"""Tool `retrieve_health_guideline`: hỏi đáp trên tài liệu chuẩn (WHO AQG 2021, QCVN 05:2023, QĐ 1459/QĐ-TCMT)
bằng RAG chain variant tốt nhất (`settings.rag.tool_variant`).

- Chain (embedding, reranker trên GPU) chỉ dựng một lần, lần gọi đầu tiên.
- Trả câu trả lời kèm các nguồn được trích [n]; lỗi (Groq 429, timeout...) → `ToolError`, không raise.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.settings import settings
from src.rag.chain import build_rag_chain
from src.rag.prompts import NO_INFO_EN, NO_INFO_VI
from src.tools.schemas import ToolError

# Trích dẫn trong câu trả lời: [1], [1][2], 【1】, 【1†L1-L4】
_CITATION_RE = re.compile(r"[\[【](\d+)")


class GuidelineInput(BaseModel):
    """Input của retrieve_health_guideline."""

    query: str = Field(
        min_length=1,
        description="Câu hỏi cần tra trong tài liệu, viết đầy đủ ý (vd 'VN_AQI 160 thì trẻ em có nên ra ngoài "
        "chơi không?', 'Giới hạn PM2,5 trung bình 24 giờ theo QCVN 05:2023'). A self-contained question.",
    )


class GuidelineSource(BaseModel):
    """Một đoạn tài liệu được trích trong câu trả lời."""

    ref: int = Field(description="Số thứ tự [n] dùng trong câu trả lời")
    doc_title: str | None
    section: str | None
    source: str | None = Field(description="Mã tài liệu (qcvn_05_2023 | who_aqg_2021 | qd_1459_vn_aqi | aqi_categories)")
    chunk_id: str | None


class GuidelineResult(BaseModel):
    """Output thành công của retrieve_health_guideline."""

    answer: str = Field(description="Câu trả lời dựa trên tài liệu, có trích dẫn [n]")
    found: bool = Field(description="False nếu tài liệu không có thông tin về câu hỏi")
    sources: list[GuidelineSource]
    variant: str


@lru_cache(maxsize=1)
def _get_chain(variant: str) -> Runnable:
    """Dựng chain một lần (tải embedding + reranker lên GPU tốn vài giây)."""
    return build_rag_chain(variant)


def cited_refs(answer: str) -> list[int]:
    """Các số [n] được trích trong câu trả lời, theo thứ tự xuất hiện, không trùng."""
    return list(dict.fromkeys(int(n) for n in _CITATION_RE.findall(answer)))


def is_no_info(answer: str) -> bool:
    """Câu trả lời là câu từ chối cố định 'tài liệu không có thông tin'."""
    return NO_INFO_VI in answer or NO_INFO_EN in answer


def to_sources(docs: list[Document], refs: list[int]) -> list[GuidelineSource]:
    """Nguồn của các đoạn được trích; câu trả lời không trích [n] nào → trả mọi đoạn đã đưa vào prompt."""
    keep = [r for r in refs if 1 <= r <= len(docs)] or list(range(1, len(docs) + 1))
    return [
        GuidelineSource(
            ref=r,
            doc_title=docs[r - 1].metadata.get("doc_title"),
            section=docs[r - 1].metadata.get("section"),
            source=docs[r - 1].metadata.get("source"),
            chunk_id=docs[r - 1].metadata.get("chunk_id"),
        )
        for r in keep
    ]


def retrieve_guideline(query: str, variant: str | None = None) -> dict[str, Any]:
    """Chạy RAG cho 1 câu hỏi. Trả `GuidelineResult` hoặc `ToolError` dưới dạng dict."""
    query = (query or "").strip()
    if not query:
        return ToolError(error="invalid_input", message="Câu hỏi rỗng.").model_dump()
    variant = variant or settings.rag.tool_variant
    try:
        out = _get_chain(variant).invoke({"question": query})
    except Exception as e:  # noqa: BLE001 — tool không được làm sập agent
        return ToolError(error="api_error", message=f"Không tra được tài liệu: {type(e).__name__}: {e}").model_dump()

    answer = out["answer"].strip()
    found = not is_no_info(answer)
    sources = to_sources(out["contexts"], cited_refs(answer)) if found else []
    return GuidelineResult(answer=answer, found=found, sources=sources, variant=variant).model_dump()


@tool("retrieve_health_guideline", args_schema=GuidelineInput)
def retrieve_health_guideline(query: str) -> dict[str, Any]:
    """Tra tài liệu chuẩn về chất lượng không khí và sức khỏe: khuyến nghị sức khỏe theo mức VN_AQI
    (QĐ 1459/QĐ-TCMT), giới hạn nồng độ theo QCVN 05:2023/BTNMT, mức hướng dẫn WHO 2021, cách tính VN_AQI.

    Bắt buộc gọi trước khi đưa khuyến nghị sức khỏe (ra ngoài, tập thể dục, mở cửa sổ, đeo khẩu trang,
    trẻ em / người già / người bệnh hô hấp, tim mạch...). Với khuyến nghị theo chỉ số, đưa giá trị VN_AQI vào
    câu hỏi (vd "VN_AQI 160, người già có nên đi bộ buổi sáng?"). Chỉ dùng nội dung trong `answer`;
    `found=false` nghĩa là tài liệu không có thông tin — không tự bổ sung.

    Look up official guidelines (Vietnamese AQI health advice, QCVN 05:2023 limits, WHO 2021 guidelines).
    Must be called before giving any health recommendation. Returns {answer, found, sources, variant}
    or {error, message}.
    """
    return retrieve_guideline(query)
