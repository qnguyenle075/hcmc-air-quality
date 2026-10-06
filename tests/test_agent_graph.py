"""Test định tuyến của LangGraph agent với LLM giả và tool giả (không gọi Groq / API ngoài)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx
import pytest
from groq import APIConnectionError
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from src.agent.graph import build_graph, run_config
from src.agent.prompts import build_system_prompt
from src.agent.run import ask

NOW = datetime(2026, 10, 6, 9, 30)


class FakeToolLLM(FakeMessagesListChatModel):
    """LLM giả trả lần lượt các AIMessage cho trước; ghi lại input mỗi lượt gọi."""

    seen: list[list[Any]] = []

    def bind_tools(self, tools: Any, **kwargs: Any) -> FakeToolLLM:
        return self

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.seen.append(list(input))
        # Bản sao không id: trả lại cùng một AIMessage (đã có id) thì add_messages ghi đè thay vì nối thêm
        return super().invoke(input, config, **kwargs).model_copy(update={"id": None})


def call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@tool
def geocode_address(address: str) -> dict:
    """Giả lập geocode."""
    return {"lat": 10.74, "lng": 106.72, "ward": "Phường Tân Thuận", "display_name": address}


@tool
def get_air_quality(lat: float, lng: float) -> dict:
    """Giả lập chất lượng không khí."""
    return {"vn_aqi": 160, "category": "Xấu", "dominant_pollutant": "pm25"}


@tool
def retrieve_health_guideline(query: str) -> dict:
    """Giả lập tra tài liệu."""
    return {"answer": "Nên ở trong nhà [1].", "found": True, "sources": [], "variant": "v4"}


TOOLS = [geocode_address, get_air_quality, retrieve_health_guideline]


def make_graph(responses: list[AIMessage]) -> tuple[Any, FakeToolLLM]:
    llm = FakeToolLLM(responses=responses, seen=[])
    return build_graph(llm=llm, tools=TOOLS, now=lambda: NOW), llm


def tool_trajectory(messages: list[Any]) -> list[str]:
    return [m.name for m in messages if isinstance(m, ToolMessage)]


def test_luong_day_du_geocode_aqi_rag_roi_tra_loi() -> None:
    graph, llm = make_graph([
        call("geocode_address", {"address": "phường Tân Thuận"}, "c1"),
        call("get_air_quality", {"lat": 10.74, "lng": 106.72}, "c2"),
        call("retrieve_health_guideline", {"query": "VN_AQI 160, trẻ em có nên ra ngoài?"}, "c3"),
        AIMessage(content="VN_AQI 160 (Xấu) ..."),
    ])
    out = graph.invoke({"messages": [HumanMessage("Tân Thuận hôm nay cho con ra ngoài được không?")]},
                       run_config("t1"))
    assert tool_trajectory(out["messages"]) == ["geocode_address", "get_air_quality", "retrieve_health_guideline"]
    assert out["messages"][-1].content == "VN_AQI 160 (Xấu) ..."
    assert len(llm.seen) == 4
    # Kết quả tool được đưa lại cho LLM ở lượt sau
    assert '"vn_aqi": 160' in llm.seen[2][-1].content


def test_khong_co_tool_call_thi_ket_thuc_ngay() -> None:
    graph, llm = make_graph([AIMessage(content="Xin lỗi, tôi chỉ hỗ trợ chất lượng không khí tại TP.HCM.")])
    out = graph.invoke({"messages": [HumanMessage("Giá vàng hôm nay?")]}, run_config("t2"))
    assert tool_trajectory(out["messages"]) == []
    assert len(out["messages"]) == 2
    assert len(llm.seen) == 1


def test_system_prompt_moi_luot_co_ngay_hien_tai_va_khong_luu_vao_state() -> None:
    graph, llm = make_graph([AIMessage(content="ok")])
    out = graph.invoke({"messages": [HumanMessage("chào")]}, run_config("t3"))
    first = llm.seen[0][0]
    assert isinstance(first, SystemMessage)
    assert "06/10/2026 09:30" in first.content
    assert not any(isinstance(m, SystemMessage) for m in out["messages"])


def test_hoi_tiep_cung_thread_giu_lich_su_khong_geocode_lai() -> None:
    graph, llm = make_graph([
        call("geocode_address", {"address": "Quận 7"}, "c1"),
        call("get_air_quality", {"lat": 10.74, "lng": 106.72}, "c2"),
        AIMessage(content="VN_AQI 160 (Xấu)."),
        # lượt 2: LLM dùng lại AQI trong lịch sử, chỉ tra khuyến nghị
        call("retrieve_health_guideline", {"query": "VN_AQI 160, có nên mở cửa sổ?"}, "c3"),
        AIMessage(content="Không nên ..."),
    ])
    cfg = run_config("t4")
    graph.invoke({"messages": [HumanMessage("Không khí Quận 7 thế nào?")]}, cfg)
    out = graph.invoke({"messages": [HumanMessage("Vậy có nên mở cửa sổ không?")]}, cfg)
    # Lượt 2: LLM thấy toàn bộ lịch sử lượt 1 (gồm kết quả get_air_quality)
    assert any(isinstance(m, ToolMessage) and m.name == "get_air_quality" for m in llm.seen[3])
    assert tool_trajectory(out["messages"]) == ["geocode_address", "get_air_quality", "retrieve_health_guideline"]
    assert out["messages"][-1].content == "Không nên ..."


def test_thread_khac_khong_dung_chung_bo_nho() -> None:
    graph, llm = make_graph([AIMessage(content="a")])
    graph.invoke({"messages": [HumanMessage("câu 1")]}, run_config("t5"))
    out = graph.invoke({"messages": [HumanMessage("câu 2")]}, run_config("t6"))
    assert [m.content for m in out["messages"]] == ["câu 2", "a"]


def test_vong_lap_tool_bi_chan_boi_recursion_limit() -> None:
    graph, _ = make_graph([call("geocode_address", {"address": "x"}, "c1")])  # LLM giả gọi tool mãi
    with pytest.raises(GraphRecursionError):
        graph.invoke({"messages": [HumanMessage("x")]}, run_config("t7"))


def test_recursion_limit_mac_dinh_du_cho_luong_day_du() -> None:
    # geocode → AQI → RAG → trả lời = 7 bước (4 agent + 3 tools) phải lọt giới hạn mặc định
    graph, _ = make_graph([
        call("geocode_address", {"address": "Q1"}, "c1"),
        call("get_air_quality", {"lat": 10.77, "lng": 106.70}, "c2"),
        call("retrieve_health_guideline", {"query": "q"}, "c3"),
        AIMessage(content="xong"),
    ])
    out = graph.invoke({"messages": [HumanMessage("Q1?")]}, run_config("t8"))
    assert out["messages"][-1].content == "xong"


def test_tool_loi_khong_lam_sap_graph() -> None:
    @tool
    def geocode_address(address: str) -> dict:
        """Tool giả raise exception."""
        raise RuntimeError("Nominatim sập")

    llm = FakeToolLLM(responses=[call("geocode_address", {"address": "x"}, "c1"),
                                 AIMessage(content="Không lấy được tọa độ.")], seen=[])
    graph = build_graph(llm=llm, tools=[geocode_address], now=lambda: NOW)
    out = graph.invoke({"messages": [HumanMessage("x")]}, run_config("t9"))
    err = [m for m in out["messages"] if isinstance(m, ToolMessage)][0]
    assert err.status == "error"
    assert out["messages"][-1].content == "Không lấy được tọa độ."


def test_system_prompt_noi_dung_bat_buoc() -> None:
    text = build_system_prompt(NOW)
    assert "Thứ Ba, 06/10/2026 09:30" in text
    for name in ("geocode_address", "get_air_quality", "retrieve_health_guideline"):
        assert name in text
    assert "không phải trạm quan trắc" in text
    assert "không thay thế tư vấn y tế" in text


class FailingLLM(FakeToolLLM):
    """LLM giả luôn lỗi kết nối (thay cho 429 / 503 của Groq)."""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        raise APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))


def test_ask_bat_loi_llm_khong_vang_traceback() -> None:
    graph = build_graph(llm=FailingLLM(responses=[], seen=[]), tools=TOOLS, now=lambda: NOW)
    answer = ask(graph, "Không khí ở Quận 7 thế nào?", "t-loi")
    assert answer.startswith("Không gọi được LLM (APIConnectionError")
