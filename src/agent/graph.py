"""LangGraph agent: node `agent` (LLM gắn 3 tool) ⇄ node `tools` (ToolNode).

START → agent ─(có tool call)→ tools → agent ... ─(không tool call)→ END

- System prompt không lưu trong state: dựng lại mỗi lượt gọi LLM để luôn có thời điểm hiện tại.
- Checkpointer lưu lịch sử theo `thread_id` → câu hỏi tiếp theo dùng lại kết quả tool của lượt trước.
- Giới hạn số bước bằng `recursion_limit` (truyền qua config, xem `run_config`) để tránh vòng lặp.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from config.settings import settings
from src.agent.prompts import build_system_prompt


def default_tools() -> list[BaseTool]:
    """3 tool của agent. Import trễ: tool RAG kéo theo embedding / reranker."""
    from src.rag.tool import retrieve_health_guideline
    from src.tools.air_quality import get_air_quality
    from src.tools.geocode import geocode_address

    return [geocode_address, get_air_quality, retrieve_health_guideline]


def now_vn() -> datetime:
    """Thời điểm hiện tại theo giờ Việt Nam."""
    return datetime.now(ZoneInfo(settings.api.timezone))


def build_graph(
    llm: BaseChatModel | None = None,
    tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    now: Callable[[], datetime] = now_vn,
) -> CompiledStateGraph:
    """Dựng và compile graph. Mặc định: LLM Groq (LLM_MODEL), 3 tool thật, bộ nhớ trong RAM."""
    if llm is None:
        from src.utils.llm import get_llm

        llm = get_llm()
    tools = list(tools) if tools is not None else default_tools()
    model = llm.bind_tools(tools)

    def agent(state: MessagesState) -> dict:
        response = model.invoke([SystemMessage(build_system_prompt(now())), *state["messages"]])
        return {"messages": [response]}

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    # handle_tool_errors=True: exception bất kỳ trong tool → ToolMessage lỗi cho LLM, không làm sập graph
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


def run_config(thread_id: str, recursion_limit: int | None = None) -> RunnableConfig:
    """Config cho 1 hội thoại: thread_id (bộ nhớ) + giới hạn số bước."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit or settings.agent.recursion_limit,
    }
