"""CLI chạy thử agent, in trace tool call (tên tool + tham số + kết quả rút gọn) để debug.

Chat nhiều lượt:   python -m src.agent.run
Hỏi 1 câu rồi thoát: python -m src.agent.run "Không khí ở phường Tân Thuận hôm nay thế nào?"

Lệnh trong chat: /new = hội thoại mới (xóa bộ nhớ), /exit = thoát.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid

from groq import APIError
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from src.agent.graph import build_graph, run_config

_PREVIEW_CHARS = 300  # độ dài tối đa khi in kết quả tool


def _preview(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + " …"


def ask(graph: CompiledStateGraph, question: str, thread_id: str) -> str:
    """Gửi 1 câu hỏi, in trace từng bước, trả câu trả lời cuối."""
    start, answer = time.perf_counter(), ""
    try:
        for update in graph.stream(
            {"messages": [HumanMessage(question)]}, run_config(thread_id), stream_mode="updates"
        ):
            for node, out in update.items():
                for msg in (out or {}).get("messages", []):
                    if isinstance(msg, AIMessage) and msg.tool_calls:
                        for call in msg.tool_calls:
                            print(f"  → {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
                    elif isinstance(msg, ToolMessage):
                        print(f"  ← {msg.name}: {_preview(msg.content)}")
                    elif isinstance(msg, AIMessage) and node == "agent":
                        answer = msg.content
    except GraphRecursionError:
        answer = "Agent vượt quá số bước cho phép (recursion_limit) — dừng để tránh vòng lặp."
    except APIError as exc:
        # Lỗi LLM của agent (429 hết quota, 503 quá tải, mất kết nối): báo ngắn gọn thay vì văng traceback
        status = getattr(exc, "status_code", None)
        answer = f"Không gọi được LLM ({type(exc).__name__}{f', HTTP {status}' if status else ''}) — thử lại sau."
    print(f"  ({time.perf_counter() - start:.1f}s)")
    return answer


def main() -> None:
    """Chat nhiều lượt trên một thread; có câu hỏi ở tham số dòng lệnh thì hỏi 1 câu rồi thoát."""
    sys.stdout.reconfigure(encoding="utf-8")  # console Windows mặc định cp1252 → lỗi khi in tiếng Việt
    parser = argparse.ArgumentParser(description="Chạy thử HCMC Air Quality agent")
    parser.add_argument("question", nargs="?", help="Câu hỏi (bỏ trống để chat nhiều lượt)")
    args = parser.parse_args()

    graph = build_graph()
    thread_id = uuid.uuid4().hex
    if args.question:
        print(ask(graph, args.question, thread_id))
        return

    print("HCMC Air Quality agent — /new: hội thoại mới, /exit: thoát")
    while True:
        try:
            question = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question in ("/exit", "/quit"):
            break
        if question == "/new":
            thread_id = uuid.uuid4().hex
            print("(hội thoại mới)")
            continue
        print(f"\nAgent: {ask(graph, question, thread_id)}")


if __name__ == "__main__":
    main()
