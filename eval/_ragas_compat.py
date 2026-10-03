"""Shim tương thích: ragas 0.4.3 import `langchain_community.chat_models.vertexai.ChatVertexAI`,
module này đã bị gỡ khỏi langchain-community 0.4.x → ImportError ngay khi `import ragas`.

ragas chỉ dùng ChatVertexAI trong 1 tuple isinstance (ragas/llms/base.py) để kiểm tra model có hỗ trợ
nhiều completion không. Project không dùng Vertex AI → thay bằng class rỗng là an toàn.
Import module này TRƯỚC khi import ragas. Gỡ bỏ khi ragas phát hành bản sửa lỗi.
"""

from __future__ import annotations

import importlib
import sys
import types


def apply() -> None:
    """Đăng ký module giả nếu langchain-community không còn module vertexai."""
    name = "langchain_community.chat_models.vertexai"
    try:
        importlib.import_module(name)
    except ImportError:
        stub = types.ModuleType(name)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
        sys.modules[name] = stub


apply()
