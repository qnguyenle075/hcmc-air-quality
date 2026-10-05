"""Schema dùng chung cho output lỗi của các tool."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ErrorCode = Literal["invalid_input", "not_found", "out_of_scope", "api_error", "no_data"]


class ToolError(BaseModel):
    """Lỗi có cấu trúc trả cho agent thay vì raise exception."""

    error: ErrorCode = Field(description="Mã lỗi")
    message: str = Field(description="Mô tả lỗi cho agent / người dùng")
