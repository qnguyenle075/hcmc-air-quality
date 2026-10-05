"""Gọi API ngoài: timeout, retry tối đa `max_retries` lần với backoff lũy thừa."""

from __future__ import annotations

import time
from typing import Any

import httpx

from config.settings import settings

# Mã HTTP đáng thử lại (quá tải / lỗi tạm thời phía server)
_RETRY_STATUS = {429, 500, 502, 503, 504}


class APIError(Exception):
    """Lỗi gọi API sau khi đã hết lượt thử lại."""


def get_json(url: str, params: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
    """GET và parse JSON. Thử lại khi timeout / lỗi mạng / mã 429, 5xx; lỗi khác raise ngay `APIError`."""
    cfg = settings.api
    last_err = ""
    for attempt in range(cfg.max_retries + 1):
        if attempt:
            time.sleep(cfg.backoff_s * 2 ** (attempt - 1))
        try:
            resp = httpx.get(url, params=params, headers=headers, timeout=cfg.timeout_s)
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = f"{type(e).__name__}: {e}"
            continue
        if resp.status_code in _RETRY_STATUS:
            last_err = f"HTTP {resp.status_code}"
            continue
        if resp.status_code != 200:
            raise APIError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        try:
            return resp.json()
        except ValueError as e:
            raise APIError(f"Response không phải JSON: {e}") from e
    raise APIError(f"Thất bại sau {cfg.max_retries + 1} lần gọi ({last_err})")
