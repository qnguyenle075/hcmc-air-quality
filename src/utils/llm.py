"""Khởi tạo LLM: generator/agent qua Groq, judge RAGAS qua Cerebras (hoặc Groq)."""

from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config.settings import settings


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> ChatGroq:
    """Tạo ChatGroq; mặc định dùng LLM_MODEL trong .env. Ẩn phần reasoning khỏi output."""
    name = model or settings.llm.model
    if not name:
        raise RuntimeError("Chưa đặt LLM_MODEL trong .env")
    kwargs = {}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatGroq(
        model=name,
        temperature=settings.llm.temperature if temperature is None else temperature,
        timeout=settings.api.timeout_s * 3,
        max_retries=settings.api.max_retries,
        reasoning_format="hidden",
        callbacks=callbacks,
        **kwargs,
    )


def get_judge_llm(callbacks: list[BaseCallbackHandler] | None = None) -> BaseChatModel:
    """LLM chấm điểm RAGAS theo JUDGE_PROVIDER / JUDGE_MODEL.

    - cerebras: API tương thích OpenAI → ChatOpenAI + base_url; có rate limiter phía client (5 RPM);
      reasoning_effort="none" để tắt reasoning (qwen-3.8-27b).
    - groq: ChatGroq, reasoning effort thấp để tiết kiệm quota.
    """
    llm_cfg = settings.llm
    if not llm_cfg.judge_model:
        raise RuntimeError("Chưa đặt JUDGE_MODEL trong .env")
    if llm_cfg.judge_provider == "cerebras":
        if not llm_cfg.cerebras_api_key:
            raise RuntimeError("Chưa đặt CEREBRAS_API_KEY trong .env")
        return ChatOpenAI(
            model=llm_cfg.judge_model,
            base_url=llm_cfg.cerebras_base_url,
            api_key=llm_cfg.cerebras_api_key,
            temperature=0,
            max_tokens=llm_cfg.judge_max_tokens,
            reasoning_effort=llm_cfg.judge_reasoning_effort,
            timeout=settings.api.timeout_s * 6,
            max_retries=settings.api.max_retries,
            rate_limiter=InMemoryRateLimiter(
                requests_per_second=llm_cfg.judge_requests_per_minute / 60,
                check_every_n_seconds=0.5,
                max_bucket_size=1,
            ),
            callbacks=callbacks,
        )
    if llm_cfg.judge_provider == "groq":
        return get_llm(
            model=llm_cfg.judge_model,
            reasoning_effort=llm_cfg.judge_reasoning_effort,
            max_tokens=llm_cfg.judge_max_tokens,
            callbacks=callbacks,
        )
    raise ValueError(f"JUDGE_PROVIDER không hỗ trợ: {llm_cfg.judge_provider!r}")
