"""Cấu hình tập trung: đọc biến môi trường từ .env và khai báo các hằng số có thể tinh chỉnh.

Mọi tham số (chunk size, k, trọng số hybrid, model...) đặt ở đây, không hardcode rải rác.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str = "") -> str:
    """Đọc biến môi trường, bỏ khoảng trắng thừa."""
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Paths:
    """Đường dẫn dữ liệu và index."""

    root: Path = ROOT_DIR
    raw: Path = ROOT_DIR / "data" / "raw"
    processed: Path = ROOT_DIR / "data" / "processed"
    curated: Path = ROOT_DIR / "data" / "curated"
    chroma: Path = (ROOT_DIR / _env("CHROMA_DIR", "./chroma_db")).resolve()
    eval_datasets: Path = ROOT_DIR / "eval" / "datasets"
    eval_results: Path = ROOT_DIR / "eval" / "results"
    geocode_cache: Path = ROOT_DIR / ".cache" / "geocode.json"


@dataclass(frozen=True)
class LLMSettings:
    """Cấu hình LLM qua Groq."""

    groq_api_key: str = _env("GROQ_API_KEY")
    model: str = _env("LLM_MODEL")
    judge_model: str = _env("JUDGE_MODEL")
    temperature: float = 0.0


@dataclass(frozen=True)
class RAGSettings:
    """Tham số pipeline RAG (V0–V4)."""

    embedding_model: str = _env("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_fallback: str = "intfloat/multilingual-e5-base"
    reranker_model: str = _env("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    device: str = _env("DEVICE", "cuda")
    use_fp16: bool = True  # GPU 4 GB VRAM → bắt buộc fp16
    max_seq_length: int = 512
    embed_batch_size: int = 8

    # Chunking (đơn vị: token)
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Retrieval
    top_k: int = 5  # số chunk đưa vào prompt
    hybrid_weights: tuple[float, float] = (0.5, 0.5)  # (bm25, dense)
    multi_query_n: int = 3
    rerank_fetch_k: int = 20  # V3: lấy rộng trước khi rerank
    rerank_top_n: int = 5

    collection_name: str = "hcmc_aq_guidelines"


@dataclass(frozen=True)
class APISettings:
    """Cấu hình API ngoài (WAQI, Nominatim)."""

    waqi_token: str = _env("WAQI_TOKEN")
    waqi_base_url: str = "https://api.waqi.info"
    nominatim_url: str = "https://nominatim.openstreetmap.org/search"
    nominatim_user_agent: str = _env("NOMINATIM_USER_AGENT", "hcmc-aq-agent/0.1")
    nominatim_min_interval_s: float = 1.0  # chính sách Nominatim: ≤ 1 req/s
    timeout_s: float = 10.0
    max_retries: int = 2
    backoff_s: float = 1.0
    station_far_km: float = 10.0  # trạm xa hơn ngưỡng này → cảnh báo


@dataclass(frozen=True)
class GeoSettings:
    """Phạm vi địa lý: TP.HCM CŨ (trước sáp nhập 1/7/2025).

    Bbox là giá trị xấp xỉ, cần đối chiếu lại với ranh giới OSM ở Phase 2.
    """

    # (lat_min, lat_max, lng_min, lng_max)
    hcmc_bbox: tuple[float, float, float, float] = (10.37, 11.17, 106.35, 107.03)
    city_suffix: str = ", Thành phố Hồ Chí Minh"


@dataclass(frozen=True)
class AgentSettings:
    """Cấu hình LangGraph agent."""

    recursion_limit: int = 8


@dataclass(frozen=True)
class Settings:
    """Gom toàn bộ cấu hình."""

    paths: Paths = field(default_factory=Paths)
    llm: LLMSettings = field(default_factory=LLMSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)
    api: APISettings = field(default_factory=APISettings)
    geo: GeoSettings = field(default_factory=GeoSettings)
    agent: AgentSettings = field(default_factory=AgentSettings)


settings = Settings()
