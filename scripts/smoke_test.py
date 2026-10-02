"""Smoke test Phase 0: kiểm tra Groq, WAQI, Nominatim và embedding trên GPU.

Chạy: uv run python scripts/smoke_test.py
Không in token/API key ra màn hình.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import settings  # noqa: E402

# Trung tâm Quận 1 cũ
TEST_LAT, TEST_LNG = 10.7769, 106.7009
TEST_ADDRESS = "Chợ Bến Thành" + settings.geo.city_suffix


def check_groq() -> bool:
    """Gọi thử LLM 1 câu. Nếu chưa chọn LLM_MODEL thì liệt kê model Groq hiện có."""
    if not settings.llm.groq_api_key:
        print("[Groq] BỎ QUA: chưa có GROQ_API_KEY trong .env")
        return False
    if not settings.llm.model:
        resp = httpx.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {settings.llm.groq_api_key}"},
            timeout=settings.api.timeout_s,
        )
        resp.raise_for_status()
        ids = sorted(m["id"] for m in resp.json()["data"])
        print("[Groq] Chưa đặt LLM_MODEL. Các model hiện có:")
        for model_id in ids:
            print(f"   - {model_id}")
        return False

    from langchain_groq import ChatGroq

    llm = ChatGroq(model=settings.llm.model, temperature=0, timeout=settings.api.timeout_s)
    t0 = time.perf_counter()
    answer = llm.invoke("Trả lời đúng 1 câu: PM2.5 là gì?").content
    print(f"[Groq] OK ({settings.llm.model}, {time.perf_counter() - t0:.2f}s): {answer}")
    return True


def check_waqi() -> bool:
    """Lấy dữ liệu WAQI tại 1 tọa độ."""
    if not settings.api.waqi_token:
        print("[WAQI] BỎ QUA: chưa có WAQI_TOKEN trong .env")
        return False
    url = f"{settings.api.waqi_base_url}/feed/geo:{TEST_LAT};{TEST_LNG}/"
    resp = httpx.get(url, params={"token": settings.api.waqi_token}, timeout=settings.api.timeout_s)
    resp.raise_for_status()
    body = resp.json()
    if body.get("status") != "ok":
        print(f"[WAQI] LỖI: {body.get('data')}")
        return False
    data = body["data"]
    iaqi = {k: v.get("v") for k, v in data.get("iaqi", {}).items()}
    print(
        f"[WAQI] OK: trạm={data['city']['name']} | geo={data['city']['geo']} | "
        f"aqi(US)={data['aqi']} | dominentpol={data.get('dominentpol')} | "
        f"time={data['time']['s']} | iaqi={iaqi}"
    )
    return True


def check_nominatim() -> bool:
    """Geocode 1 địa chỉ."""
    resp = httpx.get(
        settings.api.nominatim_url,
        params={"q": TEST_ADDRESS, "format": "json", "limit": 1, "countrycodes": "vn"},
        headers={"User-Agent": settings.api.nominatim_user_agent},
        timeout=settings.api.timeout_s,
    )
    resp.raise_for_status()
    results = resp.json()
    if not results:
        print(f"[Nominatim] LỖI: không tìm thấy '{TEST_ADDRESS}'")
        return False
    r = results[0]
    print(f"[Nominatim] OK: {r['lat']}, {r['lon']} | {r['display_name']}")
    return True


def check_embedding() -> bool:
    """Nạp bge-m3 (fp16) trên GPU và embed thử 2 câu."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = settings.rag.device if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model_kwargs = {"torch_dtype": torch.float16} if (device == "cuda" and settings.rag.use_fp16) else {}
    model = SentenceTransformer(settings.rag.embedding_model, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = settings.rag.max_seq_length
    vecs = model.encode(
        ["Giới hạn PM2.5 trung bình 24 giờ", "24-hour PM2.5 limit"],
        normalize_embeddings=True,
    )
    sim = float(vecs[0] @ vecs[1])
    vram = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
    print(
        f"[Embedding] OK: {settings.rag.embedding_model} trên {device} | dim={vecs.shape[1]} | "
        f"cos(vi,en)={sim:.3f} | VRAM peak={vram:.2f} GB | {time.perf_counter() - t0:.1f}s"
    )
    return True


def main() -> int:
    """Chạy tất cả check, trả exit code 0 nếu tất cả OK."""
    results: dict[str, bool] = {}
    for name, fn in [
        ("groq", check_groq),
        ("waqi", check_waqi),
        ("nominatim", check_nominatim),
        ("embedding", check_embedding),
    ]:
        try:
            results[name] = fn()
        except Exception as exc:  # smoke test: báo lỗi gọn, chạy tiếp check khác
            print(f"[{name}] LỖI: {type(exc).__name__}: {exc}")
            results[name] = False
    print("\nTổng kết:", {k: ("OK" if v else "FAIL/SKIP") for k, v in results.items()})
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
