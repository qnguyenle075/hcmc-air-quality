"""Tool `geocode_address`: địa chỉ / địa danh TP.HCM → tọa độ qua Nominatim (OpenStreetMap).

- Chuẩn hóa viết tắt ("Q7", "Q.7" → "Quận 7"; "P. Tân Thuận" → "Phường Tân Thuận"), thêm hậu tố thành phố.
- Lấy vài kết quả, chọn kết quả đầu tiên nằm trong TP.HCM CŨ (polygon, xem `boundary.py`); không có → lỗi out_of_scope.
- Tên phường/xã MỚI (sau 1/7/2025) lấy từ địa chỉ Nominatim; thiếu (vd node lịch sử "Quận 7") → reverse geocode.
- Cache kết quả vào file JSON; giới hạn ≤ 1 request/giây, User-Agent riêng của project.
"""

from __future__ import annotations

import json
import re
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.settings import settings
from src.tools import _http
from src.tools.boundary import in_hcmc_old
from src.tools.schemas import ToolError

SOURCE = "Nominatim (OpenStreetMap)"

# Chuỗi (đã bỏ dấu, chữ thường) cho biết địa chỉ đã nêu thành phố → không thêm hậu tố
_CITY_MARKERS = ("ho chi minh", "hcm", "sai gon", "saigon")
# Tiền tố tên đơn vị hành chính cấp xã sau sáp nhập
_WARD_PREFIXES = ("Phường ", "Xã ", "Đặc khu ")
# Khóa địa chỉ Nominatim có thể chứa phường/xã
_WARD_KEYS = ("suburb", "quarter", "city_district", "village", "town", "municipality")

CACHE_PATH: Path = settings.paths.geocode_cache
_cache: dict[str, dict[str, Any]] | None = None
_lock = threading.Lock()
_last_request = 0.0


class GeocodeInput(BaseModel):
    """Input của geocode_address."""

    address: str = Field(
        min_length=1,
        description="Địa chỉ / địa danh ở TP.HCM, giữ nguyên cách người dùng gọi (vd 'Quận 7', 'phường Tân Thuận', "
        "'chợ Bến Thành'). Address or place name in Ho Chi Minh City.",
    )


class GeocodeResult(BaseModel):
    """Output thành công của geocode_address."""

    lat: float
    lng: float
    display_name: str = Field(description="Tên đầy đủ theo OSM (đơn vị hành chính mới)")
    ward: str | None = Field(description="Phường/xã/đặc khu hiện hành (sau 1/7/2025) chứa điểm này")
    query: str = Field(description="Chuỗi đã gửi tới Nominatim sau chuẩn hóa")
    source: str = SOURCE


def _strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def normalize_address(address: str) -> str:
    """Chuẩn hóa NFC, khoảng trắng, viết tắt quận/phường; thêm ", Thành phố Hồ Chí Minh" nếu chưa có."""
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFC", address)).strip(" ,.")
    text = re.sub(r"(?i)\bq\.?\s*(\d{1,2})\b", r"Quận \1", text)  # Q7, Q.7, q 7
    text = re.sub(r"(?i)\bp\.\s*(?=\D)", "Phường ", text)  # P. Tân Thuận (số phường cũ "P.5" giữ nguyên)
    if not any(m in _strip_accents(text).lower() for m in _CITY_MARKERS):
        text += settings.geo.city_suffix
    return text


def extract_ward(address: dict[str, str]) -> str | None:
    """Tìm phường/xã/đặc khu trong `address` của Nominatim."""
    for key in _WARD_KEYS:
        value = address.get(key, "")
        if value.startswith(_WARD_PREFIXES):
            return value
    return None


def _throttle() -> None:
    """Chính sách Nominatim: tối đa 1 request/giây."""
    global _last_request
    with _lock:
        wait = settings.api.nominatim_min_interval_s - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


def _nominatim(url: str, params: dict[str, Any]) -> Any:
    _throttle()
    headers = {"User-Agent": settings.api.nominatim_user_agent, "Accept-Language": "vi"}
    return _http.get_json(url, {**params, "format": "jsonv2", "addressdetails": 1}, headers)


def _get_cache() -> dict[str, dict[str, Any]]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _cache = {}
    return _cache


def _save_cache(key: str, value: dict[str, Any]) -> None:
    cache = _get_cache()
    cache[key] = value
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass  # cache chỉ để tiết kiệm request, ghi lỗi không ảnh hưởng kết quả


def _reverse_ward(lat: float, lng: float) -> str | None:
    """Reverse geocode để lấy phường/xã mới; lỗi API → None (không chặn kết quả chính)."""
    try:
        data = _nominatim(
            settings.api.nominatim_reverse_url,
            {"lat": lat, "lon": lng, "zoom": settings.api.nominatim_reverse_zoom},
        )
    except _http.APIError:
        return None
    return extract_ward(data.get("address", {})) if isinstance(data, dict) else None


def geocode(address: str) -> dict[str, Any]:
    """Geocode 1 địa chỉ ở TP.HCM cũ. Trả `GeocodeResult` hoặc `ToolError` dưới dạng dict."""
    if not address or not address.strip():
        return ToolError(error="invalid_input", message="Địa chỉ rỗng.").model_dump()
    query = normalize_address(address)
    key = query.lower()
    if key in _get_cache():
        return _get_cache()[key]

    try:
        results = _nominatim(
            settings.api.nominatim_url,
            {"q": query, "limit": settings.api.nominatim_limit, "countrycodes": "vn"},
        )
    except _http.APIError as e:
        return ToolError(error="api_error", message=f"Không gọi được Nominatim: {e}").model_dump()

    if not results:
        return ToolError(error="not_found", message=f"Không tìm thấy địa điểm '{address}' trên OpenStreetMap.").model_dump()

    for r in results:
        lat, lng = float(r["lat"]), float(r["lon"])
        if not in_hcmc_old(lat, lng):
            continue
        ward = extract_ward(r.get("address", {})) or _reverse_ward(lat, lng)
        out = GeocodeResult(lat=lat, lng=lng, display_name=r["display_name"], ward=ward, query=query).model_dump()
        _save_cache(key, out)
        return out

    first = results[0]
    out = ToolError(
        error="out_of_scope",
        message=f"'{first['display_name']}' ({float(first['lat']):.4f}, {float(first['lon']):.4f}) nằm ngoài "
        "TP.HCM cũ (ranh giới trước 1/7/2025) — ngoài phạm vi hệ thống.",
    ).model_dump()
    _save_cache(key, out)
    return out


@tool("geocode_address", args_schema=GeocodeInput)
def geocode_address(address: str) -> dict[str, Any]:
    """Tìm tọa độ (lat, lng) và phường/xã hiện hành của một địa chỉ / địa danh ở TP.HCM.

    Dùng khi người dùng nhắc tới một địa điểm (quận/phường cũ hoặc mới, đường, chợ, trường...) và cần
    tọa độ để gọi get_air_quality. Không tự đoán tọa độ. Phạm vi: TP.HCM cũ (trước 1/7/2025, không gồm
    Bình Dương, Bà Rịa–Vũng Tàu cũ); ngoài phạm vi trả lỗi "out_of_scope".

    Use when the user mentions a place in Ho Chi Minh City and coordinates are needed for get_air_quality.
    Returns {lat, lng, display_name, ward, query, source} or {error, message}.
    """
    return geocode(address)
