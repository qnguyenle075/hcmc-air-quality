"""Tool `get_air_quality`: nồng độ chất ô nhiễm theo tọa độ (Open-Meteo / CAMS) → VN_AQI giờ (QĐ 1459).

Open-Meteo trả chuỗi TB 1 giờ (µg/m³) của hôm qua + hôm nay; chỉ dùng các giờ ≤ thời điểm `current`
(bỏ giờ dự báo). PM2.5/PM10 dùng Nowcast 12 giờ; O3, NO2, SO2, CO dùng giá trị giờ hiện tại.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config.settings import settings
from src.tools import _http
from src.tools.boundary import in_hcmc_old
from src.tools.schemas import ToolError
from src.tools.vn_aqi import category, hourly_aqi

DATA_SOURCE = "Open-Meteo / CAMS (mô hình)"
METHOD = "VN_AQI giờ theo QĐ 1459/QĐ-TCMT (2019); PM2.5, PM10 dùng Nowcast 12 giờ"
NOTE = (
    "Số liệu mô hình CAMS (qua Open-Meteo), không phải số đo của trạm quan trắc. Độ phân giải thô nên các "
    "phường gần nhau có thể cho giá trị gần như giống nhau. VN_AQI được tính theo công thức QĐ 1459/QĐ-TCMT "
    "(vốn áp dụng cho dữ liệu trạm quan trắc tự động) — chỉ mang tính tham khảo."
)

# Tên biến Open-Meteo → khóa thông số trong vn_aqi
VAR_MAP: dict[str, str] = {
    "pm2_5": "pm25",
    "pm10": "pm10",
    "nitrogen_dioxide": "no2",
    "ozone": "o3",
    "sulphur_dioxide": "so2",
    "carbon_monoxide": "co",
}
_UG_M3 = {"µg/m³", "μg/m³"}  # micro (U+00B5) và mu Hy Lạp (U+03BC)


class AirQualityInput(BaseModel):
    """Input của get_air_quality."""

    lat: float = Field(ge=-90, le=90, description="Vĩ độ (lấy từ geocode_address) / latitude")
    lng: float = Field(ge=-180, le=180, description="Kinh độ (lấy từ geocode_address) / longitude")


class AirQualityResult(BaseModel):
    """Output thành công của get_air_quality."""

    vn_aqi: int = Field(description="VN_AQI giờ tổng hợp = max AQI thông số")
    category: str = Field(description="Mức chất lượng không khí (Bảng 1, QĐ 1459)")
    category_color: str
    dominant_pollutant: str = Field(description="Thông số có AQI cao nhất")
    sub_indices: dict[str, int | None] = Field(description="AQI giờ của từng thông số")
    pm25: float | None = Field(description="Nồng độ TB 1 giờ hiện tại, µg/m³")
    pm10: float | None
    no2: float | None
    o3: float | None
    so2: float | None
    co: float | None
    pm25_nowcast: float | None = Field(description="Nowcast PM2.5 dùng để tính AQI, µg/m³")
    pm10_nowcast: float | None
    unit: str = "µg/m³"
    measured_at: str = Field(description="Giờ của dữ liệu (giờ Việt Nam, ISO 8601)")
    data_source: str = DATA_SOURCE
    grid_lat: float = Field(description="Vĩ độ tâm ô lưới mô hình thực tế")
    grid_lng: float
    method: str = METHOD
    note: str = NOTE


def _fetch(lat: float, lng: float) -> dict[str, Any]:
    cfg = settings.api
    variables = ",".join(cfg.open_meteo_vars)
    return _http.get_json(
        cfg.open_meteo_aq_url,
        {
            "latitude": lat,
            "longitude": lng,
            "hourly": variables,
            "current": variables,
            "past_days": cfg.open_meteo_past_days,
            "forecast_days": cfg.open_meteo_forecast_days,
            "timezone": cfg.timezone,
        },
    )


def parse_response(data: dict[str, Any]) -> dict[str, Any]:
    """Response Open-Meteo → `AirQualityResult` hoặc `ToolError` (dict). Hàm thuần, test được với mock."""
    hourly = data.get("hourly") or {}
    times: list[str] = hourly.get("time") or []
    now = (data.get("current") or {}).get("time")
    if not times or not now:
        return ToolError(error="no_data", message="Open-Meteo không trả chuỗi dữ liệu giờ.").model_dump()

    units = data.get("hourly_units") or {}
    bad = {v: units.get(v) for v in VAR_MAP if v in units and units[v] not in _UG_M3}
    if bad:
        return ToolError(error="no_data", message=f"Đơn vị không phải µg/m³: {bad}").model_dump()

    # Chỉ dùng giờ ≤ hiện tại (chuỗi ISO cùng định dạng → so sánh chuỗi được)
    past = [i for i, t in enumerate(times) if t <= now]
    if not past:
        return ToolError(error="no_data", message="Không có giờ dữ liệu nào trước thời điểm hiện tại.").model_dump()
    idx = past[-1]
    series = {key: (hourly.get(var) or [])[: idx + 1] for var, key in VAR_MAP.items()}

    res = hourly_aqi(series)
    if res.aqi is None:
        return ToolError(
            error="no_data",
            message="Thiếu dữ liệu PM2.5 và PM10 nên không tính được VN_AQI (QĐ 1459 bắt buộc có ít nhất 1 trong 2).",
        ).model_dump()

    current = {key: (s[-1] if s else None) for key, s in series.items()}
    cat = category(res.aqi)
    return AirQualityResult(
        vn_aqi=res.aqi,
        category=cat.name,
        category_color=cat.color,
        dominant_pollutant=res.dominant_pollutant,
        sub_indices=res.sub_indices,
        **current,
        pm25_nowcast=None if res.inputs["pm25"] is None else round(res.inputs["pm25"], 1),
        pm10_nowcast=None if res.inputs["pm10"] is None else round(res.inputs["pm10"], 1),
        measured_at=f"{times[idx]}{_utc_offset(data)}",
        grid_lat=data["latitude"],
        grid_lng=data["longitude"],
    ).model_dump()


def _utc_offset(data: dict[str, Any]) -> str:
    """Hậu tố múi giờ ISO 8601 từ `utc_offset_seconds` (vd +07:00)."""
    sec = int(data.get("utc_offset_seconds", 0))
    sign = "+" if sec >= 0 else "-"
    h, m = divmod(abs(sec) // 60, 60)
    return f"{sign}{h:02d}:{m:02d}"


def air_quality(lat: float, lng: float) -> dict[str, Any]:
    """Lấy dữ liệu và tính VN_AQI cho 1 tọa độ trong TP.HCM cũ. Trả dict (kết quả hoặc lỗi)."""
    if not in_hcmc_old(lat, lng):
        return ToolError(
            error="out_of_scope",
            message=f"Tọa độ ({lat:.4f}, {lng:.4f}) nằm ngoài TP.HCM cũ (ranh giới trước 1/7/2025).",
        ).model_dump()
    try:
        data = _fetch(lat, lng)
    except _http.APIError as e:
        return ToolError(error="api_error", message=f"Không lấy được dữ liệu Open-Meteo: {e}").model_dump()
    return parse_response(data)


@tool("get_air_quality", args_schema=AirQualityInput)
def get_air_quality(lat: float, lng: float) -> dict[str, Any]:
    """Lấy chất lượng không khí hiện tại tại một tọa độ ở TP.HCM: VN_AQI giờ, mức (Tốt/Trung bình/Kém/Xấu/
    Rất xấu/Nguy hại), thông số chính và nồng độ PM2.5, PM10, NO2, O3, SO2, CO (µg/m³).

    Dùng sau geocode_address, khi người dùng hỏi không khí / AQI / bụi mịn tại một địa điểm. Không tự đoán
    chỉ số. Số liệu là mô hình CAMS (Open-Meteo), không phải trạm đo — phải nói rõ điều này khi trả lời.
    Tool này không đưa khuyến nghị sức khỏe: dùng retrieve_health_guideline cho phần đó.

    Use after geocode_address to get the current Vietnamese AQI (VN_AQI) and pollutant concentrations at a
    location in Ho Chi Minh City. Returns the result fields or {error, message}.
    """
    return air_quality(lat, lng)
