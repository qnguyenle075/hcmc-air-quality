"""Tính chỉ số VN_AQI theo QĐ 1459/QĐ-TCMT (2019) — hàm thuần, không gọi API.

Nguồn: `data/processed/qd_1459_vn_aqi.md` (Bảng 1, Bảng 2, mục 2.2), đã đối chiếu ảnh trang PDF.
Đơn vị nồng độ: µg/m³ cho mọi thông số (kể cả CO), đúng như Bảng 2 và đơn vị Open-Meteo trả về.

Quy ước cài đặt (văn bản gốc không nói rõ):
- Làm tròn: half-up (0,5 → lên), áp cho AQI thông số và AQI tổng hợp. Khớp các ví dụ ở mục 2.3,
  trừ AQI giờ NO2 (bản gốc ghi 60, tính lại theo công thức ra 59,35 → 59; xem data/processed/NOTES.md).
- Nồng độ ≥ mức cao nhất của Bảng 2 (mức 8, ghi "≥") → AQI thông số = 500 (không ngoại suy).
- Nowcast: dùng công thức chuẩn hóa Σ w^(i−1)·ci / Σ w^(i−1) cho cả trường hợp w = 1/2. Bản gốc viết
  trường hợp w = 1/2 là Σ (1/2)^i·ci (không chia), tương đương gần đúng vì Σ(i=1→12) (1/2)^(i−1) ≈ 2.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# Bảng 2 — giá trị Ii của từng mức i = 1..8
AQI_LEVELS: tuple[int, ...] = (0, 50, 100, 150, 200, 300, 400, 500)

# Bảng 2 — BPi (µg/m³) theo mức i = 1..8. None = văn bản không quy định mức đó ("-").
BREAKPOINTS: dict[str, tuple[float | None, ...]] = {
    "o3_1h": (0, 160, 200, 300, 400, 800, 1000, 1200),
    "o3_8h": (0, 100, 120, 170, 210, 400, None, None),
    "co": (0, 10000, 30000, 45000, 60000, 90000, 120000, 150000),
    "so2": (0, 125, 350, 550, 800, 1600, 2100, 2630),
    "no2": (0, 100, 200, 700, 1200, 2350, 3100, 3850),
    "pm10": (0, 50, 150, 250, 350, 420, 500, 600),
    "pm25": (0, 25, 50, 80, 150, 250, 350, 500),
}

# Thứ tự thông số (dùng khi trùng AQI thông số để chọn thông số chính)
POLLUTANTS: tuple[str, ...] = ("pm25", "pm10", "o3", "no2", "so2", "co")
PM_POLLUTANTS: tuple[str, ...] = ("pm25", "pm10")

# Mục 2.2.2.b: không tính AQI O3 theo TB 8 giờ khi TB 8 giờ lớn nhất trong ngày > 400 µg/m³
O3_8H_MAX_VALID = 400.0

NOWCAST_HOURS = 12


@dataclass(frozen=True)
class AQICategory:
    """Một mức trong Bảng 1 (QĐ 1459)."""

    low: int
    high: int
    name: str
    color: str
    rgb: tuple[int, int, int]


# Bảng 1: Khoảng giá trị AQI và đánh giá chất lượng không khí
CATEGORIES: tuple[AQICategory, ...] = (
    AQICategory(0, 50, "Tốt", "Xanh", (0, 228, 0)),
    AQICategory(51, 100, "Trung bình", "Vàng", (255, 255, 0)),
    AQICategory(101, 150, "Kém", "Da cam", (255, 126, 0)),
    AQICategory(151, 200, "Xấu", "Đỏ", (255, 0, 0)),
    AQICategory(201, 300, "Rất xấu", "Tím", (143, 63, 151)),
    AQICategory(301, 500, "Nguy hại", "Nâu", (126, 0, 35)),
)


@dataclass
class AQIResult:
    """Kết quả tính VN_AQI (giờ hoặc ngày)."""

    aqi: int | None  # None nếu không có AQI của PM2.5 lẫn PM10 (mục 2.1.b)
    sub_indices: dict[str, int | None] = field(default_factory=dict)
    dominant_pollutant: str | None = None
    inputs: dict[str, float | None] = field(default_factory=dict)  # Cx / Nowcast đã dùng


def round_half_up(x: float) -> int:
    """Làm tròn half-up; làm tròn trước 6 chữ số để tránh sai số dấu phẩy động (109,4999999 → 110)."""
    return math.floor(round(x, 6) + 0.5)


def _is_missing(c: float | None) -> bool:
    return c is None or (isinstance(c, float) and math.isnan(c))


def sub_index(param: str, conc: float | None) -> float | None:
    """AQI thông số (chưa làm tròn) theo Công thức 1 (nội suy tuyến tính trên Bảng 2).

    `param` là khóa của BREAKPOINTS ("pm25", "pm10", "o3_1h", "o3_8h", "no2", "so2", "co").
    Với PM ở AQI giờ, truyền giá trị Nowcast (Công thức 2 có cùng dạng). Trả None nếu thiếu dữ liệu.
    """
    if _is_missing(conc) or conc < 0:
        return None
    if param == "o3_8h" and conc > O3_8H_MAX_VALID:
        return None
    bps = BREAKPOINTS[param]
    for i in range(len(bps) - 1):
        lo, hi = bps[i], bps[i + 1]
        if hi is None:
            break
        if conc <= hi:
            i_lo, i_hi = AQI_LEVELS[i], AQI_LEVELS[i + 1]
            return (i_hi - i_lo) / (hi - lo) * (conc - lo) + i_lo
    # Vượt mức cao nhất của Bảng 2 ("≥" ở mức 8) → giới hạn ở 500
    return float(AQI_LEVELS[-1])


def nowcast(values: Sequence[float | None]) -> float | None:
    """Giá trị Nowcast (mục 2.2.1.a) từ chuỗi giờ, phần tử đầu là c1 = giờ hiện tại.

    Chỉ dùng 12 giá trị đầu. Cần ít nhất 2 trong 3 giá trị c1, c2, c3; ci thiếu → trọng số 0.
    """
    c = list(values[:NOWCAST_HOURS])
    if sum(not _is_missing(x) for x in c[:3]) < 2:
        return None
    avail = [x for x in c if not _is_missing(x)]
    c_min, c_max = min(avail), max(avail)
    if c_max <= 0:
        return 0.0
    w_star = c_min / c_max
    w = 0.5 if w_star <= 0.5 else w_star
    num = den = 0.0
    for i, x in enumerate(c):
        if _is_missing(x):
            continue
        num += w**i * x
        den += w**i
    return num / den


def max_rolling_mean(values: Sequence[float | None], window: int = 8) -> float | None:
    """Giá trị trung bình `window` giờ lớn nhất (mục 2.2.2.a), bỏ qua cửa sổ có giờ thiếu dữ liệu."""
    means = [
        sum(values[i - window + 1 : i + 1]) / window
        for i in range(window - 1, len(values))
        if not any(_is_missing(x) for x in values[i - window + 1 : i + 1])
    ]
    return max(means) if means else None


def _combine(sub: dict[str, float | None], inputs: dict[str, float | None]) -> AQIResult:
    """AQI tổng hợp = max các AQI thông số (đã làm tròn). Bắt buộc có PM2.5 hoặc PM10."""
    rounded = {p: (None if v is None else round_half_up(v)) for p, v in sub.items()}
    if all(rounded.get(p) is None for p in PM_POLLUTANTS):
        return AQIResult(aqi=None, sub_indices=rounded, inputs=inputs)
    valid = {p: v for p, v in rounded.items() if v is not None}
    # max() giữ phần tử đầu khi trùng → ưu tiên theo thứ tự POLLUTANTS
    dominant = max((p for p in POLLUTANTS if p in valid), key=lambda p: valid[p])
    return AQIResult(aqi=valid[dominant], sub_indices=rounded, dominant_pollutant=dominant, inputs=inputs)


def hourly_aqi(hourly: Mapping[str, Sequence[float | None]]) -> AQIResult:
    """AQI giờ (mục 2.2.1) từ chuỗi TB 1 giờ theo thời gian tăng dần (phần tử cuối = giờ hiện tại).

    Khóa: "pm25", "pm10", "o3", "no2", "so2", "co" (thiếu khóa = không có dữ liệu thông số đó).
    PM dùng Nowcast 12 giờ gần nhất; O3 dùng BPi O3 (1h); SO2, CO, NO2 dùng giá trị giờ hiện tại.
    """
    inputs: dict[str, float | None] = {}
    sub: dict[str, float | None] = {}
    for p in POLLUTANTS:
        series = list(hourly.get(p) or [])
        if p in PM_POLLUTANTS:
            cx = nowcast(series[::-1]) if series else None
        else:
            cx = series[-1] if series else None
            cx = None if _is_missing(cx) else cx
        inputs[p] = cx
        sub[p] = sub_index("o3_1h" if p == "o3" else p, cx)
    return _combine(sub, inputs)


def daily_aqi(
    *,
    pm25_24h: float | None = None,
    pm10_24h: float | None = None,
    o3_1h_max: float | None = None,
    o3_8h_max: float | None = None,
    no2_1h_max: float | None = None,
    so2_1h_max: float | None = None,
    co_1h_max: float | None = None,
) -> AQIResult:
    """AQI ngày (mục 2.2.2) từ các giá trị đã tổng hợp trong ngày.

    O3: lấy AQI lớn hơn giữa TB 1 giờ lớn nhất (BPi O3 1h) và TB 8 giờ lớn nhất (BPi O3 8h);
    bỏ phần 8 giờ khi TB 8 giờ lớn nhất > 400 µg/m³.
    """
    o3_parts = [v for v in (sub_index("o3_1h", o3_1h_max), sub_index("o3_8h", o3_8h_max)) if v is not None]
    sub = {
        "pm25": sub_index("pm25", pm25_24h),
        "pm10": sub_index("pm10", pm10_24h),
        "o3": max(o3_parts) if o3_parts else None,
        "no2": sub_index("no2", no2_1h_max),
        "so2": sub_index("so2", so2_1h_max),
        "co": sub_index("co", co_1h_max),
    }
    inputs = {
        "pm25": pm25_24h,
        "pm10": pm10_24h,
        "o3_1h": o3_1h_max,
        "o3_8h": o3_8h_max,
        "no2": no2_1h_max,
        "so2": so2_1h_max,
        "co": co_1h_max,
    }
    return _combine(sub, inputs)


def category(aqi: int) -> AQICategory:
    """Mức chất lượng không khí theo Bảng 1. AQI > 500 xếp vào "Nguy hại"."""
    if aqi < 0:
        raise ValueError(f"AQI âm: {aqi}")
    for cat in CATEGORIES:
        if aqi <= cat.high:
            return cat
    return CATEGORIES[-1]
