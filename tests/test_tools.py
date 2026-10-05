"""Test Phase 2: VN_AQI (giá trị tính tay theo QĐ 1459), ranh giới TP.HCM cũ, geocode và air quality
với response giả. Integration test gọi API thật: `pytest -m integration`."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from src.tools import _http, air_quality, geocode
from src.tools.boundary import in_hcmc_old
from src.tools.vn_aqi import (
    category,
    daily_aqi,
    hourly_aqi,
    max_rolling_mean,
    nowcast,
    round_half_up,
    sub_index,
)

# ---------------------------------------------------------------- VN_AQI (ví dụ mục 2.3, QĐ 1459)

# PM2.5 TB 1 giờ 09:00 → 20:00 (mục 2.3.a); c1 = 20:00
PM25_EXAMPLE = [26.9, 24.7, 20.5, 23.5, 19.5, 16.5, 19.0, 16.5, 20.3, 22.4, 19.6, 20.6]


def test_nowcast_vi_du_van_ban() -> None:
    assert round(nowcast(PM25_EXAMPLE[::-1]), 1) == 20.3


def test_nowcast_w_nho_hon_mot_nua_dung_w_bang_0_5() -> None:
    # Cmin/Cmax = 0,1 ≤ 0,5 → w = 0,5: Σ 0,5^(i−1)·ci / Σ 0,5^(i−1)
    vals = [100.0, 10.0]
    assert nowcast(vals) == pytest.approx((100 + 0.5 * 10) / 1.5)


def test_nowcast_can_2_trong_3_gio_gan_nhat() -> None:
    assert nowcast([None, None, 10.0, 10.0]) is None
    assert nowcast([None, 10.0, 10.0]) == pytest.approx(10.0)


def test_nowcast_gio_thieu_co_trong_so_0() -> None:
    # c2 thiếu → bỏ khỏi cả tử và mẫu; còn c1 = c3 = 30 → Nowcast = 30
    assert nowcast([30.0, None, 30.0]) == pytest.approx(30.0)


def test_aqi_gio_thong_so_vi_du_van_ban() -> None:
    assert round_half_up(sub_index("o3_1h", 136.1)) == 43
    assert round_half_up(sub_index("pm25", 20.3)) == 41
    # Bản gốc ghi 60; công thức cho (100−50)/(200−100)×(118,7−100)+50 = 59,35 → 59
    # (sai số trong bản gốc, xem data/processed/NOTES.md)
    assert sub_index("no2", 118.7) == pytest.approx(59.35)
    assert round_half_up(sub_index("no2", 118.7)) == 59


def test_hourly_aqi_tong_hop() -> None:
    res = hourly_aqi({"o3": [136.1], "no2": [118.7], "pm25": PM25_EXAMPLE})
    assert res.sub_indices["o3"] == 43
    assert res.sub_indices["pm25"] == 41
    assert res.aqi == 59 and res.dominant_pollutant == "no2"
    assert res.sub_indices["co"] is None


def test_hourly_aqi_khong_co_pm_thi_khong_tinh() -> None:
    res = hourly_aqi({"o3": [136.1], "no2": [118.7]})
    assert res.aqi is None


def test_daily_aqi_vi_du_van_ban() -> None:
    res = daily_aqi(o3_8h_max=89.3, o3_1h_max=114.6, no2_1h_max=130.8, pm25_24h=55.7)
    assert round_half_up(sub_index("o3_8h", 89.3)) == 45
    assert round_half_up(sub_index("o3_1h", 114.6)) == 36
    assert res.sub_indices == {"pm25": 110, "pm10": None, "o3": 45, "no2": 65, "so2": None, "co": None}
    assert res.aqi == 110 and res.dominant_pollutant == "pm25"


def test_o3_8h_tren_400_khong_tinh() -> None:
    assert sub_index("o3_8h", 401) is None
    res = daily_aqi(pm25_24h=10, o3_8h_max=450, o3_1h_max=500)
    assert res.sub_indices["o3"] == round_half_up(sub_index("o3_1h", 500))  # chỉ dùng 1 giờ


def test_max_rolling_mean_vi_du_o3_van_ban() -> None:
    # O3 TB 1 giờ 18:00 → 0:00 hôm sau (mục 2.2.2.a); TB 8 giờ lớn nhất = 74,4
    o3 = [15.7, 14.2, 17.7, 18.9, 19.3, 15.7, 19.7, 22.6, 27.1, 29.0, 31.9, 25.3, 34.7, 35.2, 41.6, 45.7,
          49.2, 55.8, 69.6, 78.3, 91.5, 97.7, 81.2, 71.8, 43.5, 34.3, 21.5, 20.5, 19.4, 20.4, 21.3]
    assert round(max_rolling_mean(o3, 8), 1) == 74.4


def test_sub_index_bien_va_vuot_thang() -> None:
    assert sub_index("pm25", 25) == pytest.approx(50)
    assert sub_index("pm25", 500) == pytest.approx(500)
    assert sub_index("pm25", 900) == 500
    assert sub_index("pm25", -1) is None
    assert sub_index("pm25", None) is None
    assert sub_index("co", 10000) == pytest.approx(50)


def test_round_half_up() -> None:
    assert round_half_up(0.5) == 1
    assert round_half_up(42.5) == 43
    assert round_half_up(42.49) == 42


@pytest.mark.parametrize(
    ("aqi", "name"),
    [(0, "Tốt"), (50, "Tốt"), (51, "Trung bình"), (100, "Trung bình"), (101, "Kém"), (150, "Kém"),
     (151, "Xấu"), (200, "Xấu"), (201, "Rất xấu"), (300, "Rất xấu"), (301, "Nguy hại"), (500, "Nguy hại"),
     (501, "Nguy hại")],
)
def test_category_bang_1(aqi: int, name: str) -> None:
    assert category(aqi).name == name


# ---------------------------------------------------------------- Ranh giới TP.HCM cũ

INSIDE = {
    "Chợ Bến Thành": (10.7725, 106.6980),
    "Phường Tân Thuận (Q7 cũ)": (10.7525, 106.7284),
    "Phường Thủ Đức": (10.8506, 106.7585),
    "Phường Bình Tân": (10.7937, 106.5898),
    "Ấp Củ Chi": (10.9744, 106.4949),
    "Cần Thạnh (Cần Giờ)": (10.411, 106.954),
    "Nhà Bè": (10.695, 106.74),
}
OUTSIDE = {
    "Thủ Dầu Một": (10.9809, 106.6537),  # trong bbox nhưng ngoài polygon
    "Dĩ An": (10.906, 106.769),
    "Biên Hòa": (10.957, 106.843),
    "Vũng Tàu": (10.346, 107.084),
    "Bến Lức": (10.64, 106.49),
}


@pytest.mark.parametrize("name", INSIDE)
def test_trong_tphcm_cu(name: str) -> None:
    assert in_hcmc_old(*INSIDE[name])


@pytest.mark.parametrize("name", OUTSIDE)
def test_ngoai_tphcm_cu(name: str) -> None:
    assert not in_hcmc_old(*OUTSIDE[name])


# ---------------------------------------------------------------- Geocode (mock Nominatim)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Q7", "Quận 7, Thành phố Hồ Chí Minh"),
        ("q.10", "Quận 10, Thành phố Hồ Chí Minh"),
        ("P. Tân Thuận", "Phường Tân Thuận, Thành phố Hồ Chí Minh"),
        ("chợ Bến Thành, TP.HCM", "chợ Bến Thành, TP.HCM"),
        ("Ben Thanh market, Ho Chi Minh City", "Ben Thanh market, Ho Chi Minh City"),
        ("  Quận   1 ", "Quận 1, Thành phố Hồ Chí Minh"),
        ("QL1A, Bình Chánh", "QL1A, Bình Chánh, Thành phố Hồ Chí Minh"),
    ],
)
def test_normalize_address(raw: str, expected: str) -> None:
    assert geocode.normalize_address(raw) == expected


def _hit(lat: float, lng: float, name: str, address: dict[str, str] | None = None) -> dict[str, Any]:
    return {"lat": str(lat), "lon": str(lng), "display_name": name, "address": address or {}}


class FakeNominatim:
    """Thay `_http.get_json`: trả response theo endpoint, đếm số lần gọi."""

    def __init__(self, search: Any = None, reverse: Any = None, error: bool = False) -> None:
        self.search, self.reverse, self.error = search or [], reverse or {}, error
        self.calls: list[str] = []

    def __call__(self, url: str, params: dict[str, Any], headers: dict[str, str] | None = None) -> Any:
        self.calls.append(url)
        assert headers and headers["User-Agent"]
        if self.error:
            raise _http.APIError("timeout")
        return self.reverse if "reverse" in url else self.search


@pytest.fixture
def fake_geo(monkeypatch: pytest.MonkeyPatch, tmp_path: Any):
    """Cache tạm, bỏ giới hạn 1 req/s; trả hàm cài FakeNominatim."""
    monkeypatch.setattr(geocode, "CACHE_PATH", tmp_path / "geocode.json")
    monkeypatch.setattr(geocode, "_cache", None)
    monkeypatch.setattr(geocode, "_throttle", lambda: None)

    def install(fake: FakeNominatim) -> FakeNominatim:
        monkeypatch.setattr(_http, "get_json", fake)
        return fake

    return install


def test_geocode_chon_ket_qua_dau_tien_trong_tphcm_cu(fake_geo) -> None:
    fake_geo(FakeNominatim(search=[
        _hit(10.6310, 107.3816, "Ấp Bình Tân, Xã Hòa Hội", {"city_district": "Xã Hòa Hội"}),  # BR-VT cũ
        _hit(10.7937, 106.5898, "Phường Bình Tân, Thành phố Hồ Chí Minh", {"suburb": "Phường Bình Tân"}),
    ]))
    out = geocode.geocode("Bình Tân")
    assert out["ward"] == "Phường Bình Tân"
    assert out["lat"] == pytest.approx(10.7937)
    assert out["query"] == "Bình Tân, Thành phố Hồ Chí Minh"


def test_geocode_ngoai_pham_vi(fake_geo) -> None:
    fake_geo(FakeNominatim(search=[_hit(10.9809, 106.6537, "Phường Thủ Dầu Một, Thành phố Hồ Chí Minh")]))
    out = geocode.geocode("Thủ Dầu Một")
    assert out["error"] == "out_of_scope"
    assert "Thủ Dầu Một" in out["message"]


def test_geocode_khong_tim_thay(fake_geo) -> None:
    fake_geo(FakeNominatim(search=[]))
    assert geocode.geocode("xyz không tồn tại")["error"] == "not_found"


def test_geocode_loi_api_khong_raise(fake_geo) -> None:
    fake_geo(FakeNominatim(error=True))
    assert geocode.geocode("Quận 1")["error"] == "api_error"


def test_geocode_input_rong() -> None:
    assert geocode.geocode("   ")["error"] == "invalid_input"


def test_geocode_dia_danh_cu_reverse_lay_phuong_moi(fake_geo) -> None:
    fake = fake_geo(FakeNominatim(
        search=[_hit(10.7379, 106.7297, "Quận 7, Thành phố Hồ Chí Minh", {"historic": "Quận 7"})],
        reverse={"address": {"suburb": "Phường Tân Hưng"}},
    ))
    out = geocode.geocode("Q7")
    assert out["ward"] == "Phường Tân Hưng"
    assert len(fake.calls) == 2  # search + reverse


def test_geocode_cache_khong_goi_lai(fake_geo) -> None:
    fake = fake_geo(FakeNominatim(search=[_hit(10.7725, 106.6980, "Chợ Bến Thành", {"suburb": "Phường Bến Thành"})]))
    first = geocode.geocode("Chợ Bến Thành")
    second = geocode.geocode("chợ bến thành")
    assert first == second
    assert len(fake.calls) == 1
    assert geocode.CACHE_PATH.exists()


def test_geocode_tool_invoke(fake_geo) -> None:
    fake_geo(FakeNominatim(search=[_hit(10.7725, 106.6980, "Chợ Bến Thành", {"suburb": "Phường Bến Thành"})]))
    out = geocode.geocode_address.invoke({"address": "Chợ Bến Thành"})
    assert set(out) == {"lat", "lng", "display_name", "ward", "query", "source"}


# ---------------------------------------------------------------- Air quality (mock Open-Meteo)


def _om_response(now_idx: int = 23, **overrides: Any) -> dict[str, Any]:
    """Response giả 48 giờ (hôm qua + hôm nay); giờ sau `now_idx` là dự báo với giá trị rất cao."""
    times = [f"2026-10-0{4 + h // 24}T{h % 24:02d}:00" for h in range(48)]

    def series(value: float) -> list[float]:
        return [value if i <= now_idx else 999.0 for i in range(48)]

    hourly = {
        "time": times,
        "pm2_5": series(20.0),
        "pm10": series(30.0),
        "nitrogen_dioxide": series(118.7),
        "ozone": series(40.0),
        "sulphur_dioxide": series(10.0),
        "carbon_monoxide": series(500.0),
    }
    hourly.update(overrides)
    return {
        "latitude": 10.8,
        "longitude": 106.7,
        "utc_offset_seconds": 25200,
        "current": {"time": times[now_idx]},
        "hourly": hourly,
        "hourly_units": {v: "μg/m³" for v in air_quality.VAR_MAP},
    }


def test_parse_response_tinh_vn_aqi_bo_gio_du_bao() -> None:
    out = air_quality.parse_response(_om_response(now_idx=30))
    assert out["vn_aqi"] == 59  # NO2 118,7 → 59; PM2.5 = 20 → 40
    assert out["dominant_pollutant"] == "no2"
    assert out["category"] == "Trung bình"
    assert out["sub_indices"]["pm25"] == 40
    assert out["pm25"] == 20.0 and out["pm25_nowcast"] == 20.0
    assert out["measured_at"] == "2026-10-05T06:00+07:00"
    assert (out["grid_lat"], out["grid_lng"]) == (10.8, 106.7)
    assert out["data_source"] == "Open-Meteo / CAMS (mô hình)"
    assert "không phải số đo" in out["note"]


def test_parse_response_thieu_pm() -> None:
    out = air_quality.parse_response(_om_response(pm2_5=[None] * 48, pm10=[None] * 48))
    assert out["error"] == "no_data"


def test_parse_response_sai_don_vi() -> None:
    data = _om_response()
    data["hourly_units"]["carbon_monoxide"] = "mg/m³"
    assert air_quality.parse_response(data)["error"] == "no_data"


def test_parse_response_rong() -> None:
    assert air_quality.parse_response({})["error"] == "no_data"


def test_air_quality_ngoai_pham_vi_khong_goi_api(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("không được gọi API")

    monkeypatch.setattr(_http, "get_json", boom)
    out = air_quality.get_air_quality.invoke({"lat": 10.9809, "lng": 106.6537})  # Thủ Dầu Một
    assert out["error"] == "out_of_scope"


def test_air_quality_loi_api_khong_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*a: Any, **k: Any) -> Any:
        raise _http.APIError("timeout")

    monkeypatch.setattr(_http, "get_json", fail)
    assert air_quality.air_quality(10.7769, 106.7009)["error"] == "api_error"


# ---------------------------------------------------------------- HTTP retry


def test_get_json_retry_roi_thanh_cong(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [httpx.TimeoutException("t"), httpx.Response(503), httpx.Response(200, json={"ok": 1})]

    def fake_get(*a: Any, **k: Any) -> httpx.Response:
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(_http.time, "sleep", lambda s: None)
    assert _http.get_json("http://x", {}) == {"ok": 1}


def test_get_json_het_luot_thu(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_get(*a: Any, **k: Any) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectTimeout("t")

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(_http.time, "sleep", lambda s: None)
    with pytest.raises(_http.APIError):
        _http.get_json("http://x", {})
    assert len(calls) == 3  # 1 lần đầu + 2 lần thử lại


def test_get_json_loi_4xx_khong_thu_lai(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def fake_get(*a: Any, **k: Any) -> httpx.Response:
        calls.append(1)
        return httpx.Response(400, text="bad")

    monkeypatch.setattr(httpx, "get", fake_get)
    with pytest.raises(_http.APIError):
        _http.get_json("http://x", {})
    assert len(calls) == 1


# ---------------------------------------------------------------- Integration (API thật)


@pytest.mark.integration
@pytest.mark.parametrize(
    "address",
    ["Quận 1", "Q7", "phường Tân Thuận", "Thủ Đức", "Bình Tân", "Củ Chi", "Cần Giờ"],
)
def test_integration_geocode_trong_pham_vi(address: str) -> None:
    out = geocode.geocode(address)
    assert "error" not in out, out
    assert in_hcmc_old(out["lat"], out["lng"])


@pytest.mark.integration
def test_integration_geocode_thu_dau_mot_bi_tu_choi() -> None:
    assert geocode.geocode("Thủ Dầu Một")["error"] == "out_of_scope"


@pytest.mark.integration
def test_integration_air_quality_q1() -> None:
    out = air_quality.air_quality(10.7769, 106.7009)
    assert "error" not in out, out
    assert 0 <= out["vn_aqi"] <= 500
