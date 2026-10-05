"""Kiểm tra tọa độ có thuộc TP.HCM CŨ (trước sáp nhập 1/7/2025) hay không.

Hai bước: bbox (lọc nhanh) → point-in-polygon (ray casting, quy tắc chẵn–lẻ) trên ranh giới
`settings.geo.boundary_file` (geoBoundaries gbHumanitarian, Chính phủ VN qua HDX/OCHA, 2020).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from config.settings import settings

Ring = list[tuple[float, float]]  # (lng, lat) theo thứ tự GeoJSON


@lru_cache(maxsize=4)
def load_rings(path: Path | None = None) -> tuple[Ring, ...]:
    """Đọc mọi vòng (ngoài + lỗ) của Polygon/MultiPolygon trong file GeoJSON."""
    data = json.loads(Path(path or settings.geo.boundary_file).read_text(encoding="utf-8"))
    features = data["features"] if data.get("type") == "FeatureCollection" else [data]
    rings: list[Ring] = []
    for feat in features:
        geom = feat.get("geometry", feat)
        polygons = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polygons:
            rings.extend([(float(p[0]), float(p[1])) for p in ring] for ring in poly)
    return tuple(rings)


def _in_bbox(lat: float, lng: float) -> bool:
    lat_min, lat_max, lng_min, lng_max = settings.geo.hcmc_bbox
    return lat_min <= lat <= lat_max and lng_min <= lng <= lng_max


def point_in_rings(lat: float, lng: float, rings: tuple[Ring, ...]) -> bool:
    """Ray casting chẵn–lẻ trên mọi vòng → tự xử lý lỗ (hole) của polygon."""
    inside = False
    for ring in rings:
        n = len(ring)
        for i in range(n):
            x1, y1 = ring[i]
            x2, y2 = ring[(i + 1) % n]
            if (y1 > lat) != (y2 > lat) and lng < (x2 - x1) * (lat - y1) / (y2 - y1) + x1:
                inside = not inside
    return inside


def in_hcmc_old(lat: float, lng: float) -> bool:
    """True nếu (lat, lng) nằm trong ranh giới TP.HCM cũ."""
    return _in_bbox(lat, lng) and point_in_rings(lat, lng, load_rings())
