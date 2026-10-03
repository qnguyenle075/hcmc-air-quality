"""Làm sạch text trích xuất từ PDF: chuẩn hóa unicode, sửa ký hiệu, bỏ header/footer, nối dòng."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ingest.extract import PageText

MICRO = "µ"  # µ (micro sign) — chuẩn chung cho toàn corpus

# Dòng chỉ chứa số trang
_PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,3}\s*$")
# Dòng kết thúc câu / mục → không nối với dòng sau
_SENTENCE_END_RE = re.compile(r"[.:;!?]\s*$")
# Dòng tiêu đề mục đánh số ngắn, vd "1.2. Đối tượng áp dụng" → không nối dòng sau vào
_SHORT_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)+\.?\s+\S.{0,70}$")
# Dòng bắt đầu một mục mới → không nối vào dòng trước
_NEW_ITEM_RE = re.compile(r"^\s*(?:[-•■]|\d+(?:\.\d+)*\.?\s|[a-zđ]\.\s|Bảng\s|Table\s|Ghi chú|Điều\s)")


def normalize_unicode(text: str) -> str:
    """Chuẩn hóa NFC (bắt buộc cho tiếng Việt) và thống nhất ký hiệu micro thành µ."""
    text = unicodedata.normalize("NFC", text)
    return text.replace("μ", MICRO)  # μ (Greek mu) → µ


def fix_symbol_font(text: str) -> str:
    """Đổi ký tự private-use của font Symbol về unicode chuẩn.

    QCVN 05:2023 dùng font Symbol: chữ "m" của Symbol (µ) bị trích xuất thành U+F06D,
    ví dụ "Đơn vị: \\uf06dg/Nm3", "bằng 10 \\uf06dm" (đã kiểm tra bằng mắt với PDF).
    """
    return text.replace("\uf06d", MICRO)


def remove_headers_footers(text: str, header_patterns: tuple[str, ...], edge_lines: int = 3) -> str:
    """Bỏ header/footer lặp lại theo regex và số trang.

    Số trang chỉ bị bỏ khi nằm trong `edge_lines` dòng đầu/cuối trang, để không xóa nhầm
    các ô số đứng riêng một dòng trong bảng (vd "350", "125").
    """
    patterns = [re.compile(p) for p in header_patterns]
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    n = len(lines)
    kept = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        at_edge = idx < edge_lines or idx >= n - edge_lines
        if at_edge and _PAGE_NUMBER_RE.match(stripped):
            continue
        if any(p.match(stripped) for p in patterns):
            continue
        kept.append(line)
    return "\n".join(kept)


def join_broken_lines(text: str) -> str:
    """Nối các dòng bị ngắt giữa câu; giữ nguyên dòng trống và đầu mục mới."""
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if out and out[-1] != "":
                out.append("")
            continue
        prev = out[-1] if out else ""
        prev_is_heading = bool(_SHORT_HEADING_RE.match(prev)) and stripped[:1].isupper()
        if (
            prev
            and not _SENTENCE_END_RE.search(prev)
            and not prev_is_heading
            and not _NEW_ITEM_RE.match(stripped)
        ):
            out[-1] = f"{out[-1]} {stripped}"
        else:
            out.append(stripped)
    return re.sub(r" {2,}", " ", "\n".join(out)).strip()


def clean_text(text: str, header_patterns: tuple[str, ...] = ()) -> str:
    """Pipeline làm sạch cho 1 đoạn text."""
    text = normalize_unicode(text)
    text = fix_symbol_font(text)
    text = remove_headers_footers(text, header_patterns)
    return join_broken_lines(text)


def clean_pages(pages: list[PageText], header_patterns: tuple[str, ...] = ()) -> str:
    """Làm sạch từng trang rồi ghép lại, giữ marker trang để đối chiếu với PDF gốc."""
    parts = [f"<!-- page {p.page_no} -->\n{clean_text(p.text, header_patterns)}" for p in pages]
    return "\n\n".join(parts) + "\n"
