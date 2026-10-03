"""Trích xuất text từ PDF gốc (pymupdf) và bảng (pdfplumber).

Chạy: uv run python -m src.ingest.extract
→ ghi text đã làm sạch tự động vào data/processed/auto/<doc_id>.txt (có đánh dấu trang).
Bản .md cuối cùng trong data/processed/ được chỉnh tay từ output này (xem data/processed/NOTES.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
import pymupdf

from config.settings import settings
from src.ingest.clean import clean_pages


@dataclass(frozen=True)
class SourceDoc:
    """Thông tin 1 tài liệu nguồn trong corpus."""

    doc_id: str
    filename: str
    title: str
    language: str
    # Regex các dòng header/footer lặp lại cần bỏ (ngoài số trang đứng riêng)
    header_patterns: tuple[str, ...] = field(default_factory=tuple)
    # Trang (1-based) bỏ qua: bìa, trang trống, trang license, bìa sau...
    skip_pages: tuple[int, ...] = field(default_factory=tuple)


SOURCE_DOCS: tuple[SourceDoc, ...] = (
    SourceDoc(
        doc_id="who_aqg_2021",
        filename="who_aqg_2021_exec_summary.pdf",
        title="WHO global air quality guidelines 2021 — Executive summary",
        language="en",
        header_patterns=(r"^EXECUTIVE SUMMARY$", r"^WHO GLOBAL AIR QUALITY GUIDELINES$"),
        skip_pages=(1, 2, 3, 4, 15, 16),
    ),
    SourceDoc(
        doc_id="qcvn_05_2023",
        filename="qcvn_05_2023_btnmt.pdf",
        title="QCVN 05:2023/BTNMT — Quy chuẩn kỹ thuật quốc gia về chất lượng không khí",
        language="vi",
        header_patterns=(r"^QCVN 05:2023/BTNMT$", r"^QCVN …\. : 2022/BTNMT$"),
        skip_pages=(1,),
    ),
    SourceDoc(
        doc_id="qd_1459_vn_aqi",
        filename="qd_1459_tcmt_2019.pdf",
        title="Quyết định 1459/QĐ-TCMT — Hướng dẫn kỹ thuật tính toán và công bố VN_AQI",
        language="vi",
        header_patterns=(),
        skip_pages=(),
    ),
)


@dataclass(frozen=True)
class PageText:
    """Text của 1 trang PDF."""

    page_no: int  # 1-based
    text: str


def extract_pages(pdf_path: Path, skip_pages: tuple[int, ...] = ()) -> list[PageText]:
    """Lấy text từng trang bằng pymupdf, bỏ các trang trong skip_pages."""
    with pymupdf.open(pdf_path) as doc:
        return [
            PageText(page_no=i + 1, text=page.get_text())
            for i, page in enumerate(doc)
            if (i + 1) not in skip_pages
        ]


def extract_tables(pdf_path: Path, page_no: int) -> list[list[list[str | None]]]:
    """Lấy các bảng trên 1 trang bằng pdfplumber (dùng hỗ trợ dựng lại bảng bị vỡ)."""
    with pdfplumber.open(pdf_path) as pdf:
        return pdf.pages[page_no - 1].extract_tables()


def run(output_dir: Path | None = None) -> list[Path]:
    """Trích xuất + làm sạch tự động mọi tài liệu, ghi ra data/processed/auto/."""
    out_dir = output_dir or settings.paths.processed / "auto"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for src in SOURCE_DOCS:
        pages = extract_pages(settings.paths.raw / src.filename, src.skip_pages)
        cleaned = clean_pages(pages, src.header_patterns)
        out_path = out_dir / f"{src.doc_id}.txt"
        out_path.write_text(cleaned, encoding="utf-8")
        written.append(out_path)
        print(f"{src.doc_id}: {len(pages)} trang → {out_path.relative_to(settings.paths.root)}")
    return written


if __name__ == "__main__":
    run()
