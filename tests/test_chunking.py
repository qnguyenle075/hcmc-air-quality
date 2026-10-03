"""Test chunking: không chunk rỗng, metadata đủ, bảng không bị cắt giữa chừng."""

from __future__ import annotations

import re

import pytest

from config.settings import settings
from src.ingest.chunk import (
    count_tokens,
    load_corpus_chunks,
    parse_front_matter,
    split_blocks,
    split_large_table,
    split_sections,
)

REQUIRED_METADATA = {"source", "doc_title", "section", "language", "chunk_id"}
_TABLE_ROW = re.compile(r"^\|")
_TABLE_SEP = re.compile(r"^\|(\s*:?-+:?\s*\|)+\s*$")


@pytest.fixture(scope="module")
def chunks():
    return load_corpus_chunks()


# ---------- Unit test trên dữ liệu nhỏ ----------


def test_parse_front_matter():
    meta, body = parse_front_matter("---\ndoc_id: x\nlanguage: vi\n---\n# Tiêu đề\nnội dung")
    assert meta == {"doc_id": "x", "language": "vi"}
    assert body.startswith("# Tiêu đề")


def test_split_sections_builds_heading_path():
    body = "# Doc\n## 1. A\ntext a\n### 1.1. B\ntext b\n## 2. C\ntext c"
    sections = split_sections(body)
    assert [p for p, _ in sections] == [["Doc", "1. A"], ["Doc", "1. A", "1.1. B"], ["Doc", "2. C"]]


def test_caption_attached_to_table_even_after_blank_line():
    content = "Đoạn mở đầu.\n\nBảng 5: Khuyến nghị\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\nGhi chú sau bảng."
    blocks = split_blocks(content)
    tables = [b for b in blocks if b.is_table]
    assert len(tables) == 1
    assert tables[0].text.startswith("Bảng 5: Khuyến nghị")
    assert not any(b.text.startswith("Bảng 5") for b in blocks if not b.is_table)


def test_split_large_table_repeats_caption_and_header():
    rows = "\n".join(f"| {i} | chất ô nhiễm số {i} với tên khá dài để tăng số token | 24 giờ |" for i in range(60))
    table = f"Bảng 2: Test\n\n| TT | Thông số | Thời gian |\n|---|---|---|\n{rows}"
    parts = split_large_table(table, max_tokens=200)
    assert len(parts) > 1
    for part in parts:
        lines = part.splitlines()
        assert lines[0] == "Bảng 2: Test"
        assert lines[2] == "| TT | Thông số | Thời gian |"
        assert _TABLE_SEP.match(lines[3])
    # Không mất / không lặp dòng dữ liệu
    data_rows = [ln for p in parts for ln in p.splitlines()[4:]]
    assert data_rows == rows.splitlines()


# ---------- Test trên corpus thật ----------


def test_corpus_not_empty(chunks):
    assert len(chunks) > 0
    assert {c.metadata["source"] for c in chunks} == {
        "who_aqg_2021",
        "qcvn_05_2023",
        "qd_1459_vn_aqi",
        "aqi_categories",
    }


def test_no_empty_chunks(chunks):
    for c in chunks:
        body = c.page_content.split("\n\n", 1)[-1]  # bỏ dòng header ngữ cảnh
        assert body.strip(), c.metadata["chunk_id"]


def test_metadata_complete(chunks):
    for c in chunks:
        missing = REQUIRED_METADATA - c.metadata.keys()
        assert not missing, f"{c.metadata.get('chunk_id')}: thiếu {missing}"
        assert all(str(c.metadata[k]).strip() for k in REQUIRED_METADATA)
        assert c.metadata["language"] in {"vi", "en"}


def test_chunk_ids_unique(chunks):
    ids = [c.metadata["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_size_within_limit(chunks):
    """Nội dung chunk (không tính header) ≤ chunk_size; cả chunk ≤ max_seq_length của embedding."""
    for c in chunks:
        body = c.page_content.split("\n\n", 1)[-1]
        assert count_tokens(body) <= settings.rag.chunk_size, c.metadata["chunk_id"]
        assert count_tokens(c.page_content) <= settings.rag.max_seq_length, c.metadata["chunk_id"]


def test_tables_not_cut(chunks):
    """Mỗi đoạn bảng trong chunk phải bắt đầu bằng hàng header + dòng phân cách |---|."""
    for c in chunks:
        lines = c.page_content.splitlines()
        for i, line in enumerate(lines):
            starts_table = _TABLE_ROW.match(line) and (i == 0 or not _TABLE_ROW.match(lines[i - 1]))
            if starts_table:
                assert i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]), (
                    f"{c.metadata['chunk_id']}: bảng bắt đầu không có header ở dòng {i}"
                )


def test_no_orphan_table_caption(chunks):
    """Không chunk nào kết thúc bằng dòng tiêu đề bảng (tiêu đề bị tách khỏi bảng)."""
    for c in chunks:
        last = [ln for ln in c.page_content.splitlines() if ln.strip()][-1]
        assert not re.match(r"^(Bảng|Table)\s+[\d.]+", last), c.metadata["chunk_id"]


def test_key_threshold_table_kept_whole(chunks):
    """Bảng 1 QCVN (7 thông số cơ bản) và Bảng 2 QĐ 1459 (breakpoint) nằm trọn trong 1 chunk."""
    qcvn = [c for c in chunks if c.metadata["source"] == "qcvn_05_2023" and "Bảng 1:" in c.page_content]
    assert len(qcvn) == 1
    assert all(p in qcvn[0].page_content for p in ("| 1 | SO2 |", "| 7 | Bụi PM2,5 |", "45(*)"))

    bp = [c for c in chunks if c.metadata["source"] == "qd_1459_vn_aqi" and "Bảng 2: Các giá trị BPi" in c.page_content]
    assert len(bp) == 1
    assert "| 1 | 0 |" in bp[0].page_content and "| 8 | 500 |" in bp[0].page_content
