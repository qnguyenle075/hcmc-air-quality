"""Chunking corpus markdown theo heading/điều khoản, giữ bảng nguyên vẹn, gắn metadata.

Chiến lược:
1. Đọc front matter YAML của từng file .md (doc_id, doc_title, language).
2. Cắt theo heading markdown (#..####) → mỗi mục (section) có đường dẫn heading đầy đủ.
3. Trong mỗi mục, tách thành khối: khối bảng (dòng bắt đầu bằng "|", kèm dòng tiêu đề "Bảng .../Table ...")
   và khối văn bản.
4. Gộp các khối liên tiếp vào chunk cho tới khi chạm chunk_size (đếm token bằng tokenizer của model embedding).
   - Khối văn bản quá dài → RecursiveCharacterTextSplitter (có overlap).
   - Bảng không bao giờ bị cắt giữa chừng. Bảng dài hơn chunk_size được chia theo nhóm dòng,
     MỖI phần đều lặp lại tiêu đề bảng + hàng header để tự đứng được.
5. Mỗi chunk được thêm dòng ngữ cảnh "doc_title > section" ở đầu để retrieval biết chunk thuộc đâu.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config.settings import settings

_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
_TABLE_LINE_RE = re.compile(r"^\s*\|")
_CAPTION_RE = re.compile(r"^\s*(Bảng|Table)\s+[\d.]+", re.IGNORECASE)


@dataclass
class Block:
    """Một khối nội dung trong 1 mục: văn bản thường hoặc bảng."""

    text: str
    is_table: bool


@lru_cache(maxsize=1)
def _tokenizer():
    """Tokenizer của model embedding (chỉ tải tokenizer, không tải trọng số)."""
    return AutoTokenizer.from_pretrained(settings.rag.embedding_model)


def count_tokens(text: str) -> int:
    """Đếm token theo tokenizer của model embedding."""
    return len(_tokenizer().encode(text, add_special_tokens=False))


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Tách front matter YAML (giữa 2 dòng '---') khỏi thân tài liệu."""
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        return yaml.safe_load(fm) or {}, body.lstrip("\n")
    return {}, raw


def split_sections(body: str) -> list[tuple[list[str], str]]:
    """Cắt thân tài liệu theo heading → [(đường dẫn heading, nội dung mục)]."""
    sections: list[tuple[list[str], str]] = []
    path: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        content = "\n".join(buf).strip()
        if content:
            sections.append((path.copy(), content))
        buf.clear()

    for line in body.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            level = len(m.group(1))
            path[:] = path[: level - 1] + [m.group(2).strip("* ")]
        else:
            buf.append(line)
    flush()
    return sections


def split_blocks(content: str) -> list[Block]:
    """Tách nội dung 1 mục thành các khối bảng / văn bản (theo đoạn trống)."""
    blocks: list[Block] = []
    lines = content.splitlines()
    i = 0
    para: list[str] = []

    def flush_para() -> None:
        text = "\n".join(para).strip()
        if text:
            blocks.append(Block(text, is_table=False))
        para.clear()

    while i < len(lines):
        line = lines[i]
        if _TABLE_LINE_RE.match(line):
            # Kéo dòng tiêu đề bảng ("Bảng 1: ...") ngay phía trên vào cùng khối bảng.
            # Tiêu đề có thể nằm trong đoạn đang gom, hoặc là khối riêng (cách bảng 1 dòng trống).
            caption = ""
            if para and _CAPTION_RE.match(para[-1]):
                caption = para.pop()
            flush_para()
            if (
                not caption
                and blocks
                and not blocks[-1].is_table
                and "\n" not in blocks[-1].text
                and _CAPTION_RE.match(blocks[-1].text)
            ):
                caption = blocks.pop().text
            rows = []
            while i < len(lines) and _TABLE_LINE_RE.match(lines[i]):
                rows.append(lines[i].rstrip())
                i += 1
            blocks.append(Block("\n".join(([caption, ""] if caption else []) + rows), is_table=True))
            continue
        if not line.strip():
            flush_para()
        else:
            para.append(line)
        i += 1
    flush_para()
    return blocks


def split_large_table(table: str, max_tokens: int) -> list[str]:
    """Chia bảng dài theo nhóm dòng; mỗi phần lặp lại tiêu đề + header + dòng phân cách."""
    lines = table.splitlines()
    first_row = next(i for i, ln in enumerate(lines) if _TABLE_LINE_RE.match(ln))
    head = lines[: first_row + 2]  # caption (nếu có) + header + |---|
    rows = lines[first_row + 2 :]
    parts: list[str] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join(head + current + [row])
        if current and count_tokens(candidate) > max_tokens:
            parts.append("\n".join(head + current))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join(head + current))
    return parts


def _text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _tokenizer(),
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )


def chunk_section(content: str, max_tokens: int) -> list[tuple[str, bool]]:
    """Gộp các khối của 1 mục thành chunk ≤ max_tokens. Trả về [(text, has_table)]."""
    units: list[Block] = []
    for block in split_blocks(content):
        n = count_tokens(block.text)
        if n <= max_tokens:
            units.append(block)
        elif block.is_table:
            units.extend(Block(t, True) for t in split_large_table(block.text, max_tokens))
        else:
            units.extend(Block(t, False) for t in _text_splitter().split_text(block.text))

    chunks: list[tuple[str, bool]] = []
    buf: list[Block] = []
    for unit in units:
        candidate = "\n\n".join(b.text for b in buf + [unit])
        if buf and count_tokens(candidate) > max_tokens:
            chunks.append(("\n\n".join(b.text for b in buf), any(b.is_table for b in buf)))
            buf = []
        buf.append(unit)
    if buf:
        chunks.append(("\n\n".join(b.text for b in buf), any(b.is_table for b in buf)))
    return chunks


def chunk_document(path: Path) -> list[Document]:
    """Chunk 1 file corpus .md thành list Document có metadata đầy đủ."""
    meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
    doc_id = meta.get("doc_id", path.stem)
    doc_title = meta.get("doc_title", path.stem)
    language = meta.get("language", "vi")
    # Chừa chỗ cho dòng ngữ cảnh ở đầu chunk
    max_tokens = settings.rag.chunk_size

    docs: list[Document] = []
    for section_path, content in split_sections(body):
        section = " > ".join(section_path[1:] or section_path)  # bỏ heading H1 (tên tài liệu)
        for text, has_table in chunk_section(content, max_tokens):
            header = f"[{doc_title}] {section}".strip()
            docs.append(
                Document(
                    page_content=f"{header}\n\n{text}",
                    metadata={
                        "source": doc_id,
                        "doc_title": doc_title,
                        "section": section,
                        "language": language,
                        "has_table": has_table,
                        "source_file": str(path.relative_to(settings.paths.root).as_posix()),
                    },
                )
            )
    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = f"{doc_id}-{i:03d}"
    return docs


def corpus_files() -> list[Path]:
    """Danh sách file .md trong corpus (theo settings), đã loại file ghi chú."""
    files: list[Path] = []
    for d in settings.rag.corpus_dirs:
        files += sorted(
            p for p in (settings.paths.root / d).glob("*.md") if p.name not in settings.rag.corpus_exclude
        )
    return files


def load_corpus_chunks() -> list[Document]:
    """Chunk toàn bộ corpus. Hàm tất định: cùng input → cùng chunk (dùng chung cho dense và BM25)."""
    return [doc for path in corpus_files() for doc in chunk_document(path)]


if __name__ == "__main__":
    chunks = load_corpus_chunks()
    for c in chunks:
        m = c.metadata
        print(f"{m['chunk_id']:<22} {count_tokens(c.page_content):>4} tok  table={m['has_table']!s:<5}  {m['section'][:70]}")
    print(f"\nTổng: {len(chunks)} chunk")
