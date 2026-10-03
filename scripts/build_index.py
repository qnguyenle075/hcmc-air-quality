"""Build index: corpus .md → chunk → embed (bge-m3) → Chroma.

Chạy: uv run python scripts/build_index.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings  # noqa: E402
from src.ingest.chunk import load_corpus_chunks  # noqa: E402
from src.rag.vectorstore import build_vectorstore  # noqa: E402


def main() -> None:
    """Chunk toàn bộ corpus và index vào Chroma."""
    t0 = time.perf_counter()
    chunks = load_corpus_chunks()
    print(f"Chunk: {len(chunks)} | theo nguồn: {dict(Counter(c.metadata['source'] for c in chunks))}")
    print(f"Config: chunk_size={settings.rag.chunk_size}, overlap={settings.rag.chunk_overlap}, "
          f"embedding={settings.rag.embedding_model}, device={settings.rag.device}")
    store = build_vectorstore(chunks)
    print(f"Đã index {store._collection.count()} chunk vào {settings.paths.chroma} "  # noqa: SLF001
          f"({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
