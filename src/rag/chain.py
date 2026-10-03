"""Dựng RAG chain theo variant (V0–V4) từ cùng một codebase để ablation tái lập được.

Output của chain: {"question", "answer", "contexts": list[Document], "sources": list[dict]}
— giữ contexts để RAGAS chấm context_precision / context_recall / faithfulness.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from src.rag.prompts import RAG_PROMPT_V0
from src.rag.retrievers import get_dense_retriever
from src.utils.llm import get_llm

VARIANTS = ("v0",)  # sẽ mở rộng v1..v4 theo từng bước


def format_context(docs: list[Document]) -> str:
    """Đánh số các đoạn context kèm nguồn + mục để LLM trích dẫn [n]."""
    return "\n\n".join(
        f"[{i}] (Nguồn: {d.metadata.get('doc_title')} — Mục: {d.metadata.get('section')})\n"
        f"{d.page_content.split(chr(10) * 2, 1)[-1]}"  # bỏ dòng header ngữ cảnh đã có trong "Nguồn"
        for i, d in enumerate(docs, start=1)
    )


def sources_of(docs: list[Document]) -> list[dict]:
    """Danh sách nguồn gọn để hiển thị / trả về từ tool."""
    return [
        {"ref": i, "source": d.metadata.get("source"), "section": d.metadata.get("section"),
         "chunk_id": d.metadata.get("chunk_id")}
        for i, d in enumerate(docs, start=1)
    ]


def build_rag_chain(variant: str = "v0") -> Runnable:
    """Tạo RAG chain cho 1 variant. Input: {"question": str}."""
    variant = variant.lower()
    if variant not in VARIANTS:
        raise NotImplementedError(f"Variant {variant!r} chưa được cài đặt (hiện có: {VARIANTS})")

    retriever = get_dense_retriever()
    prompt = RAG_PROMPT_V0
    llm = get_llm()

    generate = (
        RunnableLambda(lambda x: {"context": format_context(x["contexts"]), "question": x["question"]})
        | prompt
        | llm
        | StrOutputParser()
    )
    return (
        RunnablePassthrough.assign(contexts=lambda x: retriever.invoke(x["question"]))
        | RunnablePassthrough.assign(answer=generate)
        | RunnablePassthrough.assign(sources=lambda x: sources_of(x["contexts"]))
    )
