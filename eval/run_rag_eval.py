"""Chạy RAGAS cho 1 variant RAG và ghi kết quả.

Chạy: uv run python -m eval.run_rag_eval --variant v0 [--limit N]

- Sinh câu trả lời cho toàn bộ testset bằng build_rag_chain(variant), đo latency từng câu.
- RAGAS (judge = JUDGE_MODEL) chấm các câu trong phạm vi: context_precision, context_recall,
  faithfulness, answer_relevancy.
- Câu out_of_scope: đo tỉ lệ từ chối đúng. Câu trong phạm vi: đo tỉ lệ từ chối nhầm.
- Lưu eval/results/<YYYYMMDD-HHMM>_<variant>.json và cập nhật eval/results/ablation.md.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

from eval import _ragas_compat  # noqa: F401  — phải import trước ragas

from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig  # noqa: E402
from langchain_core.callbacks import UsageMetadataCallbackHandler  # noqa: E402

from config.settings import settings  # noqa: E402
from src.rag.chain import build_rag_chain  # noqa: E402
from src.rag.embeddings import get_embeddings  # noqa: E402
from src.rag.prompts import NO_INFO_EN, NO_INFO_VI  # noqa: E402
from src.utils.llm import get_judge_llm  # noqa: E402

TESTSET = settings.paths.eval_datasets / "rag_testset.jsonl"
METRIC_KEYS = ("llm_context_precision_with_reference", "context_recall", "faithfulness", "answer_relevancy")
_REFUSAL_RE = re.compile(
    rf"{re.escape(NO_INFO_VI.rstrip('.'))}|{re.escape(NO_INFO_EN.rstrip('.'))}|"
    r"không có thông tin|do not contain information|does not contain information",
    re.IGNORECASE,
)

# Disclaimer y tế (mục 4.4) không phải nội dung trả lời → cắt trước khi chấm RAGAS, nếu không judge
# coi là claim không có trong context (đo 2026-10-03: q001 mất 2/4 claim faithfulness vì disclaimer).
# Cắt theo câu (không theo dòng) để không mất nội dung nếu disclaimer viết liền dòng với câu trả lời;
# bắt cả biến thể LLM tự viết lại ("…tham khảo…y tế…" / "…reference…medical advice…").
_DISCLAIMER_RE = re.compile(
    r"[*_]*[^.!?\n*_]*(?:tham khảo[^.!?\n]*y tế|reference[^.!?\n]*medical advice)[^.!?\n]*[.!?]?[*_]*",
    re.IGNORECASE,
)

# Cấu hình chạy RAGAS: judge đã có rate limiter (judge_requests_per_minute) → timeout mỗi job phải đủ dài để chờ lượt
RAGAS_RUN_CONFIG = dict(max_workers=4, max_retries=10, max_wait=60, timeout=600)


def is_refusal(answer: str) -> bool:
    """Câu trả lời có phải là từ chối 'tài liệu không có thông tin' không."""
    return bool(_REFUSAL_RE.search(answer))


def strip_disclaimer(answer: str) -> str:
    """Bỏ dòng disclaimer y tế khỏi câu trả lời (chỉ dùng cho bước chấm RAGAS)."""
    return _DISCLAIMER_RE.sub("", answer).strip()


def load_testset(limit: int | None = None) -> list[dict]:
    """Đọc testset JSONL."""
    rows = [json.loads(line) for line in TESTSET.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def generate_answers(variant: str, rows: list[dict], usage: UsageMetadataCallbackHandler) -> list[dict]:
    """Chạy RAG chain cho từng câu (tuần tự để đo latency chính xác)."""
    chain = build_rag_chain(variant)
    results = []
    for row in rows:
        t0 = time.perf_counter()
        out = chain.invoke({"question": row["question"]}, config={"callbacks": [usage]})
        latency = time.perf_counter() - t0
        results.append({
            **row,
            "answer": out["answer"],
            "answer_scored": strip_disclaimer(out["answer"]),  # bản đưa cho RAGAS chấm
            "retrieved_contexts": [d.page_content for d in out["contexts"]],
            "retrieved_chunk_ids": [d.metadata["chunk_id"] for d in out["contexts"]],
            "latency_s": round(latency, 3),
            "refused": is_refusal(out["answer"]),
        })
        print(f"  {row['id']} ({latency:.1f}s) refused={results[-1]['refused']}")
    return results


def run_ragas(results: list[dict], usage: UsageMetadataCallbackHandler) -> list[dict]:
    """Chấm RAGAS cho các câu trong phạm vi; trả về điểm từng câu."""
    in_scope = [r for r in results if r["group"] != "out_of_scope"]
    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer_scored"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["ground_truth"],
            reference_contexts=r["reference_contexts"],
        )
        for r in in_scope
    ])
    # bypass_n: Groq không hỗ trợ n > 1 → wrapper tự gọi n lần (cần cho ResponseRelevancy)
    judge = LangchainLLMWrapper(get_judge_llm(callbacks=[usage]), bypass_n=True)
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    scores = evaluate(
        dataset=dataset,
        metrics=[LLMContextPrecisionWithReference(), LLMContextRecall(), Faithfulness(), ResponseRelevancy()],
        llm=judge,
        embeddings=emb,
        run_config=RunConfig(**RAGAS_RUN_CONFIG),
        raise_exceptions=False,
    )
    df = scores.to_pandas()
    for r, (_, row) in zip(in_scope, df.iterrows()):
        r["ragas"] = {k: (None if _nan(row.get(k)) else float(row[k])) for k in METRIC_KEYS}
    return results


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _mean(values: list[float | None]) -> tuple[float | None, int]:
    vals = [v for v in values if v is not None]
    return (round(statistics.mean(vals), 4) if vals else None), len(values) - len(vals)


def summarize(results: list[dict]) -> dict:
    """Tổng hợp điểm trung bình + tỉ lệ từ chối + latency."""
    in_scope = [r for r in results if r["group"] != "out_of_scope"]
    oos = [r for r in results if r["group"] == "out_of_scope"]
    summary: dict = {}
    for k in METRIC_KEYS:
        mean, n_fail = _mean([r.get("ragas", {}).get(k) for r in in_scope])
        summary[k] = mean
        summary[f"{k}_failed"] = n_fail
    summary["oos_refusal_rate"] = round(sum(r["refused"] for r in oos) / len(oos), 4) if oos else None
    summary["false_refusal_rate"] = round(sum(r["refused"] for r in in_scope) / len(in_scope), 4) if in_scope else None
    summary["latency_mean_s"] = round(statistics.mean(r["latency_s"] for r in results), 3)
    summary["latency_p50_s"] = round(statistics.median(r["latency_s"] for r in results), 3)
    summary["by_group"] = {
        g: {k: _mean([r.get("ragas", {}).get(k) for r in in_scope if r["group"] == g])[0] for k in METRIC_KEYS}
        for g in sorted({r["group"] for r in in_scope})
    }
    return summary


def run_config_snapshot(variant: str) -> dict:
    """Cấu hình đầy đủ của lần chạy (để tái lập)."""
    rag = settings.rag
    return {
        "variant": variant,
        "chunk_size": rag.chunk_size, "chunk_overlap": rag.chunk_overlap, "top_k": rag.top_k,
        "hybrid_weights": rag.hybrid_weights, "multi_query_n": rag.multi_query_n,
        "rerank_fetch_k": rag.rerank_fetch_k, "rerank_top_n": rag.rerank_top_n,
        "embedding_model": rag.embedding_model, "reranker_model": rag.reranker_model,
        "generator_model": settings.llm.model,
        "judge_provider": settings.llm.judge_provider, "judge_model": settings.llm.judge_model,
        "judge_reasoning_effort": settings.llm.judge_reasoning_effort,
        "ragas_run_config": RAGAS_RUN_CONFIG,
    }


def update_ablation_md(variant: str, summary: dict, result_file: Path, config: dict, n: int) -> None:
    """Cập nhật dòng của variant trong bảng ablation + thêm 1 dòng lịch sử chạy."""
    path = settings.paths.eval_results / "ablation.md"
    labels = {"v0": "V0 dense", "v1": "V1 +hybrid", "v2": "V2 +multi-query", "v3": "V3 +rerank", "v4": "V4 +prompt"}
    if not path.exists():
        rows = "\n".join(f"| {labels[v]} | | | | | | | | |" for v in labels)
        path.write_text(
            "# Ablation RAG (V0 → V4)\n\n"
            "Metric RAGAS tính trên các câu trong phạm vi (không gồm out_of_scope). "
            "OOS refusal = tỉ lệ từ chối đúng ở nhóm out_of_scope; False refusal = tỉ lệ từ chối nhầm ở câu trong phạm vi.\n\n"
            "| Variant | Ctx Precision | Ctx Recall | Faithfulness | Answer Rel. | OOS refusal | False refusal | Latency (s) | Ghi chú |\n"
            "|---|---|---|---|---|---|---|---|---|\n"
            f"{rows}\n\n## Lịch sử chạy\n\n"
            "| Thời điểm | Variant | Số câu | Generator | Judge | Kết quả |\n|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )

    def fmt(x):
        return "" if x is None else f"{x:.3f}"

    s = summary
    new_row = (
        f"| {labels[variant]} | {fmt(s['llm_context_precision_with_reference'])} | {fmt(s['context_recall'])} | "
        f"{fmt(s['faithfulness'])} | {fmt(s['answer_relevancy'])} | {fmt(s['oos_refusal_rate'])} | "
        f"{fmt(s['false_refusal_rate'])} | {fmt(s['latency_mean_s'])} | {result_file.name} |"
    )
    text = path.read_text(encoding="utf-8")
    text = re.sub(rf"^\| {re.escape(labels[variant])} \|.*$", new_row, text, count=1, flags=re.MULTILINE)
    text = text.rstrip("\n") + (
        f"\n| {datetime.now():%Y-%m-%d %H:%M} | {variant} | {n} | {config['generator_model']} | "
        f"{config['judge_model']} | {result_file.name} |\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy RAGAS cho 1 variant RAG")
    parser.add_argument("--variant", default="v0")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu (debug)")
    args = parser.parse_args()

    rows = load_testset(args.limit)
    config = run_config_snapshot(args.variant)
    t_start = time.perf_counter()
    print(f"[1/2] Sinh câu trả lời ({args.variant}, {len(rows)} câu)...")
    gen_usage, judge_usage = UsageMetadataCallbackHandler(), UsageMetadataCallbackHandler()
    results = generate_answers(args.variant, rows, gen_usage)
    print("[2/2] Chấm RAGAS...")
    results = run_ragas(results, judge_usage)
    summary = summarize(results)
    elapsed = round(time.perf_counter() - t_start, 1)

    settings.paths.eval_results.mkdir(parents=True, exist_ok=True)
    out_file = settings.paths.eval_results / f"{datetime.now():%Y%m%d-%H%M}_{args.variant}.json"
    out_file.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config, "n_questions": len(rows), "runtime_s": elapsed,
        "summary": summary,
        "token_usage": {"generator": gen_usage.usage_metadata, "judge": judge_usage.usage_metadata},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.limit is None:
        update_ablation_md(args.variant, summary, out_file, config, len(rows))

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Token usage:", json.dumps({"generator": gen_usage.usage_metadata, "judge": judge_usage.usage_metadata}, default=str))
    print(f"Đã lưu {out_file} ({elapsed}s)")


if __name__ == "__main__":
    main()
