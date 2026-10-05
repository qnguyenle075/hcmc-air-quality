"""Chạy RAGAS cho 1 variant RAG và ghi kết quả.

Chạy: uv run python -m eval.run_rag_eval --variant v0 [--split dev|test] [--limit N] [--resume]

- --split dev (mặc định): bộ dev 28 câu (rag_devset.jsonl) — dùng cho ablation V0–V4, phân tích lỗi, chỉnh prompt.
  --split test: bộ test 27 câu (rag_testset.jsonl) — chỉ đo một lần cuối Phase 1 cho V0 và variant tốt nhất;
  không chỉnh hệ thống theo kết quả bộ này.
- Sinh câu trả lời cho toàn bộ testset bằng build_rag_chain(variant), đo latency từng câu.
- RAGAS (judge = JUDGE_MODEL) chấm các câu trong phạm vi: context_precision, context_recall,
  faithfulness, answer_relevancy.
- Câu out_of_scope: đo tỉ lệ từ chối đúng. Câu trong phạm vi: đo tỉ lệ từ chối nhầm.
- Lưu eval/results/<YYYYMMDD-HHMM>_<variant>.json (bộ test: ..._<variant>_test.json) và cập nhật
  eval/results/ablation.md (bảng dev hoặc bảng test).
- Checkpoint: mỗi câu sinh xong được ghi ngay vào eval/results/.partial/<variant>[_test].jsonl. Nếu bị dừng
  giữa chừng (vd Groq 429 hết quota ngày), chạy lại với --resume để chỉ sinh các câu còn thiếu.
  Checkpoint chỉ được dùng lại khi config và câu hỏi khớp hoàn toàn; chạy xong thì checkpoint bị xóa.
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
from langchain_core.messages.ai import add_usage  # noqa: E402

from config.settings import settings  # noqa: E402
from src.rag.chain import build_rag_chain  # noqa: E402
from src.rag.embeddings import get_embeddings  # noqa: E402
from src.rag.prompts import NO_INFO_EN, NO_INFO_VI  # noqa: E402
from src.utils.llm import get_judge_llm  # noqa: E402

# dev: chỉnh hệ thống + ablation; test: chỉ đo cuối (tách 2026-10-04 để tránh tune theo test set)
DATASETS = {
    "dev": settings.paths.eval_datasets / "rag_devset.jsonl",
    "test": settings.paths.eval_datasets / "rag_testset.jsonl",
}
ABLATION_TEST_HEADING = "## Bộ test (đo một lần cuối Phase 1)"
CHECKPOINT_DIR = settings.paths.eval_results / ".partial"
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


def load_testset(split: str = "dev", limit: int | None = None) -> list[dict]:
    """Đọc bộ câu hỏi JSONL của split (dev / test)."""
    path = DATASETS[split]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows[:limit] if limit else rows


def load_checkpoint(path: Path, config: dict, rows: list[dict]) -> dict[str, dict]:
    """Đọc checkpoint → {id: kết quả}. Dừng nếu config hoặc câu hỏi không khớp lần chạy hiện tại."""
    if not path.exists():
        return {}
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0].get("config") != json.loads(json.dumps(config)):  # tuple → list như khi đọc JSON
        raise SystemExit(f"Checkpoint {path} có config khác lần chạy này → xóa file hoặc bỏ --resume")
    questions = {r["id"]: r["question"] for r in rows}
    done = {r["id"]: r for r in (line["result"] for line in lines[1:])}
    if any(questions.get(i) != r["question"] for i, r in done.items()):
        raise SystemExit(f"Checkpoint {path} có câu hỏi không khớp testset hiện tại → xóa file hoặc bỏ --resume")
    return done


def merge_usage(usages: list[dict]) -> dict:
    """Cộng dồn token usage {model: UsageMetadata} của nhiều câu."""
    total: dict = {}
    for u in usages:
        for model, meta in u.items():
            total[model] = add_usage(total[model], meta) if model in total else meta
    return total


def generate_answers(
    variant: str, rows: list[dict], config: dict, checkpoint: Path, resume: bool = False
) -> tuple[list[dict], dict, int]:
    """Chạy RAG chain cho từng câu (tuần tự để đo latency chính xác), ghi checkpoint sau mỗi câu.

    Trả về (kết quả, token usage generator cộng dồn mọi lượt chạy, số câu lấy lại từ checkpoint).
    """
    done = load_checkpoint(checkpoint, config, rows) if resume else {}
    if done:
        print(f"  Resume: lấy lại {len(done)} câu từ {checkpoint}")
    else:  # chạy mới: ghi đè checkpoint cũ, dòng đầu là config
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"config": config}, ensure_ascii=False) + "\n", encoding="utf-8")
    chain = build_rag_chain(variant) if len(done) < len(rows) else None
    results = []
    for row in rows:
        if row["id"] in done:
            results.append(done[row["id"]])
            continue
        usage = UsageMetadataCallbackHandler()  # tách theo câu để cộng dồn đúng qua các lượt resume
        t0 = time.perf_counter()
        out = chain.invoke({"question": row["question"]}, config={"callbacks": [usage]})
        latency = time.perf_counter() - t0
        result = {
            **row,
            "answer": out["answer"],
            "answer_scored": strip_disclaimer(out["answer"]),  # bản đưa cho RAGAS chấm
            "retrieved_contexts": [d.page_content for d in out["contexts"]],
            "retrieved_chunk_ids": [d.metadata["chunk_id"] for d in out["contexts"]],
            "latency_s": round(latency, 3),
            "refused": is_refusal(out["answer"]),
            "gen_token_usage": usage.usage_metadata,
        }
        results.append(result)
        with checkpoint.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"result": result}, ensure_ascii=False) + "\n")
        print(f"  {row['id']} ({latency:.1f}s) refused={result['refused']}")
    return results, merge_usage([r.get("gen_token_usage", {}) for r in results]), len(done)


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


def run_config_snapshot(variant: str, split: str = "dev") -> dict:
    """Cấu hình đầy đủ của lần chạy (để tái lập)."""
    rag = settings.rag
    return {
        "variant": variant, "split": split, "dataset": DATASETS[split].name,
        "chunk_size": rag.chunk_size, "chunk_overlap": rag.chunk_overlap, "top_k": rag.top_k,
        "hybrid_weights": rag.hybrid_weights, "multi_query_n": rag.multi_query_n,
        "rerank_fetch_k": rag.rerank_fetch_k, "rerank_top_n": rag.rerank_top_n,
        "embedding_model": rag.embedding_model, "reranker_model": rag.reranker_model,
        "generator_model": settings.llm.model,
        "judge_provider": settings.llm.judge_provider, "judge_model": settings.llm.judge_model,
        "judge_reasoning_effort": settings.llm.judge_reasoning_effort,
        "ragas_run_config": RAGAS_RUN_CONFIG,
    }


LABELS = {"v0": "V0 dense", "v1": "V1 +hybrid", "v2": "V2 +multi-query", "v3": "V3 +rerank", "v4": "V4 +prompt"}
_TABLE_HEADER = (
    "| Variant | Ctx Precision | Ctx Recall | Faithfulness | Answer Rel. | OOS refusal | False refusal | Latency (s) | Ghi chú |\n"
    "|---|---|---|---|---|---|---|---|---|\n"
)


def upsert_ablation_row(text: str, split: str, label: str, new_row: str) -> str:
    """Thay dòng của variant trong bảng của split; bảng test chưa có dòng đó thì thêm vào cuối bảng.

    Bảng dev đứng đầu file, bảng test nằm dưới ABLATION_TEST_HEADING (tạo trước mục lịch sử nếu chưa có).
    """
    row_re = rf"^\| {re.escape(label)} \|.*$"
    if split == "dev":
        head, sep, tail = text.partition(ABLATION_TEST_HEADING)
        return re.sub(row_re, lambda _: new_row, head, count=1, flags=re.MULTILINE) + sep + tail
    if ABLATION_TEST_HEADING not in text:
        section = (f"{ABLATION_TEST_HEADING}\n\nCùng metric như bảng dev, trên bộ test "
                   f"(`rag_testset.jsonl`, không dùng để chỉnh hệ thống).\n\n{_TABLE_HEADER}\n")
        history = text.find("## Lịch sử chạy")
        text = text[:history] + section + text[history:] if history >= 0 else text.rstrip("\n") + "\n\n" + section
    head, sep, tail = text.partition(ABLATION_TEST_HEADING)
    if re.search(row_re, tail, flags=re.MULTILINE):
        tail = re.sub(row_re, lambda _: new_row, tail, count=1, flags=re.MULTILINE)
    else:  # chèn sau dòng cuối của bảng test (bảng đầu tiên dưới heading)
        lines = tail.split("\n")
        first = next(i for i, line in enumerate(lines) if line.startswith("|"))
        last = first
        while last + 1 < len(lines) and lines[last + 1].startswith("|"):
            last += 1
        tail = "\n".join(lines[: last + 1] + [new_row] + lines[last + 1:])
    return head + sep + tail


def update_ablation_md(
    variant: str, summary: dict, result_file: Path, config: dict, n: int, split: str = "dev"
) -> None:
    """Cập nhật dòng của variant trong bảng ablation (dev hoặc test) + thêm 1 dòng lịch sử chạy."""
    path = settings.paths.eval_results / "ablation.md"
    if not path.exists():
        rows = "\n".join(f"| {LABELS[v]} | | | | | | | | |" for v in LABELS)
        path.write_text(
            "# Ablation RAG (V0 → V4)\n\n"
            "Bảng ablation đo trên bộ **dev** (`rag_devset.jsonl`). "
            "Metric RAGAS tính trên các câu trong phạm vi (không gồm out_of_scope). "
            "OOS refusal = tỉ lệ từ chối đúng ở nhóm out_of_scope; False refusal = tỉ lệ từ chối nhầm ở câu trong phạm vi.\n\n"
            f"{_TABLE_HEADER}{rows}\n\n## Lịch sử chạy\n\n"
            "| Thời điểm | Split | Variant | Số câu | Generator | Judge | Kết quả |\n|---|---|---|---|---|---|---|\n",
            encoding="utf-8",
        )

    def fmt(x):
        return "" if x is None else f"{x:.3f}"

    s = summary
    new_row = (
        f"| {LABELS[variant]} | {fmt(s['llm_context_precision_with_reference'])} | {fmt(s['context_recall'])} | "
        f"{fmt(s['faithfulness'])} | {fmt(s['answer_relevancy'])} | {fmt(s['oos_refusal_rate'])} | "
        f"{fmt(s['false_refusal_rate'])} | {fmt(s['latency_mean_s'])} | {result_file.name} |"
    )
    text = upsert_ablation_row(path.read_text(encoding="utf-8"), split, LABELS[variant], new_row)
    text = text.rstrip("\n") + (
        f"\n| {datetime.now():%Y-%m-%d %H:%M} | {split} | {variant} | {n} | {config['generator_model']} | "
        f"{config['judge_model']} | {result_file.name} |\n"
    )
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy RAGAS cho 1 variant RAG")
    parser.add_argument("--variant", default="v0")
    parser.add_argument("--split", choices=tuple(DATASETS), default="dev",
                        help="dev: ablation + chỉnh hệ thống; test: chỉ đo cuối Phase 1")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu (debug)")
    parser.add_argument("--resume", action="store_true", help="Dùng lại câu trả lời đã sinh trong checkpoint")
    args = parser.parse_args()

    rows = load_testset(args.split, args.limit)
    config = run_config_snapshot(args.variant, args.split)
    split_suffix = "_test" if args.split == "test" else ""  # dev giữ cách đặt tên như các lượt chạy trước khi tách
    suffix = f"_limit{args.limit}" if args.limit else ""
    checkpoint = CHECKPOINT_DIR / f"{args.variant}{split_suffix}{suffix}.jsonl"
    t_start = time.perf_counter()
    print(f"[1/2] Sinh câu trả lời ({args.variant}, {args.split}, {len(rows)} câu)...")
    judge_usage = UsageMetadataCallbackHandler()
    results, gen_usage, n_resumed = generate_answers(args.variant, rows, config, checkpoint, args.resume)
    print("[2/2] Chấm RAGAS...")
    results = run_ragas(results, judge_usage)
    summary = summarize(results)
    elapsed = round(time.perf_counter() - t_start, 1)

    settings.paths.eval_results.mkdir(parents=True, exist_ok=True)
    out_file = settings.paths.eval_results / f"{datetime.now():%Y%m%d-%H%M}_{args.variant}{split_suffix}.json"
    out_file.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config, "n_questions": len(rows), "runtime_s": elapsed,
        # runtime_s chỉ tính lượt chạy cuối; latency từng câu vẫn đo ở lượt sinh ra câu đó
        "n_resumed_from_checkpoint": n_resumed,
        "summary": summary,
        "token_usage": {"generator": gen_usage, "judge": judge_usage.usage_metadata},
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.limit is None:
        update_ablation_md(args.variant, summary, out_file, config, len(rows), args.split)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    checkpoint.unlink(missing_ok=True)  # đã lưu kết quả đầy đủ → không cần checkpoint nữa
    print("Token usage:", json.dumps({"generator": gen_usage, "judge": judge_usage.usage_metadata}, default=str))
    print(f"Đã lưu {out_file} ({elapsed}s)")


if __name__ == "__main__":
    main()
