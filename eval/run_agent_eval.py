"""Đánh giá LangGraph agent trên eval/datasets/agent_testset.jsonl (Phase 4) và ghi kết quả.

Chạy: uv run python -m eval.run_agent_eval [--limit N] [--ids a001,a020] [--resume]

- Mỗi câu chạy trên 1 thread mới, LLM + 3 tool thật. Câu `follow_up` chạy trước các lượt `setup_turns` trên cùng
  thread; chỉ lượt cuối (`question`) được chấm.
- Chấm theo luật (agentevals, không tốn token):
  - trajectory match theo `match_mode` của từng câu (strict / subset / superset), bỏ qua tham số tool;
  - tool selection: tập tool đã gọi so với tập kỳ vọng (cùng quy tắc mode, không xét thứ tự / số lần);
  - args hints: tham số tool có chứa chuỗi gợi ý không (không phân biệt dấu, hoa thường);
  - số bước (số tool call) so với tối thiểu, loop (vượt recursion_limit), latency lượt cuối.
- Chấm bằng LLM (judge = JUDGE_MODEL, không cần reference): rubric riêng của project (đúng tool, không bịa số liệu,
  khuyến nghị phải qua RAG, từ chối đúng phạm vi...), đạt / không đạt + lý do.
- Lưu eval/results/<YYYYMMDD-HHMM>_agent.json và thêm 1 lượt vào eval/results/agent_eval.md.
- Rate limit Groq (8K token/phút): nghỉ `eval_cooldown_s` giữa các câu (không tính vào latency). Câu vẫn gặp 429
  (ở agent, hoặc bên trong tool RAG — tool trả lỗi thay vì raise) → chờ rồi chạy lại cả câu trên thread mới.
- Checkpoint: mỗi câu chạy xong ghi ngay vào eval/results/.partial/agent.jsonl. Bị dừng giữa chừng (vd Groq 429
  hết quota ngày) → chạy lại với --resume để chỉ chạy các câu còn thiếu. Checkpoint chỉ dùng lại khi config và
  câu hỏi khớp hoàn toàn; chạy xong thì checkpoint bị xóa.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import unicodedata
import uuid
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from agentevals.trajectory.llm import create_trajectory_llm_as_judge
from agentevals.trajectory.match import create_trajectory_match_evaluator
from groq import RateLimitError
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.messages.ai import add_usage
from langgraph.errors import GraphRecursionError

from config.settings import settings
from src.agent.graph import build_graph, run_config
from src.utils.llm import get_judge_llm

DATASET = settings.paths.eval_datasets / "agent_testset.jsonl"
CHECKPOINT = settings.paths.eval_results / ".partial" / "agent.jsonl"
REPORT = settings.paths.eval_results / "agent_eval.md"
CATEGORIES = ("full", "aqi_only", "knowledge_only", "out_of_scope", "follow_up", "ambiguous")

# Rubric cho judge (tiếng Anh cho judge dễ theo). Không dùng dấu ngoặc nhọn ngoài {outputs}: prompt được format.
JUDGE_PROMPT = """You are an expert grader of an AI agent that answers questions about air quality in
Ho Chi Minh City (HCMC), Vietnam. The agent has three tools:
- geocode_address: place name -> coordinates (rejects places outside the old HCMC boundary with out_of_scope)
- get_air_quality: coordinates -> VN_AQI and pollutant concentrations (CAMS model data)
- retrieve_health_guideline: searches official documents (QCVN 05:2023, WHO 2021 guidelines, VN_AQI guidance)

Grade ONLY how the agent handled the LAST user message (earlier turns are context).

<Rubric>
  A correct trajectory:
  - Calls the tools the request needs, in a sensible order, and no unnecessary ones
    (location air quality -> geocode then get_air_quality; health advice -> retrieve_health_guideline;
    pure document knowledge -> retrieve_health_guideline only).
  - Never fabricates data: every coordinate, VN_AQI value, concentration, threshold or health recommendation in
    the final answer comes from a tool output or an earlier turn of the conversation.
  - Gives health recommendations only after calling retrieve_health_guideline, consistent with its answer;
    if the documents have no information, says so instead of advising from general knowledge.
  - Politely refuses without fetching air quality data when the request is not about air quality, or the place is
    outside the old HCMC area (e.g. other provinces, former Binh Duong, former Ba Ria - Vung Tau).
  - Asks the user for a specific location when the location is missing or ambiguous, instead of guessing.
  - For a follow-up about the same place, reuses earlier results instead of geocoding again;
    for a new place, looks it up again.
  - Answers in the same language as the user.
  - Is reasonably efficient (no repeated or looping calls).
</Rubric>

<trajectory>
{outputs}
</trajectory>
"""


# ---------------------------------------------------------------- hàm thuần (có unit test)


def fold(text: str) -> str:
    """Bỏ dấu tiếng Việt + chữ thường (so khớp gợi ý tham số không phân biệt dấu)."""
    text = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()


def last_turn(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Các message của lượt cuối: từ HumanMessage cuối cùng đến hết."""
    idx = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return list(messages[idx:])


def tool_calls_of(messages: Sequence[BaseMessage]) -> list[dict]:
    """Danh sách tool call (name, args) theo thứ tự trong các message."""
    return [
        {"name": c["name"], "args": c["args"]}
        for m in messages if isinstance(m, AIMessage)
        for c in m.tool_calls
    ]


def reference_messages(question: str, expected: Sequence[str]) -> list[BaseMessage]:
    """Trajectory tham chiếu cho agentevals: mỗi tool call 1 AIMessage + ToolMessage, cuối là câu trả lời."""
    msgs: list[BaseMessage] = [HumanMessage(question)]
    for i, name in enumerate(expected):
        msgs.append(AIMessage("", tool_calls=[{"name": name, "args": {}, "id": f"ref{i}"}]))
        msgs.append(ToolMessage("...", tool_call_id=f"ref{i}", name=name))
    msgs.append(AIMessage("(câu trả lời cuối)"))
    return msgs


def tool_selection_ok(actual: Sequence[str], expected: Sequence[str], mode: str) -> bool:
    """Chọn tool đúng: so tập tool (không xét thứ tự / số lần) theo match_mode của câu."""
    a, e = set(actual), set(expected)
    if mode == "subset":
        return a <= e
    if mode == "superset":
        return a >= e
    return a == e


def args_hints_ok(calls: Sequence[dict], hints: dict[str, str]) -> bool | None:
    """Mọi gợi ý có xuất hiện trong tham số của ít nhất 1 lần gọi tool đó không.

    Bỏ qua gợi ý của tool không được gọi (lỗi đó đã tính ở trajectory). Không có gợi ý nào áp dụng → None.
    """
    checks = []
    for tool, hint in hints.items():
        args = [fold(json.dumps(c["args"], ensure_ascii=False)) for c in calls if c["name"] == tool]
        if args:
            checks.append(any(fold(hint) in a for a in args))
    return all(checks) if checks else None


def merge_usage(usages: list[dict]) -> dict:
    """Cộng dồn token usage {model: UsageMetadata} của nhiều câu (như eval/run_rag_eval.py, tránh import ragas)."""
    total: dict = {}
    for u in usages:
        for model, meta in u.items():
            total[model] = add_usage(total[model], meta) if model in total else meta
    return total


def hit_rate_limit(messages: Sequence[BaseMessage]) -> bool:
    """Có tool nào trả lỗi 429 không (tool RAG bắt exception của Groq và trả về message lỗi)."""
    return any(isinstance(m, ToolMessage) and "RateLimitError" in str(m.content) for m in messages)


def min_steps(expected: Sequence[str], mode: str) -> int:
    """Số tool call tối thiểu để đạt: subset cho phép không gọi tool nào."""
    return 0 if mode == "subset" else len(expected)


def serialize(messages: Sequence[BaseMessage]) -> list[dict]:
    """Message → dict gọn để lưu JSON."""
    out = []
    for m in messages:
        d: dict[str, Any] = {"type": m.type, "content": m.content}
        if isinstance(m, AIMessage) and m.tool_calls:
            d["tool_calls"] = [{"name": c["name"], "args": c["args"]} for c in m.tool_calls]
        if isinstance(m, ToolMessage):
            d["name"] = m.name
        out.append(d)
    return out


def score_rules(row: dict, turn: Sequence[BaseMessage]) -> dict:
    """Các điểm chấm theo luật cho 1 câu (không gọi LLM)."""
    calls = tool_calls_of(turn)
    names = [c["name"] for c in calls]
    expected, mode = row["expected_trajectory"], row["match_mode"]
    matcher = create_trajectory_match_evaluator(trajectory_match_mode=mode, tool_args_match_mode="ignore")
    match = matcher(outputs=list(turn), reference_outputs=reference_messages(row["question"], expected))
    return {
        "actual_trajectory": names,
        "trajectory_match": bool(match["score"]),
        "tool_selection_ok": tool_selection_ok(names, expected, mode),
        "args_hints_ok": args_hints_ok(calls, row.get("expected_args_hints", {})),
        "n_steps": len(calls),
        "min_steps": min_steps(expected, mode),
    }


# ---------------------------------------------------------------- chạy agent


def load_testset(limit: int | None = None, ids: set[str] | None = None) -> list[dict]:
    """Đọc agent_testset.jsonl, lọc theo --ids / --limit."""
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    if ids:
        rows = [r for r in rows if r["id"] in ids]
    return rows[:limit] if limit else rows


def load_checkpoint(path: Path, config: dict, rows: list[dict]) -> dict[str, dict]:
    """Đọc checkpoint → {id: kết quả}. Dừng nếu config hoặc câu hỏi không khớp lần chạy hiện tại."""
    if not path.exists():
        return {}
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0].get("config") != json.loads(json.dumps(config)):
        raise SystemExit(f"Checkpoint {path} có config khác lần chạy này → xóa file hoặc bỏ --resume")
    questions = {r["id"]: r["question"] for r in rows}
    done = {r["id"]: r for r in (line["result"] for line in lines[1:])}
    if any(questions.get(i) != r["question"] for i, r in done.items()):
        raise SystemExit(f"Checkpoint {path} có câu hỏi không khớp testset hiện tại → xóa file hoặc bỏ --resume")
    return done


def run_case(graph: Any, row: dict) -> dict:
    """Chạy 1 câu (kèm setup_turns) trên thread mới; trả kết quả + điểm theo luật.

    Lỗi API (vd Groq 429) không bắt ở đây → lan ra ngoài, câu này chạy lại khi --resume.
    """
    cfg = run_config(f"eval-{row['id']}-{uuid.uuid4().hex[:8]}")
    usage = UsageMetadataCallbackHandler()
    cfg["callbacks"] = [usage]
    for q in row.get("setup_turns", []):
        graph.invoke({"messages": [HumanMessage(q)]}, cfg)
    looped = False
    t0 = time.perf_counter()
    try:
        graph.invoke({"messages": [HumanMessage(row["question"])]}, cfg)
    except GraphRecursionError:
        looped = True
    latency = time.perf_counter() - t0
    messages = graph.get_state(cfg).values["messages"]
    if hit_rate_limit(messages):
        raise RateLimited(row["id"])
    turn = last_turn(messages)
    final = turn[-1].content if isinstance(turn[-1], AIMessage) and not turn[-1].tool_calls else ""
    return {
        **row,
        **score_rules(row, turn),
        "looped": looped,
        "latency_s": round(latency, 3),
        "final_answer": final,
        "conversation": serialize(messages),
        "gen_token_usage": usage.usage_metadata,
    }


class RateLimited(Exception):
    """Tool trả lỗi 429 trong lúc chạy câu → kết quả không hợp lệ, cần chạy lại."""


def run_case_retrying(graph: Any, row: dict, wait_s: float, retries: int) -> dict:
    """run_case; gặp 429 (agent hoặc tool) thì chờ `wait_s` rồi chạy lại cả câu, tối đa `retries` lần."""
    for attempt in range(retries + 1):
        try:
            return {**run_case(graph, row), "rate_limit_retries": attempt}
        except (RateLimitError, RateLimited) as exc:
            if attempt == retries:
                raise
            print(f"  {row['id']}: 429 ({type(exc).__name__}) → chờ {wait_s:.0f}s, chạy lại lần {attempt + 1}")
            time.sleep(wait_s)
    raise AssertionError("không tới được")


def run_cases(rows: list[dict], config: dict, checkpoint: Path, resume: bool = False,
              cooldown_s: float = 0.0, wait_s: float = 0.0, retries: int = 0) -> tuple[list[dict], int]:
    """Chạy tuần tự mọi câu, ghi checkpoint sau mỗi câu. Trả (kết quả, số câu lấy lại từ checkpoint).

    `cooldown_s`: nghỉ trước mỗi câu (trừ câu đầu) cho cửa sổ TPM hồi lại; `wait_s`, `retries`: xem run_case_retrying.
    """
    done = load_checkpoint(checkpoint, config, rows) if resume else {}
    if done:
        print(f"  Resume: lấy lại {len(done)} câu từ {checkpoint}")
    else:
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"config": config}, ensure_ascii=False) + "\n", encoding="utf-8")
    graph = build_graph() if len(done) < len(rows) else None
    results = []
    first = True
    for row in rows:
        if row["id"] in done:
            results.append(done[row["id"]])
            continue
        if not first and cooldown_s:
            time.sleep(cooldown_s)
        first = False
        result = run_case_retrying(graph, row, wait_s, retries)
        results.append(result)
        with checkpoint.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"result": result}, ensure_ascii=False) + "\n")
        print(f"  {row['id']} ({result['latency_s']:.1f}s) {result['actual_trajectory']} "
              f"match={result['trajectory_match']}{' LOOP' if result['looped'] else ''}")
    return results, len(done)


def run_judge(results: list[dict], usage: UsageMetadataCallbackHandler) -> None:
    """LLM-as-judge (không reference) cho từng câu, chấm trên toàn bộ hội thoại của thread."""
    judge = create_trajectory_llm_as_judge(prompt=JUDGE_PROMPT, judge=get_judge_llm(callbacks=[usage]))
    for r in results:
        try:
            out = judge(outputs=_to_messages(r["conversation"]))
            r["judge"] = {"pass": bool(out["score"]), "reasoning": out.get("comment")}
        except Exception as exc:  # judge lỗi 1 câu không làm hỏng cả lượt
            r["judge"] = {"pass": None, "reasoning": f"judge lỗi: {type(exc).__name__}: {exc}"}
        print(f"  judge {r['id']}: {r['judge']['pass']}")


def _to_messages(conversation: list[dict]) -> list[BaseMessage]:
    """Dựng lại LangChain message từ bản serialize (để judge đọc được cả tool call)."""
    msgs: list[BaseMessage] = []
    pending: list[str] = []  # id tool call chờ ToolMessage
    for i, d in enumerate(conversation):
        if d["type"] == "human":
            msgs.append(HumanMessage(d["content"]))
        elif d["type"] == "ai":
            calls = [{"name": c["name"], "args": c["args"], "id": f"c{i}_{j}"}
                     for j, c in enumerate(d.get("tool_calls", []))]
            pending = [c["id"] for c in calls]
            msgs.append(AIMessage(d["content"], tool_calls=calls))
        elif d["type"] == "tool":
            msgs.append(ToolMessage(d["content"], tool_call_id=pending.pop(0) if pending else "", name=d.get("name")))
    return msgs


# ---------------------------------------------------------------- tổng hợp + báo cáo


def _rate(values: list[bool | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def summarize(results: list[dict]) -> dict:
    """Tổng hợp metric toàn bộ + theo nhóm."""
    def block(rs: list[dict]) -> dict:
        return {
            "n": len(rs),
            "trajectory_match": _rate([r["trajectory_match"] for r in rs]),
            "tool_selection": _rate([r["tool_selection_ok"] for r in rs]),
            "args_hints": _rate([r["args_hints_ok"] for r in rs]),
            "judge_pass": _rate([r.get("judge", {}).get("pass") for r in rs]),
            "steps_mean": round(statistics.mean(r["n_steps"] for r in rs), 3),
            "min_steps_mean": round(statistics.mean(r["min_steps"] for r in rs), 3),
            "loop_rate": _rate([r["looped"] for r in rs]),
            "latency_mean_s": round(statistics.mean(r["latency_s"] for r in rs), 3),
        }
    summary = block(results)
    summary["latency_p50_s"] = round(statistics.median(r["latency_s"] for r in results), 3)
    summary["judge_failed"] = sum(r.get("judge", {}).get("pass") is None for r in results)
    summary["by_category"] = {c: block([r for r in results if r["category"] == c])
                              for c in CATEGORIES if any(r["category"] == c for r in results)}
    return summary


def run_config_snapshot() -> dict:
    """Cấu hình đầy đủ của lần chạy (để tái lập)."""
    return {
        "dataset": DATASET.name,
        "generator_model": settings.llm.model,
        "rag_tool_variant": settings.rag.tool_variant,
        "recursion_limit": settings.agent.recursion_limit,
        "eval_cooldown_s": settings.agent.eval_cooldown_s,
        "judge_provider": settings.llm.judge_provider, "judge_model": settings.llm.judge_model,
        "judge_reasoning_effort": settings.llm.judge_reasoning_effort,
    }


def _fmt(x: float | None) -> str:
    return "" if x is None else f"{x:.3f}"


def update_report(summary: dict, results: list[dict], result_file: Path, config: dict) -> None:
    """Thêm 1 lượt chạy vào agent_eval.md: dòng tổng hợp + bảng theo nhóm + danh sách câu chưa đạt."""
    header = ("| Thời điểm | Số câu | Trajectory match | Tool selection | Args hints | Judge pass | "
              "Bước TB / tối thiểu | Loop | Latency TB (s) | Generator | Judge | Kết quả |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    if not REPORT.exists():
        REPORT.write_text(
            "# Đánh giá agent (Phase 4)\n\n"
            "Bộ câu hỏi: `eval/datasets/agent_testset.jsonl`. Trajectory match / tool selection / args hints chấm theo "
            "luật (agentevals); Judge pass = LLM-as-judge không reference (rubric trong `eval/run_agent_eval.py`). "
            "Bước = số tool call ở lượt được chấm. Câu `follow_up` chỉ chấm lượt cuối.\n\n"
            f"## Các lượt chạy\n\n{header}\n## Chi tiết từng lượt\n",
            encoding="utf-8",
        )
    s = summary
    stamp = f"{datetime.now():%Y-%m-%d %H:%M}"
    row = (f"| {stamp} | {s['n']} | {_fmt(s['trajectory_match'])} | {_fmt(s['tool_selection'])} | "
           f"{_fmt(s['args_hints'])} | {_fmt(s['judge_pass'])} | {s['steps_mean']:.2f} / {s['min_steps_mean']:.2f} | "
           f"{_fmt(s['loop_rate'])} | {s['latency_mean_s']:.1f} | {config['generator_model']} | "
           f"{config['judge_model']} | {result_file.name} |")
    text = REPORT.read_text(encoding="utf-8")
    head, sep, tail = text.partition("\n## Chi tiết từng lượt")
    head = head.rstrip("\n") + "\n" + row + "\n"

    lines = [f"\n### {stamp} — {result_file.name}\n",
             "| Nhóm | Số câu | Trajectory match | Tool selection | Args hints | Judge pass | Bước TB / tối thiểu | Latency TB (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for cat, b in s["by_category"].items():
        lines.append(f"| {cat} | {b['n']} | {_fmt(b['trajectory_match'])} | {_fmt(b['tool_selection'])} | "
                     f"{_fmt(b['args_hints'])} | {_fmt(b['judge_pass'])} | "
                     f"{b['steps_mean']:.2f} / {b['min_steps_mean']:.2f} | {b['latency_mean_s']:.1f} |")
    failed = [r for r in results if not r["trajectory_match"] or r.get("judge", {}).get("pass") is False
              or r["args_hints_ok"] is False or r["looped"]]
    lines.append("\nCâu chưa đạt (trajectory, args hints, judge hoặc loop):\n")
    lines += [f"- **{r['id']}** ({r['category']}): kỳ vọng {r['expected_trajectory']} ({r['match_mode']}), "
              f"thực tế {r['actual_trajectory']}; args hints={r['args_hints_ok']}; "
              f"judge={r.get('judge', {}).get('pass')}{'; LOOP' if r['looped'] else ''}" for r in failed] or ["- (không có)"]
    REPORT.write_text(head + sep + tail.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá LangGraph agent (trajectory + LLM-as-judge)")
    parser.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu (debug)")
    parser.add_argument("--ids", default=None, help="Chỉ chạy các id, cách nhau dấu phẩy (debug)")
    parser.add_argument("--resume", action="store_true", help="Dùng lại kết quả đã chạy trong checkpoint")
    args = parser.parse_args()

    ids = set(args.ids.split(",")) if args.ids else None
    rows = load_testset(args.limit, ids)
    partial = bool(args.limit or ids)
    config = run_config_snapshot()
    checkpoint = CHECKPOINT.with_name("agent_partial.jsonl") if partial else CHECKPOINT
    t_start = time.perf_counter()
    print(f"[1/2] Chạy agent ({len(rows)} câu)...")
    agent_cfg = settings.agent
    results, n_resumed = run_cases(rows, config, checkpoint, args.resume, cooldown_s=agent_cfg.eval_cooldown_s,
                                   wait_s=agent_cfg.eval_rate_limit_wait_s, retries=agent_cfg.eval_rate_limit_retries)
    print("[2/2] LLM-as-judge...")
    judge_usage = UsageMetadataCallbackHandler()
    run_judge(results, judge_usage)
    summary = summarize(results)
    elapsed = round(time.perf_counter() - t_start, 1)
    gen_usage = merge_usage([r.get("gen_token_usage", {}) for r in results])

    settings.paths.eval_results.mkdir(parents=True, exist_ok=True)
    out_file = settings.paths.eval_results / f"{datetime.now():%Y%m%d-%H%M}_agent{'_partial' if partial else ''}.json"
    out_file.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": config, "n_questions": len(rows), "runtime_s": elapsed,
        "n_resumed_from_checkpoint": n_resumed,
        "summary": summary,
        "token_usage": {"generator": gen_usage, "judge": judge_usage.usage_metadata},
        "results": results,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if not partial:
        update_report(summary, results, out_file, config)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    checkpoint.unlink(missing_ok=True)
    print("Token usage:", json.dumps({"generator": gen_usage, "judge": judge_usage.usage_metadata}, default=str))
    print(f"Đã lưu {out_file} ({elapsed}s)")


if __name__ == "__main__":
    main()
