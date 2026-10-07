"""Test eval/run_agent_eval.py: chấm theo luật, checkpoint / --resume (LLM và tool giả, không gọi API)."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from eval import run_agent_eval as ev
from src.agent.graph import build_graph
from tests.test_agent_graph import NOW, TOOLS, FakeToolLLM, call


def _turn(*names: str, question: str = "q") -> list:
    """Lượt hội thoại giả: gọi lần lượt các tool rồi trả lời."""
    msgs: list = [HumanMessage(question)]
    for i, n in enumerate(names):
        msgs += [call(n, {"address": "Quận 3"} if n == "geocode_address" else {"query": "VN_AQI 160, hen suyễn"}, f"t{i}"),
                 ToolMessage("{}", tool_call_id=f"t{i}", name=n)]
    return msgs + [AIMessage("trả lời")]


def _row(expected: list[str], mode: str = "strict", hints: dict | None = None) -> dict:
    return {"id": "a", "question": "q", "expected_trajectory": expected, "match_mode": mode,
            "expected_args_hints": hints or {}}


FULL = ["geocode_address", "get_air_quality", "retrieve_health_guideline"]


def test_fold_bo_dau_va_chu_hoa() -> None:
    assert ev.fold("Đường Bến Thành") == "duong ben thanh"


def test_strict_dung_thu_tu_moi_dat() -> None:
    assert ev.score_rules(_row(FULL), _turn(*FULL))["trajectory_match"]
    swapped = ev.score_rules(_row(FULL), _turn("get_air_quality", "geocode_address", "retrieve_health_guideline"))
    assert not swapped["trajectory_match"]
    assert swapped["tool_selection_ok"]  # đúng tập tool, sai thứ tự


def test_strict_thieu_hoac_thua_tool_khong_dat() -> None:
    assert not ev.score_rules(_row(FULL), _turn("geocode_address", "get_air_quality"))["trajectory_match"]
    assert not ev.score_rules(_row([]), _turn("geocode_address"))["trajectory_match"]
    assert ev.score_rules(_row([]), _turn())["trajectory_match"]


def test_subset_cho_phep_khong_goi_tool_nhung_cam_tool_ngoai_danh_sach() -> None:
    row = _row(["geocode_address"], "subset")
    assert ev.score_rules(row, _turn())["trajectory_match"]
    assert ev.score_rules(row, _turn("geocode_address"))["trajectory_match"]
    bad = ev.score_rules(row, _turn("geocode_address", "get_air_quality"))
    assert not bad["trajectory_match"] and not bad["tool_selection_ok"]
    assert ev.score_rules(row, _turn())["min_steps"] == 0


def test_superset_cho_phep_goi_lap() -> None:
    row = _row(["retrieve_health_guideline"], "superset")
    assert ev.score_rules(row, _turn("retrieve_health_guideline", "retrieve_health_guideline"))["trajectory_match"]
    assert not ev.score_rules(row, _turn())["trajectory_match"]


def test_args_hints_khong_phan_biet_dau() -> None:
    calls = [{"name": "geocode_address", "args": {"address": "Phu Nhuan"}}]
    assert ev.args_hints_ok(calls, {"geocode_address": "Phú Nhuận"}) is True
    assert ev.args_hints_ok(calls, {"geocode_address": "Gò Vấp"}) is False
    assert ev.args_hints_ok(calls, {"retrieve_health_guideline": "tim"}) is None  # tool không được gọi


def test_last_turn_chi_lay_luot_cuoi() -> None:
    msgs = _turn("geocode_address", question="lượt 1") + _turn("retrieve_health_guideline", question="lượt 2")
    turn = ev.last_turn(msgs)
    assert turn[0].content == "lượt 2"
    assert [c["name"] for c in ev.tool_calls_of(turn)] == ["retrieve_health_guideline"]


def test_serialize_roi_dung_lai_giu_tool_call() -> None:
    msgs = _turn("geocode_address", "get_air_quality")
    back = ev._to_messages(ev.serialize(msgs))
    assert [type(m) for m in back] == [type(m) for m in msgs]
    assert ev.tool_calls_of(back) == ev.tool_calls_of(msgs)
    assert back[2].tool_call_id == back[1].tool_calls[0]["id"]


def test_run_case_follow_up_chi_cham_luot_cuoi() -> None:
    llm = FakeToolLLM(responses=[
        call("geocode_address", {"address": "Tân Bình"}, "1"), call("get_air_quality", {"lat": 1, "lng": 2}, "2"),
        AIMessage("VN_AQI 160"),
        call("retrieve_health_guideline", {"query": "VN_AQI 160, mở cửa sổ"}, "3"), AIMessage("Nên đóng cửa sổ."),
    ], seen=[])
    graph = build_graph(llm=llm, tools=TOOLS, now=lambda: NOW)
    row = {"id": "a020", "category": "follow_up", "question": "Vậy có nên mở cửa sổ không?",
           "setup_turns": ["Không khí ở Tân Bình thế nào?"], "expected_trajectory": ["retrieve_health_guideline"],
           "match_mode": "strict", "expected_args_hints": {}}
    r = ev.run_case(graph, row)
    assert r["actual_trajectory"] == ["retrieve_health_guideline"]
    assert r["trajectory_match"] and not r["looped"]
    assert r["final_answer"] == "Nên đóng cửa sổ."
    assert sum(m["type"] == "human" for m in r["conversation"]) == 2


def test_resume_chi_chay_cau_con_thieu(monkeypatch, tmp_path) -> None:
    rows = [{"id": f"a{i}", "question": f"câu {i}"} for i in range(3)]
    calls: list[str] = []

    def fake_run_case(graph, row, fail_on=()):
        calls.append(row["id"])
        if row["id"] in fail_on:
            raise RuntimeError("429 tokens per day")
        return {**row, "actual_trajectory": [], "trajectory_match": True, "looped": False, "latency_s": 0.1}

    monkeypatch.setattr(ev, "build_graph", lambda: None)
    monkeypatch.setattr(ev, "run_case", lambda g, r: fake_run_case(g, r, fail_on={"a1"}))
    ckpt = tmp_path / "agent.jsonl"
    with pytest.raises(RuntimeError):
        ev.run_cases(rows, {"m": 1}, ckpt)
    assert len(ckpt.read_text(encoding="utf-8").splitlines()) == 2  # header + a0

    calls.clear()
    monkeypatch.setattr(ev, "run_case", fake_run_case)
    results, n_resumed = ev.run_cases(rows, {"m": 1}, ckpt, resume=True)
    assert calls == ["a1", "a2"] and n_resumed == 1 and [r["id"] for r in results] == ["a0", "a1", "a2"]

    with pytest.raises(SystemExit):  # config khác → không dùng lại checkpoint
        ev.run_cases(rows, {"m": 2}, ckpt, resume=True)


def test_hit_rate_limit_phat_hien_loi_429_trong_tool() -> None:
    ok = [ToolMessage('{"answer": "..."}', tool_call_id="1", name="retrieve_health_guideline")]
    bad = [ToolMessage('{"error": "api_error", "message": "Không tra được tài liệu: RateLimitError: 429"}',
                       tool_call_id="1", name="retrieve_health_guideline")]
    assert not ev.hit_rate_limit(ok) and ev.hit_rate_limit(bad)


def test_run_case_retrying_chay_lai_khi_429(monkeypatch) -> None:
    attempts: list[int] = []

    def flaky(graph, row):
        attempts.append(1)
        if len(attempts) < 3:
            raise ev.RateLimited(row["id"])
        return {**row, "latency_s": 0.1}

    monkeypatch.setattr(ev, "run_case", flaky)
    r = ev.run_case_retrying(None, {"id": "a"}, wait_s=0, retries=3)
    assert r["rate_limit_retries"] == 2 and len(attempts) == 3

    attempts.clear()
    with pytest.raises(ev.RateLimited):  # hết số lần retry → để lỗi lan ra (chạy lại bằng --resume)
        ev.run_case_retrying(None, {"id": "a"}, wait_s=0, retries=1)
