"""Test checkpoint / --resume của eval/run_rag_eval.py với RAG chain giả (không gọi LLM)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from eval import run_rag_eval as ev

ROWS = [{"id": f"q{i}", "group": "threshold", "question": f"câu {i}"} for i in range(4)]
CONFIG = {"variant": "vx", "hybrid_weights": (0.5, 0.5)}


class _FakeChain:
    """Trả lời giả; ném lỗi (giả lập Groq 429) khi gặp câu nằm trong fail_on."""

    def __init__(self, fail_on: set[str] = frozenset()) -> None:
        self.fail_on, self.calls = fail_on, []

    def invoke(self, x: dict, config: dict | None = None) -> dict:
        self.calls.append(x["question"])
        if x["question"] in self.fail_on:
            raise RuntimeError("429 tokens per day")
        return {"answer": f"đáp {x['question']}", "contexts": [Document(page_content="c", metadata={"chunk_id": "k"})]}


def _use(monkeypatch, chain: _FakeChain) -> None:
    monkeypatch.setattr(ev, "build_rag_chain", lambda variant: chain)


def test_resume_chi_sinh_cau_con_thieu(monkeypatch, tmp_path) -> None:
    ckpt = tmp_path / "vx.jsonl"
    first = _FakeChain(fail_on={"câu 2"})
    _use(monkeypatch, first)
    with pytest.raises(RuntimeError):
        ev.generate_answers("vx", ROWS, CONFIG, ckpt)
    assert len(ckpt.read_text(encoding="utf-8").splitlines()) == 3  # header + 2 câu xong

    second = _FakeChain()
    _use(monkeypatch, second)
    results, _, n_resumed = ev.generate_answers("vx", ROWS, CONFIG, ckpt, resume=True)
    assert n_resumed == 2
    assert second.calls == ["câu 2", "câu 3"]
    assert [r["id"] for r in results] == ["q0", "q1", "q2", "q3"]


def test_khong_resume_thi_chay_lai_tu_dau(monkeypatch, tmp_path) -> None:
    ckpt = tmp_path / "vx.jsonl"
    _use(monkeypatch, _FakeChain())
    ev.generate_answers("vx", ROWS, CONFIG, ckpt)
    again = _FakeChain()
    _use(monkeypatch, again)
    ev.generate_answers("vx", ROWS, CONFIG, ckpt)  # không --resume → ghi đè
    assert len(again.calls) == 4


def test_resume_tu_choi_config_khac(monkeypatch, tmp_path) -> None:
    ckpt = tmp_path / "vx.jsonl"
    _use(monkeypatch, _FakeChain())
    ev.generate_answers("vx", ROWS[:2], CONFIG, ckpt)
    with pytest.raises(SystemExit):
        ev.generate_answers("vx", ROWS, {**CONFIG, "variant": "vy"}, ckpt, resume=True)


def test_resume_tu_choi_cau_hoi_doi(monkeypatch, tmp_path) -> None:
    ckpt = tmp_path / "vx.jsonl"
    _use(monkeypatch, _FakeChain())
    ev.generate_answers("vx", ROWS[:2], CONFIG, ckpt)
    changed = [{**ROWS[0], "question": "câu khác"}, *ROWS[1:]]
    with pytest.raises(SystemExit):
        ev.generate_answers("vx", changed, CONFIG, ckpt, resume=True)


def test_merge_usage_cong_don() -> None:
    u = {"m": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}}
    total = ev.merge_usage([u, u, {}])
    assert total["m"]["total_tokens"] == 24
    assert total["m"]["input_tokens"] == 20
