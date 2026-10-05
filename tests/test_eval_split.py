"""Test tách dev/test của eval/run_rag_eval.py (đọc đúng file, ghi đúng bảng trong ablation.md)."""

from __future__ import annotations

from eval import run_rag_eval as ev

ABLATION = f"""# Ablation

| Variant | A | Ghi chú |
|---|---|---|
| V0 dense | 0.1 | dev_v0.json |
| V3 +rerank | 0.3 | dev_v3.json |

{ev.ABLATION_TEST_HEADING}

| Variant | A | Ghi chú |
|---|---|---|

## Lịch sử chạy

| Thời điểm | Split |
|---|---|
"""


def test_load_testset_dev_va_test_la_hai_bo_khac_nhau() -> None:
    dev, test = ev.load_testset("dev"), ev.load_testset("test")
    assert all(r["id"].startswith("q") for r in dev)
    assert all(r["id"].startswith("t") for r in test)
    assert not {r["question"] for r in dev} & {r["question"] for r in test}


def test_dev_chi_sua_bang_dev() -> None:
    out = ev.upsert_ablation_row(ABLATION, "dev", "V0 dense", "| V0 dense | 0.9 | new.json |")
    head, _, tail = out.partition(ev.ABLATION_TEST_HEADING)
    assert "| V0 dense | 0.9 | new.json |" in head
    assert "V0 dense" not in tail


def test_test_them_dong_vao_bang_test_rong() -> None:
    out = ev.upsert_ablation_row(ABLATION, "test", "V0 dense", "| V0 dense | 0.5 | t_v0.json |")
    head, _, tail = out.partition(ev.ABLATION_TEST_HEADING)
    assert "| V0 dense | 0.1 | dev_v0.json |" in head  # bảng dev giữ nguyên
    table, _, history = tail.partition("## Lịch sử chạy")
    assert table.strip().splitlines()[-1] == "| V0 dense | 0.5 | t_v0.json |"
    assert "V0 dense" not in history


def test_test_thay_dong_da_co_va_them_variant_moi() -> None:
    out = ev.upsert_ablation_row(ABLATION, "test", "V0 dense", "| V0 dense | 0.5 | a.json |")
    out = ev.upsert_ablation_row(out, "test", "V3 +rerank", "| V3 +rerank | 0.6 | b.json |")
    out = ev.upsert_ablation_row(out, "test", "V0 dense", "| V0 dense | 0.7 | c.json |")
    tail = out.partition(ev.ABLATION_TEST_HEADING)[2].partition("## Lịch sử chạy")[0]
    rows = [line for line in tail.splitlines() if line.startswith("| V") and not line.startswith("| Variant")]
    assert rows == ["| V0 dense | 0.7 | c.json |", "| V3 +rerank | 0.6 | b.json |"]


def test_tao_bang_test_neu_chua_co() -> None:
    no_test = ABLATION.replace(ABLATION.partition(ev.ABLATION_TEST_HEADING)[1]
                               + ABLATION.partition(ev.ABLATION_TEST_HEADING)[2].partition("## Lịch sử chạy")[0], "")
    assert ev.ABLATION_TEST_HEADING not in no_test
    out = ev.upsert_ablation_row(no_test, "test", "V0 dense", "| V0 dense | 0.5 | t.json |")
    assert out.index(ev.ABLATION_TEST_HEADING) < out.index("## Lịch sử chạy")
    assert "| V0 dense | 0.5 | t.json |" in out.partition(ev.ABLATION_TEST_HEADING)[2]
