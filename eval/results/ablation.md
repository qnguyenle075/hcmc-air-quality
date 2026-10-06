# Ablation RAG (V0 → V4)

Bảng ablation đo trên bộ **dev** (`rag_devset.jsonl`, 28 câu). Metric RAGAS tính trên các câu trong phạm vi (không gồm out_of_scope). OOS refusal = tỉ lệ từ chối đúng ở nhóm out_of_scope; False refusal = tỉ lệ từ chối nhầm ở câu trong phạm vi.

| Variant | Ctx Precision | Ctx Recall | Faithfulness | Answer Rel. | OOS refusal | False refusal | Latency (s) | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| V0 dense | 0.710 | 0.891 | 0.887 | 0.817 | 1.000 | 0.087 | 17.146 | 20261003-1550_v0.json |
| V1 +hybrid | 0.691 | 0.746 | 0.786 | 0.706 | 1.000 | 0.217 | 16.312 | 20261003-1654_v1.json |
| V2 +multi-query | 0.861 | 0.833 | 0.869 | 0.810 | 1.000 | 0.043 | 20.022 | 20261004-1240_v2.json |
| V3 +rerank | 0.810 | 0.920 | 0.871 | 0.842 | 1.000 | 0.043 | 20.268 | 20261005-1346_v3.json |
| V4 +prompt | 0.812 | 0.928 | 0.922 | 0.808 | 1.000 | 0.043 | 22.312 | Trung bình 2 lần: 20261005-1434_v4.json, 20261006-1509_v4.json |

## Bộ test (đo một lần cuối Phase 1)

Cùng metric như bảng dev, trên bộ test (`rag_testset.jsonl`, không dùng để chỉnh hệ thống).

| Variant | Ctx Precision | Ctx Recall | Faithfulness | Answer Rel. | OOS refusal | False refusal | Latency (s) | Ghi chú |
|---|---|---|---|---|---|---|---|---|

## Lịch sử chạy

Các lượt trước 2026-10-05 chạy trên file `rag_testset.jsonl` cũ — nay là `rag_devset.jsonl` (cùng 28 câu).

| Thời điểm | Split | Variant | Số câu | Generator | Judge | Kết quả |
|---|---|---|---|---|---|---|
| 2026-10-03 15:50 | dev | v0 | 28 | openai/gpt-oss-120b | qwen-3.8-27b | 20261003-1550_v0.json |
| 2026-10-03 16:54 | dev | v1 | 28 | openai/gpt-oss-120b | qwen-3.8-27b | 20261003-1654_v1.json |
| 2026-10-04 12:40 | dev | v2 | 28 | openai/gpt-oss-120b | qwen-3.8-27b | 20261004-1240_v2.json |
| 2026-10-05 13:46 | dev | v3 | 28 | openai/gpt-oss-120b | qwen-3.8-27b | 20261005-1346_v3.json |
| 2026-10-05 14:34 | dev | v4 | 28 | openai/gpt-oss-120b | qwen-3.8-27b | 20261005-1434_v4.json |
| 2026-10-06 15:09 | dev | v4 | 28 | openai/gpt-oss-120b | qwen-3.8-27b | 20261006-1509_v4.json |
