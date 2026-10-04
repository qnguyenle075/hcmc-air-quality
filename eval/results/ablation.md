# Ablation RAG (V0 → V4)

Metric RAGAS tính trên các câu trong phạm vi (không gồm out_of_scope). OOS refusal = tỉ lệ từ chối đúng ở nhóm out_of_scope; False refusal = tỉ lệ từ chối nhầm ở câu trong phạm vi.

| Variant | Ctx Precision | Ctx Recall | Faithfulness | Answer Rel. | OOS refusal | False refusal | Latency (s) | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| V0 dense | 0.710 | 0.891 | 0.887 | 0.817 | 1.000 | 0.087 | 17.146 | 20261003-1550_v0.json |
| V1 +hybrid | 0.691 | 0.746 | 0.786 | 0.706 | 1.000 | 0.217 | 16.312 | 20261003-1654_v1.json |
| V2 +multi-query | 0.861 | 0.833 | 0.869 | 0.810 | 1.000 | 0.043 | 20.022 | 20261004-1240_v2.json |
| V3 +rerank | | | | | | | | |
| V4 +prompt | | | | | | | | |

## Lịch sử chạy

| Thời điểm | Variant | Số câu | Generator | Judge | Kết quả |
|---|---|---|---|---|---|
| 2026-10-03 15:50 | v0 | 28 | openai/gpt-oss-120b | qwen-3.8-27b | 20261003-1550_v0.json |
| 2026-10-03 16:54 | v1 | 28 | openai/gpt-oss-120b | qwen-3.8-27b | 20261003-1654_v1.json |
| 2026-10-04 12:40 | v2 | 28 | openai/gpt-oss-120b | qwen-3.8-27b | 20261004-1240_v2.json |
