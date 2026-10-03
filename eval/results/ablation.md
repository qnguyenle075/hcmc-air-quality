# Ablation RAG (V0 → V4)

Metric RAGAS tính trên các câu trong phạm vi (không gồm out_of_scope). OOS refusal = tỉ lệ từ chối đúng ở nhóm out_of_scope; False refusal = tỉ lệ từ chối nhầm ở câu trong phạm vi.

| Variant | Ctx Precision | Ctx Recall | Faithfulness | Answer Rel. | OOS refusal | False refusal | Latency (s) | Ghi chú |
|---|---|---|---|---|---|---|---|---|
| V0 dense | 0.710 | 0.891 | 0.887 | 0.817 | 1.000 | 0.087 | 17.146 | 20261003-1550_v0.json |
| V1 +hybrid | | | | | | | | |
| V2 +multi-query | | | | | | | | |
| V3 +rerank | | | | | | | | |
| V4 +prompt | | | | | | | | |

## Lịch sử chạy

| Thời điểm | Variant | Số câu | Generator | Judge | Kết quả |
|---|---|---|---|---|---|
| 2026-10-03 15:50 | v0 | 28 | openai/gpt-oss-120b | qwen-3.8-27b | 20261003-1550_v0.json |
