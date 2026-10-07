# Đánh giá agent (Phase 4)

Bộ câu hỏi: `eval/datasets/agent_testset.jsonl`. Trajectory match / tool selection / args hints chấm theo luật (agentevals); Judge pass = LLM-as-judge không reference (rubric trong `eval/run_agent_eval.py`). Bước = số tool call ở lượt được chấm. Câu `follow_up` chỉ chấm lượt cuối.

Phân tích lỗi (phân loại nguyên nhân + đọc tay 25 câu): xem `agent_error_analysis.md`.

## Các lượt chạy

| Thời điểm | Số câu | Trajectory match | Tool selection | Args hints | Judge pass | Bước TB / tối thiểu | Loop | Latency TB (s) | Generator | Judge | Kết quả |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-07 18:57 | 25 | 0.960 | 0.960 | 1.000 | 0.958 | 1.12 / 1.24 | 0.000 | 14.9 | openai/gpt-oss-120b | qwen-3.8-27b | 20261007-1857_agent.json |

## Chi tiết từng lượt

### 2026-10-07 18:57 — 20261007-1857_agent.json

| Nhóm | Số câu | Trajectory match | Tool selection | Args hints | Judge pass | Bước TB / tối thiểu | Latency TB (s) |
|---|---|---|---|---|---|---|---|
| full | 5 | 0.800 | 0.800 | 1.000 | 0.800 | 2.40 / 3.00 | 26.6 |
| aqi_only | 4 | 1.000 | 1.000 | 1.000 | 1.000 | 2.00 / 2.00 | 11.0 |
| knowledge_only | 5 | 1.000 | 1.000 | 1.000 | 1.000 | 0.80 / 0.80 | 4.7 |
| out_of_scope | 5 | 1.000 | 1.000 |  | 1.000 | 0.00 / 0.00 | 3.4 |
| follow_up | 3 | 1.000 | 1.000 | 1.000 | 1.000 | 1.33 / 1.33 | 50.6 |
| ambiguous | 3 | 1.000 | 1.000 |  | 1.000 | 0.00 / 0.00 | 1.3 |

Câu chưa đạt (trajectory, args hints, judge hoặc loop):

- **a004** (full): kỳ vọng ['geocode_address', 'get_air_quality', 'retrieve_health_guideline'] (strict), thực tế []; args hints=None; judge=True
- **a005** (full): kỳ vọng ['geocode_address', 'get_air_quality', 'retrieve_health_guideline'] (strict), thực tế ['geocode_address', 'get_air_quality', 'retrieve_health_guideline']; args hints=True; judge=False
