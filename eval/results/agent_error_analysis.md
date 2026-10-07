# Phân tích lỗi agent (Phase 4)

Lượt chạy: `20261007-1857_agent.json` (25 câu, generator `openai/gpt-oss-120b`, RAG tool V4, judge `qwen-3.8-27b`).
Số liệu tổng hợp: xem `agent_eval.md`.

Ngoài chấm theo luật (agentevals) và LLM-as-judge, cả 25 câu trả lời được **đọc tay** (Claude Code, 2026-10-07),
đối chiếu với output của tool trong từng hội thoại. Phần đọc tay là đánh giá của một người chấm, chỉ dùng để phân
loại lỗi, không thay số liệu tự động.

## Tóm tắt

| Nguồn chấm | Đạt |
|---|---|
| Trajectory match (luật) | 24/25 (0.960) |
| Judge pass (LLM) | 23/24 (0.958), 1 câu judge lỗi |
| Đọc tay — không có lỗi nặng | 19/25 (0.76) |

Lỗi nặng = sai luồng tool, khuyến nghị ngoài tài liệu, thiếu thông tin mà tài liệu có, trả lời sai ngôn ngữ.
Lỗi nhẹ = lẫn ngôn ngữ một phần, diễn giải thêm nhỏ, tên địa danh.

Không có loop; nhóm `out_of_scope` từ chối đúng 5/5, nhóm `ambiguous` hỏi lại địa điểm 3/3; câu `follow_up`
dùng lại địa điểm từ lượt trước, không geocode lại (3/3).

## Phân loại theo nguyên nhân

| Nguyên nhân | Câu | Mức |
|---|---|---|
| Chọn tool sai (agent) | a004 | Nặng |
| Tool trả kết quả sai (geocode / air quality) | — | — |
| Lỗi do RAG (retrieval) | a012 | Nặng |
| Tổng hợp câu trả lời — khuyến nghị ngoài tài liệu | a005, a020 | Nặng |
| Tổng hợp câu trả lời — sai ngôn ngữ | a002, a005, a024 | Nặng |
| Tổng hợp câu trả lời — lẫn ngôn ngữ một phần | a021 | Nhẹ |
| Tổng hợp câu trả lời — diễn giải thêm nhỏ | a003, a011 | Nhẹ |
| Tổng hợp câu trả lời — tên địa danh | a001 | Nhẹ |

(a005 có 2 lỗi nặng, tính 1 câu.)

## Chi tiết

### Chọn tool sai

- **a004** "Q3 không khí giờ sao, người bị hen suyễn có nên ra ngoài không?" — kỳ vọng
  geocode → get_air_quality → retrieve_health_guideline, thực tế không gọi tool nào. LLM không nhận ra "Q3" là
  Quận 3 nên hỏi lại địa điểm. `geocode_address` đã chuẩn hóa "Q3" → "Quận 3" nhưng không được gọi tới.

### Lỗi do RAG

- **a012** "Ngưỡng PM2.5 trung bình 24 giờ của WHO và QCVN khác nhau thế nào?" — luồng tool đúng; RAG trả đúng
  QCVN (50 → 45 µg/Nm³) nhưng nói "Tài liệu không đề cập đến WHO 2021", dù Bảng 0.1 của WHO có trong corpus. Agent
  chuyển lại trung thực. Cùng kiểu lỗi đã ghi ở RAG (`error_analysis.md`, q022): câu hỏi gộp 2 nguồn chỉ lấy được
  một nửa context; query agent gửi là tiếng Việt trong khi tài liệu WHO là tiếng Anh.

### Khuyến nghị ngoài tài liệu

- **a005** "I live in Hoc Mon. Should I keep the windows open today?" — RAG trả "Tài liệu không đề cập đến việc mở
  cửa sổ", agent vẫn kết luận "người bình thường có thể mở cửa sổ mà không gây lo ngại sức khỏe". Judge bắt được.
- **a020** (follow-up) "Vậy có nên mở cửa sổ không?" — agent có nói tài liệu không có khuyến nghị, nhưng vẫn thêm
  "bạn có thể mở cửa sổ nếu cảm thấy thoải mái". Nhẹ hơn a005 nhưng cùng loại; judge không bắt.

### Sai ngôn ngữ

- **a002**, **a005** (câu `full` tiếng Anh) — toàn bộ câu trả lời bằng tiếng Việt. Agent gửi query RAG bằng tiếng
  Việt, RAG trả lời tiếng Việt, agent chép lại theo ngôn ngữ đó.
- **a024** "Should I go jogging now?" — không gọi tool (đúng), nhưng hỏi lại địa điểm bằng tiếng Việt.
- **a021** (follow-up tiếng Anh) — thân câu trả lời tiếng Anh, dòng nguồn và disclaimer tiếng Việt (nhẹ).
- Câu tiếng Anh không gọi RAG (a007, a010, a016, a019) trả lời đúng tiếng Anh.

### Lỗi nhẹ khác

- **a003** thêm "tránh hoạt động mạnh nếu xuất hiện bất thường"; **a011** thêm "để giảm tiếp xúc với bụi và khí
  độc" — không có trong output RAG, không đổi ý khuyến nghị.
- **a001** ghi "Phường Hạnh Thông, Quận Gò Vấp" — `ward` trả về là "Phường Hạnh Thông", agent tự thêm cấp quận cũ
  (đã bỏ sau sáp nhập).

## Hạn chế của judge

- **a004**: judge chấm đạt vì cũng không nhận ra "Q3" là địa điểm → đạt sai.
- **a002, a024**: rubric có tiêu chí "trả lời cùng ngôn ngữ" nhưng judge không bắt; **a020** cũng không bắt.
- **a018**: judge lỗi `LengthFinishReasonError` (sinh hết 8192 token, không ra JSON) → không có điểm; agent từ chối
  đúng (đọc tay).
- → Judge pass 0.958 đánh giá cao hơn thực tế; số đọc tay 0.76 cho thấy các lỗi judge bỏ sót tập trung ở tiêu chí
  ngôn ngữ và "không khuyên khi tài liệu không có".

## Ghi chú khác

- **a017 (Vũng Tàu), a018 (Dĩ An)**: agent từ chối dựa trên hiểu biết sẵn có, không gọi `geocode_address` để kiểm
  tra bằng polygon ranh giới. Testset cho phép (`subset`), kết quả đúng, nhưng phạm vi địa lý không được tool xác minh.
- Chi phí: lượt chạy tốn ~175K token Groq (gần giới hạn 200K/ngày) + ~33K token judge (Cerebras).

## Hướng sửa (chưa áp dụng)

| Lỗi | Hướng sửa |
|---|---|
| a004 viết tắt địa danh | System prompt: câu có tên / viết tắt địa danh (Q1–Q12, "Thủ Đức"...) → gọi `geocode_address`, không hỏi lại |
| Sai ngôn ngữ | System prompt: luôn trả lời theo ngôn ngữ câu hỏi gốc, dịch nội dung RAG nếu khác ngôn ngữ |
| Khuyến nghị ngoài tài liệu | System prompt: RAG nói tài liệu không có → chỉ nói vậy, không tự kết luận nên / không nên |
| a012 (RAG) | Ở tầng RAG: tách câu hỏi nhiều nguồn, hoặc query tiếng Anh cho tài liệu WHO |
| Judge | Tăng `max_tokens` / retry khi `LengthFinishReasonError`; tách tiêu chí ngôn ngữ thành kiểm tra theo luật |

Lưu ý: bộ agent eval không tách dev/test. Nếu sửa prompt theo các câu trên rồi đo lại trên chính bộ này, kết quả sẽ
lạc quan — cần ghi rõ khi báo cáo.
