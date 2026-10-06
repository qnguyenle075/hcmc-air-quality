# Phân tích lỗi RAG — V0 → V4

Ngày phân tích: 2026-10-04. Dữ liệu: `20261003-1550_v0.json`, `20261003-1654_v1.json`, `20261004-1240_v2.json`
(28 câu, generator `openai/gpt-oss-120b`, judge `qwen-3.8-27b`). V3 chưa có kết quả tại thời điểm viết.

Phương pháp:
- So điểm RAGAS từng câu giữa các variant.
- Đo retrieval không dùng LLM: chunk "đúng" = chunk chứa nguyên văn một `reference_context` (so khớp sau chuẩn hóa NFC,
  chữ thường, gộp khoảng trắng); ghi hạng của chunk đúng trong BM25, dense và hybrid 0.5/0.5 (corpus 53 chunk).
- Đọc câu trả lời và đối chiếu với nội dung chunk.

## 1. Vì sao V1 (hybrid) tụt so với V0

Số câu (trên 23 câu trong phạm vi) có chunk đúng nằm trong top 5:

| Retriever | Top 5 có chunk đúng |
|---|---|
| Dense (bge-m3) | 21/23 |
| BM25 | 15/23 |
| Hybrid 0.5/0.5 (V1) | 19/23 |

Các câu V1 lấy sai đều là **câu hỏi khác ngôn ngữ với tài liệu chứa đáp án**:

| Câu | Ngôn ngữ câu hỏi → tài liệu | Hạng BM25 | Hạng dense | Hậu quả ở V1 |
|---|---|---|---|---|
| q003 | EN → QCVN (VI) | 13 | 3 | Từ chối nhầm |
| q014 | EN → bảng VN_AQI (VI) | 51/53 | 2 | Từ chối nhầm |
| q017 | EN → bảng VN_AQI (VI) | 48 | 2 | Chunk đúng tụt xuống hạng 4, precision giảm |
| q022 | EN → QCVN + WHO | 14 | 15 | Từ chối (cả V0, V1, V2) |
| q007 | VI → WHO (EN) | 30 | 1 | Precision giảm |
| q004 | VI đời thường ("Ozone… 8 tiếng") → văn bản ghi "O3… 8 giờ" | 9 | 3 | Từ chối nhầm |

BM25 so khớp từ vựng: câu tiếng Anh chỉ khớp chunk WHO (tiếng Anh) và đẩy chunk tiếng Việt chứa đáp án ra khỏi top 5.
Ba câu từ chối nhầm (q003, q004, q014) bị 0 ở mọi metric, kéo trung bình V1 xuống.

BM25 **có ích** khi câu hỏi cùng ngôn ngữ và dùng thuật ngữ của văn bản:

| Câu | Hạng dense | Hạng BM25 |
|---|---|---|
| q010 | 5 | 2 |
| q015 | 8 | 2 |
| q018 | 7 | 1 |

**V2 bù được lỗi này:** prompt multi-query (`src/rag/prompts.py`) yêu cầu ít nhất một biến thể tiếng Việt và một biến thể
tiếng Anh, nên BM25 luôn có một truy vấn cùng ngôn ngữ với tài liệu. q003, q004, q014 trả lời đúng lại ở V2;
false refusal giảm từ 0.217 (V1) xuống 0.043 (V2).

**Quyết định (2026-10-04):** giữ trọng số hybrid 0.5/0.5, không tinh chỉnh. Giảm trọng số BM25 chỉ chữa triệu chứng,
làm mất phần lợi ở các câu cùng ngôn ngữ, và buộc đo lại V1–V3. Thử nhiều bộ trọng số trên bộ test cũng là tune theo
test set (mục 4.5 CLAUDE.md).

## 2. Lỗi còn lại ở V2

| Loại lỗi | Câu | Chi tiết |
|---|---|---|
| Generator thêm nội dung ngoài context | q013, q018 | q013 thêm "cân nhắc thực hiện các bài tập nhẹ hơn"; q018 thêm "các bệnh nền khác". Không cụm nào có trong corpus |
| Generator lẫn mức / nhóm người | q014 | AQI 250 (Rất xấu): văn bản khuyên nhóm nhạy cảm "Nên ở trong nhà và giảm hoạt động mạnh"; "đeo khẩu trang đạt tiêu chuẩn" là khuyến nghị cho người bình thường ở mức này, nhưng câu trả lời gán cho người già. Câu trả lời cũng không nêu tên mức |
| Dùng nhãn US EPA | q020 | Câu trả lời nhắc "Good, Moderate, Unhealthy…" (CLAUDE.md cấm). Bảng mức VN_AQI (`qd_1459_vn_aqi-004`, hạng dense 14) không được lấy về |
| Từ chối cả câu thay vì trả lời phần có thông tin | q022 | Có bảng WHO, thiếu bảng QCVN → từ chối toàn bộ (V0, V1, V2 đều vậy) |
| Retrieval thiếu ý phụ ở câu nhiều ý | q020, q023 | q023: chunk về yêu cầu bắt buộc PM10/PM2.5 (`qd_1459_vn_aqi-005`) ở hạng dense 8 |
| Judge chấm nhiễu | q001, q016 | q001 trả lời đúng hoàn toàn nhưng faithfulness 0.5. q016 trả lời đúng, context recall 0 dù chunk `qd_1459_vn_aqi-014` đã lấy về chứa đúng câu đáp án |

## 3. Đề xuất cho V4 (prompt)

Chỉ thực hiện sau khi V3 đo xong, chờ người dùng duyệt:

1. Chỉ dùng câu chữ có trong context; không thêm ví dụ, diễn giải hay lời khuyên ngoài văn bản.
2. Câu hỏi theo mức AQI: nêu khoảng giá trị và tên mức VN_AQI trước, sau đó trích khuyến nghị đúng dòng của mức đó
   và đúng nhóm người được hỏi.
3. Không dùng nhãn US EPA (Good, Moderate…), chỉ dùng tên mức VN_AQI.
4. Context chỉ đủ một phần câu hỏi → trả lời phần có thông tin, nói rõ phần tài liệu không có; không từ chối cả câu.

Lỗi retrieval câu nhiều ý (q020, q023) thuộc phạm vi V3 (rerank với fetch_k=20), không xử lý bằng prompt.

## 4. V4 (variant chốt) — lỗi còn lại

Ngày phân tích: 2026-10-06. Dữ liệu: `20261005-1346_v3.json`, hai lượt V4 `20261005-1434_v4.json` (V4a) và
`20261006-1509_v4.json` (V4b), bộ dev 28 câu. V4 dùng cùng retrieval với V3 (chỉ khác prompt), nên context
precision / recall từng câu gần như trùng V3; khác biệt nằm ở câu trả lời.

**Quyết định (2026-10-06):** chốt V4 làm variant mặc định. Faithfulness tăng 0.871 → 0.922 (trung bình 2 lượt);
đổi lại answer relevancy giảm 0.842 → 0.808 (xem mục 4.2).

### 4.1 Lỗi đã sửa được so với V2/V3

| Câu | V3 | V4 (cả 2 lượt) |
|---|---|---|
| q011 | Thêm "có thể cần dùng thuốc hen suyễn thường xuyên hơn" — lấy từ mức 101–150, sai mức | Chỉ trích đúng dòng mức 151–200 cho nhóm nhạy cảm |
| q014 | Faithfulness 0.25 | Nêu khoảng + tên mức, khuyến nghị đúng nhóm; faithfulness 1.00 |
| q020 | Dùng nhãn US EPA (lỗi ở V2) | Dùng tên mức VN_AQI; phần thiếu nói bằng câu "không đề cập" (trả lời một phần, đúng quy tắc mới) |

### 4.2 Answer relevancy giảm — đánh đổi do prompt, không phải lỗi sai nội dung

9 câu giảm answer relevancy > 0.05 ở **cả hai** lượt V4 (q002, q003, q005, q009, q012, q013, q015, q017, q021);
3 câu tăng (q010, q014, q023). Câu trả lời V4 ngắn hơn (trung vị 170 → ~148 ký tự) và không còn câu kết luận
trả lời thẳng câu hỏi đời thường:

| Câu | V3 | V4 |
|---|---|---|
| q012 (AQI 75, người khỏe chạy bộ được không) | "...tự do thực hiện các hoạt động ngoài trời, **bao gồm việc chạy bộ**" | "Tự do thực hiện các hoạt động ngoài trời" |
| q015 (AQI 320 có nên mở cửa sổ) | "...**Vì vậy, không nên mở cửa sổ**" | Chỉ trích "đóng cửa ra vào và cửa sổ" |

Nguyên nhân: quy tắc V4 "không thêm ... kết luận mà context không nêu" khiến generator bỏ bước nối khuyến nghị với
câu hỏi. RAGAS answer relevancy sinh ngược câu hỏi từ câu trả lời, nên câu trả lời thiếu ý "chạy bộ"/"mở cửa sổ" bị
điểm thấp hơn. Nội dung V4 vẫn đúng văn bản; đây là đánh đổi faithfulness ↔ relevancy.

### 4.3 Lỗi sai nội dung còn lại (cần lưu ý khi ghép vào agent)

| Loại lỗi | Câu | Chi tiết |
|---|---|---|
| Đọc nhầm cột bảng | q007 | Cả hai lượt V4 trả **40 µg/m³** (cột Interim target 1) thay vì **10 µg/m³** (cột AQG level) cho NO2 trung bình năm, dù Bảng 0.1 WHO có trong context (`who_aqg_2021-004`). V3 trả đúng (1 lượt). Judge bắt được (faithfulness 0) |
| Không biết ngày hiện tại | q001 | Hỏi giới hạn PM2,5 24 giờ "hiện nay": cả hai lượt V4 trả 50 µg/Nm³ (giá trị trước 01/01/2026) thay vì 45. Prompt không có ngày hiện tại nên generator không xác định được giá trị nào đang hiệu lực. **Judge không bắt được** (faithfulness 1.00 vì 50 có trong context) |
| Từ chối cả câu dù có một nửa context | q022 | Retrieval lấy được dòng NO2 của QCVN (`qcvn_05_2023-004`) nhưng không lấy được Bảng 0.1 WHO; generator vẫn từ chối cả câu, không theo quy tắc "trả lời một phần" của V4. Lỗi ở cả V0–V4 |
| Retrieval thiếu ý phụ | q019, q023 | Context recall 0.50–0.67, như V3 — rerank chưa kéo được chunk ý phụ lên top 5 |
| Judge chấm nhiễu | q011, q013, q018 | Câu trả lời V4 trích gần nguyên văn nhưng faithfulness dao động giữa 2 lượt (q011: 0.60 / 0.25; q013: 1.00 / 0.75; q018: 0.80 / 1.00) |

### 4.4 Đề xuất (chưa thực hiện, chờ người dùng duyệt)

- **Agent (Phase 3):** đưa ngày hiện tại vào system prompt agent / prompt RAG để xử lý câu hỏi "hiện nay" với các
  giá trị có hiệu lực theo mốc thời gian (q001).
- Không chỉnh thêm prompt RAG trong Phase 1: V4 đã chốt, bộ test đo trên V4 nguyên trạng. Lỗi q007, q022 ghi vào hạn
  chế đã biết.
