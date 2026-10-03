# Ghi chú xử lý dữ liệu (data/processed/)

Quy trình: `uv run python -m src.ingest.extract` → text tự động trong `data/processed/auto/*.txt`
(pymupdf + `clean.py`: NFC, sửa ký tự font Symbol, bỏ header/footer, nối dòng). Các file `.md`
trong thư mục này được **chỉnh tay** từ output đó và **đối chiếu bằng mắt** với ảnh trang PDF
(render 90–110 dpi bằng pymupdf). File `.md` là nguồn chuẩn cho bước chunk/embed.

Quy ước: nội dung trong `[...]` hoặc `[Ghi chú biên soạn: ...]` là phần do người xử lý thêm vào,
**không có trong văn bản gốc**.

Ngày xử lý: 2026-10-03.

## Lỗi trích xuất chung

| Lỗi | Nguyên nhân | Cách xử lý |
|---|---|---|
| Ký hiệu µ thành ký tự U+F06D | Font Symbol (QCVN 05:2023) | `clean.fix_symbol_font` đổi U+F06D → `µ` (U+00B5) |
| µ (U+00B5) và μ (U+03BC) lẫn lộn | Khác font giữa các tài liệu | `clean.normalize_unicode` thống nhất về U+00B5 |
| Bảng bị vỡ thành mỗi ô một dòng | pymupdf đọc theo luồng text | Dựng lại bảng markdown thủ công, đối chiếu ảnh trang |

## who_aqg_2021.md (WHO AQG 2021 — Executive summary, 16 trang PDF)

- Bỏ trang 1–4 (bìa, trang trống, trang license/cataloguing), trang 15 (trống), trang 16 (bìa sau,
  lặp lại nội dung tóm tắt), và danh sách tài liệu tham khảo (trang 14). Thông tin license và
  trích dẫn được giữ trong front matter.
- Chú thích chân trang (footnote 1–4: định nghĩa PM2.5/PM10, BC/EC, UFP) được đưa vào thân câu
  tương ứng trong ngoặc / gạch ngang.
- Bảng 0.1 (trang 8): text trích xuất bị đảo thứ tự cột → dựng lại; **đã kiểm tra bằng mắt từng ô**
  với ảnh trang 8. Bảng 0.2 (trang 9) dựng lại từ text, khớp thứ tự.
- Bảng 0.3 (trang 11): giữ dạng danh sách đánh số theo nhóm BC/EC, UFP, SDS.
- Heading "Air quality guidelines with short averaging times that remain valid" do người xử lý
  đặt (bản gốc không có heading cho đoạn này) để tách mục khi chunk.

## qcvn_05_2023.md (QCVN 05:2023/BTNMT, 12 trang PDF)

- Bản PDF có chữ ký số (VGCA), lấy từ trang Sở TN&MT Lạng Sơn (không tìm thấy bản trên scem.gov.vn).
- Bỏ trang 1 (bìa). Header trang 3 trong bản gốc ghi nhầm "QCVN …. : 2022/BTNMT" (sót từ bản dự thảo) → bỏ.
- Lời nói đầu: số hiệu và ngày của Thông tư để trống trong bản ký số → điền `[01]`, `[13]`, `[03]`
  theo Thông tư 01/2023/TT-BTNMT ngày 13/03/2023 (nguồn: datafiles.chinhphu.vn, luatmoitruong.vn).
- Bảng 1 (trang 4–5, vắt qua 2 trang): dựng lại; **đã kiểm tra bằng mắt** với ảnh trang 5.
  Ô PM2,5 – trung bình 24 giờ trong bản gốc chia đôi "50 | 45(*)" → ghi `50; 45(*)` + ghi chú biên soạn.
- Bảng 2 (trang 5–7): các chất có 2 thời gian trung bình (ô gộp) được tách thành 2 dòng lặp lại tên chất.
- Bảng 3 (trang 7–12): dựng lại, các phương pháp nối bằng dấu `;`. Giữ nguyên lỗi chính tả gốc
  ("Naphtalene", "NIOSH 7 Method 300", "Tetrachloethylene").
- Thêm heading cấp 3 cho mục 2.1, 2.2, 3.1, 3.2 (bản gốc là đoạn văn đánh số) để chunk theo điều khoản.

## qd_1459_vn_aqi.md (QĐ 1459/QĐ-TCMT, 11 trang PDF)

- Bản PDF có chữ ký số, lấy từ vea.mae.gov.vn (Cục Môi trường). Số hiệu "1459" và ngày "12/11"
  nằm ở lớp chữ ký → ghi vào front matter và câu dẫn đầu.
- Trang 1 (Quyết định): giữ Điều 1–3, bỏ phần căn cứ pháp lý, nơi nhận, chữ ký.
- **Công thức (trang 4, 5, 7, 8, 9) bị mất hoặc rối hoàn toàn** khi trích xuất (là đối tượng
  phương trình/ảnh). Đã **chép lại thủ công từ ảnh trang** dưới dạng text thuần:
  - Trang 4: w* = Cmin/Cmax; điều kiện w; 2 công thức Nowcast.
  - Trang 5, 7: Công thức 1 và Công thức 2 (nội suy tuyến tính AQIx).
  - Trang 8–9: phần tính toán mẫu (Nowcast = 20,3; AQI^h = 60; AQI^d = 110).
- Bảng 2 (breakpoint BPi, trang 5): text trích xuất đúng thứ tự; **đã kiểm tra bằng mắt từng ô**.
- Bảng ví dụ O3 (trang 6–7): chuyển từ bảng lưới sang dạng liệt kê "giờ: giá trị". Đã tự kiểm tra
  giá trị TB8h lúc 1:00 = trung bình 8 giá trị 18:00→1:00 = 17,975 ≈ 18,0 (khớp bản gốc).
- Bảng 5 (trang 11): dòng 301–500 (Nguy hại) là **ô gộp cho cả hai nhóm** (đã xem ảnh) → lặp lại
  nội dung ở cả hai cột kèm ghi chú biên soạn.
- Nhận xét công thức Nowcast khi w = 1/2: bản gốc viết Nowcast = Σ (1/2)^i·ci (không chia tổng trọng
  số), trong khi trường hợp w > 1/2 dùng Σ w^(i−1)·ci / Σ w^(i−1). Hai cách viết **tương đương
  (gần đúng)** vì Σ(i=1→12) (1/2)^(i−1) ≈ 2. Giữ nguyên cách viết gốc trong .md; khi cài đặt
  `vn_aqi.py` sẽ dùng công thức chuẩn hóa chung và ghi chú lại.
- **Kiểm tra lại phần tính toán mẫu bằng code** (2026-10-03): Nowcast = 20,3 ✓; AQI^d: 45, 36, 65,
  110 ✓; AQI^h: O3 = 43 ✓, PM2.5 = 41 ✓, nhưng **NO2: bản gốc ghi 60, tính lại ra 59,35** (làm tròn
  → 59; AQI^h tổng hợp vẫn = 60 nếu làm tròn lên). Không có quy tắc làm tròn nào (round/ceil) khớp
  đồng thời mọi ví dụ (ceil cho NO2 ngày ra 66 ≠ 65) → nhiều khả năng là sai số trong bản gốc.
  Giữ nguyên số "60" của bản gốc trong .md; unit test của `vn_aqi.py` cần ghi chú trường hợp này.

## data/curated/aqi_health_categories.md

- Tự soạn, ghép **nguyên văn** Bảng 1, 4, 5 của QĐ 1459 theo từng mức VN_AQI (mỗi mức một heading
  → một chunk hoàn chỉnh).
- **Không thêm** các nội dung mà CLAUDE.md từng gợi ý nhưng văn bản gốc không có (phụ nữ mang thai
  trong nhóm nhạy cảm, máy lọc không khí, ...). Định nghĩa nhóm nhạy cảm lấy đúng Bảng 4.
- Trạng thái: **Đã duyệt (2026-10-03)**. Người dùng chốt: chỉ dùng nội dung văn bản chính phủ, không bổ sung.
