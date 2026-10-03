---
doc_id: qd_1459_vn_aqi
source_file: data/raw/qd_1459_tcmt_2019.pdf
source_url: https://vea.mae.gov.vn/Data/files/QD%201459%20TCMT%20ngay%2012_11_2019%20AQI(1).pdf
doc_title: Hướng dẫn kỹ thuật tính toán và công bố chỉ số chất lượng không khí Việt Nam (VN_AQI)
doc_number: 1459/QĐ-TCMT
issued_by: Tổng cục Môi trường (Bộ Tài nguyên và Môi trường)
issued_date: 2019-11-12
status_checked: 2026-10-03 — còn được Cục Môi trường (Bộ Nông nghiệp và Môi trường) đăng tải và trích dẫn, chưa thấy văn bản thay thế
language: vi
processing: Trích xuất pymupdf + chỉnh tay; công thức và bảng chép lại từ ảnh trang PDF (xem data/processed/NOTES.md)
---

# Quyết định 1459/QĐ-TCMT về việc ban hành Hướng dẫn kỹ thuật tính toán và công bố chỉ số chất lượng không khí Việt Nam (VN_AQI)

Ngày 12/11/2019, Tổng cục trưởng Tổng cục Môi trường ban hành Quyết định số 1459/QĐ-TCMT.

Điều 1. Ban hành kèm theo Quyết định này Hướng dẫn kỹ thuật tính toán và công bố chỉ số chất lượng không khí Việt Nam.

Điều 2. Quyết định này có hiệu lực kể từ ngày ban hành và thay thế Quyết định 878/QĐ-TCMT ngày 01 tháng 7 năm 2011 của Tổng cục trưởng Tổng cục Môi trường về việc ban hành Sổ tay hướng dẫn tính toán chỉ số chất lượng không khí.

Điều 3. Chánh văn phòng Tổng cục Môi trường, Vụ trưởng Vụ Quản lý chất lượng môi trường, Giám đốc Trung tâm Quan trắc môi trường miền Bắc, Thủ trưởng các đơn vị trực thuộc Tổng cục Môi trường và các tổ chức, cá nhân có liên quan chịu trách nhiệm thi hành Quyết định này.

# Hướng dẫn kỹ thuật tính toán và công bố chỉ số chất lượng không khí Việt Nam (VN_AQI)

## 1. Quy định chung

### 1.1. Mục đích

Văn bản này hướng dẫn việc tính toán và công bố chỉ số chất lượng không khí (VN_AQI) từ dữ liệu quan trắc của trạm quan trắc không khí tự động, liên tục.

### 1.2. Đối tượng sử dụng

Đối tượng sử dụng văn bản này là các cơ quan quản lý nhà nước về môi trường; các tổ chức, cá nhân có tham gia vào việc xây dựng, công bố thông tin về chất lượng môi trường cho cộng đồng tại Việt Nam.

### 1.3. Giải thích từ ngữ

1.3.1. Chỉ số chất lượng không khí Việt Nam (viết tắt là VN_AQI) là chỉ số được tính toán từ các thông số quan trắc các chất ô nhiễm trong không khí ở Việt Nam, nhằm cho biết tình trạng chất lượng không khí và mức độ ảnh hưởng đến sức khỏe con người, được biểu diễn qua một thang điểm.

1.3.2. AQI ngày (AQI^d) là giá trị AQI đại diện cho chất lượng không khí trong 1 ngày.

1.3.3. AQI giờ (AQI^h) là giá trị AQI đại diện cho chất lượng không khí trong 1 giờ.

1.3.4. AQI thông số (AQIx) là giá trị tính toán AQI cho từng thông số quan trắc. AQI thông số bao gồm AQI thông số theo ngày (AQIx^d) và AQI thông số theo giờ (AQIx^h).

1.3.5. Giá trị Nowcast là giá trị trung bình có trọng số được tính toán từ 12 giá trị trung bình 1 giờ gần nhất so với thời điểm tính toán. Giá trị Nowcast được tính toán cho thông số PM10 và PM2.5.

1.3.6. Trạm quan trắc không khí tự động liên tục là trạm quan trắc cố định hoặc di động và có khả năng đo tự động, liên tục các thông số về chất lượng không khí.

### 1.4. Đánh giá chỉ số chất lượng không khí

Chỉ số chất lượng không khí được tính theo thang điểm (khoảng giá trị AQI) tương ứng với biểu tượng và các màu sắc để cảnh báo chất lượng không khí và mức độ ảnh hưởng tới sức khỏe con người, cụ thể như sau:

Bảng 1: Khoảng giá trị AQI và đánh giá chất lượng không khí

| Khoảng giá trị AQI | Chất lượng không khí | Màu sắc | Mã màu RGB |
|---|---|---|---|
| 0 – 50 | Tốt | Xanh | 0;228;0 |
| 51 – 100 | Trung bình | Vàng | 255;255;0 |
| 101 – 150 | Kém | Da cam | 255;126;0 |
| 151 – 200 | Xấu | Đỏ | 255;0;0 |
| 201 – 300 | Rất xấu | Tím | 143;63;151 |
| 301 – 500 | Nguy hại | Nâu | 126;0;35 |

## 2. Hướng dẫn tính toán chỉ số chất lượng không khí

### 2.1. Yêu cầu đối với việc tính toán VN_AQI

a. Yêu cầu đối với số liệu sử dụng để tính toán VN_AQI

- Thiết bị quan trắc phải được kiểm soát chất lượng hệ thống và đo lường theo các quy định của pháp luật.
- Dữ liệu quan trắc được đưa vào tính toán đã qua xử lý, đảm bảo đã loại bỏ các giá trị sai lệch, đạt yêu cầu đối với quy trình quy phạm về đảm bảo và kiểm soát chất lượng số liệu.

b. Cách thức sử dụng số liệu để tính toán VN_AQI

- VN_AQI được tính toán bao gồm AQI giờ và AQI ngày. Số liệu sử dụng để tính toán VN_AQI là giá trị quan trắc trung bình 1 giờ, trung bình 8 giờ và trung bình 24 giờ.
- VN_AQI được tính toán cho dữ liệu của từng trạm quan trắc không khí tự động liên tục đối với môi trường không khí xung quanh.
- Đối với mỗi trạm quan trắc, AQIx được tính toán cho từng thông số quan trắc, giá trị AQI cuối cùng là giá trị lớn nhất trong các giá trị AQIx của mỗi thông số.
- Các thông số được sử dụng để tính VN_AQI bao gồm: SO2, CO, NO2, O3, PM10, và PM2.5.
- Phương pháp tính toán VN_AQI yêu cầu bắt buộc phải có tối thiểu 01 trong 02 thông số PM10, PM2.5 trong công thức tính.

### 2.2. Tính toán giá trị VN_AQI

#### 2.2.1. Tính toán giá trị AQI giờ (AQI^h)

Số liệu để tính toán AQI giờ là giá trị quan trắc trung bình 1 giờ.

**a. Tính giá trị Nowcast đối với thông số PM2.5 và PM10**

Gọi c1, c2, …, c12 là 12 giá trị quan trắc trung bình 1 giờ (với c1 là giá trị quan trắc trung bình 1 giờ hiện tại, c12 là giá trị quan trắc trung bình 1 giờ cách 12 giờ so với hiện tại).

Tính giá trị trọng số: w* = Cmin / Cmax

Trong đó: Cmin là giá trị nhỏ nhất trong số 12 giá trị trung bình 1 giờ; Cmax là giá trị lớn nhất trong số 12 giá trị trung bình 1 giờ.

- Nếu w* ≤ 1/2 thì lấy w = 1/2.
- Nếu w* > 1/2 thì lấy w = w*.

Trong trường hợp w > 1/2 thì giá trị Nowcast = [Σ(i=1→12) w^(i−1) × ci] / [Σ(i=1→12) w^(i−1)]

Trong trường hợp w = 1/2 thì Nowcast = (1/2)·c1 + (1/2)^2·c2 + … + (1/2)^12·c12

Chú ý:

- Nếu có ít nhất 2 trong 3 giá trị c1, c2, c3 có dữ liệu thì mới tính được giá trị Nowcast, ngược lại coi như "không có dữ liệu" (không tính được giá trị Nowcast).
- Nếu ci không có giá trị thì lấy w^(i−1) = 0.

**b. Tính giá trị AQI^h của từng thông số (AQIx)**

Giá trị AQI^h của các thông số SO2, CO, NO2, O3 được tính toán theo công thức 1, giá trị AQI^h của các thông số PM10, PM2.5 được tính toán theo công thức 2:

Công thức 1: AQIx = [(I(i+1) − Ii) / (BP(i+1) − BPi)] × (Cx − BPi) + Ii

Công thức 2: AQIx = [(I(i+1) − Ii) / (BP(i+1) − BPi)] × (Nowcastx − BPi) + Ii

Trong đó:

- AQIx: Giá trị AQI thông số của thông số x
- BPi: Nồng độ giới hạn dưới của giá trị thông số quan trắc được quy định trong Bảng 2 tương ứng với mức i
- BP(i+1): Nồng độ giới hạn trên của giá trị thông số quan trắc được quy định trong Bảng 2 tương ứng với mức i+1
- Ii: Giá trị AQI ở mức i đã cho trong bảng tương ứng với giá trị BPi
- I(i+1): Giá trị AQI ở mức i+1 cho trong bảng tương ứng với giá trị BP(i+1)
- Cx: Giá trị quan trắc trung bình 1 giờ của thông số x
- Nowcastx: Giá trị Nowcast được tính toán ở phần a

Bảng 2: Các giá trị BPi đối với các thông số (Đơn vị: µg/m3)

| i | Ii | O3 (1h) | O3 (8h) | CO | SO2 | NO2 | PM10 | PM2.5 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 50 | 160 | 100 | 10.000 | 125 | 100 | 50 | 25 |
| 3 | 100 | 200 | 120 | 30.000 | 350 | 200 | 150 | 50 |
| 4 | 150 | 300 | 170 | 45.000 | 550 | 700 | 250 | 80 |
| 5 | 200 | 400 | 210 | 60.000 | 800 | 1.200 | 350 | 150 |
| 6 | 300 | 800 | 400 | 90.000 | 1.600 | 2.350 | 420 | 250 |
| 7 | 400 | 1.000 | - | 120.000 | 2.100 | 3.100 | 500 | 350 |
| 8 | 500 | ≥1.200 | - | ≥150.000 | ≥2.630 | ≥3.850 | ≥600 | ≥500 |

Ghi chú:

- Tính toán AQI giờ (AQI^h) của O3 sử dụng giá trị BPi cho O3 (1h).
- Tính toán AQI ngày (AQI^d) của O3 sử dụng giá trị BPi cho O3 (1h) và O3 (8h).

**c. Giá trị AQI giờ tổng hợp**

Sau khi đã có giá trị AQIx của mỗi thông số, chọn giá trị AQI lớn nhất của các thông số để lấy làm giá trị AQI giờ tổng hợp.

AQI^h = max(AQIx)

Ghi chú: Giá trị AQI giờ được làm tròn thành số nguyên.

#### 2.2.2. Tính toán giá trị AQI ngày (AQI^d)

Giá trị AQI ngày được tính toán dựa trên các giá trị như sau:

- Thông số PM2.5 và PM10: giá trị trung bình 24 giờ.
- Thông số O3: giá trị trung bình 1 giờ lớn nhất trong ngày và giá trị trung bình 8 giờ lớn nhất trong ngày.
- Thông số SO2, NO2 và CO: giá trị trung bình 1 giờ lớn nhất trong ngày.

**a. Xác định giá trị trung bình 1 giờ lớn nhất trong ngày, giá trị trung bình 8 giờ lớn nhất trong ngày và giá trị trung bình 24 giờ**

- Giá trị trung bình 1 giờ lớn nhất trong ngày là giá trị lớn nhất trong số các giá trị quan trắc trung bình 1 giờ.
- Giá trị quan trắc trung bình 8 giờ lớn nhất trong ngày là giá trị lớn nhất trong số các giá trị trung bình 8 giờ. Giá trị trung bình 8 giờ là trung bình cộng các giá trị trung bình 1 giờ trong 8 giờ liên tiếp.

Ví dụ tính giá trị trung bình 8 giờ lớn nhất trong ngày: bảng số liệu trung bình 1 giờ của thông số O3 như sau:

O3 (TB1h), µg/m3 theo giờ: 18:00: 15,7; 19:00: 14,2; 20:00: 17,7; 21:00: 18,9; 22:00: 19,3; 23:00: 15,7; 00:00: 19,7; 1:00: 22,6; 2:00: 27,1; 3:00: 29,0; 4:00: 31,9; 5:00: 25,3; 6:00: 34,7; 7:00: 35,2; 8:00: 41,6; 9:00: 45,7; 10:00: 49,2; 11:00: 55,8; 12:00: 69,6; 13:00: 78,3; 14:00: 91,5; 15:00: 97,7; 16:00: 81,2; 17:00: 71,8; 18:00: 43,5; 19:00: 34,3; 20:00: 21,5; 21:00: 20,5; 22:00: 19,4; 23:00: 20,4; 0:00: 21,3.

Tính giá trị trung bình 8h được tính toán trong bảng như sau (mỗi ngày có 24 giá trị trung bình 8 giờ):

O3 (TB8h), µg/m3 theo giờ: 1:00: 18,0; 2:00: 19,4; 3:00: 21,2; 4:00: 23,0; 5:00: 23,8; 6:00: 25,8; 7:00: 28,2; 8:00: 30,9; 9:00: 33,8; 10:00: 36,6; 11:00: 39,9; 12:00: 44,7; 13:00: 51,3; 14:00: 58,4; 15:00: 66,2; 16:00: 71,1; 17:00: 74,4; 18:00: 73,7; 19:00: 71,0; 20:00: 65,0; 21:00: 57,8; 22:00: 48,7; 23:00: 39,1; 0:00: 31,6.

Giá trị trung bình 8 giờ lớn nhất trong ngày là: 74,4

- Giá trị trung bình 24 giờ: trung bình cộng các giá trị quan trắc trung bình 1 giờ trong 1 ngày (từ 01:00 giờ đến 00:00 hôm sau).

**b. Tính giá trị AQI^d của từng thông số (AQIx)**

Giá trị AQI ngày của các thông số SO2, CO, NO2, O3, PM10, PM2.5 được tính toán theo công thức 1 như sau:

Công thức 1: AQIx = [(I(i+1) − Ii) / (BP(i+1) − BPi)] × (Cx − BPi) + Ii

Trong đó:

- Bảng giá trị BPi và Ii lấy trong Bảng 2.
- AQIx: Giá trị AQI^d thông số của thông số x
- BPi: Nồng độ giới hạn dưới của giá trị thông số quan trắc được quy định trong Bảng 2 tương ứng với mức i
- BP(i+1): Nồng độ giới hạn trên của giá trị thông số quan trắc được quy định trong Bảng 2 tương ứng với mức i+1
- Ii: Giá trị AQI ở mức i đã cho trong bảng tương ứng với giá trị BPi
- I(i+1): Giá trị AQI ở mức i+1 cho trong bảng tương ứng với giá trị BP(i+1)
- Cx: được quy định cụ thể như sau:
  - Đối với thông số PM2.5 và PM10: Cx là giá trị trung bình 24 giờ.
  - Đối với thông số O3: Cx là giá trị lớn nhất trong giá trị trung bình 1 giờ lớn nhất trong ngày và giá trị trung bình 8 giờ lớn nhất trong ngày. Lưu ý: Không tính toán AQI thông số O3 khi giá trị trung bình 8 giờ lớn nhất trong ngày cao hơn 400 µg/m3 (lúc này chỉ tính toán AQI đối với trung bình 1 giờ lớn nhất trong ngày).
  - Đối với thông số SO2, NO2 và CO: Cx là giá trị trung bình 1 giờ lớn nhất trong ngày.

**c. Giá trị AQI ngày tổng hợp**

Sau khi đã có giá trị AQIx ngày của mỗi thông số, chọn giá trị AQI lớn nhất của các thông số để lấy làm giá trị AQI ngày tổng hợp.

AQI^d = max(AQIx)

Ghi chú: Giá trị AQI ngày được làm tròn thành số nguyên.

### 2.3. Tính toán mẫu

**a. Tính giá trị Nowcast**

Giả sử có bảng số liệu quan trắc PM2.5 như sau:

| 09:00 | 10:00 | 11:00 | 12:00 | 13:00 | 14:00 | 15:00 | 16:00 | 17:00 | 18:00 | 19:00 | 20:00 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 26,9 | 24,7 | 20,5 | 23,5 | 19,5 | 16,5 | 19,0 | 16,5 | 20,3 | 22,4 | 19,6 | 20,6 |

Tính giá trị w* = Cmin / Cmax = 16,5 / 26,9 = 0,61

Do w* = 0,61 > 0,5 vì vậy lấy w = w* = 0,61

Nowcast = (0,61^0 × 20,6 + 0,61^1 × 19,6 + 0,61^2 × 22,4 + … + 0,61^11 × 26,9) / (0,61^0 + 0,61^1 + 0,61^2 + … + 0,61^11) = 20,3 (µg/m3)

**b. Tính giá trị AQI giờ**

Giả sử có bảng số liệu quan trắc trung bình 1 giờ như sau:

| O3 (µg/m3) | NO2 (µg/m3) | Nowcast (PM2.5) (µg/m3) |
|---|---|---|
| 136,1 | 118,7 | 20,3 |

Tính toán các giá trị AQI thông số như sau:

- AQI(O3) = (50 − 0) / (160 − 0) × (136,1 − 0) + 0 = 43
- AQI(NO2) = (100 − 50) / (200 − 100) × (118,7 − 100) + 50 = 60
- AQI(PM2.5) = (50 − 0) / (25 − 0) × (20,3 − 0) + 0 = 41

Như vậy giá trị AQI^h = 60.

**c. Tính giá trị AQI ngày**

Giả sử có bảng số liệu quan trắc như sau:

| O3 (µg/m3) – Trung bình 8 giờ lớn nhất trong ngày | O3 (µg/m3) – Trung bình 1 giờ lớn nhất trong ngày | NO2 (µg/m3) – Trung bình 1 giờ lớn nhất trong ngày | PM2.5 (µg/m3) – Trung bình 24 giờ |
|---|---|---|---|
| 89,3 | 114,6 | 130,8 | 55,7 |

Tính toán các giá trị AQI thông số như sau:

- AQI(O3 8h) = (50 − 0) / (100 − 0) × (89,3 − 0) + 0 = 45
- AQI(O3 1h) = (50 − 0) / (160 − 0) × (114,6 − 0) + 0 = 36
- AQI(NO2) = (100 − 50) / (200 − 100) × (130,8 − 100) + 50 = 65
- AQI(PM2.5) = (150 − 100) / (80 − 50) × (55,7 − 50) + 100 = 110

Như vậy giá trị AQI^d = 110.

## 3. Hướng dẫn công bố chỉ số chất lượng không khí

Chỉ thực hiện công bố thông tin về VN_AQI khi nguồn số liệu và kết quả tính toán đáp ứng các yêu cầu quy định tại Mục 2 của Hướng dẫn này.

Thông tin về VN_AQI được công bố cho cộng đồng thông qua các phương tiện thông tin đại chúng như báo chí, truyền thanh, truyền hình, bảng thông tin điện tử, trang thông tin điện tử, ứng dụng trên thiết bị di động.

Bảng 3: Yêu cầu đối với nội dung thông tin công bố về VN_AQI

| Nội dung công bố | Báo chí | Phát thanh | Truyền hình | Bảng điện tử | Trang tin điện tử | Thiết bị di động |
|---|---|---|---|---|---|---|
| Màu sắc tương ứng mức cảnh báo | X | - | X | X | X | X |
| Thông tin về trạm quan trắc | Tên trạm rút gọn | Tên trạm rút gọn | Tên trạm rút gọn | Tên trạm rút gọn | Địa điểm, tọa độ | Địa điểm, tọa độ |
| Thông số quan trắc sử dụng để tính toán AQI | Thông số có (AQIx) cao nhất | Thông số có (AQIx) cao nhất | Thông số có (AQIx) cao nhất | Thông số có (AQIx) cao nhất | Tất cả các thông số. Chỉ rõ thông số có (AQIx) cao nhất | Tất cả các thông số. Chỉ rõ thông số có (AQIx) cao nhất |
| Khuyến nghị, cảnh báo mức tác động tới sức khỏe | X | X | X | X | X | X |
| Phương pháp tính AQI sử dụng (VN_AQI) | Rút gọn theo tên cơ quan ban hành công thức AQI | Rút gọn theo tên cơ quan ban hành công thức AQI | Rút gọn theo tên cơ quan ban hành công thức AQI | Rút gọn theo tên cơ quan ban hành công thức AQI | Trình bày chi tiết cách tính | Trình bày chi tiết cách tính |
| Nguồn cung cấp dữ liệu để tính toán AQI | X | X | X | X | X | X |

Ghi chú: (X): thông tin bắt buộc; (-): không bắt buộc

Bảng 4: Các mức VN_AQI tương ứng ảnh hưởng tới sức khỏe

| Khoảng giá trị AQI | Ảnh hưởng tới sức khỏe con người |
|---|---|
| 0 – 50 (Tốt) | Chất lượng không khí tốt, không ảnh hưởng tới sức khỏe |
| 51 – 100 (Trung bình) | Chất lượng không khí ở mức chấp nhận được. Tuy nhiên, đối với những người nhạy cảm (người già, trẻ em, người mắc các bệnh hô hấp, tim mạch…) có thể chịu những tác động nhất định tới sức khỏe. |
| 101 – 150 (Kém) | Những người nhạy cảm gặp phải các vấn đề về sức khỏe, những người bình thường ít ảnh hưởng. |
| 151 – 200 (Xấu) | Những người bình thường bắt đầu có các ảnh hưởng tới sức khỏe, nhóm người nhạy cảm có thể gặp những vấn đề sức khỏe nghiêm trọng hơn. |
| 201 – 300 (Rất xấu) | Cảnh báo hưởng tới sức khỏe: mọi người bị ảnh hưởng tới sức khỏe nghiêm trọng hơn. |
| 301 – 500 (Nguy hại) | Cảnh báo khẩn cấp về sức khỏe: Toàn bộ dân số bị ảnh hưởng tới sức khỏe tới mức nghiêm trọng. |

Bảng 5: Các mức VN_AQI và một số hoạt động khuyến nghị

| Khoảng giá trị AQI | Khuyến nghị hoạt động cho những người bình thường | Khuyến nghị hoạt động cho nhóm người nhạy cảm |
|---|---|---|
| 0 – 50 (Tốt) | Tự do thực hiện các hoạt động ngoài trời | Tự do thực hiện các hoạt động ngoài trời |
| 51 – 100 (Trung bình) | Tự do thực hiện các hoạt động ngoài trời | Nên theo dõi các triệu chứng như ho hoặc khó thở, nhưng vẫn có thể hoạt động bên ngoài. |
| 101 – 150 (Kém) | Những người thấy có triệu chứng đau mắt, ho hoặc đau họng… nên cân nhắc giảm các hoạt động ngoài trời. Đối với học sinh, có thể hoạt động bên ngoài, nhưng nên giảm bớt việc tập thể dục kéo dài. | Nên giảm các hoạt động mạnh và giảm thời gian hoạt động ngoài trời. Những người mắc bệnh hen suyễn có thể cần sử dụng thuốc thường xuyên hơn. |
| 151 – 200 (Xấu) | Mọi người nên giảm các hoạt động mạnh khi ở ngoài trời, tránh tập thể dục kéo dài và nghỉ ngơi nhiều hơn trong nhà. | Nên ở trong nhà và giảm hoạt động mạnh. Nếu cần thiết phải ra ngoài, hãy đeo khẩu trang đạt tiêu chuẩn. |
| 201 – 300 (Rất xấu) | Mọi người hạn chế tối đa các hoạt động ngoài trời và chuyển tất cả các hoạt động vào trong nhà. Nếu cần thiết phải ra ngoài, hãy đeo khẩu trang đạt tiêu chuẩn. | Nên ở trong nhà và giảm hoạt động mạnh. |
| 301 – 500 (Nguy hại) | Mọi người nên ở trong nhà, đóng cửa ra vào và cửa sổ. Nếu cần thiết phải ra ngoài, hãy đeo khẩu trang đạt tiêu chuẩn. [Ghi chú biên soạn: ô gộp, áp dụng chung cho cả hai nhóm] | Mọi người nên ở trong nhà, đóng cửa ra vào và cửa sổ. Nếu cần thiết phải ra ngoài, hãy đeo khẩu trang đạt tiêu chuẩn. [Ghi chú biên soạn: ô gộp, áp dụng chung cho cả hai nhóm] |
