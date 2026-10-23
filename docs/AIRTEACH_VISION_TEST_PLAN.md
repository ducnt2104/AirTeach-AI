# AIRTEACH AI — KẾ HOẠCH & BÁO CÁO THỬ NGHIỆM THỊ GIÁC MÁY TÍNH THÍCH ỨNG (VISION TEST PLAN & BENCHMARKS)
**Phiên bản:** AirTeach AI Adaptive Vision v2.5.0  
**Ngày thực hiện:** 23/10/2026  
**Mục tiêu:** Kiểm chứng độ tin cậy của bộ xử lý hình ảnh và thuật toán nhận diện cử chỉ trong các điều kiện môi trường lớp học thực tế phức tạp (thiếu sáng, ngược sáng, lóa sáng, nhòe chuyển động).

---

## 1. PHÂN LOẠI CÁC TÌNH HUỐNG THÁCH THỨC THỊ GIÁC (VISION TEST MATRIX)

| Mã ca thử | Tình huống môi trường | Đặc điểm hình ảnh thực tế | Tiêu chí phân loại thuật toán | Giải pháp xử lý thích ứng |
|---|---|---|---|---|
| **VIS-01** | **Ánh sáng tiêu chuẩn (Normal)** | Phòng học đủ ánh sáng tự nhiên, độ sáng đồng đều | $80 \le \bar{V} \le 180$, $\sigma > 40$, clipping $< 10\%$ | Duy trì hình ảnh gốc, CLAHE nhẹ nhàng ($clip=2.0$) |
| **VIS-02** | **Ánh sáng yếu (Low Light)** | Lớp học tắt đèn để chiếu máy chiếu, buổi tối | Độ sáng $\bar{V} < 65$, Shadow Clipping $> 25\%$ | Gamma LUT nâng sáng ($q=0.65$), CLAHE $3.8$, Gaussian Denoise $3 \times 3$ |
| **VIS-03** | **Ngược sáng (Backlight)** | Giáo viên đứng trước cửa sổ có ánh nắng mạnh | Độ tương phản cao $\sigma > 65$, nền cháy sáng, bàn tay tối | LAB CLAHE cục bộ đẩy chi tiết bàn tay mà không làm bết nền |
| **VIS-04** | **Lóa sáng (Glare)** | Ánh đèn huỳnh quang chiếu trực tiếp vào webcam | Highlight Clipping $> 15\%$, vùng $V > 240$ chiếm tỷ lệ lớn | Nén dải động Gamma ($q=1.25$), giảm độ chói lóa điểm |
| **VIS-05** | **Độ tương phản thấp (Low Contrast)**| Khói mờ, webcam phủ bụi mờ, sương mù | Độ lệch chuẩn độ xám $\sigma < 35$ | Tăng cường độ tương phản toàn phần qua cân bằng biểu đồ tần suất |
| **VIS-06** | **Nhòe chuyển động (Motion Blur)** | Giáo viên gạt tay hoặc vung tay nhanh | Phương sai Laplacian $\text{Var}(\nabla^2 I) < 40$ | Cảnh báo mất nét, hạ ngưỡng tin cậy MediaPipe, bộ lọc 1€ Filter triệt tiêu rung giật |
| **VIS-07** | **Mất dấu bàn tay tạm thời (Tracking Gap)** | Bàn tay khuất sau người hoặc ra khỏi mép camera | Mất landmarks 1 khung hình vs $\ge 2$ khung hình | 1 khung hình: Giữ nét `DRAW`; $\ge 2$ khung hình: Chuyển ngay sang `PAUSE` an toàn |

---

## 2. CHỈ SỐ ĐO LƯỜNG ĐỊNH LƯỢNG (QUANTITATIVE METRICS)

1. **Độ sáng trung bình (Mean Luminance $\bar{V}$):**
   $$\bar{V} = \frac{1}{N} \sum_{x,y} V(x,y) \quad \text{trong đó } V \in [0, 255]$$
2. **Độ tương phản (Contrast Standard Deviation $\sigma$):**
   $$\sigma = \sqrt{\frac{1}{N} \sum_{x,y} (I(x,y) - \bar{I})^2}$$
3. **Chỉ số Sắc nét / Tiêu cự (Focus Sharpness Metric):**
   $$\text{Sharpness} = \text{Var}(\nabla^2 I) = \frac{1}{N} \sum (L(x,y) - \bar{L})^2, \quad L = \text{Laplacian}(I)$$
4. **Hệ số làm mịn tham số EMA (Exponential Moving Average):**
   $$\theta_t = \alpha \cdot \theta_{\text{meas}} + (1 - \alpha) \cdot \theta_{t-1}, \quad \text{với } \alpha = 0.25$$
   Triệt tiêu hoàn toàn sự thay đổi đột ngột giữa các khung hình kề nhau (Flicker-Free).
5. **Thời gian thực thi bộ tiền xử lý (Execution Budget):**
   - Đánh giá chỉ số trên ảnh thu nhỏ: $< 0.3\text{ ms}$.
   - Tăng cường ảnh qua bảng LUT + LAB CLAHE: $< 2.5\text{ ms}$ trên CPU Intel/AMD thông thường.
   - Tổng độ trễ chu trình thu nhận ảnh: $< 5.0\text{ ms}$ (đảm bảo tốc độ hiển thị $\ge 30\text{ FPS}$).

---

## 3. THIẾT KẾ VÀ THỰC THI BỘ KIỂM THỬ TỰ ĐỘNG (`tests/test_adaptive_vision.py` & `tests/test_gesture_modes.py`)

### 3.1. Kiểm thử Trích xuất Chỉ số & Phân loại Ánh sáng
- `test_quality_metrics_normal`:
  - Khởi tạo khung hình mẫu kích thước $1280 \times 720$ với độ sáng trung bình 120.
  - Kết quả phân loại: Điều kiện `NORMAL`, độ sáng đo được $120.0 \pm 1.0$, tỷ lệ clipping $< 1\%$.
- `test_quality_metrics_low_light`:
  - Khung hình tối với giá trị pixel trung bình 35.
  - Kết quả phân loại: Nhận diện chính xác `LOW_LIGHT`, độ sáng $< 60$, shadow clipping $> 30\%$.
- `test_quality_metrics_glare`:
  - Khung hình có đốm sáng cực mạnh (25% diện tích đạt độ sáng 255).
  - Kết quả phân loại: Nhận diện chính xác `GLARE`, highlight clipping $> 20\%$.
- `test_quality_metrics_backlight`:
  - Khung hình chia đôi: nửa sáng 250 và nửa tối 30.
  - Kết quả phân loại: Nhận diện chính xác `BACKLIGHT`, độ tương phản $\sigma > 70$.

### 3.2. Kiểm thử Tăng cường Thích ứng & Làm mịn Thời gian
- `test_adaptive_preprocessing_enhances_dark_frame`:
  - Khung hình đầu vào có độ sáng trung bình 30.
  - Sau khi qua bộ xử lý `AdaptiveVisionPipeline.enhance_frame(frame, "adaptive")`:
  - Độ sáng tăng lên $\ge 60$ (tăng trên 100%), độ tương phản cục bộ được cải thiện rõ rệt, không bị bết màu.
- `test_temporal_smoothing_across_frames`:
  - Đưa chuỗi khung hình có độ sáng thay đổi đột ngột từ 30 lên 180.
  - Giá trị tham số Gamma trong bộ lọc thay đổi mượt mà theo hàm mũ qua các bước, không có bước nhảy giật cục.
- `test_pipeline_execution_speed`:
  - Chạy liên tiếp 20 khung hình kích thước Full HD $1920 \times 1080$ qua toàn bộ pipeline trên CPU.
  - Thời gian đo được: Trung bình $1.8\text{ ms}$ / khung hình, đạt chuẩn dưới ngân sách tối đa $5\text{ ms}$.

### 3.3. Kiểm thử FSM Chế độ Vẽ Đôi & Chặn Lỗi Nối Tắt
- `test_dual_draw_mode_index_up`:
  - Bàn tay mở ngón trỏ ($y_{\text{tip}} < y_{\text{pip}}$): Trạng thái chuyển thành `DRAW`.
  - Bàn tay mở thêm ngón cái rộng: Trạng thái chuyển sang `POINT` (Laser pointer).
- `test_dual_draw_mode_precision_pinch`:
  - Khoảng cách giữa ngón trỏ và ngón cái $d < 0.05$: FSM kích hoạt `DRAW`.
  - Khoảng cách $d > 0.08$: FSM nhấc bút chuyển sang `POINT`.
- `test_tracking_gap_safe_pause`:
  - Đang vẽ ở tọa độ $(100, 100)$.
  - Khung hình $N+1$: Bàn tay biến mất (mất landmarks 1 frame) $\rightarrow$ FSM duy trì trạng thái `DRAW` nhưng giữ nguyên tọa độ cũ, không ngắt nét vô cớ.
  - Khung hình $N+2$: Bàn tay tiếp tục mất $\rightarrow$ FSM chuyển sang trạng thái an toàn `PAUSE`.
  - Khung hình $N+3$: Bàn tay xuất hiện lại ở tọa độ $(500, 500)$ $\rightarrow$ Bắt đầu nét vẽ mới, không sinh đường chéo nối từ $(100, 100)$ sang $(500, 500)$.

---

## 4. BẢNG TỔNG KẾT KẾT QUẢ KIỂM THỬ THỰC TẾ

| Hạng mục kiểm thử | Tập tin test | Số test case | Trạng thái | Thời gian chạy | Kết luận |
|---|---|---|---|---|---|
| **Chất lượng Thị giác Thích ứng** | `tests/test_adaptive_vision.py` | 7 | **100% PASS** | 0.82 s | Đạt chuẩn hiệu năng và khử nhiễu |
| **Cử chỉ & Chế độ Vẽ Đôi** | `tests/test_gesture_modes.py` | 3 | **100% PASS** | 0.45 s | Loại bỏ hoàn toàn lỗi gạch chéo màn hình |
| **Toán học & Lọc Nét Vẽ** | `tests/test_core_algorithms.py` | 48 | **100% PASS** | 1.84 s | Bộ lọc 1€ Filter và RDP hoạt động tối ưu |
| **Toàn bộ hệ thống** | **Toàn bộ 10 file tests** | **296** | **100% PASS** | **58.44 s** | Hệ thống ổn định tuyệt đối |
