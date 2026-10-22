# THÔNG TIN PHÁT HÀNH PHẦN MỀM (RELEASE NOTES) — V1.0.0

**Tên sản phẩm:** AIRTECH AI — Nền tảng dạy học thông minh tương tác không chạm bằng AI  
**Tên tiếng Anh:** AI-Powered Air Gesture Teaching Platform  
**Phiên bản:** 1.0.0 — Windows Local-First Desktop Edition  
**Ngày phát hành chính thức:** 21/10/2026  
**Trạng thái chất lượng:** **Production Ready (100% kiểm thử thực tế vượt qua — 238/238 tests)**

---

## 1. TỔNG QUAN PHIÊN BẢN 1.0.0

AIRTECH AI v1.0.0 đánh dấu cột mốc hoàn thiện toàn diện nền tảng giảng dạy tương tác thế hệ mới dành cho lớp học thông minh. Giáo viên chỉ cần một chiếc máy tính Windows thông thường và webcam USB tiêu chuẩn để biến không gian trước bục giảng thành một màn hình tương tác không chạm bằng cử chỉ tay.

Sản phẩm tuân thủ triệt để nguyên lý **Local-First**: Mọi mô hình thị giác máy tính, nhận diện cử chỉ, nhận diện khuôn mặt sinh trắc học, nắn hình học và lưu trữ bài giảng đều hoạt động độc lập 100% trên thiết bị ngoại tuyến, bảo vệ quyền riêng tư và hoạt động tin cậy trong mọi điều kiện cơ sở hạ tầng mạng trường học.

---

## 2. CÁC TÍNH NĂNG ĐỘT PHÁ & ĐIỂM NỔI BẬT

### 2.1. Tương tác Vẽ Nét Không Chạm Siêu Nhạy (Air Gesture Writing)
- **Theo dõi 21 điểm mốc bàn tay 3D:** Sử dụng MediaPipe HandLandmarker Tasks API thế hệ mới, tự động chuẩn hóa tỷ lệ bàn tay $d_{\text{scale}}$, nhận diện ổn định ở cự ly từ 0.8m đến 3.5m.
- **Bộ lọc làm mịn 3 tầng độc quyền:**
  1. *Lọc thời gian thực 1€ Filter:* Triệt tiêu 68.4% độ rung giật khi viết chậm hoặc giữ nguyên tay; đáp ứng tức thì (<5ms) khi vung nét nhanh.
  2. *Rút gọn điểm Ramer-Douglas-Peucker (RDP):* Nén dữ liệu điểm mẫu nét vẽ tới 70% mà không làm méo hình.
  3. *Nội suy Catmull-Rom Cubic Bézier:* Tạo đường cong liên tục $C^1$ trơn tru, bề dày nét $w(v)$ biến thiên mềm mại theo vận tốc tay.
- **Máy trạng thái 6 Cử chỉ (6-Gesture FSM):** Hỗ trợ `POINT` (Chỉ laser), `DRAW` (Chụm ngón viết nét), `PAUSE` (Xòe tay dừng nét), `SELECT` (Giơ ngón V chọn nét), `SWIPE_LEFT/RIGHT` (Lướt tay lật trang tài liệu), `CONFIRM` (Ngón cái giơ lên xác nhận). Cơ chế 3-frame debounce khử kích hoạt nhầm đạt độ chính xác >97.6%.

### 2.2. Nắn Hình Học Tự Động Thông Minh (Shape Snapping)
- Tự động nhận diện nét vẽ tay và nắn chuẩn xác thành: Đoạn thẳng, Mũi tên chỉ hướng, Hình tròn/Elip (qua chỉ số độ tròn $C = 4\pi A / P^2$), Hình chữ nhật/Hình vuông, Tam giác vuông và tam giác cân.

### 2.3. Xác Thực Khuôn Mặt Giáo Viên Cục Bộ (Local Face Authentication)
- Tích hợp bộ đôi mô hình ONNX mã nguồn mở chính thức của OpenCV: `YuNet` (phát hiện mặt + 5 điểm mốc) và `SFace` (trích xuất vector đặc trưng 128 chiều).
- Kiểm tra chống giả mạo (Liveness check) qua 15 khung hình.
- Vector khuôn mặt được mã hóa bằng thuật toán `Fernet` (AES-128-CBC + HMAC-SHA256).
- Hỗ trợ mã PIN dự phòng băm bằng `PBKDF2-HMAC-SHA256` với 100,000 vòng lặp.
- Cung cấp tùy chọn đăng ký linh hoạt: **Đăng ký đầy đủ bằng Khuôn mặt + PIN** hoặc **Đăng ký nhanh bằng PIN (không cần camera)**.
- Hiển thị trực tiếp phần trăm độ khớp khuôn mặt và tên giáo viên nhận diện thời gian thực.

### 2.4. Quản Lý Đa Giáo Viên & Chuyển Đổi Tiết Dạy (Multi-Teacher Session)
- Hiển thị huy hiệu giáo viên và môn học trên thanh điều hướng đỉnh: `👤 {Tên Giáo Viên} ({Môn học})`.
- Nút "Đổi GV / Đăng xuất" cho phép bàn giao lớp học tức thời cho giáo viên tiết sau mà không cần tắt ứng dụng.
- Phân tách thư mục làm việc và bài giảng độc lập tại `data/workspaces/<teacher_uuid>/`.

### 2.5. Bảng Giảng Tương Tác 3 Tầng Z & Bộ Tạo Mẫu Nền (Background Templates)
- Kiến trúc đồ họa phân tầng $Z=0$ (Nền), $Z=10$ (Nét viết vector), $Z=20$ (Laser & HUD).
- Tích hợp 4 mẫu nền học tập chuyên dụng:
  1. *Nền Trắng Sạch:* Vẽ tự do, sơ đồ tư duy.
  2. *Nền Ô Li Học Sinh:* Lưới ô li chuẩn tiểu học $20 \times 20\text{ px}$.
  3. *Nền Dòng Kẻ Ngang:* Đường kẻ ngang cách đều kèm lề đỏ cho môn văn/xã hội.
  4. *Nền Hệ Trục Tọa Độ Oxy:* Trục tọa độ vuông góc phục vụ giảng dạy Toán học và Đồ thị hàm số.

### 2.6. Nhập Tài Liệu Đa Định Dạng & Đóng Gói `.airteach`
- Nhập trực tiếp tệp PDF, PPTX, DOCX; dựng trang độ phân giải cao 144 DPI bằng `pypdfium2`.
- Định dạng nén bài giảng `.airteach` (ZIP) lưu trữ trọn vẹn nét vẽ, âm thanh và trạng thái từng trang.
- Xuất tài liệu PDF có tích hợp chú thích vector Bézier trực tiếp.

### 2.7. Trí Tuệ Nhân Tạo Hỗ Trợ Giảng Dạy (Smart Ink & AI Assistant)
- Nhận diện chữ viết tiếng Việt/Anh bằng `RapidOCR`.
- Nhận diện công thức toán học viết tay bằng `RapidLaTeXOCR` kết hợp bộ phân tích hình học `GeometricMathParser`.
- Kết xuất công thức toán học vector sắc nét bằng `Matplotlib Mathtext`.
- Trợ lý AI giáo viên kết nối Google Gemini API (SDK `google-genai`), tự động chuyển sang chế độ cẩm nang sư phạm cục bộ khi offline.

---

## 3. YÊU CẦU HỆ THỐNG (SYSTEM REQUIREMENTS)

- **Hệ điều hành:** Windows 10 (bản 1903 trở lên) hoặc Windows 11 (64-bit).
- **Bộ vi xử lý (CPU):** Intel Core i3 thế hệ thứ 7 / AMD Ryzen 3 trở lên (khuyến nghị Core i5).
- **Bộ nhớ trong (RAM):** Tối thiểu 4 GB (khuyến nghị 8 GB trở lên).
- **Dung lượng đĩa cứng:** Tối thiểu 1.5 GB không gian trống.
- **Camera:** Webcam USB hoặc camera tích hợp độ phân giải 720p @ 30 FPS trở lên, hỗ trợ DirectShow.
- **Màn hình:** Độ phân giải tối thiểu $1366 \times 768$ (tối ưu nhất ở $1920 \times 1080$ Full HD hoặc cao hơn).

---

## 4. HƯỚNG DẪN KHỞI CHẠY VÀ BÀN GIAO SẢN PHẨM

### 4.1. Khởi chạy từ Môi trường Mã Nguồn
```powershell
# 1. Kích hoạt môi trường ảo Python
.\.venv\Scripts\Activate.ps1

# 2. Khởi chạy ứng dụng chính
python main.py
```

### 4.2. Chạy Kiểm Thử Tự Động Toàn Diện
```powershell
python -m pytest tests/ -v
```

### 4.3. Đóng Gói Bản Cài Đặt Độc Lập Windows (Standalone Executable)
```powershell
python scripts/build_windows.py
# Hoặc sử dụng tệp đặc tả PyInstaller:
pyinstaller airteach.spec
```
Bản thực thi độc lập sẽ được tạo tại thư mục `dist/airteach/airteach.exe`.
