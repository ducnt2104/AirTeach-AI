# KẾ HOẠCH KIỂM THỬ TỔNG THỂ (TEST PLAN)

**Dự án:** AirTeach AI  
**Cập nhật:** 20/10/2026  
**Chính sách kiểm thử:** 100% Thực tế, Zero Mock Data, Tự động hóa kết hợp Thẩm định Thị giác (Visual QA).

---

## 1. CÁC TẦNG KIỂM THỬ (TEST LEVELS)

### 1.1. Kiểm thử đơn vị & Thuật toán cốt lõi (`tests/test_core_algorithms.py`)
- Kiểm tra tính toán học của bộ lọc 1€ Filter (giảm rung jitter, thích ứng vận tốc cao, làm mới trạng thái).
- Kiểm tra thuật toán Ramer-Douglas-Peucker (giảm điểm, bảo tồn 2 đầu mút, xử lý epsilon).
- Kiểm tra nội suy đường cong Catmull-Rom Bézier spline (liên tục $C^1$, biến thiên độ dày nét).
- Kiểm tra máy trạng thái 6 cử chỉ (FSM) và bộ đệm khử rung 3 khung hình.
- Kiểm tra nắn hình học tự động (Đường thẳng, Mũi tên, Hình tròn, Elip, Chữ nhật, Tam giác).

### 1.2. Kiểm thử dịch vụ & Lưu trữ dữ liệu (`tests/test_services_and_storage.py`)
- Kiểm tra mã hóa Fernet và bảo vệ khóa `.keystore`.
- Kiểm tra cơ sở dữ liệu SQLite hồ sơ giáo viên và mã băm PIN PBKDF2.
- Kiểm tra dịch vụ nhận diện khuôn mặt OpenCV YuNet + SFace.
- Kiểm tra nhập học liệu PDF và kết xuất trang bằng `pypdfium2`.
- Kiểm tra chu trình nén/đọc gói bài giảng `.airteach` (manifest, pages, strokes.json).
- Kiểm tra bộ phân tích toán học hình học `GeometricMathParser`.
- Kiểm tra trợ lý Gemini và cơ chế fallback ngoại tuyến.

### 1.3. Kiểm thử giao diện & Tương tác người dùng (`tests/test_ui_components.py` & `test_phase3_advanced_ui.py`)
- Kiểm tra bảng viết `LessonBoard` và `LessonBoardCanvas` (phân tách tầng Z, undo/redo, tẩy nét).
- Kiểm tra `CameraWidget`, `ChatPanel`, `SettingsDialog`, `FaceAuthDialog`, `LibraryAndPrepWidget`.
- Kiểm tra thanh điều khiển phát lại bài giảng `ReplayController`.
- Kiểm tra chuyển trang và phân tách nét vẽ độc lập đa trang.
