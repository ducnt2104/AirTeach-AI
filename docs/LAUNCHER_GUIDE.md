# AIRTEACH AI - ONE-CLICK WINDOWS LAUNCHER GUIDE
## Hướng Dẫn Vận Hành & Khởi Chạy 1-Click: `Run_AirTeach_AI.bat`

Tài liệu hướng dẫn khởi chạy phần mềm AirTeach AI trên hệ điều hành Microsoft Windows bằng tệp kịch bản tự động `Run_AirTeach_AI.bat`.

---

### 1. KHỞI CHẠY NHANH (QUICK START)

1. Mở thư mục gốc của dự án `AirTeach AI`.
2. Nhấp đúp chuột (Double-click) vào tệp **`Run_AirTeach_AI.bat`**.
3. Cửa sổ dòng lệnh sẽ tự động:
   - Chuyển mã trang ký tự sang UTF-8 (Code Page 65001).
   - Kiểm tra và tự động ưu tiên môi trường ảo `.venv\Scripts\python.exe`.
   - Nếu không có `.venv`, tự động sử dụng Python hệ thống từ biến môi trường `PATH`.
   - Kiểm tra phiên bản Python (yêu cầu $\ge 3.10$).
   - Kiểm tra các thư viện cốt lõi: `PySide6`, `opencv-python`, `mediapipe`, `numpy`, `cryptography`, `python-dotenv`.
   - Nếu phát hiện thiếu thư viện, tự động chạy `pip install -r requirements.txt`.
   - Khởi chạy màn hình Đăng nhập Giáo viên (`LoginWindow`) hoặc chuyển thẳng vào bảng vẽ.

---

### 2. CÁC THAM SỐ DÒNG LỆNH HỖ TRỢ (CLI OPTIONS)

Người dùng hoặc kỹ sư có thể chạy launcher kèm các tham số dòng lệnh:

| Lệnh chạy | Mục đích |
| :--- | :--- |
| `Run_AirTeach_AI.bat` | Khởi chạy giao diện người dùng đầy đủ của ứng dụng. |
| `Run_AirTeach_AI.bat --version` | In phiên bản phần mềm hiện tại (`AirTeach AI v2.5.0-Release`) và thoát. |
| `Run_AirTeach_AI.bat --check` | Kiểm tra tính toàn vẹn của mô hình ONNX, MediaPipe, cơ sở dữ liệu SQLite và thoát (Exit code 0). |
| `Run_AirTeach_AI.bat --dry-run` | Kiểm tra pre-flight toàn bộ hệ thống mà không mở cửa sổ giao diện. |

---

### 3. NHẬT KÝ KHỞI CHẠY (LAUNCH LOGGING)

Mọi lần khởi chạy đều được ghi nhận tự động vào tệp nhật ký:
`logs\launcher.log`

Nội dung nhật ký mẫu:
```text
[Fri 10/23/2026 14:31:13.31] AirTeach AI Launcher starting... 
[INFO] Virtual environment detected: .venv 
[INFO] Launching main.py at 14:31:15.02 
[CHECK] AirTeach AI system components and models verification...
[CHECK] Open-source AI models verified.
[CHECK] SQLite profile store and facial recognition pipeline verified.
[CHECK] All systems operational. Ready to launch.
[Fri 10/23/2026 14:31:16.32] Application exited with code 0 
```

---

### 4. XỬ LÝ SỰ CỐ PHỔ BIẾN (TROUBLESHOOTING)

1. **Lỗi `[ERROR] Khong tim thay Python 3.10+ tren may tinh!`:**
   - **Khắc phục:** Cài đặt Python 3.10 hoặc mới hơn từ trang chủ [python.org](https://www.python.org/downloads/). Nhớ tích chọn **"Add Python to PATH"** trong quá trình cài đặt.

2. **Lỗi camera không mở được hoặc màn hình camera màu đen:**
   - **Khắc phục:** Kiểm tra cáp webcam hoặc đảm bảo không có ứng dụng nào khác (Zoom, Teams, Camera app) đang chiếm quyền điều khiển webcam. Vào Cài đặt Windows $\to$ Privacy & Security $\to$ Camera $\to$ Cho phép ứng dụng truy cập camera.

3. **Lỗi thiếu mô hình AI:**
   - **Khắc phục:** Khi có kết nối mạng, hệ thống tự động tải các mô hình vào thư mục `models/`. Nếu máy tính hoàn toàn offline, hãy sao chép thư mục `models/` từ máy đã cài đặt sang.
