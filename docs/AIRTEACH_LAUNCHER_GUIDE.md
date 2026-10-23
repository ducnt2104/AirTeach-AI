# AIRTEACH AI - ONE-CLICK WINDOWS LAUNCHER GUIDE
## Hướng Dẫn Vận Hành & Khởi Chạy 1-Click: `Run_AirTeach_AI.bat`

**Dự án:** AirTeach AI — Nền tảng dạy học thông minh tương tác không chạm bằng AI  
**Tệp kịch bản:** `Run_AirTeach_AI.bat` (Thư mục gốc)  
**Nền tảng hỗ trợ:** Microsoft Windows 10, Windows 11 (64-bit Architecture AMD64)  
**Yêu cầu môi trường tối thiểu:** Python 3.10+, RAM 4GB, Webcam USB 720p/1080p

---

### 1. QUY TRÌNH KHỞI CHẠY NHANH 1-CLICK (QUICK START)

Để bắt đầu giảng dạy bằng AirTeach AI trên bất kỳ máy tính Windows nào:

1. Điều hướng tới thư mục gốc của dự án `AirTeach AI`.
2. Nhấp đúp chuột (Double-click) vào tệp **`Run_AirTeach_AI.bat`**.
3. Kịch bản batch thông minh sẽ tự động thực hiện tuần tự:
   - Chuyển đổi bảng mã bàn phím và console sang chuẩn **UTF-8 (Code Page 65001)** để hiển thị tiếng Việt có dấu hoàn hảo.
   - Quét tìm và tự động ưu tiên môi trường ảo cục bộ `.venv\Scripts\python.exe`.
   - Nếu không tìm thấy `.venv`, tự động quét và lựa chọn trình thông dịch Python hệ thống từ biến môi trường `PATH`.
   - Kiểm tra phiên bản Python (yêu cầu nghiêm ngặt phiên bản $\ge 3.10$).
   - Kiểm tra tính đầy đủ của các gói thư viện cốt lõi: `PySide6`, `opencv-python`, `mediapipe`, `numpy`, `cryptography`, `python-dotenv`.
   - Nếu phát hiện thiếu bất kỳ gói phụ thuộc nào, launcher sẽ tự động chạy lệnh `pip install -r requirements.txt` để hoàn thiện môi trường.
   - Kích hoạt ứng dụng và hiển thị màn hình Đăng nhập Khuôn mặt Giáo viên (`LoginWindow`) hoặc chuyển tiếp vào Bảng vẽ thông minh (`MainWindow`).

---

### 2. CÁC THAM SỐ DÒNG LỆNH HỖ TRỢ (COMMAND-LINE INTERFACE)

Kịch bản `Run_AirTeach_AI.bat` hỗ trợ đầy đủ các cờ tham số dòng lệnh phục vụ việc kiểm tra tự động và tích hợp hệ thống:

| Tham số dòng lệnh | Mục đích sử dụng | Hành vi thực thi |
| :--- | :--- | :--- |
| `Run_AirTeach_AI.bat` | Khởi chạy tiêu chuẩn | Khởi động giao diện người dùng đầy đủ (GUI) của phần mềm. |
| `Run_AirTeach_AI.bat --version` | Kiểm tra phiên bản | In chuỗi định danh phiên bản (`AirTeach AI v2.5.0-Release (Windows Desktop Academic Edition)`) và thoát với mã `0`. |
| `Run_AirTeach_AI.bat --check` | Tự kiểm tra hệ thống (Pre-flight) | Quét và xác thực tính toàn vẹn của các mô hình ONNX trong `models/`, cơ sở dữ liệu SQLite `data/profiles.db`, pipeline nhận diện khuôn mặt và camera rồi thoát với mã `0`. |
| `Run_AirTeach_AI.bat --dry-run` | Chạy thử nghiệm ngầm | Kiểm tra nạp toàn bộ cấu hình, dịch vụ và bộ nhớ mà không mở cửa sổ giao diện chính. |

---

### 3. ĐẶC TẢ KỸ THUẬT KỊCH BẢN BATCH (TECHNICAL SPECIFICATIONS)

Kịch bản `Run_AirTeach_AI.bat` được thiết kế theo các tiêu chuẩn kỹ thuật nghiêm ngặt dành cho môi trường Windows:

1. **Định dạng ký tự ngắt dòng (CRLF):**
   - Tuyệt đối sử dụng chuẩn ngắt dòng Windows CRLF (`\r\n`). Kịch bản ngăn ngừa hoàn toàn các lỗi cú pháp bí ẩn của `cmd.exe` khi gặp tệp batch có ngắt dòng Unix LF.
2. **Khóa chống chiếm dụng Camera (Single-Instance Inter-Process Mutex):**
   - Khi người dùng nhấp đúp vào launcher lần thứ hai trong lúc ứng dụng đang chạy:
     - Ứng dụng thứ hai phát hiện IPC server `airteach_ai_single_instance_ipc`.
     - Gửi tín hiệu đánh thức (`WAKEUP`) đến ứng dụng đầu tiên để kéo cửa sổ lên trước màn hình.
     - Ứng dụng thứ hai tự động dừng lại và trả về mã thoát `0`. Tránh hoàn toàn việc hai ứng dụng cùng tranh chấp một thiết bị webcam làm đen hình hoặc treo máy.
3. **Quản lý mã thoát (Exit Code Propagation):**
   - Launcher ghi nhận mã thoát chính xác từ ứng dụng Python (`ERRORLEVEL`) và trả mã đó về cho hệ điều hành (`exit /b %APP_EXIT_CODE%`).
4. **Nhật ký khởi chạy chi tiết (Logging):**
   - Mọi tiến trình khởi chạy, phát hiện môi trường, thời gian bắt đầu và kết thúc đều được nối tự động vào tệp nhật ký `logs\launcher.log`.

---

### 4. NỘI DUNG NHẬT KÝ MẪU (`logs\launcher.log`)

```text
[Fri 10/23/2026 15:48:00.12] AirTeach AI Launcher starting... 
[INFO] Virtual environment detected: .venv 
[INFO] Launching main.py at 15:48:01.05 
[CHECK] AirTeach AI system components and models verification...
[CHECK] Open-source AI models verified.
[CHECK] SQLite profile store and facial recognition pipeline verified.
[CHECK] All systems operational. Ready to launch.
[Fri 10/23/2026 15:48:01.88] Application exited with code 0 
```

---

### 5. HƯỚNG DẪN XỬ LÝ SỰ CỐ THƯỜNG GẶP (TROUBLESHOOTING)

1. **Lỗi: `[ERROR] Khong tim thay Python 3.10+ tren may tinh!`**
   - **Nguyên nhân:** Máy tính chưa cài đặt Python hoặc chưa tích chọn đưa Python vào biến môi trường hệ thống.
   - **Cách xử lý:** Truy cập [python.org](https://www.python.org/downloads/) để tải và cài đặt Python 3.10, 3.11 hoặc 3.12. Trong màn hình cài đặt đầu tiên, **bắt buộc đánh dấu tích vào ô "Add Python to PATH"**.

2. **Khung camera góc dưới màu đen hoặc không nhận diện tay:**
   - **Nguyên nhân:** Webcam đang bị phần mềm khác (như Zoom, Microsoft Teams, hoặc ứng dụng Camera mặc định của Windows) chiếm quyền độc quyền.
   - **Cách xử lý:** Đóng các phần mềm họp trực tuyến khác. Mở **Windows Settings $\to$ Privacy & Security $\to$ Camera** và đảm bảo tùy chọn **"Let desktop apps access your camera"** đang ở trạng thái **ON**.

3. **Thông báo lỗi thiếu tệp mô hình trong thư mục `models/`:**
   - **Nguyên nhân:** Lần đầu chạy trên máy tính không có kết nối Internet để tải mô hình tự động.
   - **Cách xử lý:** Kết nối mạng Internet trong lần khởi chạy đầu tiên để ứng dụng tự tải 3 mô hình (`face_detection_yunet_2023mar.onnx`, `face_recognition_sface_2021dec.onnx`, `hand_landmarker.task`), hoặc sao chép thư mục `models/` từ máy đã cài đặt sẵn.
