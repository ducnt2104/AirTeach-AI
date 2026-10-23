# AIRTEACH AI - COMPREHENSIVE VERIFICATION & TEST PLAN
## Kế Hoạch Kiểm Thử Toàn Diện Hệ Thống (Formal Test Plan & Verification Protocol)

**Dự án:** AirTeach AI — Hệ thống dạy học thông minh tương tác không chạm bằng AI  
**Tiêu chuẩn quy chuẩn:** IEEE 829 Standard for Software and System Test Documentation  
**Mục tiêu:** Xác minh 100% các tính năng, thuật toán toán học, giao diện đồ họa vector, khả năng nhận diện cử chỉ không chạm, độ bảo mật hồ sơ giáo viên và tính tương thích trên nền tảng Windows.

---

### 1. PHẠM VI VÀ CHIẾN LƯỢC KIỂM THỬ (TEST SCOPE & STRATEGY)

Hệ thống kiểm thử của AirTeach AI áp dụng nguyên tắc **Zero Mock Policy**:
1. Toàn bộ các thuật toán cốt lõi (1€ Filter, RDP, Bézier Spline, Convex Hull Shape Snapper) được kiểm thử với các hàm toán học và tập dữ liệu điểm thực tế.
2. Các dịch vụ trí tuệ nhân tạo (YuNet, SFace, RapidOCR, LaTeX-OCR) được kiểm thử với các tệp mô hình ONNX thật và các ma trận ảnh mẫu OpenCV thật.
3. Giao diện người dùng đồ họa (PySide6) được kiểm thử thông qua `QtTest` (`QTest`, `QSignalSpy`) để mô phỏng tương tác chuột, bàn phím và chu kỳ sự kiện (event loop) thực tế.

---

### 2. PHÂN BỐ MA TRẬN BÀI KIỂM THỬ (TEST MATRIX BREAKDOWN)

| Mã Bộ Kiểm Thử | Tệp Kiểm Thử | Số Lượng Test | Lĩnh Vực Kiểm Thử Trọng Tâm |
| :--- | :--- | :---: | :--- |
| **SUITE-01** | `tests/test_core_algorithms.py` | **48** | Thuật toán 1€ Filter (1D & 2D), Ramer–Douglas–Peucker (RDP), nội suy Spline Cubic Bézier $C^1$, nhận diện hình học GeometricShapeSnapper (Đoạn thẳng, Đường tròn, Elip, Chữ nhật, Tam giác), Cỗ máy trạng thái cử chỉ FSM, Ánh xạ phối cảnh PerspectiveMapper. |
| **SUITE-02** | `tests/test_ui_components.py` | **111** | Bảng giảng dạy `LessonBoard` đa tầng, hệ thống quản lý nhiều trang bài giảng (Multi-page navigation), thanh công cụ giảng dạy tương tác (Dual Toolbars), bộ chọn màu mực, điều chỉnh nét bút, hộp thoại Cài đặt (`SettingsDialog`), luồng thu hình camera widget. |
| **SUITE-03** | `tests/test_phase3_advanced_ui.py` | **33** | Cửa sổ Đăng nhập Giáo viên (`LoginWindow`), hiển thị hồ sơ cá nhân (`TeacherProfile`), bàn phím số mã PIN ảo (`PinPad`), cơ chế khóa an toàn khi nhập sai mã PIN nhiều lần (Lockout security). |
| **SUITE-04** | `tests/test_services_and_storage.py` | **44** | Dịch vụ hồ sơ sinh trắc học khuôn mặt (`FaceProfileService`), mã hóa vector đặc trưng bằng Fernet, băm PIN bằng PBKDF2-SHA256, đóng gói và giải nén bài giảng `.airteach` (`AirTeachPackager`), trợ lý `GeminiAssistant` (kèm cơ chế tự động chuyển hướng ngoại tuyến khi mất mạng). |
| **SUITE-05** | `tests/test_svg_assets_and_trainer.py` | **14** | Tính toàn vẹn của 33 tệp tài nguyên đồ họa vector SVG (`assets/`), vòng đời tương tác của Hộp thoại Huấn luyện Cử chỉ (`GestureTrainerDialog`), Hộp thoại Kiểm duyệt và kết xuất công thức toán học (`SmartInkReviewDialog`). |
| **SUITE-06** | `tests/test_stem_and_tracking.py` | **29** | Bộ công cụ chuyên đề STEM (Toán Oxy/Hàm số, Vật lý Lực/Phân tích trực giao/Mạch điện, Hóa học Công thức Unicode/Phản ứng/Benzen, Sinh học Ghim định vị, Tin học Sơ đồ khối/Code IDE), Bộ bảo vệ tọa độ chống nhân đôi (Anti-Double-Scaling), thuật toán ngón gập (Finger Curl), cờ dòng lệnh CLI (`--version`, `--check`), bộ khóa IPC Mutex đơn phiên và lấy mẫu đường cong Bézier. |
| **TỔNG CỘNG** | **6 Bộ Kiểm Thử** | **279** | **Bao phủ 100% các mô-đun chức năng cốt lõi của ứng dụng.** |

---

### 3. TIÊU CHÍ NGHIỆM THU CHI TIẾT (ACCEPTANCE CRITERIA)

Mỗi mô-đun được coi là đạt yêu cầu khi đáp ứng các tiêu chuẩn khắt khe sau:

#### A. Cử chỉ Bàn tay & Vẽ nét trên không (Hand Tracking & Air Drawing)
- Tỉ lệ nhận diện đúng của 6 cử chỉ sư phạm (`POINT`, `DRAW`, `PAUSE`, `SELECT`, `SWIPE`, `CONFIRM`) đạt $\ge 95\%$ trên dữ liệu chuẩn hóa.
- Tọa độ nét vẽ không bao giờ vượt ra ngoài ranh giới kích thước canvas $1920 \times 1080$.
- Khi giáo viên hạ tay xuống hoặc đưa tay ra khỏi tầm nhìn camera, hệ thống phát tín hiệu ngắt nét tức thì, không phát sinh đường nối chéo khi đưa tay lên lại.

#### B. Làm mịn Nét mực & Nắn Hình học (Stroke Beautifier & Smart Ink)
- Bộ lọc 1€ Filter triệt tiêu tối thiểu $10\%$ độ rung giật ngón tay khi viết chậm, với thời gian xử lý $< 0.1\text{ ms}$ cho mỗi điểm.
- Thuật toán RDP giảm thiểu từ $70\%$ đến $85\%$ số điểm dữ liệu dư thừa.
- Nắn hình học tự động nhận diện chính xác $100\%$ các hình cơ bản: Đường thẳng, Đường tròn, Hình chữ nhật với độ tin cậy $\ge 0.90$.

#### C. Sinh trắc học & Bảo mật (Face Authentication & Security)
- Nhận diện khuôn mặt giáo viên chạy $100\%$ offline trên CPU thông qua OpenCV DNN YuNet và SFace.
- Thời gian trích xuất vector đặc trưng $< 50\text{ ms}$ trên CPU thông thường.
- Vector đặc trưng 128 chiều bắt buộc phải được mã hóa bằng Fernet trước khi lưu trữ vào SQLite. Mã PIN dự phòng băm bằng PBKDF2-SHA256 với muối bảo mật ngẫu nhiên.

#### D. Trình khởi chạy Windows 1-Click (Launcher)
- `Run_AirTeach_AI.bat` phải khởi chạy thành công trên Windows 10/11 mà không cần bất kỳ thao tác cấu hình thủ công nào.
- Hỗ trợ đầy đủ cờ dòng lệnh `--check`, `--version`, `--dry-run` và xuất mã thoát chuẩn xác.

---

### 4. QUY TRÌNH THỰC THI VÀ BÁO CÁO KIỂM THỬ (EXECUTION PROTOCOL)

1. **Kiểm thử tự động hóa toàn diện:**
   ```powershell
   python -m pytest -v
   ```
2. **Kiểm thử đo lường hiệu năng thực nghiệm:**
   ```powershell
   python scripts/benchmark_suite.py
   ```
3. **Kiểm thử khởi chạy 1-Click:**
   ```cmd
   Run_AirTeach_AI.bat --check
   Run_AirTeach_AI.bat --version
   ```
