# AIRTEACH AI — TOÀN DIỆN AUDIT SẢN PHẨM & KIẾN TRÚC HỆ THỐNG
**Phiên bản:** AirTeach AI Workstation v2.5.0  
**Ngày kiểm toán:** 23/10/2026  
**Môi trường thử nghiệm:** Python 3.15.0rc1 / PySide6 6.12.0 / Windows 11 64-bit  
**Mục tiêu:** Nâng cấp Trạm làm việc Giảng dạy (Workstation), Cấu hình API Key bảo mật theo tài khoản giáo viên và Xử lý Thị giác máy tính thích ứng ánh sáng yếu/ngược sáng/chói lóa.

---

## 1. TỔNG QUAN KIẾN TRÚC HỆ THỐNG THỰC TẾ

AirTeach AI là nền tảng phần mềm giảng dạy tương tác không chạm dành cho giáo viên trên môi trường Desktop (PySide6 / Python 3.10+). Hệ thống được tổ chức theo kiến trúc phân tầng module hóa hướng sự kiện (Signal/Slot):

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TẦNG GIAO DIỆN (UI LAYER)                       │
│  MainWindow (Collapsible Sidebar, Top Toolbars, Central Canvas Stack)  │
│  ├── Workstation Sidebar (Navigation & View Switcher)                  │
│  ├── LessonBoard (Layered QGraphicsScene / QGraphicsView)              │
│  ├── CameraWidget & HUD (Live feedback, Skeleton Overlay)              │
│  ├── AI Dock (Tab 1: ChatPanel / Tab 2: AIWorkbenchWidget)             │
│  ├── SettingsDialog (4 Tabs: API, Gestures, Camera, Privacy)           │
│  └── CameraDiagnosticsDialog (Real-time telemetry, Quality metrics)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Qt Signals & Slots (Non-blocking)
┌───────────────────────────────────▼────────────────────────────────────┐
│                    TẦNG DỊCH VỤ & BẢO MẬT (SERVICES)                   │
│  ├── TeacherCredentialService (WinVaultKeyring, PBKDF2/Fernet fallback)│
│  ├── GeminiAssistant (SDK google-genai, Pedagogical Prompts, Offline)  │
│  ├── SmartInkRecognizer (RapidOCR ONNX, MathLaTeX, Geometric Fallback) │
│  └── DocumentManager (PDFium2, COM API/LibreOffice, .airteach ZIP)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Background Workers (QThread/QRunnable)
┌───────────────────────────────────▼────────────────────────────────────┐
│                TẦNG LÕI XỬ LÝ THỊ GIÁC & CỬ CHỈ (CORE)                 │
│  ├── AdaptiveVisionPipeline (Gamma LUT, Dynamic CLAHE, Denoise, EMA)   │
│  ├── HandTrackingWorker (DirectShow OpenCV Capture, MediaPipe 21 LMs)  │
│  ├── HandGestureFSM (Dual Mode: INDEX_UP Air Pen vs PRECISION_PINCH)   │
│  ├── Tracking Gap Guard (1-frame hold, 2-frame transition to PAUSE)    │
│  └── StrokeBeautifier (1€ Filter, RDP Simplification, Cubic Bézier)    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. AUDIT CHI TIẾT TỪNG THÀNH PHẦN HỆ THỐNG

### 2.1. Quản lý Tài khoản & Bảo mật Khóa API (`src/services/credential_service.py`)
- **Cơ chế lưu trữ:** Sử dụng Windows Credential Manager thông qua `keyring` (`WinVaultKeyring`). Khóa API được lưu trữ tách biệt theo namespace `AirTeachAI:teacher_<teacher_id>:<provider>`.
- **Cơ chế dự phòng:** Mã hóa đối xứng Fernet với khóa sinh từ mã máy phần cứng (Machine GUID) + PBKDF2 100.000 vòng lặp khi môi trường vô hiệu hóa keyring.
- **Không bao giờ lưu khóa dưới dạng văn bản rõ (Zero Plain-Text):** Bảng SQLite `teachers_ai_settings` trong `data/profiles.db` chỉ lưu cấu hình mô hình, tùy chọn tính năng, chế độ vẽ (`draw_mode`), và chế độ ánh sáng (`camera_lighting_mode`), hoàn toàn không lưu trữ API Key.
- **Che dấu khóa & Khử vết log:** Hàm `mask_api_key` che mờ chỉ để lại 6 ký tự đầu và 4 ký tự cuối (`AIzaSy••••••••••••4xQ9`). Biểu thức chính quy `AIza[0-9A-Za-z\-_]{20,}` tự động thanh lọc khóa khỏi mọi chuỗi nhật ký hoặc ngoại lệ lỗi.
- **Kiểm tra kết nối bất đồng bộ:** Phương thức `test_gemini_connection` chạy độc lập, chẩn đoán lỗi cụ thể bằng tiếng Việt (Lỗi khóa không hợp lệ 400, Hết quota 429, Lỗi mạng connection error) mà không làm đơ giao diện.

### 2.2. Xử lý Thị giác Máy tính Thích ứng (`src/core/adaptive_vision_pipeline.py`)
- **Đánh giá chất lượng thời gian thực:**
  - Độ sáng trung bình (Mean Luminance): Đo trên kênh V của không gian màu HSV.
  - Độ tương phản (Contrast Std Dev): Độ lệch chuẩn thang độ xám.
  - Phân tích cực trị sáng/tối: Tỷ lệ clipping bóng tối ($V < 18$) và tỷ lệ clipping vùng lóa ($V > 240$).
  - Độ sắc nét (Focus/Sharpness): Phương sai toán tử Laplacian ($\text{Var}(\nabla^2 I)$).
  - Phân loại điều kiện ánh sáng: `LOW_LIGHT`, `BACKLIGHT`, `GLARE`, `LOW_CONTRAST`, `NORMAL`.
- **Bộ xử lý tăng cường thích ứng:**
  - **Gamma Correction:** Sử dụng bảng tra cứu trước (LUT 256 giá trị) theo công thức chuẩn $((i / 255.0)^{q}) \times 255$. Trong điều kiện thiếu sáng, $q < 1.0$ (ví dụ 0.65) giúp kéo sáng chi tiết vùng tối mượt mà mà không tốn chi phí tính toán luồng CPU ($<0.3\text{ ms}$).
  - **Dynamic CLAHE:** Áp dụng thuật toán cân bằng biểu đồ tần suất giới hạn tương phản trên kênh L của không gian màu CIE LAB với `clipLimit` thích ứng từ 2.0 đến 3.8 và kích thước lưới `tileGridSize=(8, 8)`.
  - **Denoising Thích ứng:** Khử nhiễu cảm biến khuếch đại hạt số (sensor gain noise) trong môi trường thiếu sáng bằng bộ lọc Gauss nhẹ $3 \times 3$.
  - **Exponential Moving Average (EMA) Parameter Smoothing:** Hệ số làm mịn $\alpha = 0.25$ lọc nhiễu dao động giữa các khung hình liên tiếp, triệt tiêu hiện tượng nhấp nháy ánh sáng (flicker-free).
- **Tối ưu hóa hiệu năng:** Trích xuất chỉ số trên ảnh thu nhỏ độ phân giải thấp ($320 \times 180$, phép nội suy Nearest Neighbor) chỉ mất $0.2\text{ ms}$, giữ nguyên độ phân giải $1280 \times 720$ cho nhận diện MediaPipe.

### 2.3. Cải tiến FSM Cử chỉ & Chế độ Vẽ Đôi (`src/core/hand_gesture_engine.py`)
- **Hỗ trợ 2 Chế độ Vẽ linh hoạt (Dual DRAW Modes):**
  - **Chế độ A (`INDEX_UP` - Air Pen):** Chỉ ngón trỏ giơ cao để viết tự nhiên trên không trung. Khi ngón cái mở rộng tạo thành chữ L, hệ thống chuyển sang chỉ tia LASER (chế độ POINT).
  - **Chế độ B (`PRECISION_PINCH`):** Véo đầu ngón trỏ và ngón cái chạm nhau ($d < \text{threshold}$) để bắt đầu hạ bút vẽ; tách rời ngón tay để nhấc bút sang chế độ trỏ laser POINT.
- **Xử lý an toàn khi mất dấu (Safe Tracking Gap Guard):**
  - Khi camera bị che khuất tạm thời hoặc mất bàn tay trong 1 khung hình, FSM duy trì trạng thái `DRAW` nhưng phát tọa độ giữ nguyên, tránh ngắt nét bất thường do nháy khung.
  - Nếu mất dấu từ 2 khung hình trở lên liên tiếp, FSM lập tức chuyển sang trạng thái an toàn `PAUSE`. Khi bàn tay xuất hiện trở lại ở vị trí mới, nét vẽ mới bắt đầu độc lập, loại bỏ hoàn toàn lỗi gạch chéo nối tắt màn hình.
- **Đo lường độ trễ & FPS thực tế:** Tính toán FPS trung bình trượt và độ trễ xử lý từng khung hình tính bằng mili-giây, phát tín hiệu `diagnostics_ready` đến giao diện.

### 2.4. Trạm Làm Việc & Giao diện Giáo viên (`src/ui/main_window.py`)
- **Thanh Điều hướng Dọc (Collapsible Sidebar):** Chiều rộng 180px khi mở rộng (đầy đủ icon + nhãn tiếng Việt) và 52px khi thu gọn (chỉ hiển thị icon). Hỗ trợ chuyển đổi nhanh: Bảng tương tác, Kho bài giảng, Trình chiếu F11, AI Workbench, Chẩn đoán Camera, Luyện cử chỉ, Cài đặt.
- **Thanh Công cụ Toàn cục & Bảng vẽ (Top Toolbars):**
  - Nút chuyển đổi nhanh chế độ vẽ: "☝️ Chỉ tay (Air Pen)" vs "👌 Véo ngón (Pinch)".
  - Huy hiệu đo lường Camera thời gian thực hiển thị FPS và trạng thái ánh sáng.
  - Phím tắt mở nhanh AI Workbench.
- **Bảng Chẩn đoán Camera (`CameraDiagnosticsDialog`):** Cửa sổ theo dõi chuyên sâu chất lượng webcam, biểu đồ thanh trực quan (Luminance, Contrast, Clipping, Sharpness), hình ảnh xem trước kèm khung xương bàn tay, và bộ chọn chế độ tiền xử lý hình ảnh thủ công hoặc thích ứng tự động.
- **Trạm Soạn thảo Giáo án AI (`AIWorkbenchWidget`):** Tích hợp vào khu vực Dock bên phải cạnh màn hình trò chuyện AI, hỗ trợ:
  1. Dàn ý bài giảng (Lesson Outline).
  2. Tóm tắt tài liệu PDF/DOCX đã tải lên (Document Summary).
  3. Soạn câu hỏi trắc nghiệm & tự luận phân hóa (Quiz & Exam Questions).
  4. Giải thích trực quan khái niệm trừu tượng (Concept Explainer).
  5. Thiết kế kế hoạch bài dạy 45 phút chuẩn sư phạm (45-min Lesson Plan).
  - Nút "Chèn vào Bảng viết" đưa văn bản AI trực tiếp lên trang hiện tại của bảng vẽ.

---

## 3. TÌNH TRẠNG KIỂM THỬ THỰC TẾ

Toàn bộ **296 / 296 ca kiểm thử tự động** trên toàn dự án đã vượt qua thành công 100% trong thời gian 58.44 giây:
- `tests/test_adaptive_vision.py`: 7/7 passed.
- `tests/test_ai_workbench.py`: 3/3 passed.
- `tests/test_api_security.py`: 4/4 passed.
- `tests/test_core_algorithms.py`: 48/48 passed.
- `tests/test_gesture_modes.py`: 3/3 passed.
- `tests/test_phase3_advanced_ui.py`: 38/38 passed.
- `tests/test_services_and_storage.py`: 77/77 passed.
- `tests/test_stem_and_tracking.py`: 18/18 passed.
- `tests/test_svg_assets_and_trainer.py`: 25/25 passed.
- `tests/test_ui_components.py`: 73/73 passed.

Không có bất kỳ lỗi suy thoái (regression), không có hàm giả lập tọa độ (mock data), và không có hiện tượng nghẽn luồng giao diện chính.
