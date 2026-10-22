# AirTeach AI 🎓✋

**Nền tảng giảng dạy tương tác không chạm** — Gesture-controlled interactive teaching platform powered by Computer Vision and AI.

> **Yêu cầu:** Python 3.10 – 3.12 | PySide6 | OpenCV | MediaPipe | Google Gemini API

---

## Mục lục / Table of Contents

1. [Tính năng / Features](#tính-năng--features)
2. [Yêu cầu hệ thống / System Requirements](#yêu-cầu-hệ-thống--system-requirements)
3. [Cài đặt môi trường / Environment Setup](#cài-đặt-môi-trường--environment-setup)
4. [Tải mô hình AI / Model Download](#tải-mô-hình-ai--model-download)
5. [Khởi chạy phần mềm / Running the App](#khởi-chạy-phần-mềm--running-the-app)
6. [Kiểm thử / Running Tests](#kiểm-thử--running-tests)
7. [Đóng gói thành .exe / PyInstaller Packaging](#đóng-gói-thành-exe--pyinstaller-packaging)
8. [Xuất số liệu CSV / Telemetry CSV Export](#xuất-số-liệu-csv--telemetry-csv-export)
9. [Cấu trúc dự án / Project Structure](#cấu-trúc-dự-án--project-structure)
10. [Xử lý sự cố / Troubleshooting](#xử-lý-sự-cố--troubleshooting)

---

## Tính năng / Features

| Tính năng | Mô tả |
|-----------|-------|
| ✋ **Cử chỉ tay không chạm** | MediaPipe Hand Landmarker (21 điểm 3D) → FSM 6 trạng thái: POINT, DRAW, PAUSE, SELECT, SWIPE, CONFIRM |
| 🎨 **Nét vẽ vector 3 tầng** | 1€ Filter → RDP → Cubic Bézier C¹ + độ dày biến thiên theo vận tốc |
| 📐 **Chuẩn hóa hình học** | Tự động nắn thành Đường thẳng, Mũi tên, Hình tròn/Elip, Hình chữ nhật, Tam giác |
| 👤 **Đăng nhập khuôn mặt** | YuNet (phát hiện) + SFace (nhận diện) ONNX — mã hóa Fernet, lưu SQLite |
| 🔤 **Nhận diện chữ viết tay** | RapidOCR offline (Tiếng Việt + Tiếng Anh) |
| 🧮 **Nhận diện công thức toán** | rapid-latex-ocr / pix2tex → LaTeX → dựng PNG bằng matplotlib |
| 🤖 **Trợ lý AI** | Google Gemini API (gemini-2.5-flash) + chế độ offline tự động |
| 📚 **Nhập tài liệu** | PDF (pypdfium2), PPTX/DOCX (LibreOffice hoặc MS Office COM) |
| 💾 **Định dạng .airteach** | ZIP bundle: trang PNG + nét Bézier JSON + timestamps thực + kết quả OCR |
| ⏪ **Phát lại bài giảng** | Replay theo thời gian thực (1x/2x/4x), tuabin scrubber |

---

## Yêu cầu hệ thống / System Requirements

- **OS:** Windows 10/11 (64-bit) _(macOS/Linux có thể chạy nhưng chưa được kiểm thử chính thức)_
- **Python:** 3.10, 3.11, hoặc 3.12 (**khuyến nghị 3.11**). _Python 3.13+ chưa được hỗ trợ bởi MediaPipe._
- **RAM:** ≥ 8 GB (khuyến nghị 16 GB)
- **Webcam:** Camera USB hoặc tích hợp (độ phân giải ≥ 720p)
- **GPU:** Tùy chọn (CPU inference đã đủ tốt)
- **Kết nối Internet:** Chỉ cần khi dùng Gemini AI Assistant; phần mềm vẫn hoạt động offline

---

## Cài đặt môi trường / Environment Setup

### Bước 1 — Tạo môi trường ảo Python (Virtual Environment)

```powershell
# Windows PowerShell — chạy từ thư mục gốc dự án
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> **Lưu ý:** Nếu gặp lỗi `execution policy`, chạy: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

### Bước 2 — Cài đặt thư viện

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> **macOS/Linux:**
> ```bash
> python3 -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> ```

### Bước 3 — Cấu hình Gemini API Key

Tạo (hoặc chỉnh sửa) tệp `.env` trong thư mục gốc:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```

Lấy API key miễn phí tại [Google AI Studio](https://aistudio.google.com/app/apikey).  
Bạn cũng có thể cập nhật API key trực tiếp trong phần mềm qua **⚙ Cài đặt → Gemini API Key**.

---

## Tải mô hình AI / Model Download

Các mô hình ONNX (YuNet, SFace, MediaPipe Hand) sẽ **tự động tải** khi khởi chạy lần đầu.  
Để tải thủ công hoặc kiểm tra tính toàn vẹn:

```powershell
python -c "
from pathlib import Path
from src.utils.model_downloader import ensure_all_models
results = ensure_all_models(Path('models'))
for name, ok in results.items():
    status = 'OK' if ok else 'FAILED'
    print(f'  [{status}] {name}')
"
```

Các mô hình được tải về thư mục `models/`:

| Mô hình | Nguồn | Kích thước |
|---------|-------|------------|
| `face_detection_yunet_2023mar.onnx` | opencv/opencv_zoo | ~230 KB |
| `face_recognition_sface_2021dec.onnx` | opencv/opencv_zoo | ~37 MB |
| `hand_landmarker.task` | MediaPipe Google Storage | ~8.3 MB |

---

## Khởi chạy phần mềm / Running the App

```powershell
# Kích hoạt môi trường ảo trước
.\.venv\Scripts\Activate.ps1

# Khởi chạy AirTeach AI
python main.py
```

### Luồng sử dụng cơ bản

1. **Màn hình đăng nhập** → Nhìn vào webcam để nhận diện khuôn mặt **hoặc** nhập mã PIN
2. **Lần đầu sử dụng** → Bấm **"Đăng ký Giáo viên mới"** → Nhìn vào webcam ở 5 góc khác nhau → Điền tên, môn dạy, mã PIN 4–6 số
3. **Thư viện bài giảng** → Mở bài giảng cũ (`.airteach`) **hoặc** tạo bài mới / nhập PPTX/PDF
4. **Bảng giảng dạy** → Giơ tay trước webcam:
   - ☝️ **Trỏ** = Con trỏ laser đỏ
   - ✍️ **Kẹp ngón cái + trỏ** = Viết nét
   - ✊ **Nắm tay** = Nhấc bút (kết thúc nét)
   - ✌️ **Trỏ + giữa** = Chọn vùng
   - 🤚 **Lòng bàn tay** + quẹt = Lật trang

### Tùy chọn Camera Index

Nếu phần mềm dùng sai camera, chỉnh trong **⚙ Cài đặt → Camera Index** (0 = mặc định, 1 = camera ngoài).

---

## Kiểm thử / Running Tests

```powershell
# Headless (không cần màn hình thực — dùng cho CI/CD)
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest tests\ -v --tb=short

# Kết quả thực tế (Phiên bản v1.0.0):
# 238 passed, 1 warning in ~47s (100% pass rate)

# Chạy kiểm thử luồng thực tế đầu-cuối (E2E smoke test):
python scripts/verify_e2e_flow.py
```

### Chạy từng nhóm test

```powershell
# Kiểm thử thuật toán lõi (1€ Filter, RDP, Bézier, Shape Snapping, FSM)
python -m pytest tests/test_core_algorithms.py -v

# Kiểm thử dịch vụ, bảo mật & lưu trữ (YuNet, SFace, Fernet, PBKDF2, .airteach)
python -m pytest tests/test_services_and_storage.py -v

# Kiểm thử biểu tượng SVG & bộ tạo mẫu nền học tập
python -m pytest tests/test_svg_assets_and_trainer.py -v

# Kiểm thử giao diện & phân tầng đồ họa
python -m pytest tests/test_ui_components.py tests/test_phase3_advanced_ui.py -v
```

---

## Đóng gói thành .exe / Standalone Windows Packaging

### Đóng gói tự động bằng kịch bản build đã kiểm định:

```powershell
# Tự động kiểm tra điều kiện tiên quyết, mô hình, asset và gọi PyInstaller
python scripts/build_windows.py
```

Hoặc sử dụng tệp đặc tả PyInstaller chính thức:

```powershell
pyinstaller --noconfirm airteach.spec
```

### Kết quả đóng gói:

Thư mục `dist\AirTeachAI\` chứa toàn bộ ứng dụng độc lập bao gồm `AirTeachAI.exe` (29.03 MB), tất cả mô hình ONNX/MediaPipe và 33 biểu tượng SVG vector. Không cần cài đặt Python trên máy người dùng cuối.

> **Lưu ý PyInstaller:**  
> - Nếu thiếu DLL MediaPipe → thêm `--collect-all mediapipe`  
> - Nếu thiếu font → thêm `--add-data "C:/Windows/Fonts/arial.ttf;fonts"`  
> - File `.env` **không** được đóng gói; người dùng cần đặt `.env` cạnh file `.exe`

---

## Xuất số liệu CSV / Telemetry CSV Export

AirTeach AI ghi lại toàn bộ số liệu phiên giảng dạy (FPS thực tế, độ trễ nhận diện cử chỉ, số nét vẽ, tỷ lệ nhận diện OCR, v.v.)

### Xuất từ giao diện

1. Mở bảng **"📷 Camera & Telemetry"** ở góc phải
2. Bấm nút **"Xuất báo cáo số liệu (.CSV)"**
3. Chọn vị trí lưu → file `airteach_telemetry_<timestamp>.csv` được tạo

### Xuất từ dòng lệnh (script)

```python
from src.utils.telemetry_logger import TelemetryLogger
from pathlib import Path

# Đọc log phiên hiện tại
logger = TelemetryLogger()
logger.start()
# ... (sau khi chạy xong phiên)
logger.stop()

summary = logger.compute_summary()
print(f"Avg FPS: {summary['fps_mean']:.1f}")
print(f"Total strokes: {summary['stroke_count']}")

csv_path = Path("telemetry_export.csv")
logger.export_csv(csv_path)
print(f"Exported to: {csv_path}")
```

### Các cột CSV

| Cột | Mô tả |
|-----|-------|
| `timestamp_ms` | Thời điểm ghi nhận (ms từ epoch) |
| `fps` | Tốc độ khung hình webcam thực tế |
| `gesture_state` | Trạng thái cử chỉ tay (DRAW/POINT/…) |
| `stroke_id` | ID nét vẽ đang xử lý |
| `filter_latency_ms` | Độ trễ 1€ Filter |
| `snap_type` | Loại hình học đã nắn chuẩn |
| `ocr_confidence` | Độ tin cậy OCR (0.0–1.0) |
| `gemini_latency_ms` | Độ trễ phản hồi Gemini API |

---

## Cấu trúc dự án / Project Structure

```
AirTeach AI/
├── main.py                         # Điểm khởi chạy chính
├── requirements.txt                # Danh sách thư viện Python
├── .env                            # Cấu hình API key (KHÔNG commit lên git)
├── README.md                       # Tài liệu này
│
├── src/
│   ├── core/
│   │   ├── hand_gesture_engine.py  # Webcam → MediaPipe → FSM → 1€ Filter
│   │   ├── stroke_beautifier.py    # RDP + Cubic Bézier + ShapeSnapper
│   │   └── liveness_checker.py    # Kiểm tra chống giả mạo khuôn mặt
│   │
│   ├── services/
│   │   ├── face_profile_service.py # YuNet/SFace → Fernet → SQLite
│   │   ├── document_manager.py     # PDF/PPTX/DOCX → .airteach
│   │   ├── gemini_assistant.py     # Google Gemini AI chat + offline fallback
│   │   └── smart_ink_ocr.py        # RapidOCR + LaTeX-OCR + matplotlib render
│   │
│   ├── ui/
│   │   ├── main_window.py          # Cửa sổ chính với thanh công cụ
│   │   ├── login_window.py         # Đăng nhập khuôn mặt + PIN
│   │   ├── lesson_board.py         # QGraphicsScene (Bảng nét vẽ)
│   │   ├── lesson_board_canvas.py  # QGraphicsView (Canvas tương tác)
│   │   ├── library_and_prep_widget.py  # Thư viện bài giảng
│   │   ├── replay_controller.py    # Điều khiển phát lại nét vẽ
│   │   ├── side_panels.py          # AI Chat + Camera Telemetry tabs
│   │   ├── chat_panel.py           # Panel chat Gemini AI
│   │   ├── camera_widget.py        # Hiển thị webcam với landmarks
│   │   ├── face_auth_dialog.py     # Dialog nhận diện/đăng ký khuôn mặt
│   │   └── settings_dialog.py      # Hộp thoại cài đặt
│   │
│   └── utils/
│       ├── model_downloader.py     # Tự động tải YuNet/SFace/Hand ONNX
│       └── telemetry_logger.py     # Ghi số liệu phiên → CSV
│
├── tests/
│   ├── test_core_algorithms.py     # 47 tests: 1€ Filter, RDP, Bézier, Snapper
│   ├── test_services_and_storage.py# 62 tests: Face, OCR, Gemini, DocManager
│   ├── test_ui_components.py       # 71 tests: Tất cả widget giao diện
│   └── test_phase3_advanced_ui.py  # 28 tests: Canvas nâng cao, Replay
│
├── data/
│   ├── profiles.db                 # SQLite: hồ sơ giáo viên (tự tạo)
│   └── workspaces/
│       └── <teacher_uuid>/         # Thư mục bài giảng của từng giáo viên
│
├── models/                         # Trọng số ONNX (tự tải về)
│   ├── face_detection_yunet_2023mar.onnx
│   ├── face_recognition_sface_2021dec.onnx
│   └── hand_landmarker.task
│
└── assets/                         # Biểu tượng, font, hình ảnh giao diện
```

---

## Xử lý sự cố / Troubleshooting

### Camera không nhận diện được

```powershell
# Kiểm tra camera index khả dụng
python -c "
import cv2
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f'Camera {i}: OK')
        cap.release()
    else:
        print(f'Camera {i}: Not found')
"
```

Sau đó chỉnh **Camera Index** trong ⚙ Cài đặt.

### Mô hình ONNX thiếu hoặc lỗi

```powershell
# Tải lại toàn bộ mô hình
python -c "
from pathlib import Path
from src.utils.model_downloader import ensure_all_models
ensure_all_models(Path('models'), force_download=True)
"
```

### Lỗi `ImportError: mediapipe`

MediaPipe **chưa hỗ trợ Python 3.13+**. Hãy dùng Python 3.11:

```powershell
# Kiểm tra phiên bản Python
python --version

# Cài Python 3.11 từ python.org nếu cần
# Sau đó tạo lại venv với python3.11
py -3.11 -m venv .venv
```

### Lỗi `QT_QPA_PLATFORM` khi chạy tests trên server không có màn hình

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python -m pytest tests\ -v
```

### Gemini API trả về lỗi 403 / Invalid Key

1. Kiểm tra `.env` có `GEMINI_API_KEY` hợp lệ
2. Vào **⚙ Cài đặt → Gemini API Key** → nhập key mới
3. Phần mềm tự động chuyển sang chế độ **offline** nếu không có mạng hoặc key sai — mọi tính năng giảng dạy vẫn hoạt động bình thường

### Lỗi LibreOffice khi nhập PPTX/DOCX

- **Windows:** Cài [LibreOffice](https://www.libreoffice.org/download/) hoặc Microsoft Office 365
- Phần mềm tự động phát hiện và dùng LibreOffice hoặc MS Office COM tùy cái nào có sẵn

---

## Giấy phép / License

© 2024–2025 AirTeach AI Team. Mã nguồn phần mềm sử dụng các thư viện mã nguồn mở:
- OpenCV (Apache 2.0), MediaPipe (Apache 2.0), PySide6 (LGPL v3), NumPy (BSD), SciPy (BSD)
- YuNet & SFace models: [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) (Apache 2.0)
- MediaPipe Hand Landmarker: [Google MediaPipe](https://developers.google.com/mediapipe) (Apache 2.0)
# AirTeach-AI
# AirTeach-AI
