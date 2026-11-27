# AirTeach AI (DNT - Victer)

**Bảng Thông Minh Không Chạm Thế Hệ Mới** — Điều khiển bằng cử chỉ 2 tay 60 FPS (MediaPipe Hand Landmarker + Camera chống ngược sáng 3m), trình chiếu trực tiếp PDF / Word (`.docx`) / PowerPoint (`.pptx`), nhận dạng khuôn mặt ArcFace 512-D, ghi âm bài giảng và Trợ lý Sư phạm AI (Hybrid Local LLM + Gemini).

---

## Hướng dẫn Cài đặt & Chạy Nhanh 2 Bước (Windows 10 / 11 — Chuẩn Máy Thi KHKT)

### Bước 1: Chạy `setup_and_build.bat` để Tự động Cài đặt & Đóng gói `.exe`
Nháy đúp hoặc chạy trong Terminal tại thư mục gốc dự án:
```bat
setup_and_build.bat
```
Script sẽ tự động thực hiện trọn gói từ máy trắng:
1. Kiểm tra & cài đặt **Microsoft Visual C++ 2015–2022 Redistributable (x64)** và **Python 3.11 (64-bit)** (ưu tiên `winget`, tự dự phòng bằng PowerShell nếu thiếu `winget`).
2. Tạo môi trường ảo cô lập `.venv` và cài đầy đủ toàn bộ thư viện trong `requirements.txt` (`PySide6`, `mediapipe`, `opencv-python`, `PyMuPDF`, `python-docx`, `python-pptx`, `Pillow`, `numpy`, `google-genai`, `pywin32`, `requests`, `pyinstaller`, `maturin`, ...).
3. Tự kiểm tra **Rust Toolchain (`cargo`)**; nếu có sẽ biên dịch nhân tăng tốc `rust_core` (`airteach_core_rs`), nếu máy không cài Rust sẽ tự thông báo rõ và dùng Python fallback mượt mà.
4. Tự động chạy `scripts/fetch_assets.py` để tải & kiểm tra SHA-256:
   - Model MediaPipe `assets/models/hand_landmarker.task` & `models/blaze_face_short_range.tflite`
   - Bộ font tiếng Việt đầy đủ dấu `assets/fonts/NotoSans-Regular.ttf`, `NotoSans-Bold.ttf`
   - Biểu tượng đa kích thước `assets/icons/airteach.ico`
5. Đóng gói bản độc lập bằng `AirTeachAI.spec` vào `dist\AirTeachAI\AirTeachAI.exe`, sao chép `Chay_AirTeachAI_AutoFix.bat` sang bên cạnh `.exe` và tự kiểm thử (`--smoke-test`).

---

### Bước 2: Chạy `Chay_AirTeachAI_AutoFix.bat` để Mở trên Máy Thi
Copy toàn bộ thư mục **`dist\AirTeachAI\`** sang máy thi (hoặc chạy trực tiếp từ USB) và nháy đúp:
```bat
Chay_AirTeachAI_AutoFix.bat
```
Trình khởi chạy bảo vệ máy thi sẽ tự động:
- Kiểm tra `vcruntime140_1.dll` trong `C:\Windows\System32`; nếu thiếu sẽ tự tải và cài đặt im lặng `vc_redist.x64.exe /install /quiet /norestart`.
- Khởi tạo sẵn các thư mục dữ liệu ghi tại `%APPDATA%\AirTeachAI\` (`logs`, `doc_cache`, `face_embeddings`, `recordings`, `models`) để ứng dụng chạy ổn định ngay cả khi mở từ thư mục chỉ đọc hoặc USB.
- Kiểm tra & tự bật quyền truy cập **Camera Windows** cho ứng dụng desktop.
- Khởi chạy `AirTeachAI.exe`. Nếu phát sinh lỗi thoát khác `0`, tự động xuất báo cáo chẩn đoán tiếng Việt ra **`crash_log.txt`** kèm 60 dòng log gần nhất từ `%APPDATA%\AirTeachAI\logs\app.log`.

---

## Điều khiển Cử chỉ Không chạm & Phím tắt
- **1 Ngón trỏ giơ lên:** Vẽ nét mực mượt lên bảng / Slide (`DRAW`).
- **2 Ngón (Trỏ + Giữa):** Di chuyển con trỏ định hướng không để lại mực (`HOVER`).
- **Chụm ngón cái + trỏ:** Nhấn nút giao diện / kích hoạt công cụ (`CLICK`).
- **Xòe 5 ngón tay:** Tẩy nét vẽ thông minh (`ERASE`).
- **Tay trái hỗ trợ (`Fist` / `2 Ngón`):** Khóa đường thẳng (`LINE_LOCK`) / khóa hình học (`SHAPE_LOCK`), thu phóng 2 tay.

## Cấu hình Gemini API Key (Tùy chọn)
- Nhập trực tiếp trong giao diện cài đặt của AirTeach AI (lưu an toàn tại `%APPDATA%\AirTeachAI\config.json`), hoặc copy `.env.example` thành `.env` và điền `GEMINI_API_KEY`.
- Khi mất mạng hoặc không có API Key, phần mềm vẫn hoạt động đầy đủ bảng vẽ, nhận diện cử chỉ tay, trình chiếu Word/PDF/PPTX và kết nối Local LLM (Ollama / LM Studio / llama.cpp).
