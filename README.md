# AirTeach AI (DNT - Victer)

**Bảng Thông Minh Không Chạm Thế Hệ Mới** — Ứng dụng giảng dạy tương tác chạy cục bộ 60 FPS kết hợp lõi tính toán **Rust Native (`airteach_core_rs` v2.0.0)**, nhận diện cử chỉ **1 Tay & 2 Tay** (MediaPipe Hand Landmarker), bảng vẽ vô tận theo kiến trúc **Object Canvas**, **Bảng Chọn Tròn Chuột Phải 8 hướng + vòng chọn con**, trình chiếu đa định dạng **PDF / PPTX / DOCX / Ảnh**, đính kèm tệp tiếng Việt UTF-8 vào Trợ lý AI và kiến trúc **AI Hybrid (Gemini Cloud Full Model ↔ Local AI GGUF Offline)**.

---

## Mục Lục

1. [Tổng quan tính năng nổi bật](#1-tổng-quan-tính-năng-nổi-bật)
2. [Kiến trúc hiệu năng Rust + Python PySide6](#2-kiến-trúc-hiệu-năng-rust--python-pyside6)
3. [Bảng tra cứu Cử chỉ 1 Tay & 2 Tay](#3-bảng-tra-cứu-cử-chỉ-1-tay--2-tay)
4. [Bảng Phím tắt, Object Canvas & Bảng chọn tròn Chuột phải](#4-bảng-phím-tắt-object-canvas--bảng-chọn-tròn-chuột-phải)
5. [Trình chiếu đa định dạng & Đính kèm file UTF-8 vào AI Chatbot](#5-trình-chiếu-đa-định-dạng--đính-kèm-file-utf-8-vào-ai-chatbot)
6. [AI Cloud (Full Model Gemini) & AI Local Offline (GGUF Model Hub)](#6-ai-cloud-full-model-gemini--ai-local-offline-gguf-model-hub)
7. [Interactive Spotlight Tour & Phòng Thử Cử Chỉ (Gesture Sandbox)](#7-interactive-spotlight-tour--phòng-thử-cử-chỉ-gesture-sandbox)
8. [Cài đặt & Đóng gói `.exe` 1-Click trên Windows 10/11](#8-cài-đặt--đóng-gói-exe-1-click-trên-windows-1011)
9. [Thư mục cấu hình an toàn (`%APPDATA%\AirTeachAI\`) & Xử lý sự cố](#9-thư-mục-cấu-hình-an-toàn-appdataairteachai--xử-lý-sự-cố)

---

## 1. Tổng quan tính năng nổi bật

- **Điều khiển không chạm 1 Tay & 2 Tay (60 FPS):** Nhận diện chính xác từ khoảng cách `0.5m – 3.2m`, chống ngược sáng tự động, lọc nhiễu không trễ và chống tạo nét vẽ thừa (*Zero-Stray-Stroke FSM*).
- **Lõi hiệu năng cao viết bằng Rust (`airteach_core_rs`):** Xử lý lọc tín hiệu, dự đoán động học, phân loại cử chỉ 3D, không gian 2D Quadtree, nắn hình học thông minh và xử lý ảnh đa luồng dưới mức mili-giây (`< 1 ms`).
- **Object Canvas độc lập & Infinite Pan/Zoom (`25% – 400%`):** Mọi nét mực, hình khối, văn bản, giấy ghi chú (Sticky Note) và ảnh chèn đều là đối tượng độc lập có thể chọn, di chuyển, co giãn 8 góc, xoay, đổi màu, nhân bản và hoàn tác/làm lại nhiều cấp.
- **Bảng Chọn Tròn Chuột Phải (Radial Pie Menu):** Nhấn giữ chuột phải bất kỳ đâu trên bảng để chọn nhanh 8 công cụ chính và kéo ra vòng ngoài chọn màu mực, loại hình khối, tạo trang mới hoặc mở AI Chat.
- **Trình chiếu đa định dạng không phụ thuộc phần mềm văn phòng:** Mở trực tiếp `.pdf`, `.pptx`, `.ppt`, `.docx`, `.doc`, ảnh đơn và thư mục ảnh với hệ thống chuyển đổi 3 tầng (LibreOffice → MS Office COM → Pure-Python offline fallback) + cache SHA-256.
- **Trợ lý Sư phạm AI Hybrid (Cloud ↔ Offline):** Hỗ trợ đầy đủ danh mục **Gemini Cloud** (có tự xoay vòng nhiều API Key & fallback) và **Local AI Offline** với kho tải model `.gguf` trực tiếp từ HuggingFace/GitHub có hỗ trợ tạm dừng/tải tiếp (HTTP Range Resume) và kiểm tra SHA-256.
- **Đính kèm tệp tiếng Việt UTF-8 sạch vào Chatbot:** Đọc `.pdf`, `.docx`, `.pptx`, `.txt`, `.md`, `.csv`, `.json` và hình ảnh có chuẩn hóa Unicode NFC, không lỗi vỡ dấu trên Windows; câu trả lời hỗ trợ Streaming, Markdown Code Block, Bảng biểu, LaTeX và nút *Đưa nội dung lên bảng*.
- **Onboarding trực quan:** Hướng dẫn **Interactive Spotlight Tour (6 bước)** chiếu sáng từng khu vực giao diện và **Phòng Thử Cử Chỉ (Gesture Sandbox)** phản hồi trực tiếp qua Webcam.
- **Sẵn sàng máy thi / triển khai thực tế:** Bộ script 1-click (`setup_all.bat`, `build_exe.bat`, `auto_setup_and_build.bat`, `run_airteach.bat`, `Chay_AirTeachAI_AutoFix.bat`) đóng gói độc lập kèm DLL VC++ và cơ chế tự chẩn đoán lỗi.

---

## 2. Kiến trúc hiệu năng Rust + Python PySide6

AirTeach AI tách biệt rạch ròi giữa **tầng tính toán thời gian thực viết bằng Rust (`rust_core/`)** biên dịch qua **PyO3 / Maturin** thành extension native `airteach_core_rs`, và **tầng giao diện / trình chiếu / AI viết bằng Python PySide6 (`main.py`, `cv_engine/`)**.

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                   AirTeach AI — PySide6 Desktop Studio                     │
│  SmartboardCanvas • RadialPieMenu • SpotlightTour • AI Chatbot • Presenter │
└───────────────────┬────────────────────────────────────┬───────────────────┘
                    │ Python Bridge (auto fallback)      │ Async Workers
                    ▼                                    ▼
┌──────────────────────────────────────────┐  ┌──────────────────────────────┐
│   Rust Native Core (`airteach_core_rs`)  │  │   MediaPipe • PyMuPDF • AI   │
│  • RustOneEuroFilter 1D/2D/3D            │  │  • Hand Landmarker (21 3D)   │
│  • RustKalmanVelocityPredictor2D         │  │  • PyMuPDF + Office/PyRender │
│  • RustKinematic60HzExtrapolator         │  │  • Gemini Cloud Full Models  │
│  • RustGestureStateMachine (3D FSM)      │  │  • Local GGUF / Ollama / LM  │
│  • RustDualHandStateMachine & Swipe      │  └──────────────────────────────┘
│  • RustSpatialIndex (2D Quadtree)        │
│  • Catmull-Rom + RDP + Shape Engine      │
│  • Multi-threaded Image Resize/Sharpen   │
└──────────────────────────────────────────┘
```

### Các mô-đun tăng tốc trong `airteach_core_rs` (v2.0.0)

| Thành phần Rust | Chức năng cốt lõi | Vai trò trong trải nghiệm thực tế |
|---|---|---|
| `RustOneEuroFilter1D` / `2D` / `3D` | Lọc thông thấp thích ứng vận tốc | Giữ đầu bút đứng yên không rung khi viết chậm, bám sát không trễ khi lia tay nhanh |
| `RustKalmanVelocityPredictor2D` & `RustKinematic60HzExtrapolator` | Dự đoán quỹ đạo & ngoại suy 60Hz | Bù độ trễ webcam 30 FPS lên chuyển động con trỏ mượt 60 FPS |
| `classify_hand_3d_rs` & `RustGestureStateMachine` | Phân loại 21 khớp 3D bất biến tỉ lệ + FSM chống nét thừa | Nhận diện ổn định từ `0.5m` đến `3.2m`, cắt đuôi nét thừa (*Tail-Trimming*), giữ nét qua khoảng mất dấu `150ms` |
| `RustDualHandStateMachine`, `RustSwipeDetector`, `compute_dual_hand_kinematics_rs` | Điều phối cử chỉ 2 tay & vẫy chuyển trang | Dual-Pinch Zoom mượt quanh tâm 2 tay, Dual-Palm Pan, khóa nét thẳng/hình học, chống lật slide nhầm |
| `RustSpatialIndex` (2D Quadtree), `hit_test_objects_rs`, `marquee_select_objects_rs` | Chỉ mục không gian 2D + hit-test OBB/polyline | Chọn đối tượng, kéo khung Marquee và viewport culling `< 1 ms` ngay cả khi bảng có hàng nghìn Object |
| `rdp_simplify_rs`, `resample_arc_length_rs`, `catmull_rom_spline_rs`, `snap_geometric_shape_rs` | Rút gọn điểm RDP, nội suy Catmull-Rom, nắn hình học | Biến nét vẽ tay thành đường cong mềm mại hoặc hình tròn, vuông, chữ nhật, tam giác, đường thẳng bắt góc, mũi tên chuẩn xác |
| `preprocess_image_resize_sharpen_rs`, `analyze_frame_luma_rs`, `apply_gamma_contrast_rs`, `compute_roi_crop_box_rs` | Resize/Sharpen đa luồng (`std::thread::scope`, nhả GIL) + chống ngược sáng & ROI | Render ảnh/tài liệu sắc nét không nghẽn UI; tự tăng sáng vùng tối/ngược sáng cho camera |

> **Cơ chế an toàn (`cv_engine/rust_bridge.py`):** Nếu chạy trên môi trường chưa cài Rust Toolchain, ứng dụng tự động chuyển sang bộ thực thi dự phòng thuần Python tương thích hoàn toàn API mà không gây gián đoạn.

---

## 3. Bảng tra cứu Cử chỉ 1 Tay & 2 Tay

### Cử chỉ 1 Tay

| Cử chỉ | Ký hiệu / Trạng thái | Tác vụ trên bảng | Ghi chú kỹ thuật |
|---|---|---|---|
| **Giơ 1 Ngón Trỏ** | `INDEX` (`DRAW`) | Vẽ nét mực mượt 60Hz | Tự động làm mượt Catmull-Rom & hỗ trợ Nắn hình thông minh |
| **Giơ 2 Ngón Chữ V (Trỏ + Giữa)** | `TWO_FINGERS` (`HOVER`) | Di chuyển con trỏ / Chỉ bài | Không để lại nét mực thừa khi chỉ giảng |
| **Chụm Ngón Cái + Ngón Trỏ** | `PINCH` (`GRAB` / `CLICK`) | Cầm & kéo Object trên bảng / Kích hoạt nút | Ngưỡng vào `< 0.28` giữ `>= 3` frame, ngưỡng thoát `> 0.40` chống nhấp nháy |
| **Xòe cả 5 Ngón** | `OPEN_PALM` (`ERASE`) | Tẩy nét thông minh / Dừng vẽ | Hướng lòng bàn tay về camera; bán kính tẩy hiển thị trực quan |
| **Giơ 3 Ngón Tay** | `THREE_FINGERS` (`UNDO`) | Hoàn tác thao tác gần nhất | Giữ ổn định `~220ms`, cooldown `400ms` chống hoàn tác lặp |
| **Vẫy Tay Ngang (Trái / Phải)** | `SWIPE_LEFT` / `SWIPE_RIGHT` | Trang kế tiếp / Trang trước | Bảo vệ tuyệt đối mực trên slide (không tẩy nhầm khi vẫy) |
| **Vẫy Tay Dọc (Lên / Xuống)** | `SWIPE_UP` / `SWIPE_DOWN` | Cuộn xuống / Cuộn lên | Hoạt động trong chế độ **Cuộn tài liệu liên tục** |

### Cử chỉ 2 Tay

| Cử chỉ 2 Tay | Chế độ FSM | Tác vụ trên bảng | Mô tả thao tác |
|---|---|---|---|
| **2 Tay Cùng Chụm (Dual Pinch)** | `DUAL_PINCH_ZOOM` | Phóng to / Thu nhỏ bảng (`25% – 400%`) | Kéo 2 tay ra xa để phóng to, khép lại để thu nhỏ quanh đúng tâm giữa 2 tay |
| **2 Tay Cùng Nắm hoặc Cùng Xòe** | `DUAL_PALM_PAN` | Kéo trượt bảng vô tận / Cuộn PDF & DOCX | Di chuyển 2 tay song song theo mọi hướng để Pan bảng mượt mà |
| **Tay phụ NẮM + Tay chính VẼ** | `LINE_LOCK` | Khóa vẽ đoạn thẳng tuyệt đối | Tự bắt góc chuẩn (`0°`, `45°`, `90°`) khi kẻ trục tọa độ, bảng biểu |
| **Tay phụ GIƠ 2 NGÓN + Tay chính VẼ** | `SHAPE_LOCK` | Tăng cường chuẩn hóa hình học | Ép chuẩn hình tròn, vuông, chữ nhật, tam giác, mũi tên vector |
| **Tay phụ GIƠ 3 NGÓN** | `UNDO` | Hoàn tác tức thì | Hỗ trợ hoàn tác nhanh bằng tay trái trong khi tay phải đang thao tác |

---

## 4. Bảng Phím tắt, Object Canvas & Bảng chọn tròn Chuột phải

### Bảng Phím tắt & Điều hướng Chuột

| Phím tắt / Thao tác chuột | Chức năng |
|---|---|
| `V` | Công cụ **Chọn / Di chuyển / Resize / Xoay Object** (`SELECT`) |
| `S` | Công cụ **Vẽ Hình Khối** (`SHAPE`) |
| `T` | Công cụ **Thêm Hộp Văn Bản** (`TEXT`) |
| `N` | Công cụ **Thêm Giấy Ghi Chú** (`STICKY`) |
| `L` | Công cụ **Khoanh Vùng Hỏi AI** (`LASSO`) |
| `Giữ Chuột Phải + Kéo` | Mở **Bảng Chọn Tròn 8 Hướng + Vòng Chọn Con** |
| `Space + Kéo Chuột Trái` hoặc `Kéo Chuột Giữa` | Di chuyển bảng vô tận (**Pan** 60 FPS) |
| `Cuộn Chuột (Wheel)` | Phóng to / Thu nhỏ đúng tâm con trỏ (`25% – 400%`) |
| `Shift + Cuộn Chuột` | Cuộn bảng theo phương ngang |
| `Ctrl + 0` | Đặt lại thu phóng về `100%` và căn giữa bảng |
| `Ctrl + Z` | Hoàn tác (`Undo`) |
| `Ctrl + Y` hoặc `Ctrl + Shift + Z` | Làm lại (`Redo`) |
| `Ctrl + D` | Nhân bản Object đang chọn |
| `Delete` | Xóa Object đang chọn |
| `Shift` / `Ctrl + Click` hoặc kéo khung Marquee | Chọn nhiều Object cùng lúc |
| `←` / `→` hoặc `PageUp` / `PageDown` | Lùi / Tiến trang Slide hoặc tài liệu |
| `F1` | Mở **Hướng Dẫn Sử Dụng, Phím Tắt & Gesture Sandbox** |
| `H` | Ẩn / Hiện thanh **Mini HUD** trạng thái cử chỉ |
| `Ctrl + Shift + D` | Bật / Tắt **Debug Telemetry HUD** (FPS, độ trễ, ngưỡng tay) |
| `Esc` | Hủy thao tác đang vẽ / Đóng Bảng chọn tròn / Thoát Spotlight Tour |

### Hệ thống Object Canvas Độc Lập (`cv_engine/canvas_scene.py`)

- Mỗi **Nét vẽ tự do**, **Hình khối vector**, **Hộp văn bản**, **Giấy ghi chú (Sticky Note)** và **Hình ảnh chèn lên bảng** được quản lý thành một đối tượng độc lập trong Scene Graph theo từng trang.
- Hỗ trợ chọn trực tiếp hoặc kéo khung **Marquee Select**, kéo **8 điểm neo** để co giãn, kéo **tay cầm xoay phía trên** để xoay góc tự do, **đổi màu nhanh** nhóm đối tượng đang chọn, **nhân bản (`Ctrl+D`)**, **nhấn đúp** để chỉnh sửa văn bản/ghi chú, và lưu lịch sử **Undo/Redo** trọn vẹn.
- Toàn bộ thao tác `hit_test`, `marquee_select` và lọc đối tượng trong khung nhìn (*viewport culling*) được tăng tốc qua **`RustSpatialIndex` (Quadtree)** giúp giữ mượt 60 FPS trên các trang bài giảng dày đặc nội dung.

### Bảng Chọn Tròn Chuột Phải (8 Hướng Chính + Vòng Chọn Con Ngoài)

Nhấn giữ **Chuột phải** tại bất kỳ vị trí nào trên bảng để mở vòng tròn glassmorphic ngay dưới con trỏ. Kéo chuột qua vòng chết tâm (`> 22px`) về hướng mong muốn rồi thả để kích hoạt; thả trong vòng tâm (`< 22px`) để **Hủy** an toàn. Với 3 nhóm có ký hiệu `▸`, kéo xa hơn ra **vòng ngoài (`> 144px`)** để chọn nhanh mục con:

| Hướng | Công cụ chính | Các mục Vòng Chọn Con Ngoài (`> 144px`) |
|---|---|---|
| **N (Bắc — 12h)** | ✎ **Bút Vẽ (`PEN`)** `▸` | Mực Cyan (`#00F5D4`) · Mực Vàng (`#FEE440`) · Mực Đỏ (`#F43F5E`) · Bút Dạ Quang (`HIGHLIGHTER`) · Bật/Tắt Nắn Hình Thông Minh |
| **NE (Đông Bắc)** | ⌫ **Tẩy Nét (`ERASER`)** | — |
| **E (Đông — 3h)** | ⬉ **Chọn & Kéo (`SELECT`)** | — |
| **SE (Đông Nam)** | ✥ **Kéo Bảng (`PAN`)** | — |
| **S (Nam — 6h)** | **T Thêm Chữ (`TEXT`)** | — |
| **SW (Tây Nam)** | ◇ **Hình Khối (`SHAPES`)** `▸` | Hình Chữ Nhật · Hình Tròn · Hình Tam Giác · Mũi Tên Vector · Giấy Ghi Chú (`STICKY`) |
| **W (Tây — 9h)** | ↶ **Hoàn Tác (`UNDO`)** | — |
| **NW (Tây Bắc)** | ✦ **Cử Chỉ / AI** *(mở Sơ Đồ Cử Chỉ an toàn, không xóa mực)* `▸` | Trang Mới (`NEW_PAGE`) · Xóa Bảng (`CLEAR_BOARD`) · Khoanh Vùng Hỏi AI (`ASK_AI_REGION`) · **Sơ Đồ Cử Chỉ (`OPEN_GESTURE_MAP`)** · **Mở AI Chat (`OPEN_AI_CHAT`)** |

---

## 5. Trình chiếu đa định dạng & Đính kèm file UTF-8 vào AI Chatbot

### Trình chiếu PDF, PPTX, DOCX & Ảnh (`cv_engine/document_presenter.py`)

- **Định dạng hỗ trợ:** `.pdf`, `.pptx`, `.ppt`, `.ppsx`, `.potx`, `.docx`, `.doc`, ảnh (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.svg`) và **Thư mục Ảnh** (tự sắp xếp thành bộ Slide).
- **Pipeline chuyển đổi PPTX/DOCX 3 tầng có Cache SHA-256** (lưu tại `%APPDATA%\AirTeachAI\doc_cache\`, mở lại `< 0.2 giây`):
  1. **Tầng 1:** Headless **LibreOffice** (`soffice --headless --convert-to pdf`) cho độ trung thực bố cục cao nhất.
  2. **Tầng 2:** **Microsoft Office COM Automation** (PowerPoint / Word) trên máy Windows đã cài MS Office.
  3. **Tầng 3 (Tích hợp sẵn 100% Offline):** Bộ render thuần Python (`python-docx`, `python-pptx`, `Pillow`, `PyMuPDF` + font tiếng Việt `NotoSans`) — đảm bảo mở được bài giảng Word/PowerPoint trên mọi máy ngay cả khi **không cài Office lẫn LibreOffice**.
- **Hiển thị & Xuất bài giảng:** Render PDF sắc nét đến `400%` bằng PyMuPDF, cache LRU kết hợp pre-render trang kề, chế độ **Cuộn dọc liên tục**, lưu lớp Object/chú thích độc lập cho từng trang, và **Xuất trọn vẹn bài giảng kèm chú thích** ra tệp **PDF** hoặc bộ ảnh **PNG**.

### Đính kèm tệp tiếng Việt UTF-8 vào AI Chatbot (`cv_engine/file_attachment_parser.py`)

- Hỗ trợ đính kèm trực tiếp tài liệu (`.pdf`, `.docx`, `.pptx`), tệp văn bản/dữ liệu (`.txt`, `.md`, `.csv`, `.json`) và hình ảnh (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.svg`) — **tương thích đầy đủ đường dẫn & tên tệp tiếng Việt có dấu trên Windows**.
- **Xử lý mã hóa & Unicode chuẩn xác:**
  - Tự động thử giải mã theo thứ tự `utf-8-sig` → `utf-8` → `utf-16` → `cp1258` → `latin-1`.
  - Chuẩn hóa toàn bộ văn bản sang **Unicode NFC**, loại bỏ BOM/null byte/ký tự điều khiển lỗi để tiếng Việt và ký hiệu toán học không bao giờ bị vỡ dấu (*mojibake*).
  - Tự ánh xạ MIME chuẩn (không lệ thuộc registry Windows lỗi) và tạo ảnh xem trước trang đầu (PNG `<= 1024px`) gửi kèm cho mô hình AI đa phương thức (Vision).
- **Hiển thị phản hồi chuyên sâu (`cv_engine/output_sanitizer.py`):** Streaming thời gian thực tới `8192` token, hiển thị rõ **khối mã (`<pre><code>`)**, **bảng Markdown có viền (`<table>`)**, **công thức toán LaTeX** ở cả Dark/Light mode, kèm nút **📋 Sao chép** và **📌 Đưa nội dung lên bảng**.

---

## 6. AI Cloud (Full Model Gemini) & AI Local Offline (GGUF Model Hub)

### AI Cloud — Full Model Catalog (Google Gemini)

Cấu hình trực tiếp trong ngăn Trợ lý AI (nút **API Key**, hỗ trợ nhập 1 hoặc nhiều `GEMINI_API_KEY` cách nhau bằng dấu phẩy để tự xoay vòng khi hết quota) hoặc sao chép `.env.example` thành `.env`:

| Mã Model | Nhãn hiển thị | Phù hợp cho |
|---|---|---|
| *(Rỗng — Mặc định)* | **Auto (Chuỗi Fallback tối ưu)** | Tự động chuyển đổi model dự phòng khi gặp giới hạn băng thông/quota |
| `gemini-2.5-flash` | **Gemini 2.5 Flash (Nhanh & Mạnh)** | Giảng dạy trực tiếp, giải đáp đa phương thức tốc độ cao |
| `gemini-2.5-pro` | **Gemini 2.5 Pro (Suy luận chuyên sâu)** | Bài toán khó, lập luận nhiều bước, phân tích học thuật |
| `gemini-2.5-flash-lite` | **Gemini 2.5 Flash-Lite (Siêu nhẹ)** | Phản hồi tức thì, tiết kiệm quota |
| `gemini-2.0-flash` | **Gemini 2.0 Flash (Ổn định)** | Dự phòng ổn định cho lớp học đông |
| `gemini-3.8-flash` | **Gemini 3.8 Flash (Preview)** | Trải nghiệm năng lực suy luận thế hệ mới |
| `gemini-3.5-flash` | **Gemini 3.5 Flash** | Tốc độ cao và tối ưu ngữ cảnh dài |

### AI Local Offline & Kho Model GGUF (`cv_engine/gguf_model_manager.py`)

Mở nút **🤖 Quản Lý AI Local (Model Hub)** để chuyển 1-click giữa 3 chế độ định tuyến: **Hybrid (Ưu tiên Cloud, tự dự phòng Local)** · **Cloud Only** · **Local Offline 100%**.

- **Tải Model 1-click có Resume & SHA-256:** Hỗ trợ tạm dừng/tải tiếp bằng HTTP Range (`206 Partial Content`), hiển thị tốc độ MB/s theo thời gian thực, kiểm tra header `GGUF` và xác thực mã băm **SHA-256**, lưu tại `%APPDATA%\AirTeachAI\models\`.
- **Hỗ trợ dán URL tùy chỉnh:** Dán bất kỳ đường dẫn `.gguf` nào từ HuggingFace (`/blob/` tự quy đổi thành `/resolve/`) hoặc GitHub Releases (`/blob/` tự quy đổi thành `/raw/`).
- **Danh mục GGUF tuyển chọn sẵn:**

| Model GGUF | Dung lượng | RAM đề xuất | Điểm mạnh |
|---|---|---|---|
| **Qwen 2.5 1.5B Instruct (Q4_K_M)** *(Khuyên dùng)* | `1.04 GB` | `>= 4 GB` | Cân bằng tốc độ & chất lượng tiếng Việt, Toán & Khoa học trên CPU phổ thông |
| **Qwen 2.5 3B Instruct (Q4_K_M)** *(Khuyên dùng)* | `1.96 GB` | `>= 6 GB` | Tiếng Việt rất tốt, lập luận sư phạm sâu, bảng biểu & LaTeX chuẩn |
| **Qwen 2.5 0.5B Instruct (Q4_K_M)** | `469 MB` | `>= 2 GB` | Siêu nhẹ cho máy cấu hình thấp hoặc demo offline nhanh |
| **Meta Llama 3.2 1B Instruct (Q4_K_M)** *(Khuyên dùng)* | `770 MB` | `>= 3 GB` | Khởi chạy cực nhanh trên laptop CPU văn phòng (AVX2) |
| **Meta Llama 3.2 3B Instruct (Q4_K_M)** | `1.88 GB` | `>= 6 GB` | Trả lời mạch lạc, tốc độ ổn định |
| **Google Gemma 2 2B IT (Q4_K_M)** | `1.59 GB` | `>= 6 GB` | Trình bày khái niệm khoa học STEM rõ ràng |
| **Microsoft Phi-3.5 Mini Instruct (Q4_K_M)** | `2.23 GB` | `>= 8 GB` | Mạnh vượt trội về Toán, Vật lý, Code & Logic đa bước |

- **Runtime tự phát hiện:** `llama-cpp-python` (chạy trực tiếp trong ứng dụng, tự thử GPU offload rồi fallback CPU AVX2), **Ollama** (`127.0.0.1:11434`), **LM Studio** (`127.0.0.1:1234`), và **`llama-server`** (`127.0.0.1:8080`).

---

## 7. Interactive Spotlight Tour & Phòng Thử Cử Chỉ (Gesture Sandbox)

### Interactive Spotlight Tour (`cv_engine/spotlight_tour.py`)

- Khi khởi chạy lần đầu (hoặc nhấn nút **🔦 Tour Spotlight** trong cửa sổ Hướng Dẫn `?` / `F1`), lớp phủ làm tối toàn bộ cửa sổ và khoét vùng sáng có viền Neon Cyan nổi bật lần lượt **6 bước**:
  1. **Bảng chọn tròn chuột phải 8 hướng + vòng chọn con**
  2. **Di chuyển bảng (Pan) & Zoom-to-Cursor**
  3. **Hệ thống Object Canvas độc lập** (Chọn, Di chuyển, Co giãn 8 góc, Xoay, Đổi màu, Xóa)
  4. **Cử chỉ 1 Tay & 2 Tay không chạm** *(kèm nút nhảy nhanh `🧪 Thử Cử Chỉ`)*
  5. **Mở bài giảng đa định dạng & ghi chú từng trang**
  6. **Trợ lý AI Cloud Gemini Full Model & Local AI GGUF Offline**
- Thẻ hướng dẫn tự căn vị trí thông minh không tràn màn hình (`1366×768` đến `4K`), hỗ trợ chọn *"Không hiện lại lần sau"* và phím `Esc` để đóng tức thì.

### Sơ Đồ Cử Chỉ Trực Quan 2 Cột & Phòng Thử Cử Chỉ (`cv_engine/user_guide_dialog.py`)

- Nhấn nút **`Sơ đồ Cử chỉ`** ngay trên thanh tiêu đề, trên Mini Camera HUD góc phải, trong bảng Camera hoặc từ lát cắt **Cử Chỉ / AI** của Bảng Chọn Tròn Chuột Phải → mở cửa sổ trực quan chia **2 cột (Cử Chỉ 1 Tay | Cử Chỉ 2 Tay)** có hình minh họa vector bàn tay, ngưỡng kích hoạt chi tiết và **thanh Test thử cử chỉ trực tiếp** qua Webcam.
- Nhấn nút **`? Hướng dẫn`** trên thanh trên để chạy **Interactive Spotlight Tour (6 bước)**, hoặc nhấn phím **`F1`** để mở sổ tay đầy đủ 5 tab kèm **Phòng Thử Cử Chỉ (Gesture Sandbox)** phản hồi Xanh (`✓ ĐÃ NHẬN DIỆN`) / Đỏ theo thời gian thực.

---

## 8. Cài đặt & Đóng gói `.exe` 1-Click trên Windows 10/11

### Tổng quan bộ Script tự động

| Script | Mục đích sử dụng | File Log đầu ra |
|---|---|---|
| `setup_all.bat` | **Bước 1 — Cài đặt toàn bộ môi trường từ máy trắng:** VC++ 2015–2022 x64, Python 3.11/3.12 x64, Rust/Cargo (`rustup`), `.venv`, `requirements.txt` (retry 3 lần), tải/xác thực MediaPipe models + NotoSans fonts + icons + locales, kiểm tra LibreOffice & MS Office | `setup_all.log` |
| `build_exe.bat` | **Bước 2 — Biên dịch Rust & Đóng gói `.exe` độc lập:** Kích hoạt `.venv` (tự gọi `setup_all.bat` nếu chưa có), chạy `maturin develop --release` (`lto = true`, `opt-level = 3`, `strip = true`), đóng gói PyInstaller (`AirTeachAI.spec`), cập nhật launcher gốc `AirTeach-AI.exe`, copy launcher vào `dist\AirTeachAI\` và chạy `--smoke-test` | `build_exe.log` |
| `AirTeach-AI.exe` | **Launcher Native gốc siêu nhẹ (~150 KB):** Tự động khởi chạy `dist\AirTeachAI\AirTeachAI.exe` (hoặc `.venv\Scripts\pythonw.exe main.py`) khi nháy đúp trực tiếp tại thư mục gốc | — |
| `auto_setup_and_build.bat` | **Gộp 1-Click trọn gói:** Chạy liên hoàn cả cài môi trường + biên dịch Rust + đóng gói `.exe` + smoke test trong 1 lần duy nhất | `setup_build.log` |
| `run_airteach.bat` | **Khởi chạy nhanh:** Ưu tiên mở `AirTeachAI.exe` độc lập; nếu đang ở chế độ phát triển sẽ tự chạy `.venv\Scripts\python.exe main.py` | — |
| `Chay_AirTeachAI_AutoFix.bat` | **Launcher bảo vệ Máy Thi / USB:** Tự cài VC++ x64 nếu thiếu, tạo `%APPDATA%\AirTeachAI\`, tự mở quyền Camera Windows, chạy `AirTeachAI.exe` và tự xuất `crash_log.txt` tiếng Việt kèm 60 dòng log cuối nếu có sự cố | `crash_log.txt` |

### Quy trình khuyến nghị (2 bước tách biệt)

```bat
:: Bước 1: Cài đặt toàn bộ môi trường (VC++, Python, Rust, .venv, assets)
setup_all.bat

:: Bước 2: Biên dịch lõi Rust & đóng gói dist\AirTeachAI\AirTeachAI.exe + smoke test
build_exe.bat
```

Hoặc chạy gộp 1 bước:

```bat
auto_setup_and_build.bat
```

### Triển khai sang Máy Thi / Máy Trình Chiếu

1. Copy toàn bộ thư mục **`dist\AirTeachAI\`** sang máy đích hoặc USB.
2. Nháy đúp **`Chay_AirTeachAI_AutoFix.bat`** (hoặc `run_airteach.bat`).

### Chạy tự động trong CI / Script không dừng ở `pause`

Tất cả các `.bat` đều hỗ trợ biến môi trường `AIRTEACH_NO_PAUSE=1` (hoặc `CI=1`):

```powershell
$env:AIRTEACH_NO_PAUSE = "1"
cmd /c setup_all.bat
cmd /c build_exe.bat
```

---

## 9. Thư mục cấu hình an toàn (`%APPDATA%\AirTeachAI\`) & Xử lý sự cố

### Cấu trúc thư mục dữ liệu người dùng

AirTeach AI **không ghi dữ liệu cấu hình vào `Program Files` hay thư mục cài đặt chỉ đọc**, mà lưu toàn bộ tại `%APPDATA%\AirTeachAI\` (tương đương `C:\Users\<TenNguoiDung>\AppData\Roaming\AirTeachAI\`):

```text
%APPDATA%\AirTeachAI\
├── config.json              # Gemini API Key, model đã chọn, chế độ AI, tùy chọn Spotlight Tour
├── vision_config.json       # Cấu hình camera, độ nhạy cử chỉ, ngưỡng chống ngược sáng
├── local_ai_cache.sqlite    # Cache phản hồi AI cục bộ
├── logs\
│   └── app.log              # Nhật ký runtime xoay vòng (2 MB x 3 bản)
├── doc_cache\               # PDF đã chuyển đổi từ PPTX/DOCX (định danh theo SHA-256)
├── models\                  # Các tệp model Local AI (.gguf & .sha256) tải từ Model Hub
├── face_embeddings\         # Vector nhận diện khuôn mặt cục bộ
└── recordings\              # Bản ghi âm bài giảng
```

### Xử lý sự cố thường gặp

| Triệu chứng / Mã lỗi | Nguyên nhân phổ biến | Cách khắc phục nhanh |
|---|---|---|
| Lỗi `0xC0000135` hoặc báo thiếu `vcruntime140_1.dll` | Máy Windows thiếu Microsoft Visual C++ 2015–2022 Redistributable (x64) | Chạy **`Chay_AirTeachAI_AutoFix.bat`** hoặc **`setup_all.bat`** để tự tải và cài đặt im lặng |
| Không mở được Webcam / Màn hình camera đen | Windows Privacy tắt quyền Camera hoặc ứng dụng khác (Zoom/Teams/OBS) đang giữ camera | Đóng app đang chiếm camera; chạy `Chay_AirTeachAI_AutoFix.bat` hoặc bật **Settings → Privacy & security → Camera → Let desktop apps access your camera** |
| Cử chỉ tay khó nhận hoặc bị giật | Đứng quá xa (`> 3.2m`), ngược sáng mạnh hoặc lòng bàn tay quay nghiêng | Mở **Tab 5 (`F1`) Gesture Sandbox** để căn góc lòng bàn tay hướng vào camera; giữ khoảng cách tối ưu `0.6m – 2.5m` |
| Không cài được Rust / `maturin` báo lỗi biên dịch | Thiếu mạng khi tải `rustup` hoặc thiếu MSVC Build Tools | Xem chi tiết trong `setup_all.log` / `build_exe.log`; ứng dụng vẫn chạy đầy đủ bằng **Python Fallback** |
| Máy thi không có Microsoft Office hay LibreOffice | Không có phần mềm văn phòng cài sẵn | Không cần cài thêm — AirTeach AI tự động dùng **bộ render DOCX/PPTX thuần Python tích hợp sẵn** |
| AI Cloud báo hết quota hoặc mất Internet | Giới hạn lượt gọi API hoặc phòng thi ngắt mạng | Nhập thêm API Key dự phòng (cách nhau dấu phẩy) hoặc mở **🤖 Model Hub** chuyển sang chế độ **Local Offline (GGUF)** |
| Ứng dụng đóng đột ngột trên máy thi | Lỗi driver GPU/OpenGL hoặc môi trường Windows | Kiểm tra **`crash_log.txt`** (cùng thư mục `.exe`) và **`%APPDATA%\AirTeachAI\logs\app.log`** để xem nguyên nhân chi tiết bằng tiếng Việt |

---

**AirTeach AI — DNT - Victer** · *Smartboard Không Chạm, Hiệu Năng Rust, Trình Chiếu Đa Định Dạng & AI Hybrid Sư Phạm.*
