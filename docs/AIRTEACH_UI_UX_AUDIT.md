# AIRTEACH AI — ĐÁNH GIÁ THIẾT KẾ GIAO DIỆN & TRẢI NGHIỆM NGƯỜI DÙNG (UI/UX AUDIT)
**Phiên bản:** AirTeach AI Workstation v2.5.0  
**Ngày phát hành:** 23/10/2026  
**Chuẩn thiết kế:** Modern Desktop Classroom Workstation (PySide6 / Qt 6)

---

## 1. TỔNG QUAN ĐỔI MỚI GIAO DIỆN

Trước đây, giao diện của hệ thống tập trung chủ yếu vào một bảng vẽ đơn giản kèm khung camera nhỏ. Trong đợt nâng cấp toàn diện này, AirTeach AI đã được tái thiết kế thành một **Trạm làm việc Giảng dạy Chuyên nghiệp (Smart Teaching Workstation)**, cân bằng giữa không gian viết bảng tối đa và các công cụ hỗ trợ thông minh xung quanh.

```
┌────┬───────────────────────────────────────────────────────────────────┬───────────────┐
│ ☰  │ 📂 Mở | 💾 Lưu | 📄 Xuất PDF | ◀ ▶ Trang | 📹 30 FPS • Normal | ✨ AI │ 💬 AI Chat   │
│ A  ├───────────────────────────────────────────────────────────────────┤ ✨ AI Work-  │
│ I  │ ✏️ Bút (P) | 🔴 Laser | 🧹 Tẩy | 🔲 Chọn | ☝️ Chỉ tay | 🎨 Màu | 📏 Nét│    bench      │
│ R  ├───────────────────────────────────────────────────────────────────┤               │
│ T  │                                                                   │ - Soạn đề     │
│ E  │                                                                   │ - Tóm tắt PDF │
│ A  │                     BẢNG DẠY HỌC TƯƠNG TÁC                        │ - Giáo án     │
│ C  │                  (Layered QGraphicsView Scene)                    │ - Dàn ý       │
│ H  │                                                                   ├───────────────┤
│    │               [Nét vẽ vector + Nắn hình học + OCR]                │ 📷 WEBCAM     │
│ S  │                                                                   │   PREVIEW     │
│ I  │                                                                   │ + Skeleton    │
│ D  │                                                                   │ + Mode HUD    │
│ E  ├───────────────────────────────────────────────────────────────────┴───────────────┤
│ B  │ Mode: DRAW (Air Pen)  |  Trang: 1/1  |  Webcam: DirectShow (30.0 FPS) | Lưu tự động │
│ AR │                                                                                   │
└────┴───────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. CHI TIẾT CÁC THÀNH PHẦN GIAO DIỆN ĐƯỢC NÂNG CẤP

### 2.1. Thanh Điều Hướng Bên Trái Thu Gọn Đa Năng (Collapsible Navigation Sidebar)
- **Mục tiêu UX:** Cho phép giáo viên truy cập nhanh vào mọi không gian chức năng nhưng không chiếm dụng diện tích bài giảng.
- **Hành vi động:**
  - **Trạng thái Mở rộng (Expanded - 180px):** Hiển thị logo, tiêu đề "AirTeach AI" và các mục có đầy đủ biểu tượng và nhãn tiếng Việt rõ ràng:
    - 🎨 *Bảng tương tác* (Màn hình chính)
    - 📂 *Kho bài giảng* (Thư viện bài học `.airteach`)
    - 📑 *Trình chiếu* (Bật/tắt chế độ toàn màn hình Fullscreen F11)
    - ✨ *AI Workbench* (Mở nhanh tab soạn thảo giáo án)
    - 🔬 *Chẩn đoán Camera* (Cửa sổ đo lường thông số webcam)
    - 🎯 *Luyện cử chỉ* (Màn hình hiệu chuẩn góc camera và 6 cử chỉ)
    - ⚙️ *Cài đặt AI & Máy* (Hộp thoại cấu hình tài khoản)
    - 🚪 *Đăng xuất* (Chuyển đổi giáo viên an toàn)
  - **Trạng thái Thu gọn (Collapsed - 52px):** Nút hamburger `☰` thu nhỏ thanh sang dải icon tối giản có tooltip giải thích khi rê chuột, giải phóng thêm 128px chiều rộng cho bảng viết.

### 2.2. Thanh Công Cụ Toàn Cục & Giảng Dạy (Dual Header Toolbars)
- **Thanh 1 — Quản lý Bài giảng (Navigation & File Toolbar):**
  - Chức năng lưu, mở, xuất PDF và chuyển trang trực quan.
  - **Huy hiệu Chẩn đoán Camera (`self._cam_diag_btn`):** Hiển thị liên tục FPS thực tế và điều kiện ánh sáng (ví dụ: `📹 30 FPS • Normal` hoặc `📹 28 FPS • Low Light`). Khi giáo viên nhấn vào, cửa sổ chẩn đoán chi tiết lập tức mở ra.
  - **Nút Kích hoạt AI Workbench (`self._ai_workbench_btn`):** Nổi bật với màu xanh Google Blue nhẹ nhàng (`#e8f0fe`), đưa giáo viên vào trạm soạn thảo chỉ với một cú nhấp chuột.
  - **Hồ sơ giáo viên:** Huy hiệu thông tin hiển thị rõ mã số, họ tên và môn học của giáo viên đang giảng dạy.
- **Thanh 2 — Công cụ Giảng dạy Bảng viết (Canvas Teaching Toolbar):**
  - Nhóm nút chọn công cụ độc quyền: Bút viết (P), Tia Laser (L), Tẩy nét (E), Chọn vùng (S).
  - **Nút Chuyển đổi Chế độ Vẽ Đôi (`self._draw_mode_btn`):** Cho phép giáo viên chuyển đổi trực tiếp giữa:
    - `☝️ Chỉ tay (Air Pen)`: Vẽ bằng ngón trỏ tự nhiên.
    - `👌 Véo ngón (Pinch)`: Véo ngón trỏ và ngón cái để vẽ chính xác.
  - Lựa chọn nhanh 5 màu mực sư phạm chuẩn (Đen, Xanh dương, Đỏ sửa bài, Xanh lục, Tím), 4 cấp độ nét bút (Mảnh 1.5px, Vừa 2.5px, Đậm 4.5px, Dạ quang 8.0px), và các mẫu phông nền (Trắng, Ô li học sinh, Dòng kẻ ngang, Hệ trục tọa độ Oxy).

### 2.3. Bảng Chẩn Đoán & Đo Lường Camera Chuyên Sâu (`CameraDiagnosticsDialog`)
- Được thiết kế dưới dạng popup tiện ích với bảng điều khiển trực quan:
  - **Bảng chỉ số đo lường thực:** Tốc độ khung hình (FPS), Độ phân giải (Resolution), Độ sáng trung bình (Luminance), Độ tương phản (Contrast), Tỷ lệ cháy sáng/tối (Clipping %), và Điểm số sắc nét (Focus/Sharpness).
  - **Khung xem trước trực tiếp (Live Camera Feed):** Hiển thị hình ảnh webcam thời gian thực kèm khung xương 21 điểm mốc bàn tay màu cyan/cam và tâm ngón trỏ màu đỏ khi đang viết.
  - **Bộ chuyển đổi chế độ tiền xử lý ánh sáng:** Tùy chọn nhanh `Adaptive Auto`, `Low Light Enhance`, `Backlight Boost`, `Glare Reduce`, `Off (Raw)` có tác dụng ngay lập tức mà không cần khởi động lại ứng dụng.
  - **Nút Hiệu chuẩn góc nhìn:** Khởi chạy thuật toán Perspective Calibration căn chỉnh 4 góc webcam tương ứng 4 góc bảng.

### 2.4. Trạm Soạn Thảo Giáo Án & Trợ Giảng AI (`AIWorkbenchWidget`)
- Tích hợp liền mạch thành Tab thứ hai trong Dock bên phải:
  - **Thực đơn tác vụ sư phạm:**
    - 📋 *Dàn ý bài giảng*: Cấu trúc bài học theo mục tiêu năng lực.
    - 📑 *Tóm tắt tài liệu*: Trích xuất luận điểm chính từ tài liệu đính kèm.
    - ❓ *Bộ câu hỏi trắc nghiệm*: Soạn 4 câu hỏi 4 mức độ nhận thức (Nhận biết, Thông hiểu, Vận dụng, Vận dụng cao) có đáp án và giải thích chi tiết.
    - 💡 *Giải thích khái niệm*: Đơn giản hóa kiến thức cho học sinh bằng ngôn ngữ trực quan và ví dụ đời sống.
    - ⏱️ *Kế hoạch bài dạy 45 phút*: Phân bổ tiến trình giảng dạy khoa học.
  - **Hiển thị luồng văn bản thời gian thực (Streaming):** Giáo viên nhìn thấy kết quả được sinh ra từng đoạn mà không phải đợi toàn bộ bài viết hoàn thành.
  - **Thao tác nhanh:** Nút "📋 Sao chép" đưa vào bộ nhớ tạm và nút "📌 Chèn vào Bảng viết" đưa thẳng văn bản vào trang bảng hiện tại với cỡ chữ và vị trí căn giữa tối ưu.

### 2.5. Hộp Thoại Cài Đặt Toàn Diện Đa Tài Khoản (`SettingsDialog`)
- Tổ chức theo 4 Tab chuyên biệt với bố cục gọn gàng, hỗ trợ phím tắt Enter/Esc:
  1. **🤖 AI & API Settings:**
     - Lựa chọn mô hình (`gemini-2.5-flash`, `gemini-1.5-pro`, ...).
     - Khóa API cá nhân hiển thị dạng mật khẩu kèm nút `👁 Hiện/Ẩn` và `🗑 Xóa key`.
     - Nút `🔍 Kiểm tra kết nối` chạy kiểm tra bất đồng bộ, hiển thị nhãn trạng thái xanh lục kèm biểu tượng tích khi thành công, hoặc màu đỏ kèm giải thích chi tiết khi thất bại.
     - Các nút bật/tắt tính năng sư phạm: Tóm tắt bài học, Gợi ý giảng dạy, Phân tích nét viết Smart Ink.
  2. **✋ Cử chỉ & Bút vẽ:**
     - Chế độ vẽ ưu tiên (Chỉ tay tự nhiên vs Véo ngón tay).
     - Ngưỡng khoảng cách véo ngón (Pinch Threshold) có thanh trượt điều chỉnh từ 0.03 đến 0.12.
     - Tham số làm mịn nét 1€ Filter ($\beta$) và làm mượt RDP ($\epsilon$).
     - Nút bật/tắt tính năng Tự động nắn hình học (Shape Snapping).
  3. **📷 Camera & Thị giác:**
     - Bộ chọn chỉ số camera kết nối (Webcam tích hợp hoặc Camera ngoài qua cổng USB).
     - Chế độ tiền xử lý thích ứng ánh sáng.
     - Tùy chọn lật gương hình ảnh (Mirror Camera) và cân bằng sáng động CLAHE.
     - Phím tắt mở Wizard Hiệu chuẩn cử chỉ.
  4. **🔒 Quyền riêng tư & Bảo mật:**
     - Mô tả rõ ràng cơ chế bảo vệ của Windows Credential Manager.
     - Cam kết Zero Leak: Không gửi dữ liệu hình ảnh học sinh lên máy chủ đám mây, mọi vector đặc trưng và nhận diện bàn tay được xử lý cục bộ 100% trên CPU giáo viên.

---

## 3. TIÊU CHUẨN ĐỒ HỌA, MÀU SẮC & TYPOGRAPHY

- **Typography:**
  - Giao diện người dùng: Font hệ thống chuẩn `Segoe UI`, `SF Pro Display`, `Arial` với độ đậm (Font Weight) phân cấp rõ ràng (SemiBold cho tiêu đề, Regular cho văn bản).
  - Khối mã nguồn và HUD thị giác: `Consolas`, `Cascadia Code`, `Fira Code` giúp hiển thị số đo FPS và tọa độ chính xác.
  - Hỗ trợ 100% tiếng Việt có dấu đầy đủ (Unicode UTF-8) và các ký hiệu khoa học $\LaTeX$, $\vec{F}$, $\Omega$, $C_6H_6$.
- **Hệ thống bảng màu (Design Tokens):**
  - Màu nền chính: Dark Slate `#1e1e2e` cho thanh bên, Crisp Clean `#ffffff` cho bảng vẽ, và Cool Gray `#f8f9fa` cho thanh công cụ.
  - Màu nhấn (Accent): Google Blue `#1a73e8` và Sapphire `#89b4fa`.
  - Màu trạng thái: Thành công `#137333` (Green), Cảnh báo `#f9ab00` (Amber), Báo lỗi `#d93025` (Red).
