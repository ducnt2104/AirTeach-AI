# HỆ THỐNG THIẾT KẾ GIAO DIỆN AIRTEACH AI (DESIGN SYSTEM SPECIFICATION)

**Dự án:** AIRTECH AI — AI-Powered Air Gesture Teaching Platform  
**Phiên bản:** 1.0.0 — Windows Local-First Desktop Edition  
**Chuẩn thiết kế:** Modern Fluent / Clean Classroom Interactive UI  
**Thư viện áp dụng:** PySide6 (Qt Quick / QtWidgets), QSS (Qt Style Sheets), QGraphicsScene

---

## 1. NGUYÊN TẮC THIẾT KẾ (DESIGN PRINCIPLES)

1. **Hiển thị Rõ nét trong Lớp học (Classroom High Legibility):**
   Giao diện được tối ưu hóa đặc biệt cho điều kiện ánh sáng đa dạng của phòng học thực tế (từ ánh sáng tự nhiên mạnh đến phòng tối bật máy chiếu). Độ tương phản văn bản đạt chuẩn WCAG AA/AAA (tỷ lệ tương phản tối thiểu 4.5:1 cho văn bản thường và 7:1 cho tiêu đề).
2. **Tương tác Không Chạm Tự nhiên (Air-First Ergonomics):**
   Các nút công cụ và mục điều khiển có kích thước tối thiểu $44 \times 44\text{ px}$ nhằm hỗ trợ việc chọn bằng cử chỉ chỉ tay từ xa hoặc con trỏ ngón tay mà không bị nhầm lẫn.
3. **Đồ họa Vector Sắc nét (Resolution Independence):**
   100% biểu tượng được thiết kế dưới dạng vector SVG (`assets/icons/*.svg`), hỗ trợ thu phóng vô hạn không vỡ hạt trên màn hình độ phân giải cao 2K/4K và tỷ lệ High-DPI scaling từ 100% đến 250%.
4. **Phản hồi Trực quan Thời gian thực (Instant Visual Feedback):**
   Mọi trạng thái cử chỉ (Nhận diện bàn tay, Khóa nét vẽ, Chuyển cử chỉ, Đang lưu nét) đều có hiệu ứng vòng phản hồi (Feedback Ring) và thanh chỉ thị màu sắc thời gian thực.

---

## 2. BẢNG MÀU CHUẨN (COLOR PALETTE & DESIGN TOKENS)

### 2.1. Màu sắc Chủ đạo (Primary & Accent Tokens)

| Tên Token | Mã Hex | RGB | Ứng dụng trong Giao diện |
|---|---|---|---|
| `--color-primary` | `#1A73E8` | `rgb(26, 115, 232)` | Màu nhận diện thương hiệu, nút hành động chính, viền tiêu điểm |
| `--color-primary-hover` | `#1557B0` | `rgb(21, 87, 176)` | Trạng thái rê chuột hoặc kích hoạt nút chính |
| `--color-accent-teal` | `#00897B` | `rgb(0, 137, 123)` | Trạng thái cử chỉ sẵn sàng, chỉ báo công cụ thước đo hình học |
| `--color-accent-laser` | `#FF1744` | `rgb(255, 23, 68)` | Chấm laser ngón tay, chỉ báo ghi nét vẽ trực tiếp |
| `--color-success` | `#2E7D32` | `rgb(46, 125, 50)` | Xác thực khuôn mặt thành công, lưu bài giảng thành công |
| `--color-warning` | `#F57C00` | `rgb(245, 124, 0)` | Cảnh báo góc nghiêng khuôn mặt, cảnh báo mất dấu bàn tay |
| `--color-error` | `#D32F2F` | `rgb(211, 47, 47)` | Lỗi nhập mã PIN, lỗi đọc tập tin, camera bận |

### 2.2. Màu Nền & Phân lớp Bề mặt (Surface & Background Tokens)

| Tên Token | Mã Hex | Ứng dụng |
|---|---|---|
| `--surface-canvas-white` | `#FFFFFF` | Nền bảng viết trơn, trang giấy tài liệu |
| `--surface-canvas-grid` | `#F8F9FA` | Nền bảng có dòng kẻ hoặc ô li học sinh |
| `--surface-panel-bg` | `#F1F3F4` | Khung điều khiển, thanh công cụ bên cạnh |
| `--surface-dark-toolbar`| `#202124` | Thanh công cụ chế độ tương phản cao lớp học |
| `--border-subtle` | `#DADCE0` | Đường phân cách, viền ô nhập liệu |
| `--text-primary` | `#202124` | Tiêu đề chính, nội dung giảng dạy |
| `--text-secondary` | `#5F6368` | Nhãn phụ, thông số FPS, gợi ý thao tác |

---

## 3. QUY CHUẨN KIỂU CHỮ (TYPOGRAPHY SCALE)

Phần mềm sử dụng hệ thống phông chữ gốc của hệ điều hành Windows nhằm đảm bảo hiệu năng kết xuất tối đa và độ sắc nét qua engine ClearType:

- **Phông chữ mặc định:** `Segoe UI, Arial, 'Noto Sans', sans-serif`
- **Phông chữ toán học & công thức:** `Cambria Math, 'Times New Roman', STIX Two Math`
- **Phông chữ mã nguồn:** `Consolas, 'Cascadia Code', monospace`

| Cấp bậc (Hierarchy) | Cỡ chữ (Size) | Độ đậm (Weight) | Khoảng cách dòng (Line Height) |
|---|---|---|---|
| **Display Header (H1)** | 22 pt (29 px) | Bold (700) | 1.25 |
| **Section Header (H2)** | 16 pt (21 px) | SemiBold (600) | 1.30 |
| **Component Title (H3)**| 13 pt (17 px) | Medium (500) | 1.35 |
| **Body Regular** | 10 pt (13 px) | Regular (400) | 1.40 |
| **Caption / HUD Label** | 8.5 pt (11 px) | Medium (500) | 1.20 |

---

## 4. BỘ TẠO MẪU NỀN HỌC TẬP (BACKGROUND TEMPLATE GENERATOR)

Bảng giảng tương tác tích hợp 4 loại mẫu nền sư phạm chuyên dụng được tạo tự động bởi `BackgroundTemplateGenerator` (kích thước chuẩn $1920 \times 1080$):

1. **Trắng Sạch (Plain White):**
   - Nền màu trắng tinh `#FFFFFF`.
   - Phù hợp vẽ tự do, thuyết trình sơ đồ tư duy, vẽ phác thảo.
2. **Ô Li Học Sinh (Student Grid):**
   - Lưới ô li vuông kích thước $20 \times 20\text{ px}$.
   - Màu lưới xanh lam nhạt `#E3E8EE` với đường kẻ chính mỗi 5 ô màu `#C7D2DF`.
   - Phù hợp luyện chữ viết tay, bài giảng tiểu học, bài tập chính tả.
3. **Dòng Kẻ Ngang (Lined Notebook):**
   - Các đường kẻ ngang cách đều $32\text{ px}$.
   - Màu kẻ `#E0E0E0` với lề đỏ bên trái cách mép $80\text{ px}$ (`#FFCDD2`).
   - Phù hợp viết văn, dàn bài, bài giảng xã hội.
4. **Hệ Trục Tọa Độ Vuông Góc Oxy (Cartesian Oxy Axes):**
   - Trục hoành Ox và trục tung Oy đi qua tâm bảng $(960, 540)$ với màu `#1A73E8`, bề dày 2px kèm mũi tên chỉ hướng.
   - Lưới tọa độ phụ cách đều $40\text{ px}$ với màu xám nhạt `#EEF2F6`.
   - Phù hợp giảng dạy Toán học, Đồ thị hàm số, Vật lý và Hình học giải tích.

---

## 5. HỆ THỐNG BIỂU TƯỢNG VECTOR (SVG ICONOGRAPHY)

Toàn bộ các biểu tượng trong ứng dụng tuân thủ tiêu chuẩn:
- **Định dạng:** SVG 1.1 Vector chuẩn.
- **ViewBox:** `0 0 24 24`.
- **Stroke Width:** 2px đồng nhất (`stroke-linecap="round"`, `stroke-linejoin="round"`).
- **Bộ biểu tượng cốt lõi:**
  - `pen.svg`, `eraser.svg`, `shapes.svg`, `laser.svg` (Công cụ tương tác)
  - `undo.svg`, `redo.svg`, `clear.svg`, `export.svg` (Thao tác biên tập)
  - `ocr.svg`, `latex.svg`, `ai_assistant.svg` (Trí tuệ nhân tạo)
  - `play.svg`, `pause.svg`, `step.svg` (Phát lại bài giảng)
  - `grid.svg`, `camera.svg`, `user.svg`, `settings.svg` (Điều khiển hệ thống)
