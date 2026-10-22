# BÁO CÁO KẾT QUẢ KIỂM THỬ HỒI QUY (TEST RESULTS)

**Dự án:** AirTeach AI  
**Lần chạy gần nhất:** 20/10/2026 21:43:28  
**Lệnh thực thi:** `py -3.15 -m pytest -q`  
**Môi trường:** Python 3.15.0rc1, PySide6 6.12.0, pytest 9.1.1, Windows x86_64  
**Kết quả tổng quan:** **231 / 231 PASSED (100% Pass Rate)** trong 39.12s.

---

## 1. BẢNG CHI TIẾT THEO TỪNG TẬP KIỂM THỬ

| Tệp kiểm thử | Số lượng bài test | Kết quả | Thời gian chạy | Các chức năng được xác thực |
|---|---|---|---|---|
| `tests/test_core_algorithms.py` | 47 | **47 PASSED** | $\approx 4.2\text{ s}$ | 1€ Filter, RDP simplification, Cubic Bézier spline, Shape Snapper (Line, Arrow, Circle, Ellipse, Rect, Tri), FSM 6 cử chỉ, Perspective warping |
| `tests/test_phase3_advanced_ui.py` | 28 | **28 PASSED** | $\approx 3.8\text{ s}$ | 4 mẫu nền ô li / tọa độ, NewLessonDialog, ReplayController (play/pause/scrub), SidePanelsWidget, LoginWindow PIN keypad |
| `tests/test_services_and_storage.py` | 69 | **69 PASSED** | $\approx 18.5\text{ s}$ | Fernet key generation & persistence, SQLite teachers table schema, YuNet+SFace authentication, PBKDF2 PIN hashing, DocumentManager PDF import/export, .airteach ZIP bundle roundtrip, GeometricMathParser |
| `tests/test_ui_components.py` | 73 | **73 PASSED** | $\approx 12.4\text{ s}$ | LessonBoard layers & items, CameraWidget, ChatPanel offline/online toggle, SettingsDialog parameters, FaceAuthDialog modes, MainWindow smoke & multi-page stroke segregation |
| `tests/test_svg_assets_and_trainer.py` | 14 | **14 PASSED** | $\approx 4.4\text{ s}$ | 33 SVG vector assets (icons, gestures, diagrams), GestureTrainerDialog HUD/skeleton/calibration, SmartInkReviewDialog LaTeX rendering & edit, MainWindow dual toolbars, teaching modes (PEN/LASER/ERASER/SELECT), palette & Undo/Redo stack |
| **TỔNG CỘNG** | **231** | **231 PASSED** | **39.12s** | **Tỷ lệ thành công: 100% (0 hồi quy)** |

---

## 2. KẾT QUẢ KIỂM TRA THỊ GIÁC (VISUAL INSPECTION RESULTS)

- **Cửa sổ Đăng nhập:** Đã chụp và kiểm định `screen_login_win.png`. Giao diện chuẩn xác, phông Segoe UI hiển thị dấu tiếng Việt hoàn hảo.
- **Cửa sổ Giảng dạy chính (v2):** Đã chụp và kiểm định `screen_main_v2_win.png`. Tích hợp đầy đủ Dual Toolbars chuẩn SVG, thanh công cụ giảng dạy, bảng màu nhanh, bộ chọn độ dày nét và con trỏ Laser.
- **Hộp thoại Huấn luyện Cử chỉ:** Đã chụp và kiểm định `screen_gesture_trainer_win.png`. 7 thẻ minh họa SVG cử chỉ, luồng skeleton 21 điểm mốc thời gian thực, bảng HUD đo lường và nút hiệu chuẩn.
- **Hộp thoại Kiểm duyệt Smart Ink:** Đã chụp và kiểm định `screen_smart_ink_review_win.png`. So sánh nét viết tay gốc vs công thức vector MathText, thanh đo độ tin cậy và bộ chỉnh sửa LaTeX.
- **Thư viện bài giảng:** Đã chụp và kiểm định `screen_prep_win.png`. Nút tạo bài giảng và nhập tài liệu hoạt động mượt mà.
- **Hộp thoại Cài đặt:** Đã chụp và kiểm định `screen_settings_win.png`. Đầy đủ 3 nhóm cấu hình API, Camera, và Nét vẽ.
