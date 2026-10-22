# AIRTEACH AI — AGENT SKILLS USAGE MATRIX
**Dự án:** AirTeach AI — AI-Powered Air Gesture Teaching Platform  
**Phiên bản mục tiêu:** 1.0.0 — Windows Local-First Desktop Edition  
**Thời điểm cập nhật:** 21/10/2026

---

## 1. TỔNG QUAN VẬN DỤNG CÁC AGENT SKILLS

Bảng dưới đây ghi nhận chi tiết việc áp dụng các Agent Skills chuyên biệt vào quá trình tái thiết kế, sửa lỗi, nâng cấp đồ họa và nghiệm thu sản phẩm:

| STT | Tên Skill | Nhóm | Tác vụ cụ thể trong AirTeach AI | Bằng chứng và Thay đổi Mã nguồn | Trạng thái |
|---|---|---|---|---|---|
| 1 | `systematic-debugging` | Workflow & QA | Điều tra nguyên nhân gốc rễ lỗi đăng ký/đăng nhập, deadlock giải phóng camera giữa các màn hình | `docs/BASELINE_AUDIT.md`, sửa `_LoginCameraWorker`, `TeacherEnrollDialog` | ĐÃ ÁP DỤNG |
| 2 | `writing-plans` | Workflow & QA | Lập kế hoạch kiến trúc chi tiết từng bước có tiêu chí nghiệm thu | `docs/IMPLEMENTATION_STATUS.md`, `docs/ARCHITECTURE.md` | ĐÃ ÁP DỤNG |
| 3 | `test-driven-development` | Workflow & QA | Viết unit/integration tests cho luồng đăng ký, xác thực, chuyển màn hình | `tests/test_services_and_storage.py`, `tests/test_ui_components.py` | ĐÃ ÁP DỤNG |
| 4 | `accidental-data-loss-prevention` | An toàn hệ thống | Bảo vệ cơ sở dữ liệu `data/profiles.db`, không xóa đè workspace của giáo viên | Sử dụng giao dịch an toàn SQLite, mã hóa Fernet độc lập | ĐÃ ÁP DỤNG |
| 5 | `airteach-local-face-auth` | Chuyên biệt AirTeach | YuNet ONNX phát hiện + SFace 128D trích xuất vector, Fernet AES-128, PBKDF2 PIN | `src/services/face_profile_service.py` | ĐÃ ÁP DỤNG |
| 6 | `airteach-mediapipe-gestures` | Chuyên biệt AirTeach | MediaPipe HandLandmarker 21 3D landmarks, chuẩn hóa khoảng cách theo bàn tay, 6 cử chỉ FSM | `src/core/hand_gesture_engine.py` | ĐÃ ÁP DỤNG |
| 7 | `airteach-vector-ink` | Chuyên biệt AirTeach | 1€ Filter lọc rung thời gian thực, RDP giảm điểm, Catmull-Rom Bézier spline biến thiên độ dày | `src/core/stroke_beautifier.py` | ĐÃ ÁP DỤNG |
| 8 | `airteach-shape-snapping` | Chuyên biệt AirTeach | Phân tích bao lồi, circularity index, nắn hình tự động (Line, Arrow, Circle, Ellipse, Rect, Tri) | `src/core/stroke_beautifier.py` | ĐÃ ÁP DỤNG |
| 9 | `airteach-pyside6-ui` | Chuyên biệt AirTeach | QGraphicsScene kiến trúc 3 tầng Z (0: Nền, 10: Mực, 20: Laser/HUD), QueuedConnection an toàn luồng | `src/ui/lesson_board.py`, `src/ui/main_window.py` | ĐÃ ÁP DỤNG |
| 10 | `airteach-qt-performance` | Chuyên biệt AirTeach | Loại bỏ nghẽn luồng giao diện, tối ưu vòng lặp camera, ngắt luồng an toàn | `src/ui/login_window.py`, `src/ui/main_window.py` | ĐÃ ÁP DỤNG |
| 11 | `airteach-document-import` | Chuyên biệt AirTeach | pypdfium2 dựng PDF 144 DPI, COM / LibreOffice headless cho PPTX/DOCX, gói `.airteach` ZIP | `src/services/document_manager.py` | ĐÃ ÁP DỤNG |
| 12 | `airteach-offline-ocr` | Chuyên biệt AirTeach | Nhận diện chữ viết tay cục bộ tiếng Việt/tiếng Anh, crop vùng mực, trích xuất văn bản | `src/services/smart_ink_ocr.py` | ĐÃ ÁP DỤNG |
| 13 | `airteach-math-formula-recognition` | Chuyên biệt AirTeach | GeometricMathParser thuần Python + kết xuất vector công thức toán học qua matplotlib.mathtext | `src/services/smart_ink_ocr.py`, `src/ui/smart_ink_review_dialog.py` | ĐÃ ÁP DỤNG |
| 14 | `ui-ux-pro-max` & `design-system` | Giao diện & Thiết kế | Chuẩn hóa bảng màu Academic (Slate/Deep Navy/Accent Blue), tỷ lệ không gian, Typography Segoe UI | `docs/DESIGN_SYSTEM.md`, `main.py`, `src/ui/login_window.py` | ĐÃ ÁP DỤNG |
| 15 | `frontend-design` & `brand` | Giao diện & Thiết kế | Tinh chỉnh tính thẩm mỹ của Teacher Dashboard, bảng viết kỹ thuật số chuyên nghiệp | Nâng cấp giao diện `MainWindow`, `TeacherWorkspaceDashboard` | ĐÃ ÁP DỤNG |
| 16 | `ai-software-architect-developer` | Kiến trúc & AI | Thiết kế kiến trúc Local-First, tách bạch luồng dữ liệu giữa camera, inference, render và database | `docs/ARCHITECTURE.md`, `src/services/` | ĐÃ ÁP DỤNG |
| 17 | `project-documentation-architect` | Tài liệu khoa học | Xây dựng bộ hồ sơ kỹ thuật đầy đủ: Baseline, Architecture, Auth Flow, AI Pipeline, QA Strategy | Toàn bộ các tệp tài liệu trong `docs/` | ĐÃ ÁP DỤNG |
| 18 | `managing-python-dependencies` | Môi trường | Đảm bảo môi trường thực thi an toàn, không làm hỏng thư viện hệ thống, quản lý fallback offline | `requirements.txt`, `config.py` | ĐÃ ÁP DỤNG |
| 19 | `critical-project-reviewer` | Đánh giá phản biện | Rà soát các giả định, kiểm tra điều kiện biên (mất camera, ánh sáng yếu, không có webcam) | `docs/QA_STRATEGY.md`, bộ test ca biên | ĐÃ ÁP DỤNG |
