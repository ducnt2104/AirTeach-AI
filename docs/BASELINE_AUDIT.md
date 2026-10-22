# AIRTEACH AI — BASELINE AUDIT REPORT
**Phiên bản mục tiêu:** 1.0.0 — Windows Local-First Desktop Edition  
**Thời điểm thực hiện:** 21/10/2026  
**Chế độ thực thi:** Antigravity `/goal`  
**Chính sách chất lượng:** Zero Mock Data Policy — 100% kiểm định trên terminal và giao diện thực tế.

---

## 1. MÔI TRƯỜNG THỰC THI THỰC TẾ (RUNTIME BASELINE)

- **Hệ điều hành:** Windows 10/11 x86_64
- **Python Runtime:** Python 3.15.0rc1 (System interpreter tại `C:\Users\ADMIN\AppData\Local\Programs\Python\Python315\python.exe`)
- **PySide6 / Qt:** PySide6 6.12.0 (Qt 6.12)
- **Thư viện thị giác máy tính:** OpenCV (`cv2`) 5.0.0.93 + MediaPipe 1.1.0 (`mp.tasks.vision.HandLandmarker`)
- **Kho mô hình offline (`models/`):**
  - `face_detection_yunet_2023mar.onnx` (232,589 bytes) — Present & Verified
  - `face_recognition_sface_2021dec.onnx` (38,696,353 bytes) — Present & Verified
  - `hand_landmarker.task` (7,819,105 bytes) — Present & Verified
- **Cơ sở dữ liệu hồ sơ giáo viên:** SQLite tại `data/profiles.db` (Schema: `teachers` table với Fernet AES-128 ciphertext + PBKDF2 HMAC-SHA256 PIN salt/hash)
- **Kiểm thử hiện hữu:** 231 tests trong `tests/` — 231/231 PASSED trong 39.68s.

---

## 2. PHÂN TÍCH NGUYÊN NHÂN GỐC RỄ CÁC LỖI TRỌNG YẾU (ROOT CAUSE ANALYSIS)

### Lỗi 1: Luồng đăng ký giáo viên chưa hoàn chỉnh hoặc chuyển trạng thái sai
- **Hiện tượng:** Giáo viên nhập họ tên, môn học, mã PIN nhưng camera chụp mẫu khuôn mặt không phản hồi hoặc bị văng về màn hình đăng nhập.
- **Nguyên nhân gốc rễ:**
  1. Trong `src/ui/login_window.py`, `TeacherEnrollDialog` là một hộp thoại con không có khung xem trước camera. Khi người dùng bấm "Bắt đầu chụp", dialog này đóng lại (`accept()`), đẩy người dùng về `LoginWindow`.
  2. Tại `_LoginCameraWorker`, điều kiện lấy mẫu khuôn mặt dùng ngưỡng `confidence >= 0.5`. Tuy nhiên, khi chuyển sang `enroll_from_frames` trong `face_profile_service.py`, hàm này lại gọi `detect_faces` lần 2 với `score_threshold = 0.6`. Nếu khung hình có score từ 0.50 đến 0.59, `detect_faces` trả về `None`, dẫn đến `ValueError: Only X valid face detections from 5 frames. Need at least 2.` và làm hỏng toàn bộ tiến trình đăng ký.
  3. Không có cơ chế đăng ký bằng PIN dự phòng trực tiếp khi thiết bị không có webcam hoặc ánh sáng không đủ điều kiện phát hiện khuôn mặt.

### Lỗi 2: Thông báo quét thành công nhưng nhận diện Face ID không đăng nhập vào tài khoản
- **Hiện tượng:** Camera hiện khung xanh 95%+ trên mặt, thanh liveness đạt 15/15 frames, nhưng hệ thống không tự động đăng nhập.
- **Nguyên nhân gốc rễ:**
  1. Khi chưa có giáo viên nào trong cơ sở dữ liệu (`data/profiles.db`), `_get_all_teachers()` trả về danh sách rỗng `[]`. `authenticate_by_face` trả về `None`. Tuy nhiên, giao diện vẫn tiếp tục hiển thị khung nhận diện màu xanh và thanh tiến trình xanh lá mà không có bất kỳ thông báo hướng dẫn nào cho giáo viên mới ("Chưa có tài khoản, vui lòng đăng ký trước").
  2. `_liveness.is_ready()` kiểm tra `len(history) >= 15`. Sau 15 khung hình, hàm luôn trả về `True` liên tục 30 lần mỗi giây, khiến `authenticate_by_face` liên tục giải mã Fernet và tính cosine similarity trên mỗi khung hình ở luồng camera, làm sụt giảm FPS.
  3. Khi không khớp hồ sơ nào (`best_score < COSINE_THRESHOLD`), hệ thống không cập nhật trạng thái trực quan để giáo viên biết độ khớp là bao nhiêu (ví dụ: `Độ khớp: 18% / Cần 36.3%`).

### Lỗi 3: Sau đăng ký, ứng dụng quay lại màn hình tên, ID và môn học thay vì chuyển sang màn hình làm việc
- **Hiện tượng:** Đăng ký thành công xong không chuyển sang Workspace/Studio mà bị kẹt hoặc quay lại form đăng nhập/đăng ký.
- **Nguyên nhân gốc rễ:**
  1. Trong `LoginWindow._on_enroll_finished`, sau khi tạo giáo viên thành công, hệ thống chỉ gọi `refresh_teacher_dropdown()` và đóng camera `_stop_camera()`.
  2. Tại `_stop_camera()`, phương thức `quit()` và `wait(1000)` nếu bị timeout do DirectShow driver đang bận sẽ gọi `terminate()`. Lệnh `terminate()` làm luồng camera bị ngắt cưỡng bức trước khi `cap.release()` kịp chạy, dẫn đến việc thiết bị camera bị giữ khóa (locked device handle). Khi `MainWindow` khởi động và cố mở lại `cv2.VideoCapture(0, cv2.CAP_DSHOW)`, camera bị từ chối truy cập và ứng dụng bị kẹt.
  3. `main.py` không kiểm tra việc đóng và mở cửa sổ một cách an toàn (`QApplication.setQuitOnLastWindowClosed(False)`), dẫn đến nguy cơ ứng dụng bị đóng sự kiện khi `login_win` đóng trước khi `main_win` kịp hiển thị hoàn toàn.

### Lỗi 4: Màn hình sau đăng nhập cần là Dashboard Trung tâm Giảng dạy tích hợp
- **Hiện tượng:** Sau khi vào tài khoản, người dùng bị đưa ngay vào một bảng trắng trơn không có tiêu đề bài học, không có danh sách bài giảng gần đây, không có hướng dẫn cử chỉ.
- **Giải pháp:** Cung cấp Trung tâm Giảng dạy Giáo viên (Teacher Workspace Dashboard) tích hợp:
  - Thẻ thông tin giáo viên, môn phụ trách, phiên làm việc hiện tại.
  - Phím tắt mở nhanh: **Air Gesture Studio (Giảng dạy cử chỉ không chạm)**, **Tạo bài giảng mới**, **Thư viện học liệu (PPTX/PDF/Word)**, **Huấn luyện & Hiệu chuẩn cử chỉ (Gesture Trainer)**.
  - Điều hướng mượt mà, chuyển đổi linh hoạt giữa Dashboard và Canvas Bảng viết.

---

## 3. MA TRẬN ĐÁNH GIÁ CÁC PHÂN HỆ CỐT LÕI

| Phân hệ | File thực thi | Trạng thái hiện tại | Vấn đề cần khắc phục |
|---|---|---|---|
| **Xác thực khuôn mặt & PIN** | `src/services/face_profile_service.py` | Hoạt động | Cần nới lỏng ngưỡng lấy mẫu đồng bộ, hỗ trợ đăng ký PIN-only khi không có webcam, cung cấp phản hồi độ khớp rõ ràng |
| **Giao diện Đăng nhập & Đăng ký** | `src/ui/login_window.py` | Cần cải tổ | Thiết kế lại hộp thoại đăng ký tích hợp camera trực quan + hỗ trợ PIN fallback; khắc phục deadlock giải phóng camera |
| **Theo dõi cử chỉ & 21 Landmarks** | `src/core/hand_gesture_engine.py` | Hoạt động tốt | Hỗ trợ 6 cử chỉ, bộ lọc debounce 3 frames, perspective warping |
| **Bộ làm nét nét viết 3 tầng** | `src/core/stroke_beautifier.py` | Hoạt động tốt | 1€ Filter, RDP, Catmull-Rom Bézier spline, Shape Snapping 6 dạng hình học |
| **Bảng viết tương tác** | `src/ui/lesson_board.py` | Hoạt động | Hỗ trợ 3 tầng Z, hoàn tác/làm lại, xóa nét, đổi màu, đổi độ dày |
| **Quản lý học liệu & Đóng gói .airteach** | `src/services/document_manager.py` | Hoạt động | pypdfium2 PDF, COM/LibreOffice PPTX/DOCX, gói ZIP chuẩn |
| **Trung tâm thư viện & Soạn bài** | `src/ui/library_and_prep_widget.py` | Hoạt động | Cần tích hợp nổi bật vào Teacher Dashboard |
| **Nhận diện chữ & công thức toán** | `src/services/smart_ink_ocr.py` | Hoạt động | GeometricMathParser + Matplotlib MathText offline |
| **Trợ lý AI giảng dạy** | `src/services/gemini_assistant.py` | Hoạt động | Google GenAI SDK + offline educational handbook |

---

## 4. KẾ HOẠCH HÀNH ĐỘNG KHÔI PHỤC TOÀN DIỆN (ROADMAP)

1. **Giai đoạn 1:** Sửa triệt để luồng đăng ký & xác thực giáo viên (`face_profile_service.py`, `login_window.py`, `main.py`).
2. **Giai đoạn 2:** Giải quyết triệt để xung đột giải phóng camera giữa các màn hình, đảm bảo chuyển trạng thái mượt mà 100%.
3. **Giai đoạn 3:** Xây dựng Teacher Workspace Dashboard trung tâm, thống nhất trải nghiệm từ Đăng nhập → Dashboard → Bảng viết Air Gesture Studio → Thư viện bài giảng.
4. **Giai đoạn 4:** Nâng cấp hệ thống thiết kế giao diện (Design System, Academic Dark/Light, Typography, SVG icons, Motion).
5. **Giai đoạn 5:** Kiểm thử tự động hóa, kiểm thử giao diện thực tế và đóng gói bản phân phối release.
