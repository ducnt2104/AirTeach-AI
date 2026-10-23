# NHẬT KÝ THI CÔNG DỰ ÁN (WORK LOG)

**Dự án:** AirTeach AI — Nền tảng giảng dạy tương tác không chạm bằng cử chỉ tay  
**Khởi tạo nhật ký:** 20/10/2026

---

## NHẬT KÝ CHI TIẾT THEO MỐC THỜI GIAN

### [20/10/2026 21:11] — Khởi động /goal và Rà soát Không gian làm việc
- Kiểm tra môi trường hệ thống: Windows Desktop, Python 3.15.0rc1, Python 3.13, PySide6 6.12.0, OpenCV 5.0.0.93, MediaPipe 1.1.0.
- Chạy thử toàn bộ 217 bài test kiểm thử hồi quy: 217 passed trong 38.96s.
- Rà soát thư mục `models/`: Có sẵn 3 tệp mô hình chính xác:
  - `face_detection_yunet_2023mar.onnx`
  - `face_recognition_sface_2021dec.onnx`
  - `hand_landmarker.task`
- Rà soát thư mục `assets/`: Hiện đang rỗng, cần tạo toàn bộ hệ thống đồ họa vector SVG.

### [20/10/2026 21:15] — Phát hiện & Sửa lỗi Nghiêm trọng (Critical Bugfixes)
1. **Lỗi MediaPipe 1.x:** `src/core/hand_gesture_engine.py` gọi `mp.solutions.hands`, gây lỗi `AttributeError: module 'mediapipe' has no attribute 'solutions'` vì MediaPipe 1.1.0 đã chuyển sang `mp.tasks.vision.HandLandmarker`.
   - *Khắc phục:* Viết lại tầng nạp mô hình hỗ trợ song song cả Tasks API (`models/hand_landmarker.task`) lẫn solutions kế thừa.
2. **Lỗi Luồng Camera Treo khi Đóng Cửa Sổ:** `FaceAuthDialog` và `LoginWindow` không dừng luồng nền an toàn khi camera không khả dụng (`camera_index < 0`).
   - *Khắc phục:* Thêm điều kiện kiểm tra `camera_index < 0`, bổ sung timeout và `terminate()` dự phòng khi `wait()` quá 1000ms. Thêm phương thức `closeEvent()` và `reject()` cho `FaceAuthDialog`.

### [20/10/2026 21:20] — Kiểm toán Thị giác & Phát hiện Phông chữ
- Render ảnh chụp màn hình các cửa sổ chính: `screen_login_win.png`, `screen_main_win.png`, `screen_prep_win.png`, `screen_settings_win.png`, `screen_face_enroll_win.png`.
- Phát hiện: Chế độ offscreen QPA làm mất font hệ thống (hiện ô vuông `□`). Trên nền tảng Windows DirectWrite gốc, phông Segoe UI dựng chữ tiếng Việt chuẩn xác 100%.

### [20/10/2026 21:24] — Tác giả hóa 11 Agent Skills & Tài liệu Chuẩn hóa
- Khởi tạo 11 specialized agent skills tại `.agents/skills/`.
- Hoàn thiện toàn bộ hệ thống tài liệu kiểm toán: `docs/PRODUCT_AUDIT.md`, `docs/GRAPHICS_AND_UX_AUDIT.md`, `docs/AI_MODEL_AUDIT.md`, `docs/PERFORMANCE_BASELINE.md`, `docs/TEST_PLAN.md`, `docs/TEST_RESULTS.md`, `docs/ASSET_AND_MODEL_LICENSES.md`, `docs/IMPLEMENTATION_BACKLOG.md`, `docs/WORK_LOG.md`.

### [20/10/2026 21:35] — Xây dựng Bộ Tài nguyên Vector SVG & Hộp thoại Tương tác Mới
1. **Thiết kế & Tạo lập 33 Vector SVG Assets:**
   - 24 biểu tượng công cụ dạy học chuẩn Qt6 (`assets/icons/`).
   - 7 thẻ minh họa cử chỉ không chạm trực quan (`assets/gestures/`).
   - Sơ đồ kiến trúc xử lý nét viết 3 tầng Stroke Beautifier (`assets/diagrams/pipeline_stroke_beautifier.svg`).
   - Tranh minh họa thư viện bài giảng trống (`assets/illustrations/empty_library.svg`).
2. **Hộp thoại Huấn luyện & Hiệu chuẩn Cử chỉ (`GestureTrainerDialog`):**
   - Lưới hiển thị 7 thẻ cử chỉ tương tác, chuyển màu xanh lá khi giáo viên thực hiện ổn định.
   - Luồng camera hiển thị bộ xương bàn tay 21 điểm 3D (`cv2.LINE_AA` & `cv2.circle`).
   - HUD đo lường thời gian thực: Cử chỉ hiện tại, độ tin cậy %, khoảng cách pinch, số khung hình debounce.
   - Nút khôi phục góc hiệu chuẩn chuẩn (Full Canvas).
3. **Hộp thoại Kiểm duyệt Nhận dạng Mực thông minh (`SmartInkReviewDialog`):**
   - So sánh trực quan: Ảnh crop nét viết tay gốc vs Công thức vector kết xuất.
   - Kết xuất động công thức toán học qua `matplotlib.mathtext` (không phụ thuộc TeX cài ngoài).
   - Ô nhập mã LaTeX / văn bản nhận dạng cho phép giáo viên chỉnh sửa trực tiếp theo thời gian thực.
   - Tùy chọn chèn: "Chèn công thức song song" hoặc "Thay thế hoàn toàn nét viết tay".

### [20/10/2026 21:42] — Nâng cấp MainWindow Dual Toolbars & Bộ Test Mới (231/231 Passed)
1. **MainWindow UI/UX:**
   - Tích hợp 2 thanh công cụ chuyên biệt: Thanh điều hướng bài giảng (Navigation Toolbar) & Thanh giảng dạy bảng viết (Classroom Canvas Toolbar).
   - Bộ chuyển đổi chế độ công cụ: `PEN`, `LASER`, `ERASER`, `SELECT`.
   - Bảng chọn nhanh 5 màu mực chuẩn bảng viết & bộ chọn độ dày nét (1.5px, 2.5px, 4.5px, 8.0px).
   - Ngăn xếp Hoàn tác / Làm lại (Undo / Redo Stack) đầy đủ cho các nét vẽ tay.
2. **Kiểm thử tự động hóa mới (`tests/test_svg_assets_and_trainer.py`):**
   - Viết 14 ca kiểm thử chuyên sâu: Xác thực toàn bộ 33 tệp SVG, kiểm thử vòng đời `GestureTrainerDialog`, kiểm duyệt công thức `SmartInkReviewDialog`, và thanh công cụ giảng dạy `MainWindow`.
   - Chạy toàn bộ test suite dự án: **231 / 231 tests PASSED (100%) trong 39.12s**, 0 lỗi, 0 hồi quy.

### [20/10/2026 21:48] — Đóng gói Bản phân phối Windows Standalone (PyInstaller)
1. **Cấu hình `airteach.spec`:**
   - Tích hợp các plugin Qt6 (`QtSvg`, `QtSvgWidgets`, `QtPrintSupport`), các mô hình ONNX (`models/`), toàn bộ tài nguyên SVG (`assets/`), và mô-đun toán học `matplotlib.mathtext`.
   - Cập nhật `config.py` và `main.py` tự động nhận diện `sys.frozen` và `sys._MEIPASS`.
2. **Biên dịch & Xác thực Phân phối (`scripts/build_windows.py`):**
   - Tạo thành công tệp thực thi độc lập `dist/AirTeachAI/AirTeachAI.exe` (29.03 MB).
   - Xác minh tự động 100%: 3 mô hình ONNX / MediaPipe task bundle đầy đủ trong `_internal/models/`, 33 tệp vector SVG hiện diện đầy đủ trong `_internal/assets/`.
   - Chụp ảnh Visual QA trên Windows DirectWrite: `screen_main_v2_win.png`, `screen_gesture_trainer_win.png`, `screen_smart_ink_review_win.png`.

### [23/10/2026 14:10] — Chiến Dịch Kỹ Thuật Tự Trị: Khắc Phục Triệt Để Lỗi Vẽ Không Chạm P0 & Bộ Công Cụ STEM
1. **Khắc phục lỗi chặn tương tác P0 (Air Drawing Defect):**
   - *Nguyên nhân 1 (Double-Scaling):* `_scene_coords` trong `src/ui/main_window.py` nhân đôi tọa độ pixel `(cx, cy)` với `BOARD_WIDTH` (1920) và `BOARD_HEIGHT` (1080), làm văng tọa độ lên $(1.843.200, 583.200)$. Đã khắc phục bằng bộ lọc bảo vệ và kẹp tọa độ $[0, 1920] \times [0, 1080]$.
   - *Nguyên nhân 2 (Dead-Zone & Tilt Flaw):* Ngưỡng ngón cái cũ tạo vùng chết $0.35 - 0.41$ gây rớt cử chỉ về `NONE`. Thuật toán vươn ngón phụ thuộc trục Y màn hình làm hỏng cử chỉ khi bàn tay nghiêng. Đã khắc phục bằng khoảng cách cổ tay - khớp gốc ngón tay và thuật toán kiểm tra ngón gập (curl).
   - *Nguyên nhân 3 (NONE Event Dropping):* `_QtCameraWorker` không phát tín hiệu `NONE` khi hạ tay, khiến nét vẽ không kết thúc và tạo đường nối chéo khổng lồ khi đưa tay lên lại. Đã sửa để luôn phát `NONE` và gọi `_finalize_stroke()`.
   - *Hiển thị chẩn đoán (Diagnostic Skeleton Overlay):* Vẽ trực tiếp 21 khớp, đoạn xương, vòng trạng thái đầu ngón tay (Đỏ = DRAW, Xanh = POINT) và huy hiệu HUD thời gian thực lên khung hình camera.
2. **Xây dựng Bộ Công cụ Chuyên đề STEM (`src/ui/stem_tools.py`):**
   - *Toán học:* Hệ trục tọa độ vuông góc Oxy (lưới, mũi tên, vạch chia, nhãn) và bộ vẽ đồ thị hàm số $f(x)$ liên tục (`numpy.linspace` + `eval` an toàn).
   - *Vật lý:* Vectơ lực $\vec{F}$, phân tích lực trực giao $F_x, F_y$ kèm đường dóng nét đứt, linh kiện sơ đồ mạch điện (Điện trở, Tụ điện, Nguồn DC).
   - *Hóa học:* Tự động định dạng công thức hóa học Unicode ($H_2SO_4 \to H_2SO_4$, $Cu^{2+} \to Cu^{2+}$, $Fe^{3+} \to Fe^{3+}$), mũi tên phản ứng thuận nghịch $\rightleftharpoons$ kèm điều kiện nhiệt độ/xúc tác/áp suất, vòng thơm Benzen $C_6H_6$.
   - *Sinh học:* Ghim định vị và đường chỉ dẫn chú thích giải phẫu/tế bào.
   - *Tin học:* Khối sơ đồ thuật toán (Thực thi, Điều kiện, Vào/Ra, Bắt đầu/Kết thúc) và khung hiển thị code IDE tối màu.
   - Tích hợp thanh công cụ STEM chuyên biệt vào `MainWindow` chuyển đổi động theo môn học.
3. **Trình khởi chạy 1-Click Windows (`Run_AirTeach_AI.bat`):**
   - Thiết kế kịch bản batch chuẩn Windows với ngắt dòng CRLF và UTF-8 Code Page 65001.
   - Tự động nhận diện `.venv` hoặc Python hệ thống, kiểm tra phiên bản $\ge 3.10$.
   - Tự động kiểm tra và cài đặt thiếu phụ thuộc từ `requirements.txt`.
   - Hỗ trợ tham số `--version`, `--check`, `--dry-run`, bẫy mã lỗi và ghi log vào `logs/launcher.log`.
4. **Kiểm thử hồi quy toàn diện:**
   - Bổ sung 25 ca kiểm thử chuyên sâu trong `tests/test_stem_and_tracking.py`.
   - Toàn bộ test suite đạt **275 / 275 tests PASSED (100%)**, không có bất kỳ lỗi hồi quy nào.
