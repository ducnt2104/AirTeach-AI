# AIRTEACH AI - WORD DOSSIER TRACEABILITY MATRIX
## Ma Trận Truy Vết Kỹ Thuật Giữa Bản Thuyết Minh & Mã Nguồn Thực Tế

Tài liệu này đối chiếu 100% các yêu cầu kỹ thuật, công thức toán học, thuật toán và kiến trúc được mô tả trong **Bản Thuyết Minh Dự Án (`Ban_Thuyet_Minh_AirTeach_AI_Nang_Cap.docx`)** với mã nguồn thực tế đã cài đặt trong dự án **AirTeach AI**.

---

### BẢNG ĐỐI CHIẾU HỆ THỐNG (REQUIREMENT-TO-CODE TRACEABILITY)

| Mục Thuyết Minh | Đặc tả Kỹ thuật | Trạng thái | Tệp mã nguồn cài đặt | Chi tiết triển khai & Kiểm thử |
| :--- | :--- | :--- | :--- | :--- |
| **I. TỔNG QUAN HỆ THỐNG** | Nền tảng dạy học thông minh tương tác không chạm bằng cử chỉ tay qua Webcam máy tính thường (Desktop Windows). | **HOÀN TẤT** | `main.py`<br>`Run_AirTeach_AI.bat` | Ứng dụng Desktop chạy độc lập trên Windows, hỗ trợ Python 3.10+, tự động thiết lập môi trường và quản lý vòng đời ứng dụng. |
| **II.1. THU HÌNH & THEO DÕI TAY** | OpenCV VideoCapture DirectShow + MediaPipe 21 điểm mốc 3D (`mp.tasks.vision.HandLandmarker`). | **HOÀN TẤT** | `src/core/hand_gesture_engine.py`<br>`models/hand_landmarker.task` | Thu nhận webcam ở luồng nền `HandTrackingWorker`, trích xuất 21 điểm mốc 3D chuẩn hóa, hiển thị khung xương màu thời gian thực và tự động bù sáng CLAHE. |
| **II.2. BỘ LỌC THỜI GIAN THỰC (TẦNG 1)** | Bộ lọc thông thấp thích ứng vận tốc **1€ Filter (Casiez et al., 2012)**: $f_c = f_{c,\min} + \beta \|\dot{x}\|$. | **HOÀN TẤT** | `src/core/stroke_beautifier.py`<br>(`OneEuroFilter2D`) | Lọc rung tay (jitter) ở tốc độ chậm ($f_{c,\min} = 1.0\text{ Hz}$), triệt tiêu độ trễ khi lướt bút nhanh ($\beta = 0.007$). Đạt 100% test trong `test_core_algorithms.py`. |
| **II.3. LÀM MỊN HẬU KỲ (TẦNG 2)** | Rút gọn điểm Ramer–Douglas–Peucker (RDP) + Nội suy đường cong Cubic Bézier Spline $C^1$ có bề dày biến thiên theo vận tốc. | **HOÀN TẤT** | `src/core/stroke_beautifier.py`<br>(`ramer_douglas_peucker`, `fit_cubic_bezier_spline`) | Rút gọn 70-85% điểm thừa, tính tiếp tuyến Catmull-Rom liên tục $C^1$, điều biến bề dày nét $w(v) = w_{\max} - (w_{\max}-w_{\min})\frac{v}{v_{\max}}$. |
| **II.4. NẮN HÌNH HỌC TỰ ĐỘNG (TẦNG 3)** | Nhận dạng hình học: Đoạn thẳng, Mũi tên, Đường tròn, Elip, Chữ nhật, Tam giác qua Bao lồi (`ConvexHull`) & Độ tròn $C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$. | **HOÀN TẤT** | `src/core/stroke_beautifier.py`<br>(`GeometricShapeSnapper`) | Phân tích tỉ lệ khoảng cách đầu-cuối $d(P_0, P_N)/L$, độ lồi bao lồi, góc nội tiếp tam giác và tự động khóa góc $\pm 5^\circ$ theo trục tọa độ. |
| **II.5. CỖ MÁY TRẠNG THÁI CỬ CHỈ (FSM)** | Finite State Machine 6 cử chỉ: `POINT` (Tia laser), `DRAW` (Vẽ nét), `PAUSE` (Nắm tay/Nhấc bút), `SELECT` (Chọn vùng), `SWIPE_LEFT/RIGHT` (Chuyển trang), `CONFIRM` (👍). | **HOÀN TẤT** | `src/core/hand_gesture_engine.py`<br>(`HandGestureFSM`) | Chuẩn hóa theo tỉ lệ bàn tay $L_{\text{hand}} = \|P_0 - P_9\|_2$, bộ đệm chống nhấp nháy 3 frames, ngưỡng chống kẹt dead-zone hysteresis và kiểm tra ngón gập (curl). |
| **III.1. XÁC THỰC GIÁO VIÊN CỤC BỘ** | Nhận diện khuôn mặt 100% Offline: OpenCV DNN **YuNet** (Phát hiện) + **SFace** (Trích xuất 128D) + Mã hóa Fernet + PIN dự phòng PBKDF2-SHA256 + SQLite. | **HOÀN TẤT** | `src/services/face_profile_service.py`<br>`src/ui/login_window.py`<br>`data/profiles.db` | Không gửi ảnh ra ngoài Internet; kiểm tra Liveness qua độ dịch chuyển góc mặt và kích thước mắt (EAR); cô lập không gian làm việc tại `data/workspaces/<uuid>/`. |
| **III.2. NHẬN DẠNG CHỮ & CÔNG THỨC TOÁN** | Offline Smart Ink: RapidOCR (PaddleOCR ONNX) cho chữ viết tay + Rapid-LaTeX-OCR / Pix2Tex cho công thức toán + Dựng vector `matplotlib.mathtext`. | **HOÀN TẤT** | `src/services/offline_recognizer.py`<br>`src/ui/lesson_board.py` | Chuyển đổi chữ viết tay Việt/Anh và phương trình toán học thành chuỗi mã LaTeX và dựng lớp phủ vector sắc nét trên bảng giảng dạy. |
| **IV.1. TRỢ LÝ AI TRỰC TUYẾN** | Trợ lý hướng dẫn và giải đáp sư phạm qua Google AI Studio API (`google-genai` SDK), tự động chuyển sang chế độ hướng dẫn ngoại tuyến khi mất mạng. | **HOÀN TẤT** | `src/services/gemini_assistant.py`<br>`src/ui/chat_panel.py` | Đọc khóa từ `GEMINI_API_KEY` trong `.env`, chụp ảnh vùng bảng được chọn (ROI snapshot) để giảng giải trực quan, hỗ trợ streaming phản hồi. |
| **IV.2. BỘ CÔNG CỤ CHUYÊN ĐỀ STEM** | Thanh công cụ chuyên sâu cho 5 bộ môn: Toán (Hệ Oxy, đồ thị $f(x)$), Vật lý (Vectơ lực, phân tích $F_x, F_y$, mạch điện), Hóa học (Công thức Unicode, mũi tên phản ứng ⇄, vòng Benzen), Sinh học (Ghim chú thích), Tin học (Khối thuật toán, Code block). | **HOÀN TẤT** | `src/ui/stem_tools.py`<br>`src/ui/main_window.py` | Tích hợp menu chuyên đề STEM động trên thanh công cụ chính, cho phép chèn tức thì các cấu trúc vector sư phạm chuẩn xác vào bài học. |
| **V.1. NHẬP HỌC LIỆU & GÓI BÀI GIẢNG** | Nhập tệp PDF (`pypdfium2` + `pypdf`), DOCX/PPTX (COM API / LibreOffice headless). Lưu trữ bài giảng dạng nén `.airteach` kèm lịch sử nét vẽ `timestamps_ms`. | **HOÀN TẤT** | `src/services/document_manager.py`<br>`src/services/airteach_packager.py`<br>`src/ui/replay_controller.py` | Hỗ trợ lưu trữ hoàn chỉnh và tính năng Replay (tua lại quá trình viết bài của giáo viên theo đúng thời gian thực). |
| **VI. TRÌNH KHỞI CHẠY WINDOWS 1-CLICK** | Tệp khởi chạy `Run_AirTeach_AI.bat` tại thư mục gốc, tự động phát hiện Python, cài đặt phụ thuộc, kiểm tra mô hình và ghi log. | **HOÀN TẤT** | `Run_AirTeach_AI.bat`<br>`logs/launcher.log` | Chuẩn mã hóa UTF-8 với ngắt dòng CRLF chuẩn Windows, hỗ trợ các tham số dòng lệnh `--check`, `--version`, bẫy mã lỗi chi tiết. |

---

### TRUY VẾT CÁC CÔNG THỨC TOÁN HỌC CỐT LÕI

1. **Công thức 1€ Filter (Tần số cắt thích ứng):**
   $$f_c = f_{c,\min} + \beta \cdot \|\dot{x}\|$$
   $$\alpha = \frac{1}{1 + \frac{\tau}{\Delta t}}, \quad \tau = \frac{1}{2\pi f_c}$$
   - *Cài đặt:* `src/core/stroke_beautifier.py`, lớp `OneEuroFilter1D._alpha()` và `OneEuroFilter2D.filter()`.
   - *Tham số mặc định:* $f_{c,\min} = 1.0\text{ Hz}, \beta = 0.007, d_{\text{cutoff}} = 1.0\text{ Hz}$.

2. **Công thức Độ tròn nhận dạng đường tròn/elip:**
   $$C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$$
   - *Cài đặt:* `src/core/stroke_beautifier.py`, phương thức `GeometricShapeSnapper._is_circle_or_ellipse()`.
   - *Tiêu chuẩn:* $C \ge 0.78$ kết hợp tỉ lệ khép kín đầu-cuối $d(P_0, P_N)/L \le 0.18$.

3. **Công thức Nội suy Bézier bậc 3 ($C^1$ Continuity):**
   $$B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3, \quad t \in [0, 1]$$
   - *Cài đặt:* `src/core/stroke_beautifier.py`, hàm `fit_cubic_bezier_spline()`.
   - *Điểm điều khiển:* Tính toán dựa trên vận tốc và tiếp tuyến Catmull-Rom tại mỗi đỉnh điểm mốc.

4. **Công thức Chuẩn hóa Tỉ lệ Bàn tay (Scale-Invariance):**
   $$L_{\text{hand}} = \|P_{\text{wrist}} - P_{\text{middle\_mcp}}\|_2$$
   - *Cài đặt:* `src/core/hand_gesture_engine.py`, hàm `HandGestureFSM.process_landmarks()`.
   - *Ý nghĩa:* Đảm bảo khoảng cách ngón trỏ và ngón cái khi chụm vẽ (DRAW) hoặc chỉ (POINT) không bị phụ thuộc vào việc giáo viên đứng gần hay xa webcam.
