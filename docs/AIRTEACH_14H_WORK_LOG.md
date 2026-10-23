# AIRTEACH AI - 14-HOUR AUTONOMOUS ENGINEERING WORK LOG
## Nhật Ký Kỹ Thuật Chi Tiết Chiến Dịch Phát Triển Tự Trị (14-Hour Work Log)

**Chiến dịch:** `AIRTEACH-AI-AUTONOMOUS-14H-V3`  
**Dự án:** AirTeach AI — Hệ thống dạy học thông minh tương tác không chạm bằng AI  
**Nền tảng:** Windows Desktop (Python 3.10+, PySide6, OpenCV, MediaPipe, Local AI)  
**Tài liệu tham chiếu:** `Ban_Thuyet_Minh_AirTeach_AI_Nang_Cap.docx`

---

### MỐC THỜI GIAN VÀ TIẾN ĐỘ THỰC HIỆN CHIẾN DỊCH

#### GIAI ĐOẠN 1: KHỞI ĐỘNG CHIẾN DỊCH, RÀ SOÁT VÀ TÁI HIỆN LỖI CHẶN P0
- **Nhiệm vụ:** Tiếp nhận yêu cầu người dùng: *"Người dùng giơ một ngón tay trước camera nhưng AirTeach AI không nhận diện đúng; tính năng vẽ nét trên không (air drawing) không hoạt động."*
- **Kiểm toán môi trường:**
  - Hệ điều hành: Windows 11 x64, Python 3.15.0rc1 / Python 3.12 (.venv).
  - Thư viện: PySide6 6.12.0, OpenCV 5.0.0.93, MediaPipe 1.1.0, NumPy 2.2.6, Cryptography 46.0.5.
- **Rà soát mã nguồn & Tái hiện lỗi:**
  - Phát hiện hàm `MainWindow._scene_coords` nhân đôi tọa độ pixel `(cx, cy)` đã ánh xạ với kích thước màn hình `1920` và `1080`. Điểm nét vẽ bị đẩy ra tọa độ $(1.843.200, 583.200)$, nằm hoàn toàn ngoài tầm nhìn.
  - Phát hiện cỗ máy trạng thái `HandGestureFSM` trong `src/core/hand_gesture_engine.py` có vùng chết ngưỡng ngón cái giữa $0.34$ và $0.42$.
  - Phát hiện việc kiểm tra vươn ngón trỏ phụ thuộc trục thẳng đứng Y ($y_{\text{tip}} < y_{\text{pip}}$), khiến cử chỉ thất bại khi bàn tay nghiêng.
  - Phát hiện `_QtCameraWorker` bỏ qua việc phát sự kiện khi không có bàn tay, khiến nét vẽ cũ không được ngắt và tạo đường kẻ nối chéo khi đưa tay lên lại.

#### GIAI ĐOẠN 2: THIẾT KẾ & KHẮC PHỤC TRIỆT ĐỂ LỖI P0
- **Khắc phục lỗi nhân đôi tọa độ:**
  - Viết lại `MainWindow._scene_coords`: Tự động nhận diện hệ tọa độ chuẩn hóa $[0, 1]$ so với hệ tọa độ pixel $[0, 1920]$, kẹp chặt biên an toàn trong $[0, 1920] \times [0, 1080]$.
- **Khắc phục vùng chết & góc nghiêng bàn tay:**
  - Chuẩn hóa khoảng cách ngón cái - ngón trỏ $r_{\text{thumb}} = \|P_4 - P_6\| / L_{\text{hand}}$: $\le 0.38$ là DRAW, $> 0.38$ là POINT. Xóa bỏ hoàn toàn khoảng hở vùng chết.
  - Dùng khoảng cách Euclid từ cổ tay tới đầu ngón so với đốt khớp gốc để phát hiện vươn ngón độc lập với góc xoay bàn tay.
  - Bổ sung hàm kiểm tra ngón gập `_is_finger_curled` cho ngón giữa, áp út và út.
- **Khắc phục ngắt nét vẽ khi hạ tay:**
  - Cập nhật `_QtCameraWorker` luôn phát tín hiệu `GestureState.NONE` khi không có bàn tay để kích hoạt `_finalize_stroke()` và ẩn laser.
- **Hiển thị khung xương chẩn đoán (Diagnostic Skeleton Overlay):**
  - Cài đặt hàm vẽ trực tiếp lên khung hình camera: 21 khớp tròn màu lục lam (#00D2FF), đoạn xương liên kết (#00A8FF), vòng trạng thái đầu ngón trỏ (Đỏ: DRAW, Xanh lục: POINT) và huy hiệu HUD Mode Badge hiển thị thời gian thực.

#### GIAI ĐOẠN 3: XÂY DỰNG TRÌNH KHỞI CHẠY 1-CLICK & BỘ KHÓA CAMERA IPC
- **Trình khởi chạy 1-Click `Run_AirTeach_AI.bat`:**
  - Soạn thảo kịch bản batch với mã hóa UTF-8 (Code Page 65001) và ngắt dòng Windows CRLF.
  - Tự động nhận diện môi trường ảo `.venv` hoặc Python hệ thống trong `PATH`.
  - Tự động kiểm tra phiên bản $\ge 3.10$ và tự động cài đặt gói thiếu từ `requirements.txt`.
  - Hỗ trợ các cờ dòng lệnh `--version`, `--check`, `--dry-run`, bẫy mã lỗi thoát và ghi nhật ký vào `logs/launcher.log`.
- **Bộ khóa camera đơn phiên (Single-Instance IPC Mutex):**
  - Cài đặt `QLocalServer`/`QLocalSocket` trong `main.py` với tên pipe `airteach_ai_single_instance_ipc`.
  - Ngăn ngừa tình trạng người dùng mở nhiều cửa sổ làm xung đột thiết bị webcam. Phiên bản sau tự động gửi thông điệp `"WAKEUP\n"`, kéo phiên bản trước lên đầu màn hình và tự thoát với mã 0.

#### GIAI ĐOẠN 4: THIẾT KẾ BỘ CÔNG CỤ CHUYÊN ĐỀ STEM HOÀN CHỈNH
- **Xây dựng `src/ui/stem_tools.py`:**
  - *Toán học (`MathTools`):* Hệ trục tọa độ Oxy trực quan (lưới, mũi tên, vạch chia, nhãn gốc $O$) và bộ vẽ đồ thị hàm số liên tục $y=f(x)$ bằng `numpy.linspace`.
  - *Vật lý (`PhysicsTools`):* Vectơ lực $\vec{F}$, phân tích lực trực giao $\vec{F}_x, \vec{F}_y$ nét đứt và ký hiệu linh kiện điện ($R, C, E$).
  - *Hóa học (`ChemistryTools`):* Bộ định dạng hóa học Unicode ($H_2SO_4 \to H_2SO_4$, $Cu^{2+} \to Cu^{2+}$, $2H_2O \to 2H_2O$), mũi tên phản ứng 1 chiều/thuận nghịch ($\to, \rightleftharpoons$) kèm nhãn xúc tác/nhiệt độ, vòng Benzen $C_6H_6$.
  - *Sinh học (`BiologyTools`):* Ghim định vị giải phẫu/tế bào tròn kèm đường chỉ dẫn góc khuỷu và hộp chú thích văn bản.
  - *Tin học (`InformaticsTools`):* Các khối sơ đồ thuật toán chuẩn ISO (Process, Decision, I/O, Terminator) và code block phong cách IDE tối màu.
- **Tích hợp giao diện `MainWindow`:**
  - Bổ sung menu công cụ STEM động trên thanh công cụ chính và kết nối 12 sự kiện xử lý chuyên dụng.

#### GIAI ĐOẠN 5: KHẮC PHỤC LẤY MẪU QUỸ ĐẠO SMART INK DỌC ĐƯỜNG CONG BÉZIER
- **Vấn đề phát hiện:** Phương thức `MainWindow._get_strokes_in_rect` trước đó chỉ truyền một điểm đơn lẻ cho `SmartInkRecognizer`, khiến mô hình OCR không nhận được nét viết hoàn chỉnh.
- **Khắc phục:** Cập nhật thuật toán lấy mẫu: duyệt qua toàn bộ các đoạn cong Bézier cắt qua vùng chọn và lấy mẫu 4 điểm dọc theo tham số $t \in [0.0, 0.33, 0.67, 1.0]$. Tạo ra quỹ đạo nét mực liên tục, đầy đủ giúp OCR chữ và công thức toán hoạt động chính xác.

#### GIAI ĐOẠN 6: ĐO KIỂM HIỆU NĂNG THỰC NGHIỆM ĐỘC LẬP
- **Xây dựng `scripts/benchmark_suite.py`:**
  - Đo đạc thực tế 100% trên CPU không dùng GPU:
    - 1€ Filter: **179,373 điểm/giây** (độ trễ: **5.57 $\mu$s/điểm**).
    - RDP Simplification: Rút gọn **76.0% – 89.6%** điểm thừa trong $4.0 – 35.1\text{ ms}$.
    - Cubic Bézier Spline: Dựng 299 đoạn cong liên tục $C^1$ trong **10.23 ms**.
    - Shape Snapper: Line ($0.56\text{ ms}$), Circle ($1.07\text{ ms}$), Rectangle ($0.43\text{ ms}$) đạt độ tin cậy 1.0.
    - YuNet Face Detector: **5.27 ms/khung hình** (**189.9 FPS** trên CPU).
    - SFace Feature Extractor: **15.25 ms/khuôn mặt**.
  - Xuất dữ liệu nghiệm thu vào `data/benchmark_results.json` và biên soạn báo cáo `docs/AIRTEACH_PERFORMANCE_BASELINE.md`.

#### GIAI ĐOẠN 7: KIỂM THỬ HỒI QUY TOÀN HỆ THỐNG & ĐỒNG BỘ TÀI LIỆU
- **Mở rộng bộ bài kiểm thử:** Bổ sung 25 ca test chuyên sâu trong `tests/test_stem_and_tracking.py`.
- **Kết quả kiểm thử toàn diện:** **275 / 275 tests PASSED (100%)** trong 51.21s, 0 lỗi, 0 hồi quy.
- **Kiểm tra launcher:** `Run_AirTeach_AI.bat --check` và `Run_AirTeach_AI.bat --version` chạy thành công với mã thoát 0.
- **Đồng bộ hệ thống tài liệu kỹ thuật chuẩn:** Biên soạn đầy đủ 10 tài liệu kỹ thuật chuyên nghiệp trong `docs/`.
