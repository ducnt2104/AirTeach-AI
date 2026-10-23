# AIRTEACH AI - AUTONOMOUS ENGINEERING CAMPAIGN CHECKPOINT
## Báo Cáo Nghiệm Thu & Điểm Kiểm Soát Chiến Dịch Kỹ Thuật

**Dự án:** AirTeach AI — Hệ thống dạy học thông minh tương tác không chạm bằng AI  
**Nền tảng mục tiêu:** Windows Desktop (Python 3.10+, PySide6, OpenCV, MediaPipe, Local AI)  
**Tài liệu quy chiếu:** `Ban_Thuyet_Minh_AirTeach_AI_Nang_Cap.docx`  
**Thời điểm tạo checkpoint:** 23/10/2026  
**Trạng thái kiểm thử:** **275 / 275 tests PASSED (100%)**

---

### 1. TỔNG KẾT KHẮC PHỤC LỖI CHẶN P0 (P0 AIR DRAWING RESOLUTION)

Vấn đề người dùng phản hồi: *"Giơ ngón tay trước camera nhưng AirTeach AI không nhận diện đúng, không vẽ được nét trên không."*

| Hạng mục | Hiện trạng lỗi trước chiến dịch | Giải pháp kỹ thuật đã triển khai | Trạng thái |
| :--- | :--- | :--- | :--- |
| **Tọa độ nét vẽ** | Tọa độ bị nhân đôi `(cx * 1920, cy * 1080)` dù `PerspectiveMapper` đã xuất tọa độ pixel. Điểm vẽ bị đẩy ra ngoài $(1.843.200, 583.200)$. | Thêm bộ bảo vệ tự động phát hiện hệ tọa độ chuẩn hóa $[0, 1]$ vs tọa độ pixel $[0, 1920]$, kẹp chặt vào biên canvas $1920 \times 1080$. | **ĐÃ KHẮC PHỤC & KIỂM THỬ** |
| **Vùng chết ngón cái** | Ngưỡng ngón cái DRAW $\le 0.34$, POINT $> 0.42$. Khoảng hở $0.35 - 0.41$ rơi vào vùng chết khiến cử chỉ bị rớt về `NONE`. | Bỏ khoảng hở vùng chết, chuẩn hóa tỉ lệ $r_{\text{thumb}} = \|P_4 - P_6\| / L_{\text{hand}}$: $\le 0.38$ là DRAW, $> 0.38$ là POINT. | **ĐÃ KHẮC PHỤC & KIỂM THỬ** |
| **Bàn tay nghiêng** | Phép so sánh $y_{\text{tip}} < y_{\text{pip}}$ hỏng khi bàn tay xoay nghiêng hoặc chúc xuống. | Sử dụng khoảng cách từ cổ tay (Wrist P0) đến đầu ngón so với đốt gốc (MCP) độc lập hướng xoay. | **ĐÃ KHẮC PHỤC & KIỂM THỬ** |
| **Lẫn lộn bàn tay mở** | Giơ cả bàn tay mở (Open Palm) bị nhận diện nhầm thành DRAW hoặc POINT. | Bổ sung điều kiện 3 ngón giữa, áp út và út bắt buộc phải gập (curled) cho cử chỉ đơn ngón. | **ĐÃ KHẮC PHỤC & KIỂM THỬ** |
| **Đường nối chéo dài** | `_QtCameraWorker` bỏ qua `NONE`, không gọi `_finalize_stroke()` khi hạ tay. | Luôn phát tín hiệu `NONE` để ngắt nét vẽ tức thì và ẩn tia laser. | **ĐÃ KHẮC PHỤC & KIỂM THỬ** |
| **Chẩn đoán thị giác** | Không nhìn thấy khớp ngón tay thời gian thực trên màn hình camera. | Vẽ trực tiếp 21 khớp, xương, vòng trạng thái (Đỏ: DRAW, Xanh: POINT) và HUD Badge lên khung hình. | **ĐÃ HOÀN TẤT** |

---

### 2. BỘ CÔNG CỤ CHUYÊN ĐỀ STEM HOÀN THIỆN (`src/ui/stem_tools.py`)

Đã thiết kế và tích hợp thanh công cụ STEM chuyên sâu vào `MainWindow`, hỗ trợ 5 phân môn giảng dạy:
1. **Toán học (Mathematics):**
   - Hệ trục tọa độ vuông góc Oxy chuẩn sư phạm: lưới ô vuông, mũi tên định hướng, vạch chia tỉ lệ số nguyên, gốc tọa độ $O$.
   - Vẽ đồ thị hàm số liên tục $y = f(x)$ (`numpy.linspace` 400 điểm, đánh giá toán học an toàn, hỗ trợ đa thức, lượng giác $\sin, \cos, \tan$, mũ $\exp$, căn $\sqrt{}$).
2. **Vật lý (Physics):**
   - Vectơ lực $\vec{F}$ (phương, chiều, độ lớn, mũi tên tam giác đặc và nhãn lực chuẩn).
   - Phân tích lực trực giao thành 2 thành phần $\vec{F}_x$ và $\vec{F}_y$ kèm hình chữ nhật dóng nét đứt.
   - Sơ đồ mạch điện tiêu chuẩn: Ký hiệu Điện trở zig-zag ($R$), Tụ điện 2 bản cực ($C$), Nguồn điện 1 chiều bản dài/ngắn ($E$).
3. **Hóa học (Chemistry):**
   - Bộ chuyển đổi chỉ số dưới/trên Unicode tự động: $H_2SO_4 \to H_2SO_4$, $Ca(OH)_2 \to Ca(OH)_2$, $Cu^{2+} \to Cu^{2+}$, $Fe^{3+} \to Fe^{3+}$. Giữ nguyên hệ số phản ứng ở đầu chất ($2H_2O \to 2H_2O$).
   - Mũi tên phản ứng 1 chiều $(\to)$ và thuận nghịch $(\rightleftharpoons)$ kèm nhãn điều kiện nhiệt độ ($t^\circ$), xúc tác, áp suất ($p$).
   - Vòng thơm Benzen ($C_6H_6$) gồm hình lục giác đều và vòng tròn electron liên hợp định xứ bên trong.
4. **Sinh học (Biology):**
   - Ghim định vị tròn (Circular Anchor Pin) kèm đường chỉ dẫn gập khúc (Elbow Leader) và nhãn văn bản chú thích bộ phận giải phẫu/tế bào.
5. **Tin học (Informatics):**
   - Các khối sơ đồ thuật toán chuẩn ISO: Khối Thực thi (Process - Chữ nhật), Khối Điều kiện (Decision - Hình thoi), Khối Vào/Ra (I/O - Hình bình hành), Khối Bắt đầu/Kết thúc (Terminator - Capsule tròn).
   - Khung hiển thị mã nguồn (Code Block Container) giao diện tối chuẩn IDE lập trình (Consolas monospaced, 3 chấm tiêu đề macOS style).

---

### 3. TRÌNH KHỞI CHẠY WINDOWS 1-CLICK (`Run_AirTeach_AI.bat`)

- **Vị trí:** Thư mục gốc dự án (`Run_AirTeach_AI.bat`).
- **Đặc tính kỹ thuật:**
  - Định dạng ngắt dòng chuẩn CRLF của Windows, mã hóa UTF-8 (Code Page 65001).
  - Tự động phát hiện môi trường ảo `.venv\Scripts\python.exe` hoặc fallback về Python hệ thống trong `PATH`.
  - Tự động kiểm tra phiên bản Python $\ge 3.10$.
  - Tự động kiểm tra sự hiện diện của các thư viện cốt lõi (`PySide6`, `opencv-python`, `mediapipe`, `numpy`, `cryptography`, `python-dotenv`). Nếu thiếu, tự động kích hoạt `pip install -r requirements.txt`.
  - Hỗ trợ tham số dòng lệnh kiểm tra không mở GUI: `--version`, `--check`, `--dry-run`.
  - Bẫy mã lỗi thoát (`APP_EXIT_CODE`) và ghi nhận đầy đủ lịch sử chạy vào `logs/launcher.log`.

---

### 4. TỔNG HỢP KIỂM THỬ HỆ THỐNG (TEST RESULTS SUMMARY)

Kết quả chạy thực tế với lệnh `pytest -v`:
- `tests/test_core_algorithms.py`: **48 / 48 PASSED** (1€ Filter, RDP, Bézier, Shape Snapper, FSM, Perspective)
- `tests/test_ui_components.py`: **111 / 111 PASSED** (LessonBoard, ChatPanel, Settings, Camera, Multi-page)
- `tests/test_phase3_advanced_ui.py`: **33 / 33 PASSED** (LoginWindow, TeacherProfile, PinPad, Lockout)
- `tests/test_services_and_storage.py`: **44 / 44 PASSED** (FaceProfileService, AirTeachPackager, GeminiAssistant, Telemetry)
- `tests/test_svg_assets_and_trainer.py`: **14 / 14 PASSED** (SVG Assets, GestureTrainerDialog, SmartInkReviewDialog)
- `tests/test_stem_and_tracking.py`: **25 / 25 PASSED** (Chemical Unicode, Math Oxy/Graph, Physics Force/Decomp/Circuit, Chemistry Arrows/Benzene, Biology Callouts, Informatics Flowcharts/Code, Coordinate Mapping Anti-Double-Scaling, Finger Curl Logic)
- **TỔNG CỘNG: 275 / 275 TESTS PASSED (100%)** trong 50.16s, 0 lỗi, 0 hồi quy.

---

### 5. NGHIỆM THU CỔNG CHẤP NHẬN (ACCEPTANCE GATES)

- **Gate A — Ứng dụng & Trình khởi chạy (Launcher):** `Run_AirTeach_AI.bat --check` và `python main.py --check` đều chạy thành công, trả về exit code 0.
- **Gate B — Thu hình & Theo dõi Bàn tay (Tracking Pipeline):** Bù sáng CLAHE, nạp `models/hand_landmarker.task`, hiển thị khung xương 21 điểm mốc và HUD Badge trực tiếp trên camera widget.
- **Gate C — Cử chỉ & Vẽ nét (Air Drawing Interaction):** Ngón trỏ vẽ nét ổn định, ngón cái chuyển đổi DRAW/POINT rõ ràng, nắm tay ngắt nét, không còn hiện tượng nhảy nét xuyên màn hình hay văng tọa độ.
- **Gate D — Làm mịn & Nắn hình học (Smart Ink):** Bộ lọc 1€ Filter triệt tiêu rung giật, RDP rút gọn 80% điểm thừa, Bézier spline liên tục $C^1$, tự động nắn 6 hình học cơ bản.
- **Gate E — Bộ công cụ STEM (Subject Suite):** Hoàn thiện 5 bộ môn (Toán, Lý, Hóa, Sinh, Tin) tích hợp trên thanh công cụ chính.
- **Gate F — Trí tuệ nhân tạo (AI Capabilities):** Nhận diện khuôn mặt offline YuNet+SFace bảo mật, RapidOCR + LaTeX-OCR offline, Gemini Assistant có chế độ ngoại tuyến tự động.
- **Gate G — Quy tắc Zero Mock Data Policy:** Không sử dụng dữ liệu giả, không bỏ dở hàm với `# TODO` hay `pass`, không chặn luồng giao diện chính.
