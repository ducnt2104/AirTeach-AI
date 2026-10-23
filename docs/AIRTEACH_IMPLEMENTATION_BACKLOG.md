# AIRTEACH AI - IMPLEMENTATION BACKLOG & ENGINEERING ROADMAP
## Bảng Quản Lý Nhiệm Vụ Kỹ Thuật & Lộ Trình Phát Triển (Feature Backlog & Roadmap)

**Dự án:** AirTeach AI — Nền tảng dạy học thông minh tương tác không chạm bằng AI  
**Chu kỳ kỹ thuật:** Chiến dịch Kỹ thuật Tự trị (Autonomous Engineering Campaign v3)  
**Tài liệu tham chiếu:** `Ban_Thuyet_Minh_AirTeach_AI_Nang_Cap.docx`

---

### 1. BẢNG TIẾN ĐỘ THỰC HIỆN CÁC NHIỆM VỤ CỐT LÕI (CORE IMPLEMENTATION BACKLOG)

| Mã Nhiệm Vụ | Mức Ưu Tiên | Hạng Mục Kỹ Thuật | Mô Tả Chi Tiết Công Việc | Trạng Thái Hoàn Thành |
| :--- | :---: | :--- | :--- | :---: |
| **TSK-P0-01** | **P0** | Khắc phục lỗi văng tọa độ nét vẽ | Sửa hàm `_scene_coords` trong `MainWindow`, loại bỏ double-scaling `(cx * 1920, cy * 1080)`, kẹp tọa độ trong $[0, 1920] \times [0, 1080]$. | **HOÀN THÀNH (100%)** |
| **TSK-P0-02** | **P0** | Khắc phục vùng chết cử chỉ DRAW/POINT | Điều chỉnh ngưỡng $r_{\text{thumb}} \lessgtr 0.38$, xóa bỏ khoảng hở $0.35-0.41$ gây rớt cử chỉ về `NONE`. | **HOÀN THÀNH (100%)** |
| **TSK-P0-03** | **P0** | Nhận diện độc lập góc xoay & ngón gập | Sử dụng khoảng cách cổ tay - đầu ngón; bổ sung thuật toán kiểm tra ngón gập (curl check) cho 3 ngón còn lại. | **HOÀN THÀNH (100%)** |
| **TSK-P0-04** | **P0** | Khắc phục đường kẻ chéo khi hạ tay | `_QtCameraWorker` luôn phát tín hiệu `NONE` khi không có bàn tay để `MainWindow` tự động kết thúc nét vẽ (`_finalize_stroke()`). | **HOÀN THÀNH (100%)** |
| **TSK-P0-05** | **P0** | Hiển thị khung xương chẩn đoán thời gian thực | Vẽ trực tiếp 21 khớp, các đoạn xương, vòng trạng thái đầu ngón trỏ và HUD Mode Badge lên khung camera góc phải. | **HOÀN THÀNH (100%)** |
| **TSK-P1-01** | **P1** | Trình khởi chạy 1-Click Windows | Tạo tệp kịch bản `Run_AirTeach_AI.bat` chuẩn Windows CRLF UTF-8, tự động nhận diện `.venv`, kiểm tra dependencies và cờ `--check`. | **HOÀN THÀNH (100%)** |
| **TSK-P1-02** | **P1** | Bộ khóa camera đơn phiên (IPC Mutex) | Tích hợp `QLocalServer`/`QLocalSocket` trong `main.py` ngăn ngừa xung đột webcam khi khởi chạy nhiều phiên bản. | **HOÀN THÀNH (100%)** |
| **TSK-P1-03** | **P1** | Bộ công cụ chuyên đề STEM toàn diện | Thiết kế `src/ui/stem_tools.py` tích hợp 5 phân môn: Toán học (Oxy/Hàm số), Vật lý (Lực/Mạch điện), Hóa học (Unicode/Phản ứng/Benzen), Sinh học (Ghim chỉ dẫn), Tin học (Khối thuật toán/Code IDE). | **HOÀN THÀNH (100%)** |
| **TSK-P1-04** | **P1** | Khắc phục lấy mẫu quỹ đạo Smart Ink | Sửa `MainWindow._get_strokes_in_rect` thay thế điểm đơn bằng việc lấy mẫu 4 điểm dọc theo đường cong Bézier ($t=0.0, 0.33, 0.67, 1.0$) cho OCR. | **HOÀN THÀNH (100%)** |
| **TSK-P2-01** | **P2** | Bộ đo kiểm chuẩn hóa hiệu năng thực tế | Xây dựng `scripts/benchmark_suite.py` đo kiểm độc lập 1€ Filter, RDP, Bézier, Shape Snapper, YuNet, SFace; xuất dữ liệu JSON và Markdown. | **HOÀN THÀNH (100%)** |
| **TSK-P2-02** | **P2** | Kiểm thử tự động hóa 275 bài | Mở rộng bộ kiểm thử lên 275 bài tests, đạt tỉ lệ đậu 100%, bảo toàn nguyên tắc Zero Mock Policy. | **HOÀN THÀNH (100%)** |
| **TSK-P2-03** | **P2** | Hoàn thiện 10 tài liệu kỹ thuật chuẩn | Biên soạn toàn bộ 10 tài liệu kỹ thuật chi tiết trong thư mục `docs/` đồng bộ với thuyết minh dự án. | **HOÀN THÀNH (100%)** |

---

### 2. LỘ TRÌNH PHÁT TRIỂN NÂNG CẤO (FUTURE ENGINEERING ROADMAP)

Các tính năng nâng cao tiềm năng cho các bản cập nhật tiếp theo (v3.0+):

1. **Tương tác Đa Bàn tay Đồng thời (Multi-Hand Collaborative Teaching):**
   - Hỗ trợ phát hiện và theo dõi đồng thời 2 bàn tay qua MediaPipe.
   - Bàn tay trái: Điều khiển thanh trượt thu phóng (Zoom) hoặc chuyển đổi màu mực nhanh.
   - Bàn tay phải: Giữ vai trò viết nét (Draw) hoặc chỉ trỏ (Point).
   - Chế độ 2 giáo viên cùng tương tác trên màn hình tương tác khổ lớn.

2. **Tăng tốc Phần cứng DirectML / OpenVINO:**
   - Cung cấp tùy chọn tăng tốc phần cứng cho suy luận ONNX trên card đồ họa tích hợp Intel Iris Xe hoặc card rời NVIDIA thông qua DirectML Execution Provider.
   - Rút ngắn thời gian xử lý OCR cho các bài giảng có hàng chục trang viết phức tạp.

3. **Đồng bộ Đám mây Không gian làm việc Giáo viên (Cloud Workspace Sync):**
   - Tùy chọn đồng bộ mã hóa hai đầu (End-to-End Encrypted) các gói bài giảng `.airteach` lên Google Drive hoặc máy chủ lưu trữ nội bộ của trường học.

4. **Bộ Thu âm Giảng bài Đồng bộ Thời gian thực (Audio-Visual Lecture Recording):**
   - Tích hợp thu âm micro giảng bài đồng bộ với trục thời gian nét vẽ $t_{\text{ms}}$ của tính năng Replay, cho phép học sinh nghe lại lời giảng tương ứng với từng nét viết xuất hiện trên bảng.
