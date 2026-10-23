# AIRTEACH AI - HAND TRACKING & CAMERA PIPELINE DIAGNOSTICS
## Hướng Dẫn Kỹ Thuật: Thu Hình Camera, Khung Xương MediaPipe & Khử Lỗi Tọa Độ

**Dự án:** AirTeach AI — Hệ thống dạy học thông minh tương tác không chạm bằng AI  
**Mục tiêu tài liệu:** Cung cấp tài liệu kỹ thuật chi tiết về kiến trúc luồng thu hình thời gian thực, thuật toán trích xuất 21 điểm mốc 3D bàn tay qua MediaPipe, giải pháp chẩn đoán thị giác trực tiếp và các bản sửa lỗi triệt để cho vấn đề không vẽ được nét (P0 Air Drawing Defect).

---

### 1. KIẾN TRÚC TOÀN DIỆN LUỒNG DỮ LIỆU CAMERA (PIPELINE ARCHITECTURE)

```
[Webcam Hardware (DirectShow / USB)]
       │
       ▼ (OpenCV DirectShow Backend: 640x480 / 1280x720 @ 30 FPS)
[HandTrackingWorker / _QtCameraWorker (Background QThread)]
       │
       ├──► 1. Bù sáng thích ứng: CLAHE (Contrast Limited Adaptive Histogram Equalization)
       │       └── Tự động kích hoạt khi giá trị trung bình kênh V trong không gian HSV < 85
       │
       ├──► 2. Trích xuất MediaPipe HandLandmarker (21 3D Landmarks)
       │       ├── Mô hình ONNX/Task Bundle: models/hand_landmarker.task
       │       └── Chuẩn hóa tỉ lệ bàn tay: L_hand = ||P0 - P9||₂ (Wrist đến Middle MCP)
       │
       ├──► 3. Cỗ máy trạng thái FSM (HandGestureFSM)
       │       ├── Phân tích hình học: Ngón trỏ, ngón cái, kiểm tra ngón gập (curl check)
       │       ├── Bộ đệm chống rung (Debounce Counter: 3 khung hình)
       │       └── Phân loại: DRAW, POINT, PAUSE, SELECT, SWIPE_LEFT/RIGHT, CONFIRM
       │
       ├──► 4. Bộ lọc thời gian thực 1€ Filter (OneEuroFilter2D)
       │       └── Khử rung lắc cho tọa độ đầu ngón trỏ P8: fc = fc_min + beta * ||v||
       │
       ├──► 5. Vẽ khung xương chẩn đoán (Diagnostic Skeleton Overlay trực tiếp lên frame)
       │       ├── 21 Khớp ngón tay (Joints) với mã màu theo chức năng
       │       ├── 21 Đoạn xương kết nối (Skeleton bones) nét mảnh khử răng cưa
       │       ├── Vòng tròn trạng thái tại đầu ngón trỏ (Đỏ: DRAW, Xanh lục: POINT)
       │       └── Huy hiệu hiển thị chế độ cử chỉ hiện tại (HUD Mode Badge)
       │
       ├──► 6. Single-Instance Mutex (QLocalServer / QLocalSocket)
       │       └── Ngăn ngừa 2 phiên bản phần mềm cùng mở một thiết bị webcam
       │
       └──► Qt QueuedConnection Signals (Thread-Safe)
               ├── frame_ready(np.ndarray) ──► CameraWidget (Hiển thị góc dưới bên phải)
               └── gesture_event(state, cx, cy) ──► MainWindow / LessonBoard (Canvas vẽ)
```

---

### 2. PHÂN TÍCH GỐC RỄ & GIẢI PHÁP SỬA LỖI P0 (P0 AIR DRAWING RESOLUTION)

Vấn đề thực tế từ người dùng: *"Người dùng giơ một ngón tay trước camera nhưng AirTeach AI không nhận diện đúng; tính năng vẽ nét trên không (air drawing) không hoạt động."*

Đội ngũ kỹ sư đã tiến hành rà soát từng dòng mã của luồng tương tác và phát hiện 3 nguyên nhân cốt lõi cùng 1 lỗi hiển thị:

#### A. Nguyên nhân gốc rễ 1: Phóng đại tọa độ 2 lần (Double-Scaling Bug)
- **Cơ chế lỗi:** Trong `HandGestureFSM`, lớp `PerspectiveMapper` đã thực hiện chuyển đổi tọa độ chuẩn hóa $[0, 1]$ sang kích thước pixel của bảng ($1920 \times 1080$). Tuy nhiên, hàm `_scene_coords` trong `MainWindow` trước đó lại tiếp tục nhân tọa độ nhận được với `BOARD_WIDTH` (1920) và `BOARD_HEIGHT` (1080) một lần nữa.
- **Hệ quả:** Tọa độ $(960, 540)$ tại tâm màn hình bị phóng đại thành $(1.843.200, 583.200)$ — bắn văng ra ngoài không gian canvas hiển thị, khiến giáo viên không thể nhìn thấy bất kỳ nét vẽ nào dù hệ thống vẫn âm thầm tạo điểm vẽ.
- **Giải pháp dứt điểm:** Cài đặt bộ bảo vệ tự động phát hiện hệ tọa độ và kẹp biên an toàn:
  ```python
  def _scene_coords(self, cx: float, cy: float) -> tuple[float, float]:
      """Safely convert incoming coordinates to canvas pixel coordinates.
      Guards against double-scaling: cx, cy might already be [0, 1920] x [0, 1080].
      """
      if cx > 1.0 or cy > 1.0:
          bx = max(0.0, min(float(BOARD_WIDTH), cx))
          by = max(0.0, min(float(BOARD_HEIGHT), cy))
      else:
          bx = max(0.0, min(float(BOARD_WIDTH), cx * BOARD_WIDTH))
          by = max(0.0, min(float(BOARD_HEIGHT), cy * BOARD_HEIGHT))
      return bx, by
  ```

#### B. Nguyên nhân gốc rễ 2: Lỗi phân loại khi bàn tay nghiêng & Vùng chết ngón cái (Dead-Zone)
- **Cơ chế lỗi:** Thuật toán nhận diện cũ kiểm tra ngón tay vươn thẳng bằng biểu thức `tip.y < pip.y`. Khi giáo viên đứng giảng bài tự nhiên, bàn tay thường nghiêng ngang hoặc chếch nhẹ xuống dưới, khiến trục Y bị xoay và hệ thống không nhận ra ngón trỏ đang vươn thẳng.
- Đồng thời, ngưỡng ngón cái cũ quy định: $\le 0.34$ là DRAW, $> 0.42$ là POINT. Khoảng hở từ $0.35$ đến $0.41$ là một "vùng chết" (dead-zone) khiến cử chỉ bị rơi về `NONE` khi ngón cái chuyển động tự nhiên.
- **Giải pháp dứt điểm:**
  1. Sử dụng khoảng cách Euclid từ cổ tay (Wrist P0) đến đầu ngón tay so với khớp gốc (MCP P5) để kiểm tra vươn ngón độc lập 100% với góc xoay bàn tay.
  2. Bổ sung kiểm tra ngón gập (curl check) cho ngón giữa, ngón áp út và ngón út (`_is_finger_curled`), triệt tiêu việc nhận nhầm bàn tay xòe mở (Open Palm) thành cử chỉ vẽ.
  3. Xóa bỏ hoàn toàn vùng chết: Tỉ lệ khoảng cách ngón cái - ngón trỏ $r_{\text{thumb}} = \|P_4 - P_6\| / L_{\text{hand}} \le 0.38$ là `DRAW`, và $> 0.38$ là `POINT`.

#### C. Nguyên nhân gốc rễ 3: Không phát tín hiệu ngắt nét khi hạ tay
- **Cơ chế lỗi:** Luồng camera `_QtCameraWorker` trước đây bỏ qua việc phát tín hiệu `gesture_event` khi trạng thái là `GestureState.NONE`. Khi giáo viên hạ tay xuống hoặc đưa tay ra khỏi tầm nhìn camera, `_finalize_stroke()` không được gọi. Khi đưa tay lên vẽ tiếp, hệ thống nối điểm mới với điểm cũ bằng một đường kẻ chéo dài xuyên màn hình.
- **Giải pháp dứt điểm:** Luôn phát tín hiệu `GestureState.NONE` khi không thấy bàn tay để `MainWindow` tự động kết thúc nét vẽ và ẩn tia laser con trỏ ngay lập tức.

---

### 3. HỆ THỐNG MÃ MÀU HIỂN THỊ KHUNG XƯƠNG CHẨN ĐOÁN (DIAGNOSTIC OVERLAY)

Khung hình camera góc dưới bên phải được tích hợp lớp phủ chẩn đoán thời gian thực giúp giáo viên và kỹ sư giám sát tức thì:

1. **Khớp bàn tay (Landmark Joints):**
   - Vòng tròn bán kính 4px màu lục lam (#00D2FF) viền trắng cho toàn bộ 21 điểm mốc.
2. **Đoạn xương liên kết (Bones):**
   - Đường thẳng màu xanh nhạt (#00A8FF) với độ dày 2px kết nối các đốt ngón tay theo cấu trúc giải phẫu MediaPipe.
3. **Đầu ngón trỏ (Index Tip P8 - Điểm điều khiển nét vẽ):**
   - Vòng tròn kích thước lớn (bán kính 9px) kèm vòng phát quang kép.
   - **Màu Đỏ rực rỡ (#EA4335):** Trạng thái **DRAW** (Đang vẽ nét vector Smart Ink).
   - **Màu Xanh lục (#34A853):** Trạng thái **POINT** (Đang di chuyển con trỏ laser chỉ bảng).
   - **Màu Vàng (#FBBC04):** Trạng thái **SELECT** (Đang chọn vùng nội dung).
   - **Màu Xanh dương (#1A73E8):** Trạng thái **PAUSE** (Nắm tay tạm dừng / nhấc bút).
4. **Huy hiệu Chế độ (HUD Mode Badge):**
   - Khung chữ nhật bo góc bán trong suốt hiển thị trên đỉnh khung camera:
     - `MODE: DRAW (INK)` — Chế độ vẽ nét viết.
     - `MODE: POINT (LASER)` — Chế độ tia laser chỉ bảng.
     - `MODE: PAUSE (FIST)` — Chế độ nắm tay ngắt nét.
     - `MODE: SELECT (ROI)` — Chế độ chọn vùng bài học.

---

### 4. QUẢN LÝ THIẾT BỊ CAMERA & BỘ KHÓA ĐƠN PHIÊN (SINGLE-INSTANCE MUTEX)

Để tránh hiện tượng hai tiến trình cùng cố gắng truy cập webcam (dẫn đến webcam bị chiếm dụng, camera đen hoặc ứng dụng thứ hai báo lỗi):

- `main.py` khởi tạo một `QLocalServer` với định danh IPC duy nhất: `airteach_ai_single_instance_ipc`.
- Khi người dùng nhấp đúp vào `Run_AirTeach_AI.bat` trong khi ứng dụng đã đang chạy:
  1. Tiến trình thứ hai kết nối tới `QLocalServer` thông qua `QLocalSocket`.
  2. Gửi thông điệp kích hoạt `"WAKEUP\n"`.
  3. Cửa sổ ứng dụng hiện tại tự động được nâng lên trên cùng (`raise_()`), kích hoạt (`activateWindow()`) và hủy thu nhỏ (`showNormal()`).
  4. Tiến trình thứ hai thoát an toàn với mã `0` mà không chạm tới phần cứng webcam.
