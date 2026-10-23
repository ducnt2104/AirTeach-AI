# AIRTEACH AI - HAND TRACKING & CAMERA PIPELINE DIAGNOSTICS
## Hướng Dẫn Kỹ Thuật: Thu Hình Camera, Khung Xương MediaPipe & Khử Lỗi Tọa Độ

Tài liệu này cung cấp chi tiết kiến trúc chẩn đoán và quy trình xử lý hình ảnh thời gian thực của AirTeach AI, tập trung vào việc hiển thị khung xương 21 điểm mốc, trạng thái cử chỉ, và thuật toán chuyển đổi tọa độ chống lỗi lệch hình.

---

### 1. KIẾN TRÚC THU HÌNH & CHU TRÌNH DỮ LIỆU (DATA PIPELINE)

```
[Webcam Hardware]
       │
       ▼ (OpenCV DirectShow Backend: 640x480 / 1280x720 @ 30 FPS)
[HandTrackingWorker (Background QThread)]
       │
       ├──► 1. Bù sáng thích ứng: CLAHE (Contrast Limited Adaptive Histogram Equalization)
       │
       ├──► 2. Trích xuất MediaPipe HandLandmarker (21 3D Landmarks)
       │       └── Tỉ lệ chuẩn hóa: L_hand = ||P0 - P9||₂
       │
       ├──► 3. Cỗ máy trạng thái FSM (HandGestureFSM)
       │       └── Phân loại: DRAW, POINT, PAUSE, SELECT, SWIPE, CONFIRM
       │
       ├──► 4. Bộ lọc 1€ Filter (Khử rung giật thời gian thực cho đầu ngón trỏ P8)
       │
       ├──► 5. Vẽ khung xương chẩn đoán (Diagnostic Skeleton Overlay trực tiếp lên frame)
       │       ├── 21 Khớp ngón tay với mã màu theo chức năng
       │       ├── 21 Đoạn xương kết nối (Skeleton bones)
       │       ├── Vòng tròn trạng thái tại đầu ngón trỏ (Đỏ: DRAW, Xanh: POINT)
       │       └── Huy hiệu hiển thị chế độ cử chỉ hiện tại (HUD Mode Badge)
       │
       └──► Qt QueuedConnection Signals
               ├── frame_ready(np.ndarray) ──► CameraWidget (UI hiển thị camera góc phải)
               └── gesture_event(state, cx, cy) ──► MainWindow / LessonBoard (Canvas vẽ)
```

---

### 2. CHI TIẾT SỬA LỖI P0: KHỬ LỖI TỌA ĐỘ VÀ VÙNG CHẾT NGÓN TAY

#### A. Nguyên nhân gốc rễ 1: Phóng đại tọa độ 2 lần (Double-Scaling Bug)
- **Cơ chế lỗi:** `PerspectiveMapper` trong FSM đã ánh xạ tọa độ chuẩn hóa $[0, 1]$ sang kích thước pixel của bảng ($1920 \times 1080$). Tuy nhiên, hàm `_scene_coords` trong `MainWindow` trước đó lại tiếp tục nhân tọa độ nhận được với `BOARD_WIDTH` (1920) và `BOARD_HEIGHT` (1080) một lần nữa.
- **Hệ quả:** Tọa độ $(960, 540)$ bị bắn ra ngoài vũ trụ thành $(1.843.200, 583.200)$, khiến nét vẽ hoàn toàn biến mất khỏi tầm nhìn giáo viên.
- **Giải pháp khắc phục:**
  ```python
  def _scene_coords(self, cx: float, cy: float) -> tuple[float, float]:
      """Safely convert incoming coordinates to canvas pixel coordinates."""
      # cx, cy are already mapped to pixel canvas [0, BOARD_WIDTH] x [0, BOARD_HEIGHT]
      if cx > 1.0 or cy > 1.0:
          bx = max(0.0, min(float(BOARD_WIDTH), cx))
          by = max(0.0, min(float(BOARD_HEIGHT), cy))
      else:
          bx = max(0.0, min(float(BOARD_WIDTH), cx * BOARD_WIDTH))
          by = max(0.0, min(float(BOARD_HEIGHT), cy * BOARD_HEIGHT))
      return bx, by
  ```

#### B. Nguyên nhân gốc rễ 2: Lỗi phân loại khi bàn tay nghiêng và Vùng chết ngưỡng ngón cái
- **Cơ chế lỗi:** Thuật toán cũ kiểm tra ngón tay vươn thẳng bằng `tip.y < pip.y`. Khi bàn tay giáo viên nghiêng ngang hoặc hướng xuống, trục Y đảo ngược dẫn đến hệ thống không nhận ra ngón trỏ đang giơ lên. Ngoài ra, điều kiện `DRAW` yêu cầu khoảng cách ngón cái $\le 0.34$, trong khi `POINT` yêu cầu $> 0.42$. Khoảng hở từ $0.35$ đến $0.41$ là "vùng chết" (dead-zone) khiến cử chỉ bị rơi về `NONE`.
- **Giải pháp khắc phục:**
  1. Sử dụng khoảng cách từ cổ tay (Wrist P0) đến đầu ngón tay so với đốt khớp (MCP P5) để kiểm tra vươn ngón độc lập với góc xoay bàn tay.
  2. Bổ sung kiểm tra ngón giữa, ngón áp út và ngón út bắt buộc phải gập (curled) để phân biệt chính xác với bàn tay mở tự nhiên (Open Palm).
  3. Xóa bỏ vùng chết: khoảng cách ngón cái $\le 0.38$ là `DRAW`, $> 0.38$ là `POINT`.

#### C. Nguyên nhân gốc rễ 3: Không ngắt nét vẽ khi hạ tay
- **Cơ chế lỗi:** `_QtCameraWorker` trước đây bỏ qua việc phát tín hiệu `gesture_event` khi trạng thái là `GestureState.NONE`. Do đó, khi giáo viên hạ tay xuống, nét vẽ không được kết thúc (`_finalize_stroke()` không chạy), dẫn đến khi đưa tay lên lại sẽ xuất hiện một đường nối chéo khổng lồ xuyên qua màn hình.
- **Giải pháp khắc phục:** Luôn luôn phát tín hiệu `GestureState.NONE` khi không có bàn tay để `MainWindow` tự động gọi `_finalize_stroke()` và ẩn tia laser tức thì.

---

### 3. HỆ THỐNG MÃ MÀU HIỂN THỊ KHUNG XƯƠNG CHẨN ĐOÁN

Trên khung hình camera trực tiếp tại góc dưới bên phải, hệ thống vẽ thời gian thực:
- **Khớp bàn tay (Landmarks):** Vòng tròn bán kính 4px màu lục lam (#00D2FF).
- **Đốt ngón trỏ (Index Finger):** Vòng tròn dày 2px với tâm sáng.
- **Đầu ngón trỏ (Index Tip P8):**
  - **Màu Đỏ (#EA4335):** Đang ở chế độ vẽ nét (**DRAW** - Smart Ink).
  - **Màu Xanh lục (#34A853):** Đang ở chế độ con trỏ (**POINT** - Laser Pointer).
  - **Màu Vàng (#FBBC04):** Đang ở chế độ chọn vùng (**SELECT**).
  - **Màu Xanh lam (#1A73E8):** Đang ở chế độ tạm dừng/nắm tay (**PAUSE**).
- **Huy hiệu trạng thái (HUD Mode Badge):** Khung chữ nhật bán trong suốt hiển thị tên cử chỉ thời gian thực (VD: `MODE: DRAW (INK)`, `MODE: POINT (LASER)`).
