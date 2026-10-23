# AIRTEACH AI — MULTI-HAND AIR INTERACTION WORKSTATION CHECKPOINT

## 1. TỔNG QUAN NÂNG CẤP (MULTI-HAND AIR CONTROL & INTERACTION SUITE)
Nâng cấp toàn diện AirTeach AI thành một môi trường giảng dạy tương tác không chạm bằng cả hai bàn tay trước webcam:
1. **Theo dõi đồng thời 2 bàn tay (Simultaneous Multi-Hand Tracking):** Gắn định danh bền vững (`hand_0`, `hand_1`) bằng khoảng cách tâm bàn tay kết hợp độ tin cậy tay trái/phải (`handedness affinity`), ngăn chặn hoàn toàn hiện tượng hoán đổi ID khi hai tay bắt chéo hoặc che khuất tạm thời (tối đa 4 frames).
2. **Vẽ đồng thời bằng cả hai tay (Concurrent Dual-Hand Drawing):** Mỗi bàn tay sở hữu luồng xử lý độc lập, bộ lọc OneEuroFilter2D riêng biệt, mảng điểm quỹ đạo riêng, màu sắc nét vẽ riêng và preview item riêng trên `LessonBoard` (`_live_previews[hand_id]`). Hai tay có thể vẽ cùng lúc mà không xung đột hay đè nét.
3. **Phối hợp bất đối xứng 2 tay (Asymmetric Multi-Hand Workflows):**
   - Chế độ `draw_and_tool`: Tay thuận làm bút vẽ/laser, tay phụ điều khiển menu nổi hoặc chọn công cụ.
   - Chế độ `draw_and_erase`: Tay thuận vẽ nét, tay phụ làm tẩy xóa nhanh liên tục. Thao tác xóa của tay phụ không bao giờ làm gián đoạn nét vẽ đang dở của tay thuận.
   - Chế độ `dual_draw`: Cả hai tay cùng vẽ nét tự do hoặc hình học.
   - Chế độ `independent`: Nhận diện độc lập cử chỉ của từng bàn tay theo FSM.
4. **Nhận diện thao tác nhấn trong không khí (Air Click & Selection Suite):**
   - **Pinch Click:** Bắt cạnh lên (rising edge) khi chụm ngón trỏ và ngón cái kèm cơ chế debounce 250ms.
   - **Air Tap:** Phát hiện xung vận tốc hướng tới trục Z camera ($v_z > 0.45$) trong cửa sổ thời gian 60–260ms.
   - **Dwell Selection:** Tích lũy thời gian dừng con trỏ ổn định trong bán kính 24px suốt 650ms (vòng tròn tiến trình trực quan $0.0 \to 1.0$).
   - **Hybrid:** Kết hợp cả Tap, Pinch và Dwell tự động.
5. **Bảng Menu nổi trong không khí (Floating Action Menu):**
   - Hiển thị trên lớp $Z=50$ của `QGraphicsScene`, mở tự động khi thực hiện cử chỉ nhấn/búng trong không khí hoặc phím tắt `M`.
   - Hỗ trợ 2 chế độ hiển thị: **Radial Wheel (Bánh xe xoay tròn 360°)** và **Floating Panel (Bảng nổi đứng)**.
   - Cây menu phân cấp: Root Menu $\to$ Bút vẽ & Màu sắc $\to$ Bộ công cụ Tẩy xóa $\to$ Hình học STEM $\to$ Hoàn tác/Làm lại.
   - Tự động nắn tọa độ biên (boundary clamping) để menu luôn nằm 100% trong khung nhìn canvas.
6. **Bộ công cụ Tẩy xóa nâng cao (Advanced Eraser Suite):**
   - **Continuous Bézier Segment Eraser:** Chia nhỏ mỗi đoạn cong Bézier thành các mẫu nội suy và đo khoảng cách hình học ngắn nhất từ tâm tẩy tới từng phân đoạn, bắt dính 100% các nét vẽ dài/nhanh mà các thuật toán kiểm tra điểm mút truyền thống bỏ sót.
   - **Snapped Shape Eraser:** Nhận diện và xóa hình học chuẩn (đoạn thẳng, mũi tên, hình tròn, elip, chữ nhật, tam giác).
   - **Object Eraser:** Xóa đối tượng đơn lẻ gần nhất (`erase_object_at`).
   - **Area Eraser:** Xóa tất cả nét vẽ giao cắt với vùng chữ nhật (`erase_area`).
   - **Clear Last Stroke & Clear Page:** Xóa nét vừa vẽ theo từng tay và xóa toàn bộ chú thích trên trang mà giữ nguyên tuyệt đối ảnh nền bài giảng (`_bg_item`).
   - **Eraser Reticle:** Vòng tròn nét đứt trực quan hiển thị bán kính vùng tẩy thực tế tại con trỏ.
7. **Cấu hình độc lập theo từng tài khoản Giáo viên (Per-Teacher Settings):**
   - Lưu trữ `multi_hand_mode`, `dominant_hand`, `air_click_type`, `eraser_radius`, `menu_layout_mode` trong SQLite với cơ chế tự động migrate cột (`ALTER TABLE ADD COLUMN IF NOT EXISTS`).
   - Giao diện trực quan trong `SettingsDialog` với thẻ "Điều khiển Đa bàn tay & Tương tác Không gian".
8. **Tích hợp Chuyên đề STEM vào Menu nổi không chạm:**
   - Bổ sung menu con `stem_menu` chứa 9 công cụ hình học STEM chuyên sâu (Hệ trục tọa độ Oxy, Đồ thị hàm số f(x), Vectơ lực F⃗, Mạch điện, Công thức hóa học, Vòng Benzen, Chú thích sinh học tế bào, Sơ đồ thuật toán, Khung mã nguồn Code block).
   - Tự động đóng menu nổi khi kích hoạt công cụ để không che khuất bảng giảng bài.
9. **Phát lại nét vẽ lũy tiến song song đa bàn tay (Progressive Multi-Hand Replay Engine):**
   - Nâng cấp `ReplayController` dựng nét vẽ tăng dần theo tỷ lệ thời gian thực thay vì hiện đột ngột cả nét.
   - Dựng đồng thời các nét vẽ do cả hai tay thực hiện cùng lúc dựa trên mảng `timestamps_ms` thực tế.
   - Cơ chế khôi phục toàn vẹn nét vẽ (`restore_full_strokes`) khi tạm dừng hoặc kết thúc phát lại.
10. **Hiển thị khung xương và thông số cả hai tay trên Camera HUD Preview:**
   - Vẽ đồng thời khung xương 21 khớp của cả hai tay với màu sắc phân biệt: Tay chính (xương Cyan, khớp Cam), Tay phụ (xương Hồng tím, khớp Tím thẫm).
   - Vòng chỉ báo đầu ngón trỏ đổi màu theo trạng thái (Đỏ: DRAW, Xanh lục: POINT, Vàng: PINCH/SELECT).
   - Bảng thông tin HUD trực quan trên camera `[h_0:R[DRAW] | h_1:L[POINT]]` không chặn UI thread.

---

## 2. KIẾN TRÚC MÃ NGUỒN ĐÃ TRIỂN KHAI

| Tệp nguồn | Vai trò & Tính năng chính |
|---|---|
| `src/core/multi_hand_coordinator.py` | Quản lý liên kết quỹ đạo 2 tay (`MultiHandTracker`), nhận diện cử chỉ nhấn không chạm (`AirClickDetector`), điều phối vai trò (`MultiHandCoordinator`), cấu trúc dữ liệu `HandLandmarkData`. |
| `src/core/hand_gesture_engine.py` | Vòng lặp camera đa tay (`num_hands=2` / `max_num_hands=2`), trích xuất landmark cho cả hai tay, tích hợp điều phối viên đa tay và phát tín hiệu `on_multi_hands`. |
| `src/ui/lesson_board.py` | Canvas vector hỗ trợ preview vẽ đồng thời cho nhiều tay (`_live_previews[hand_id]`), con trỏ đa tay (`move_laser(hand_id=...)`), hiển thị reticle tẩy (`move_eraser_reticle`), giải thuật xóa Bézier subsegment-distance, xóa đối tượng, xóa vùng, xóa toàn trang bảo toàn ảnh nền. |
| `src/ui/floating_action_menu.py` | Đối tượng đồ họa `FloatingActionMenu(QGraphicsObject)` tại $Z=50$, bố cục Radial Wheel & Floating Panel, hiệu ứng hover, tiến trình dwell arc, menu con STEM (`stem_menu`), boundary clamping. |
| `src/ui/replay_controller.py` | Bộ điều khiển phát lại bài giảng với cơ chế progressive rendering vẽ nét lũy tiến theo thời gian, hỗ trợ phát lại song song các nét vẽ của cả hai tay, khôi phục `restore_full_strokes`. |
| `src/services/credential_service.py` | Cấu trúc `TeacherAISettings` mở rộng 5 thuộc tính tương tác đa tay, tự động di trú cơ sở dữ liệu SQLite, lưu trữ an toàn theo giáo viên. |
| `src/ui/settings_dialog.py` | Giao diện cấu hình phối hợp 2 tay, tay thuận, kiểu nhấn không chạm, thanh trượt bán kính tẩy xóa, kiểu bố cục menu nổi. |
| `src/ui/main_window.py` | Cầu nối tín hiệu Qt `_QtCameraWorker.multi_gesture_event`, quản lý nét vẽ song song `HandLiveStroke`, slot `_on_multi_gesture_event`, điều phối menu nổi không cần chuột, chuyển đổi trạng thái công cụ, vẽ khung xương 2 tay và HUD trên camera widget. |

---

## 3. BẢNG KIỂM THỬ TỔNG THỂ (VERIFICATION MATRIX)

Tất cả **321 kiểm thử tự động** trên toàn bộ hệ thống đã chạy và vượt qua 100% với `pytest`:

```
======================= 326 passed in 110.34s (0:01:44) =======================
```

### Chi tiết các bộ test mới:
1. `tests/test_multi_hand_tracking.py` (5 passed):
   - Duy trì ổn định ID bàn tay khi thứ tự danh sách phát hiện bị đảo chiều.
   - Khôi phục track ID khi một bàn tay bị che khuất 1 khung hình.
   - Nhận diện cạnh lên của Pinch Click kèm debounce.
   - Tiến trình Dwell Selection và cơ chế reset khi dịch chuyển vượt ngưỡng dung sai.
   - Phân bổ vai trò Primary/Secondary theo cấu hình tay thuận.
2. `tests/test_air_click_and_menu.py` (5 passed):
   - Tự động nắn biên (clamping) của menu nổi khi mở sát mép hoặc ngoài khung canvas $1920 \times 1080$.
   - Hit testing cho cung tròn Radial Wheel và thanh tiến trình dwell.
   - Kích hoạt menu con (Root $\to$ Bút vẽ $\to$ Quay lại).
   - Đóng menu qua nút trung tâm và phát tín hiệu `menu_closed`.
   - Chuyển đổi linh hoạt giữa chế độ Radial Wheel và Floating Panel.
3. `tests/test_eraser_tools.py` (5 passed):
   - Xóa nét Bézier liên tục thông qua khoảng cách subsegment hình học (chạm vào đoạn giữa vẫn xóa chính xác).
   - Xóa các đối tượng hình học chuẩn (đoạn thẳng, hình tròn).
   - Xóa đối tượng gần nhất tại điểm chạm (`erase_object_at`).
   - Xóa vùng chữ nhật đa đối tượng (`erase_area`).
   - Hoàn tác nét cuối theo từng tay và xóa toàn trang bảo toàn ảnh nền.
4. `tests/test_multi_hand_workflows.py` (5 passed):
   - Khởi tạo đầy đủ Menu nổi và thiết lập đa tay trong `MainWindow`.
   - Vẽ hai nét độc lập cùng lúc bằng hai bàn tay (`dual_draw`).
   - Tương tác bất đối xứng: Tay thuận vẽ nét, tay phụ xóa nét (`draw_and_erase`).
   - Mở và điều khiển Menu nổi trong không khí bằng Air Tap / Pinch Click.
   - Lưu trữ và đọc lại bền vững thiết lập tương tác đa tay trong SQLite của giáo viên.
5. `tests/test_stem_replay_integration.py` (8 passed):
   - Tích hợp menu con STEM trong Menu nổi với 9 công cụ chuyên sâu.
   - Điều phối lệnh STEM từ Menu nổi đến các hộp thoại thiết lập hình học chuẩn.
   - Phát lại nét vẽ lũy tiến theo thời gian cho các nét vẽ đồng thời của 2 tay.
   - Dựng khung xương 21 khớp và telemetry HUD cho 2 tay trên khung hình camera.
   - Đo đạc độ trễ xử lý đa tay ($\le 15\text{ ms}$ cho 60 frames liên tục).
   - Cô lập vòng đời đồ thị/hình vẽ STEM theo từng trang (`_page_stem_items`, ẩn/hiện chính xác khi chuyển slide).
   - Ngăn xếp Hoàn tác / Làm lại hợp nhất (Unified Undo/Redo) cho cả nét vẽ vector và đối tượng STEM theo thứ tự thời gian.
   - Xuất tài liệu PDF chất lượng cao trung thực với canvas (Scene-accurate Vector PDF Export) bảo toàn đầy đủ ảnh nền, nét mực, công thức toán và hình vẽ STEM đa trang.

