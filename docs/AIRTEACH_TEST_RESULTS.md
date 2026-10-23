# AIRTEACH AI - COMPREHENSIVE TEST SUITE EXECUTION REPORT
## Báo Cáo Kết Quả Kiểm Thử Toàn Diện Hệ Thống (Automated Test Execution Results)

**Dự án:** AirTeach AI — Hệ thống dạy học thông minh tương tác không chạm bằng AI  
**Thời điểm thực thi:** 23/10/2026 - 24/10/2026  
**Môi trường thực thi:** Microsoft Windows 11 (AMD64), Python 3.15.0rc1, PySide6 6.12.0, OpenCV 5.0.0.93, MediaPipe 1.1.1, SciPy 1.18.1, Keyring 25.7.0  
**Khung kiểm thử:** `pytest` 9.1.1 kết hợp `pytest-qt` 4.5.0  
**Kết quả chung cuộc:** **579 / 579 BÀI TEST VƯỢT QUA THÀNH CÔNG (100% PASSED) TRÊN 39 BỘ KIỂM THỬ**

---

### 1. BẢNG TỔNG HỢP KẾT QUẢ THEO TỪNG BỘ KIỂM THỬ

| Mã Suite | Tệp Mã Nguồn Kiểm Thử | Mục tiêu kiểm thử | Số Lượng Bài Test | Trạng Thái | Thời Gian Chạy | Tỉ Lệ Đạt |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **SUITE-01** | `tests/test_adaptive_vision.py` | Pipeline thị giác thích ứng, Gamma LUT, CLAHE in-place, Denoise | 7 | **PASSED** | 2.06s | **100%** |
| **SUITE-02** | `tests/test_ai_workbench.py` | Trạm soạn thảo giáo án AI, streaming, chèn bảng | 3 | **PASSED** | 0.45s | **100%** |
| **SUITE-03** | `tests/test_air_click_and_menu.py` | Menu nổi không chạm (Radial/Panel), clamping, hit-test | 5 | **PASSED** | 1.20s | **100%** |
| **SUITE-04** | `tests/test_api_security.py` | Cách ly khóa API, mã hóa WinVaultKeyring, chống va chạm namespace | 6 | **PASSED** | 0.65s | **100%** |
| **SUITE-05** | `tests/test_core_algorithms.py` | 1€ Filter, RDP (NumPy vectorized), Bézier, Nắn hình học Shape Snapper, Cử chỉ Air Swipe | 50 | **PASSED** | 2.24s | **100%** |
| **SUITE-06** | `tests/test_eraser_tools.py` | Bộ công cụ tẩy xóa (nét Bézier, đối tượng, vùng chữ nhật, toàn trang) | 5 | **PASSED** | 1.45s | **100%** |
| **SUITE-07** | `tests/test_gesture_modes.py` | Chế độ vẽ đôi (Air Pen vs Pinch), safe tracking gap | 3 | **PASSED** | 0.42s | **100%** |
| **SUITE-08** | `tests/test_multi_hand_tracking.py` | Theo dõi hai tay song song, hoán đổi ID, pinch debounce, dwell | 5 | **PASSED** | 1.15s | **100%** |
| **SUITE-09** | `tests/test_multi_hand_workflows.py` | Luồng làm việc 2 tay (vẽ đôi, vẽ và xóa, điều khiển menu nổi) | 6 | **PASSED** | 28.32s | **100%** |
| **SUITE-10** | `tests/test_phase3_advanced_ui.py` | Đăng nhập khuôn mặt, PIN, thư viện bài giảng | 38 | **PASSED** | 13.88s | **100%** |
| **SUITE-11** | `tests/test_selection_and_cache.py` | Chọn đối tượng không chạm, kéo thả biến đổi, LRU Pixmap Cache, .airteach | 11 | **PASSED** | 8.20s | **100%** |
| **SUITE-12** | `tests/test_services_and_storage.py` | SQLite, Fernet, Document Manager, Gemini Assistant, CLI Flags (--help/--check) | 80 | **PASSED** | 40.39s | **100%** |
| **SUITE-13** | `tests/test_stem_and_tracking.py` | Công cụ STEM (Tam giác, Đường tròn, Cung góc, Mạch điện, Hóa hữu cơ) | 43 | **PASSED** | 14.77s | **100%** |
| **SUITE-14** | `tests/test_stem_replay_integration.py` | STEM menu, Cổng logic, Mũi tên luồng, Xoắn kép DNA, Khung Punnett, Quang hình, Phân tử 3D, Replay song song, HUD, PDF, Undo/Redo | 23 | **PASSED** | 54.13s | **100%** |
| **SUITE-15** | `tests/test_svg_assets_and_trainer.py` | Tài nguyên SVG, Gesture Trainer, Perspective Calibration | 14 | **PASSED** | 3.50s | **100%** |
| **SUITE-16** | `tests/test_ui_components.py` | LessonBoard, MainWindow, SettingsDialog, CameraWidget, CameraDiagnosticsDialog, Grid & Spacing | 83 | **PASSED** | 15.00s | **100%** |
| **SUITE-17** | `tests/test_performance_and_stress.py` | Benchmark 1000-pt Bézier (2.67ms), điều hướng 50 trang, 300 nét vẽ, menu 60 FPS | 6 | **PASSED** | 7.87s | **100%** |
| **SUITE-18** | `tests/test_adversarial_redteam.py` | Chống ZeroDivisionError, đầu vào suy biến, điểm NaN/None, định dạng hóa học | 5 | **PASSED** | 1.94s | **100%** |
| **SUITE-19** | `tests/test_perspective_replay_and_package.py` | Phối cảnh 4 góc, Replay 0.5x, Gói .airteach (Nét Bézier + LaTeX + Text hoàn chỉnh) | 13 | **PASSED** | 8.18s | **100%** |
| **SUITE-20** | `tests/test_measurement_tools.py` | Thước kẻ centimet, Thước đo độ 180°, Đèn rọi Spotlight, Kính lúp (Virtual Loupe), Đồng hồ đếm ngược lớp học (Presentation Timer) | 13 | **PASSED** | 23.34s | **100%** |
| **SUITE-21** | `tests/test_audio_feedback.py` | Động cơ âm thanh tổng hợp PCM WAV, phản hồi tương tác (Click, Snap, Page, Erase, Timer Chime, Undo), SettingsDialog | 4 | **PASSED** | 1.06s | **100%** |
| **SUITE-22** | `tests/test_desktop_overlay.py` | Chế độ vẽ đè Desktop trong suốt (Transparent Desktop Overlay Mode), thanh công cụ nổi acrylic draggable (DesktopOverlayBar), F11 toggle, LessonBoard transparent mode | 4 | **PASSED** | 7.89s | **100%** |
| **SUITE-23** | `tests/test_gesture_guide_and_bimanual.py` | Thẻ hướng dẫn cử chỉ tay trực quan (GestureGuideOverlay), thắp sáng thẻ theo thời gian thực, điều chế 2 tay (Bimanual) khẩu độ Đèn rọi & độ phóng đại Kính lúp | 3 | **PASSED** | 8.25s | **100%** |
| **SUITE-24** | `tests/test_export_and_chalkboard.py` | Động cơ xuất ảnh bài giảng độ nét cao (1080p, 4K UHD, PNG/JPEG), Lọc lớp phủ tạm thời, Chế độ Bảng xanh phấn (Chalkboard), Vở 4 ô ly tiểu học (Vietnamese Ô ly), Bảng tối | 17 | **PASSED** | 17.84s | **100%** |
| **SUITE-25** | `tests/test_slide_drawer_and_sticky_notes.py` | Ngăn kéo danh sách slide bài giảng (SlideThumbnailDrawer), Điều hướng ngẫu nhiên, Thêm/nhân bản/xóa slide, Thẻ ghi chú dán Acrylic nổi (StickyNoteItem, 5 chủ đề màu pastel) | 11 | **PASSED** | 8.96s | **100%** |
| **SUITE-26** | `tests/test_laser_trail_and_calibration.py` | Động cơ vệt sáng laser phát quang ("Hiệu ứng pháo hoa"), tự phân rã theo thời gian, 0% CPU idle, Thuật sĩ hiệu chỉnh cử chỉ & khoảng cách thực tế (GestureCalibrationDialog) | 15 | **PASSED** | 10.42s | **100%** |
| **SUITE-27** | `tests/test_zoom_and_sound_preview.py` | Bộ điều khiển thu phóng bảng vẽ (CanvasZoomController 0.25x - 4.0x), widget HUD nổi, Ctrl+Wheel, kéo bảng Pan mode, cử chỉ 2 tay chụm phóng đại (Bimanual Pinch Zoom), thử âm thanh SettingsDialog | 10 | **PASSED** | 14.02s | **100%** |
| **SUITE-28** | `tests/test_interactive_logic_gates.py` | Cổng logic số mô phỏng tương tác (AND, OR, NOT, NAND, NOR, XOR), lật bit ngõ vào (0/1), ngõ ra phát sáng halo, bảng chân lý BCL HUD, bắt dính pin từ tính, dây dẫn vuông góc Manhattan | 6 | **PASSED** | 7.98s | **100%** |
| **SUITE-29** | `tests/test_classroom_metronome.py` | Máy đếm nhịp lớp học & Trợ lý luyện đọc thơ/tiết tấu âm nhạc (40-240 BPM, LED pulse, âm thanh tổng hợp woodblock/chime, 2/4-6/8, chạm không chạm air-click) | 10 | **PASSED** | 15.63s | **100%** |
| **SUITE-30** | `tests/test_canvas_minimap.py` | Bản đồ toàn cảnh thu nhỏ (Canvas Mini-Map HUD 16:9, tỷ lệ 1:12, khung viewport phát sáng thời gian thực, click/drag cuộn nhanh bảng vẽ, điều khiển không chạm) | 8 | **PASSED** | 12.15s | **100%** |
| **SUITE-31** | `tests/test_laser_customization.py` | Tùy biến màu sắc & kích thước tia Laser (Đỏ Ruby, Lục Bảo, Cyan, Hổ Phách, Tím), điều chỉnh bán kính 4-36px, vệt sáng pháo hoa, phím tắt Ctrl+L, SettingsDialog | 7 | **PASSED** | 8.45s | **100%** |
| **SUITE-32** | `tests/test_virtual_compass.py` | Compa hình học vẽ đường tròn tương tác (Virtual Compass, R=1-12cm, chia vạch mm, góc quay 0-360°, dập đường tròn vector, air click, Ctrl+Shift+C) | 10 | **PASSED** | 8.12s | **100%** |
| **SUITE-33** | `tests/test_interactive_circuit_wire.py` | Dây dẫn mạch logic Manhattan 90° (Interactive Circuit Wire), truyền tín hiệu logic 0/1 thời gian thực, halo neon ngõ ra, cập nhật tọa độ tự động khi kéo cổng, air click | 8 | **PASSED** | 3.56s | **100%** |
| **SUITE-34** | `tests/test_cartesian_coordinate_hud.py` | HUD hiển thị tọa độ Oxy thời gian thực (Cartesian Coordinate HUD), chuyển đổi trục Y ngược toán học, bắt dính lưới (Snap to Grid), tâm O và trục Ox/Oy, air click | 7 | **PASSED** | 8.58s | **100%** |
| **SUITE-35** | `tests/test_interactive_angle_tool.py` | Thước đo góc hình học động (Interactive Angle Measure Item), tính góc BAC tự động theo tích vô hướng, tự động bắt góc vuông 90°, nhãn phân loại (nhọn/vuông/tù/bẹt), dập vector lên bảng | 8 | **PASSED** | 8.02s | **100%** |
| **SUITE-36** | `tests/test_interactive_triangle_tool.py` | Định lý tam giác hình học động & Đường thẳng Euler (Interactive Triangle Item), tính $a,b,c,P,S,R,r$, các tâm $G,O,H,I$, định lý Euler $HG=2GO$, thẻ HUD định lý, dập vector | 8 | **PASSED** | 9.90s | **100%** |
| **SUITE-37** | `tests/test_interactive_function_plotter.py` | Đồ thị hàm số & Tiếp tuyến đạo hàm động (Interactive Function Plotter), tính đạo hàm giải tích số $f'(x_0)$, tiếp tuyến động, thẻ HUD giải tích (đồng biến/nghịch biến/cực trị), dập vector | 8 | **PASSED** | 8.62s | **100%** |
| **SUITE-38** | `tests/test_interactive_physics_incline.py` | Phân tích hệ lực mặt phẳng nghiêng (Interactive Inclined Plane), góc dốc $\alpha$, trọng lực $\vec{P}$, phản lực $\vec{N}$, ma sát $\vec{F}_{ms}$, lực kéo $\vec{F}_{net}$, gia tốc $a$, trượt/cân bằng, dập vector | 8 | **PASSED** | 9.15s | **100%** |
| **SUITE-39** | `tests/test_interactive_optics_refraction.py` | Khúc xạ ánh sáng & Phản xạ toàn phần (Interactive Snell's Law Optics), góc tới $i$, khúc xạ $r$, phản xạ $i'$, góc tới hạn $i_{gh}$, preset chiết suất $n_1/n_2$, hoán đổi môi trường, dập vector | 8 | **PASSED** | 7.92s | **100%** |
| **TỔNG KẾT** | **Toàn Bộ Hệ Thống** | **39 Bộ kiểm thử chuyên sâu toàn diện** | **579** | **ALL PASSED** | **370.7s** | **100.0%** |

---

### 2. CHI TIẾT CÁC BỘ KIỂM THỬ TRỌNG YẾU MỚI NHẤT

#### 2.1. Thẻ Hướng Dẫn Cử Chỉ Tay Nổi & Điều Chế Hai Tay Song Song (`tests/test_gesture_guide_and_bimanual.py`)
- `test_gesture_card_properties_and_states`: Thẻ hiển thị cử chỉ bo tròn 10px viền phát sáng xanh lá dạ quang `#00ff88` khi được nhận diện, mô tả chi tiết thao tác và phím tắt.
- `test_gesture_guide_overlay_lifecycle_and_drag`: Thẻ hướng dẫn tổng thể 6 cử chỉ cốt lõi (Air Pen ✌️, Laser Pointer ☝️, Tẩy xóa/Menu ✋, Pinch Click 👌, Tạm dừng ✊, Lật trang 👈👉). Thử nghiệm kéo thả chuột tự do trên màn hình và tự động kích hoạt viền thẻ tương ứng khi bàn tay di chuyển.
- `test_main_window_gesture_guide_and_bimanual_tuning`:
  - Kiểm tra phím tắt `F1` và nút thanh công cụ bật/tắt thẻ hướng dẫn.
  - Kiểm tra tính năng **điều chế hai tay đồng thời (Bimanual air interaction)**: Khi giáo viên đang bật Đèn rọi Spotlight (`SPOTLIGHT`) hoặc Kính lúp (`LOUPE`), tay thuận trỏ điểm cần nhấn mạnh trên bảng, còn tay phụ giơ lên/xuống trước camera để tự động phóng to/thu nhỏ khẩu độ đèn rọi $[40\text{px}, 350\text{px}]$ hoặc thay đổi hệ số phóng đại kính lúp $[1.5\times, 4.0\times]$ theo thời gian thực mà không cần chạm chuột.

#### 2.2. Chế Độ Vẽ Đè Màn Hình Desktop Trong Suốt (`tests/test_desktop_overlay.py`)
- `test_desktop_overlay_bar_properties_and_signals`: Thanh công cụ nổi acrylic bán trong suốt bo tròn 18px với các cờ `FramelessWindowHint` và `WindowStaysOnTopHint`. Kiểm tra phát tín hiệu và chuyển đổi trạng thái công cụ (Bút, Tẩy, Laser, Đèn rọi, Kính lúp, Hẹn giờ, Hoàn tác, Xóa hết, Thoát chế độ).
- `test_desktop_overlay_bar_dragging_and_paint`: Thử nghiệm di chuột kéo thả tự do thanh công cụ nổi trên màn hình và render nền acrylic chống giật.
- `test_lesson_board_desktop_overlay_mode`: Chuyển đổi cọ vẽ nền của `LessonBoard` sang trong suốt hoàn toàn (`Qt.GlobalColor.transparent`) và ẩn ảnh nền bài giảng, cho phép nhìn xuyên qua màn hình Desktop, đồng thời vô hiệu hóa vẽ lưới tọa độ Oxy khi ở chế độ Desktop.
- `test_main_window_desktop_overlay_integration`: Kiểm tra vòng đời chuyển đổi toàn diện giữa Bảng dạy học và Desktop Overlay (ẩn/hiện thanh điều hướng sidebar, chat panel, chuyển đổi cờ cửa sổ, gán phím tắt `F11`, thanh công cụ trên cùng và liên kết menu nổi không chạm).

#### 2.3. Động Cơ Âm Thanh Phản Hồi Cử Chỉ Tổng Hợp PCM WAV (`src/utils/audio_feedback.py`)
- `test_synthesize_pcm_wav_header_and_structure`: Tổng hợp sóng PCM WAV 16-bit mono trực tiếp trong bộ nhớ RAM, đường bao suy giảm hàm mũ (exponential decay), điều chế tần số (pitch bend) và tiêu chuẩn container RIFF/WAVE.
- `test_audio_feedback_manager_singleton_and_controls`: Singleton `AudioFeedbackManager.instance()`, điều chỉnh âm lượng kẹp $[0.0, 1.0]$, bật/tắt tiếng (mute).
- `test_audio_feedback_manager_play_methods_safe_execution`: Phát âm thanh bất đồng bộ `winsound.PlaySound` (SND_ASYNC | SND_MEMORY) hoàn toàn không chặn UI thread, bảo vệ an toàn ngoại lệ khi thiết bị âm thanh bận.
- `test_settings_dialog_audio_feedback_integration`: Kiểm tra checkbox bật âm thanh cử chỉ, spinbox âm lượng %, nút "Thử âm thanh" và lưu cấu hình hồ sơ giáo viên.

#### 2.2. Xuất Ảnh Bài Giảng Độ Phân Giải Cao & Chế Độ Bảng Xanh Phấn / Vở 4 Ô Ly (`tests/test_export_and_chalkboard.py`)
- `test_board_theme_constants` & `test_theme_configs_integrity`: Xác thực 4 chế độ bảng viết (`BoardTheme.WHITE`, `BoardTheme.CHALKBOARD`, `BoardTheme.OLY`, `BoardTheme.SLATE`) với đầy đủ bảng màu nét chữ, màu lưới, chấm tọa độ, và trục Oxy.
- `test_lesson_board_set_theme` & `test_lesson_board_cycle_theme`: Kiểm tra chu kỳ xoay vòng bảng viết (`Ctrl+B`, thanh công cụ, menu nổi không chạm) và phát tín hiệu `board_theme_changed` để đồng bộ màu bút/phấn.
- `test_lesson_board_draw_background_all_themes_and_grids`: Kiểm thử dựng nền và lưới ô ly tiểu học Việt Nam (lưới 4 ô con với nét phụ siêu mảnh `subgrid_pen` và nét chính `grid_pen`) không gây lỗi ngoại lệ.
- `test_render_scene_to_image_1080p` & `test_render_scene_to_image_4k_supersampling`: Kiểm thử dựng ảnh bảng vẽ kích thước chuẩn 1080p ($1920 \times 1080$) và siêu lấy mẫu 2x 4K UHD ($3840 \times 2160$) siêu sắc nét.
- `test_render_scene_transparent_background`: Kiểm thử xuất ảnh PNG với kênh alpha trong suốt (transparent canvas).
- `test_render_scene_hides_overlays`: Tự động ẩn các thành phần tương tác tạm thời (chấm laser pointer, vòng tròn tẩy xóa reticle, đèn rọi, kính lúp) trong lúc chụp ảnh xuất tệp và tự động phục hồi sau khi kết thúc.
- `test_export_scene_to_file_png`, `test_export_scene_to_file_jpeg`, `test_export_all_pages_whiteboard`: Kiểm thử lưu ảnh thực tế xuống ổ đĩa và xuất hàng loạt toàn bộ bài giảng.
- `test_main_window_toggle_board_theme_integration`: Tích hợp toàn diện trên MainWindow, tự động chuyển màu bút sang phấn trắng khi vào bảng xanh và đổi sang mực tím than khi vào vở ô ly.

#### 2.3. Ngăn Kéo Danh Sách Slide Bài Giảng & Ghi Chú Dán Acrylic Nổi (`tests/test_slide_drawer_and_sticky_notes.py`)
- `test_slide_drawer_initialization` & `test_slide_drawer_set_pages`: Khởi tạo ngăn kéo danh sách slide (`SlideThumbnailDrawer`), sinh thumbnail tự động từ ảnh slide 16:9, đánh số thứ tự `Trang 1`, `Trang 2`.
- `test_slide_drawer_selection_signal`: Nhấp chuột chọn slide phát tín hiệu `page_selected(idx)` cho phép giáo viên nhảy ngay lập tức đến bất kỳ slide nào trong bài giảng (thay vì phải lật tuần tự).
- `test_slide_drawer_action_buttons`: Kiểm thử các nút tác vụ nhanh trên thanh công cụ ngăn kéo (Thêm trang trắng mới `➕`, Nhân bản slide `📋`, Xóa slide `🗑️`).
- `test_sticky_note_themes_integrity` & `test_sticky_note_theme_cycling`: Kiểm thử 5 chủ đề màu ghi chú pastel (`yellow`, `green`, `blue`, `coral`, `lavender`) với đường viền, dải ruy-băng đầu trang và ghim đính 📌.
- `test_sticky_note_dimensions_and_collapse`: Kiểm thử co giãn thẻ ghi chú giữa dạng thẻ mở rộng ($240 \times 160$) và dạng huy hiệu thu gọn ($160 \times 36$) để không chiếm diện tích bảng khi đang giảng bài.
- `test_sticky_note_serialization`: Kiểm thử chuyển đổi 2 chiều giữa đối tượng đồ họa và từ điển JSON để lưu trữ bền vững trong gói bài giảng `.airteach`.
- `test_lesson_board_add_and_get_sticky_notes`: Quản lý vòng đời ghi chú trên `LessonBoard` (thêm mới, truy vấn ghi chú đang hiển thị, xóa từng thẻ hoặc xóa toàn bộ).
- `test_main_window_slide_drawer_and_sticky_notes`: Tích hợp toàn diện trên MainWindow, kiểm tra phím tắt `F2` ẩn/hiện ngăn kéo slide, phím tắt `Ctrl+N` tạo ghi chú dán, lưu trữ và phục hồi độc lập ghi chú theo từng trang bài giảng.

#### 2.4. Vệt Sáng Laser Dần Biến Mất & Thuật Sĩ Hiệu Chỉnh Cử Chỉ Khoảng Cách (`tests/test_laser_trail_and_calibration.py`)
- `test_laser_trail_item_properties_and_buffering`: Khởi tạo đối tượng `LaserTrailItem` với deque lưu vết điểm, lọc micro-jitter khi di chuyển < 2px, phân rã theo thời gian thực (lifetime 1.0s - 2.5s).
- `test_laser_trail_decay_and_idle_timer`: Kiểm thử cơ chế 0% idle CPU overhead: bộ đếm `QTimer` 30 FPS tự động dừng chạy khi hàng đợi điểm rỗng và chỉ kích hoạt khi giáo viên chỉ trỏ.
- `test_laser_trail_neon_paint_rendering`: Dựng hình 2 tầng (vệt neon tỏa sáng bên ngoài và lõi nhiệt phát quang bên trong) với độ mờ đục tỷ lệ nghịch với tuổi thọ điểm.
- `test_lesson_board_laser_trail_integration`: Tích hợp đa bàn tay độc lập trên `LessonBoard`, hỗ trợ bật/tắt vệt sáng và tùy biến thời gian tồn tại qua `set_laser_trail_lifetime()`.
- `test_gesture_calibration_dialog_lifecycle`: Thuật sĩ hiệu chỉnh cử chỉ trực quan `GestureCalibrationDialog`, đo khoảng cách thực tế đến webcam (chuẩn 1.5m - 2.5m), thanh trượt độ nhạy con trỏ (0.5x - 2.5x), độ mịn bộ lọc 1€ chống rung tay, thời gian dừng dwell click (300ms - 1200ms) và bàn tay thuận (Phải / Trái).
- `test_calibration_target_practice_pad`: Bàn tập luyện chạm không chạm (Air Click Practice Pad) tương tác với bộ đếm đo độ trễ mili-giây thời gian thực.
- `test_main_window_gesture_calibration_integration`: Kiểm tra menu *Công cụ* trên `MainWindow` kích hoạt thuật sĩ hiệu chỉnh và phát tín hiệu lưu trữ cấu hình.

#### 2.5. Thu Phóng Bảng Vẽ & Kiểm Thử Âm Thanh Đa Dạng (`tests/test_zoom_and_sound_preview.py`)
- `test_settings_dialog_audio_preview_combo`: Kiểm tra danh sách thả xuống thử nghiệm âm thanh với đầy đủ 6 loại âm thanh cử chỉ ("Snap", "Air Click", "Page Turn", "Erase", "Timer Chime", "Undo").
- `test_settings_dialog_play_test_sound`: Kiểm thử hàm `play_test_sound` thực thi an toàn không phát sinh lỗi ngoại lệ, cho phép nghe thử âm thanh ở mức âm lượng thiết lập ngay cả khi tính năng âm thanh đang tạm tắt.
- `test_settings_dialog_on_test_audio_slot`: Kiểm tra nút `🔊 Thử âm thanh` kết nối đồng bộ giữa combo box và âm lượng thiết lập.
- `test_canvas_zoom_widget_controls_and_signals`: Khối điều khiển thu phóng nổi acrylic `CanvasZoomWidget` với 4 nút (`btn_pan`, `btn_zoom_out`, `btn_reset`, `btn_zoom_in`), kiểm thử phát các tín hiệu điều khiển và hiển thị phần trăm phóng đại (`100%`, `150%`, `75%`).
- `test_canvas_zoom_controller_defaults_and_clamping`: Bộ điều khiển `CanvasZoomController` giới hạn nghiêm ngặt khoảng phóng đại $[0.25\times, 4.0\times]$, bước tăng $1.20\times$, bước giảm $0.8333\times$, và đưa về chuẩn $100\%$ (`reset_zoom()`).
- `test_canvas_zoom_controller_pan_mode_and_drag`: Kiểm tra chế độ cuộn/kéo bảng (Pan mode), chuyển đổi con trỏ chuột `OpenHandCursor` / `ClosedHandCursor` và dịch chuyển thanh cuộn theo pixel màn hình.
- `test_canvas_zoom_controller_wheel_event_filtering`: Lọc sự kiện con lăn chuột: cuộn thông thường không ảnh hưởng bảng, giữ `Ctrl + Wheel` phóng to/thu nhỏ mượt mà tâm điểm chuột.
- `test_main_window_zoom_and_pan_integration`: Tích hợp toàn diện trên `MainWindow`, phím tắt `Ctrl++`, `Ctrl+-`, `Ctrl+0`, `H`, các mục menu *Hiển thị*, và kích hoạt thông qua menu nổi không chạm.
- `test_bimanual_pinch_to_zoom_and_air_panning`: Tính toán khoảng cách Euclid giữa 2 đầu ngón tay trỏ được chuẩn hóa $D = \sqrt{\Delta x^2 + \Delta y^2}$, khi 2 tay chụm ngón tay và tách xa nhau thì phóng to bảng vẽ, khi đưa lại gần thì thu nhỏ; và chế độ cuộn bảng không chạm (Touchless Air Panning) chuyển dịch bảng khi giáo viên vẫy tay mà không tạo nét vẽ thừa.

#### 2.6. Cổng Logic Số Tương Tác & Định Tuyến Mạch Vuông Góc Manhattan (`tests/test_interactive_logic_gates.py`)
- `test_interactive_gate_boolean_evaluations`: Mô phỏng số học logic xác thực 100% cho 6 loại cổng logic số cơ bản (AND, OR, NOT, NAND, NOR, XOR) theo đúng bảng chân lý đại số Boole.
- `test_interactive_gate_pin_toggle_and_signals`: Kiểm thử lật trạng thái bit ngõ vào $0 \leftrightarrow 1$ khi nhấp chuột/chạm không chạm, phát tín hiệu `state_changed(gate_type, output)` và bật/tắt bảng chân lý `truth_table_toggled`.
- `test_interactive_gate_magnetic_pin_snapping`: Tính toán chính xác tọa độ cổng kết nối trên scene (`get_input_scene_pos`, `get_output_scene_pos`) và thuật toán bắt dính từ tính (Magnetic Pin Snapping) tự động hút đầu dây dẫn vào đúng chân linh kiện trong bán kính dung sai 20px.
- `test_interactive_gate_air_click`: Kiểm thử cử chỉ chạm không chạm (Touchless Air Click): giáo viên chỉ cần vẫy chạm đầu ngón tay trong không trung trúng nút huy hiệu chân A/B để đảo bit $0/1$ hoặc chạm nút `📊 BCL` để mở bảng chân lý mà không cần đụng chạm chuột.
- `test_informatics_tools_flowchart_pins_and_manhattan_wire`: 
  - Tính toán 4 điểm neo chính (cardinal pins: top, bottom, left, right) cho các khối lưu đồ thuật toán (hình chữ nhật Process, hình thoi Decision, hình bình hành I/O, oval Terminator).
  - Thuật toán định tuyến dây dẫn vuông góc 90° Manhattan (`create_manhattan_wire`) với điểm nối hàn tròn (solder junction dot) tự động gấp khúc góc vuông thẩm mỹ chuẩn sơ đồ mạch điện tử EDA.
- `test_main_window_interactive_logic_gate_integration`: Tích hợp toàn diện trên MainWindow, chèn cổng logic tương tác vào bảng vẽ, lắng nghe tín hiệu hiển thị trạng thái lên thanh status bar, và bắt sự kiện chạm không chạm của bàn tay MediaPipe để kích hoạt cổng.

#### 2.7. Phân Tích Hệ Lực Mặt Phẳng Nghiêng Vật Lý Cơ Học (`tests/test_interactive_physics_incline.py`)
- `test_incline_item_initialization_and_geometry`: Khởi tạo mặt phẳng nghiêng với góc dốc $\alpha = 30^\circ$, khối lượng $m = 2.0\text{ kg}$, hệ số ma sát $\mu = 0.25$. Tính toán chính xác hình học tam giác vuông đáy nằm ngang và đỉnh dốc.
- `test_incline_physics_calculations_and_slip_condition`: Phân tích đầy đủ hệ lực vật lý cổ điển Newton:
  - Trọng lực $P = mg$, thành phần trượt song song dốc $P_\parallel = P \sin\alpha$, thành phần áp sát mặt dốc $P_\perp = P \cos\alpha$.
  - Phản lực pháp tuyến $N = P_\perp$.
  - Lực ma sát trượt $F_{ms} = \mu N$.
  - Lực hợp lực kéo vật $F_{net} = P_\parallel - F_{ms}$, gia tốc $a = F_{net}/m$.
  - Điều kiện trượt: $\tan\alpha > \mu$ (vật trượt xuống dốc) so với $\tan\alpha \le \mu$ (vật nằm cân bằng tĩnh).
- `test_incline_angle_adjustment_and_clamping`: Điều chỉnh góc dốc $\alpha$ từ $5^\circ$ đến $75^\circ$, phát tín hiệu `angle_changed` và `physics_updated`.
- `test_incline_mass_and_friction_adjustment`: Cập nhật linh hoạt khối lượng $m$ và hệ số ma sát $\mu$, tự động tái tính toán toàn bộ véctơ lực tức thì.
- `test_incline_apex_drag_with_mouse`: Kéo thả chuột tại đỉnh dốc $W_3$, tính toán góc dốc mới theo vị trí chuột và cập nhật hình học tương tác.
- `test_incline_touchless_air_gesture_drag`: Bắt cử chỉ véo tay không chạm (Touchless Air Pinch) để kéo đỉnh dốc thay đổi góc nghiêng mượt mà.
- `test_incline_stamp_to_canvas`: Dập sơ đồ hệ lực véctơ phân tích vĩnh viễn lên bảng vẽ `LessonBoard` thông qua `PhysicsTools.create_force_vector`.
- `test_lesson_board_and_main_window_integration`: Tích hợp vòng đời hiển thị trên `LessonBoard` (`show_inclined_plane`, `toggle_inclined_plane`), phím tắt `Ctrl+Shift+I`, nút thanh công cụ và liên kết menu nổi không chạm.

#### 2.8. Mô Phỏng Quang Hình Học Khúc Xạ Ánh Sáng & Phản Xạ Toàn Phần (`tests/test_interactive_optics_refraction.py`)
- `test_optics_initialization_and_geometry`: Khởi tạo mô phỏng khúc xạ ánh sáng với 2 môi trường trong suốt ngăn cách bởi mặt phân cách nằm ngang, nguồn sáng $S$, điểm tới $I$, và pháp tuyến $NN'$.
- `test_snells_law_and_refraction_angle`: Kiểm chứng chính xác 100% định luật khúc xạ Snell ($n_1 \sin i = n_2 \sin r$) và định luật phản xạ ($i' = i$). Thử nghiệm môi trường không khí ($n_1 = 1.0$) sang thủy tinh ($n_2 = 1.5$) với góc tới $i = 30^\circ \implies r = \arcsin(1/3) \approx 19.47^\circ$, tia khúc xạ bẻ cong lại gần pháp tuyến ($r < i$).
- `test_total_internal_reflection_and_critical_angle`: Thử nghiệm hiện tượng Phản xạ Toàn phần (TIR) khi truyền từ môi trường chiết quang hơn sang kém chiết quang hơn ($n_1 = 1.5 > n_2 = 1.0$):
  - Góc tới hạn phản xạ toàn phần: $i_{gh} = \arcsin(1.0 / 1.5) \approx 41.81^\circ$.
  - Khi $i = 30^\circ < 41.81^\circ$: Có tia khúc xạ đi vào môi trường 2 với $r \approx 48.59^\circ$.
  - Khi $i = 60^\circ > 41.81^\circ$: **Phản xạ toàn phần xảy ra hoàn toàn**! Toàn bộ chùm sáng bị phản xạ trở lại môi trường 1, triệt tiêu hoàn toàn tia khúc xạ, hiển thị huy hiệu cảnh báo neon `⚡ PHẢN XẠ TOÀN PHẦN`.
- `test_media_presets_and_swapping`: Thử nghiệm các bộ preset môi trường quang học thực tế (Không khí $1.00$, Nước $1.333$, Thủy tinh Crown $1.52$, Dầu hỏa $1.48$, Kim cương $2.417$) và nút đảo ngược 2 môi trường `[🔄 Đổi n₁ ⇄ n₂]` để giảng giải hiện tượng phản xạ toàn phần tức thì trong giảng dạy.
- `test_speed_of_light_in_media`: Tính toán vận tốc truyền ánh sáng trong từng môi trường $v = c / n$ hiển thị trực tiếp trên thẻ HUD giải thích.
- `test_touchless_air_gestures`: Hỗ trợ thao tác không chạm hoàn chỉnh: kéo nguồn sáng $S$ để thay đổi góc tới $i$, dịch chuyển điểm tới $I$, và vẫy chạm ngón tay (Air Tap) để đảo môi trường hoặc dập vector.
- `test_stamp_to_canvas_and_lessonboard`: Dập vĩnh viễn sơ đồ quang học véctơ chuẩn xác cao lên bảng vẽ `LessonBoard` với các tia sáng mũi tên, vạch góc, nhãn môi trường và công thức Snell.
- `test_lessonboard_and_main_window_integration`: Tích hợp vòng đời hiển thị trên `LessonBoard` (`show_optics_refraction`, `toggle_optics_refraction`), phím tắt `Ctrl+Shift+O`, nút thanh công cụ `🌈 Khúc xạ ánh sáng` và liên kết menu nổi không chạm.
