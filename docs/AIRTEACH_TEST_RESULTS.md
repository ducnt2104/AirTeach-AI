# AIRTEACH AI - COMPREHENSIVE TEST SUITE EXECUTION REPORT
## Báo Cáo Kết Quả Kiểm Thử Toàn Diện Hệ Thống (Automated Test Execution Results)

**Dự án:** AirTeach AI — Hệ thống dạy học thông minh tương tác không chạm bằng AI  
**Thời điểm thực thi:** 23/10/2026 - 24/10/2026  
**Môi trường thực thi:** Microsoft Windows 11 (AMD64), Python 3.15.0rc1, PySide6 6.12.0, OpenCV 5.0.0.93, MediaPipe 1.1.1, SciPy 1.18.1, Keyring 25.7.0  
**Khung kiểm thử:** `pytest` 9.1.1 kết hợp `pytest-qt` 4.5.0  
**Kết quả chung cuộc:** **689 / 689 BÀI TEST VƯỢT QUA THÀNH CÔNG (100% PASSED) TRÊN 53 BỘ KIỂM THỬ**

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
| **SUITE-40** | `tests/test_interactive_standing_wave.py` | Sóng cơ & Sóng dừng cộng hưởng (Interactive Standing Wave), điều kiện 2 đầu cố định $L=k(\lambda/2)$ & 1 đầu tự do, nút N / bụng B, bao sóng đối xứng, hoạt họa 40 FPS, dập vector | 8 | **PASSED** | 7.83s | **100%** |
| **SUITE-41** | `tests/test_interactive_vsepr_geometry.py` | Mô phỏng 3D hình học phân tử VSEPR & Lai hóa orbital (Interactive 3D VSEPR Geometry), xoay 3D Euler Pitch/Yaw, 8 mô hình AX₂-AX₆, thùy e tự do, dập vector | 7 | **PASSED** | 8.74s | **100%** |
| **SUITE-42** | `tests/test_interactive_dna_replication.py` | Tự nhân đôi DNA & Nguyên tắc bổ sung (Interactive DNA Replication), $A=T$ (2 lk H), $G \equiv C$ (3 lk H), chạc ba tái bản, đột biến điểm, dập vector | 8 | **PASSED** | 7.81s | **100%** |
| **SUITE-43** | `tests/test_interactive_kepler_orbit.py` | Mô phỏng 3 Định luật Kepler & Chuyển động thiên thể (Interactive Kepler Orbit Item), quỹ đạo Elip $r(\theta)=\frac{a(1-e^2)}{1+e\cos\theta}$, quét diện tích $S_1=S_2$, Vis-Viva $v(\theta)$, 4 preset hành tinh, dập vector | 8 | **PASSED** | 4.68s | **100%** |
| **SUITE-44** | `tests/test_interactive_graph_pathfinding.py` | Lý thuyết đồ thị & Đường đi ngắn nhất Dijkstra / A* (Interactive Graph Pathfinding), đồ thị có trọng số $G=(V, E)$, Priority Queue Min-Heap, hàm $f=g+h$, kéo thả đỉnh, dập vector | 8 | **PASSED** | 2.21s | **100%** |
| **SUITE-45** | `tests/test_interactive_electric_field.py` | Đường sức điện trường & Định luật Coulomb tĩnh điện (Interactive Electric Field), RK4 streamline integration, đa điện tích điểm $+/-$, đầu dò vector $E(P)$ & điện thế $V(P)$, dập vector | 8 | **PASSED** | 2.35s | **100%** |
| **SUITE-46** | `tests/test_interactive_lorentz_force.py` | Mô phỏng Từ trường & Lực từ Lorentz làm lệch hạt mang điện (Interactive Lorentz Force), bảo toàn động năng $W=0$, bán kính Cyclotron $R=mv/(qB)$, Boris rotation solver, đảo chiều B-field $\otimes/\odot$, dập vector | 8 | **PASSED** | 3.12s | **100%** |
| **SUITE-47** | `tests/test_interactive_trophic_web.py` | Lưới thức ăn sinh thái & Động học quần thể Lotka-Volterra (Interactive Trophic Web Item), tích phân số RK4, bảo toàn $V(x,y)$, sốc sinh thái (hạn hán, dịch bệnh), kim tự tháp Lindeman 10%, dập vector | 8 | **PASSED** | 1.64s | **100%** |
| **SUITE-48** | `tests/test_interactive_gas_laws.py` | Định luật khí lý tưởng Clapeyron & Phân bố vận tốc Maxwell-Boltzmann (Interactive Gas Laws), piston di động, quá trình Đẳng nhiệt/Đẳng áp/Đẳng tích/Tự do, khí He/N₂/Ar, đun nóng/làm lạnh, dập vector | 8 | **PASSED** | 1.42s | **100%** |
| **SUITE-49** | `tests/test_interactive_acid_base_titration.py` | Chuẩn độ Axit - Bazơ & Đường cong pH đệm Henderson-Hasselbalch (Interactive Titration), Buret thủy tinh nhỏ giọt, bình tam giác đổi màu chỉ thị, đạo hàm dpH/dV, dập vector | 7 | **PASSED** | 1.68s | **100%** |
| **SUITE-50** | `tests/test_interactive_harmonic_oscillator.py` | Dao động điều hòa, Tắt dần & Cộng hưởng cơ học (Interactive Harmonic Oscillator), tích phân số RK4, lò xo xoắn ốc động, kéo vật air pinch, bao sóng tắt dần, cộng hưởng Lorentzian, dập vector | 8 | **PASSED** | 1.45s | **100%** |
| **SUITE-51** | `tests/test_interactive_radioactive_decay.py` | Định luật phóng xạ hạt nhân & Chuỗi phân rã (Interactive Radioactive Decay), ma trận 100 hạt Monte Carlo, chớp phát xạ alpha/beta/gamma, máy đếm Geiger, tuổi mẫu vật, dập vector | 8 | **PASSED** | 1.47s | **100%** |
| **SUITE-52** | `tests/test_interactive_chemical_kinetics.py` | Động hóa học, Năng lượng hoạt hóa Arrhenius & Cân bằng Le Chatelier (Interactive Chemical Kinetics), sơ đồ thế năng Ea/ΔH, xúc tác, đồ thị nồng độ [A]/[C], dập vector | 8 | **PASSED** | 1.56s | **100%** |
| **SUITE-53** | `tests/test_interactive_cryptography.py` | Mật mã học RSA & Số học đồng dư (Interactive Cryptography), thuật toán Euclid mở rộng, nghịch đảo modulo, mã hóa/giải mã RSA, Diffie-Hellman, mạng dây cung Cardioid, dập vector | 8 | **PASSED** | 3.67s | **100%** |
| **TỔNG KẾT** | **Toàn Bộ Hệ Thống** | **53 Bộ kiểm thử chuyên sâu toàn diện** | **689** | **ALL PASSED** | **340.9s** | **100.0%** |

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

#### 2.9. Sóng Cơ Học & Hiện Tượng Sóng Dừng Trên Sợi Dây Đàn Hồi (`tests/test_interactive_standing_wave.py`)
- `test_standing_wave_initialization_and_geometry`: Khởi tạo sợi dây sóng dừng với chiều dài $L = 480\text{ px}$, họa âm $k = 3$, biên độ $A = 30\text{ px}$, tần số $f = 2.0\text{ Hz}$, vị trí cân bằng và ngàm cố định hai đầu.
- `test_standing_wave_fixed_fixed_resonance`: Kiểm chứng điều kiện sóng dừng trên dây hai đầu cố định:
  - Công thức cộng hưởng: $L = k \frac{\lambda}{2} \implies \lambda = \frac{2L}{k}$.
  - Vận tốc truyền sóng: $v = \lambda \cdot f$.
  - Số nút sóng: $N = k + 1 = 4$ nút ($u=0$ cố định tại $x = 0, L/3, 2L/3, L$).
  - Số bụng sóng: $B = k = 3$ bụng sóng (dao động với biên độ cực đại $2A$).
- `test_standing_wave_fixed_free_resonance`: Kiểm chứng điều kiện sóng dừng một đầu cố định, một đầu tự do (ngàm trượt vòng nhẫn):
  - Công thức cộng hưởng: $L = (2k + 1) \frac{\lambda}{4} \implies \lambda = \frac{4L}{2k + 1}$.
  - Số nút sóng = số bụng sóng = $k + 1 = 3$ (với $k = 2$).
- `test_wave_displacement_and_envelope`: Kiểm tra hàm độ dời tức thời $u(x, t) = 2A \sin(kx) \cos(\omega t)$:
  - Tại mọi nút sóng $x_n$, độ dời triệt tiêu hoàn toàn $u(x_n, t) \equiv 0$ với mọi thời điểm $t$.
  - Tại các bụng sóng $x_b$, đường bao biên độ đạt cực đại đối xứng $\pm A$.
- `test_harmonic_and_amplitude_mutators`: Thay đổi linh hoạt họa âm $k \in [1, 6]$, tự động tái tính toán bước sóng $\lambda$, vị trí các nút và bụng, điều chỉnh biên độ $A$ và tần số $f$.
- `test_play_pause_and_animation_timer`: Bật/tắt hoạt họa dao động sợi dây 40 FPS bằng `QTimer`, kiểm tra dừng mượt mà và chạy tiếp mà không lệch pha.
- `test_mouse_and_touchless_air_gestures`: Kéo thả tay cầm biên độ $A$ ở đỉnh bụng sóng bằng chuột hoặc cử chỉ chụm tay không chạm (Air Pinch), vẫy chạm không chạm (Air Tap) vào các nút chọn họa âm $k=1..6$, nút chuyển chế độ ngàm và nút tạm dừng.
- `test_stamp_to_canvas_and_main_window_integration`: Dập sơ đồ vector sóng dừng vĩnh viễn lên bảng vẽ `LessonBoard` (đường vị trí cân bằng, 2 đường bao đối xứng, ảnh chụp sợi dây dao động, điểm nút/bụng, thước đo bước sóng $\lambda$ và thẻ công thức), tích hợp phím tắt `Ctrl+Shift+W`, nút thanh công cụ `🌊 Sóng dừng` và liên kết menu nổi không chạm.

#### 2.10. Mô Phỏng 3D Hình Học Phân Tử VSEPR & Lai Hóa Orbital Hóa Học (`tests/test_interactive_vsepr_geometry.py`)
- `test_vsepr_initialization_and_models`: Khởi tạo mô phỏng hình học phân tử 3D tương tác với cấu hình chuẩn $AX_4$ (Tứ diện đều, Metan $CH_4$, góc $109.5^\circ$, lai hóa $sp^3$, 4 liên kết, 0 cặp electron tự do).
- `test_vsepr_model_selection_and_properties`: Kiểm chứng đầy đủ 8 cấu hình hình học kinh điển của thuyết đẩy cặp electron hóa trị (VSEPR):
  1. $AX_2$: Tuyến tính ($180^\circ$, $sp$, $CO_2$)
  2. $AX_3$: Tam giác phẳng ($120^\circ$, $sp^2$, $BF_3$)
  3. $AX_2E$: Gấp khúc ($\approx 119^\circ$, $sp^2$, 1 cặp e tự do, $SO_2$)
  4. $AX_4$: Tứ diện đều ($109.5^\circ$, $sp^3$, $CH_4$)
  5. $AX_3E$: Chóp tam giác ($\approx 107.3^\circ$, $sp^3$, 1 cặp e tự do đẩy các liên kết ép nhỏ lại, $NH_3$)
  6. $AX_2E_2$: Gấp khúc kiểu nước ($\approx 104.5^\circ$, $sp^3$, 2 cặp e tự do tạo lực đẩy mạnh nhất, $H_2O$)
  7. $AX_5$: Lưỡng tháp tam giác ($90^\circ$ và $120^\circ$, $sp^3d$, $PCl_5$)
  8. $AX_6$: Bát diện đều ($90^\circ$, $sp^3d^2$, $SF_6$)
- `test_3d_rotation_and_projection`: Kiểm chứng thuật toán quay Euler 3D hai góc Pitch $\theta_x$ và Yaw $\theta_y$ kết hợp phép chiếu phối cảnh ảo ($d = 450\text{ px}$) không gây biến dạng tọa độ, bao gói góc quay $[-180^\circ, 180^\circ]$.
- `test_auto_spin_animation`: Bật/tắt chế độ tự xoay 3D mượt mà `[🔄 Tự xoay]` bằng `QTimer` ~33 FPS, tự động quay quanh trục Y giúp học sinh hình dung không gian 3 chiều trực quan.
- `test_touchless_air_gestures`: Hỗ trợ thao tác không chạm hoàn chỉnh: véo tay trong không trung (Air Pinch Drag) để xoay phân tử 3D theo cử chỉ bàn tay, vẫy chạm không chạm (Air Tap) vào các nút chọn mô hình $AX_2 \dots AX_6$ và nút đóng dấu.
- `test_stamp_to_canvas_and_lessonboard`: Dập mô hình phân tử 3D vĩnh viễn lên bảng vẽ `LessonBoard` với thuật toán tô bóng hình cầu CPK `QRadialGradient`, liên kết ống trụ 3D và thẻ thông tin VSEPR thông qua `ChemistryTools.create_molecular_geometry`.
- `test_lessonboard_and_main_window_integration`: Tích hợp vòng đời hiển thị trên `LessonBoard` (`show_vsepr_geometry`, `toggle_vsepr_geometry`), phím tắt `Ctrl+Shift+Y`, nút thanh công cụ `⚛️ Hình học VSEPR` và liên kết menu nổi không chạm.

#### 2.11. Mô Phỏng Tự Nhân Đôi DNA & Nguyên Tắc Bổ Sung Watson-Crick (`tests/test_interactive_dna_replication.py`)
- `test_dna_initialization_and_properties`: Khởi tạo chuỗi xoắn kép DNA với 12 cặp bazơ nitơ chuẩn, tự động sinh mạch bổ sung theo nguyên tắc $A-T, G-C$, tiến độ tháo xoắn ban đầu $40\%$, nhãn đầu $5'$ và $3'$ ngược chiều.
- `test_watson_crick_base_pairing_and_hydrogen_bonds`: Kiểm chứng quy tắc liên kết hydro:
  - Cặp $A - T$: 2 liên kết hydro ($A = T$).
  - Cặp $G - C$: 3 liên kết hydro ($G \equiv C$).
  - Chuỗi mẫu 12 cặp (6 cặp $A-T$ và 6 cặp $G-C$) tính toán chính xác tuyệt đối $6 \times 2 + 6 \times 3 = 30$ liên kết hydro.
- `test_unzip_progress_and_fork_clamping`: Kéo thanh trượt tháo xoắn chạc ba tái bản (Replication Fork) từ $0.0$ (chuỗi xoắn kép nguyên vẹn) đến $1.0$ (tách đôi hoàn toàn 2 mạch đơn), giới hạn nghiêm ngặt $[0.0, 1.0]$.
- `test_auto_replication_animation`: Bật/tắt chế độ tự động nhân đôi DNA `[▶ Nhân đôi]` bằng `QTimer` ~28 FPS, mô phỏng enzyme Helicase tách mạch mẹ kết hợp DNA Polymerase tổng hợp liên tục mạch mới con theo nguyên tắc bán bảo tồn.
- `test_point_mutation_and_reset`: Thử nghiệm đột biến điểm `[🧬 Thử đột biến điểm]` (thay thế cặp nucleotit thứ 6), phát hiện sai hỏng cấu trúc và nút `[↺ Khôi phục]` đưa về chuỗi hoang dại ban đầu.
- `test_touchless_air_gestures`: Hỗ trợ thao tác không chạm: véo tay trượt chạc ba tái bản (Air Drag Fork), vẫy chạm không chạm (Air Tap) vào các nút đột biến, khôi phục và tự nhân đôi.
- `test_stamp_to_canvas_and_lessonboard`: Dập chuỗi xoắn kép vector lên bảng vẽ `LessonBoard` thông qua `BiologyTools.create_dna_double_helix`.

#### 2.12. Mô Phỏng 3 Định Luật Kepler & Chuyển Động Thiên Thể Thiên Văn Học (`tests/test_interactive_kepler_orbit.py`)
- `test_kepler_orbit_init_and_properties`: Khởi tạo mô phỏng chuyển động thiên thể hình elip với bán trục lớn $a = 190.0\text{ px}$, tâm sai $e = 0.45$, bán trục bé $b = a\sqrt{1-e^2}$, tiêu cự $c = a \cdot e$, khoảng cách cận nhật $r_p = a(1-e)$ và viễn nhật $r_a = a(1+e)$, vị trí tiêu điểm Mặt Trời $F_1$ và tiêu điểm phụ $F_2$.
- `test_kepler_polar_equation`: Kiểm chứng phương trình tọa độ cực chính xác của quỹ đạo Kepler $r(\theta) = \frac{a(1-e^2)}{1+e\cos\theta}$:
  - Tại cận nhật $\theta = 0$: $r = a(1-e) = r_p$.
  - Tại viễn nhật $\theta = \pi$: $r = a(1+e) = r_a$.
  - Tại góc vuông $\theta = \pi/2$: $r = a(1-e^2)$ (bán trục thông số $p$).
- `test_kepler_presets`: Kiểm chứng các giá trị thực tế của hệ Mặt Trời: Trái Đất ($e=0.017$), Sao Hỏa ($e=0.093$), Sao Thủy ($e=0.206$), Sao Chổi Halley ($e=0.800$), và khả năng kẹp biên tâm sai tùy chỉnh $[0.0, 0.85]$.
- `test_kepler_vis_viva_velocity`: Kiểm chứng phương trình Vis-Viva $v = \sqrt{GM(2/r - 1/a)}$: vận tốc cực đại tại cận nhật $P$ và cực tiểu tại viễn nhật $A$, tỉ số tốc độ $v_p / v_a = \frac{1+e}{1-e}$ đúng chuẩn cơ học thiên thể.
- `test_kepler_toggle_and_tick`: Bật/tắt chuyển động hành tinh `[▶ Chuyển động]`, bật/tắt diện tích quét bằng nhau $S_1 = S_2$ theo định luật 2 Kepler, tích phân thời gian thực $\frac{d\theta}{dt} = \frac{h}{r^2}$ với $h = \frac{2\pi ab}{T}$.
- `test_kepler_air_gestures`: Hỗ trợ thao tác không chạm: véo tay điều chỉnh tâm sai tại điểm cận nhật, vẫy chạm không chạm (Air Tap) chọn các hành tinh và bật/tắt hoạt họa.
- `test_kepler_stamping`: Dập quỹ đạo elip vector sắc nét, Mặt Trời tỏa sáng và thẻ định luật Kepler lên bảng vẽ thông qua `MathTools.create_conic_section`.
- `test_lesson_board_kepler_integration`: Tích hợp vòng đời hiển thị trên `LessonBoard` (`show_kepler_orbit`, `toggle_kepler_orbit`), phím tắt `Ctrl+Shift+K`, nút thanh công cụ `🪐 Quỹ đạo Kepler` và liên kết menu nổi không chạm.

#### 2.13. Lý Thuyết Đồ Thị & Thuật Toán Tìm Đường Đi Ngắn Nhất Dijkstra / A* (`tests/test_interactive_graph_pathfinding.py`)
- `test_graph_init_and_properties`: Khởi tạo mô hình đồ thị có trọng số $G = (V, E)$ gồm 7 đỉnh, 12 cạnh, đỉnh xuất phát $A$, đỉnh đích $G$, khởi tạo chi phí ban đầu $g(A) = 0.0$ và $g(V \setminus \{A\}) = \infty$, trạng thái đỉnh xuất phát `OPEN` đưa vào hàng đợi ưu tiên Min-Heap.
- `test_dijkstra_full_run_optimal_path`: Kiểm chứng thuật toán Dijkstra tìm kiếm chính xác đường đi tối ưu tuyệt đối trên topo mạng `NETWORK_7`: Lộ trình tối ưu $A \to C (2) \to F (4) \to E (3) \to G (3)$ với tổng chi phí ngắn nhất bằng $12.0$, cập nhật trạng thái các đỉnh trên lộ trình thành `PATH` và làm sáng các cạnh tương ứng với viền vàng dạ quang (`#fbbf24`).
- `test_astar_search_and_heuristic`: Kiểm chứng thuật toán A* Search với hàm đánh giá $f(n) = g(n) + h(n)$ sử dụng khoảng cách Euclid làm heuristic chấp nhận được (admissible), đảm bảo tìm ra lộ trình ngắn nhất trùng khớp với Dijkstra trong khi giảm số đỉnh cần duyệt.
- `test_step_by_step_execution_and_signals`: Kiểm chứng bộ máy máy trạng thái hữu hạn (FSM) từng bước: mỗi bước lấy đỉnh có $f$ nhỏ nhất ra khỏi Min-Heap, gán nhãn `CURRENT`, duyệt giãn (relax) các đỉnh kề, cập nhật chi phí $g$ và con trỏ cha (parent), phát tín hiệu `step_advanced` và `path_found`.
- `test_graph_presets_switching`: Chuyển đổi linh hoạt giữa các topo đồ thị mẫu: Mạng 7 đỉnh (DAG), Vòng lục giác 6 đỉnh (Ring - tìm ra lộ trình tối ưu $A \to F \to E \to D$ chi phí 8.0 thay vì $A \to B \to C \to D$ chi phí 9.0), và Cây nhị phân 7 đỉnh (Binary Tree).
- `test_node_repositioning`: Kéo thả chuột hoặc véo tay không chạm (Air Pinch Drag) di chuyển đỉnh tự do trên bảng vẽ; hệ thống tự động tính toán lại góc kết nối, vị trí nhãn trọng số cạnh và khoảng cách heuristic theo thời gian thực.
- `test_touchless_air_gestures`: Hỗ trợ thao tác không chạm: véo tay di chuyển đỉnh đồ thị, vẫy chạm không chạm (Air Tap) vào các nút điều khiển (`[▶ Chạy tự động]`, `[⏭ Từng bước]`, `[↺ Đặt lại]`, `[Thuật toán: Dijkstra ↔ A*]`, `[📌 Đóng dấu]`, `[✕]`).
- `test_stamp_to_canvas_and_lessonboard`: Dập mô hình đồ thị vector sắc nét, nhãn trọng số cạnh, lộ trình vàng tối ưu và thẻ lý thuyết thuật toán lên bảng vẽ `LessonBoard` thông qua `InformaticsTools.create_graph_diagram`.
- `test_lessonboard_and_main_window_integration`: Tích hợp vòng đời hiển thị trên `LessonBoard` (`show_graph_pathfinding`, `toggle_graph_pathfinding`), phím tắt `Ctrl+Shift+P`, nút thanh công cụ `🕸️ Đồ thị Dijkstra` và liên kết menu nổi không chạm.

#### 2.14. Mô Phỏng Đường Sức Điện Trường & Định Luật Coulomb Tĩnh Điện (`tests/test_interactive_electric_field.py`)
- `test_electric_field_init_and_properties`: Khởi tạo mô phỏng tĩnh điện Coulomb với cấu hình lưỡng cực điện (Dipole) gồm 2 điện tích điểm: $Q_1 = +1.0\ \mu\text{C}$ tại $(220, 220)$ và $Q_2 = -1.0\ \mu\text{C}$ tại $(440, 220)$, kích hoạt đầu dò vector kiểm tra và tích phân dòng đường sức ban đầu.
- `test_coulomb_law_superposition_dipole`: Kiểm chứng nguyên lý chồng chất điện trường $\vec{E} = k_e \sum \frac{q_i}{r_i^3} \vec{r}_i$ và điện thế $V = k_e \sum \frac{q_i}{r_i}$ trên lưỡng cực điện:
  - Tại trung điểm $M(330, 220)$: vector điện trường hướng sang phải về phía điện tích âm ($E_x > 0, E_y = 0$).
  - Điện thế $V(M) = 0.0\text{ V}$ triệt tiêu hoàn hảo do tính đối xứng của mặt phẳng trung trực.
- `test_coulomb_law_like_charges_neutral_point`: Kiểm chứng hai điện tích cùng dấu dương ($+1\ \mu\text{C}, +1\ \mu\text{C}$):
  - Tại trung điểm, hai vector lực điện trường đẩy nhau ngược chiều và triệt tiêu hoàn toàn ($\vec{E} = \vec{0}$, điểm trung hòa điện trường).
  - Điện thế $V(M) > 0$ do chồng chất dương cùng dấu.
- `test_rk4_field_lines_computation`: Kiểm chứng thuật toán giải tích số Runge-Kutta bậc 4 (RK4) tích phân đường cong vi phân $\frac{d\vec{r}}{ds} = \frac{\vec{E}}{\|\vec{E}\|}$:
  - Các đường sức điện xuất phát từ điện tích dương, uốn cong mượt mà theo phương vector điện trường và kết thúc tại điện tích âm hoặc biên mô phỏng.
  - Mỗi đường cong có tối thiểu $\ge 3$ điểm tọa độ liên tục, không tự cắt nhau.
- `test_electrostatic_presets_and_inversion`: Kiểm chứng các cấu hình tĩnh điện chuẩn: Lưỡng cực điện (Dipole), Hai điện tích dương ($+/+$), Hai điện tích âm ($-/-$), Tứ cực điện (Quadrupole 4 điện tích xen kẽ), và nút đổi dấu `[± Đổi dấu]` đảo ngược cực tính toàn bộ hệ thống.
- `test_probe_movement_and_signals`: Đầu dò vector $P(x, y)$ có thể kéo thả di chuyển tự do:
  - Hiển thị vector mũi tên cường độ điện trường $\vec{E}$ thời gian thực.
  - Thẻ thông số nổi hiển thị độ lớn $|\vec{E}|$ ($\text{V/m}$) và điện thế $V$ ($\text{V}$), phát tín hiệu `probe_updated`.
- `test_touchless_air_gestures`: Hỗ trợ thao tác không chạm: véo tay di chuyển hạt điện tích và đầu dò, vẫy chạm không chạm (Air Tap) vào các nút điều khiển (`[🎯 Đầu dò E⃗]`, `[± Đổi dấu]`, `[📌 Đóng dấu]`, `[✕]`, các nút preset).
- `test_stamp_to_canvas_and_lessonboard`: Dập đường sức điện trường vector, mũi tên định hướng, các hạt điện tích phát sáng và công thức định luật Coulomb lên bảng vẽ `LessonBoard` thông qua `PhysicsTools.create_electric_field_diagram`.
- `test_lessonboard_and_main_window_integration`: Tích hợp vòng đời hiển thị trên `LessonBoard` (`show_electric_field`, `toggle_electric_field`), phím tắt `Ctrl+Shift+E`, nút thanh công cụ `⚡ Điện trường Coulomb` và liên kết menu nổi không chạm.

#### 2.15. Mô Phỏng Từ Trường & Lực Từ Lorentz Làm Lệch Hạt Mang Điện (`tests/test_interactive_lorentz_force.py`)
- `test_lorentz_init_and_properties`: Khởi tạo buồng từ trường đồng nhất với hạt electron ($q=-1.0, m=1.0$), vận tốc ban đầu $v_0 = 160.0\text{ px/s}$, cảm ứng từ $B = 1.0\text{ T}$ hướng vào trang giấy ($\otimes$), tính toán bán kính Cyclotron lý thuyết $R = \frac{mv}{|q|B} = 160.0\text{ px}$ và chu kỳ quay $T = \frac{2\pi m}{|q|B} = 2\pi\text{ s}$.
- `test_lorentz_force_perpendicularity`: Kiểm chứng phương trình định luật Lorentz $\vec{F}_L = q(\vec{v} \times \vec{B})$. Do từ trường vuông góc với mặt phẳng vận tốc ($B_z \neq 0, B_x = B_y = 0$), lực từ Lorentz luôn vuông góc tuyệt đối với vận tốc tức thời: $\vec{F}_L \cdot \vec{v} = 0$, sinh công tức thời $dW = \vec{F}_L \cdot d\vec{r} = 0$.
- `test_kinetic_energy_conservation`: Kiểm chứng định luật bảo toàn động năng trong từ trường: Sau 60 bước tích phân số liên tục bằng thuật toán quay vận tốc Boris (Boris rotation solver), độ lớn vận tốc $|\vec{v}|$ và động năng $E_k = \frac{1}{2}mv^2$ được bảo toàn chính xác tuyệt đối ở mức sai số máy ($\Delta E_k / E_k < 10^{-12}$), triệt tiêu hoàn toàn hiện tượng năng lượng giả (artificial energy drift) của phương pháp Euler thông thường.
- `test_particle_types_deflection_and_cyclotron_radius`: Kiểm chứng phản ứng từ của các loại hạt mang điện:
  - Electron ($e^-$: $q=-1, m=1$): Bán kính nhỏ, quỹ đạo uốn cong ngược chiều kim đồng hồ.
  - Proton ($p^+$: $q=+1, m=6$): Bán kính lớn gấp 6 lần, uốn cong thuận chiều kim đồng hồ.
  - Hạt Alpha ($\alpha$: $q=+2, m=12$): Bán kính $R = 6 \cdot R_e$, lệch thuận chiều kim đồng hồ.
- `test_b_field_direction_reversal`: Đảo chiều vector cảm ứng từ $\vec{B}$ giữa $\otimes$ (Hạt bay vào trong) và $\odot$ (Hạt bay ra ngoài): hướng uốn cong của quỹ đạo hạt tự động đảo chiều $180^\circ$ đúng quy tắc bàn tay trái.
- `test_play_pause_and_reset`: Thử nghiệm phím bấm tạm dừng `[⏸ Dừng]`, phát tiếp `[▶ Tiếp tục]`, và khôi phục `[↺ Đặt lại]` hạt về vị trí nguồn phát ban đầu.
- `test_touchless_air_gestures`: Hỗ trợ thao tác không chạm: véo tay di chuyển nguồn phát hạt, vẫy chạm không chạm (Air Tap) vào các nút đổi hạt ($e^-, p^+, \alpha$), đảo chiều từ trường, tăng/giảm cảm ứng từ và vận tốc.
- `test_stamp_to_canvas_and_lessonboard`: Dập mô hình buồng từ trường vector, ký hiệu $\otimes/\odot$, quỹ đạo tròn hạt, vector vận tốc $\vec{v}$ (xanh lá), vector lực Lorentz $\vec{F}_L$ (đỏ) và thẻ công thức Cyclotron lên bảng vẽ `LessonBoard` thông qua `PhysicsTools.create_lorentz_force_diagram`.

#### 2.16. Lưới Thức Ăn Sinh Thái & Động Học Quần Thể Lotka-Volterra (`tests/test_interactive_trophic_web.py`)
- `test_trophic_web_init_and_properties`: Khởi tạo hệ sinh thái gồm 4 bậc dinh dưỡng (Sinh vật sản xuất $P$, Sinh vật tiêu thụ bậc 1 $H$, Bậc 2 $C$, Đầu bảng $A$), hệ số Lotka-Volterra chuẩn ($\alpha = 0.80, \beta = 0.02, \gamma = 0.40, \delta = 0.005$), số lượng khởi điểm Thỏ $x_0 = 100.0$, Cáo $y_0 = 30.0$, tính toán chính xác điểm cân bằng bền $x^* = \gamma/\delta = 80.0$, $y^* = \alpha/\beta = 40.0$.
- `test_lotka_volterra_rk4_dynamics`: Kiểm chứng thuật toán giải tích số vi phân phi tuyến Runge-Kutta bậc 4 (RK4):
  $$\frac{dx}{dt} = \alpha x - \beta x y, \quad \frac{dy}{dt} = \delta x y - \gamma y$$
  Hệ thống mô phỏng trơn tru dao động điều hòa tự nhiên lệch pha kinh điển (quần thể thú săn mồi trễ pha $\approx \pi/2$ so với quần thể con mồi), không bị bùng nổ hay tuyệt chủng.
- `test_conserved_quantity_approximate_invariance`: Kiểm chứng tích phân thứ nhất (hàm Liapunov bất biến) bảo toàn của hệ Lotka-Volterra:
  $$V(x, y) = \delta x - \gamma \ln(x) + \beta y - \alpha \ln(y) = \text{const}$$
  Sau nhiều bước tích phân, sai số độ lệch tương đối của $V$ dưới $2\%$.
- `test_ecological_perturbations_drought_and_disease`: Thử nghiệm các cú sốc sinh thái tương tác:
  - Hạn hán (`[⚡ Hạn hán]`): Giảm $50\%$ số lượng thực vật và con mồi, kích hoạt cảnh báo vàng và hiển thị chu kỳ hồi phục tự nhiên.
  - Dịch bệnh (`[🎯 Dịch bệnh]`): Giảm $60\%$ quần thể kẻ săn mồi, kích hoạt hiện tượng bùng nổ dân số con mồi (trophic cascade).
  - Bổ sung cá thể (`[🌾 +30 Thỏ]`, `[🦊 +20 Cáo]`): Tiêm xung số lượng cá thể vào hệ sinh thái.
  - Đặt lại (`[🔄 Đặt lại]`): Khôi phục nguyên vẹn trạng thái cân bằng sinh thái ban đầu.
- `test_parameter_tuning_and_pyramid_toggle`: Tinh chỉnh hệ số sinh sản $\alpha$ và săn mồi $\beta$, bật/tắt thẻ trực quan hóa Kim tự tháp năng lượng sinh thái.
- `test_touchless_air_gestures`: Hỗ trợ thao tác không chạm: véo tay kéo thả các đỉnh loài sinh vật trong lưới thức ăn để bố trí lại sơ đồ mạng, vẫy chạm không chạm (Air Tap) vào các nút hạn hán, dịch bệnh, kim tự tháp, đóng dấu.
- `test_stem_tools_create_trophic_pyramid_and_stamping`: Kiểm chứng hàm dựng hình vector `BiologyTools.create_trophic_pyramid` sinh 4 bậc hình thang kim tự tháp Lindeman (100% $\to$ 10% $\to$ 1% $\to$ 0.1%), nhãn năng lượng $\text{kcal/m}^2/\text{năm}$, mũi tên thất thoát nhiệt hô hấp $90\%$, dập chuẩn xác lên `LessonBoard`.
- `test_lessonboard_and_main_window_integration`: Tích hợp vòng đời hiển thị trên `LessonBoard` (`show_trophic_web`, `toggle_trophic_web`), phím tắt `Ctrl+Shift+U`, nút thanh công cụ `🌿 Lưới thức ăn`, liên kết menu nổi không chạm và loại trừ khỏi vùng tẩy xóa.

#### 2.17. Định Luật Khí Lý Tưởng & Phân Bố Vận Tốc Maxwell-Boltzmann (`tests/test_interactive_gas_laws.py`)
- `test_gas_laws_init_and_properties`: Khởi tạo xilanh khí lý tưởng với khí Helium ($m=4\text{ u}$), nhiệt độ phòng $T = 300.0\text{ K}$, tỷ lệ thể tích ban đầu $70\%$ ($V = 15.68\text{ L}$), tính toán chính xác áp suất lý thuyết theo phương trình Clapeyron-Mendeleev:
  $$P = \frac{nRT}{V} = \frac{0.5 \times 8.314 \times 300}{0.01568 \times 1000} \approx 79.5\text{ kPa}$$
  và vận tốc căn quân phương phân tử $v_{\text{rms}} = \sqrt{\frac{2k_BT}{m}} > 0$.
- `test_kinetic_particle_step_and_pressure_updates`: Mô phỏng va chạm đàn hồi hoàn hảo của các phân tử khí vi mô với các thành xilanh và piston di động:
  - Tọa độ tất cả phân tử luôn được giới hạn chặt chẽ trong $[x_{\text{left}}, x_{\text{piston}}] \times [y_{\text{top}}, y_{\text{bot}}]$.
  - Tích lũy độ biến thiên động lượng $\Delta p = \sum 2m|v_\perp|$ trên các thành chứa làm cơ sở xác định áp suất vĩ mô thực tế, lọc mượt bằng bộ lọc số đáp ứng chuẩn xác áp suất lý thuyết.
- `test_isothermal_process_boyle_mariotte`: Kiểm chứng quá trình Đẳng nhiệt (Boyle-Mariotte, $T = \text{const}$):
  - Khi khóa chế độ `ISOTHERMAL`, nhiệt độ $T$ được bảo toàn tuyệt đối khi nén piston từ $0.70 \to 0.40$ ($V \downarrow$).
  - Tích số $P \cdot V = \text{const}$ ($P_1 V_1 = P_2 V_2$) với sai số $< 0.1\%$, áp suất $P_2 > P_1$ tăng tương ứng với mức độ nén thể tích.
- `test_isobaric_process_charles`: Kiểm chứng quá trình Đẳng áp (Charles, $P = \text{const}$):
  - Khi tăng nhiệt độ chất khí từ $300\text{ K} \to 400\text{ K}$, piston tự động giãn nở tăng thể tích xilanh sao cho tỉ số $V/T = \text{const}$ ($V_1/T_1 = V_2/T_2$).
  - Áp suất $P$ giữ nguyên giá trị không đổi.
- `test_isochoric_process_gay_lussac`: Kiểm chứng quá trình Đẳng tích (Gay-Lussac, $V = \text{const}$):
  - Khi khóa chế độ `ISOCHORIC`, vị trí piston được cố định cứng (thể tích $V$ không thể thay đổi).
  - Tăng nhiệt độ chất khí $\Delta T = +150\text{ K}$ làm áp suất trong bình tăng tỉ lệ thuận: $\frac{P_1}{T_1} = \frac{P_2}{T_2}$ ($P_2 > P_1$).
- `test_gas_species_switching_and_speeds`: Hoán đổi giữa 3 loại khí: Helium ($m=4$), Nitơ ($m=28$), Argon ($m=40$):
  - Tốc độ chuyển động nhiệt $v_{\text{rms}}$ của các hạt khí nhẹ (He) nhanh hơn rõ rệt so với khí nặng (Ar), thỏa mãn tỉ số căn bậc hai khối lượng mol: $\frac{v_{\text{rms}}(\text{He})}{v_{\text{rms}}(\text{Ar})} = \sqrt{\frac{M_{\text{Ar}}}{M_{\text{He}}}} = \sqrt{10} \approx 3.162$.
- `test_touchless_air_gestures`: Hỗ trợ thao tác không chạm: véo tay (Air Pinch Drag) kéo thanh gạt piston để nén/giãn thể tích xilanh khí, vẫy chạm không chạm (Air Tap) vào các nút đun nóng `[🔥 Đun nóng]`, làm mát `[❄️ Làm mát]`, chọn khí mẫu, chuyển chế độ đẳng nhiệt/đẳng áp/đẳng tích và đặt lại.
- `test_stem_tools_create_gas_chamber_and_stamping`: Kiểm chứng hàm dựng hình vector `ChemistryTools.create_gas_chamber_diagram` sinh đầy đủ xilanh, piston kim loại, đồng hồ đo áp suất có kim chỉ, nhiệt kế thủy ngân có cột chất lỏng đỏ, các hạt phân tử có vệt vector vận tốc và thẻ phương trình $PV=nRT$, dập chuẩn xác lên `LessonBoard`.
- `test_lessonboard_and_main_window_integration`: Tích hợp vòng đời hiển thị trên `LessonBoard` (`show_gas_laws`, `toggle_gas_laws`), phím tắt `Ctrl+Shift+J`, nút thanh công cụ `🎈 Khí lý tưởng`, liên kết menu nổi không chạm và loại trừ khỏi vùng tẩy xóa.

#### 2.18. Chuẩn Độ Axit - Bazơ & Đường Cong pH Đệm Henderson-Hasselbalch (`tests/test_interactive_acid_base_titration.py`)
- `test_titration_init_and_properties`: Khởi tạo hệ chuẩn độ phân tích hóa học với axit axetic yếu $CH_3COOH$ ($pK_a = 4.76$, $C_a = 0.10\text{ M}$, $V_a = 25.0\text{ mL}$), dung dịch chuẩn bazơ $NaOH$ ($C_b = 0.10\text{ M}$), tính toán chính xác thể tích tương đương lý thuyết $V_{eq} = \frac{C_a V_a}{C_b} = 25.0\text{ mL}$, pH ban đầu $pH_0 \approx 2.88$, và chỉ thị Phenolphthalein.
- `test_weak_acid_henderson_hasselbalch_buffer_and_equivalence`: Kiểm chứng các điểm đặc trưng then chốt trên đường cong chuẩn độ axit yếu:
  - Điểm nửa tương đương (Half-equivalence point, $V = 12.5\text{ mL}$): Nồng độ axit và muối axetat bằng nhau $[CH_3COO^-] = [CH_3COOH]$, phương trình Henderson-Hasselbalch cho kết quả chính xác tuyệt đối $pH = pK_a = 4.76$.
  - Vùng đệm hóa học (Buffer capacity): Trong khoảng $V \in [10.0, 15.0]\text{ mL}$, pH biến thiên cực kỳ chậm rãi chứng minh khả năng kháng thay đổi pH của hệ đệm.
  - Điểm tương đương ($V = 25.0\text{ mL}$): Thủy phân muối bazơ yếu $CH_3COONa$ tạo môi trường kiềm nhẹ $pH = 8.73 > 7.00$.
  - Sau tương đương ($V = 40.0\text{ mL}$): Lượng dư bazơ mạnh $NaOH$ đẩy pH vọt lên vùng kiềm mạnh $pH > 12.0$.
- `test_strong_acid_equivalence_point_neutral`: Hoán đổi sang hệ axit mạnh ($HCl + NaOH$):
  - pH ban đầu: $[H^+] = 0.10\text{ M} \implies pH = 1.00$.
  - Điểm tương đương ($V = 25.0\text{ mL}$): Muối trung tính $NaCl$ không thủy phân cho $pH = 7.00$ chính xác tuyệt đối.
  - Bước nhảy pH (pH jump) dốc đứng từ $pH \approx 3 \to pH \approx 11$ chỉ trong vài giọt chuẩn độ.
- `test_first_derivative_peak_detection`: Kiểm chứng đạo hàm bậc nhất giải tích số $\frac{dpH}{dV}$:
  - Đạt đỉnh cực đại nhọn hoắt tại $V = 25.0\text{ mL}$ ($\frac{dpH}{dV} > 5.0$) so với vùng đệm phẳng ($\frac{dpH}{dV} < 0.2$), xác thực thuật toán dò tìm tự động điểm uốn tương đương.
- `test_indicator_color_transitions`: Kiểm chứng sự đổi màu của các chất chỉ thị màu axit-bazơ:
  - Phenolphthalein: Không màu ở môi trường axit/trung tính ($pH < 8.2$) $\to$ hồng cánh sen/đỏ fuchsia rực rỡ ở môi trường bazơ ($pH > 8.2$).
  - Metyl Da Cam (Methyl Orange): Đỏ ở $pH < 3.1$ $\to$ da cam $\to$ vàng ở $pH > 4.4$.
  - Chỉ thị vạn năng (Universal Indicator): Biến đổi tuần tự qua 7 sắc cầu vồng theo thang pH chuẩn từ đỏ $\to$ cam $\to$ vàng $\to$ lục $\to$ lam $\to$ tím.
- `test_touchless_air_gestures`: Hỗ trợ thao tác không chạm: véo tay trượt cột chất lỏng trên buret để điều chỉnh thể tích dung dịch chuẩn thêm vào, vẫy chạm không chạm (Air Tap) vào các nút nhỏ giọt `[💧 +0.5 mL]`, xả nhanh `[💧 +2.0 mL]`, nhỏ tự động `[▶ Nhỏ tự động]`, đổi chỉ thị và đổi hệ axit.
- `test_stem_tools_create_titration_curve_and_stamping`: Kiểm chứng hàm dựng hình vector `ChemistryTools.create_titration_curve_diagram` sinh đầy đủ giá đỡ buret, buret thủy tinh có vạch chia, khóa van stopcock, bình tam giác Erlenmeyer chứa dung dịch đổi màu, hệ trục tọa độ đường cong chuẩn độ pH, điểm tương đương $E(V_{eq}, pH_{eq})$ và điểm nửa tương đương $pH = pK a$, dập chuẩn xác lên `LessonBoard`.
- `test_lessonboard_and_main_window_integration`: Tích hợp vòng đời hiển thị trên `LessonBoard` (`show_titration`, `toggle_titration`), phím tắt `Ctrl+Shift+Q`, nút thanh công cụ `🧪 Chuẩn độ pH`, liên kết menu nổi không chạm và loại trừ khỏi vùng tẩy xóa.

#### 2.19. Dao Động Điều Hòa, Tắt Dần & Cộng Hưởng Cơ Học (`tests/test_interactive_harmonic_oscillator.py`)
- `test_harmonic_oscillator_energy_conservation_shm`: Kiểm chứng tích phân số Runge-Kutta bậc 4 (RK4) giải phương trình $m\ddot{x} + b\dot{x} + kx = F_0\cos(\omega t)$. Trong chế độ dao động điều hòa tự do không ma sát ($\zeta=0$), cơ năng $E = E_k + E_p$ được bảo toàn nghiêm ngặt với sai số dưới 0.1% qua nhiều chu kỳ liên tiếp.
- `test_damping_regimes_and_envelope`: Kiểm chứng 3 chế độ tắt dần:
  - Dưới tới hạn ($\zeta < 1.0$): Dao động tắt dần bám sát đường bao mũ $x(t) \le A_0 e^{-\gamma t}$.
  - Tới hạn ($\zeta = 1.0$): Vật về vị trí cân bằng nhanh nhất mà không đổi chiều dao động.
  - Quá tới hạn ($\zeta > 1.0$): Dịch chuyển chậm về VTCB do cản lớn.
- `test_resonance_lorentzian_curve`: Kiểm chứng biên độ dao động cưỡng bức đạt cực đại sắc nhọn khi tần số kích thích tiệm cận tần số riêng $\omega \approx \omega_0 = \sqrt{k/m}$.
- `test_air_pinch_and_stamping`: Véo tay điều chỉnh độ biến dạng lò xo và dập sơ đồ cơ học lên bảng qua `PhysicsTools.create_harmonic_oscillator_diagram`.

#### 2.20. Định Luật Phóng Xạ & Chuỗi Phân Rã Hạt Nhân (`tests/test_interactive_radioactive_decay.py`)
- `test_radioactive_decay_law_theory`: Kiểm chứng định luật phân rã phóng xạ $N(t) = N_0 2^{-t/T_{1/2}}$ qua 5 đồng vị phóng xạ thực nghiệm (Carbon-14, Radon-222, Iodine-131, Uranium-238, Cobalt-60).
- `test_monte_carlo_lattice_simulation`: Tích hợp mạng 100 hạt nhân Monte Carlo rời rạc với xác suất phân rã mỗi chu kỳ $P = 1 - 2^{-\Delta t/T_{1/2}}$, xuất hiện hạt phát xạ $\alpha, \beta^-, \gamma$ và xung máy đếm Geiger-Müller.
- `test_radiometric_dating_calculator`: Tính toán chính xác tuổi mẫu vật từ tỷ lệ số hạt còn lại: $t = -T_{1/2} \log_2(N/N_0)$.
- `test_stamping_and_lessonboard_integration`: Dập sơ đồ phóng xạ lên bảng qua `PhysicsTools.create_radioactive_decay_diagram`, phím tắt `Ctrl+Alt+R`, và loại trừ khỏi vùng tẩy xóa.

#### 2.21. Tốc Độ Phản Ứng, Năng Lượng Hoạt Hóa Arrhenius & Cân Bằng Le Chatelier (`tests/test_interactive_chemical_kinetics.py`)
- `test_chemical_kinetics_rate_equations`: Giải hệ phương trình vi phân tốc độ phản ứng thuận nghịch: $v_f = k_f [A][B], v_r = k_r [C][D]$, tại trạng thái cân bằng $v_f = v_r$ và thương số phản ứng $Q_c = K_c$.
- `test_arrhenius_activation_energy`: Kiểm chứng phương trình Arrhenius $k(T) = A e^{-E_a/RT}$. Chất xúc tác hạ rào cản hoạt hóa $\Delta E_{\text{cat}}$ đồng thời trên cả chiều thuận và nghịch, tăng tốc độ phản ứng mà giữ nguyên hằng số cân bằng $K_c$.
- `test_le_chatelier_perturbations`: Nhiễu loạn cân bằng theo nguyên lý Le Chatelier (bơm thêm chất phản ứng/sản phẩm hoặc tăng/giảm nhiệt độ cho phản ứng tỏa/thu nhiệt) làm dịch chuyển cân bằng đúng chiều nhiệt động học.
- `test_stamping_and_lessonboard_integration`: Dập biểu đồ thế năng và nồng độ qua `ChemistryTools.create_reaction_kinetics_diagram`, phím tắt `Ctrl+Alt+K`.

#### 2.22. Mật Mã Học RSA & Số Học Đồng Dư (`tests/test_interactive_cryptography.py`)
- `test_extended_euclidean_and_modular_inverse`: Kiểm chứng thuật toán Euclid mở rộng tìm ước số chung lớn nhất $ax + by = \gcd(a,b)$ và tính nghịch đảo nhân modulo $d = e^{-1} \pmod{\phi(n)}$ (ví dụ với $e=7, \phi=60 \implies d=43$).
- `test_rsa_key_generation_and_presets`: Tạo khóa công khai $(n, e)$ và khóa bí mật $(n, d)$ từ tích 2 số nguyên tố $n = p \cdot q$ và phi hàm Euler $\phi(n) = (p-1)(q-1)$ trên 4 cặp số nguyên tố mẫu chuẩn.
- `test_rsa_encryption_decryption_roundtrip`: Mã hóa và giải mã thông điệp $M$: $C = M^e \pmod n$, $M' = C^d \pmod n$ chứng minh $M' \equiv M$ với 100% độ chính xác số học.
- `test_diffie_hellman_shared_secret`: Giao thức trao đổi khóa Diffie-Hellman trên số nguyên tố $p=23$ và căn nguyên thủy $g=5$, tính khóa chung của Alice và Bob độc lập $K_A = B^a \pmod p = K_B = A^b \pmod p = 2$.
- `test_air_tap_and_pinch_interactions`: Véo tay điều chỉnh thông điệp $M$ trên vòng tròn số học đồng dư, chạm không chạm chuyển đổi chế độ và dập sơ đồ vector lên bảng qua `InformaticsTools.create_rsa_crypto_diagram`, phím tắt `Ctrl+Alt+C`.
