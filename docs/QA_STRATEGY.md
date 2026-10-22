# CHIẾN LƯỢC ĐẢM BẢO CHẤT LƯỢNG PHẦN MỀM (QA STRATEGY & TEST HARNESS)

**Dự án:** AIRTECH AI — Nền tảng dạy học thông minh tương tác không chạm bằng AI  
**Phiên bản:** 1.0.0 — Windows Local-First Desktop Edition  
**Công cụ kiểm thử chính:** `pytest`, `pytest-cov`, `pytest-qt`, `unittest.mock` (cho giới hạn phần cứng camera ảo)  
**Tần suất thực thi:** Kiểm thử hồi quy tự động toàn bộ trước mỗi lần cập nhật.

---

## 1. NGUYÊN TẮC VÀ MỤC TIÊU CHIẾN LƯỢC QA

1. **Tuân thủ Tuyệt đối Quy chuẩn Zero Mock (Zero Mock Policy):**
   Mọi thuật toán cốt lõi (1€ Filter, RDP, Catmull-Rom Bézier spline, Shape Snapping, băm PBKDF2, mã hóa Fernet, nén file `.airteach`, xuất PDF vector) bắt buộc phải kiểm thử bằng tính toán toán học thực tế trên mảng số liệu, không dùng hàm giả lập trả về chuỗi cố định.
2. **Kiểm soát Độ trễ và Hiệu năng Phần cứng (Performance & SLA Enforcement):**
   Mọi hàm xử lý khung hình hoặc lọc nét vẽ đều có bài kiểm tra giới hạn thời gian (Execution Time Constraint):
   - Thuật toán 1€ Filter: $\le 0.5\text{ ms}$ trên mỗi điểm tọa độ.
   - Thuật toán RDP + Bézier: $\le 5.0\text{ ms}$ cho chuỗi 200 điểm nét vẽ.
   - Thuật toán nắn hình học: $\le 10.0\text{ ms}$ cho toàn bộ hình.
3. **Độ bao phủ Toàn diện Đa Tầng (Multi-tier Test Coverage):**
   - **Tầng 1 (Unit Testing):** Kiểm tra từng hàm toán học, bộ lọc, dịch chuyển trạng thái FSM, mã hóa và băm mật mã.
   - **Tầng 2 (Integration Testing):** Kiểm tra liên kết giữa Database SQLite, Face Service, Quản lý tài liệu PDF/PPTX và Trình đóng gói `.airteach`.
   - **Tầng 3 (UI Component Testing):** Kiểm tra tạo widget PySide6, phân tầng canvas $Z=0, 10, 20$, đồng bộ thanh công cụ, và phát tín hiệu Signal/Slot an toàn.
   - **Tầng 4 (System Regression Testing):** Kiểm tra toàn bộ luồng sử dụng từ đăng ký, đăng nhập, vẽ nét, chuyển trang đến xuất bản PDF.

---

## 2. PHÂN BỔ BỘ KIỂM THỬ (TEST SUITE STRUCTURE)

Hệ thống kiểm thử gồm 5 tập tin chính với **238 bài kiểm thử hoàn chỉnh**:

```
tests/
├── test_core_algorithms.py        # 47 tests: 1€ Filter, RDP, Catmull-Rom, Shape Snapping, FSM, Landmark Norm
├── test_phase3_advanced_ui.py      # 33 tests: Replay Controller, LessonBoardCanvas, Layers Z, Multi-selection
├── test_services_and_storage.py    # 71 tests: YuNet, SFace, FaceProfileService, PIN PBKDF2, Fernet, Documents, .airteach ZIP
├── test_svg_assets_and_trainer.py  # 14 tests: SVG Vector Assets, BackgroundTemplateGenerator, Icon integrity
└── test_ui_components.py          # 73 tests: LoginWindow, MainWindow, CameraWidget, ChatPanel, SettingsDialog
```

---

## 3. MA TRẬN TRUY XUẤT YÊU CẦU ĐẾN KIỂM THỬ (REQUIREMENT TRACEABILITY MATRIX)

| Mã Yêu Cầu | Tính năng / Phân hệ | Bài kiểm thử tương ứng (Test Cases) | Tiêu chí Nghiệm thu Đạt |
|---|---|---|---|
| **REQ-01** | Theo dõi 21 điểm mốc 3D bàn tay | `test_core_algorithms.py::TestLandmarkNormalization` | Tọa độ chuẩn hóa bất biến theo tỷ lệ bàn tay $d_{\text{scale}}$ |
| **REQ-02** | Khử rung nét vẽ bằng 1€ Filter | `test_core_algorithms.py::TestOneEuroFilter` | Giảm $\ge 60\%$ jitter, thời gian lọc $< 0.1\text{ ms}$ |
| **REQ-03** | Làm mịn Bézier & biến thiên độ dày | `test_core_algorithms.py::TestBezierAndRDP` | Nét liên tục $C^1$, giảm $\ge 50\%$ số điểm mẫu |
| **REQ-04** | Nắn hình học tự động (Shape Snapping)| `test_core_algorithms.py::TestShapeSnapping` | Nhận diện đúng đoạn thẳng, hình tròn ($C>0.78$), hình chữ nhật, tam giác |
| **REQ-05** | Máy trạng thái cử chỉ (6 gestures FSM)| `test_core_algorithms.py::TestGestureStateMachine` | Chống nhảy nét nhầm 3-frame debounce, tỷ lệ chính xác $>95\%$ |
| **REQ-06** | Xác thực khuôn mặt offline (YuNet+SFace) | `test_services_and_storage.py::TestFaceProfileService` | Nhận diện đúng giáo viên đã đăng ký, từ chối mặt lạ |
| **REQ-07** | Đăng ký tài khoản (Face + PIN & PIN-only) | `test_services_and_storage.py::TestEnrollmentFlows` | Đăng ký thành công cả 2 chế độ, không bị kẹt form |
| **REQ-08** | Bảo mật SQLite, Fernet & PBKDF2 | `test_services_and_storage.py::TestSecurityAndCrypto` | Vector được mã hóa, mật khẩu salt 100,000 vòng |
| **REQ-09** | Nhập tài liệu đa định dạng (PDF/PPTX) | `test_services_and_storage.py::TestDocumentImport` | Dựng trang ảnh chất lượng cao 144 DPI |
| **REQ-10** | Đóng gói bài giảng chuẩn `.airteach` | `test_services_and_storage.py::TestAirteachPackaging` | Nén/giải nén ZIP nguyên vẹn manifest, strokes, pages |
| **REQ-11** | Bảng giảng tương tác phân tầng đồ họa | `test_phase3_advanced_ui.py::TestLessonBoardCanvas` | Phân tầng đúng $Z=0, 10, 20$, không lem nét giữa các trang |
| **REQ-12** | Bộ phát lại bài giảng (Replay Controller) | `test_phase3_advanced_ui.py::TestReplayController` | Phát lại chính xác theo timestamp ms, tua 1x-4x |
| **REQ-13** | Bộ tạo mẫu nền học tập chuyên dụng | `test_svg_assets_and_trainer.py::TestBackgroundTemplates` | Dựng đúng 4 mẫu: Trắng, Ô li, Kẻ ngang, Oxy |
| **REQ-14** | Chuyển giao phiên & Đăng xuất đa giáo viên| `test_ui_components.py::TestSessionAndLogout` | Đăng xuất an toàn, mở lại màn hình đăng nhập |

---

## 4. QUY TRÌNH CHẠY KIỂM THỬ VÀ BẢO ĐẢM HỒI QUY (REGRESSION EXECUTION)

Lệnh thực thi kiểm thử chuẩn hóa trên môi trường Windows PowerShell:

```powershell
# Chạy toàn bộ 238 bài kiểm thử với báo cáo chi tiết
python -m pytest tests/ -v

# Chạy kiểm thử đo độ trễ hiệu năng
python -m pytest tests/test_core_algorithms.py -k "test_filter_performance"

# Chạy kiểm thử với đo lường độ bao phủ mã nguồn
python -m pytest --cov=src tests/
```
