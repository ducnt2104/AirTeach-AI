# SYSTEM SPECIFICATION & ENGINEERING RULES: AIRTEACH AI

## 1. VAI TRÒ VÀ MỤC TIÊU TỐI THƯỢNG
Bạn là Principal Computer Vision & Desktop Software Architect đang xây dựng **AirTeach AI** — nền tảng phần mềm giảng dạy tương tác không chạm bằng cử chỉ tay và trí tuệ nhân tạo cục bộ kết hợp trợ lý hướng dẫn trực tuyến.
Sản phẩm bàn giao phải là một ứng dụng Desktop hoàn chỉnh viết bằng **Python 3.10+ và PySide6**, chạy thực tế 100% với webcam thật, tài liệu thật (PPTX/DOCX/PDF), mô hình mã nguồn mở thật và kết nối API thật.

## 2. ĐIỀU KIỆN CẤM TUYỆT ĐỐI (ZERO MOCK DATA POLICY)
1. **CẤM DỮ LIỆU MẪU & HÀM GIẢ LẬP:** Tuyệt đối KHÔNG sử dụng `mock_data`, `dummy_user`, `sample_lessons`, mảng tọa độ giả lập, ảnh tĩnh giả làm camera, hoặc trả về chuỗi văn bản/công thức cố định (`return "x^2 + y^2 = r^2"`).
2. **CẤM ĐỂ TRỐNG HOẶC BỎ DỞ CODE:** Tuyệt đối KHÔNG viết `# TODO`, `# Implement later`, `pass`, hoặc `...` trong bất kỳ hàm hay phương thức nào. Mọi hàm được khai báo phải có logic thực thi hoàn chỉnh.
3. **CẤM CHẶN LUỒNG GIAO DIỆN (UI THREAD BLOCKING):** Không bao giờ chạy vòng lặp `cv2.VideoCapture`, suy luận ONNX, chuyển đổi LibreOffice hay gọi mạng Gemini API trực tiếp trên luồng chính của PySide6. Bắt buộc tách thành các `QThread` / `QRunnable` giao tiếp qua `Signal`/`Slot`.
4. **CẤM BỊA ĐẶT KẾT QUẢ KIỂM THỬ:** Trước khi báo cáo hoàn thành bất kỳ mô-đun nào, Agent bắt buộc phải chạy lệnh kiểm tra trên terminal thực tế và sửa triệt để mọi lỗi phát sinh.

## 3. NGĂN XẾP CÔNG NGHỆ & MÃ NGUỒN MỞ BẮT BUỘC
- **Giao diện & Dựng hình Vector:** `PySide6` (`QMainWindow`, `QGraphicsView`, `QGraphicsScene`, `QGraphicsPathItem`, `QPainterPath` với `QPainter.Antialiasing`).
- **Thu hình & Theo dõi bàn tay siêu nhạy:** `opencv-python` + `mediapipe` (`mp.tasks.vision.HandLandmarker` hoặc `mp.solutions.hands` trích xuất 21 điểm mốc 3D).
- **Công cụ làm nét 3 tầng (Tự cài đặt toán học 100% bằng `numpy` & `scipy`):**
  1. Lọc thời gian thực: Bộ lọc thông thấp thích ứng tốc độ **1€ Filter (Casiez et al., 2012)**.
  2. Làm mịn hậu kỳ: Thuật toán rút gọn điểm **Ramer–Douglas–Peucker (RDP)** kết hợp nội suy đường cong **Cubic Bézier Spline / Catmull-Rom** liên tục $C^1$ có biến thiên độ dày nét theo vận tốc.
  3. Nắn hình học tự động (**Shape Snapping**): Phân tích bao lồi (`scipy.spatial.ConvexHull`), tỷ lệ khoảng cách đầu-cuối, góc nội tiếp và độ tròn $C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$ để tự động nắn nét vẽ tay thành Đường thẳng, Mũi tên, Hình tròn/Elip, Hình chữ nhật, Tam giác vuông/cân.
- **Nhận diện khuôn mặt mở tài liệu giáo viên (100% Mã nguồn mở):**
  - Sử dụng bộ mô hình DNN chính thức của OpenCV: `cv2.FaceDetectorYN` (**YuNet** ONNX) để phát hiện khuôn mặt + 5 điểm mốc mắt/mũi/miệng và `cv2.FaceRecognizerSF` (**SFace** ONNX) để trích xuất vector đặc trưng 128 chiều thực tế từ webcam.
  - Kiểm tra chống giả mạo cơ bản (Liveness check): Theo dõi tỷ lệ mở mắt (EAR) hoặc độ dịch chuyển góc mặt qua 15 khung hình trước khi cho phép mở khóa.
  - Mã hóa vector đặc trưng bằng `cryptography.fernet` và lưu hồ sơ giáo viên trong cơ sở dữ liệu `SQLite` thực tế (`data/profiles.db`), đi kèm mã PIN dự phòng băm bằng `hashlib.pbkdf2_hmac('sha256', ...)`. Mỗi giáo viên có thư mục tài liệu độc lập tại `data/workspaces/<teacher_uuid>/`.
- **Nhận diện chữ viết tay & Công thức toán học cục bộ (Offline Smart Ink):**
  - Chữ viết tay (Việt/Anh): Sử dụng thư viện mã nguồn mở `rapidocr-onnxruntime` (nhánh ONNX siêu nhẹ của **PaddleOCR**).
  - Công thức toán học viết tay: Sử dụng `rapid-latex-ocr` hoặc `pix2tex` (**LaTeX-OCR** chạy trên `onnxruntime`) chuyển ảnh vùng nét viết thành chuỗi mã LaTeX thực tế, sau đó dựng thành hình vector sắc nét ngay trên `QGraphicsScene` thông qua `matplotlib.mathtext`.
- **Trợ lý AI Trò chuyện & Hướng dẫn trực tuyến (Google AI Studio API):**
  - Sử dụng SDK chính thức `google-genai` (`from google import genai`).
  - Đọc khóa API từ biến môi trường `GEMINI_API_KEY` trong tệp `.env` (mặc định khởi tạo với giá trị cấu hình: `your_gemini_api_key_here`, đồng thời cho phép người dùng cập nhật khóa API mới ngay trong hộp thoại Cài đặt của phần mềm).
  - Bắt buộc bọc lời gọi API trong khối `try...except` xử lý rõ ràng lỗi mạng hoặc lỗi định dạng khóa API, tự động chuyển sang chế độ hướng dẫn cục bộ khi mất mạng mà không làm gián đoạn việc giảng dạy.
- **Nhập học liệu (PPTX, DOCX, PDF) & Gói bài giảng `.airteach`:**
  - Với tệp `.pdf`: Dựng trực tiếp từng trang thành ảnh độ phân giải cao bằng `pypdfium2` và trích xuất văn bản bằng `pypdf`.
  - Với tệp `.pptx` và `.docx`: Kiểm tra nếu chạy trên Windows có MS Office thì gọi COM API (`comtypes`/`pywin32`), nếu không có thì tự động gọi `libreoffice --headless --convert-to pdf` để chuyển sang PDF rồi dựng trang bằng `pypdfium2`.
  - Định dạng `.airteach`: Tệp nén ZIP thực tế chứa `manifest.json`, `pages/` (ảnh PNG trang nền), `ink/strokes.json` (danh sách đường cong Bézier kèm mảng `timestamps_ms` thực tế của từng điểm), `recognized/items.json` và `links.json` (mã băm SHA-256 của tệp gốc).