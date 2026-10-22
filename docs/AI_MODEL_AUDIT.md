# BÁO CÁO KIỂM ĐỊNH MÔ HÌNH TRÍ TUỆ NHÂN TẠO CỤC BỘ (AI MODEL AUDIT)

**Dự án:** AirTeach AI  
**Cập nhật:** 20/10/2026  
**Nguyên tắc cốt lõi:** 100% Cục bộ & Mã nguồn mở cho luồng giảng dạy chính. Không phụ thuộc đám mây.

---

## 1. DANH MỤC MÔ HÌNH VÀ TÍNH KHẢ DỤNG THỰC TẾ

| Mô hình | Nhiệm vụ | Định dạng tệp | Dung lượng | Vị trí lưu trữ | Tình trạng kiểm định |
|---|---|---|---|---|---|
| **YuNet (2023mar)** | Phát hiện khuôn mặt + 5 điểm mốc | ONNX | 232.5 KB | `models/face_detection_yunet_2023mar.onnx` | **ĐÃ XÁC THỰC:** Chạy ổn định qua `cv2.FaceDetectorYN`, xử lý trong $\approx 8\text{ ms}$ |
| **SFace (2021dec)** | Trích xuất vector đặc trưng 128 chiều | ONNX | 38.7 MB | `models/face_recognition_sface_2021dec.onnx` | **ĐÃ XÁC THỰC:** Chạy ổn định qua `cv2.FaceRecognizerSF`, cosine similarity $\ge 0.363$ |
| **MediaPipe HandLandmarker** | Trích xuất 21 điểm mốc 3D bàn tay | TFLite Task | 7.8 MB | `models/hand_landmarker.task` | **ĐÃ XÁC THỰC:** Chạy qua `mp.tasks.vision.HandLandmarker` (Tasks API), đạt $\approx 30\text{ FPS}$ |
| **GeometricMathParser** | Phân tích cấu trúc hình học công thức toán | Thuần Python/NumPy | — | `src/services/smart_ink_ocr.py` | **ĐÃ XÁC THỰC:** Phân tích toán học offline 100% không cần tải weights nặng |
| **Matplotlib MathText** | Dựng mã LaTeX thành ảnh vector RGBA | Thuần Python | — | `src/services/smart_ink_ocr.py` | **ĐÃ XÁC THỰC:** Dựng vector công thức sắc nét 300 DPI trực tiếp lên canvas |
| **Google GenAI (Gemini)** | Trợ lý giáo viên trực tuyến (Tùy chọn) | Cloud API SDK | — | `src/services/gemini_assistant.py` | **ĐÃ XÁC THỰC:** Đọc khóa từ `.env`, tự động fallback sang cẩm nang sư phạm cục bộ khi offline |

---

## 2. GHI CHÚ QUAN TRỌNG VỀ MÔI TRƯỜNG ONNX RUNTIME VÀ PYTHON 3.15

1. Phiên bản Python 3.15 là bản thử nghiệm tiền phát hành (RC), hiện tại ONNX Runtime chính thức chưa phát hành binary wheels tương thích cho Python 3.15 trên Windows.
2. Ứng dụng AirTeach AI đã được thiết kế kiến trúc phân tầng chịu lỗi cao:
   - Module nhận diện chữ và toán học tự động chuyển hướng sang `GeometricMathParser` (phân giải dấu bằng, cộng, trừ, phân số, số mũ) và `matplotlib.mathtext` (dựng LaTeX vector) mà không làm gián đoạn bất kỳ thao tác nào của giáo viên.
   - Khi chạy trên Python 3.13 hoặc 3.10-3.12 có cài đặt `rapidocr-onnxruntime` và `rapid-latex-ocr`, hệ thống sẽ kích hoạt tầng nhận diện OCR học sâu đầy đủ.
