# BÁO CÁO KIỂM TOÁN ĐỒ HỌA VÀ TRẢI NGHIỆM NGƯỜI DÙNG (GRAPHICS & UX AUDIT)

**Dự án:** AirTeach AI  
**Cập nhật:** 20/10/2026  
**Định hướng thiết kế:** Academic Teaching Workstation — Chuyên nghiệp, thanh lịch, tối ưu cho máy chiếu lớp học và màn hình tương tác.

---

## 1. PHÂN TÍCH THỊ GIÁC VÀ BẰNG CHỨNG THỰC TẾ (VISUAL AUDIT)

Dựa trên các ảnh chụp màn hình thực tế kết xuất từ ứng dụng:
1. **Cửa sổ Đăng nhập (Login Window - `screen_login_win.png`):**
   - Bố cục 2 cột cân xứng: Khung camera nhận diện khuôn mặt trực tiếp bên trái, Bàn phím số PIN tương tác bên phải.
   - Thẻ hiển thị sắc thái Google Academic Theme (#f8f9fa nền, #ffffff thẻ, viền #dadce0, nút xanh lá #34a853 cho đăng nhập PIN và xanh dương #1a73e8 cho đăng ký).
   - Đạt độ tương phản cao, phông chữ Segoe UI rõ ràng.
2. **Cửa sổ Giảng dạy chính (Main Teaching Window - `screen_main_win.png`):**
   - Không gian bảng trắng rộng rãi (1920x1080 canvas), tách biệt hoàn toàn với vùng camera giám sát.
   - Thanh công cụ điều hướng trên đỉnh đang dùng văn bản kèm ký tự emoji (`📚 Thư viện`, `◀ Trang trước`, `Trang sau ▶`, `Nhận dạng chữ (OCR)`, `∑ Nhận dạng công thức`, `⏱ Replay`, `🗑 Xóa nét`).
   - Cần thay thế emoji bằng bộ icon SVG vector chuyên dụng để nâng cao tính đồng bộ và thẩm mỹ công nghệ giáo dục.
3. **Thư viện bài giảng (Library & Prep Widget - `screen_prep_win.png`):**
   - Trạng thái rỗng (Empty state) thân thiện, có nút bấm nhanh "Tạo Bài giảng Mới" và "Nhập PPTX / PDF / Word".
   - Cung cấp 4 mẫu nền bảng viết chuẩn học đường: Trắng, Kẻ ô li học sinh, Dòng kẻ ngang, Hệ trục tọa độ Oxy.
4. **Hộp thoại Cài đặt (Settings Dialog - `screen_settings_win.png`):**
   - 3 tab mạch lạc: AI API (Gemini API key & model), Camera (chọn webcam, độ phân giải, lật gương, CLAHE), Vẽ & Làm nét (độ dày nét, màu mực, độ nhạy 1€ Filter, RDP epsilon, bật/tắt nắn hình học).

---

## 2. KẾ HOẠCH HÀNH ĐỘNG NÂNG CẤP ĐỒ HỌA (ACTION PLAN)

- [x] Tạo bộ biểu tượng vector SVG bản quyền riêng trong thư mục `assets/icons/`.
- [x] Tạo bộ tranh minh họa 6 cử chỉ điều khiển không chạm trong `assets/gestures/`.
- [x] Tạo sơ đồ kiến trúc và chuỗi xử lý làm nét 3 tầng trong `assets/diagrams/`.
- [x] Bổ sung thanh công cụ vẽ chuyên dụng trên bảng (Floating Canvas Teaching Toolbar) gồm Bút viết, Con trỏ Laser, Tẩy, Chọn vùng, Bảng màu mực nhanh, và Độ dày nét.
- [x] Tích hợp Hộp thoại luyện tập và hiệu chuẩn cử chỉ (`GestureTrainerDialog`) có hình ảnh hướng dẫn trực quan.
