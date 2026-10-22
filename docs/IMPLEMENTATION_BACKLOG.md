# DANH MỤC CÔNG VIỆC VÀ TIẾN ĐỘ THI CÔNG (IMPLEMENTATION BACKLOG)

**Dự án:** AirTeach AI  
**Cập nhật:** 20/10/2026  
**Chính sách:** Hoàn thiện tuần tự theo thứ tự ưu tiên P0 $\to$ P6, không để lại stub hay mã giả.

---

## BẢNG TIẾN ĐỘ THI CÔNG CHI TIẾT

| Mã công việc | Nhiệm vụ kỹ thuật | Độ ưu tiên | Trạng thái | Ghi chú & Kết quả |
|---|---|---|---|---|
| **TASK-01** | Kiểm toán toàn diện dự án & khởi tạo các Agent Skills | P0 | **HOÀN THÀNH** | 11 agent skills tại `.agents/skills/` |
| **TASK-02** | Khắc phục tương thích MediaPipe 1.x (Tasks API) | P1 | **HOÀN THÀNH** | Tương thích song song Tasks API và solutions |
| **TASK-03** | Khắc phục ngắt luồng an toàn cho Camera và Dialog | P1 | **HOÀN THÀNH** | Tránh treo ứng dụng khi đóng cửa sổ |
| **TASK-04** | Xây dựng bộ tài nguyên vector SVG đầy đủ | P2 | **HOÀN THÀNH** | 33 SVG tại `assets/icons/`, `gestures/`, `diagrams/`, `illustrations/` |
| **TASK-05** | Tích hợp SVG Icon vào Toolbar và Canvas Controls | P2 | **HOÀN THÀNH** | Thay thế toàn bộ ký tự emoji trên thanh công cụ |
| **TASK-06** | Xây dựng thanh công cụ giảng dạy tích hợp (Dual Toolbars) | P3 | **HOÀN THÀNH** | Chế độ Bút, Tẩy, Laser, Vùng chọn, Bảng 5 màu, Độ dày nét, Undo/Redo |
| **TASK-07** | Hộp thoại Huấn luyện & Hiệu chuẩn cử chỉ | P3 | **HOÀN THÀNH** | `GestureTrainerDialog` với 7 cử chỉ, live skeleton, HUD đo lường |
| **TASK-08** | Hộp thoại Xem & Duyệt Smart Ink Review | P4 | **HOÀN THÀNH** | `SmartInkReviewDialog` crop nét viết + render công thức Matplotlib MathText |
| **TASK-09** | Bổ sung các bài test tự động cho UI/SVG/Trainer | P5 | **HOÀN THÀNH** | 231 / 231 tests PASSED (100%), 0 lỗi, 0 hồi quy |
| **TASK-10** | Đóng gói bản phân phối Windows Standalone (PyInstaller) | P6 | **HOÀN THÀNH** | `dist/AirTeachAI/AirTeachAI.exe` (29.03 MB), bundled models & SVG verified 100% |
