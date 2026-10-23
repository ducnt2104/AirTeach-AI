# AIRTEACH AI — KIẾN TRÚC BẢO MẬT KHÓA API & CÁCH LY TÀI KHOẢN (API SECURITY SPECIFICATION)
**Phiên bản:** AirTeach AI Security v2.5.0  
**Ngày phát hành:** 23/10/2026  
**Chuẩn tuân thủ:** Zero Plain-Text Secrets & Strict Multi-Tenant Account Isolation

---

## 1. NGUYÊN TẮC BẢO MẬT CỐT LÕI (SECURITY PRINCIPLES)

1. **Không bao giờ lưu khóa dưới dạng văn bản rõ (Zero Plain-Text):**
   - Khóa Google Gemini API và các khóa dịch vụ bên thứ ba tuyệt đối không được ghi vào tệp cơ sở dữ liệu SQLite (`data/profiles.db`), tệp `.env`, tệp cấu hình JSON, hoặc nhật ký hệ thống.
2. **Cách ly tuyệt đối giữa các tài khoản giáo viên (Per-Teacher Namespace Isolation):**
   - Mỗi giáo viên sở hữu không gian lưu trữ khóa riêng biệt trong kho bảo mật hệ điều hành. Giáo viên A đăng nhập không thể đọc, sử dụng hoặc xóa khóa của Giáo viên B.
3. **Che dấu & Thanh lọc khóa (Key Masking & Redaction):**
   - Trên giao diện người dùng, khóa luôn được che dạng mật khẩu hoặc mặt nạ bảo mật (`AIzaSy••••••••••••4xQ9`).
   - Mọi thông báo ngoại lệ (exceptions), thông điệp lỗi hoặc nhật ký gửi về hệ thống đều đi qua bộ lọc biểu thức chính quy để loại bỏ hoàn toàn các chuỗi khóa khớp mẫu.
4. **Không làm nghẽn luồng giao diện khi kiểm tra khóa (Zero GUI Blocking):**
   - Quá trình kiểm tra kết nối API qua mạng được đưa vào worker luồng phụ (`QRunnable` trên `QThreadPool`). Giao diện người dùng vẫn phản hồi mượt mà trong suốt thời gian kết nối mạng.

---

## 2. KIẾN TRÚC LƯU TRỮ VÀ MÃ HÓA

```
┌────────────────────────────────────────────────────────────────────────┐
│                   GIAO DIỆN CÀI ĐẶT (SETTINGS DIALOG)                  │
│   - Nhập khóa API: AIzaSy...                                           │
│   - Nút kiểm tra kết nối: _ApiTestRunnable (QThreadPool)               │
│   - Hiển thị che mờ: AIzaSy••••••••••••4xQ9                            │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│         DỊCH VỤ QUẢN LÝ BẢO MẬT (TeacherCredentialService)             │
│                                                                        │
│   [Kiểm tra HĐH Windows]                                               │
│   ├── Khả dụng ──► Windows Credential Manager (WinVaultKeyring)        │
│   │                Namespace: AirTeachAI                               │
│   │                Username: teacher_<teacher_id>_<provider>           │
│   │                Bảo vệ bởi: Windows DPAPI (Hardware Encrypted)      │
│   │                                                                    │
│   └── Dự phòng ──► Kho mã hóa cục bộ AES-GCM / Fernet                  │
│                    Key: PBKDF2-HMAC-SHA256 (Machine GUID + Salt,       │
│                                            100,000 iterations)         │
│                    Tệp lưu: data/workspaces/<teacher_id>/.sec_vault    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Cấu hình phi bí mật (Non-secrets)
┌───────────────────────────────────▼────────────────────────────────────┐
│         CƠ SỞ DỮ LIỆU SQLITE CÁ NHÂN (data/profiles.db)                │
│                                                                        │
│   BẢNG: teachers_ai_settings                                           │
│   - teacher_id (TEXT PRIMARY KEY)                                      │
│   - provider (TEXT)                  - model_name (TEXT)               │
│   - feature_summary (INTEGER)        - feature_quiz (INTEGER)          │
│   - feature_smart_ink (INTEGER)      - draw_mode (TEXT)                │
│   - camera_lighting_mode (TEXT)      - pinch_threshold (REAL)          │
│   - updated_at (TEXT)                                                  │
│   ★ HOÀN TOÀN KHÔNG CÓ CỘT API_KEY TRONG BẢNG NÀY!                     │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. CƠ CHẾ BẢO VỆ CHI TIẾT

### 3.1. Windows Credential Manager (`WinVaultKeyring`)
- Dịch vụ sử dụng thư viện `keyring` chuẩn với backend chính thức trên Windows là `WinVaultKeyring` (sử dụng thư viện `pywin32-ctypes`).
- Dữ liệu được mã hóa tự động ở cấp độ nhân hệ điều hành thông qua Data Protection API (DPAPI), gắn chặt với thông tin đăng nhập của người dùng Windows.
- Khi người dùng xóa khóa (`delete_api_key`), lệnh gọi `keyring.delete_password` sẽ xóa hoàn toàn bản ghi khỏi Windows Vault.

### 3.2. Cơ chế Mã hóa Dự phòng (PBKDF2 + Fernet)
- Khi ứng dụng chạy trong môi trường không có keyring (ví dụ: máy tính bị khóa policy hoặc phiên Linux container), dịch vụ tự động chuyển sang cơ chế dự phòng:
  1. Đọc mã định danh phần cứng máy tính (Machine GUID / MAC Address).
  2. Kết hợp với muối mật (Salt) cố định và chạy thuật toán `hashlib.pbkdf2_hmac` với hàm băm SHA-256 qua 100.000 vòng lặp.
  3. Mã hóa khóa bằng chuẩn Fernet (AES-128 trong chế độ CBC kết hợp xác thực HMAC-SHA256).
  4. Lưu tệp nhị phân đã mã hóa trong thư mục workspace độc lập của giáo viên: `data/workspaces/<teacher_id>/.vault_data`.

### 3.3. Thanh lọc Khóa khỏi Nhật ký & Ngoại lệ
- Biểu thức chính quy phát hiện khóa:
  ```python
  _REDACT_REGEX = re.compile(r"AIza[0-9A-Za-z\-_]{20,}")
  ```
- Mọi hàm log hoặc gửi lỗi đều gọi qua:
  ```python
  @staticmethod
  def redact_text(text: str) -> str:
      if not text:
          return ""
      return TeacherCredentialService._REDACT_REGEX.sub("[REDACTED_API_KEY]", str(text))
  ```

### 3.4. Kiểm tra Kết nối Mạng Bất đồng bộ với Chẩn đoán Tiếng Việt
Phương thức `test_gemini_connection(api_key, model_name)` gửi một prompt ngắn mẫu (`"Xin chào! Vui lòng phản hồi 'OK'."`) đến Gemini API và ánh xạ các mã lỗi HTTP sang thông báo tiếng Việt trực quan cho giáo viên:
- **Thành công (200 OK):** `"Kết nối thành công! Mô hình sẵn sàng phục vụ giảng dạy."`
- **Lỗi khóa sai (400 / API_KEY_INVALID):** `"Khóa API không hợp lệ. Vui lòng kiểm tra lại khóa sao chép từ Google AI Studio."`
- **Lỗi hết hạn mức (429 / RESOURCE_EXHAUSTED):** `"Khóa API đã vượt quá hạn mức sử dụng (Quota Exceeded). Vui lòng kiểm tra trên Google AI Studio."`
- **Lỗi mạng (ConnectionError):** `"Không thể kết nối đến máy chủ Google Gemini. Vui lòng kiểm tra kết nối Internet hoặc cấu hình mạng trường học."`

---

## 4. KẾT QUẢ KIỂM THỬ BẢO MẬT THỰC TẾ (`tests/test_api_security.py`)

Bộ kiểm thử an toàn thông tin chuyên sâu đã được thực thi và vượt qua 100%:

```
tests/test_api_security.py::test_per_teacher_isolation PASSED            [ 25%]
tests/test_api_security.py::test_key_masking PASSED                      [ 50%]
tests/test_api_security.py::test_key_redaction PASSED                    [ 75%]
tests/test_api_security.py::test_per_teacher_ai_settings_persistence PASSED [100%]
```

1. **`test_per_teacher_isolation`:** Xác minh Giáo viên 1 (`gv_toan_01`) lưu khóa `KEY_A` và Giáo viên 2 (`gv_ly_02`) lưu khóa `KEY_B`. Khi truy vấn, mỗi giáo viên chỉ đọc được đúng khóa của mình. Khi Giáo viên 1 xóa khóa, khóa của Giáo viên 2 vẫn được bảo toàn nguyên vẹn.
2. **`test_key_masking`:** Khóa `AIzaSyB345678901234567890123456784xQ9` được che chuẩn thành `AIzaSy••••••••••••4xQ9`.
3. **`test_key_redaction`:** Chuỗi văn bản lỗi chứa khóa bị thay thế thành `[REDACTED_API_KEY]`, ngăn chặn rò rỉ vào file log hoặc telemetry.
4. **`test_per_teacher_ai_settings_persistence`:** Cấu hình mô hình, tùy chọn tính năng và chế độ vẽ được lưu trữ và khôi phục chính xác từ cơ sở dữ liệu SQLite mà không chứa bất kỳ chuỗi nhạy cảm nào.
