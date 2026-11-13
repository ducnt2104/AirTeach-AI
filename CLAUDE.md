# AirTeach AI — Design System & Figma-to-Code Rules (`CLAUDE.md`)

## 1. Design Token Definitions

AirTeach AI uses a high-contrast dark studio palette optimized for classroom projectors, interactive flat panels, and desktop monitors (`1366x768` to `4K`).

### Semantic Color Tokens
| Token Name | Hex Value | RGB | Purpose & Usage |
| :--- | :--- | :--- | :--- |
| `BG_DARK` (`--bg-canvas`) | `#0F172A` | `(15, 23, 42)` | Primary Smartboard canvas, Header Bar, Bottom Drawing Dock, and input wells |
| `SURFACE_DARK` (`--bg-surface`) | `#1E293B` | `(30, 41, 59)` | Left Camera Dock (`270px`), Right AI Dock (`330px`), and inactive buttons |
| `BORDER_SLATE` (`--border-subtle`) | `#334155` | `(51, 65, 85)` | `1px` structural panel dividers, card borders, and inactive controls |
| `PRIMARY_INDIGO` (`--accent-primary`) | `#4F46E5` | `(79, 70, 229)` | Primary active buttons, selected tabs, and AI action triggers |
| `SECONDARY_TEAL` (`--accent-secondary`) | `#06B6D4` | `(6, 182, 212)` | Teacher identity badges, formula highlights, and telemetry labels |
| `NEON_CYAN` (`--ink-cyan`) | `#00F5D4` | `(0, 245, 212)` | Default digital ink, 21-bone MediaPipe hand skeleton connections, and `DRAW` cursor ring |
| `BRIGHT_YELLOW` (`--ink-yellow`) | `#FEE440` | `(254, 228, 64)` | MediaPipe 21 hand landmark joints and `HOVER` target ring |
| `ALERT_ROSE` (`--status-danger`) | `#F43F5E` | `(244, 63, 94)` | Recording indicator, `[Phân Tích Vùng]` selection box, and `ERASE` circle (`r = 45px`) |
| `TEXT_LIGHT` (`--text-primary`) | `#F8FAFC` | `(248, 250, 252)` | Primary Vietnamese UTF-8 UI typography and slide body text |
| `TEXT_MUTED` (`--text-secondary`) | `#94A3B8` | `(148, 163, 184)` | Secondary labels, hints, and inactive status text |

### Smartboard Ink Palette (6 Circular Swatches, `22x22px`)
- Cyan: `#00F5D4`
- Yellow: `#FEE440`
- White: `#FFFFFF`
- Green: `#38B000`
- Magenta: `#FF007F`
- Orange: `#FF6D00`

---

## 2. Typography & Vietnamese UTF-8 Rules

- **Primary Font Stack**: `Segoe UI` (`C:\Windows\Fonts\segoeui.ttf`), fallback `Arial` (`arial.ttf`), `Tahoma` (`tahoma.ttf`), and `Segoe UI Symbol` (`seguisym.ttf`) for mathematical operators.
- **Zero-Tofu Rule**: Never use rare multicolor emoji glyphs that fail to render in hardware-accelerated `egui` or `Qt` font atlases; use clean bracketed Vietnamese labels (`[Đăng xuất]`, `[Phân Tích Vùng]`, `[Bút vẽ]`, `[Tẩy]`, `[Laser]`).
- **Mathematical Formatting**: Render formulas using clean Mathematical Unicode (`²`, `³`, `√`, `Δ`, `∫`, `∑`, `⇒`, `≠`, `≤`, `≥`) via `DocumentLoader::format_latex_to_unicode_math` rather than raw LaTeX control sequences.

---

## 3. Studio Layout Architecture ("Clean Studio UI" — 80% Canvas Area)

1. **Single Top Header Bar (`Height: 46px`, Background `#0F172A`)**:
   - **No Second Sub-Toolbar**: Never add a second horizontal sub-toolbar below the Header Bar.
   - **Left Group**: `[Đăng xuất]` | `GV: <Tên Giáo viên>` | `[Mở PDF/Slide]` | `[ < ] Slide X/Y [ > ]` | `[Bật/Tắt Lưới Caro]`.
   - **Middle**: Flexible stretch spacer (`addStretch(1)` in Qt / `Layout::right_to_left` split in `egui`) so left and right controls never collide.
   - **Right Group**: `[Ghi Âm Bài Giảng]` | `[Ẩn/Hiện Camera]` | `[Trợ Lý AI (Đóng/Mở)]`.
2. **Minimalist Left Camera Dock (`Fixed Width: 270px`, Collapsible)**:
   - **4:3 Camera Preview (`250x188px`)** with live 21-point hand skeleton (**Neon Cyan `#00F5D4`** bones + **Bright Yellow `#FEE440`** joints) and 1 concise status line (`"Trạng thái: Đang Vẽ / Di chuyển / Tẩy / Chờ tay"`).
   - **Camera Controls**: `[Cam 0]` `[Cam 1]` + `Ngược sáng (Auto-HDR)` checkbox (`Gamma 0.6` + `CLAHE clipLimit=2.5`).
   - **4 Compact Gesture Badges (`2x2` grid)**: `[1 Ngón: Vẽ]`, `[2 Ngón: Di chuyển]`, `[Chụm tay: Click]`, `[5 Ngón: Tẩy]`.
   - **Compact Mic VU Meter** at the bottom.
3. **Clean Center Smartboard Canvas (`>= 80%` Viewport Area)**:
   - Top-left button `[Ẩn nội dung Slide để lấy bảng trống 100%]` toggles the slide card so the teacher can claim 100% blank board space at any time.
   - Zero floating status banners, zero fake `MATH_{expr}` annotations, and zero bottom subtitle overlays on the canvas.
4. **Right Gemini AI Dock (`Fixed Width: 330px`, Collapsible)**:
   - Compact API Key row (auto-collapses into `[Đổi API Key]` once saved).
   - 3 Pedagogical Action Buttons: `[1. Giải & Chữa Bài Trên Bảng]`, `[2. Tóm Tắt & Tạo Trắc Nghiệm]`, `[3. Hỏi Đáp Sư Phạm]`.
   - Clean scrollable AI response area (`gemini-2.5-flash`).
5. **Bottom Drawing Dock (`Height: 54px`, Background `#0F172A`)**:
   - `[Phân Tích Vùng]` | `[Bút vẽ]` | `[Tẩy]` | `[Laser]` | `[Nắn hình: TẮT/BẬT]` | 6 circular color buttons (`22x22px`) + thickness slider (`2px - 14px`) | `[Hoàn tác (Undo)]` | `[Làm lại (Redo)]` | `[Xóa Bảng]`.

---

## 4. Component & File Organization

- **Rust Native Portable Executable (`AirTeach-AI.exe`)**:
  - `src/main.rs` & `src/app.rs`: Application entry point, state machine, and `eframe`/`wgpu` loop.
  - `src/ui/auth_view.rs`: Two-Tab Security Gate (`TAB 1: ĐĂNG KÝ TÀI KHOẢN MỚI` -> `TAB 2: ĐĂNG NHẬP`).
  - `src/ui/toolbar_view.rs`: Single `46px` Header Bar and `54px` Bottom Drawing Dock.
  - `src/ui/canvas_view.rs`: `270px` Left Camera Dock + Center Smartboard Canvas.
  - `src/ui/chat_view.rs`: `330px` Right Gemini 2.5 Flash Pedagogical Assistant Dock.
  - `src/core/camera.rs`, `src/core/hand_tracker.rs`, `src/core/gestures.rs`: Backlight-resilient camera enhancement (`Gamma 0.6` + `CLAHE`), MediaPipe 21-point hand tracking (`confidence = 0.55`), and finger gesture FSM.
  - `scripts/mediapipe_hand_worker.py`: Persistent headless MediaPipe co-processor.
- **PySide6 Desktop Implementation (`main.py`)**:
  - Self-contained PySide6 + OpenCV + MediaPipe + `google-genai` studio implementation mirroring the exact same 4-Module architecture and persistent `data/users_db.json` + `data/face_embeddings/<username>.npy` storage.
