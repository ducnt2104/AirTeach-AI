# AIRTEACH AI — CONTINUOUS AUTONOMOUS EXECUTION CONTINUATION FILE

**Created / Updated:** 2026-10-23T20:35:00+07:00  
**Project:** AirTeach AI — Desktop Air Gesture Interactive Teaching Workstation  
**Platform:** Windows 10/11, Python 3.10+, PySide6 (Qt 6.12.0), OpenCV 5.0+, MediaPipe 1.1+  
**Current Test Suite Status:** **326 / 326 PASSED (100%)** in ~85s  

---

## 1. EXECUTIVE STATUS & CONTEXT SUMMARY

AirTeach AI is an enterprise-grade desktop application for touchless, air-gesture teaching. It combines:
- Real-time webcam tracking with adaptive lighting (CLAHE, Gamma compensation)
- Multi-hand tracking and coordination (concurrent dual-hand drawing, secondary tool selection)
- Continuous C¹ cubic Bézier spline vector ink rendering with velocity-based width variation (1€ Filter & RDP)
- Autonomous shape snapping (line, arrow, circle, ellipse, rectangle, triangle)
- STEM teaching tools (Oxy coordinate axes, mathematical function plotting, physical force vectors, logic gates, chemical structures, flowcharts, code blocks)
- In-Air Floating Action Menu with progressive air-click gesture triggers
- Progressive multi-hand stroke replay synchronization
- Document ingestion & high-fidelity vector PDF export
- Teacher profile security with Fernet AES-256 encrypted biometric face embeddings and PBKDF2-SHA256 PIN fallback

---

## 2. STRICT CONSTRAINTS & COMPLIANCE

1. **Zero Mock Data Policy (`RULE[AGENTS.md]`):**
   - No mock coordinates, no dummy camera frames, no static formula returns (`"x + y = z"` strictly prohibited).
   - All AI models (PaddleOCR / RapidOCR ONNX, LaTeX-OCR ONNX, OpenCV YuNet & SFace) execute actual inference or report honest empty results with 0.0 confidence when not detected.
2. **Zero UI Blocking:**
   - No `cv2.VideoCapture`, ONNX inference, LibreOffice COM conversions, or PDF rendering runs synchronously on the PySide6 main thread.
   - All asynchronous work runs via `QThread` (`HandTrackingWorker`), `QRunnable` (`QThreadPool`), or non-blocking event queues.
3. **Continuous Execution Protocol:**
   - Autonomous engineering loop continues iteratively without waiting or stopping unless explicit `DỪNG HẲN` is issued.

---

## 3. ARCHITECTURAL MAP & CRITICAL COMPONENTS

| Layer | Source Files | Description & Key Responsibilities |
|---|---|---|
| **Multi-Hand Tracking & Gestures** | `src/core/multi_hand_coordinator.py`<br>`src/core/hand_gesture_engine.py` | Multi-hand tracking FSM, distance-based tracking ID assignment, concurrent dual drawing, air-click debounce, pinch detection |
| **Vector Ink Engine** | `src/core/stroke_beautifier.py`<br>`src/ui/lesson_board_canvas.py` | 1€ low-pass filter, Ramer-Douglas-Peucker (RDP), C¹ cubic Bézier splines, variable pressure width, shape snapping |
| **STEM Tools Engine** | `src/ui/stem_tools.py`<br>`src/ui/main_window.py` | Cartesian Oxy axes, function graphing (LaTeX/numexpr), free-body force vectors, circuit schematics, organic chemistry rings, flowcharts, code snippets |
| **In-Air Floating Menu** | `src/ui/floating_action_menu.py` | Radial & panel layouts, hover dwell timer, air-pinch/air-click trigger, root and STEM submenus |
| **Progressive Replay** | `src/ui/replay_controller.py` | Multi-hand progressive Bézier subsegment animation, pause/resume, playback speed control, full stroke restoration |
| **Document & PDF Export** | `src/services/document_manager.py`<br>`src/ui/main_window.py` | High-fidelity vector scene rendering to multi-page PDF (capturing background, ink, STEM diagrams, and text overlays) |
| **Teacher Authentication** | `src/services/face_auth.py` | YuNet + SFace face recognition, 15-frame EAR anti-spoofing liveness check, Fernet vector encryption, PBKDF2 PIN fallback |
| **Per-Teacher AI Workspace** | `src/services/ai_service.py`<br>`src/ui/ai_workbench.py` | Isolated per-teacher API key storage, Google GenAI SDK integration, offline fallback |

---

## 4. RECENT MAJOR ACCOMPLISHMENTS

1. **Zero Mock Hardening in `smart_ink_ocr.py`:**
   - Eliminated hardcoded fallback `return "x + y = z"`.
   - Now safely falls back to geometric symbol parser or honest empty formula when no handwriting is detected.
2. **Document Manager Process Safety:**
   - Protected MS Office COM calls with `try...finally: app.Quit()` preventing zombie `POWERPNT.EXE` and `WINWORD.EXE` background processes.
   - Guarded output directory creation with `output_pdf_path.parent.mkdir(parents=True, exist_ok=True)`.
3. **Multi-Page STEM Lifecycle & Unified Undo/Redo:**
   - Solved stuck graphics bug: added `_page_stem_items` dictionary in `MainWindow`.
   - Hidden diagrams on stashed pages, restored diagrams when switching back to their original page.
   - Unified LIFO Undo/Redo stack interleaving both vector ink strokes and STEM diagram items.
4. **Scene-Accurate High-Fidelity PDF Export:**
   - Upgraded `MainWindow._on_export_pdf` to render the actual `QGraphicsScene` using antialiased vector rasterization onto 1920x1080 canvas.
   - Automatically supports multi-page imported documents and standalone whiteboard sessions.
5. **Dynamic Camera HUD Telemetry:**
   - In `_QtCameraWorker`, dynamically calculates font scale and text bounding box with `cv2.getTextSize` to prevent badge clipping on low-resolution feeds.
   - In `CameraWidget`, guarded against negative/zero widget dimensions during window resize events.
6. **Test Suite Expansion:**
   - 8 comprehensive tests in `tests/test_stem_replay_integration.py` covering STEM menu, in-air triggers, concurrent progressive dual replay, multi-hand skeleton HUD, latency budget (<= 15ms), page lifecycle isolation, unified undo/redo, and PDF export.
   - Overall suite: **326 passed tests** across all 15 test suites.

---

## 5. NEXT STEPS FOR UPCOMING ITERATIONS

1. **Layer D: Memory & Resource Leak Check:**
   - Verify that repeated loading and switching between large 50+ page PDF/PPTX presentations cleans up `QPixmap` and `QImage` memory on garbage collection.
2. **Layer E: Multi-Hand Gesture Customization UI:**
   - In Settings Dialog, add a dedicated tab to let teachers map individual hand roles (e.g. Left hand = Eraser/Menu only, Right hand = Pen only) and adjust air-click pinch sensitivity.
3. **Layer G: STEM Interactive Manipulation:**
   - Allow teachers to use the `SELECT` hand gesture to move and resize previously placed STEM diagrams directly on the canvas without using a mouse.
