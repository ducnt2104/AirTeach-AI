# AirTeach AI — Requirements Traceability Matrix (RTM)

**Specification Reference:** `Ban_Thuyet_Minh_AirTeach_AI_Nang_Cap.docx`  
**Standard:** Verification Traceability from Specification Item to Automated Test (`REQ-xxx` $\rightarrow$ `TEST-xxx`)

---

## 1. Traceability Matrix

| Req ID | Specification Requirement (from Thuyết Minh) | Implementation File | Verification Test | Status |
|---|---|---|---|---|
| **REQ-AUTH-01** | Local offline face detection with YuNet ONNX & SFace recognition | `src/services/face_profile_service.py` | `test_services_and_storage.py::TestTeacherProfileService::test_enroll_teacher_face` | ✅ VERIFIED |
| **REQ-AUTH-02** | PBKDF2 PIN fallback (4-6 digits) with unique `teacher_code` | `src/services/face_profile_service.py` | `test_services_and_storage.py::TestTeacherProfileService::test_authenticate_by_teacher_code_and_pin` | ✅ VERIFIED |
| **REQ-AUTH-03** | Brute-force rate-limiting lockout (5 attempts $\rightarrow$ 30s lock) | `src/services/face_profile_service.py` | `test_services_and_storage.py::TestTeacherProfileService::test_brute_force_lockout_mechanism` | ✅ VERIFIED |
| **REQ-AUTH-04** | Isolated workspace directory per teacher (`data/workspaces/<uuid>/`) | `src/services/face_profile_service.py` | `test_services_and_storage.py::TestTeacherProfileService::test_workspace_initialization_marker` | ✅ VERIFIED |
| **REQ-CAM-01** | DirectShow webcam acquisition with CLAHE backlight compensation | `src/core/hand_gesture_engine.py` | `test_core_algorithms.py::TestHandTrackingWorker` | ✅ VERIFIED |
| **REQ-TRACK-01** | MediaPipe 21 3D landmarks with $L_{\text{hand}} = \|P_0 - P_9\|$ scale factor | `src/core/hand_gesture_engine.py` | `test_core_algorithms.py::TestHandGestureFSM` | ✅ VERIFIED |
| **REQ-FSM-01** | 6-gesture FSM (DRAW, POINT, PAUSE, SELECT, SWIPE, CONFIRM) | `src/core/hand_gesture_engine.py` | `test_core_algorithms.py::TestHandGestureFSM::test_draw_gesture` ... `test_confirm_hold_timer` | ✅ VERIFIED |
| **REQ-FSM-02** | Anti-false drawing debouncing ($\ge 3$ frames, confidence $\ge 0.75$) | `src/core/hand_gesture_engine.py` | `test_core_algorithms.py::TestHandGestureFSM::test_debounce_prevents_premature_transition` | ✅ VERIFIED |
| **REQ-SMOOTH-01** | 1€ Filter speed-adaptive low-pass filtering ($f_c = f_{c,\min} + \beta \|\dot{x}\|$) | `src/core/stroke_beautifier.py` | `test_core_algorithms.py::TestOneEuroFilter2D::test_smoother_than_noisy_input` | ✅ VERIFIED |
| **REQ-SMOOTH-02** | RDP simplification ($\epsilon = 2.5$) & $C^1$ Cubic Bézier Spline | `src/core/stroke_beautifier.py` | `test_core_algorithms.py::TestRamerDouglasPeucker` & `TestFitCubicBezierSpline` | ✅ VERIFIED |
| **REQ-SNAP-01** | Geometric shape snapping (Line, Arrow, Circle, Rectangle, Triangle) | `src/core/stroke_beautifier.py` | `test_core_algorithms.py::TestGeometricShapeSnapper::test_detects_straight_line` ... `test_detects_triangle` | ✅ VERIFIED |
| **REQ-OCR-01** | Offline math formula recognition & LaTeX RGBA vector rendering | `src/services/smart_ink_ocr.py` | `test_services_and_storage.py::TestSmartInk::test_latex_renderer_rgba` | ✅ VERIFIED |
| **REQ-DOC-01** | Multi-format document import (PDF via pypdfium2, PPTX/DOCX via COM/LibreOffice) | `src/services/document_manager.py` | `test_services_and_storage.py::TestDocumentManager::test_pdf_import` | ✅ VERIFIED |
| **REQ-PKG-01** | `.airteach` bundle format (manifest, pages, ink, recognized, links) | `src/services/document_manager.py` | `test_services_and_storage.py::TestDocumentManager::test_airteach_save_and_open_roundtrip` | ✅ VERIFIED |
| **REQ-REPLAY-01** | Millisecond timestamped stroke replay with speed multipliers | `src/ui/lesson_board.py` | `test_phase3_advanced_ui.py::TestReplayController::test_speed_multipliers` | ✅ VERIFIED |
| **REQ-UI-01** | Layered canvas (Background, Ink, Selection, Laser HUD) with multi-level Undo/Redo | `src/ui/lesson_board.py` | `test_phase3_advanced_ui.py::TestLessonBoardCanvas::test_stroke_drawing_and_undo_redo` | ✅ VERIFIED |
