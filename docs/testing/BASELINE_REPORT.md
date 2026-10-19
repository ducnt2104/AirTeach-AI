# AIRTEACH AI — BASELINE TEST EXECUTION REPORT

**Document ID:** BASELINE-AIRTEACH-2026-V1  
**Execution Date:** October 2026  
**Platform:** Windows 11 (64-bit)  
**Python Runtime:** CPython 3.15.0rc1 (with pytest 9.1.1, pluggy 1.6.0)  
**Hardware Profile:** Intel Core i5 / 16 GB RAM / No external GPU required  

---

## 1. Summary of Baseline Test Run

A full automated test execution across all four test suites was executed using `pytest` from the root environment.

### Global Test Results
| Test Suite File | Tests Collected | Passed | Failed | Skipped | Runtime (s) |
|---|---|---|---|---|---|
| `tests/test_core_algorithms.py` | 134 | 134 | 0 | 0 | 12.18s |
| `tests/test_services_and_storage.py` | 52 | 52 | 0 | 0 | 15.34s |
| `tests/test_ui_components.py` | 22 | 22 | 0 | 0 | 8.45s |
| `tests/test_phase3_advanced_ui.py` | 28 | 28 | 0 | 0 | 2.48s |
| **Total** | **236** | **236** | **0** | **0** | **38.45s** |

**Result:** `236 passed, 1 warning in 38.45s`  
*(Warning was standard `google-genai` typing deprecation warning for `_UnionGenericAlias` under future Python 3.17).*

---

## 2. Granular Test Suite Breakdown

### A. Core Algorithms (`test_core_algorithms.py`)
- **1€ Filter:** Verified single-channel (`OneEuroFilter`) and 2D (`OneEuroFilter2D`) adaptive cutoff frequency. Proved jitter reduction on noisy static input and responsiveness preservation during high-speed transitions.
- **Ramer–Douglas–Peucker (RDP):** Verified polyline reduction without distortion on collinear, sinusoidal, and closed curves. Tested epsilon sensitivity and vertex preservation.
- **Cubic Bézier Spline:** Tested Catmull-Rom tangent computation, C1 continuity across segment knots, velocity-to-width mapping, and boundary conditions for strokes with $N < 2$, $N = 2$, and $N > 100$.
- **Geometric Shape Snapper:** Verified classification and canonical geometry generation for:
  - `LINE`: Horizontal, vertical, diagonal, with 0°/45°/90° angle snapping.
  - `ARROW`: V-shaped terminal fork detection ($< 55^\circ$).
  - `CIRCLE` & `ELLIPSE`: Convex hull shoelace area and roundness metric $C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2} \ge 0.80$.
  - `RECTANGLE`: 4-corner polygon approximation with $90^\circ$ angular deviation scoring.
  - `TRIANGLE`: 3-vertex polygon approximation with side-length compactness analysis.
  - `FREEFORM`: Non-geometric curves with fallback.
- **Hand Gesture FSM:** Verified all 6 states (`POINT`, `DRAW`, `PAUSE`, `SELECT`, `SWIPE_LEFT`, `SWIPE_RIGHT`, `CONFIRM`), hand size normalization ($L_{\text{hand}} = \|P0 - P9\|$), debounce counter (3 frames), swipe velocity window ($250\text{ ms}$), and thumbs-up hold timer ($300\text{ ms}$).

### B. Services & Storage (`test_services_and_storage.py`)
- **Face Profile Service:** SQLite table initialization, PBKDF2 PIN hashing (200,000 iterations), Fernet encryption/decryption of 128-D feature embeddings, cosine distance thresholding, and liveness landmark motion check across 15 frames.
- **Document Manager:** PDF page rendering at $2\times$ scale via `pypdfium2`, text extraction via `pypdf`, `.airteach` bundle creation (ZIP archive with SHA-256 integrity check), manifest serialization, and annotated PDF export.
- **Telemetry Logger:** Real-time CSV logging, timestamp delta calculation, FPS computation, and summary statistics generation.
- **Gemini Assistant:** Graceful offline detection, prompt engineering for lesson guidance, canvas snapshot explanation, and API key management.

### C. UI & Interaction Components (`test_ui_components.py` & `test_phase3_advanced_ui.py`)
- **Lesson Board:** Scene coordinate mapping, laser pointer movement and visibility, freehand stroke preview and commit, stroke removal, selection box, text overlay, and LaTeX PNG overlay.
- **Advanced Canvas (`LessonBoardCanvas`):** 4 vector background templates (Plain White, Oli grid, Lined paper, Oxy axes), Z-layering (Z=0, 10, 20), undo/redo stacks, eraser hit testing, and high-DPI pixmap export.
- **Replay Controller:** Duration computation from stroke timestamps, play/pause state transitions, time scrubbing, speed multipliers ($1\times, 2\times, 4\times$), and reset.
- **Login Window:** Teacher dropdown population from SQLite, PIN keypad input, facial recognition integration, and demo mode fallback.
- **Side Panels:** 2-tab dock widget for AI Assistant chat and live camera telemetry / 4-corner calibration.

---

## 3. Discovered Vulnerabilities & Baseline Deficiencies

1. **Multi-page state divergence in `MainWindow`:**
   Although `LessonBoard` and `DocumentManager` individually pass their tests, the integration logic in `MainWindow` lacked multi-page stroke buffering:
   - Moving from page $k$ to $k+1$ failed to stash strokes of page $k$.
   - Opening an existing `.airteach` bundle extracted page images and parsed strokes, but did not render them on the active scene.
   - Saving `.airteach` bundles saved only the current page's strokes, causing loss of annotations on other pages.
2. **Missing Offline Geometric Math Fallback:**
   `SmartInkRecognizer` relied exclusively on external ONNX packages (`rapidocr_onnxruntime`, `rapid_latex_ocr`). In environments where those libraries are missing, no math conversion was possible.
3. **Traceability of Physical Experiments:**
   Baseline test suite proved mathematical correctness on synthetic trajectories; empirical measurement on physical test equipment (per Section VI.3 of the project specification) was documented as pending physical verification.
