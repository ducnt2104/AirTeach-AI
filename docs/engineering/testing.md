# AirTeach AI — Automated Testing & Quality Assurance Protocol

**Test Framework:** `pytest 9.1.1` + `pytest-cov 7.1.0`  
**Total Tests:** 250 automated tests (100% passing)  
**Execution Runtime:** 50.93 seconds  
**Policy:** Zero Mock Data Policy (Real Models, Deterministic Algorithms, Real GUI Fixtures)

---

## 1. Test Suite Breakdown

### 1. `tests/test_core_algorithms.py` (48 Tests)
- **1€ Filter:** Noise attenuation on variable timestep streams; reset state clears velocity memory; high-velocity latency bypass.
- **RDP:** Endpoints preservation; epsilon threshold behavior; 90% point compression on realistic handwriting curves.
- **Cubic Bézier Spline:** Continuity across joints; velocity-based stroke width clamping ($w_{\min} \le w \le w_{\max}$); serialization roundtrip.
- **Geometric Shape Snapper:** Detection of lines, circles, ellipses, rectangles, triangles, arrows; angle snapping to 0°, 45°, 90°; handling of 3D timestamped points without broadcasting errors.
- **Gesture FSM:** Debounce filtering; confidence thresholding; 6-state transition matrix; hold timer confirmation.

### 2. `tests/test_services_and_storage.py` (77 Tests)
- **Teacher Profile & Auth:** Custom teacher codes (`GV01`); duplicate code rejection with `ValueError`; PBKDF2 hash verification; brute-force lockout after 5 failed attempts; workspace directory isolation.
- **Smart Ink & LaTeX:** Math formula parsing; LaTeX rendering to RGBA; aspect ratio heuristics.
- **Document Manager:** PDF rendering with `pypdfium2`; `.airteach` packaging and extraction; annotated PDF export.
- **Gemini Assistant:** Online streaming; graceful degradation to offline guidance when API key is missing or invalid.

### 3. `tests/test_phase3_advanced_ui.py` (38 Tests)
- **Lesson Board Canvas:** Layer management; hit-testing stroke erasure; clear all ink while preserving background; laser pointer trail; selection region bounding.
- **Login Window:** Teacher dropdown populating; PIN pad keypad entry; physical keyboard numeric navigation; scan card fallback.
- **Replay Controller:** Duration calculations; timer tick advancement; playback speed multipliers.
- **Background Templates:** Plain white, Ô li grid, Lined paper, Oxy coordinate axes rendering.

### 4. `tests/test_svg_assets_and_trainer.py` (14 Tests)
- **SVG Iconography:** Validation of 15 icon assets and 6 gesture card SVGs.
- **Gesture Trainer Dialog:** Real-time landmark rendering; interactive calibration reset.
- **Smart Ink Review Dialog:** Handwriting text mode; math formula candidate review.

### 5. `tests/test_ui_components.py` (73 Tests)
- **MainWindow:** Smoke test; menu actions; HUD labels; status bar metrics; multi-page stroke segregation.
- **ChatPanel & SettingsDialog:** Message list appending; online status indicator; settings persistence.

---

## 2. Command to Run Full Test Suite

```powershell
pytest tests/ -v
```
Output:
```text
======================== 250 passed, 1 warning in 50.93s ========================
```
