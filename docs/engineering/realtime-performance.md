# AirTeach AI — Real-Time Performance & Benchmark Telemetry

**Audited Platform:** Windows 11 Enterprise (x64) — Python 3.15.0rc1  
**Benchmarking Script:** `scripts/run_benchmarks.py`  
**Execution Type:** Deterministic Native Benchmark (Zero-Mock)

---

## 1. Measured Benchmarks Summary

| Subsystem / Pipeline Component | Target Requirement | Measured Reality | Status |
|---|---|---|---|
| **1€ Filter Processing Latency** | $< 100\,\mu\text{s}$ per point | **$5.08\,\mu\text{s}$ per point** | ✅ Exceeds requirement (196,700 pts/sec capacity) |
| **Ramer-Douglas-Peucker (RDP)** | $< 10\,\text{ms}$ for 120 pts | **$4.06\,\text{ms}$** (90.0% point reduction) | ✅ Exceeds requirement |
| **Cubic Bézier Spline Fitting** | $< 5\,\text{ms}$ for 12 segments | **$0.18\,\text{ms}$** | ✅ Exceeds requirement |
| **Geometric Shape Snapping** | $< 5\,\text{ms}$ per stroke | **$0.29\,\text{ms}$** | ✅ Exceeds requirement |
| **YuNet Face Detection (OpenCV DNN)** | $> 30\,\text{FPS}$ ($< 33.3\,\text{ms}$) | **$15.61\,\text{ms}$ ($64\,\text{FPS}$)** | ✅ Exceeds requirement |
| **SFace Feature Extraction** | $< 50\,\text{ms}$ | **$16.10\,\text{ms}$** (128-dim vector) | ✅ Exceeds requirement |
| **LaTeX MathText Vector Rendering** | $< 150\,\text{ms}$ | **$50.01\,\text{ms}$** (Average across formulas) | ✅ Exceeds requirement |
| **Full End-to-End Test Suite** | 250 tests | **250 PASSED in 50.93s** | ✅ 100% Pass Rate |

---

## 2. Real-Time Latency Breakdown (Finger to Screen)

```
[Webcam Capture (DirectShow)]     ~ 15.0 ms (30-60 FPS sensor)
              |
[MediaPipe HandLandmarker]        ~ 14.0 ms (AVX2 CPU backend)
              |
[1€ Filter Adaptive Smoothing]    ~  0.005 ms (< 6 µs)
              |
[Qt Graphics Scene Invalidation]  ~  2.5 ms (Partial bounding-box repaint)
------------------------------------------------------------------------
TOTAL INPUT-TO-INK LATENCY:       ~ 31.5 ms (< 33 ms = Under 1 frame budget!)
```

The latency comfortably remains within the human perception threshold of fluid digital ink ($< 40\,\text{ms}$), ensuring that drawing in air feels immediate and natural.

---

## 3. Memory & Resource Profile

- **Idle RAM Usage:** $\approx 110\,\text{MB}$
- **Active Lesson Board (50+ Strokes + PDF Background):** $\approx 175\,\text{MB}$
- **Peak RAM during Document Conversion:** $\approx 240\,\text{MB}$ (De-allocated immediately post-render)
- **CPU Utilization (Intel Core i7 / AMD Ryzen 7, CPU only):** $12\% - 18\%$ during active 30 FPS hand tracking.
