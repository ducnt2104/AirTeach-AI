---
name: airteach-qt-performance
description: >-
  Specialist skill for profiling, memory management, zero-GUI-blocking architectures,
  QThreadPool concurrency, and bounded frame queues in AirTeach AI.
---

# AirTeach AI — Qt Performance & Low-Latency Architecture Skill

## 1. Zero-GUI-Blocking Concurrency Model
The PySide6 main UI thread must strictly handle user events and screen painting:
- `QThread 1 (Camera & Vision)`: Runs `cv2.VideoCapture` + MediaPipe HandLandmarker + Gesture FSM at 30 FPS.
- `QThread 2 (Face Auth)`: Runs YuNet + SFace when logging in or enrolling.
- `QThreadPool (Background Tasks)`: Runs OCR rasterization, LibreOffice conversions, PDF export, and Gemini API queries.

## 2. Latency Budget Allocation (Target <= 45 ms)
- Camera frame acquisition: $\approx 10\text{ ms}$
- MediaPipe HandLandmarker inference: $\approx 18\text{ ms}$
- 1€ Filter + FSM state transition: $\approx 1\text{ ms}$
- Qt Signal delivery & QGraphicsScene paint: $\approx 8\text{ ms}$
- Total finger-to-ink pipeline latency: $\le 37\text{ ms}$ ($> 27\text{ FPS}$).

## 3. Memory & Resource Safeguards
- Always call `cap.release()`, `landmarker.close()`, and `profile_svc.close()` when windows close.
- Bounded deques (`maxlen=30`) for time-series landmark tracking prevent unbounded heap growth.
- Use page-on-demand thumbnail loading rather than keeping 100 high-res pages uncompressed in memory.
