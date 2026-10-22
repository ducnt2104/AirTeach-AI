---
name: airteach-realtime-camera
description: >-
  Specialist skill for real-time webcam acquisition, DirectShow backend,
  CLAHE backlight compensation, non-blocking frame pipelines, and fail-safe camera lifecycles.
---

# AirTeach AI — Real-Time Camera Pipeline Skill

## 1. Capture Lifecycle & Windows DirectShow
- On Windows desktop, OpenCV capture should prioritize `cv2.CAP_DSHOW` for low latency and correct format negotiation, falling back to default backend:
  ```python
  cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
  if not cap.isOpened():
      cap = cv2.VideoCapture(camera_index)
  ```
- If `camera_index < 0`, immediately skip camera initialization to allow offline headless testing without blocking or warnings.

## 2. CLAHE Backlight Compensation
Classrooms frequently suffer from harsh window backlighting or overhead fluorescent glare.
Apply CLAHE exclusively on the L (Luminance) channel of the LAB color space:
```python
lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
l_eq = clahe.apply(l)
bgr_eq = cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)
```
Do not apply histogram equalization across all RGB channels directly, which alters color tint.

## 3. Worker Thread Isolation
- The camera loop runs in `QThread.run()` or a worker object moved to a `QThread`.
- A bounded queue (drop old frames when consumer is busy) prevents stale-frame latency.
