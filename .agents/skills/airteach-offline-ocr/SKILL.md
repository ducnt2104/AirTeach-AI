---
name: airteach-offline-ocr
description: >-
  Specialist skill for offline handwriting and printed text recognition in Vietnamese/English,
  RapidOCR ONNX models, normalized crop rendering, and candidate review workflows.
---

# AirTeach AI — Offline Text & Handwriting OCR Skill

## 1. High-Resolution Stroke Rasterization
- When strokes are selected for OCR, render them onto an offscreen canvas at $2\times$ DPI scale with white background and black ink.
- Add uniform padding ($24\text{ px}$) to prevent character boundary clipping.
- Anti-aliasing (`cv2.LINE_AA`) ensures smooth stroke edges matching typical pen inputs.

## 2. Model Pipeline (RapidOCR)
- Uses lightweight ONNX models (Text Detection, Direction Classifier, Text Recognition).
- Supports Vietnamese diacritics and mixed English/Vietnamese vocabulary.
- Never runs inference on the UI thread: invoke via `QThreadPool.globalInstance().start(_OcrRunnable)`.
- If ONNX Runtime is unavailable on Python 3.15+, fall back gracefully to geometric heuristics without crashing.
