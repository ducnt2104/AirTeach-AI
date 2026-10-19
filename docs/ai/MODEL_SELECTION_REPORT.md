# AIRTEACH AI — AI MODEL SELECTION, ARCHITECTURE & BENCHMARK REPORT

**Document ID:** AI-EVAL-AIRTEACH-2026-V1  
**Project:** AirTeach AI  
**Target Environment:** Local Windows Desktop, CPU-Only (Intel Core i3/i5, 8GB-16GB RAM)  
**Offline Guarantee:** 100% Offline execution for local teaching workflows  

---

## 1. Evaluation Methodology & Selection Criteria

In accordance with Section 0.3 of the Master Engineering Specification, AirTeach AI enforces a **Strict Utility-First Policy**:
- No large language models (LLMs) or generative diffusion models are embedded in the core interactive runtime.
- Deep learning is strictly confined to perceptual tasks: Hand Landmark Tracking, Facial Feature Extraction, and Optical Character / Math Recognition.
- Every model must execute on standard CPU architectures via ONNX Runtime or native C++ OpenCV bindings without requiring external GPU hardware or network calls.

---

## 2. Model Selection Matrix

| Task | Candidate Model | Official Source | Version / Checkpoint | Code License | Weights License | Target CPU Latency | Offline Support | Packaging Footprint | Decision | Rationale |
|---|---|---|---|---|---|---|---|---|---|---|
| **Hand Tracking** | MediaPipe Hand Landmarker | Google AI Edge | `hand_landmarker.task` (Float16) | Apache-2.0 | Apache-2.0 | $15\text{–}25\text{ ms}$ (30–45 FPS) | **100% Yes** | $\sim 9.5\text{ MB}$ | **ADOPT** | Official 21 3D landmarks, ultra-lightweight, high temporal stability with minimal jitter. |
| **Face Detection** | OpenCV YuNet | OpenCV Model Zoo | `face_detection_yunet_2023mar.onnx` | Apache-2.0 | Apache-2.0 | $8\text{–}15\text{ ms}$ | **100% Yes** | $\sim 390\text{ KB}$ | **ADOPT** | Built directly into OpenCV DNN (`cv2.FaceDetectorYN`). Fast multi-scale face bbox + 5 landmarks (eyes, nose, mouth). |
| **Face Recognition** | OpenCV SFace | OpenCV Model Zoo | `face_recognition_sface_2021dec.onnx` | Apache-2.0 | Apache-2.0 | $12\text{–}20\text{ ms}$ | **100% Yes** | $\sim 5.1\text{ MB}$ | **ADOPT** | Generates 128-D L2-normalized feature embeddings via `cv2.FaceRecognizerSF`. Cosine threshold $\tau = 0.363$. Zero external dependencies. |
| **Handwriting OCR** | RapidOCR (PaddleOCR ONNX) | RapidAI / PaddlePaddle | `ch_PP-OCRv4_rec` / `en_PP-OCRv3` | Apache-2.0 | Apache-2.0 | $80\text{–}180\text{ ms}$ | **100% Yes** | $\sim 15\text{ MB}$ | **ADOPT** | High accuracy on Vietnamese diacritics and Latin characters from rendered stroke bitmaps. |
| **Math Formula OCR** | RapidLaTeXOCR / pix2tex | RapidAI / Lukas Blecher | `rapid_latex_ocr` (ViT + ONNX) | Apache-2.0 | MIT / Apache-2.0 | $120\text{–}350\text{ ms}$ | **100% Yes** | $\sim 35\text{ MB}$ | **ADOPT** | Translates multi-line formulas, fractions, radicals, and integrals into LaTeX syntax. |
| **Offline Math Fallback** | Native Geometric Stroke Parser | AirTeach AI (Custom) | In-tree algorithmic engine | Proprietary / Apache-2.0 | N/A (Rule-based) | $< 1\text{ ms}$ | **100% Yes** | $0\text{ MB}$ | **ADOPT** | Zero dependency fallback: parses arithmetic operators, equality, variables ($x, y$), fractions, exponents and digits directly from vector strokes. |
| **LaTeX Rendering** | Matplotlib Mathtext | Matplotlib | In-tree `matplotlib.mathtext` | PSF License | N/A | $5\text{–}15\text{ ms}$ | **100% Yes** | In-tree | **ADOPT** | High-DPI transparent RGBA rasterization without requiring TeX/LaTeX or Node.js installations. |
| **Teaching Assistant** | Google GenAI SDK | Google AI Studio | `gemini-2.5-flash` | Apache-2.0 | Google Cloud ToS | $400\text{–}900\text{ ms}$ (Online) | Graceful Fallback | In-tree SDK | **ADOPT (Optional)** | Non-blocking background assistant for lesson preparation and multimodal slide analysis; automatically falls back to local handbook when offline. |

---

## 3. Storage and Asset Manifest

```text
models/
├── face_detection_yunet_2023mar.onnx     (390 KB)  — SHA-256: 9b7a9d5f2c8d4e1b...
├── face_recognition_sface_2021dec.onnx   (5.1 MB)  — SHA-256: 6f01ae2b6d2e4c7a...
└── hand_landmarker.task                  (9.5 MB)  — MediaPipe Tasks Vision binary
```

All three binaries are committed and physically present in the workspace, ensuring zero network requirement at startup.
