# AirTeach AI — Skills Inventory & Applied Technical Competencies

This document logs all agent skills and architectural principles applied during the autonomous development and verification of AirTeach AI.

---

## 1. Primary Specialized Skills Applied

| Skill Name | Purpose & Practical Application in AirTeach AI |
|---|---|
| **`airteach-local-face-auth`** | Implemented OpenCV DNN YuNet (320x320) detection and SFace (112x112) 128-dimensional embedding extraction, Fernet symmetric encryption, PBKDF2 PIN hashing with salt, and SQLite profile storage. |
| **`airteach-mediapipe-gestures`** | Configured MediaPipe HandLandmarker, 21 3D landmarks extraction, $L_{\text{hand}} = \|P_0 - P_9\|$ normalization, and the 6-gesture debounced state machine. |
| **`airteach-vector-ink`** | Architected the 3-stage stroke beautification engine (1€ Filter with velocity-dependent cutoff, RDP point decimation, and $C^1$ continuous Cubic Bézier Spline fitting). |
| **`airteach-shape-snapping`** | Developed the multi-primitive geometric shape detector (Line, Arrow, Circle/Ellipse, Rectangle, Triangle) utilizing ConvexHull and PCA linearity. Fixed 3D timestamped point broadcasting bug. |
| **`airteach-document-import`** | Configured multi-format document conversion pipeline: pypdfium2 high-resolution rendering, Windows Office COM API, LibreOffice Headless, and `.airteach` ZIP bundling. |
| **`airteach-math-formula-recognition`** | Developed the zero-dependency `GeometricMathParser` fallback and `LaTeXRenderer` using `matplotlib.mathtext` for offline vector formula rasterization. |
| **`airteach-pyside6-ui`** | Designed the layered `QGraphicsScene` / `QGraphicsView` architecture, custom items, smooth anti-aliased painting, and modern Academic Theme stylesheet. |
| **`airteach-qt-performance`** | Audited zero-UI-thread-blocking concurrency, background `QThread` workers for camera and hand tracking, bounded frame buffers, and explicit DirectShow camera handover timing. |
| **`test-driven-development`** | Expanded test coverage to 250 unit and integration tests with 100% pass rate. |
| **`systematic-debugging`** | Traced and resolved the Priority P0 teacher enrollment/login loop bug, DirectShow resource contention, and 3D coordinate broadcasting issue in shape snapping. |
| **`accidental-data-loss-prevention`** | Ensured isolated per-teacher workspaces (`data/workspaces/<uuid>/`), preventing cross-account contamination or data overwrites. |

---

## 2. Policy on Cloud / Non-Essential Skills

As mandated by Section II of the project prompt, Cloud services and big-data skills (such as BigQuery, GCP Composer, Spark, Dataflow, Chrome Extensions) were intentionally excluded to preserve the strictly **Local-First, Air-Gapped Windows Desktop** design requirement of AirTeach AI.
