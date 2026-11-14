"""
AirTeach AI — Maximum Recognition Computer Vision Sidecar Engine (`cv_engine/`)
Organization: DNT - Victer

Modules:
- `preprocessing`: Adaptive LAB Luminance Diagnostics (`mean_luma`, `p5`, `p95`),
  Dynamic Adaptive Gamma Correction (`mean_luma < 70`), Highlight Roll-off (`p95 > 235`),
  LAB L-channel CLAHE (`tileGridSize=(8, 8)`, dynamic `clipLimit` in `[1.5, 4.0]`),
  Low-light Edge-Preserving Bilateral Filter (`d=5, sigmaColor=40, sigmaSpace=40`) BEFORE
  controlled Unsharp Masking (`1.45 * img - 0.45 * gaussian_blur`), and Laplacian Variance
  Blur Quality Gate (`cv2.Laplacian(gray, cv2.CV_64F).var()`).
- `hand_gesture`: MediaPipe Hand Landmarker (`model_complexity=1` Full, `min_detection_confidence=0.6`,
  `min_tracking_confidence=0.7`), ROI Dynamic Zoom Crop (`< 8%` bbox area -> `+35%` padding ->
  `384x384` `INTER_CUBIC`), 3D Vector Joint-Angle Gesture Classification (`arccos(dot(u,v)/(|u||v|))`,
  thumb palm-plane abduction angle, `Pinch Ratio = dist(4, 8) / dist(0, 9)`), One-Euro Filter,
  and Gesture FSM (`IDLE -> CANDIDATE -> ACTIVE -> GRACE_PERIOD -> IDLE`) with `FrameSkipLock`.
- `face_id`: 5-Point Canonical `112x112` Affine Face Alignment (`cv2.estimateAffinePartial2D` +
  `cv2.warpAffine`), 512-D ArcFace / InsightFace / SFace Deep Embedding Extractor, and
  5-Angle Multi-Template Cosine Similarity Matching (default threshold `0.50`).
"""

from cv_engine.preprocessing import (
    ExposureDiagnostics,
    compute_laplacian_sharpness,
    preprocess_adaptive_frame,
)
from cv_engine.hand_gesture import (
    FrameSkipLock,
    GestureFsmEngine,
    GestureFsmState,
    HandGestureEngine,
    OneEuroFilter1D,
    OneEuroFilter2D,
    OneEuroFilter3D,
    classify_hand_3d_vectors,
    compute_palm_scale_3d,
    compute_pinch_ratio_3d,
    joint_angle_3d_deg,
    thumb_palm_plane_angle_deg,
)
from cv_engine.face_id import (
    ARCFACE_TEMPLATE_112,
    ArcFace512Extractor,
    align_face_5pt_112x112,
    cosine_similarity_vec,
    multi_template_cosine_similarity,
)
from cv_engine.drawing_fsm import (
    DrawingFsmUpdate,
    DrawingHudMode,
    StrokeRingBuffer,
    ZeroStrayStrokeDrawingFsm,
)
from cv_engine.shape_recognizer import (
    BeautifiedShapeResult,
    beautify_stroke,
    catmull_rom_spline,
    rdp_simplify,
    resample_arc_length,
)

__all__ = [
    "ExposureDiagnostics",
    "compute_laplacian_sharpness",
    "preprocess_adaptive_frame",
    "FrameSkipLock",
    "GestureFsmEngine",
    "GestureFsmState",
    "HandGestureEngine",
    "OneEuroFilter1D",
    "OneEuroFilter2D",
    "OneEuroFilter3D",
    "classify_hand_3d_vectors",
    "compute_palm_scale_3d",
    "compute_pinch_ratio_3d",
    "joint_angle_3d_deg",
    "thumb_palm_plane_angle_deg",
    "ARCFACE_TEMPLATE_112",
    "ArcFace512Extractor",
    "align_face_5pt_112x112",
    "cosine_similarity_vec",
    "multi_template_cosine_similarity",
    "DrawingFsmUpdate",
    "DrawingHudMode",
    "StrokeRingBuffer",
    "ZeroStrayStrokeDrawingFsm",
    "BeautifiedShapeResult",
    "beautify_stroke",
    "catmull_rom_spline",
    "rdp_simplify",
    "resample_arc_length",
]
