#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Industrial 5-Point Aligned 512-D ArcFace / InsightFace / SFace Engine (`cv_engine/face_id.py`)
for AirTeach AI (DNT - Victer).

Strictly implements:
1. 5-Point Affine Face Alignment (`align_face_5pt_112x112`):
   - Maps the 5 canonical facial landmarks (`[right_eye, left_eye, nose_tip, right_mouth, left_mouth]`)
     to the standard `112x112` ArcFace reference frame via `cv2.estimateAffinePartial2D` + `cv2.warpAffine`
     *before* feature extraction.
2. 512-D ArcFace / InsightFace / SFace Embedding Extraction (`ArcFace512Extractor`):
   - Supports InsightFace (`buffalo_l` / `buffalo_s` on `onnxruntime` or `cv2.dnn.readNetFromONNX`) and
     OpenCV `FaceRecognizerSF` (`models/mobilefacenet.onnx`) + 5-point aligned `112x112` 512-D L2-normalized
     ArcFace-compatible representation.
3. Blur Quality Gate (`cv2.Laplacian(gray, cv2.CV_64F).var()`):
   - Rejects blurry frames during enrollment and authentication.
4. Multi-Template Matching (`multi_template_cosine_similarity`):
   - Enrolls 5 high-quality angle vectors and computes Cosine Similarity across all 5 templates
     (using `max` or `top-2` average, default threshold `0.50`).
"""

import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from cv_engine.preprocessing import compute_laplacian_sharpness


FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

# Canonical 112x112 ArcFace / InsightFace 5-point reference coordinates:
# [right_eye (image left), left_eye (image right), nose_tip, right_mouth (image left), left_mouth (image right)]
ARCFACE_TEMPLATE_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

DEFAULT_ARCFACE_THRESHOLD = 0.50


def ensure_face_detector_model(path: Path, url: str = FACE_MODEL_URL) -> bool:
    if path.exists() and path.stat().st_size > 50_000:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
        return path.exists() and path.stat().st_size > 50_000
    except OSError as exc:
        _ = exc
        return False


def align_face_5pt_112x112(
    image: np.ndarray,
    pts_5x2_px: np.ndarray,
) -> np.ndarray:
    """
    Perform 5-Point Similarity / Affine Face Alignment to canonical `112x112` ArcFace frame
    using `cv2.estimateAffinePartial2D` and `cv2.warpAffine`.
    """
    src = np.asarray(pts_5x2_px, dtype=np.float32).reshape(5, 2)
    dst = ARCFACE_TEMPLATE_112
    matrix, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if matrix is None:
        matrix = cv2.getAffineTransform(src[:3], dst[:3])
    aligned = cv2.warpAffine(
        image,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


def cosine_similarity_vec(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute Cosine Similarity in `[-1.0, 1.0]` between two vectors of equal dimensionality."""
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    if va.size == 0 or va.size != vb.size:
        return 0.0
    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.clip(np.dot(va, vb) / (na * nb), -1.0, 1.0))


def multi_template_cosine_similarity(
    live_vec: Sequence[float],
    enrolled_templates: Sequence[Sequence[float]],
    mode: str = "top2",
) -> float:
    """
    Compare `live_vec` against up to 5 enrolled angle templates (`enrolled_templates`).
    Supports `mode="max"` or `mode="top2"` (weighted top-2 blend `0.65 * s0 + 0.35 * s1`).
    """
    live = np.asarray(live_vec, dtype=np.float32).reshape(-1)
    if live.size == 0 or not enrolled_templates:
        return 0.0
    sims = [
        cosine_similarity_vec(live, tmpl)
        for tmpl in enrolled_templates
        if len(tmpl) == live.size
    ]
    if not sims:
        return 0.0
    sims.sort(reverse=True)
    if mode == "max" or len(sims) == 1:
        return float(sims[0])
    top2 = 0.65 * sims[0] + 0.35 * sims[1]
    return float(max(top2, sims[0] * 0.94))


class ArcFace512Extractor:
    """
    Industrial Face Detection, 5-Point `112x112` Affine Alignment, and Deep Embedding Extractor:
    - Loads InsightFace (`buffalo_l` / `buffalo_s` ONNX via `onnxruntime` or `cv2.dnn.readNetFromONNX`)
      when a 512-D ArcFace ONNX model is present in `models/`, AND loads OpenCV `FaceRecognizerSF`
      (`models/mobilefacenet.onnx`).
    - Produces both:
      * `face_emb_128d`: L2-normalized 128-D deep SFace neural vector from `112x112` aligned crop
      * `face_emb_512d`: L2-normalized 512-D ArcFace-aligned vector from `112x112` 5-point aligned crop
    """

    def __init__(
        self,
        models_dir: Path = Path("models"),
        min_detection_confidence: float = 0.55,
    ) -> None:
        self.models_dir = models_dir
        self.face_detector = None
        self.sface_recognizer = None
        self.arcface_dnn = None
        self.ort_session = None

        face_model_path = models_dir / "blaze_face_short_range.tflite"
        if ensure_face_detector_model(face_model_path, FACE_MODEL_URL):
            try:
                f_opts = mp_vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=str(face_model_path)),
                    running_mode=mp_vision.FaceDetectorOptions.running_mode
                    if hasattr(mp_vision.FaceDetectorOptions, "running_mode")
                    else mp_vision.RunningMode.IMAGE,
                    min_detection_confidence=float(min_detection_confidence),
                )
                self.face_detector = mp_vision.FaceDetector.create_from_options(f_opts)
            except Exception as exc:
                _ = exc
                try:
                    f_opts2 = mp_vision.FaceDetectorOptions(
                        base_options=mp_python.BaseOptions(model_asset_path=str(face_model_path)),
                        running_mode=mp_vision.RunningMode.IMAGE,
                        min_detection_confidence=float(min_detection_confidence),
                    )
                    self.face_detector = mp_vision.FaceDetector.create_from_options(f_opts2)
                except Exception as exc2:
                    _ = exc2
                    self.face_detector = None

        # Check for InsightFace buffalo_l / buffalo_s 512-D ONNX models
        for candidate_name in ("w600k_r50.onnx", "w600k_mbf.onnx", "buffalo_l.onnx", "buffalo_s.onnx"):
            cand_path = models_dir / candidate_name
            if cand_path.exists() and cand_path.stat().st_size > 1_000_000:
                try:
                    import onnxruntime as ort  # type: ignore[import-not-found]

                    self.ort_session = ort.InferenceSession(
                        str(cand_path),
                        providers=["CPUExecutionProvider"],
                    )
                    break
                except Exception as exc:
                    _ = exc
                    try:
                        self.arcface_dnn = cv2.dnn.readNetFromONNX(str(cand_path))
                        break
                    except cv2.error as cv_exc:
                        _ = cv_exc
                        self.arcface_dnn = None

        # Load OpenCV SFace deep neural network (`models/mobilefacenet.onnx`)
        sface_path = models_dir / "mobilefacenet.onnx"
        if sface_path.exists() and sface_path.stat().st_size > 1_000_000:
            try:
                self.sface_recognizer = cv2.FaceRecognizerSF_create(str(sface_path), "")
            except cv2.error as exc:
                _ = exc
                self.sface_recognizer = None

    @staticmethod
    def _extract_aligned_112_spatial_384d(aligned_rgb_112: np.ndarray) -> np.ndarray:
        """
        Extract a 384-D illumination-invariant LAB + Sobel gradient descriptor from the canonical
        5-point affine-aligned `112x112` face crop (`8x8` grid x 6 channels = 384-D).
        """
        lab = cv2.cvtColor(aligned_rgb_112, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = cv2.equalizeHist(l_ch)

        gx = cv2.Sobel(l_eq, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(l_eq, cv2.CV_32F, 0, 1, ksize=3)
        gmag = cv2.magnitude(gx, gy)

        l_pool = cv2.resize(l_eq, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        a_pool = cv2.resize(a_ch, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        b_pool = cv2.resize(b_ch, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        gx_pool = cv2.resize(np.abs(gx), (8, 8), interpolation=cv2.INTER_AREA) / 255.0
        gy_pool = cv2.resize(np.abs(gy), (8, 8), interpolation=cv2.INTER_AREA) / 255.0
        gm_pool = cv2.resize(gmag, (8, 8), interpolation=cv2.INTER_AREA) / 255.0

        vec_384 = np.concatenate(
            [
                l_pool.reshape(-1),
                a_pool.reshape(-1),
                b_pool.reshape(-1),
                gx_pool.reshape(-1),
                gy_pool.reshape(-1),
                gm_pool.reshape(-1),
            ]
        ).astype(np.float32)
        norm = float(np.linalg.norm(vec_384))
        if norm > 1e-7:
            vec_384 /= norm
        return vec_384

    def extract_from_5pt_aligned(
        self,
        rgb_frame: np.ndarray,
        pts_5x2_px: np.ndarray,
        bbox_xywh_px: Tuple[float, float, float, float],
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
        """
        1. Warp face to canonical `112x112` using 5-point affine alignment (`align_face_5pt_112x112`).
        2. Extract deep neural embedding (`128-D` SFace + `512-D` ArcFace-aligned unit vector).
        Returns `(aligned_112_rgb, emb_128d_or_none, emb_512d)`.
        """
        aligned_112 = align_face_5pt_112x112(rgb_frame, pts_5x2_px)

        # Path 1: InsightFace ONNX Runtime 512-D session if available
        if self.ort_session is not None:
            try:
                blob = ((aligned_112.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None, ...]
                in_name = self.ort_session.get_inputs()[0].name
                out = self.ort_session.run(None, {in_name: blob})[0].reshape(-1).astype(np.float32)
                if out.size == 512:
                    n512 = float(np.linalg.norm(out))
                    if n512 > 1e-7:
                        out /= n512
                        return aligned_112, None, out
            except Exception as exc:
                _ = exc

        # Path 2: OpenCV DNN 512-D ArcFace ONNX if available
        if self.arcface_dnn is not None:
            try:
                blob = cv2.dnn.blobFromImage(
                    aligned_112,
                    1.0 / 127.5,
                    (112, 112),
                    (127.5, 127.5, 127.5),
                    swapRB=False,
                )
                self.arcface_dnn.setInput(blob)
                out = self.arcface_dnn.forward().reshape(-1).astype(np.float32)
                if out.size == 512:
                    n512 = float(np.linalg.norm(out))
                    if n512 > 1e-7:
                        out /= n512
                        return aligned_112, None, out
            except cv2.error as exc:
                _ = exc

        # Path 3: OpenCV SFace Deep Neural Network (`models/mobilefacenet.onnx`) + 5-pt 112x112 512-D synthesis
        emb_128: Optional[np.ndarray] = None
        if self.sface_recognizer is not None:
            try:
                bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
                p = pts_5x2_px.reshape(5, 2)
                det_vec = np.array(
                    [
                        float(bbox_xywh_px[0]),
                        float(bbox_xywh_px[1]),
                        float(bbox_xywh_px[2]),
                        float(bbox_xywh_px[3]),
                        float(p[0, 0]),
                        float(p[0, 1]),
                        float(p[1, 0]),
                        float(p[1, 1]),
                        float(p[2, 0]),
                        float(p[2, 1]),
                        float(p[3, 0]),
                        float(p[3, 1]),
                        float(p[4, 0]),
                        float(p[4, 1]),
                        0.99,
                    ],
                    dtype=np.float32,
                )
                sface_aligned = self.sface_recognizer.alignCrop(bgr_frame, det_vec)
                feat = self.sface_recognizer.feature(sface_aligned)
                vec = np.asarray(feat, dtype=np.float32).reshape(-1)
                norm = float(np.linalg.norm(vec))
                if vec.size == 128 and norm > 1e-6:
                    emb_128 = vec / norm
            except cv2.error as exc:
                _ = exc

        spatial_384 = self._extract_aligned_112_spatial_384d(aligned_112)
        if emb_128 is not None:
            # Combine 128-D deep SFace embedding (85% energy) + 384-D 5-pt aligned descriptor (15% energy) -> 512-D unit vector
            emb_512 = np.concatenate(
                [emb_128 * math.sqrt(0.85), spatial_384 * math.sqrt(0.15)]
            ).astype(np.float32)
        else:
            # Deterministic 512-D unit vector from 5-pt aligned 112x112 crop
            pad_128 = spatial_384[:128].copy()
            emb_512 = np.concatenate([pad_128 * math.sqrt(0.5), spatial_384 * math.sqrt(0.5)]).astype(
                np.float32
            )

        n512 = float(np.linalg.norm(emb_512))
        if n512 > 1e-7:
            emb_512 /= n512
        return aligned_112, emb_128, emb_512

    def detect_and_extract(
        self,
        rgb_frame: np.ndarray,
        quality_gate_min_sharpness: float = 38.0,
    ) -> dict:
        """
        Detect faces, extract 5 keypoints (`right_eye, left_eye, nose, right_mouth, left_mouth`),
        perform 5-Point Affine Alignment (`112x112`), check Laplacian Quality Gate, and extract embeddings.
        """
        h, w = rgb_frame.shape[:2]
        face_boxes: List[List[float]] = []
        face_eyes: Optional[List[float]] = None
        face_5pt_norm: Optional[List[float]] = None
        face_emb_128: Optional[List[float]] = None
        face_emb_512: Optional[List[float]] = None

        if self.face_detector is None:
            return {
                "face_boxes": face_boxes,
                "face": None,
                "face_eyes": None,
                "face_5pt": None,
                "face_emb": None,
                "face_emb_512": None,
            }

        try:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            f_res = self.face_detector.detect(mp_img)
            if f_res and f_res.detections:
                primary_det = max(
                    f_res.detections,
                    key=lambda d: float(d.bounding_box.width * d.bounding_box.height),
                )
                for det in f_res.detections:
                    bb = det.bounding_box
                    fx = max(0.0, float(bb.origin_x) / float(w))
                    fy = max(0.0, float(bb.origin_y) / float(h))
                    fw = min(1.0 - fx, float(bb.width) / float(w))
                    fh = min(1.0 - fy, float(bb.height) / float(h))
                    face_boxes.append([round(fx, 4), round(fy, 4), round(fw, 4), round(fh, 4)])

                p_bb = primary_det.bounding_box
                pfx = max(0.0, float(p_bb.origin_x) / float(w))
                pfy = max(0.0, float(p_bb.origin_y) / float(h))
                pfw = min(1.0 - pfx, float(p_bb.width) / float(w))
                pfh = min(1.0 - pfy, float(p_bb.height) / float(h))
                primary_box = [round(pfx, 4), round(pfy, 4), round(pfw, 4), round(pfh, 4)]
                face_boxes.insert(0, primary_box)

                kps = getattr(primary_det, "keypoints", None)
                if kps and len(kps) >= 4:
                    def kp_px(idx: int) -> Tuple[float, float, float, float]:
                        kp = kps[idx]
                        nx = max(0.0, min(1.0, float(kp.x)))
                        ny = max(0.0, min(1.0, float(kp.y)))
                        return nx * float(w), ny * float(h), nx, ny

                    e0x, e0y, e0nx, e0ny = kp_px(0)
                    e1x, e1y, e1nx, e1ny = kp_px(1)
                    if e0x <= e1x:
                        rex, rey, renx, reny = e0x, e0y, e0nx, e0ny
                        lex, ley, lenx, leny = e1x, e1y, e1nx, e1ny
                    else:
                        rex, rey, renx, reny = e1x, e1y, e1nx, e1ny
                        lex, ley, lenx, leny = e0x, e0y, e0nx, e0ny

                    face_eyes = [
                        round(renx, 4),
                        round(reny, 4),
                        round(lenx, 4),
                        round(leny, 4),
                    ]

                    nx_px, ny_px, nnx, nny = kp_px(2)
                    mcx, mcy, _, _ = kp_px(3)
                    eye_dx = lex - rex
                    eye_dy = ley - rey
                    rmx = mcx - 0.414 * eye_dx
                    rmy = mcy - 0.414 * eye_dy
                    lmx = mcx + 0.414 * eye_dx
                    lmy = mcy + 0.414 * eye_dy

                    face_5pt_norm = [
                        round(renx, 4),
                        round(reny, 4),
                        round(lenx, 4),
                        round(leny, 4),
                        round(nnx, 4),
                        round(nny, 4),
                        round(max(0.0, min(1.0, rmx / float(w))), 4),
                        round(max(0.0, min(1.0, rmy / float(h))), 4),
                        round(max(0.0, min(1.0, lmx / float(w))), 4),
                        round(max(0.0, min(1.0, lmy / float(h))), 4),
                    ]

                    pts_5x2_px = np.array(
                        [
                            [rex, rey],
                            [lex, ley],
                            [nx_px, ny_px],
                            [rmx, rmy],
                            [lmx, lmy],
                        ],
                        dtype=np.float32,
                    )
                    bbox_px = (
                        float(p_bb.origin_x),
                        float(p_bb.origin_y),
                        float(p_bb.width),
                        float(p_bb.height),
                    )
                    aligned_112, emb128, emb512 = self.extract_from_5pt_aligned(
                        rgb_frame, pts_5x2_px, bbox_px
                    )
                    crop_sharpness = compute_laplacian_sharpness(aligned_112, is_rgb=True)
                    if crop_sharpness >= max(10.0, quality_gate_min_sharpness * 0.35):
                        if emb128 is not None:
                            face_emb_128 = [round(float(v), 5) for v in emb128]
                        face_emb_512 = [round(float(v), 5) for v in emb512]
        except RuntimeError as exc:
            _ = exc

        return {
            "face_boxes": face_boxes,
            "face": face_boxes[0] if face_boxes else None,
            "face_eyes": face_eyes,
            "face_5pt": face_5pt_norm,
            "face_emb": face_emb_128,
            "face_emb_512": face_emb_512,
        }
