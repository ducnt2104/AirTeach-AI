#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent Headless MediaPipe HandLandmarker + FaceDetector Co-Processor for AirTeach AI.
Implements MODULE 3 (Backlight-Resilient, High-Sensitivity Hand Tracking):
1. Automatic Backlight & Shadow Recovery Pre-Processing:
   - Applies Gamma Correction (gamma = 0.6 LUT) + LAB L-channel CLAHE (clipLimit=2.5, tileGridSize=(8, 8))
     before feeding frames to MediaPipe so hands silhouetted against a bright window are clearly visible.
2. High-Sensitivity MediaPipe Configuration:
   - min_detection_confidence = 0.55, min_tracking_confidence = 0.55
3. Lightweight Anti-Face & Minimum Area Check:
   - Hand bounding_box_area > 0.015 (1.5% of frame area)
   - Palm center (Landmark 9) does not coincide with the detected face center.
"""

import os
import sys
import math
import json
import struct
import urllib.request
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GLOG_minloglevel"] = "3"

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"

GAMMA_LUTS = {
    g: np.array([((i / 255.0) ** g) * 255.0 for i in range(256)], dtype=np.uint8)
    for g in (0.62, 0.78, 1.18, 1.32)
}
CLAHE_NORMAL = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
CLAHE_BOOST = cv2.createCLAHE(clipLimit=2.8, tileGridSize=(8, 8))
CLAHE_SOFT = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))


def ensure_model(path: Path, url: str) -> bool:
    if path.exists() and path.stat().st_size > 50_000:
        return True
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(path))
        return path.exists() and path.stat().st_size > 50_000
    except Exception:
        return False


def enhance_adaptive_lighting_rgb(rgb_frame: np.ndarray) -> np.ndarray:
    """
    Adaptive Luminance Diagnostics & Local Lighting Balance (Dark / Glare / Normal):
    1. Convert RGB -> LAB and inspect mean L, underexposed %, and overexposed % ratios.
    2. Low Light: edge-preserving Bilateral denoise + adaptive gamma lift (<1.0) + LAB L-CLAHE.
    3. Glare / Harsh Backlight: highlight soft-knee roll-off + gamma compression (>1.0) + gentle CLAHE.
    4. Normal: clean LAB L-channel CLAHE only (preserves natural A/B skin chroma).
    """
    try:
        lab = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        mean_l = float(np.mean(l_chan))
        under_ratio = float(np.mean(l_chan < 52))
        over_ratio = float(np.mean(l_chan > 224))

        if mean_l < 92.0 or under_ratio > 0.35:
            # Dark environment: suppress ISO grain with edge-preserving filter before lift
            l_clean = cv2.bilateralFilter(l_chan, d=5, sigmaColor=24, sigmaSpace=24)
            lut = GAMMA_LUTS[0.62] if mean_l < 65.0 else GAMMA_LUTS[0.78]
            l_tone = cv2.LUT(l_clean, lut)
            l_eq = CLAHE_BOOST.apply(l_tone)
        elif over_ratio > 0.14 or mean_l > 175.0:
            # Glare / overexposure suppression: compress highlights via knee + gamma > 1.0
            l_f = l_chan.astype(np.float32)
            over_mask = l_f > 195.0
            if np.any(over_mask):
                l_f[over_mask] = 195.0 + (l_f[over_mask] - 195.0) * 0.42
            l_knee = np.clip(l_f, 0, 255).astype(np.uint8)
            lut = GAMMA_LUTS[1.32] if mean_l > 195.0 else GAMMA_LUTS[1.18]
            l_tone = cv2.LUT(l_knee, lut)
            l_eq = CLAHE_SOFT.apply(l_tone)
        else:
            l_eq = CLAHE_NORMAL.apply(l_chan)

        # Controlled Unsharp Mask with coring threshold (only crispen genuine edges)
        blur = cv2.GaussianBlur(l_eq, (0, 0), 1.0)
        diff = l_eq.astype(np.int16) - blur.astype(np.int16)
        diff[np.abs(diff) < 3] = 0
        l_sharp = np.clip(l_eq.astype(np.int16) + (diff * 2) // 5, 0, 255).astype(np.uint8)

        lab_eq = cv2.merge((l_sharp, a_chan, b_chan))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    except Exception:
        return rgb_frame


def validate_real_hand(lm_list, score: float, face_boxes) -> bool:
    """
    MODULE 3.2: Backlight-Resilient Hand Validator:
    - Confidence >= 0.50 (lowered from 0.80 so backlit/tilted hands are never dropped)
    - Hand bounding_box_area > 0.015 (1.5% of frame area)
    - Palm center (Landmark 9) does not coincide with the detected Face Center
    """
    if score < 0.50 or len(lm_list) < 21:
        return False

    xs = [p[0] for p in lm_list]
    ys = [p[1] for p in lm_list]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    bounding_box_area = bbox_w * bbox_h
    if bounding_box_area <= 0.015:
        return False

    # Anti-face check: Palm center (Landmark 9) must not coincide with the face center
    px, py = lm_list[9][0], lm_list[9][1]
    for fx, fy, fw, fh in face_boxes:
        face_cx = fx + fw * 0.5
        face_cy = fy + fh * 0.5
        radius = max(fw, fh) * 0.42
        if math.hypot(px - face_cx, py - face_cy) < radius:
            return False

    return True


def main():
    models_dir = Path("models")
    hand_model_path = models_dir / "hand_landmarker.task"
    face_model_path = models_dir / "blaze_face_short_range.tflite"

    # Prefer mp.solutions.hands.Hands if available, else MediaPipe Tasks HandLandmarker
    solutions_hands = None
    hand_detector = None
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        try:
            solutions_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55,
            )
        except Exception:
            solutions_hands = None

    if solutions_hands is None:
        if not ensure_model(hand_model_path, HAND_MODEL_URL):
            sys.exit(1)
        hand_opts = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(hand_model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

    has_face_model = ensure_model(face_model_path, FACE_MODEL_URL)
    face_detector = None
    if has_face_model:
        try:
            face_opts = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(face_model_path)),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=0.55,
            )
            face_detector = vision.FaceDetector.create_from_options(face_opts)
        except Exception:
            face_detector = None

    stdin_buf = sys.stdin.buffer
    stdout_buf = sys.stdout.buffer

    # Signal readiness
    stdout_buf.write(b"READY\n")
    stdout_buf.flush()

    while True:
        hdr = stdin_buf.read(8)
        if not hdr or len(hdr) < 8:
            break
        magic, w, h = struct.unpack("<4sHH", hdr)
        if magic != b"MPHF" or w == 0 or h == 0:
            break

        nbytes = int(w) * int(h) * 3
        raw = stdin_buf.read(nbytes)
        if not raw or len(raw) < nbytes:
            break

        rgb_orig = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        rgb = enhance_adaptive_lighting_rgb(rgb_orig)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_boxes = []
        face_eyes = None
        if face_detector is not None:
            try:
                f_res = face_detector.detect(mp_img)
                if f_res and f_res.detections:
                    for det in f_res.detections:
                        bb = det.bounding_box
                        fx = max(0.0, float(bb.origin_x) / float(w))
                        fy = max(0.0, float(bb.origin_y) / float(h))
                        fw = min(1.0 - fx, float(bb.width) / float(w))
                        fh = min(1.0 - fy, float(bb.height) / float(h))
                        face_boxes.append([round(fx, 4), round(fy, 4), round(fw, 4), round(fh, 4)])
                        if face_eyes is None and getattr(det, "keypoints", None) and len(det.keypoints) >= 2:
                            k0, k1 = det.keypoints[0], det.keypoints[1]
                            p0 = (max(0.0, min(1.0, float(k0.x))), max(0.0, min(1.0, float(k0.y))))
                            p1 = (max(0.0, min(1.0, float(k1.x))), max(0.0, min(1.0, float(k1.y))))
                            left_p, right_p = (p0, p1) if p0[0] <= p1[0] else (p1, p0)
                            face_eyes = [
                                round(left_p[0], 4),
                                round(left_p[1], 4),
                                round(right_p[0], 4),
                                round(right_p[1], 4),
                            ]
            except Exception:
                pass

        left_out = None
        right_out = None

        try:
            if solutions_hands is not None:
                res = solutions_hands.process(rgb)
                if res and res.multi_hand_landmarks and res.multi_handedness:
                    for lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                        if not handed.classification:
                            continue
                        top_cat = handed.classification[0]
                        score = float(top_cat.score)
                        cat_name = (top_cat.label or "").lower()
                        lm_coords = [
                            [round(float(p.x), 4), round(float(p.y), 4), round(float(p.z), 4)]
                            for p in lms.landmark
                        ]
                        if not validate_real_hand(lm_coords, score, face_boxes):
                            continue
                        hand_obj = {"score": round(score, 4), "lm": lm_coords}
                        if cat_name == "left" and right_out is None:
                            # If only one hand is raised, treat it as the active drawing hand (right_out)
                            right_out = hand_obj
                        elif cat_name == "left":
                            left_out = hand_obj
                        else:
                            right_out = hand_obj
            elif hand_detector is not None:
                h_res = hand_detector.detect(mp_img)
                if h_res and h_res.hand_landmarks and h_res.handedness:
                    valid_hands = []
                    for lms, handed_list in zip(h_res.hand_landmarks, h_res.handedness):
                        if not handed_list:
                            continue
                        top_cat = handed_list[0]
                        score = float(top_cat.score)
                        cat_name = (top_cat.category_name or "").lower()
                        lm_coords = [
                            [round(float(p.x), 4), round(float(p.y), 4), round(float(p.z), 4)]
                            for p in lms
                        ]
                        if not validate_real_hand(lm_coords, score, face_boxes):
                            continue
                        valid_hands.append((cat_name, {"score": round(score, 4), "lm": lm_coords}))

                    if len(valid_hands) == 1:
                        # Single raised hand always controls the Smartboard cursor/pen (right_out)
                        right_out = valid_hands[0][1]
                    else:
                        for cat_name, hand_obj in valid_hands:
                            if cat_name == "left" and left_out is None:
                                left_out = hand_obj
                            else:
                                right_out = hand_obj
        except Exception:
            pass

        resp = {
            "left": left_out,
            "right": right_out,
            "face": face_boxes[0] if face_boxes else None,
            "face_eyes": face_eyes,
        }
        line = (json.dumps(resp, separators=(",", ":")) + "\n").encode("utf-8")
        stdout_buf.write(line)
        stdout_buf.flush()


if __name__ == "__main__":
    main()
