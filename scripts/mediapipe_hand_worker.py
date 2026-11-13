#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Persistent Headless MediaPipe HandLandmarker + FaceDetector + OpenCV SFace Co-Processor
for AirTeach AI (DNT - Victer).

Upgraded CV Pipeline:
1. Adaptive Lighting Pre-Processing (`enhance_adaptive_lighting_rgb`):
   - LAB L-channel statistics drive low-light Bilateral Denoise + gamma lift,
     glare highlight soft-knee compression + gamma compression, or clean LAB CLAHE,
     followed by coring-thresholded Unsharp Masking.
2. Backlight-Resilient Hand Tracking + ROI Zoom Crop:
   - `HandLandmarker` (MediaPipe Tasks) runs with high sensitivity (`0.45`), filtered by
     face-exclusion & area sanity checks (`bbox_area > 0.008`).
   - Hands occupying `< 8.5%` of frame area undergo padded cubic ROI Zoom Crop refinement
     so distant hands stay pinpoint accurate.
3. 5-Point Aligned Deep Face Recognition (`models/mobilefacenet.onnx` via `cv2.FaceRecognizerSF`):
   - Maps BlazeFace 6 keypoints -> canonical YuNet 5-point pixel descriptor (`[left_eye,
     right_eye, nose, mouth_left, mouth_right]`), runs `alignCrop` (112x112) and extracts
     an L2-normalized 128-D deep SFace embedding (`face_emb`).
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
    Adaptive Luminance Diagnostics & Local Lighting Balance (Dark / Glare / Normal).
    Operates strictly on LAB L-channel so A/B chroma and skin tone stay natural.
    """
    try:
        lab = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        mean_l = float(np.mean(l_chan))
        under_ratio = float(np.mean(l_chan < 52))
        over_ratio = float(np.mean(l_chan > 224))

        if mean_l < 92.0 or under_ratio > 0.35:
            l_clean = cv2.bilateralFilter(l_chan, d=5, sigmaColor=24, sigmaSpace=24)
            lut = GAMMA_LUTS[0.62] if mean_l < 65.0 else GAMMA_LUTS[0.78]
            l_tone = cv2.LUT(l_clean, lut)
            l_eq = CLAHE_BOOST.apply(l_tone)
        elif over_ratio > 0.14 or mean_l > 175.0:
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
    Backlight-resilient hand sanity & anti-face filter:
    - Confidence >= 0.45 (fine-tuned live on the Rust side via Calibration UI)
    - Hand bounding_box_area > 0.008 (supports normal & distant desk distances)
    - Palm center (Landmark 9) does not overlap detected face centers
    """
    if score < 0.45 or len(lm_list) < 21:
        return False

    xs = [p[0] for p in lm_list]
    ys = [p[1] for p in lm_list]
    bbox_w = max(xs) - min(xs)
    bbox_h = max(ys) - min(ys)
    if (bbox_w * bbox_h) <= 0.008:
        return False

    px, py = lm_list[9][0], lm_list[9][1]
    for fx, fy, fw, fh in face_boxes:
        face_cx = fx + fw * 0.5
        face_cy = fy + fh * 0.5
        radius = max(fw, fh) * 0.42
        if math.hypot(px - face_cx, py - face_cy) < radius:
            return False

    return True


def refine_hand_roi_zoom(
    hand_detector,
    rgb_frame: np.ndarray,
    lm_coords: list,
    orig_score: float,
    orig_cat: str,
    face_boxes: list,
):
    """
    ROI Zoom Crop refinement for small/distant hands (`bbox_area < 0.085`).
    Upsamples the padded hand crop via bicubic interpolation so `HandLandmarker`
    re-localizes fingertip & PIP/DIP joints with higher spatial resolution.
    """
    h, w = rgb_frame.shape[:2]
    xs = [p[0] for p in lm_coords]
    ys = [p[1] for p in lm_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    bw = max_x - min_x
    bh = max_y - min_y
    if (bw * bh) >= 0.085 or bw <= 0.01 or bh <= 0.01:
        return lm_coords, orig_score, orig_cat

    pad = max(bw, bh) * 0.42
    x0 = max(0, int(math.floor((min_x - pad) * w)))
    y0 = max(0, int(math.floor((min_y - pad) * h)))
    x1 = min(w, int(math.ceil((max_x + pad) * w)))
    y1 = min(h, int(math.ceil((max_y + pad) * h)))
    cw = x1 - x0
    ch = y1 - y0
    if cw < 24 or ch < 24:
        return lm_coords, orig_score, orig_cat

    roi = rgb_frame[y0:y1, x0:x1]
    target_dim = 288
    scale = max(1.0, float(target_dim) / float(max(cw, ch)))
    rw = max(cw, int(round(cw * scale)))
    rh = max(ch, int(round(ch * scale)))
    roi_up = cv2.resize(roi, (rw, rh), interpolation=cv2.INTER_CUBIC) if scale > 1.05 else np.ascontiguousarray(roi)

    try:
        roi_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_up)
        roi_res = hand_detector.detect(roi_mp)
        if not roi_res or not roi_res.hand_landmarks or not roi_res.handedness:
            return lm_coords, orig_score, orig_cat

        best_candidate = None
        for lms, handed_list in zip(roi_res.hand_landmarks, roi_res.handedness):
            if not handed_list:
                continue
            top_cat = handed_list[0]
            r_score = float(top_cat.score)
            r_cat = (top_cat.category_name or "").lower() or orig_cat
            mapped = [
                [
                    round(max(0.0, min(1.0, (x0 + float(p.x) * cw) / float(w))), 4),
                    round(max(0.0, min(1.0, (y0 + float(p.y) * ch) / float(h))), 4),
                    round(float(p.z) * (float(cw) / float(w)), 4),
                ]
                for p in lms
            ]
            if not validate_real_hand(mapped, r_score, face_boxes):
                continue
            if r_score >= orig_score * 0.90:
                if best_candidate is None or r_score > best_candidate[1]:
                    best_candidate = (mapped, r_score, r_cat)

        if best_candidate is not None:
            return best_candidate
    except Exception:
        pass

    return lm_coords, orig_score, orig_cat


def extract_face_sface_descriptor(face_recognizer, rgb_frame: np.ndarray, det, w: int, h: int):
    """
    Build YuNet-compatible 15-element pixel box from BlazeFace 6 keypoints,
    align via `FaceRecognizerSF.alignCrop` (112x112), and return `(face_5pt_norm, face_emb_128d)`.
    """
    kps = getattr(det, "keypoints", None)
    if not kps or len(kps) < 4:
        return None, None

    def kp_px(idx: int):
        kp = kps[idx]
        nx = max(0.0, min(1.0, float(kp.x)))
        ny = max(0.0, min(1.0, float(kp.y)))
        return nx * float(w), ny * float(h), nx, ny

    e0x, e0y, e0nx, e0ny = kp_px(0)
    e1x, e1y, e1nx, e1ny = kp_px(1)
    # Order eye pair left-to-right in image X (subject's right eye first for ArcFace/SFace)
    if e0x <= e1x:
        rex, rey, renx, reny = e0x, e0y, e0nx, e0ny
        lex, ley, lenx, leny = e1x, e1y, e1nx, e1ny
    else:
        rex, rey, renx, reny = e1x, e1y, e1nx, e1ny
        lex, ley, lenx, leny = e0x, e0y, e0nx, e0ny

    nx_px, ny_px, nnx, nny = kp_px(2)
    mcx, mcy, _, _ = kp_px(3)

    # Approximate left/right mouth corners along eye axis (ratio 0.414 from ArcFace 5-pt template)
    eye_dx = lex - rex
    eye_dy = ley - rey
    rmx = mcx - 0.414 * eye_dx
    rmy = mcy - 0.414 * eye_dy
    lmx = mcx + 0.414 * eye_dx
    lmy = mcy + 0.414 * eye_dy

    face_5pt = [
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

    if face_recognizer is None:
        return face_5pt, None

    bb = det.bounding_box
    det_vec = np.array(
        [
            float(bb.origin_x),
            float(bb.origin_y),
            float(bb.width),
            float(bb.height),
            rex,
            rey,
            lex,
            ley,
            nx_px,
            ny_px,
            rmx,
            rmy,
            lmx,
            lmy,
            0.99,
        ],
        dtype=np.float32,
    )

    try:
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        aligned = face_recognizer.alignCrop(bgr, det_vec)
        feat = face_recognizer.feature(aligned)
        vec = np.asarray(feat, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if vec.shape[0] == 128 and norm > 1e-6:
            vec = vec / norm
            return face_5pt, [round(float(v), 5) for v in vec]
    except Exception:
        pass

    return face_5pt, None


def main():
    models_dir = Path("models")
    hand_model_path = models_dir / "hand_landmarker.task"
    face_model_path = models_dir / "blaze_face_short_range.tflite"
    sface_model_path = models_dir / "mobilefacenet.onnx"

    if not ensure_model(hand_model_path, HAND_MODEL_URL):
        sys.exit(1)

    hand_opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(hand_model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.45,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

    has_face_model = ensure_model(face_model_path, FACE_MODEL_URL)
    face_detector = None
    if has_face_model:
        try:
            face_opts = vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(face_model_path)),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=0.50,
            )
            face_detector = vision.FaceDetector.create_from_options(face_opts)
        except Exception:
            face_detector = None

    face_recognizer = None
    if sface_model_path.exists() and sface_model_path.stat().st_size > 1_000_000:
        try:
            face_recognizer = cv2.FaceRecognizerSF_create(str(sface_model_path), "")
        except Exception:
            face_recognizer = None

    stdin_buf = sys.stdin.buffer
    stdout_buf = sys.stdout.buffer

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
        face_5pt = None
        face_emb = None
        if face_detector is not None:
            try:
                f_res = face_detector.detect(mp_img)
                if f_res and f_res.detections:
                    # Pick primary/largest face for 5-pt alignment & SFace embedding
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
                    # Put primary largest face box first
                    p_bb = primary_det.bounding_box
                    pfx = max(0.0, float(p_bb.origin_x) / float(w))
                    pfy = max(0.0, float(p_bb.origin_y) / float(h))
                    pfw = min(1.0 - pfx, float(p_bb.width) / float(w))
                    pfh = min(1.0 - pfy, float(p_bb.height) / float(h))
                    primary_box = [round(pfx, 4), round(pfy, 4), round(pfw, 4), round(pfh, 4)]
                    face_boxes.insert(0, primary_box)

                    if getattr(primary_det, "keypoints", None) and len(primary_det.keypoints) >= 2:
                        k0, k1 = primary_det.keypoints[0], primary_det.keypoints[1]
                        p0 = (max(0.0, min(1.0, float(k0.x))), max(0.0, min(1.0, float(k0.y))))
                        p1 = (max(0.0, min(1.0, float(k1.x))), max(0.0, min(1.0, float(k1.y))))
                        left_p, right_p = (p0, p1) if p0[0] <= p1[0] else (p1, p0)
                        face_eyes = [
                            round(left_p[0], 4),
                            round(left_p[1], 4),
                            round(right_p[0], 4),
                            round(right_p[1], 4),
                        ]
                        face_5pt, face_emb = extract_face_sface_descriptor(
                            face_recognizer, rgb, primary_det, int(w), int(h)
                        )
            except Exception:
                pass

        left_out = None
        right_out = None

        try:
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
                    lm_coords, score, cat_name = refine_hand_roi_zoom(
                        hand_detector, rgb, lm_coords, score, cat_name, face_boxes
                    )
                    valid_hands.append((cat_name, {"score": round(score, 4), "lm": lm_coords}))

                if len(valid_hands) == 1:
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
            "face_5pt": face_5pt,
            "face_emb": face_emb,
        }
        line = (json.dumps(resp, separators=(",", ":")) + "\n").encode("utf-8")
        stdout_buf.write(line)
        stdout_buf.flush()


if __name__ == "__main__":
    main()
