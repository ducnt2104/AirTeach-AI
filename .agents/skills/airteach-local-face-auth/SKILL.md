---
name: airteach-local-face-auth
description: >-
  Specialist skill for offline face authentication using OpenCV DNN (YuNet + SFace),
  anti-spoofing liveness check, Fernet embedding encryption, PBKDF2 PIN fallback, and SQLite profile stores.
---

# AirTeach AI — Local Face Authentication & Profile Skill

## 1. Zero Cloud Biometric Privacy
- All biometric detection and feature extraction runs locally using open-source ONNX models:
  - `face_detection_yunet_2023mar.onnx` (YuNet): Face bounding box + 5 facial landmarks (eyes, nose, mouth corners).
  - `face_recognition_sface_2021dec.onnx` (SFace): 128-dimensional normalized feature vector.
- Raw face images are NEVER stored on disk permanently.
- Extracted 128D feature vectors are encrypted at rest using AES-128-CBC (`cryptography.fernet`).
- Encryption key is stored securely in `data/.keystore` with OS-level permission restrictions.

## 2. Anti-Spoofing Liveness Check
- Tracks relative nose-to-eye displacement across 15 consecutive frames:
  $$\Delta x = |x_{\text{nose}} - x_{\text{mid-eyes}}|$$
- Rejects static 2D printed photos or smartphone screens that exhibit zero micro-head rotation.
- Requires cosine similarity $\ge 0.363$ for valid authentication.

## 3. Mandatory PIN Fallback
- Every teacher profile MUST have an associated PIN (4–6 digits), salted and hashed with `hashlib.pbkdf2_hmac("sha256", pin, salt, 100_000)`.
- If camera is unavailable, lighting is poor, or face match fails, the PIN pad provides instant, reliable access.
