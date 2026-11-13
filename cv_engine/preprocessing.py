#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Multi-Condition Image Pre-Processing Pipeline (`cv_engine/preprocessing.py`)
for AirTeach AI (DNT - Victer).

Strictly implements:
1. Luminance & Exposure Diagnostics in CIE LAB color space:
   - Computes `mean_luma`, 5th percentile `p5` (shadow floor), and 95th percentile `p95` (glare peak) on channel `L`.
2. Dynamic Low-Light / Glare / Normal Exposure Recovery:
   - If `mean_luma < 70.0`:
     * Applies Edge-Preserving Bilateral Filter (`d=5, sigmaColor=40, sigmaSpace=40`) BEFORE sharpening.
     * Applies dynamic Adaptive Gamma Correction (`gamma < 1.0` computed from `mean_luma`).
   - If `p95 > 235.0`:
     * Applies Highlight Roll-off soft-knee compression on saturated highlight regions so finger/facial contours are preserved.
   - Applies LAB L-channel CLAHE (`tileGridSize=(8, 8)`, dynamic `clipLimit` in `[1.5, 4.0]`) preserving `A`/`B` skin chrominance.
3. Controlled Unsharp Masking (Denoise-then-Sharpen):
   - Applies `1.45 * img - 0.45 * gaussian_blur` (`cv2.addWeighted(img_eq, 1.45, gaussian_blur, -0.45, 0.0)`).
4. Blur Quality Gate:
   - Computes `cv2.Laplacian(gray, cv2.CV_64F).var()` to reject motion-blurred or out-of-focus frames during Face ID enrollment/login.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import cv2
import numpy as np


@dataclass
class ExposureDiagnostics:
    mean_luma: float
    p5: float
    p95: float
    lighting_mode: str
    gamma_applied: float
    clahe_clip_limit: float
    laplacian_variance: float
    quality_gate_passed: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "mean_luma": round(self.mean_luma, 2),
            "p5": round(self.p5, 2),
            "p95": round(self.p95, 2),
            "lighting_mode": self.lighting_mode,
            "gamma_applied": round(self.gamma_applied, 3),
            "clahe_clip_limit": round(self.clahe_clip_limit, 2),
            "laplacian_variance": round(self.laplacian_variance, 2),
            "quality_gate_passed": bool(self.quality_gate_passed),
        }


def compute_laplacian_sharpness(frame: np.ndarray, is_rgb: bool = True) -> float:
    """Compute true 2D Laplacian Variance (`cv2.Laplacian(gray, cv2.CV_64F).var()`)."""
    if frame is None or frame.size == 0:
        return 0.0
    if frame.ndim == 2:
        gray = frame
    else:
        code = cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(frame, code)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def build_gamma_lut(gamma: float) -> np.ndarray:
    """Build an 8-bit Look-Up Table for `out = 255 * (in / 255) ** gamma`."""
    g = float(np.clip(gamma, 0.25, 2.50))
    indices = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(np.power(indices, g) * 255.0, 0.0, 255.0).astype(np.uint8)
    return lut


def apply_highlight_rolloff(l_channel: np.ndarray, knee: float = 195.0) -> np.ndarray:
    """
    Smoothly compress saturated highlights (`p95 > 235`) above `knee` using hyperbolic tangent
    roll-off so backlit windows or ceiling lamps do not erase finger or facial boundaries.
    """
    l_float = l_channel.astype(np.float32)
    mask = l_float > knee
    if np.any(mask):
        headroom = 255.0 - knee
        excess = (l_float[mask] - knee) / max(1.0, headroom * 1.35)
        l_float[mask] = knee + headroom * 0.78 * np.tanh(excess)
    return np.clip(l_float, 0.0, 255.0).astype(np.uint8)


def preprocess_adaptive_frame(
    frame: np.ndarray,
    is_rgb: bool = True,
    quality_gate_min: float = 38.0,
) -> Tuple[np.ndarray, ExposureDiagnostics]:
    """
    Execute the complete multi-condition adaptive preprocessing chain on an RGB or BGR frame.
    Returns `(enhanced_frame, ExposureDiagnostics)`.
    """
    if frame is None or frame.size == 0 or frame.ndim != 3:
        empty_diag = ExposureDiagnostics(
            mean_luma=0.0,
            p5=0.0,
            p95=0.0,
            lighting_mode="NORMAL",
            gamma_applied=1.0,
            clahe_clip_limit=2.0,
            laplacian_variance=0.0,
            quality_gate_passed=False,
        )
        return frame, empty_diag

    to_lab = cv2.COLOR_RGB2LAB if is_rgb else cv2.COLOR_BGR2LAB
    from_lab = cv2.COLOR_LAB2RGB if is_rgb else cv2.COLOR_LAB2BGR

    # 1. Luminance & Exposure Diagnostics in CIE LAB
    lab = cv2.cvtColor(frame, to_lab)
    l_chan, a_chan, b_chan = cv2.split(lab)

    mean_luma = float(np.mean(l_chan))
    p5, p95 = [float(x) for x in np.percentile(l_chan, [5, 95])]

    is_low_light = mean_luma < 70.0
    is_glare = p95 > 235.0

    l_work = l_chan
    gamma_applied = 1.0

    # 2. Adaptive Gamma Correction (mean_luma < 70) & Highlight Roll-off (p95 > 235)
    # Note: A backlit scene can have both low average luminance (mean_luma < 70) AND window glare (p95 > 235).
    if is_glare:
        l_work = apply_highlight_rolloff(l_work, knee=190.0)

    if is_low_light:
        lighting_mode = "LOW_LIGHT"
        # Dynamic gamma < 1.0 computed continuously from mean_luma
        gamma_applied = float(np.clip(0.46 + 0.42 * (mean_luma / 70.0), 0.46, 0.88))
        l_work = cv2.LUT(l_work, build_gamma_lut(gamma_applied))
        if is_glare:
            l_work = apply_highlight_rolloff(l_work, knee=192.0)
        clahe_clip = float(np.clip(2.6 + (70.0 - mean_luma) * (1.4 / 70.0), 1.5, 4.0))
    elif is_glare:
        lighting_mode = "GLARE"
        if mean_luma > 180.0:
            gamma_applied = float(np.clip(1.12 + (mean_luma - 180.0) / 180.0, 1.10, 1.38))
            l_work = cv2.LUT(l_work, build_gamma_lut(gamma_applied))
        clahe_clip = float(np.clip(1.9 - (p95 - 235.0) * 0.015, 1.5, 4.0))
    else:
        lighting_mode = "NORMAL"
        clahe_clip = float(np.clip(2.1 + (115.0 - abs(mean_luma - 125.0)) / 140.0, 1.5, 4.0))

    # LAB L-channel CLAHE with tileGridSize=(8, 8) and dynamic clipLimit in [1.5, 4.0]
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_work)

    # 3. Denoise-then-Sharpen:
    # In low-light (high ISO grain amplified by gamma + CLAHE), run mandatory Bilateral Filter
    # (d=5, sigmaColor=40, sigmaSpace=40) to smooth noise while preserving edges BEFORE Unsharp Masking.
    if is_low_light:
        l_eq = cv2.bilateralFilter(l_eq, d=5, sigmaColor=40, sigmaSpace=40)

    lab_eq = cv2.merge((l_eq, a_chan, b_chan))
    img_eq = cv2.cvtColor(lab_eq, from_lab)
    if is_low_light:
        img_eq = cv2.bilateralFilter(img_eq, d=5, sigmaColor=40, sigmaSpace=40)

    # Controlled Unsharp Masking: 1.45 * img - 0.45 * gaussian_blur
    gaussian_blur = cv2.GaussianBlur(img_eq, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(img_eq, 1.45, gaussian_blur, -0.45, 0.0)

    # 4. Blur Quality Gate via Laplacian Variance
    lap_var = compute_laplacian_sharpness(sharpened, is_rgb=is_rgb)
    diagnostics = ExposureDiagnostics(
        mean_luma=mean_luma,
        p5=p5,
        p95=p95,
        lighting_mode=lighting_mode,
        gamma_applied=gamma_applied,
        clahe_clip_limit=clahe_clip,
        laplacian_variance=lap_var,
        quality_gate_passed=(lap_var >= float(quality_gate_min)),
    )
    return sharpened, diagnostics
