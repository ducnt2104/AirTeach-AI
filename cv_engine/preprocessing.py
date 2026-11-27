"""
Adaptive Lighting, Anti-Flicker & Blur Quality Gate (`cv_engine/preprocessing.py` — Tier 1 Vision Engine).

Features:
1. `AdaptiveExposureController`: Samples strided luminance/contrast/flicker statistics strictly every N
   frames (`frame[::4, ::4]`) instead of scanning every pixel on every frame:
   - In normal balanced indoor lighting -> **0-ms true bypass** (`clahe_bypassed=True`).
   - In low-light, backlit, glare, or fluorescent-flicker scenes -> applies cached Gamma LUT +
     CIE LAB L-channel CLAHE (`clipLimit=2.0..3.0`, `tileGridSize=(8, 8)`) only on affected frames.
2. Zero-allocation caching for Gamma LUTs and `cv2.CLAHE` objects (`clipLimit`, `tileGridSize`).
3. Full backward-compatible `preprocess_adaptive_frame` and `compute_laplacian_sharpness` APIs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from cv_engine.rust_bridge import analyze_frame_luma_fast, apply_gamma_contrast_fast


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
    clahe_bypassed: bool = False
    contrast_span: float = 0.0
    flicker_delta: float = 0.0
    sampled_frame: bool = True

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
            "clahe_bypassed": bool(self.clahe_bypassed),
            "contrast_span": round(self.contrast_span, 2),
            "flicker_delta": round(self.flicker_delta, 2),
            "sampled_frame": bool(self.sampled_frame),
        }


def compute_laplacian_sharpness(frame: np.ndarray, is_rgb: bool = True) -> float:
    """Compute Variance of Laplacian sharpness (`Var(∇²Y)`). Detects motion/defocus blur."""
    if frame is None or frame.size == 0:
        return 0.0
    if frame.ndim == 2:
        gray = frame
    else:
        conv = cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(frame, conv)
    # Sample center 50% crop when larger than 240p to preserve sharpness calibration at 4x speedup
    h, w = gray.shape[:2]
    if h >= 320 and w >= 400:
        gray = gray[h // 4 : (3 * h) // 4, w // 4 : (3 * w) // 4]
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@lru_cache(maxsize=256)
def _cached_gamma_lut(gamma_key: int) -> np.ndarray:
    gamma = max(0.25, gamma_key / 1000.0)
    idx = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(np.power(idx, gamma) * 255.0, 0.0, 255.0).astype(np.uint8)
    return lut


def build_gamma_lut(gamma: float) -> np.ndarray:
    gamma_key = int(round(float(np.clip(gamma, 0.25, 2.50)) * 1000.0))
    return _cached_gamma_lut(gamma_key)


_CLAHE_POOL: Dict[Tuple[int, int], cv2.CLAHE] = {}


def _get_cached_clahe(clip_limit: float, tile_grid_size: int = 8) -> cv2.CLAHE:
    clip_key = int(round(max(1.0, min(6.0, float(clip_limit))) * 100.0))
    tile_key = max(4, min(16, int(tile_grid_size)))
    key = (clip_key, tile_key)
    obj = _CLAHE_POOL.get(key)
    if obj is None:
        if len(_CLAHE_POOL) > 64:
            _CLAHE_POOL.clear()
        obj = cv2.createCLAHE(clipLimit=clip_key / 100.0, tileGridSize=(tile_key, tile_key))
        _CLAHE_POOL[key] = obj
    return obj


def apply_highlight_rolloff(l_channel: np.ndarray, knee: float = 195.0) -> np.ndarray:
    """Soft shoulder compression above `knee` to suppress blown window/sun glare."""
    knee_key = int(round(knee))
    lut = _cached_rolloff_lut(knee_key)
    return cv2.LUT(l_channel, lut)


@lru_cache(maxsize=16)
def _cached_rolloff_lut(knee: int) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    k = float(max(140, min(240, knee)))
    compressed = np.where(x > k, k + (x - k) * 0.42, x)
    return np.clip(compressed, 0.0, 255.0).astype(np.uint8)


def _sample_luma_stats(frame: np.ndarray, is_rgb: bool = True, stride: int = 4) -> Tuple[float, float, float]:
    """
    Fast strided exposure statistics on `frame[::stride, ::stride]`.
    Uses Rust SIMD-friendly histogram pass (`<0.05ms`) when available, with NumPy fallback.
    """
    eff_stride = stride if (frame.shape[0] >= 120 and frame.shape[1] >= 120) else 1
    rust_stats = analyze_frame_luma_fast(frame, is_rgb=is_rgb, stride=eff_stride)
    if rust_stats is not None:
        return rust_stats
    sub = frame[::stride, ::stride] if (frame.shape[0] >= 120 and frame.shape[1] >= 120) else frame
    conv = cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY
    gray = cv2.cvtColor(sub, conv) if sub.ndim == 3 else sub
    mean_luma = float(np.mean(gray))
    p5, p95 = (float(v) for v in np.percentile(gray, (5, 95)))
    return mean_luma, p5, p95


class AdaptiveExposureController:
    """
    Stateful N-frame Adaptive Lighting & Anti-Flicker Preprocessor (Tier 1).

    - Checks luminance/contrast/flicker ONLY every `check_interval_n` frames via strided sampling.
    - In sufficient balanced lighting -> **true zero-copy bypass** (CPU ~0ms, keeps natural colors).
    - In low-light, backlit, glare, or flickering lights -> applies precomputed Gamma LUT +
      cached LAB-CLAHE (`clipLimit 2.0–3.0, tileGridSize=(8, 8)`) smoothly without per-frame spikes.
    """

    def __init__(
        self,
        check_interval_n: int = 4,
        low_light_mean_luma: float = 72.0,
        backlit_p95_luma: float = 232.0,
        flicker_delta_luma: float = 14.0,
        clahe_clip_limit: float = 2.4,
        tile_grid_size: int = 8,
        quality_gate_min: float = 22.0,
    ) -> None:
        self.check_interval_n = max(1, int(check_interval_n))
        self.low_light_mean_luma = float(low_light_mean_luma)
        self.backlit_p95_luma = float(backlit_p95_luma)
        self.flicker_delta_luma = float(flicker_delta_luma)
        self.base_clahe_clip = float(np.clip(clahe_clip_limit, 2.0, 3.0))
        self.tile_grid_size = max(4, int(tile_grid_size))
        self.quality_gate_min = float(quality_gate_min)

        self._frame_count = 0
        self._prev_sampled_mean: Optional[float] = None
        self._active_mode = "NORMAL"
        self._active_gamma = 1.0
        self._active_clip = self.base_clahe_clip
        self._active_rolloff = False
        self._last_diag = ExposureDiagnostics(
            mean_luma=120.0,
            p5=30.0,
            p95=210.0,
            lighting_mode="NORMAL",
            gamma_applied=1.0,
            clahe_clip_limit=self.base_clahe_clip,
            laplacian_variance=60.0,
            quality_gate_passed=True,
            clahe_bypassed=True,
            contrast_span=180.0,
            flicker_delta=0.0,
            sampled_frame=True,
        )

    def reset(self) -> None:
        self._frame_count = 0
        self._prev_sampled_mean = None
        self._active_mode = "NORMAL"
        self._active_gamma = 1.0

    def process(
        self,
        frame: np.ndarray,
        is_rgb: bool = True,
        enabled: bool = True,
        force_check: bool = False,
    ) -> Tuple[np.ndarray, ExposureDiagnostics]:
        if frame is None or frame.size == 0 or frame.ndim != 3:
            empty = ExposureDiagnostics(
                mean_luma=0.0,
                p5=0.0,
                p95=0.0,
                lighting_mode="RAW",
                gamma_applied=1.0,
                clahe_clip_limit=self.base_clahe_clip,
                laplacian_variance=0.0,
                quality_gate_passed=False,
                clahe_bypassed=True,
            )
            return frame, empty

        if not enabled:
            if force_check or (self._frame_count % self.check_interval_n == 0):
                mean_luma, p5, p95 = _sample_luma_stats(frame, is_rgb=is_rgb)
                lap_var = compute_laplacian_sharpness(frame, is_rgb=is_rgb)
                self._last_diag = ExposureDiagnostics(
                    mean_luma=mean_luma,
                    p5=p5,
                    p95=p95,
                    lighting_mode="RAW",
                    gamma_applied=1.0,
                    clahe_clip_limit=0.0,
                    laplacian_variance=lap_var,
                    quality_gate_passed=(lap_var >= self.quality_gate_min),
                    clahe_bypassed=True,
                    contrast_span=max(0.0, p95 - p5),
                    sampled_frame=True,
                )
            else:
                self._last_diag = ExposureDiagnostics(
                    **{**asdict(self._last_diag), "lighting_mode": "RAW", "clahe_bypassed": True, "sampled_frame": False}
                )
            self._frame_count += 1
            return frame, self._last_diag

        should_sample = force_check or (self._frame_count % self.check_interval_n == 0)
        self._frame_count += 1

        if should_sample:
            mean_luma, p5, p95 = _sample_luma_stats(frame, is_rgb=is_rgb)
            contrast_span = max(0.0, p95 - p5)
            flicker_delta = abs(mean_luma - self._prev_sampled_mean) if self._prev_sampled_mean is not None else 0.0
            self._prev_sampled_mean = mean_luma

            is_low_light = mean_luma < self.low_light_mean_luma
            is_glare = p95 > self.backlit_p95_luma
            is_backlit = is_glare and (p5 < 50.0 or mean_luma < 108.0)
            is_low_contrast = contrast_span < 58.0
            is_flicker = flicker_delta >= self.flicker_delta_luma

            if is_low_light:
                self._active_mode = "LOW_LIGHT"
                self._active_gamma = float(np.clip(0.50 + 0.38 * (mean_luma / max(1.0, self.low_light_mean_luma)), 0.48, 0.88))
                self._active_clip = float(np.clip(self.base_clahe_clip + 0.4, 2.0, 3.0))
                self._active_rolloff = is_glare
            elif is_backlit:
                self._active_mode = "BACKLIT"
                self._active_gamma = float(np.clip(0.68 + 0.16 * (mean_luma / 110.0), 0.65, 0.86))
                self._active_clip = float(np.clip(self.base_clahe_clip + 0.3, 2.0, 3.0))
                self._active_rolloff = True
            elif is_glare:
                self._active_mode = "GLARE"
                self._active_gamma = float(np.clip(1.10 + max(0.0, mean_luma - 175.0) / 180.0, 1.05, 1.32))
                self._active_clip = float(np.clip(self.base_clahe_clip - 0.2, 2.0, 3.0))
                self._active_rolloff = True
            elif is_flicker:
                self._active_mode = "FLICKER"
                self._active_gamma = 1.0
                self._active_clip = float(np.clip(self.base_clahe_clip, 2.0, 3.0))
                self._active_rolloff = False
            elif is_low_contrast:
                self._active_mode = "LOW_CONTRAST"
                self._active_gamma = 0.94
                self._active_clip = float(np.clip(self.base_clahe_clip, 2.0, 3.0))
                self._active_rolloff = False
            else:
                self._active_mode = "NORMAL"
                self._active_gamma = 1.0
                self._active_clip = self.base_clahe_clip
                self._active_rolloff = False
        else:
            mean_luma = self._last_diag.mean_luma
            p5 = self._last_diag.p5
            p95 = self._last_diag.p95
            contrast_span = self._last_diag.contrast_span
            flicker_delta = self._last_diag.flicker_delta

        # Fast-path BYPASS when lighting is normal — saves full LAB transform & memory bandwidth
        if self._active_mode == "NORMAL":
            lap_var = (
                compute_laplacian_sharpness(frame, is_rgb=is_rgb)
                if should_sample
                else self._last_diag.laplacian_variance
            )
            self._last_diag = ExposureDiagnostics(
                mean_luma=mean_luma,
                p5=p5,
                p95=p95,
                lighting_mode="NORMAL",
                gamma_applied=1.0,
                clahe_clip_limit=self._active_clip,
                laplacian_variance=lap_var,
                quality_gate_passed=(lap_var >= self.quality_gate_min),
                clahe_bypassed=True,
                contrast_span=contrast_span,
                flicker_delta=flicker_delta,
                sampled_frame=should_sample,
            )
            return frame, self._last_diag

        # Enhanced path ONLY under adverse lighting (`LOW_LIGHT` / `BACKLIT` / `GLARE` / `FLICKER` / `LOW_CONTRAST`)
        to_lab = cv2.COLOR_RGB2LAB if is_rgb else cv2.COLOR_BGR2LAB
        from_lab = cv2.COLOR_LAB2RGB if is_rgb else cv2.COLOR_LAB2BGR
        lab = cv2.cvtColor(frame, to_lab)
        l_chan, a_chan, b_chan = cv2.split(lab)

        l_work = l_chan
        if self._active_rolloff:
            l_work = apply_highlight_rolloff(l_work, knee=192.0)
        if abs(self._active_gamma - 1.0) > 0.02:
            l_work = cv2.LUT(l_work, build_gamma_lut(self._active_gamma))

        clahe = _get_cached_clahe(self._active_clip, self.tile_grid_size)
        l_eq = clahe.apply(l_work)
        lab_eq = cv2.merge((l_eq, a_chan, b_chan))
        out = cv2.cvtColor(lab_eq, from_lab)

        lap_var = (
            compute_laplacian_sharpness(out, is_rgb=is_rgb)
            if should_sample
            else self._last_diag.laplacian_variance
        )
        self._last_diag = ExposureDiagnostics(
            mean_luma=mean_luma,
            p5=p5,
            p95=p95,
            lighting_mode=self._active_mode,
            gamma_applied=self._active_gamma,
            clahe_clip_limit=self._active_clip,
            laplacian_variance=lap_var,
            quality_gate_passed=(lap_var >= self.quality_gate_min),
            clahe_bypassed=False,
            contrast_span=contrast_span,
            flicker_delta=flicker_delta,
            sampled_frame=should_sample,
        )
        return out, self._last_diag


def preprocess_adaptive_frame(
    frame: np.ndarray,
    is_rgb: bool = True,
    quality_gate_min: float = 38.0,
    auto_lightweight_normal: bool = False,
) -> Tuple[np.ndarray, ExposureDiagnostics]:
    """
    Stateless one-shot adaptive preprocessing entrypoint (100% API-compatible).
    Optimized to sample strided exposure metrics BEFORE paying full LAB conversion cost
    when `auto_lightweight_normal=True` in balanced indoor light.
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
            clahe_bypassed=True,
        )
        return frame, empty_diag

    # Strided pre-check avoids full-frame LAB split when normal light + auto_lightweight_normal
    if auto_lightweight_normal:
        mean_s, p5_s, p95_s = _sample_luma_stats(frame, is_rgb=is_rgb, stride=4)
        if 78.0 <= mean_s <= 182.0 and p95_s <= 232.0 and (p95_s - p5_s) >= 75.0:
            lap_var = compute_laplacian_sharpness(frame, is_rgb=is_rgb)
            return frame, ExposureDiagnostics(
                mean_luma=mean_s,
                p5=p5_s,
                p95=p95_s,
                lighting_mode="NORMAL",
                gamma_applied=1.0,
                clahe_clip_limit=2.0,
                laplacian_variance=lap_var,
                quality_gate_passed=(lap_var >= float(quality_gate_min)),
                clahe_bypassed=True,
                contrast_span=(p95_s - p5_s),
            )

    to_lab = cv2.COLOR_RGB2LAB if is_rgb else cv2.COLOR_BGR2LAB
    from_lab = cv2.COLOR_LAB2RGB if is_rgb else cv2.COLOR_LAB2BGR

    lab = cv2.cvtColor(frame, to_lab)
    l_chan, a_chan, b_chan = cv2.split(lab)

    mean_luma = float(np.mean(l_chan))
    l_sub = l_chan[::4, ::4] if (l_chan.shape[0] >= 120 and l_chan.shape[1] >= 120) else l_chan
    p5, p95 = (float(x) for x in np.percentile(l_sub, (5, 95)))

    is_low_light = mean_luma < 70.0
    is_glare = p95 > 235.0
    is_backlit = is_glare and (p5 < 48.0 or mean_luma < 105.0)
    dynamic_range = p95 - p5

    if (
        auto_lightweight_normal
        and not is_low_light
        and not is_glare
        and 80.0 <= mean_luma <= 180.0
        and dynamic_range >= 85.0
    ):
        lap_var = compute_laplacian_sharpness(l_chan, is_rgb=is_rgb)
        return frame, ExposureDiagnostics(
            mean_luma=mean_luma,
            p5=p5,
            p95=p95,
            lighting_mode="NORMAL",
            gamma_applied=1.0,
            clahe_clip_limit=2.0,
            laplacian_variance=lap_var,
            quality_gate_passed=(lap_var >= float(quality_gate_min)),
            clahe_bypassed=True,
            contrast_span=dynamic_range,
        )

    l_work = l_chan
    gamma_applied = 1.0

    if is_glare:
        l_work = apply_highlight_rolloff(l_work, knee=190.0)

    if is_low_light:
        lighting_mode = "LOW_LIGHT"
        gamma_applied = float(np.clip(0.46 + 0.42 * (mean_luma / 70.0), 0.46, 0.88))
        l_work = cv2.LUT(l_work, build_gamma_lut(gamma_applied))
        if is_glare:
            l_work = apply_highlight_rolloff(l_work, knee=192.0)
        clahe_clip = float(np.clip(2.6 + (70.0 - mean_luma) * (1.4 / 70.0), 1.5, 4.0))
    elif is_backlit:
        lighting_mode = "BACKLIT"
        gamma_applied = float(np.clip(0.68 + 0.18 * (mean_luma / 105.0), 0.65, 0.86))
        l_work = cv2.LUT(l_work, build_gamma_lut(gamma_applied))
        l_work = apply_highlight_rolloff(l_work, knee=194.0)
        clahe_clip = float(np.clip(2.7 - (p95 - 235.0) * 0.015, 2.0, 3.0))
    elif is_glare:
        lighting_mode = "GLARE"
        if mean_luma > 180.0:
            gamma_applied = float(np.clip(1.12 + (mean_luma - 180.0) / 180.0, 1.10, 1.38))
            l_work = cv2.LUT(l_work, build_gamma_lut(gamma_applied))
        clahe_clip = float(np.clip(2.2 - (p95 - 235.0) * 0.01, 2.0, 3.0))
    else:
        lighting_mode = "NORMAL"
        clahe_clip = float(np.clip(2.2 + (115.0 - abs(mean_luma - 125.0)) / 240.0, 2.0, 3.0))

    clahe = _get_cached_clahe(clahe_clip, tile_grid_size=8)
    l_eq = clahe.apply(l_work)

    if is_low_light or is_backlit:
        l_eq = cv2.bilateralFilter(l_eq, d=5, sigmaColor=40, sigmaSpace=40)

    lab_eq = cv2.merge((l_eq, a_chan, b_chan))
    img_eq = cv2.cvtColor(lab_eq, from_lab)
    if is_low_light:
        img_eq = cv2.bilateralFilter(img_eq, d=5, sigmaColor=40, sigmaSpace=40)

    gaussian_blur = cv2.GaussianBlur(img_eq, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(img_eq, 1.45, gaussian_blur, -0.45, 0.0)

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
        clahe_bypassed=False,
        contrast_span=dynamic_range,
    )
    return sharpened, diagnostics


def sharpen_and_contrast_roi(
    roi_frame: np.ndarray,
    *,
    is_rgb: bool = True,
    unsharp_strength: float = 0.55,
    contrast_stretch: bool = True,
) -> np.ndarray:
    """
    ROI-Only Local Sharpening & Adaptive Contrast Stretch (< 0.6ms on 224x224 ROI on i3 CPU).
    NEVER runs on full 720p frames — only on the cropped square hand ROI so distant 2.5m-3m
    finger joints become crisp for MediaPipe HandLandmarker without burning CPU.
    """
    if roi_frame is None or roi_frame.size == 0:
        return roi_frame

    work = roi_frame
    if contrast_stretch:
        sub = work[::4, ::4] if (work.shape[0] >= 64 and work.shape[1] >= 64) else work
        conv = cv2.COLOR_RGB2GRAY if is_rgb else cv2.COLOR_BGR2GRAY
        gray = cv2.cvtColor(sub, conv) if sub.ndim == 3 else sub
        mean_y = float(np.mean(gray))
        p2, p98 = (float(v) for v in np.percentile(gray, (2, 98)))

        # Low-light local gamma boost / contrast stretch (Rust fast-path with OpenCV fallback)
        if mean_y < 85.0:
            gamma = float(np.clip(0.55 + 0.35 * (mean_y / 85.0), 0.55, 0.90))
            rust_out = apply_gamma_contrast_fast(work, gamma=gamma, alpha=1.0, beta=0.0)
            work = rust_out if rust_out is not None else cv2.LUT(work, build_gamma_lut(gamma))
        elif (p98 - p2) < 90.0 and (p98 - p2) > 12.0:
            # Fast linear contrast stretch on low-contrast distant hand ROI
            alpha = min(2.2, 210.0 / max(12.0, p98 - p2))
            beta = -p2 * alpha + 18.0
            rust_out = apply_gamma_contrast_fast(work, gamma=1.0, alpha=alpha, beta=beta)
            work = rust_out if rust_out is not None else cv2.convertScaleAbs(work, alpha=alpha, beta=beta)

    strength = max(0.0, min(1.5, float(unsharp_strength)))
    if strength > 0.01:
        blur = cv2.GaussianBlur(work, (3, 3), 0.0)
        work = cv2.addWeighted(work, 1.0 + strength, blur, -strength, 0.0)
    return work
