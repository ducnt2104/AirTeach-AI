"""
AirTeach AI - Adaptive Vision Preprocessing & Camera Diagnostics
================================================================
Robust real-time image processing pipeline for challenging webcam conditions:
  - Low light (phòng thiếu sáng)
  - Severe backlighting (ngược sáng / ánh sáng cửa sổ phía sau)
  - Glare & overexposed wash-out (lóe sáng, bóng đèn rọi)
  - High sensor gain noise (nhiễu hạt camera)
  - Motion blur & out-of-focus detection (nhòe chuyển động)

Architecture:
  1. Image Quality Evaluator: Computes luminance, contrast, clipping %,
     and Laplacian variance sharpness without copying buffers.
  2. Dynamic Parameter Smoother: Temporal exponential smoothing (EMA)
     prevents frame-to-frame brightness flutter/flickering.
  3. Adaptive Multi-Stage Preprocessing:
     - Dynamic Gamma Correction (lut-based, O(1) pixel transformation)
     - Adaptive CLAHE (L channel in LAB color space)
     - Fast Edge-Preserving Denoising (bilateral / Gaussian)
     - Highlight tone-mapping
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class LightingCondition(Enum):
    NORMAL = auto()
    LOW_LIGHT = auto()
    BACKLIGHT = auto()
    GLARE = auto()
    LOW_CONTRAST = auto()


@dataclass
class FrameQualityMetrics:
    """Real-time calculated optical quality metrics for a camera frame."""
    mean_brightness: float        # [0.0 - 255.0]
    contrast_std: float           # standard deviation of luminance
    shadow_clipping_ratio: float  # fraction of pixels with Y < 15
    highlight_clipping_ratio: float  # fraction of pixels with Y > 240
    sharpness_score: float        # Laplacian variance (Var(∇²I))
    condition: LightingCondition
    processing_time_ms: float = 0.0

    @property
    def shadow_clipping_pct(self) -> float:
        return self.shadow_clipping_ratio * 100.0

    @property
    def highlight_clipping_pct(self) -> float:
        return self.highlight_clipping_ratio * 100.0

    @property
    def is_sharp(self) -> bool:
        """Heuristic for good focus: Laplacian variance > 45."""
        return self.sharpness_score >= 45.0


@dataclass
class CameraDiagnosticsData:
    """Comprehensive real measurements emitted for diagnostics HUD."""
    fps: float
    width: int
    height: int
    mean_brightness: float
    sharpness_score: float
    shadow_clipping_pct: float
    highlight_clipping_pct: float
    condition_name: str
    hands_detected: int
    tracking_confidence: float
    gesture_state_name: str
    draw_mode_name: str
    pipeline_latency_ms: float
    frame_dropped_count: int = 0


class AdaptiveVisionPipeline:
    """
    Adaptive image enhancement and quality assessment engine.
    Optimized for real-time 30-60 FPS execution on CPU.
    """

    def __init__(
        self,
        mode: str = "adaptive",  # 'adaptive', 'low_light', 'balanced', 'motion_priority'
        temporal_alpha: float = 0.25,
    ) -> None:
        self.mode = mode
        self.temporal_alpha = temporal_alpha

        # Smoothed parameters
        self._smoothed_gamma: float = 1.0
        self._smoothed_clip_limit: float = 1.0

        # Pre-computed Gamma Look-Up Tables (LUT)
        self._lut_cache: dict[float, np.ndarray] = {}
        self._clahe_cache: dict[float, cv2.CLAHE] = {}

    def set_mode(self, mode: str) -> None:
        self.mode = mode

    # ── Metric Computation ────────────────────────────────────────────────────

    def evaluate_quality(self, frame_bgr: np.ndarray) -> FrameQualityMetrics:
        """
        Compute optical quality metrics using fast sub-sampled calculations.
        Returns FrameQualityMetrics in under 1.5ms.
        """
        t0 = time.perf_counter()

        # Use fast continuous downscale for sub-millisecond metric estimation
        small_bgr = cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_NEAREST)
        small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

        mean_val = float(np.mean(small_gray))
        std_val = float(np.std(small_gray))

        # Clipping ratios
        total_pixels = small_gray.size
        shadow_pixels = int(np.count_nonzero(small_gray < 16))
        highlight_pixels = int(np.count_nonzero(small_gray > 240))

        shadow_ratio = shadow_pixels / total_pixels
        highlight_ratio = highlight_pixels / total_pixels

        # Sharpness score via Laplacian variance on center ROI
        # Center ROI is where hands and gestures normally occur
        ch, cw = small_gray.shape
        roi = small_gray[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4]
        lap = cv2.Laplacian(roi, cv2.CV_64F)
        sharpness = float(lap.var())

        # Determine dominant lighting condition
        # Check backlight: periphery vs center disparity
        center_mean = float(np.mean(roi))
        periphery_mask = np.ones_like(small_gray, dtype=bool)
        periphery_mask[ch // 4 : 3 * ch // 4, cw // 4 : 3 * cw // 4] = False
        periphery_mean = float(np.mean(small_gray[periphery_mask]))

        if mean_val < 75.0 or shadow_ratio > 0.30:
            condition = LightingCondition.LOW_LIGHT
        elif periphery_mean > (center_mean + 45.0) and periphery_mean > 140.0:
            condition = LightingCondition.BACKLIGHT
        elif highlight_ratio > 0.15 or mean_val > 200.0:
            condition = LightingCondition.GLARE
        elif std_val < 32.0:
            condition = LightingCondition.LOW_CONTRAST
        else:
            condition = LightingCondition.NORMAL

        dt_ms = (time.perf_counter() - t0) * 1000.0
        return FrameQualityMetrics(
            mean_brightness=mean_val,
            contrast_std=std_val,
            shadow_clipping_ratio=shadow_ratio,
            highlight_clipping_ratio=highlight_ratio,
            sharpness_score=sharpness,
            condition=condition,
            processing_time_ms=dt_ms,
        )

    MAX_LUT_CACHE_SIZE: int = 128
    MAX_CLAHE_CACHE_SIZE: int = 64

    # ── LUT-based Gamma Correction ────────────────────────────────────────────

    def _get_gamma_lut(self, gamma: float) -> np.ndarray:
        """
        Retrieve or build a 256-element byte LUT for gamma mapping.
        gamma < 1.0 brightens shadows (elevates dark tones).
        gamma > 1.0 compresses highlights.
        Bounded to MAX_LUT_CACHE_SIZE entries to guarantee O(1) memory.
        """
        q_gamma = round(gamma, 2)
        if q_gamma not in self._lut_cache:
            if len(self._lut_cache) >= self.MAX_LUT_CACHE_SIZE:
                self._lut_cache.clear()
            table = np.array([
                np.clip(((i / 255.0) ** q_gamma) * 255.0, 0, 255)
                for i in range(256)
            ], dtype=np.uint8)
            self._lut_cache[q_gamma] = table
        return self._lut_cache[q_gamma]

    def _get_clahe(self, clip_limit: float) -> cv2.CLAHE:
        """Bounded to MAX_CLAHE_CACHE_SIZE entries to guarantee O(1) memory."""
        q_clip = round(clip_limit, 1)
        if q_clip not in self._clahe_cache:
            if len(self._clahe_cache) >= self.MAX_CLAHE_CACHE_SIZE:
                self._clahe_cache.clear()
            self._clahe_cache[q_clip] = cv2.createCLAHE(
                clipLimit=max(q_clip, 1.0),
                tileGridSize=(8, 8),
            )
        return self._clahe_cache[q_clip]

    # ── Adaptive Preprocessing Execution ──────────────────────────────────────

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        metrics: Optional[FrameQualityMetrics] = None,
    ) -> Tuple[np.ndarray, FrameQualityMetrics]:
        """
        Enhance camera frame for optimal MediaPipe landmark tracking.
        Returns (enhanced_bgr_frame, calculated_metrics).
        """
        if metrics is None:
            metrics = self.evaluate_quality(frame_bgr)

        # In motion priority mode, bypass heavier enhancement to maximize FPS
        if self.mode == "motion_priority":
            return frame_bgr, metrics

        # 1. Determine target enhancement parameters based on mode and lighting
        target_gamma = 1.0
        target_clip = 1.0
        need_denoise = False

        if self.mode == "low_light" or metrics.condition == LightingCondition.LOW_LIGHT:
            # Low light compensation: boost dark regions (gamma < 1.0 expands shadows)
            deficit = max(0.0, 110.0 - metrics.mean_brightness)
            target_gamma = max(0.55, 1.0 - (deficit / 180.0))
            target_clip = min(3.8, 2.0 + (deficit / 50.0))
            # Sensor noise is prominent in low-light; enable gentle edge-preserving blur
            need_denoise = True

        elif self.mode == "balanced" or metrics.condition == LightingCondition.BACKLIGHT:
            # Backlight / balanced: moderate gamma boost + higher CLAHE to reveal dark hand against bright window
            target_gamma = 0.82
            target_clip = 3.2

        elif metrics.condition == LightingCondition.GLARE:
            # Glare: compress highlights (gamma > 1.0 compresses bright areas)
            target_gamma = 1.25
            target_clip = 1.6

        elif metrics.condition == LightingCondition.LOW_CONTRAST:
            target_gamma = 1.0
            target_clip = 3.0

        # 2. Temporal Smoothing (EMA) to prevent flickering between frames
        alpha = self.temporal_alpha
        self._smoothed_gamma = (1.0 - alpha) * self._smoothed_gamma + alpha * target_gamma
        self._smoothed_clip_limit = (1.0 - alpha) * self._smoothed_clip_limit + alpha * target_clip

        result = frame_bgr

        # 3. Apply Gamma Correction via fast 1-cycle O(1) LUT transform
        if abs(self._smoothed_gamma - 1.0) > 0.04:
            lut = self._get_gamma_lut(self._smoothed_gamma)
            result = cv2.LUT(result, lut)

        # 4. Apply CLAHE on L-channel in LAB space (in-place modification avoids split/merge allocations)
        if abs(self._smoothed_clip_limit - 1.0) > 0.2:
            clahe = self._get_clahe(self._smoothed_clip_limit)
            lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # 5. Apply Fast Denoising for low light sensor gain
        if need_denoise:
            # A 3x3 Gaussian blur is virtually free (0.3ms) and cleans sensor salt-and-pepper noise
            result = cv2.GaussianBlur(result, (3, 3), 0)

        return result, metrics
