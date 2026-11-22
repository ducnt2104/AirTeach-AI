#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Pipeline & Hardware Telemetry (`cv_engine/performance_pipeline.py`)
for AirTeach AI (DNT - Victer):
1. Detects available Hardware Acceleration delegates (ONNX Runtime DirectML/CUDA/CPU,
   OpenCV OpenCL / multi-threading, MediaPipe XNNPACK/GPU).
2. Enforces Dynamic Resolution Scaling: caps CV inference dimensions (`<= 640x480`)
   so Hand/Face tracking sustains 60FPS while vector strokes & math annotations render at full screen DPI.
3. Tracks live telemetry (`cv_fps`, `render_fps`, `ai_tokens_per_sec`, `ai_mode`, `active_model`)
   for the Status Pill across UI modes.
4. Supports Dirty-Rect / Offscreen Backing Cache flag tracking for zero-freeze Canvas painting.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import os
import time
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2


class PerformanceProfile(str, Enum):
    """Adaptive performance tiers for weak CPUs (Intel Core i3 4th/5th gen) up to high-end."""

    ULTRA_LIGHT = "ULTRA_LIGHT"
    BALANCED = "BALANCED"
    HIGH_QUALITY = "HIGH_QUALITY"

    @property
    def label_vi(self) -> str:
        return {
            PerformanceProfile.ULTRA_LIGHT: "Siêu nhẹ (i3 cũ)",
            PerformanceProfile.BALANCED: "Cân bằng",
            PerformanceProfile.HIGH_QUALITY: "Chất lượng cao",
        }[self]


@dataclass(frozen=True)
class ProfilePreset:
    profile: PerformanceProfile
    search_size: Tuple[int, int]
    roi_size: int
    pose_interval_frames: int
    face_interval_sec: float
    exposure_interval_n: int
    model_complexity: int
    enable_roi_sharpen: bool
    enable_full_frame_clahe: bool


PROFILE_PRESETS: Dict[PerformanceProfile, ProfilePreset] = {
    PerformanceProfile.ULTRA_LIGHT: ProfilePreset(
        profile=PerformanceProfile.ULTRA_LIGHT,
        search_size=(320, 180),
        roi_size=224,
        pose_interval_frames=15,
        face_interval_sec=2.0,
        exposure_interval_n=8,
        model_complexity=0,
        enable_roi_sharpen=True,
        enable_full_frame_clahe=False,
    ),
    PerformanceProfile.BALANCED: ProfilePreset(
        profile=PerformanceProfile.BALANCED,
        search_size=(426, 240),
        roi_size=256,
        pose_interval_frames=10,
        face_interval_sec=1.5,
        exposure_interval_n=4,
        model_complexity=1,
        enable_roi_sharpen=True,
        enable_full_frame_clahe=False,
    ),
    PerformanceProfile.HIGH_QUALITY: ProfilePreset(
        profile=PerformanceProfile.HIGH_QUALITY,
        search_size=(640, 360),
        roi_size=320,
        pose_interval_frames=6,
        face_interval_sec=1.0,
        exposure_interval_n=3,
        model_complexity=1,
        enable_roi_sharpen=True,
        enable_full_frame_clahe=True,
    ),
}


class AutoProfileController:
    """
    Monitors rolling CV inference latency (`avg_inference_ms`).
    If `avg_inference_ms > 45.0` continuously in auto mode, automatically switches to `ULTRA_LIGHT`
    (Potato-PC Mode) to keep UI/Canvas 60 FPS without CPU bottleneck.
    """

    def __init__(
        self,
        initial_profile: PerformanceProfile = PerformanceProfile.BALANCED,
        auto_adapt: bool = True,
        downgrade_threshold_ms: float = 45.0,
        window_size: int = 10,
    ) -> None:
        self.active_profile = initial_profile
        self.auto_adapt = bool(auto_adapt)
        self.downgrade_threshold_ms = float(downgrade_threshold_ms)
        self._latencies: Deque[float] = deque(maxlen=max(4, int(window_size)))
        self._high_latency_streak = 0
        self.auto_downgraded = False

    def set_manual_profile(self, profile: PerformanceProfile, auto_adapt: bool = False) -> None:
        self.active_profile = profile
        self.auto_adapt = bool(auto_adapt)
        self._high_latency_streak = 0
        self.auto_downgraded = False

    @property
    def preset(self) -> ProfilePreset:
        return PROFILE_PRESETS[self.active_profile]

    @property
    def avg_inference_ms(self) -> float:
        if not self._latencies:
            return 0.0
        return float(sum(self._latencies) / len(self._latencies))

    def record_inference_ms(self, latency_ms: float) -> PerformanceProfile:
        self._latencies.append(max(0.0, float(latency_ms)))
        if not self.auto_adapt or self.active_profile == PerformanceProfile.ULTRA_LIGHT:
            return self.active_profile

        if len(self._latencies) >= self._latencies.maxlen and self.avg_inference_ms > self.downgrade_threshold_ms:
            self._high_latency_streak += 1
            if self._high_latency_streak >= 5:
                self.active_profile = PerformanceProfile.ULTRA_LIGHT
                self.auto_downgraded = True
                self._high_latency_streak = 0
        else:
            self._high_latency_streak = 0
        return self.active_profile


class StaggeredInferenceScheduler:
    """
    Decouples heavy CV models so they NEVER run synchronously on the same frame:
    - Hand Tracking: runs every AI tick (highest priority).
    - Pose / Teacher Lock: runs every `pose_interval_frames` (10-15 frames, ~2-3 Hz).
    - Face Recognition: runs every `face_interval_sec` (>= 2.0s, ~0.5 Hz) or when teacher lost > 2s.
    """

    def __init__(self) -> None:
        self.frame_idx: int = 0
        self._last_pose_frame: int = -999
        self._last_face_ts: float = -999.0

    def next_tick(
        self,
        now_sec: float,
        preset: ProfilePreset,
        *,
        has_locked_teacher: bool,
        teacher_lost_sec: float,
    ) -> Tuple[bool, bool]:
        """Return `(run_pose, run_face)` for the current AI tick."""
        self.frame_idx += 1
        run_pose = (
            not has_locked_teacher
            or (self.frame_idx - self._last_pose_frame) >= max(1, preset.pose_interval_frames)
        )
        if run_pose:
            self._last_pose_frame = self.frame_idx

        run_face = (
            not has_locked_teacher
            or teacher_lost_sec > 2.0
            or (now_sec - self._last_face_ts) >= max(0.5, preset.face_interval_sec)
        )
        if run_face:
            self._last_face_ts = now_sec

        return run_pose, run_face


@dataclass
class HardwareAccelReport:
    cv2_threads: int
    cv2_opencl_available: bool
    ort_providers: List[str]
    selected_backend: str

    def summary_label(self) -> str:
        return f"{self.selected_backend} ({self.cv2_threads} threads)"


@dataclass
class PipelineTelemetry:
    ai_mode: str = "Hybrid"  # "Local" | "Cloud" | "Hybrid" | "Instant"
    active_model: str = "gemini-3.8-flash"
    tokens_per_sec: float = 0.0
    cv_fps: float = 0.0
    render_fps: float = 60.0
    hw_backend: str = "CPU"
    last_latency_ms: float = 0.0

    def status_pill_text(self) -> str:
        tok_part = f"{self.tokens_per_sec:.1f} tok/s" if self.tokens_per_sec > 0.1 else "Sẵn sàng"
        cv_part = f"CV {self.cv_fps:.0f} FPS" if self.cv_fps > 0.5 else "CV Standby"
        return f"AI: {self.ai_mode} ({self.active_model}) • {tok_part} • {cv_part}"


def detect_hardware_acceleration() -> HardwareAccelReport:
    """Probe & configure fastest safe local CV/ML execution delegates."""
    cpu_cnt = max(2, min(8, os.cpu_count() or 4))
    try:
        cv2.setNumThreads(cpu_cnt)
    except Exception:
        pass

    ocl_avail = False
    try:
        ocl_avail = bool(cv2.ocl.haveOpenCL())
        if ocl_avail:
            cv2.ocl.setUseOpenCL(True)
    except Exception:
        ocl_avail = False

    ort_providers: List[str] = []
    try:
        import onnxruntime as ort  # type: ignore

        ort_providers = list(ort.get_available_providers())
    except Exception:
        ort_providers = []

    if "DmlExecutionProvider" in ort_providers:
        backend = "DirectML-GPU"
    elif "CUDAExecutionProvider" in ort_providers:
        backend = "CUDA-GPU"
    elif ocl_avail:
        backend = "OpenCL-GPU"
    else:
        backend = "CPU-SIMD"

    return HardwareAccelReport(
        cv2_threads=cpu_cnt,
        cv2_opencl_available=ocl_avail,
        ort_providers=ort_providers,
        selected_backend=backend,
    )


def compute_cv_tracking_size(
    src_w: int,
    src_h: int,
    max_w: int = 640,
    max_h: int = 480,
) -> Tuple[int, int]:
    """
    Dynamic Resolution Scaling: bound CV tracking frames to `<= 640x480` (aspect-preserved)
    while Canvas/UI continues rendering crisp 1080p/Retina vectors.
    """
    if src_w <= 0 or src_h <= 0:
        return (640, 480)
    if src_w <= max_w and src_h <= max_h:
        return (src_w, src_h)
    scale = min(float(max_w) / float(src_w), float(max_h) / float(src_h))
    return (max(160, int(round(src_w * scale))), max(120, int(round(src_h * scale))))


class DirtyCanvasBackingTracker:
    """Tracks dirty state for Offscreen/Pixmap backing cache so static strokes/grid aren't repainted."""

    def __init__(self) -> None:
        self._dirty: bool = True
        self._last_key: Optional[Tuple[Any, ...]] = None

    def mark_dirty(self) -> None:
        self._dirty = True

    def needs_rebuild(self, scene_key: Tuple[Any, ...]) -> bool:
        if self._dirty or self._last_key != scene_key:
            return True
        return False

    def mark_clean(self, scene_key: Tuple[Any, ...]) -> None:
        self._dirty = False
        self._last_key = scene_key


class PerformancePipeline:
    """Coordinates telemetry metrics & dynamic scaling across CV, Render, and AI workers."""

    def __init__(self) -> None:
        self.hw_report = detect_hardware_acceleration()
        self.telemetry = PipelineTelemetry(hw_backend=self.hw_report.selected_backend)
        self.backing_tracker = DirtyCanvasBackingTracker()
        self._frame_times: List[float] = []

    def record_render_tick(self, now_sec: Optional[float] = None) -> float:
        t = time.perf_counter() if now_sec is None else float(now_sec)
        self._frame_times.append(t)
        cutoff = t - 1.0
        while self._frame_times and self._frame_times[0] < cutoff:
            self._frame_times.pop(0)
        if len(self._frame_times) >= 2:
            span = max(1e-3, self._frame_times[-1] - self._frame_times[0])
            self.telemetry.render_fps = min(240.0, (len(self._frame_times) - 1) / span)
        return self.telemetry.render_fps

    def update_cv_fps(self, fps: float) -> None:
        self.telemetry.cv_fps = max(0.0, float(fps))

    def update_ai_metrics(
        self,
        *,
        ai_mode: Optional[str] = None,
        active_model: Optional[str] = None,
        tokens_per_sec: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ) -> None:
        if ai_mode is not None:
            self.telemetry.ai_mode = ai_mode
        if active_model is not None:
            self.telemetry.active_model = active_model
        if tokens_per_sec is not None:
            self.telemetry.tokens_per_sec = max(0.0, float(tokens_per_sec))
        if latency_ms is not None:
            self.telemetry.last_latency_ms = max(0.0, float(latency_ms))
