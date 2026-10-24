"""
AirTeach AI - Hand Tracking & Gesture FSM Engine
=================================================
Runs on a dedicated QThread; communicates via Qt Signals.

Pipeline:
  1. cv2.VideoCapture  → raw BGR frame
  2. CLAHE on L-channel (LAB) → exposure-corrected frame
  3. MediaPipe HandLandmarker → 21 3D landmarks per hand
  4. Normalize all distances by hand-length L_hand = ‖P0 − P9‖₂
  5. Finite-State Machine (6 modes) with debounce counter
  6. Perspective-correction matrix maps air-writing zone → full canvas
  7. Filtered (x, y) emitted via Qt Signals for the canvas thread

Gesture states:
  POINT  – laser pointer (index extended, thumb far)
  DRAW   – write ink   (index extended, thumb pinched to index)
  PAUSE  – pen lift / stroke end  (fist)
  SELECT – region select  (index + middle extended)
  SWIPE_LEFT / SWIPE_RIGHT – page turn  (open palm + horizontal velocity)
  CONFIRM – 👍 thumbs-up held 300 ms
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from enum import Enum, auto
from typing import Callable, Optional

import cv2
import mediapipe as mp
import numpy as np

from src.core.adaptive_vision_pipeline import (
    AdaptiveVisionPipeline,
    CameraDiagnosticsData,
    FrameQualityMetrics,
    LightingCondition,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Draw Mode Enum (Dual DRAW Modes: Air Pen vs Precision Pinch)
# ─────────────────────────────────────────────────────────────────────────────
class DrawMode(Enum):
    INDEX_UP = "index_up"            # Chế độ A: Air Pen (ngón trỏ giơ cao tự nhiên)
    PRECISION_PINCH = "precision_pinch" # Chế độ B: Chụm ngón trỏ và ngón cái để vẽ


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe landmark indices (mp.solutions.hands convention)
# ─────────────────────────────────────────────────────────────────────────────
class LM:
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# ─────────────────────────────────────────────────────────────────────────────
# Gesture State Enum
# ─────────────────────────────────────────────────────────────────────────────
class GestureState(Enum):
    NONE = auto()
    POINT = auto()        # laser pointer
    DRAW = auto()         # ink writing
    PAUSE = auto()        # pen lift / stroke done
    SELECT = auto()       # region selection
    SWIPE_LEFT = auto()   # page ← 
    SWIPE_RIGHT = auto()  # page →
    CONFIRM = auto()      # 👍 thumbs-up


# ─────────────────────────────────────────────────────────────────────────────
# Perspective Transform Helper
# ─────────────────────────────────────────────────────────────────────────────
class PerspectiveMapper:
    """
    Maps a quadrilateral air-writing zone (4 corners calibrated by the user)
    to the full canvas rectangle using cv2.getPerspectiveTransform.

    The 4 source corners should be set in clockwise order:
        top-left, top-right, bottom-right, bottom-left
    in normalized [0, 1] coordinates.
    """

    def __init__(self, canvas_width: int, canvas_height: int) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        # Default: identity mapping (full frame → full canvas)
        self._src_pts = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        self._dst_pts = np.array(
            [
                [0.0, 0.0],
                [float(canvas_width - 1), 0.0],
                [float(canvas_width - 1), float(canvas_height - 1)],
                [0.0, float(canvas_height - 1)],
            ],
            dtype=np.float32,
        )
        self._M: np.ndarray = cv2.getPerspectiveTransform(self._src_pts, self._dst_pts)

    def calibrate(self, src_corners_normalized: np.ndarray) -> bool:
        """
        Update the perspective transform with new 4-corner calibration.

        Args:
            src_corners_normalized: Shape (4, 2) array of [x, y] corners
                in normalized [0, 1] space (clockwise: TL, TR, BR, BL).

        Returns:
            True if calibration succeeded, False if degenerate, collinear, or invalid.
        """
        try:
            arr = np.asarray(src_corners_normalized, dtype=np.float32)
            if arr.shape != (4, 2) or not np.isfinite(arr).all():
                logger.warning("Invalid perspective corners shape or non-finite values.")
                return False
            area = float(cv2.contourArea(arr))
            if area < 1e-4:
                logger.warning("Degenerate perspective quad (area=%f too small).", area)
                return False
            M = cv2.getPerspectiveTransform(arr, self._dst_pts)
            self._src_pts = arr
            self._M = M
            logger.info("Perspective calibration updated. Src corners: %s", self._src_pts.tolist())
            return True
        except (cv2.error, Exception) as exc:
            logger.error("cv2.getPerspectiveTransform failed: %s", exc)
            return False

    def reset_calibration(self) -> None:
        """Reset perspective mapping to default full-frame identity."""
        self._src_pts = np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        self._M = cv2.getPerspectiveTransform(self._src_pts, self._dst_pts)
        logger.info("Perspective calibration reset to identity.")

    def map(self, nx: float, ny: float) -> tuple[float, float]:
        """
        Map a normalized hand position (nx, ny) ∈ [0,1]² to canvas pixel (cx, cy).

        Args:
            nx, ny: Normalized coordinates from MediaPipe (0–1 in frame space).

        Returns:
            (canvas_x, canvas_y) in pixels.
        """
        if not math.isfinite(nx) or not math.isfinite(ny):
            return 0.0, 0.0
        pt = np.array([[[nx, ny]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self._M)
        rx = float(mapped[0][0][0])
        ry = float(mapped[0][0][1])
        if not math.isfinite(rx) or not math.isfinite(ry):
            return 0.0, 0.0
        return rx, ry


# ─────────────────────────────────────────────────────────────────────────────
# Gesture Finite-State Machine
# ─────────────────────────────────────────────────────────────────────────────
class HandGestureFSM:
    """
    Hand Gesture Finite-State Machine for AirTeach AI.

    Implements 6 gesture modes with:
      - Scale-invariant thresholds (normalized by L_hand = ‖P0−P9‖₂).
      - Debounce counter to suppress flickering (debounce_frames = 3).
      - Minimum confidence gate (min_confidence = 0.75).
      - Swipe velocity window + cooldown.
      - Thumbs-up hold timer.

    Args:
        canvas_width:       Canvas width in pixels (for perspective mapping).
        canvas_height:      Canvas height in pixels.
        debounce_frames:    Number of consecutive frames required to confirm transition.
        min_confidence:     Minimum MediaPipe hand detection confidence.
        swipe_velocity_thr: Minimum normalized wrist velocity to count as swipe.
        swipe_window_ms:    Time window for measuring swipe velocity (ms).
        swipe_cooldown_ms:  Minimum interval between successive swipe events (ms).
        confirm_hold_ms:    Time 👍 must be held to trigger CONFIRM (ms).
        on_state_change:    Optional callback(new_state, old_state).
    """

    def __init__(
        self,
        canvas_width: int = 1920,
        canvas_height: int = 1080,
        debounce_frames: int = 3,
        min_confidence: float = 0.75,
        swipe_velocity_thr: float = 0.8,
        swipe_window_ms: int = 250,
        swipe_cooldown_ms: int = 800,
        confirm_hold_ms: int = 300,
        point_thumb_ratio: float = 0.42,
        draw_thumb_ratio: float = 0.34,
        on_state_change: Optional[Callable[[GestureState, GestureState], None]] = None,
    ) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.debounce_frames = debounce_frames
        self.min_confidence = min_confidence
        self.swipe_velocity_thr = swipe_velocity_thr
        self.swipe_window_ms = swipe_window_ms
        self.swipe_cooldown_ms = swipe_cooldown_ms
        self.confirm_hold_ms = confirm_hold_ms
        self.point_thumb_ratio = point_thumb_ratio
        self.draw_thumb_ratio = draw_thumb_ratio
        self.on_state_change = on_state_change

        self.draw_mode: DrawMode = DrawMode.INDEX_UP
        self.pinch_threshold: float = 0.28
        self._gap_frames: int = 0

        self._state: GestureState = GestureState.NONE
        self._pending_state: GestureState = GestureState.NONE
        self._debounce_count: int = 0

        # Wrist position history for swipe detection: deque of (x_norm, t_ms)
        self._wrist_history: deque[tuple[float, float]] = deque(maxlen=30)
        self._last_swipe_t_ms: float = -1e9

        # Thumbs-up hold timer
        self._confirm_start_ms: Optional[float] = None

        # Perspective transform
        self.perspective = PerspectiveMapper(canvas_width, canvas_height)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> GestureState:
        return self._state

    def set_draw_mode(self, mode: str | DrawMode) -> None:
        """Switch between Air Pen (INDEX_UP) and Precision Pinch (PRECISION_PINCH)."""
        if isinstance(mode, str):
            if mode.lower() in ("precision_pinch", "pinch"):
                self.draw_mode = DrawMode.PRECISION_PINCH
            else:
                self.draw_mode = DrawMode.INDEX_UP
        elif isinstance(mode, DrawMode):
            self.draw_mode = mode

    def set_pinch_threshold(self, threshold: float) -> None:
        """Update pinch activation distance normalized by hand length."""
        self.pinch_threshold = max(0.12, min(0.55, float(threshold)))

    def process_landmarks(
        self,
        landmarks: list,            # mediapipe NormalizedLandmark list (21 items)
        handedness_score: float,    # confidence from MediaPipe
        t_ms: float,                # current timestamp in milliseconds
    ) -> tuple[GestureState, float, float]:
        """
        Process one frame's hand landmarks and update FSM state.

        Args:
            landmarks:         List of 21 NormalizedLandmark objects from MediaPipe.
            handedness_score:  Handedness detection confidence (0–1).
            t_ms:              Current frame timestamp in milliseconds.

        Returns:
            (current_state, canvas_x, canvas_y) where canvas_x/y is the
            perspective-mapped position of the index fingertip (P8).
        """
        if handedness_score < self.min_confidence or not landmarks:
            self._gap_frames += 1
            if self._state == GestureState.DRAW:
                if self._gap_frames >= 2:
                    self._transition_to(GestureState.PAUSE, t_ms)
                    return GestureState.PAUSE, -1.0, -1.0
                else:
                    # Momentary 1-frame tracking glitch: hold DRAW
                    return GestureState.DRAW, -1.0, -1.0
            self._transition_to(GestureState.NONE, t_ms)
            return GestureState.NONE, -1.0, -1.0

        self._gap_frames = 0

        # Extract 2D+Z landmark array, shape (21, 3)
        lm = np.array(
            [[l.x, l.y, l.z] for l in landmarks],
            dtype=np.float64,
        )

        # Scale-invariant hand size with foreshortening & tilt compensation:
        palm_h = float(np.linalg.norm(lm[LM.WRIST, :2] - lm[LM.MIDDLE_MCP, :2]))
        palm_w = float(np.linalg.norm(lm[LM.INDEX_MCP, :2] - lm[LM.PINKY_MCP, :2]))
        dz = float(lm[LM.WRIST, 2] - lm[LM.MIDDLE_MCP, 2]) if lm.shape[1] >= 3 else 0.0
        palm_h_3d = math.sqrt(palm_h * palm_h + dz * dz)
        l_hand = max(1e-6, max(palm_h_3d, palm_w * 1.15))

        # Detect candidate gesture
        candidate = self._classify(lm, l_hand, t_ms)

        # Debounce: require debounce_frames consecutive frames of same candidate
        # Note: Impulse events (SWIPE_LEFT, SWIPE_RIGHT) have their own velocity window
        # and cooldown timer, so they transition immediately without multi-frame debounce.
        if candidate in (GestureState.SWIPE_LEFT, GestureState.SWIPE_RIGHT):
            self._transition_to(candidate, t_ms)
            self._pending_state = candidate
            self._debounce_count = self.debounce_frames
        else:
            if candidate == self._pending_state:
                self._debounce_count += 1
            else:
                self._pending_state = candidate
                self._debounce_count = 1

            if self._debounce_count >= self.debounce_frames:
                self._transition_to(candidate, t_ms)

        # Map index tip to canvas coordinates
        tip = lm[LM.INDEX_TIP]
        cx, cy = self.perspective.map(float(tip[0]), float(tip[1]))

        return self._state, cx, cy

    # ── Classification Logic ──────────────────────────────────────────────────

    def _classify(self, lm: np.ndarray, l_hand: float, t_ms: float) -> GestureState:
        """Classify current frame landmarks into a candidate gesture state."""

        # ── Swipe: all 5 fingers open + horizontal wrist velocity ──────────
        if self._all_fingers_open(lm, l_hand):
            wrist_x = float(lm[LM.WRIST, 0])
            self._wrist_history.append((wrist_x, t_ms))
            swipe = self._detect_swipe(t_ms)
            if swipe is not None:
                return swipe
            # Open resting palm is idle, not drawing or pointing
            return GestureState.NONE
        else:
            if self._wrist_history:
                self._wrist_history.clear()

        # ── Confirm 👍: thumb extended upward, all other fingers curled ────
        if self._is_thumbs_up(lm, l_hand):
            return GestureState.CONFIRM

        # ── Fist / PAUSE: all 4 fingertips below their PIP joints ──────────
        if self._is_fist(lm):
            return GestureState.PAUSE

        # Finger extension states
        idx_ext = self._finger_extended(lm, LM.INDEX_TIP, LM.INDEX_PIP)
        mid_ext = self._finger_extended(lm, LM.MIDDLE_TIP, LM.MIDDLE_PIP)
        rng_ext = self._finger_extended(lm, LM.RING_TIP, LM.RING_PIP)
        pnk_ext = self._finger_extended(lm, LM.PINKY_TIP, LM.PINKY_PIP)

        # ── SELECT: index + middle extended, ring + pinky curled ───────────
        if idx_ext and mid_ext and not rng_ext and not pnk_ext:
            return GestureState.SELECT

        # ── DRAW or POINT: index extended, other fingers curled ────────────
        if idx_ext and not mid_ext and not rng_ext and not pnk_ext:
            if self.draw_mode == DrawMode.PRECISION_PINCH:
                # Precision Pinch: Thumb tip to Index tip distance
                pinch_dist = float(
                    np.linalg.norm(lm[LM.THUMB_TIP, :2] - lm[LM.INDEX_TIP, :2])
                ) / l_hand

                pinch_thr = self.pinch_threshold
                if self._state == GestureState.DRAW:
                    pinch_thr += 0.04
                elif self._state == GestureState.POINT:
                    pinch_thr -= 0.04

                if pinch_dist < pinch_thr:
                    return GestureState.DRAW
                else:
                    return GestureState.POINT
            else:
                # Air Pen / Index-Up Mode
                thumb_index_dist = float(
                    np.linalg.norm(lm[LM.THUMB_TIP, :2] - lm[LM.INDEX_PIP, :2])
                ) / l_hand

                # Hysteresis threshold to prevent edge jitter between DRAW and POINT
                point_thr = self.point_thumb_ratio
                if self._state == GestureState.DRAW:
                    point_thr += 0.04
                elif self._state == GestureState.POINT:
                    point_thr -= 0.04

                # L-shape: thumb extended outward into an L shape (POINT / laser)
                if thumb_index_dist > point_thr:
                    return GestureState.POINT
                else:
                    return GestureState.DRAW

        return GestureState.NONE

    # ── Finger / Hand shape helpers ───────────────────────────────────────────

    @staticmethod
    def _finger_extended(lm: np.ndarray, tip_idx: int, pip_idx: int) -> bool:
        """
        Finger is extended if tip is further from the wrist than PIP joint
        or if tip is above PIP in upright screen coordinates.
        Scale-invariant and orientation-resilient.
        """
        wrist = lm[LM.WRIST, :2]
        d_tip = float(np.linalg.norm(lm[tip_idx, :2] - wrist))
        d_pip = float(np.linalg.norm(lm[pip_idx, :2] - wrist))
        dist_extended = d_tip > (d_pip * 1.08)
        y_extended = bool(lm[tip_idx, 1] < lm[pip_idx, 1])
        return dist_extended or y_extended

    @staticmethod
    def _is_fist(lm: np.ndarray) -> bool:
        """All four fingertips are curled toward wrist/palm."""
        pairs = [
            (LM.INDEX_TIP, LM.INDEX_PIP),
            (LM.MIDDLE_TIP, LM.MIDDLE_PIP),
            (LM.RING_TIP, LM.RING_PIP),
            (LM.PINKY_TIP, LM.PINKY_PIP),
        ]
        wrist = lm[LM.WRIST, :2]
        return all(
            (lm[tip, 1] >= lm[pip, 1])
            or (float(np.linalg.norm(lm[tip, :2] - wrist)) <= float(np.linalg.norm(lm[pip, :2] - wrist)) * 1.05)
            for tip, pip in pairs
        )

    def _all_fingers_open(self, lm: np.ndarray, l_hand: float) -> bool:
        """All 5 fingers extended (open palm / swipe gesture)."""
        fingers_extended = (
            self._finger_extended(lm, LM.INDEX_TIP, LM.INDEX_PIP)
            and self._finger_extended(lm, LM.MIDDLE_TIP, LM.MIDDLE_PIP)
            and self._finger_extended(lm, LM.RING_TIP, LM.RING_PIP)
            and self._finger_extended(lm, LM.PINKY_TIP, LM.PINKY_PIP)
        )
        # Thumb also extended (CMC → TIP spread)
        thumb_spread = (
            float(np.linalg.norm(lm[LM.THUMB_TIP, :2] - lm[LM.INDEX_MCP, :2])) / l_hand
            > 0.30
        )
        return fingers_extended and thumb_spread

    def _is_thumbs_up(self, lm: np.ndarray, l_hand: float) -> bool:
        """
        Thumbs-up: thumb tip is above all other landmarks (smallest y),
        and the four fingers are all curled.
        """
        thumb_tip_y = float(lm[LM.THUMB_TIP, 1])
        # Thumb must be the highest point (smallest y in screen coords)
        other_tips_y = [
            float(lm[LM.INDEX_TIP, 1]),
            float(lm[LM.MIDDLE_TIP, 1]),
            float(lm[LM.RING_TIP, 1]),
            float(lm[LM.PINKY_TIP, 1]),
        ]
        thumb_is_highest = all(thumb_tip_y < y for y in other_tips_y)
        fingers_curled = self._is_fist(lm)  # reuse fist check for 4 fingers
        return thumb_is_highest and fingers_curled

    # ── Swipe Detection ───────────────────────────────────────────────────────

    def _detect_swipe(self, t_ms: float) -> Optional[GestureState]:
        """Measure horizontal wrist velocity over the swipe window."""
        now_cooldown_ok = (t_ms - self._last_swipe_t_ms) >= self.swipe_cooldown_ms
        if not now_cooldown_ok:
            return None

        history = list(self._wrist_history)
        if len(history) < 2:
            return None

        # Find entries within swipe_window_ms
        t_cutoff = t_ms - self.swipe_window_ms
        recent = [(x, t) for x, t in history if t >= t_cutoff]

        if len(recent) < 2:
            return None

        x_start, t_start = recent[0]
        x_end, t_end = recent[-1]
        dt_ms = t_end - t_start
        if dt_ms < 1.0:
            return None

        dx = x_end - x_start
        velocity = dx / (dt_ms / 1000.0)  # normalized units per second

        # Swipe requires significant horizontal displacement and velocity
        # (velocity in normalized screen width / s >= swipe_velocity_thr)
        # Require significant horizontal displacement (>= 12% screen width) and velocity (>= swipe_velocity_thr units/s)
        per_frame_velocity = abs(velocity)

        if abs(dx) >= 0.12 and per_frame_velocity >= self.swipe_velocity_thr:
            self._last_swipe_t_ms = t_ms
            self._wrist_history.clear()
            if dx < 0:
                return GestureState.SWIPE_LEFT
            else:
                return GestureState.SWIPE_RIGHT

        return None

    # ── State Transition ──────────────────────────────────────────────────────

    def _transition_to(self, new_state: GestureState, t_ms: float) -> None:
        """Apply state transition, call callback if state changed."""
        if new_state == self._state:
            return
        old_state = self._state
        self._state = new_state
        self._debounce_count = 0

        # Reset confirm timer when leaving CONFIRM
        if old_state == GestureState.CONFIRM:
            self._confirm_start_ms = None

        # Start confirm hold timer
        if new_state == GestureState.CONFIRM:
            self._confirm_start_ms = t_ms

        logger.debug("Gesture FSM: %s → %s", old_state.name, new_state.name)

        if self.on_state_change is not None:
            try:
                self.on_state_change(new_state, old_state)
            except Exception:
                logger.exception("on_state_change callback raised an exception")

    def is_confirm_held(self, t_ms: float) -> bool:
        """
        Returns True if the CONFIRM (thumbs-up) gesture has been held
        for at least confirm_hold_ms milliseconds continuously.
        """
        if self._state != GestureState.CONFIRM:
            return False
        if self._confirm_start_ms is None:
            return False
        return (t_ms - self._confirm_start_ms) >= self.confirm_hold_ms

    def calibrate_perspective(self, corners_normalized: np.ndarray) -> bool:
        """
        Update the perspective mapping with 4 calibration corners.

        Args:
            corners_normalized: Shape (4, 2) in normalized [0,1] space,
                                clockwise: TL, TR, BR, BL.

        Returns:
            True if calibration succeeded, False otherwise.
        """
        return self.perspective.calibrate(corners_normalized)

    def reset_perspective(self) -> None:
        """Reset the perspective mapping to default full-frame identity."""
        self.perspective.reset_calibration()


# ─────────────────────────────────────────────────────────────────────────────
# Camera & MediaPipe Worker Thread
# ─────────────────────────────────────────────────────────────────────────────

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.multi_hand_coordinator import (
        HandLandmarkData,
        HandRole,
        MultiHandCoordinator,
        MultiHandMode,
    )


class HandTrackingWorker:
    """
    Runs MediaPipe hand tracking on a background thread.

    Designed to be used with PySide6 QThread. The caller should move this
    object to a QThread and connect its signals.

    This class does NOT import PySide6 so it can be unit-tested without a
    display server. Qt-specific subclassing is done in the UI layer.

    Attributes:
        on_frame:  Callback(bgr_frame, landmarks, state, cx, cy) called per frame.
        on_error:  Callback(str) called on camera or MediaPipe errors.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        canvas_width: int = 1920,
        canvas_height: int = 1080,
        on_frame: Optional[Callable] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_diagnostics: Optional[Callable[[CameraDiagnosticsData], None]] = None,
        on_multi_hands: Optional[Callable[[list[HandLandmarkData]], None]] = None,
        lighting_mode: str = "adaptive",
        mirror: bool = True,
        multi_hand_mode: str = "draw_and_tool",
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.on_frame = on_frame
        self.on_error = on_error
        self.on_diagnostics = on_diagnostics
        self.on_multi_hands = on_multi_hands
        self.lighting_mode = lighting_mode
        self.mirror = mirror
        self.multi_hand_mode = multi_hand_mode
        self._running = False

        # Adaptive Vision Pipeline for low-light & backlight compensation
        self.vision_pipeline = AdaptiveVisionPipeline(mode=self.lighting_mode)

        # MediaPipe Hands: support both Tasks API (MediaPipe 1.x) and solutions (MediaPipe 0.10.x legacy)
        self._use_tasks_api = hasattr(mp, "tasks") and hasattr(mp.tasks, "vision") and hasattr(mp.tasks.vision, "HandLandmarker")
        self._landmarker = None
        self._mp_hands = getattr(mp, "solutions", None).hands if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands") else None
        self._hands_model = None

        # Gesture FSM (Single Hand fallback)
        self.fsm = HandGestureFSM(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )

        # Multi-Hand Coordinator (Handles dual-hand tracking & air clicks)
        from src.core.multi_hand_coordinator import MultiHandCoordinator
        self.coordinator = MultiHandCoordinator(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            mode=self.multi_hand_mode,
            draw_mode=self.fsm.draw_mode,
            pinch_threshold=self.fsm.pinch_threshold,
        )

    def set_lighting_mode(self, mode: str) -> None:
        self.lighting_mode = mode
        self.vision_pipeline.set_mode(mode)

    def set_mirror(self, mirror: bool) -> None:
        self.mirror = mirror

    def set_draw_mode(self, mode: str | DrawMode) -> None:
        self.fsm.set_draw_mode(mode)
        self.coordinator.set_draw_mode(mode)

    def set_pinch_threshold(self, threshold: float) -> None:
        self.fsm.set_pinch_threshold(threshold)
        self.coordinator.set_pinch_threshold(threshold)

    def set_multi_hand_mode(self, mode: str | MultiHandMode) -> None:
        self.multi_hand_mode = mode.value if isinstance(mode, MultiHandMode) else mode
        self.coordinator.set_mode(mode)

    def set_dominant_hand(self, dominant_hand: str) -> None:
        self.coordinator.set_dominant_hand(dominant_hand)

    def set_air_click_type(self, click_type: str) -> None:
        if hasattr(self, "coordinator"):
            self.coordinator.set_click_type(click_type)

    def calibrate_perspective(self, corners_normalized: np.ndarray) -> bool:
        """Calibrate perspective mapping across both single-hand FSM and multi-hand coordinator."""
        ok_fsm = self.fsm.calibrate_perspective(corners_normalized)
        ok_coord = self.coordinator.calibrate_perspective(corners_normalized)
        return ok_fsm and ok_coord

    def reset_perspective(self) -> None:
        """Reset perspective mapping to default identity across all engines."""
        self.fsm.reset_perspective()
        self.coordinator.reset_perspective()

    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe Hands model supporting both Tasks API and legacy solutions for up to 2 hands."""
        if self._use_tasks_api:
            from pathlib import Path
            model_path = Path(__file__).resolve().parent.parent.parent / "models" / "hand_landmarker.task"
            if model_path.exists():
                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
                    running_mode=mp.tasks.vision.RunningMode.IMAGE,
                    num_hands=2,
                    min_hand_detection_confidence=0.6,
                    min_tracking_confidence=0.6,
                )
                self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
                logger.info("MediaPipe HandLandmarker (Tasks API) initialized with %s (num_hands=2).", model_path)
                return
            else:
                logger.warning("hand_landmarker.task not found at %s; checking legacy solutions...", model_path)

        if self._mp_hands is not None:
            self._hands_model = self._mp_hands.Hands(
                model_complexity=1,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.6,
                static_image_mode=False,
            )
            logger.info("MediaPipe Hands (Solutions API) initialized (max_num_hands=2).")
        else:
            logger.error("No valid MediaPipe Hands backend available.")

    def _apply_clahe(self, bgr_frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE on the L channel (LAB color space) to reduce backlighting
        artifacts common in classroom environments.
        """
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_eq = self._clahe.apply(l_channel)
        lab_eq = cv2.merge([l_eq, a_channel, b_channel])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def start(self) -> None:
        """
        Main loop: open camera, process frames until stop() is called.
        Call this method from within a QThread.run() override.
        """
        if self.camera_index < 0:
            logger.info("Camera index is negative (%d); skipping camera capture.", self.camera_index)
            return

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            # Fallback: try without DSHOW backend
            cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            msg = f"Cannot open camera index {self.camera_index}"
            logger.error(msg)
            if self.on_error:
                self.on_error(msg)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)

        self._init_mediapipe()
        self._running = True
        logger.info("Hand tracking started on camera %d.", self.camera_index)

        fps_smoothed = float(self.fps)
        last_t = None
        frame_dropped_count = 0
        last_latency_ms = 0.0

        try:
            while self._running:
                # Dynamic backpressure & DirectShow buffer lag recovery:
                # If previous frame processing took longer than 1.5x frame budget,
                # grab (skip) stale buffered frame so in-air gestures remain real-time.
                if last_latency_ms > (1000.0 / max(1, self.fps)) * 1.5:
                    if cap.grab():
                        frame_dropped_count += 1

                ret, frame = cap.read()
                if not ret:
                    frame_dropped_count += 1
                    logger.warning("Camera read failed; retrying…")
                    time.sleep(0.01)
                    continue

                t_now = time.perf_counter()
                if last_t is not None:
                    dt = t_now - last_t
                    if dt > 0:
                        instant_fps = 1.0 / dt
                        fps_smoothed = 0.90 * fps_smoothed + 0.10 * instant_fps
                last_t = t_now
                t_ms = t_now * 1000.0

                # Flip horizontally if mirror enabled
                if self.mirror:
                    frame = cv2.flip(frame, 1)

                # Adaptive Preprocessing (Gamma, CLAHE, Denoising)
                t_proc_start = time.perf_counter()
                frame_enhanced, metrics = self.vision_pipeline.process_frame(frame)

                # Convert BGR → RGB for MediaPipe
                rgb = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False

                raw_detections: list[tuple[list, str, float]] = []

                if self._landmarker is not None:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    results = self._landmarker.detect(mp_image)
                    if results.hand_landmarks:
                        for i, hand_lm in enumerate(results.hand_landmarks):
                            hname = "Right"
                            conf = 0.85
                            if results.handedness and i < len(results.handedness) and results.handedness[i]:
                                cat = results.handedness[i][0]
                                hname = getattr(cat, "category_name", None) or getattr(cat, "display_name", "Right")
                                conf = float(getattr(cat, "score", 0.85))
                            raw_detections.append((hand_lm, hname, conf))
                elif self._hands_model is not None:
                    results = self._hands_model.process(rgb)
                    if results.multi_hand_landmarks:
                        for i, hand_lm_list in enumerate(results.multi_hand_landmarks):
                            hname = "Right"
                            conf = 0.85
                            if results.multi_handedness and i < len(results.multi_handedness):
                                cls_list = results.multi_handedness[i].classification
                                if cls_list:
                                    hname = cls_list[0].label
                                    conf = float(cls_list[0].score)
                            raw_detections.append((hand_lm_list.landmark, hname, conf))

                # Process multi-hand tracking & air clicks
                multi_hands_data = self.coordinator.process_frame_hands(raw_detections, t_ms)

                # Maintain single-hand backward compatibility for on_frame
                if multi_hands_data:
                    from src.core.multi_hand_coordinator import HandRole
                    primary_hand = next(
                        (h for h in multi_hands_data if h.role == HandRole.PRIMARY),
                        multi_hands_data[0],
                    )
                    state = primary_hand.state
                    cx = primary_hand.normalized_pos[0]
                    cy = primary_hand.normalized_pos[1]
                    landmarks_out = primary_hand.landmarks
                    confidence = primary_hand.confidence
                else:
                    state, cx, cy = self.fsm.process_landmarks([], 0.0, t_ms)
                    landmarks_out = None
                    confidence = 0.0

                latency_ms = (time.perf_counter() - t_proc_start) * 1000.0
                last_latency_ms = latency_ms

                if self.on_frame is not None:
                    self.on_frame(frame_enhanced, landmarks_out, state, cx, cy)

                if self.on_multi_hands is not None:
                    self.on_multi_hands(multi_hands_data)

                if self.on_diagnostics is not None:
                    diag = CameraDiagnosticsData(
                        fps=round(fps_smoothed, 1),
                        width=frame.shape[1],
                        height=frame.shape[0],
                        mean_brightness=round(metrics.mean_brightness, 1),
                        sharpness_score=round(metrics.sharpness_score, 1),
                        shadow_clipping_pct=round(metrics.shadow_clipping_pct, 1),
                        highlight_clipping_pct=round(metrics.highlight_clipping_pct, 1),
                        condition_name=metrics.condition.name,
                        hands_detected=len(multi_hands_data),
                        tracking_confidence=round(confidence, 2),
                        gesture_state_name=state.name,
                        draw_mode_name=self.fsm.draw_mode.name,
                        pipeline_latency_ms=round(latency_ms, 1),
                        frame_dropped_count=frame_dropped_count,
                    )
                    self.on_diagnostics(diag)

        finally:
            cap.release()
            if self._landmarker is not None:
                self._landmarker.close()
            if self._hands_model is not None:
                self._hands_model.close()
            logger.info("Hand tracking stopped. Camera released.")

    def stop(self) -> None:
        """Signal the main loop to stop."""
        self._running = False
