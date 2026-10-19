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

logger = logging.getLogger(__name__)


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

    def calibrate(self, src_corners_normalized: np.ndarray) -> None:
        """
        Update the perspective transform with new 4-corner calibration.

        Args:
            src_corners_normalized: Shape (4, 2) array of [x, y] corners
                in normalized [0, 1] space (clockwise: TL, TR, BR, BL).
        """
        self._src_pts = np.asarray(src_corners_normalized, dtype=np.float32)
        self._M = cv2.getPerspectiveTransform(self._src_pts, self._dst_pts)
        logger.info("Perspective calibration updated. Src corners: %s", self._src_pts.tolist())

    def map(self, nx: float, ny: float) -> tuple[float, float]:
        """
        Map a normalized hand position (nx, ny) ∈ [0,1]² to canvas pixel (cx, cy).

        Args:
            nx, ny: Normalized coordinates from MediaPipe (0–1 in frame space).

        Returns:
            (canvas_x, canvas_y) in pixels.
        """
        pt = np.array([[[nx, ny]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pt, self._M)
        return float(mapped[0][0][0]), float(mapped[0][0][1])


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
        swipe_velocity_thr: float = 0.35,
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

        self._state: GestureState = GestureState.NONE
        self._pending_state: GestureState = GestureState.NONE
        self._debounce_count: int = 0

        # Wrist position history for swipe detection: deque of (x_norm, t_ms)
        self._wrist_history: deque[tuple[float, float]] = deque(maxlen=30)
        self._last_swipe_t_ms: float = 0.0

        # Thumbs-up hold timer
        self._confirm_start_ms: Optional[float] = None

        # Perspective transform
        self.perspective = PerspectiveMapper(canvas_width, canvas_height)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def state(self) -> GestureState:
        return self._state

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
        if handedness_score < self.min_confidence:
            self._transition_to(GestureState.NONE, t_ms)
            return GestureState.NONE, -1.0, -1.0

        # Extract 2D+Z landmark array, shape (21, 3)
        lm = np.array(
            [[l.x, l.y, l.z] for l in landmarks],
            dtype=np.float64,
        )

        # Scale-invariant hand size: ‖P0 − P9‖₂ in normalized space
        l_hand = float(np.linalg.norm(lm[LM.WRIST, :2] - lm[LM.MIDDLE_MCP, :2]))
        if l_hand < 1e-6:
            l_hand = 1e-6  # guard against division by zero

        # Detect candidate gesture
        candidate = self._classify(lm, l_hand, t_ms)

        # Debounce: require debounce_frames consecutive frames of same candidate
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

        # ── Confirm 👍: thumb extended upward, all other fingers curled ────
        if self._is_thumbs_up(lm, l_hand):
            return GestureState.CONFIRM

        # ── Fist / PAUSE: all 4 fingertips below their PIP joints ──────────
        if self._is_fist(lm):
            return GestureState.PAUSE

        # ── SELECT: index + middle extended, ring + pinky curled ───────────
        if (
            self._finger_extended(lm, LM.INDEX_TIP, LM.INDEX_PIP)
            and self._finger_extended(lm, LM.MIDDLE_TIP, LM.MIDDLE_PIP)
            and not self._finger_extended(lm, LM.RING_TIP, LM.RING_PIP)
            and not self._finger_extended(lm, LM.PINKY_TIP, LM.PINKY_PIP)
        ):
            return GestureState.SELECT

        # ── DRAW or POINT: index extended ──────────────────────────────────
        if self._finger_extended(lm, LM.INDEX_TIP, LM.INDEX_PIP):
            thumb_index_dist = float(
                np.linalg.norm(lm[LM.THUMB_TIP, :2] - lm[LM.INDEX_PIP, :2])
            ) / l_hand

            if thumb_index_dist <= self.draw_thumb_ratio:
                return GestureState.DRAW
            elif thumb_index_dist > self.point_thumb_ratio:
                return GestureState.POINT

        return GestureState.NONE

    # ── Finger / Hand shape helpers ───────────────────────────────────────────

    @staticmethod
    def _finger_extended(lm: np.ndarray, tip_idx: int, pip_idx: int) -> bool:
        """Finger is extended if tip y-coordinate < PIP y-coordinate (screen coords)."""
        return bool(lm[tip_idx, 1] < lm[pip_idx, 1])

    @staticmethod
    def _is_fist(lm: np.ndarray) -> bool:
        """All four fingertips are below (higher y) their respective PIP joints."""
        pairs = [
            (LM.INDEX_TIP, LM.INDEX_PIP),
            (LM.MIDDLE_TIP, LM.MIDDLE_PIP),
            (LM.RING_TIP, LM.RING_PIP),
            (LM.PINKY_TIP, LM.PINKY_PIP),
        ]
        return all(lm[tip, 1] > lm[pip, 1] for tip, pip in pairs)

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

        # Convert to per-frame equivalent for threshold comparison
        # (threshold is in normalized units / frame ≈ 30 fps assumed)
        per_frame_velocity = abs(velocity) / 30.0

        if per_frame_velocity > self.swipe_velocity_thr:
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

    def calibrate_perspective(self, corners_normalized: np.ndarray) -> None:
        """
        Update the perspective mapping with 4 calibration corners.

        Args:
            corners_normalized: Shape (4, 2) in normalized [0,1] space,
                                clockwise: TL, TR, BR, BL.
        """
        self.perspective.calibrate(corners_normalized)


# ─────────────────────────────────────────────────────────────────────────────
# Camera & MediaPipe Worker Thread
# ─────────────────────────────────────────────────────────────────────────────

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
    ) -> None:
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.on_frame = on_frame
        self.on_error = on_error
        self._running = False

        # MediaPipe Hands
        self._mp_hands = mp.solutions.hands
        self._hands_model: Optional[mp.solutions.hands.Hands] = None

        # CLAHE for anti-backlight
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Gesture FSM
        self.fsm = HandGestureFSM(
            canvas_width=canvas_width,
            canvas_height=canvas_height,
        )

    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe Hands model."""
        self._hands_model = self._mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            static_image_mode=False,
        )
        logger.info("MediaPipe Hands initialized.")

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

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera read failed; retrying…")
                    time.sleep(0.01)
                    continue

                t_ms = time.perf_counter() * 1000.0

                # Flip horizontally (mirror mode, more intuitive for users)
                frame = cv2.flip(frame, 1)

                # CLAHE backlight correction
                frame_corrected = self._apply_clahe(frame)

                # Convert BGR → RGB for MediaPipe
                rgb = cv2.cvtColor(frame_corrected, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = self._hands_model.process(rgb)
                rgb.flags.writeable = True

                state = GestureState.NONE
                cx, cy = -1.0, -1.0
                landmarks_out = None

                if results.multi_hand_landmarks and results.multi_handedness:
                    hand_lm = results.multi_hand_landmarks[0]
                    hand_info = results.multi_handedness[0]
                    confidence = hand_info.classification[0].score

                    state, cx, cy = self.fsm.process_landmarks(
                        hand_lm.landmark,
                        confidence,
                        t_ms,
                    )
                    landmarks_out = hand_lm.landmark
                else:
                    self.fsm._transition_to(GestureState.NONE, t_ms)

                if self.on_frame is not None:
                    self.on_frame(frame_corrected, landmarks_out, state, cx, cy)

        finally:
            cap.release()
            if self._hands_model is not None:
                self._hands_model.close()
            logger.info("Hand tracking stopped. Camera released.")

    def stop(self) -> None:
        """Signal the main loop to stop."""
        self._running = False
