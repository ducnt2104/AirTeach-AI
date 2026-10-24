"""
AirTeach AI - Multi-Hand Tracking, Gesture Coordinator & Air Click Engine
==========================================================================
Manages multi-hand tracking association, independent hand state machines,
Air Tap / Pinch Click / Dwell selection detection, and multi-hand coordination
(Dual Draw, Draw + Menu, Draw + Erase, Independent tools).

Architecture:
  1. HandLandmarkData: Rich data structure for each tracked hand.
  2. MultiHandTracker: Frame-to-frame spatial & handedness association preventing
     hand identity swapping when hands cross or temporarily occlude.
  3. AirClickDetector: Detects Air Tap, Pinch Click, and Dwell Selection without mouse.
  4. MultiHandCoordinator: Coordinates dual-hand interactions, role assignments,
     and safe tracking gap recovery.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from src.core.hand_gesture_engine import (
    DrawMode,
    GestureState,
    HandGestureFSM,
    LM,
    PerspectiveMapper,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class HandRole(Enum):
    """Assigned role for a tracked hand."""
    PRIMARY = "primary"          # Main drawing / pointing hand
    SECONDARY = "secondary"      # Tool selection, menu, or eraser hand
    INDEPENDENT = "independent"  # Both hands perform identical active tool


class MultiHandMode(Enum):
    """Dual-hand operational mode."""
    SINGLE_HAND = "single_hand"       # Only track dominant hand
    DUAL_DRAW = "dual_draw"           # Both hands draw concurrent strokes
    DRAW_AND_TOOL = "draw_and_tool"   # Primary draws, Secondary opens/controls menu
    DRAW_AND_ERASE = "draw_and_erase" # Primary draws, Secondary erases
    INDEPENDENT = "independent"       # Full independent gesture FSM per hand


class AirClickType(Enum):
    """Trigger mode for air click / selection."""
    AIR_TAP = "air_tap"         # Quick forward tap of index finger
    PINCH_CLICK = "pinch_click" # Pinch thumb + index finger
    DWELL = "dwell"             # Hold cursor steady inside target
    HYBRID = "hybrid"           # Air tap OR pinch click with dwell fallback


# ─────────────────────────────────────────────────────────────────────────────
# Hand Landmark & Interaction Data Structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HandLandmarkData:
    """Rich normalized & canvas data for a single tracked hand."""
    track_id: str                      # Stable identifier: "hand_0", "hand_1"
    handedness: str = "Right"          # "Left", "Right", or "Unknown"
    confidence: float = 0.9            # Detection / tracking score [0, 1]
    landmarks: list = field(default_factory=list) # 21 MediaPipe landmark points
    hand_size: float = 0.2             # L_hand = ‖P0 - P9‖₂
    wrist_norm: tuple[float, float] = (0.5, 0.8) # (x, y) normalized [0, 1]
    index_tip_norm: tuple[float, float] = (0.5, 0.5) # (x, y) normalized [0, 1]
    thumb_tip_norm: tuple[float, float] = (0.5, 0.6) # (x, y) normalized [0, 1]
    canvas_x: float = 0.0              # Mapped canvas X coordinate [0, 1920]
    canvas_y: float = 0.0              # Mapped canvas Y coordinate [0, 1080]
    pinch_distance_norm: float = 0.5   # ‖P4 - P8‖₂ / L_hand
    is_pinched: bool = False           # True if pinch_distance_norm < threshold
    is_air_tap: bool = False           # True if air tap flick detected this frame
    is_pinch_click: bool = False       # True if pinch click triggered this frame
    dwell_progress: float = 0.0        # [0.0, 1.0] progress for dwell selection
    gesture_state: GestureState = GestureState.NONE
    role: HandRole = HandRole.PRIMARY
    timestamp_ms: float = 0.0

    @property
    def canvas_pos(self) -> tuple[float, float]:
        return self.canvas_x, self.canvas_y

    @property
    def state(self) -> GestureState:
        return self.gesture_state

    @property
    def normalized_pos(self) -> tuple[float, float]:
        return self.index_tip_norm


# ─────────────────────────────────────────────────────────────────────────────
# Air Click & Gesture Tap Detector
# ─────────────────────────────────────────────────────────────────────────────

class AirClickDetector:
    """
    Detects intentional Air Taps (quick forward index-finger flick/push)
    and Pinch Clicks with debounce, dead-zones, and dwell selection.
    """

    def __init__(
        self,
        click_type: AirClickType = AirClickType.HYBRID,
        tap_velocity_threshold: float = 0.45,
        pinch_threshold: float = 0.28,
        pinch_release_threshold: float = 0.38,
        dwell_time_ms: float = 650.0,
        dwell_radius_canvas: float = 24.0,
        debounce_ms: float = 250.0,
    ) -> None:
        self.click_type = click_type
        self.tap_velocity_threshold = tap_velocity_threshold
        self.pinch_threshold = pinch_threshold
        self.pinch_release_threshold = pinch_release_threshold
        self.dwell_time_ms = dwell_time_ms
        self.dwell_radius_canvas = dwell_radius_canvas
        self.debounce_ms = debounce_ms

        # State per hand track_id
        self._history: dict[str, deque[tuple[float, float, float, float]]] = {} # (x, y, z, t_ms)
        self._was_pinched: dict[str, bool] = {}
        self._last_click_t_ms: dict[str, float] = {}

        # Dwell tracking per hand: (anchor_x, anchor_y, start_t_ms)
        self._dwell_state: dict[str, tuple[float, float, float]] = {}

    def set_click_type(self, click_type: AirClickType | str) -> None:
        if isinstance(click_type, str):
            try:
                self.click_type = AirClickType(click_type.lower())
            except ValueError:
                self.click_type = AirClickType.HYBRID
        else:
            self.click_type = click_type

    def reset_hand(self, track_id: str) -> None:
        self._history.pop(track_id, None)
        self._was_pinched.pop(track_id, None)
        self._last_click_t_ms.pop(track_id, None)
        self._dwell_state.pop(track_id, None)

    def reset_all(self) -> None:
        """Reset tracking state across all hands."""
        self._history.clear()
        self._was_pinched.clear()
        self._last_click_t_ms.clear()
        self._dwell_state.clear()

    def process_hand(
        self,
        track_id: str,
        landmarks: list,
        canvas_x: float,
        canvas_y: float,
        hand_size: float,
        t_ms: float,
    ) -> tuple[bool, bool, float]:
        """
        Process hand motion for air click.
        Returns:
            (is_air_tap, is_pinch_click, dwell_progress)
        """
        if not landmarks or len(landmarks) < 21 or hand_size <= 1e-4:
            return False, False, 0.0

        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=15)
            self._was_pinched[track_id] = False
            self._last_click_t_ms[track_id] = -999999.0

        p8 = landmarks[LM.INDEX_TIP]
        p4 = landmarks[LM.THUMB_TIP]
        p5 = landmarks[LM.INDEX_MCP]

        # Normalized coordinates
        z_rel = getattr(p8, "z", 0.0)
        self._history[track_id].append((p8.x, p8.y, z_rel, t_ms))

        # Check debounce
        last_click = self._last_click_t_ms.get(track_id, -999999.0)
        debounced = (t_ms - last_click) >= self.debounce_ms

        # ── 1. Pinch Click Evaluation ──────────────────────────────────────────
        safe_hand_size = max(1e-4, hand_size)
        dist_pinch_norm = math.hypot(p8.x - p4.x, p8.y - p4.y) / safe_hand_size
        is_pinched_now = dist_pinch_norm < self.pinch_threshold

        is_pinch_click = False
        was_pinched = self._was_pinched.get(track_id, False)

        if is_pinched_now and not was_pinched and debounced:
            is_pinch_click = True
            self._last_click_t_ms[track_id] = t_ms

        self._was_pinched[track_id] = is_pinched_now

        # ── 2. Air Tap Evaluation (Forward flick along depth/flexion) ──────────
        is_air_tap = False
        hist = self._history[track_id]
        if debounced and len(hist) >= 4:
            # Check delta in last 120-220ms
            old_x, old_y, old_z, old_t = hist[0]
            dt = t_ms - old_t
            if 60.0 <= dt <= 260.0:
                # Forward motion (z decreasing towards camera in MediaPipe coordinates)
                dz = old_z - z_rel
                # Also check index tip flexion towards palm relative to index MCP
                dist_tip_mcp = math.hypot(p8.x - p5.x, p8.y - p5.y) / safe_hand_size
                vz = dz / (dt / 1000.0)

                # Tap criteria: forward pulse or swift downward jab with return
                if vz > self.tap_velocity_threshold and dist_tip_mcp > 0.40:
                    is_air_tap = True
                    self._last_click_t_ms[track_id] = t_ms

        # ── 3. Dwell Selection Evaluation ──────────────────────────────────────
        dwell_progress = 0.0
        if track_id not in self._dwell_state:
            self._dwell_state[track_id] = (canvas_x, canvas_y, t_ms)
        else:
            anchor_x, anchor_y, start_t = self._dwell_state[track_id]
            dist_drift = math.hypot(canvas_x - anchor_x, canvas_y - anchor_y)
            if dist_drift <= self.dwell_radius_canvas:
                elapsed = t_ms - start_t
                dwell_progress = min(1.0, elapsed / max(100.0, self.dwell_time_ms))
            else:
                # Reset dwell anchor
                self._dwell_state[track_id] = (canvas_x, canvas_y, t_ms)
                dwell_progress = 0.0

        # Filter output based on active click_type
        if self.click_type == AirClickType.AIR_TAP:
            return is_air_tap, False, 0.0
        elif self.click_type == AirClickType.PINCH_CLICK:
            return False, is_pinch_click, 0.0
        elif self.click_type == AirClickType.DWELL:
            return False, False, dwell_progress
        else:  # HYBRID
            return is_air_tap, is_pinch_click, dwell_progress


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Hand Spatial & Handedness Tracker (Prevents ID Swapping)
# ─────────────────────────────────────────────────────────────────────────────

class MultiHandTracker:
    """
    Maintains persistent hand identities across frames.
    Matches incoming detections to active tracks via Euclidean centroid
    distance and handedness probability, preventing ID swaps when hands cross.
    """

    def __init__(
        self,
        max_hands: int = 2,
        max_missing_frames: int = 4,
        max_match_distance_norm: float = 0.35,
    ) -> None:
        self.max_hands = max_hands
        self.max_missing_frames = max_missing_frames
        self.max_match_distance_norm = max_match_distance_norm

        # Active tracks: track_id -> dict
        # { 'centroid': (x, y), 'handedness': str, 'missing_frames': int, 'last_seen_t': float }
        self._tracks: dict[str, dict] = {}
        self._next_id_counter = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id_counter = 0

    def _get_centroid(self, landmarks: list) -> tuple[float, float]:
        """Compute palm center centroid from wrist and MCP joints."""
        if not landmarks or len(landmarks) < 21:
            return 0.5, 0.5
        pts = [
            (landmarks[LM.WRIST].x, landmarks[LM.WRIST].y),
            (landmarks[LM.INDEX_MCP].x, landmarks[LM.INDEX_MCP].y),
            (landmarks[LM.MIDDLE_MCP].x, landmarks[LM.MIDDLE_MCP].y),
            (landmarks[LM.PINKY_MCP].x, landmarks[LM.PINKY_MCP].y),
        ]
        return float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))

    def update(
        self,
        detected_hands: list[tuple[list, str, float]], # (landmarks, handedness, confidence)
        t_ms: float,
    ) -> list[tuple[str, list, str, float]]:
        """
        Associate detections to persistent tracks.
        Returns:
            list of (track_id, landmarks, handedness, confidence)
        """
        # Increment missing frames on all active tracks
        for t_info in self._tracks.values():
            t_info["missing_frames"] += 1

        # Strict input validation: filter out None or malformed landmark lists (< 21 points)
        detected_hands = [
            d for d in (detected_hands or [])
            if d is not None and len(d) >= 3 and d[0] is not None and len(d[0]) >= 21
        ]

        matched_tracks: list[tuple[str, list, str, float]] = []
        unmatched_detections = list(range(len(detected_hands)))

        # Distance matching between active tracks and detections
        track_ids = list(self._tracks.keys())
        if track_ids and detected_hands:
            cost_matrix = []
            for tid in track_ids:
                row = []
                t_centroid = self._tracks[tid]["centroid"]
                t_handedness = self._tracks[tid]["handedness"]
                for lm, hness, _ in detected_hands:
                    d_centroid = self._get_centroid(lm)
                    dist = math.hypot(t_centroid[0] - d_centroid[0], t_centroid[1] - d_centroid[1])
                    # Penalize handedness mismatch if both known
                    if t_handedness in ("Left", "Right") and hness in ("Left", "Right"):
                        if t_handedness != hness:
                            dist += 0.25
                    row.append(dist)
                cost_matrix.append(row)

            # Greedy minimum distance assignment
            assigned_det = set()
            assigned_track = set()
            while len(assigned_track) < len(track_ids) and len(assigned_det) < len(detected_hands):
                min_val = 999.0
                min_i, min_j = -1, -1
                for i, row in enumerate(cost_matrix):
                    if i in assigned_track:
                        continue
                    for j, val in enumerate(row):
                        if j in assigned_det:
                            continue
                        if val < min_val:
                            min_val = val
                            min_i, min_j = i, j

                if min_val <= self.max_match_distance_norm and min_i >= 0 and min_j >= 0:
                    tid = track_ids[min_i]
                    lm, hness, conf = detected_hands[min_j]
                    self._tracks[tid]["centroid"] = self._get_centroid(lm)
                    if hness in ("Left", "Right"):
                        self._tracks[tid]["handedness"] = hness
                    self._tracks[tid]["missing_frames"] = 0
                    self._tracks[tid]["last_seen_t"] = t_ms
                    matched_tracks.append((tid, lm, self._tracks[tid]["handedness"], conf))
                    assigned_track.add(min_i)
                    assigned_det.add(min_j)
                else:
                    break

            unmatched_detections = [j for j in range(len(detected_hands)) if j not in assigned_det]

        # Allocate new tracks for remaining detections if capacity permits
        for det_idx in unmatched_detections:
            if len(self._tracks) < self.max_hands:
                lm, hness, conf = detected_hands[det_idx]
                new_id = f"hand_{self._next_id_counter}"
                self._next_id_counter += 1
                self._tracks[new_id] = {
                    "centroid": self._get_centroid(lm),
                    "handedness": hness,
                    "missing_frames": 0,
                    "last_seen_t": t_ms,
                }
                matched_tracks.append((new_id, lm, hness, conf))

        # Evict tracks that exceeded max_missing_frames
        dead_ids = [tid for tid, info in self._tracks.items() if info["missing_frames"] > self.max_missing_frames]
        for tid in dead_ids:
            self._tracks.pop(tid, None)

        return matched_tracks


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Hand Coordinator & Interaction Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

class MultiHandCoordinator:
    """
    Central coordinator orchestrating two hands simultaneously.
    Manages:
      - Independent gesture FSMs for Hand A & Hand B.
      - Role assignment: Primary (Right/Dominant) vs Secondary (Left).
      - MultiHandMode: Dual Draw, Draw + Tool Menu, Draw + Erase, Independent.
      - Air Tap / Pinch Click / Dwell detection per hand.
      - Safe tracking gap handling to prevent cross-screen accidental ink bridging.
    """

    def __init__(
        self,
        canvas_width: int = 1920,
        canvas_height: int = 1080,
        mode: MultiHandMode = MultiHandMode.DRAW_AND_TOOL,
        dominant_hand: str = "Right", # "Right" or "Left"
        click_type: AirClickType = AirClickType.HYBRID,
        draw_mode: DrawMode = DrawMode.INDEX_UP,
        pinch_threshold: float = 0.28,
        eraser_radius: float = 32.0,
    ) -> None:
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.mode = mode
        self.dominant_hand = dominant_hand
        self.draw_mode = draw_mode
        self.pinch_threshold = pinch_threshold
        self.eraser_radius = eraser_radius

        self.tracker = MultiHandTracker(max_hands=2)
        self.click_detector = AirClickDetector(click_type=click_type, pinch_threshold=pinch_threshold)
        self.perspective = PerspectiveMapper(canvas_width, canvas_height)

        # Separate FSM instance per track ID
        self._fsms: dict[str, HandGestureFSM] = {}

    def set_mode(self, mode: MultiHandMode | str) -> None:
        if isinstance(mode, str):
            try:
                self.mode = MultiHandMode(mode.lower())
            except ValueError:
                self.mode = MultiHandMode.DRAW_AND_TOOL
        else:
            self.mode = mode
        logger.info("MultiHandCoordinator mode switched to: %s", self.mode.name)

    def set_dominant_hand(self, dominant_hand: str) -> None:
        self.dominant_hand = "Left" if dominant_hand.lower().startswith("l") else "Right"

    def set_draw_mode(self, mode: DrawMode | str) -> None:
        if isinstance(mode, str):
            self.draw_mode = DrawMode.PRECISION_PINCH if "pinch" in mode.lower() else DrawMode.INDEX_UP
        else:
            self.draw_mode = mode
        for fsm in self._fsms.values():
            fsm.set_draw_mode(self.draw_mode)

    def set_pinch_threshold(self, threshold: float) -> None:
        self.pinch_threshold = max(0.12, min(0.55, float(threshold)))
        self.click_detector.pinch_threshold = self.pinch_threshold
        for fsm in self._fsms.values():
            fsm.set_pinch_threshold(self.pinch_threshold)

    def set_click_type(self, click_type: AirClickType | str) -> None:
        self.click_detector.set_click_type(click_type)

    def calibrate_perspective(self, corners_normalized: np.ndarray) -> bool:
        """Update perspective transform across coordinator and all active hand FSMs."""
        ok = self.perspective.calibrate(corners_normalized)
        if ok:
            for fsm in self._fsms.values():
                fsm.calibrate_perspective(corners_normalized)
        return ok

    def reset_perspective(self) -> None:
        """Reset perspective transform to identity across coordinator and all active hand FSMs."""
        self.perspective.reset_calibration()
        for fsm in self._fsms.values():
            fsm.reset_perspective()

    def reset(self) -> None:
        """Full reset of tracking, FSMs, and click state."""
        self.tracker.reset()
        self.click_detector.reset_all()
        self._fsms.clear()

    def _get_or_create_fsm(self, track_id: str) -> HandGestureFSM:
        if track_id not in self._fsms:
            fsm = HandGestureFSM(
                canvas_width=self.canvas_width,
                canvas_height=self.canvas_height,
            )
            fsm.set_draw_mode(self.draw_mode)
            fsm.set_pinch_threshold(self.pinch_threshold)
            fsm.calibrate_perspective(self.perspective._src_pts)
            self._fsms[track_id] = fsm
        return self._fsms[track_id]

    def process_frame_hands(
        self,
        raw_detections: list[tuple[list, str, float]], # (landmarks, handedness, confidence)
        t_ms: float,
    ) -> list[HandLandmarkData]:
        """
        Process incoming camera frame detections through multi-hand pipeline.
        Returns:
            list of HandLandmarkData for all active hands (up to 2).
        """
        # Step 1: Persistent Identity Association
        associated = self.tracker.update(raw_detections, t_ms)

        # Evict dead tracks from FSMs and click detector
        active_track_ids = set(self.tracker._tracks.keys())
        dead_fsm_ids = [tid for tid in self._fsms if tid not in active_track_ids]
        for dead_tid in dead_fsm_ids:
            del self._fsms[dead_tid]
            self.click_detector.reset_hand(dead_tid)

        if not associated:
            # Clean up all active FSM gap counters
            for fsm in self._fsms.values():
                fsm.process_landmarks([], 0.0, t_ms)
            return []

        # Determine roles based on dominant hand configuration
        hands_data: list[HandLandmarkData] = []
        for track_id, landmarks, handedness, conf in associated:
            if landmarks is None or len(landmarks) < 21:
                continue
            fsm = self._get_or_create_fsm(track_id)
            state, mapped_x, mapped_y = fsm.process_landmarks(landmarks, conf, t_ms)

            # mapped_x, mapped_y are ALREADY perspective-mapped canvas coordinates from FSM
            if mapped_x >= 0.0 and mapped_y >= 0.0:
                canvas_x = max(0.0, min(float(self.canvas_width), mapped_x))
                canvas_y = max(0.0, min(float(self.canvas_height), mapped_y))
            else:
                canvas_x = -1.0
                canvas_y = -1.0

            p0 = landmarks[LM.WRIST]
            p9 = landmarks[LM.MIDDLE_MCP]
            p5 = landmarks[LM.INDEX_MCP]
            p17 = landmarks[LM.PINKY_MCP]
            palm_h = math.hypot(p0.x - p9.x, p0.y - p9.y)
            palm_w = math.hypot(p5.x - p17.x, p5.y - p17.y)
            z0 = getattr(p0, 'z', 0.0) or 0.0
            z9 = getattr(p9, 'z', 0.0) or 0.0
            dz = z0 - z9
            palm_h_3d = math.sqrt(palm_h * palm_h + dz * dz)
            hand_size = max(1e-4, max(palm_h_3d, palm_w * 1.15))
            p8 = landmarks[LM.INDEX_TIP]
            p4 = landmarks[LM.THUMB_TIP]

            # Pinch metrics
            pinch_dist = math.hypot(p8.x - p4.x, p8.y - p4.y) / max(1e-4, hand_size)
            is_pinched = pinch_dist < self.pinch_threshold

            # Air Click & Tap
            is_tap, is_pinch_clk, dwell = self.click_detector.process_hand(
                track_id=track_id,
                landmarks=landmarks,
                canvas_x=canvas_x,
                canvas_y=canvas_y,
                hand_size=hand_size,
                t_ms=t_ms,
            )

            # Assign Role: Primary vs Secondary
            role = HandRole.PRIMARY
            if self.mode in (MultiHandMode.DRAW_AND_TOOL, MultiHandMode.DRAW_AND_ERASE):
                # If handedness is known, check against dominant hand
                if handedness in ("Left", "Right"):
                    if handedness == self.dominant_hand:
                        role = HandRole.PRIMARY
                    else:
                        role = HandRole.SECONDARY
                else:
                    # Fallback by track order: first is primary, second is secondary
                    role = HandRole.PRIMARY if len(hands_data) == 0 else HandRole.SECONDARY
            elif self.mode in (MultiHandMode.DUAL_DRAW, MultiHandMode.INDEPENDENT):
                role = HandRole.INDEPENDENT

            h_data = HandLandmarkData(
                track_id=track_id,
                handedness=handedness,
                confidence=conf,
                landmarks=landmarks,
                hand_size=hand_size,
                wrist_norm=(p0.x, p0.y),
                index_tip_norm=(p8.x, p8.y),
                thumb_tip_norm=(p4.x, p4.y),
                canvas_x=canvas_x,
                canvas_y=canvas_y,
                pinch_distance_norm=pinch_dist,
                is_pinched=is_pinched,
                is_air_tap=is_tap,
                is_pinch_click=is_pinch_clk,
                dwell_progress=dwell,
                gesture_state=state,
                role=role,
                timestamp_ms=t_ms,
            )
            hands_data.append(h_data)

        return hands_data
