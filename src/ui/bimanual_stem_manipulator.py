"""
AirTeach AI — Bimanual STEM Transformation & Interactive Manipulator Engine.

Enables natural, touchless, dual-hand spatial manipulation (pinch-to-scale,
in-air rotation, translation, and cardinal angle snapping) of interactive STEM
and measurement objects on the classroom whiteboard scene.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable

import numpy as np
from PySide6.QtCore import QObject, QPointF, Signal
from PySide6.QtWidgets import QGraphicsItem

logger = logging.getLogger(__name__)


class BimanualTransformMode(str, Enum):
    """Degrees of freedom allowed during bimanual manipulation."""
    FULL = "full"                      # Scale + Rotate + Translate
    SCALE_AND_ROTATE = "scale_and_rotate"  # Scale + Rotate (fixed center)
    SCALE_ONLY = "scale_only"          # Scale only
    ROTATE_ONLY = "rotate_only"        # Rotate only
    TRANSLATE_ONLY = "translate_only"  # Translation only


class BimanualTransformState(str, Enum):
    """Lifecycle state of bimanual gesture recognition."""
    IDLE = "idle"
    TRACKING = "tracking"
    MANIPULATING = "manipulating"


@dataclass
class BimanualTransformTelemetry:
    """Real-time metrics produced by bimanual manipulation."""
    scale: float
    rotation_deg: float
    delta_rotation_deg: float
    center_x: float
    center_y: float
    hand_distance_px: float
    target_item_name: str
    is_active: bool


class BimanualSTEMManipulator(QObject):
    """
    Coordinates simultaneous two-hand in-air gestures to transform
    interactive STEM objects (scaling, rotating, translating).
    """

    manipulation_started = Signal(object)                       # (target_item: QGraphicsItem)
    transformed          = Signal(object, float, float, float, float)  # (item, scale, rot_deg, cx, cy)
    manipulation_finished = Signal(object)                      # (target_item: QGraphicsItem)

    CARDINAL_ANGLES_DEG = (0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0, 210.0, 225.0, 240.0, 270.0, 300.0, 315.0, 330.0, 360.0)

    def __init__(
        self,
        mode: BimanualTransformMode = BimanualTransformMode.FULL,
        cardinal_snapping: bool = True,
        cardinal_tolerance_deg: float = 3.5,
        min_distance_threshold: float = 20.0,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._mode = mode
        self._cardinal_snapping = cardinal_snapping
        self._cardinal_tolerance_deg = max(0.5, cardinal_tolerance_deg)
        self._min_distance_threshold = max(5.0, min_distance_threshold)

        self._state: BimanualTransformState = BimanualTransformState.IDLE
        self._target_item: Optional[QGraphicsItem] = None

        # Anchor metrics recorded at start of manipulation
        self._initial_distance: float = 1.0
        self._initial_angle_rad: float = 0.0
        self._initial_midpoint: tuple[float, float] = (0.0, 0.0)

        # Target item baseline transform at start of manipulation
        self._initial_item_scale: float = 1.0
        self._initial_item_rotation: float = 0.0
        self._initial_item_pos: QPointF = QPointF(0.0, 0.0)

        # Current telemetry
        self._last_telemetry: Optional[BimanualTransformTelemetry] = None

    @property
    def state(self) -> BimanualTransformState:
        return self._state

    @property
    def mode(self) -> BimanualTransformMode:
        return self._mode

    def set_mode(self, mode: BimanualTransformMode | str) -> None:
        """Configure transform degrees of freedom."""
        if isinstance(mode, str):
            mode = BimanualTransformMode(mode)
        self._mode = mode

    def set_cardinal_snapping(self, enabled: bool, tolerance_deg: float = 3.5) -> None:
        """Enable or disable locking to cardinal angles."""
        self._cardinal_snapping = enabled
        self._cardinal_tolerance_deg = max(0.5, tolerance_deg)

    def set_target_item(self, item: Optional[QGraphicsItem]) -> None:
        """Explicitly designate the target item for bimanual manipulation."""
        if self._state == BimanualTransformState.MANIPULATING and self._target_item != item:
            self.end_manipulation()
        self._target_item = item

    def get_target_item(self) -> Optional[QGraphicsItem]:
        """Return the active target item."""
        return self._target_item

    def is_manipulating(self) -> bool:
        """Return True if an object is currently being transformed."""
        return self._state == BimanualTransformState.MANIPULATING

    def snap_angle(self, angle_deg: float) -> tuple[float, bool]:
        """
        Snap angle to nearest cardinal angle if within tolerance.

        Returns:
            (snapped_angle_deg, was_snapped)
        """
        norm_deg = angle_deg % 360.0
        if not self._cardinal_snapping:
            return norm_deg, False

        for target in self.CARDINAL_ANGLES_DEG:
            diff = abs(norm_deg - target)
            diff = min(diff, 360.0 - diff)
            if diff <= self._cardinal_tolerance_deg:
                snapped = target % 360.0
                return snapped, True

        return norm_deg, False

    def process_hands(
        self,
        hand1_pos: tuple[float, float],
        hand2_pos: tuple[float, float],
        hand1_active: bool,
        hand2_active: bool,
        candidate_items: Optional[list[QGraphicsItem]] = None,
    ) -> Optional[BimanualTransformTelemetry]:
        """
        Evaluate dual-hand geometry and apply real-time transforms.

        Args:
            hand1_pos: (x, y) scene coordinates of primary hand.
            hand2_pos: (x, y) scene coordinates of secondary hand.
            hand1_active: True if primary hand is pinching or holding.
            hand2_active: True if secondary hand is pinching or holding.
            candidate_items: Optional list of items at midpoint for auto-lock.

        Returns:
            BimanualTransformTelemetry if manipulating, otherwise None.
        """
        both_active = hand1_active and hand2_active

        # If dual hands are not active together, end any existing manipulation
        if not both_active:
            if self._state == BimanualTransformState.MANIPULATING:
                self.end_manipulation()
            self._state = BimanualTransformState.IDLE
            return None

        p1_x, p1_y = hand1_pos
        p2_x, p2_y = hand2_pos

        dx = p2_x - p1_x
        dy = p2_y - p1_y
        current_distance = float(math.hypot(dx, dy))

        # Reject dual hands that are touching or overlapping to avoid division by zero
        if current_distance < self._min_distance_threshold:
            return None

        current_midpoint = ((p1_x + p2_x) * 0.5, (p1_y + p2_y) * 0.5)
        current_angle_rad = float(math.atan2(dy, dx))

        # ── Start Manipulation ────────────────────────────────────────────────
        if self._state != BimanualTransformState.MANIPULATING:
            # Auto-lock target item if none explicitly set
            if self._target_item is None and candidate_items:
                # Find first interactive item in candidates
                for cand in candidate_items:
                    if cand.isVisible() and cand.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsSelectable or hasattr(cand, "rotate_by") or hasattr(cand, "air_click_at"):
                        self._target_item = cand
                        break
                if self._target_item is None and candidate_items:
                    self._target_item = candidate_items[0]

            if self._target_item is None:
                self._state = BimanualTransformState.TRACKING
                return None

            self._state = BimanualTransformState.MANIPULATING
            self._initial_distance = max(1.0, current_distance)
            self._initial_angle_rad = current_angle_rad
            self._initial_midpoint = current_midpoint

            # Snapshot baseline transforms of the target item
            self._initial_item_scale = float(self._target_item.scale())
            self._initial_item_rotation = float(self._target_item.rotation())
            self._initial_item_pos = self._target_item.pos()

            # Ensure transform origin is centered on the item if possible
            br = self._target_item.boundingRect()
            self._target_item.setTransformOriginPoint(br.center())

            logger.info("Bimanual manipulation engaged on %s", type(self._target_item).__name__)
            self.manipulation_started.emit(self._target_item)

        # ── Apply Transformations ─────────────────────────────────────────────
        item = self._target_item
        if item is None:
            self.end_manipulation()
            return None

        # 1. Scale computation
        new_scale = self._initial_item_scale
        if self._mode in (BimanualTransformMode.FULL, BimanualTransformMode.SCALE_AND_ROTATE, BimanualTransformMode.SCALE_ONLY):
            scale_ratio = current_distance / self._initial_distance
            # Clamp scale safely to prevent extreme blow-up or microscopic shrinkage
            new_scale = max(0.15, min(6.0, self._initial_item_scale * scale_ratio))
            if hasattr(item, "set_scale"):
                try:
                    item.set_scale(new_scale)
                except Exception:
                    item.setScale(new_scale)
            else:
                item.setScale(new_scale)

        # 2. Rotation computation
        new_rot_deg = self._initial_item_rotation
        delta_deg = 0.0
        if self._mode in (BimanualTransformMode.FULL, BimanualTransformMode.SCALE_AND_ROTATE, BimanualTransformMode.ROTATE_ONLY):
            delta_rad = current_angle_rad - self._initial_angle_rad
            # Wrap delta angle to [-pi, pi]
            delta_rad = math.atan2(math.sin(delta_rad), math.cos(delta_rad))
            delta_deg = math.degrees(delta_rad)
            raw_target_deg = self._initial_item_rotation + delta_deg
            snapped_rot, _ = self.snap_angle(raw_target_deg)
            new_rot_deg = snapped_rot

            # Update item rotation
            if hasattr(item, "set_angle_deg"):
                try:
                    item.set_angle_deg(new_rot_deg)
                except Exception:
                    item.setRotation(new_rot_deg)
            else:
                item.setRotation(new_rot_deg)

        # 3. Translation computation
        center_x, center_y = current_midpoint
        if self._mode in (BimanualTransformMode.FULL, BimanualTransformMode.TRANSLATE_ONLY):
            delta_mid_x = current_midpoint[0] - self._initial_midpoint[0]
            delta_mid_y = current_midpoint[1] - self._initial_midpoint[1]
            new_pos = QPointF(self._initial_item_pos.x() + delta_mid_x, self._initial_item_pos.y() + delta_mid_y)
            item.setPos(new_pos)

        telemetry = BimanualTransformTelemetry(
            scale=round(new_scale, 3),
            rotation_deg=round(new_rot_deg, 1),
            delta_rotation_deg=round(delta_deg, 1),
            center_x=round(center_x, 1),
            center_y=round(center_y, 1),
            hand_distance_px=round(current_distance, 1),
            target_item_name=type(item).__name__,
            is_active=True,
        )
        self._last_telemetry = telemetry
        self.transformed.emit(item, new_scale, new_rot_deg, center_x, center_y)
        return telemetry

    def end_manipulation(self) -> None:
        """Conclude active bimanual manipulation and emit finished signal."""
        if self._state == BimanualTransformState.MANIPULATING and self._target_item is not None:
            logger.info("Bimanual manipulation concluded on %s", type(self._target_item).__name__)
            finished_item = self._target_item
            self._state = BimanualTransformState.IDLE
            self._target_item = None
            self._last_telemetry = None
            self.manipulation_finished.emit(finished_item)
        else:
            self._state = BimanualTransformState.IDLE
            self._target_item = None
            self._last_telemetry = None
