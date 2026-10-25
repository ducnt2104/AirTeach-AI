"""
Unit and integration tests for Bimanual STEM Transformation & Manipulator Engine.
"""

import math
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QGraphicsRectItem, QGraphicsScene

from src.ui.bimanual_stem_manipulator import (
    BimanualSTEMManipulator,
    BimanualTransformMode,
    BimanualTransformState,
)
from src.ui.interactive_trigonometric_circle import InteractiveTrigonometricUnitCircleItem
from src.ui.measurement_tools import VirtualRulerItem


class TestBimanualSTEMManipulator:
    """Verify dual-hand transformation algorithms, modes, and lifecycle."""

    def test_initialization_defaults_and_modes(self, qapp):
        manip = BimanualSTEMManipulator()
        assert manip.state == BimanualTransformState.IDLE
        assert manip.mode == BimanualTransformMode.FULL
        assert manip.get_target_item() is None
        assert manip.is_manipulating() is False

        manip.set_mode(BimanualTransformMode.SCALE_ONLY)
        assert manip.mode == BimanualTransformMode.SCALE_ONLY

        manip.set_mode("rotate_only")
        assert manip.mode == BimanualTransformMode.ROTATE_ONLY

    def test_cardinal_angle_snapping(self, qapp):
        manip = BimanualSTEMManipulator(cardinal_snapping=True, cardinal_tolerance_deg=3.5)

        # 44.0° is within 3.5° of 45.0°
        snapped, was_snapped = manip.snap_angle(44.0)
        assert was_snapped is True
        assert snapped == 45.0

        # 91.5° is within 3.5° of 90.0°
        snapped, was_snapped = manip.snap_angle(91.5)
        assert was_snapped is True
        assert snapped == 90.0

        # 70.0° is far from any cardinal angle
        snapped, was_snapped = manip.snap_angle(70.0)
        assert was_snapped is False
        assert snapped == 70.0

        # Snapping disabled
        manip.set_cardinal_snapping(False)
        snapped, was_snapped = manip.snap_angle(44.0)
        assert was_snapped is False
        assert snapped == 44.0

    def test_bimanual_manipulation_lifecycle_and_transforms(self, qapp):
        scene = QGraphicsScene()
        item = QGraphicsRectItem(0, 0, 100, 100)
        item.setPos(QPointF(200, 200))
        scene.addItem(item)

        manip = BimanualSTEMManipulator(mode=BimanualTransformMode.FULL)
        manip.set_target_item(item)

        started_items = []
        finished_items = []
        transform_events = []

        manip.manipulation_started.connect(lambda itm: started_items.append(itm))
        manip.manipulation_finished.connect(lambda itm: finished_items.append(itm))
        manip.transformed.connect(lambda itm, s, r, cx, cy: transform_events.append((itm, s, r, cx, cy)))

        # 1. Start manipulation with two hands at (100, 250) and (300, 250) -> distance = 200px, horizontal
        t1 = manip.process_hands(
            hand1_pos=(100.0, 250.0),
            hand2_pos=(300.0, 250.0),
            hand1_active=True,
            hand2_active=True,
        )
        assert manip.is_manipulating() is True
        assert len(started_items) == 1
        assert started_items[0] is item
        assert t1.scale == 1.0
        assert t1.rotation_deg == 0.0

        # 2. Scale up: Hands move apart from 200px to 300px -> scale ratio = 1.5
        t2 = manip.process_hands(
            hand1_pos=(50.0, 250.0),
            hand2_pos=(350.0, 250.0),
            hand1_active=True,
            hand2_active=True,
        )
        assert t2.scale == 1.5
        assert pytest.approx(item.scale(), 0.05) == 1.5

        # 3. Rotate: Hands rotate by 90 degrees (vertical: p1=(200, 100), p2=(200, 400))
        t3 = manip.process_hands(
            hand1_pos=(200.0, 100.0),
            hand2_pos=(200.0, 400.0),
            hand1_active=True,
            hand2_active=True,
        )
        assert t3.rotation_deg == 90.0
        assert pytest.approx(item.rotation(), 0.05) == 90.0

        # 4. Release: Hand 1 releases
        t4 = manip.process_hands(
            hand1_pos=(200.0, 100.0),
            hand2_pos=(200.0, 400.0),
            hand1_active=False,
            hand2_active=True,
        )
        assert t4 is None
        assert manip.is_manipulating() is False
        assert len(finished_items) == 1
        assert finished_items[0] is item

    def test_auto_lock_candidate_items(self, qapp):
        scene = QGraphicsScene()
        rect1 = QGraphicsRectItem(0, 0, 50, 50)
        rect2 = QGraphicsRectItem(100, 100, 80, 80)
        scene.addItem(rect1)
        scene.addItem(rect2)

        manip = BimanualSTEMManipulator()
        assert manip.get_target_item() is None

        # Provide candidate_items in process_hands
        t = manip.process_hands(
            hand1_pos=(80.0, 120.0),
            hand2_pos=(180.0, 120.0),
            hand1_active=True,
            hand2_active=True,
            candidate_items=[rect2, rect1],
        )
        assert manip.get_target_item() is rect2
        assert manip.is_manipulating() is True

        manip.end_manipulation()
        assert manip.get_target_item() is None

    def test_integration_with_interactive_trigonometric_circle(self, qapp):
        circle = InteractiveTrigonometricUnitCircleItem(center=(400.0, 300.0), radius=100.0, angle_deg=0.0)

        manip = BimanualSTEMManipulator(mode=BimanualTransformMode.FULL)
        manip.set_target_item(circle)

        # Baseline start: hands horizontal
        manip.process_hands(
            hand1_pos=(300.0, 300.0),
            hand2_pos=(500.0, 300.0),
            hand1_active=True,
            hand2_active=True,
        )
        assert circle.angle_deg == 0.0

        # Rotate dual hands to ~45 degrees
        rad45 = math.radians(45.0)
        p1 = (400.0 - 100.0 * math.cos(rad45), 300.0 - 100.0 * math.sin(rad45))
        p2 = (400.0 + 100.0 * math.cos(rad45), 300.0 + 100.0 * math.sin(rad45))

        t = manip.process_hands(
            hand1_pos=p1,
            hand2_pos=p2,
            hand1_active=True,
            hand2_active=True,
        )
        assert t is not None
        assert t.rotation_deg == 45.0
        assert pytest.approx(circle.angle_deg, 0.1) == 45.0

        manip.end_manipulation()
        assert manip.is_manipulating() is False

    def test_integration_with_virtual_ruler(self, qapp):
        ruler = VirtualRulerItem(length_cm=20.0)
        ruler.setPos(QPointF(100, 100))

        manip = BimanualSTEMManipulator(mode=BimanualTransformMode.ROTATE_ONLY)
        manip.set_target_item(ruler)

        # Start horizontal
        manip.process_hands(
            hand1_pos=(100.0, 150.0),
            hand2_pos=(300.0, 150.0),
            hand1_active=True,
            hand2_active=True,
        )

        # Rotate to 90 degrees
        t = manip.process_hands(
            hand1_pos=(200.0, 50.0),
            hand2_pos=(200.0, 250.0),
            hand1_active=True,
            hand2_active=True,
        )
        assert t is not None
        assert t.rotation_deg == 90.0
        assert pytest.approx(ruler.rotation(), 0.1) == 90.0

        manip.end_manipulation()



def test_main_window_bimanual_stem_integration(qapp):
    """Verify MainWindow owns BimanualSTEMManipulator and routes dual-pinch gestures to STEM items."""
    from src.ui.main_window import MainWindow
    from src.core.multi_hand_coordinator import HandLandmarkData, GestureState

    win = MainWindow(camera_index=-1)
    assert hasattr(win, "_bimanual_manipulator")
    assert win._bimanual_manipulator is not None

    # Insert an interactive trigonometric circle
    circle = InteractiveTrigonometricUnitCircleItem(center=(400.0, 300.0), radius=100.0, angle_deg=0.0)
    win._scene.addItem(circle)

    # Simulate two hands hovering over the circle at (300, 300) and (500, 300)
    h1 = HandLandmarkData(
        track_id="hand_0",
        handedness="Left",
        gesture_state=GestureState.SELECT,
        is_pinched=True,
        canvas_x=300.0,
        canvas_y=300.0,
        index_tip_norm=(0.3, 0.3),
    )
    h2 = HandLandmarkData(
        track_id="hand_1",
        handedness="Right",
        gesture_state=GestureState.SELECT,
        is_pinched=True,
        canvas_x=500.0,
        canvas_y=300.0,
        index_tip_norm=(0.5, 0.3),
    )

    win._on_multi_gesture_event([h1, h2])
    assert win._bimanual_manipulator.is_manipulating() is True
    assert win._bimanual_manipulator.get_target_item() is circle

    # Simulate hands rotating to vertical
    h1_rot = HandLandmarkData(
        track_id="hand_0",
        handedness="Left",
        gesture_state=GestureState.SELECT,
        is_pinched=True,
        canvas_x=400.0,
        canvas_y=200.0,
        index_tip_norm=(0.4, 0.2),
    )
    h2_rot = HandLandmarkData(
        track_id="hand_1",
        handedness="Right",
        gesture_state=GestureState.SELECT,
        is_pinched=True,
        canvas_x=400.0,
        canvas_y=400.0,
        index_tip_norm=(0.4, 0.4),
    )
    win._on_multi_gesture_event([h1_rot, h2_rot])
    assert circle.angle_deg == 90.0

    # Release hands
    h1_rel = HandLandmarkData(
        track_id="hand_0",
        handedness="Left",
        gesture_state=GestureState.POINT,
        is_pinched=False,
        canvas_x=400.0,
        canvas_y=200.0,
        index_tip_norm=(0.4, 0.2),
    )
    win._on_multi_gesture_event([h1_rel, h2_rot])
    assert win._bimanual_manipulator.is_manipulating() is False

    win.close()
