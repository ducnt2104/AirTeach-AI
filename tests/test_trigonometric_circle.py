"""
AirTeach AI - Interactive Trigonometric Unit Circle Test Suite
==============================================================
Tests for:
  - InteractiveTrigonometricUnitCircleItem
  - Exact trigonometric values lookup (0°, 30°, 45°, 60°, 90°, etc.)
  - Cardinal angle snapping
  - Mathematical projections (sin, cos, tan, cot)
  - Quadrant classification
  - Touchless air-click controls (+15°, -15°, Snap, circumference)
  - MathTools.create_trigonometric_circle_diagram integration
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.ui.interactive_trigonometric_circle import (
    InteractiveTrigonometricUnitCircleItem,
    CARDINAL_EXACT_VALUES,
)
from src.ui.stem_tools import MathTools


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


def test_trigonometric_circle_basic_properties(qapp):
    """Verify geometry, radius, angle, and trigonometric evaluations."""
    item = InteractiveTrigonometricUnitCircleItem(center=(500.0, 400.0), radius=180.0, angle_deg=30.0)

    assert item.pos().x() == 500.0
    assert item.pos().y() == 400.0
    assert item.radius == 180.0
    assert item.angle_deg == 30.0
    assert item.angle_rad == pytest.approx(math.pi / 6.0)

    # 30°: sin = 0.5, cos = sqrt(3)/2 ~= 0.8660, tan = 1/sqrt(3) ~= 0.5774
    assert item.sin_val == pytest.approx(0.5, abs=1e-5)
    assert item.cos_val == pytest.approx(math.sqrt(3.0) / 2.0, abs=1e-5)
    assert item.tan_val == pytest.approx(1.0 / math.sqrt(3.0), abs=1e-5)
    assert item.cot_val == pytest.approx(math.sqrt(3.0), abs=1e-5)
    assert item.quadrant_name == "Góc phần tư I"


def test_trigonometric_circle_quadrants_and_axes(qapp):
    """Verify correct quadrant classification across 4 quadrants and coordinate axes."""
    item = InteractiveTrigonometricUnitCircleItem()

    item.set_angle_deg(0.0)
    assert item.quadrant_name == "Trục tọa độ"

    item.set_angle_deg(45.0)
    assert item.quadrant_name == "Góc phần tư I"

    item.set_angle_deg(90.0)
    assert item.quadrant_name == "Trục tọa độ"
    assert item.tan_val is None  # Undefined at 90°

    item.set_angle_deg(135.0)
    assert item.quadrant_name == "Góc phần tư II"

    item.set_angle_deg(180.0)
    assert item.quadrant_name == "Trục tọa độ"
    assert item.cot_val is None  # Undefined at 180°

    item.set_angle_deg(225.0)
    assert item.quadrant_name == "Góc phần tư III"

    item.set_angle_deg(315.0)
    assert item.quadrant_name == "Góc phần tư IV"


def test_trigonometric_circle_exact_algebraic_values(qapp):
    """Verify exact algebraic value table mapping for cardinal angles."""
    item = InteractiveTrigonometricUnitCircleItem()

    # 45°
    item.set_angle_deg(45.0)
    vals_45 = item.get_exact_values()
    assert vals_45["rad"] == "π/4"
    assert vals_45["sin"] == "√2/2"
    assert vals_45["cos"] == "√2/2"
    assert vals_45["tan"] == "1"
    assert vals_45["cot"] == "1"

    # 60°
    item.set_angle_deg(60.0)
    vals_60 = item.get_exact_values()
    assert vals_60["rad"] == "π/3"
    assert vals_60["sin"] == "√3/2"
    assert vals_60["cos"] == "1/2"

    # Non-cardinal angle (e.g. 37.5°) returns formatted numerical floats
    item.set_angle_deg(37.5)
    vals_float = item.get_exact_values()
    assert "0.6088" in vals_float["sin"]


def test_trigonometric_circle_rotation_and_snapping(qapp):
    """Verify rotate_by and snap_to_cardinal_angle behavior."""
    item = InteractiveTrigonometricUnitCircleItem(angle_deg=40.0)

    # Rotate by +20° -> 60°
    new_ang = item.rotate_by(20.0)
    assert new_ang == 60.0
    assert item.angle_deg == 60.0

    # Near 45° (e.g. 43.0°) snaps to 45.0°
    item.set_angle_deg(43.0)
    snapped = item.snap_to_cardinal_angle(tolerance_deg=3.5)
    assert snapped == 45.0
    assert item.angle_deg == 45.0

    # Outside tolerance does not snap
    item.set_angle_deg(50.0)
    not_snapped = item.snap_to_cardinal_angle(tolerance_deg=2.0)
    assert not_snapped is None
    assert item.angle_deg == 50.0


def test_trigonometric_circle_air_clicks(qapp):
    """Verify touchless air taps on +/-15° buttons and unit circle circumference."""
    item = InteractiveTrigonometricUnitCircleItem(center=(500.0, 400.0), radius=180.0, angle_deg=30.0)
    r = item.radius

    # 1. Click on [+15°] button
    # Local pos: x in [-r+10, -r+70] = [-170, -110], y in [r+45, r+71] = [225, 251]
    # In scene coords: x = 500 - 140 = 360, y = 400 + 238 = 638
    b_plus_scene = item.mapToScene(QPointF(-r + 40.0, r + 58.0))
    handled_plus = item.air_click_at(b_plus_scene.x(), b_plus_scene.y())
    assert handled_plus is True
    assert item.angle_deg == 45.0

    # 2. Click on [-15°] button
    b_minus_scene = item.mapToScene(QPointF(-r - 30.0, r + 58.0))
    handled_minus = item.air_click_at(b_minus_scene.x(), b_minus_scene.y())
    assert handled_minus is True
    assert item.angle_deg == 30.0

    # 3. Click directly on circumference at 90° (local x=0, local y=-r)
    circ_scene = item.mapToScene(QPointF(0.0, -r))
    handled_circ = item.air_click_at(circ_scene.x(), circ_scene.y())
    assert handled_circ is True
    assert item.angle_deg == pytest.approx(90.0, abs=1.0)

    # 4. Air click far away
    assert item.air_click_at(10.0, 10.0) is False


def test_trigonometric_circle_display_toggles(qapp):
    """Verify display visibility toggles for tangent, cotangent, projections, and HUD."""
    item = InteractiveTrigonometricUnitCircleItem()

    item.set_show_tangent(False)
    assert item._show_tangent is False
    item.set_show_tangent(True)
    assert item._show_tangent is True

    item.set_show_cotangent(False)
    assert item._show_cotangent is False

    item.set_show_projections(False)
    assert item._show_projections is False

    item.set_show_hud_badge(False)
    assert item._show_hud_badge is False


def test_mathtools_create_trigonometric_circle_diagram(qapp):
    """Verify MathTools factory method returns a valid, fully configured unit circle item."""
    scene = QGraphicsScene()
    item = MathTools.create_trigonometric_circle_diagram(
        center=(600.0, 450.0),
        radius=200.0,
        angle_deg=60.0,
        show_tangent=True,
    )
    assert isinstance(item, InteractiveTrigonometricUnitCircleItem)
    assert item.radius == 200.0
    assert item.angle_deg == 60.0

    scene.addItem(item)
    assert item.scene() == scene
    rect = item.boundingRect()
    assert rect.width() > 400.0
    assert rect.height() > 400.0
