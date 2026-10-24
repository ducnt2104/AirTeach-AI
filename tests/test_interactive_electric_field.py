"""
AirTeach AI - Interactive Physics Electric Field & Coulomb's Law Unit Tests
===========================================================================
Verifies:
  - Superposition of Coulomb's Law electric field and electric potential.
  - Dipole neutral potential plane (V = 0) and constructive field vector (E > 0).
  - Like-charge neutral point (E = 0) and positive potential (V > 0).
  - 4th-Order Runge-Kutta (RK4) streamline numerical integration.
  - Electrostatic presets (Dipole, Two Positive, Two Negative, Quadrupole).
  - Interactive test probe and polarity inversion.
  - Touchless air gestures (pinch drag charges, air tap buttons).
  - Vector diagram stamping onto LessonBoard.
  - LessonBoard integration and region exclusion.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QApplication

from src.ui.interactive_electric_field import (
    InteractiveElectricFieldItem,
    ELECTROSTATIC_PRESETS,
)
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class TestInteractiveElectricField:
    """Test suite for InteractiveElectricFieldItem Coulomb electrostatic simulation."""

    def test_electric_field_init_and_properties(self, qapp):
        """Verifies initial configuration of electrostatic dipole simulation."""
        item = InteractiveElectricFieldItem(pos=(200.0, 150.0), preset="DIPOLE")

        assert item.pos() == QPointF(200.0, 150.0)
        assert item._active_preset == "DIPOLE"
        assert len(item._charges) == 2
        assert item._charges[0].q == 1.0   # +1 µC
        assert item._charges[1].q == -1.0  # -1 µC
        assert item._show_probe is True
        assert len(item._field_lines) > 0
        item.close()

    def test_coulomb_law_superposition_dipole(self, qapp):
        """
        Verifies Coulomb's Law on an electric dipole:
          - Midpoint between +q and -q has maximum Ex field pointing towards -q (+x direction).
          - Midpoint potential V = 0 due to exact antisymmetry.
        """
        item = InteractiveElectricFieldItem(preset="DIPOLE")

        q1 = item._charges[0]  # (220, 220, +1)
        q2 = item._charges[1]  # (440, 220, -1)
        mid_x = (q1.pos.x() + q2.pos.x()) / 2.0  # 330
        mid_y = (q1.pos.y() + q2.pos.y()) / 2.0  # 220

        ex, ey = item.electric_field_at(mid_x, mid_y)
        # Field must point to the right towards -q (ex > 0)
        assert ex > 0.0
        assert abs(ey) < 1e-3  # Ey is zero along the axis of symmetry

        # Potential V at midpoint must be zero (V1 + V2 = k*(+1)/d + k*(-1)/d = 0)
        v_mid = item.electric_potential_at(mid_x, mid_y)
        assert abs(v_mid) < 1e-3
        item.close()

    def test_coulomb_law_like_charges_neutral_point(self, qapp):
        """
        Verifies Coulomb's Law on two identical positive charges:
          - Midpoint is a neutral point where E = 0 (opposing repulsion vectors cancel).
          - Potential V at midpoint is strictly positive.
        """
        item = InteractiveElectricFieldItem(preset="TWO_POSITIVE")

        q1 = item._charges[0]  # (220, 220, +1)
        q2 = item._charges[1]  # (440, 220, +1)
        mid_x = (q1.pos.x() + q2.pos.x()) / 2.0
        mid_y = (q1.pos.y() + q2.pos.y()) / 2.0

        ex, ey = item.electric_field_at(mid_x, mid_y)
        # Field cancels out completely at center
        assert abs(ex) < 1e-3
        assert abs(ey) < 1e-3

        # Potential is strictly positive
        v_mid = item.electric_potential_at(mid_x, mid_y)
        assert v_mid > 0.0
        item.close()

    def test_rk4_field_lines_computation(self, qapp):
        """Verifies 4th-Order Runge-Kutta streamline integration creates smooth curves."""
        item = InteractiveElectricFieldItem(preset="DIPOLE")

        assert len(item._field_lines) >= 8
        for line in item._field_lines:
            assert len(line) >= 3
            # Check points are within simulation area
            for pt in line:
                assert -20.0 <= pt.x() <= item._width + 20.0
                assert -20.0 <= pt.y() <= item._height + 20.0
        item.close()

    def test_electrostatic_presets_and_inversion(self, qapp):
        """Verifies switching presets and inverting charge polarities."""
        item = InteractiveElectricFieldItem()

        # TWO_POSITIVE
        item.set_preset("TWO_POSITIVE")
        assert item._active_preset == "TWO_POSITIVE"
        assert all(c.q > 0 for c in item._charges)

        # TWO_NEGATIVE
        item.set_preset("TWO_NEGATIVE")
        assert item._active_preset == "TWO_NEGATIVE"
        assert all(c.q < 0 for c in item._charges)

        # QUADRUPOLE
        item.set_preset("QUADRUPOLE")
        assert item._active_preset == "QUADRUPOLE"
        assert len(item._charges) == 4

        # Invert charges
        item.set_preset("DIPOLE")
        assert item._charges[0].q == 1.0
        assert item._charges[1].q == -1.0

        item.invert_charges()
        assert item._charges[0].q == -1.0
        assert item._charges[1].q == 1.0
        item.close()

    def test_probe_movement_and_signals(self, qapp):
        """Verifies moving electric field probe emits updated physics values."""
        item = InteractiveElectricFieldItem()

        probe_emissions = []
        item.probe_updated.connect(lambda emag, ang, v: probe_emissions.append((emag, ang, v)))

        # Toggle probe
        is_shown = item.toggle_probe()
        assert is_shown is False
        is_shown = item.toggle_probe()
        assert is_shown is True

        # Recompute field lines triggers probe emission
        item.recompute_field_lines()
        assert len(probe_emissions) >= 1
        emag, ang, v = probe_emissions[-1]
        assert emag > 0.0
        item.close()

    def test_touchless_air_gestures(self, qapp):
        """Verifies touchless air tap button dispatch and air drag."""
        board = LessonBoard()
        item = InteractiveElectricFieldItem(pos=(100.0, 100.0))
        board.addItem(item)

        # 1. Air tap Probe button
        btn_probe_scene = item.mapToScene(item._rect_btn_probe.center())
        was_probe = item._show_probe
        consumed = item.handle_air_tap(btn_probe_scene, board=board)
        assert consumed is True
        assert item._show_probe != was_probe

        # 2. Air tap Invert button
        btn_invert_scene = item.mapToScene(item._rect_btn_invert.center())
        q0_before = item._charges[0].q
        consumed = item.handle_air_tap(btn_invert_scene, board=board)
        assert consumed is True
        assert item._charges[0].q == -q0_before

        # 3. Air tap Preset button (TWO_POSITIVE)
        btn_preset_scene = item.mapToScene(item._rect_btn_presets[1].center())
        consumed = item.handle_air_tap(btn_preset_scene, board=board)
        assert consumed is True
        assert item._active_preset == "TWO_POSITIVE"

        # 4. Air hover pos
        item.set_air_hover_pos(QPointF(200.0, 200.0))
        assert item._air_hover_pos is not None
        item.set_air_hover_pos(None)
        assert item._air_hover_pos is None

        # 5. Air tap Close button
        btn_close_scene = item.mapToScene(item._rect_btn_close.center())
        consumed = item.handle_air_tap(btn_close_scene, board=board)
        assert consumed is True
        assert item.scene() is None

    def test_stamp_to_canvas_and_lessonboard(self, qapp):
        """Verifies vector diagram stamping onto LessonBoard and region exclusion."""
        board = LessonBoard()
        item = InteractiveElectricFieldItem(pos=(200.0, 150.0))
        board.addItem(item)

        stamped = item.stamp_to_canvas(board)
        assert len(stamped) >= 10  # Field lines, arrowheads, charge spheres, text HUD

        for itm in stamped:
            assert itm.scene() is board

        # Test LessonBoard lifecycle
        board.show_electric_field(True, pos=(300.0, 250.0), preset="QUADRUPOLE")
        assert board.is_electric_field_visible() is True
        ef_item = board.get_electric_field_item()
        assert ef_item is not None
        assert ef_item.pos() == QPointF(300.0, 250.0)
        assert ef_item._active_preset == "QUADRUPOLE"

        # Region exclusion test
        sel_rect = QRectF(250.0, 200.0, 400.0, 350.0)
        items_inside = board.get_items_in_rect(sel_rect)
        assert ef_item not in items_inside

        # Toggle visibility
        is_vis = board.toggle_electric_field()
        assert is_vis is False
        assert board.is_electric_field_visible() is False

        is_vis = board.toggle_electric_field()
        assert is_vis is True
        assert board.is_electric_field_visible() is True

        if hasattr(ef_item, "close"):
            ef_item.close()
        item.close()
