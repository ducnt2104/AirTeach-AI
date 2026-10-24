"""
AirTeach AI - Test Suite for Interactive Fluid Dynamics Venturi Tube Simulator
=============================================================================
Verifies authentic physical calculations (Continuity, Bernoulli, U-tube manometer),
all 5 curriculum presets, parameter adjustments, animation lifecycles, touchless
air gestures, vector diagram stamping, LessonBoard integration, and MainWindow actions.
"""
import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsPolygonItem,
    QGraphicsRectItem,
    QGraphicsTextItem,
)

from src.ui.interactive_fluid_venturi import (
    GRAVITY_G,
    PRESETS,
    DENSITY_AIR,
    DENSITY_WATER,
    DENSITY_MERCURY,
    DENSITY_GASOLINE,
    FluidPreset,
    InteractiveFluidVenturiItem,
)
from src.ui.lesson_board import LessonBoard
from src.ui.main_window import MainWindow
from src.ui.stem_tools import PhysicsTools


@pytest.fixture
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def board(qapp):
    return LessonBoard()


@pytest.fixture
def venturi_item(qapp):
    return InteractiveFluidVenturiItem()


class TestFluidDynamicsPhysics:
    """Verifies rigorous mathematical and physical correctness (Zero Mock)."""

    def test_continuity_equation_throat_velocity(self, venturi_item):
        """Continuity equation: A1 * v1 = A2 * v2 => v2 = v1 * (D1 / D2)^2."""
        venturi_item.load_preset("VENTURI_METER")
        d1 = venturi_item._inlet_dia_mm    # 100 mm
        d2 = venturi_item._throat_dia_mm   # 45 mm
        v1 = venturi_item._inlet_speed_ms  # 8.0 m/s

        expected_v2 = v1 * (d1 / d2) ** 2
        assert math.isclose(venturi_item.throat_speed_ms, expected_v2, rel_tol=1e-4)
        assert venturi_item.throat_speed_ms > v1  # Flow accelerates in constriction

    def test_bernoulli_static_pressure_drop(self, venturi_item):
        """Bernoulli equation: Delta P = 0.5 * rho * (v2^2 - v1^2)."""
        venturi_item.load_preset("VENTURI_METER")
        v1 = venturi_item._inlet_speed_ms
        v2 = venturi_item.throat_speed_ms
        rho = venturi_item._preset.rho_fluid

        expected_dp = 0.5 * rho * (v2 ** 2 - v1 ** 2)
        assert math.isclose(venturi_item.pressure_drop_pa, expected_dp, rel_tol=1e-4)
        assert venturi_item.pressure_drop_pa > 0.0  # Pressure drops at throat

    def test_manometer_height_difference(self, venturi_item):
        """U-tube manometer level difference: Delta h = Delta P / (rho_m * g)."""
        venturi_item.load_preset("VENTURI_METER")
        dp = venturi_item.pressure_drop_pa
        rho_m = venturi_item._preset.rho_manometer
        expected_dh_mm = (dp / (rho_m * GRAVITY_G)) * 1000.0

        assert math.isclose(venturi_item.manometer_height_diff_mm, expected_dh_mm, rel_tol=1e-4)
        assert venturi_item.manometer_height_diff_mm > 0.0

    def test_volumetric_flow_rate(self, venturi_item):
        """Volumetric flow rate Q = A1 * v1 = pi * (D1/2)^2 * v1."""
        venturi_item.load_preset("VENTURI_METER")
        r1_m = (venturi_item._inlet_dia_mm / 2000.0)
        expected_q = math.pi * (r1_m ** 2) * venturi_item._inlet_speed_ms

        assert math.isclose(venturi_item.flow_rate_m3s, expected_q, rel_tol=1e-4)

    def test_all_five_presets(self, venturi_item):
        """Verify all 5 curriculum presets load and produce valid physical values."""
        for code in ("VENTURI_METER", "CARBURETOR_CHOKE", "AIRPLANE_WING_LIFT", "PERFUME_ASPIRATOR", "WATER_PIPE_RESTRICTION"):
            assert venturi_item.load_preset(code) is True
            assert venturi_item._active_preset_code == code
            assert venturi_item.throat_speed_ms > 0.0
            assert venturi_item.pressure_drop_pa > 0.0
            assert venturi_item.manometer_height_diff_mm > 0.0

    def test_throat_diameter_clamping(self, venturi_item):
        """Throat diameter D2 must be clamped between 15mm and D1."""
        venturi_item.load_preset("VENTURI_METER")
        d1 = venturi_item._inlet_dia_mm

        venturi_item.set_throat_diameter(5.0)  # below min 15
        assert venturi_item._throat_dia_mm == 15.0

        venturi_item.set_throat_diameter(d1 + 50.0)  # above max D1
        assert venturi_item._throat_dia_mm == d1

    def test_inlet_speed_clamping(self, venturi_item):
        """Inlet fluid speed v1 must be clamped between 0.5 and 40 m/s."""
        venturi_item.set_inlet_speed(0.1)
        assert venturi_item._inlet_speed_ms == 0.5

        venturi_item.set_inlet_speed(100.0)
        assert venturi_item._inlet_speed_ms == 40.0


class TestAnimationAndInteraction:
    """Verifies animation, touchless air gestures, and UI responsiveness."""

    def test_animation_toggle(self, venturi_item):
        """Test play/pause toggles timer."""
        initial_anim = venturi_item._is_animating
        toggled = venturi_item.toggle_animation()
        assert toggled != initial_anim
        assert venturi_item._is_animating == toggled

        # Restore
        venturi_item.toggle_animation()

    def test_particle_animation_tick(self, venturi_item):
        """Particles must move forward during animation tick."""
        p0_x = venturi_item._particles[0][0]
        venturi_item._on_animation_tick()
        p1_x = venturi_item._particles[0][0]
        assert p1_x != p0_x

    def test_air_pinch_presets(self, venturi_item):
        """Touchless air pinch on preset button loads preset."""
        preset_r = venturi_item._btn_rects["preset_CARBURETOR_CHOKE"]
        pinch_pt = venturi_item.mapToScene(preset_r.center())

        handled = venturi_item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert venturi_item._active_preset_code == "CARBURETOR_CHOKE"

    def test_air_pinch_speed_controls(self, venturi_item):
        """Touchless air pinch on +/- speed buttons modifies speed."""
        initial_speed = venturi_item._inlet_speed_ms
        inc_r = venturi_item._btn_rects["inc_speed"]
        pinch_pt = venturi_item.mapToScene(inc_r.center())

        handled = venturi_item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert venturi_item._inlet_speed_ms == initial_speed + 2.0

        dec_r = venturi_item._btn_rects["dec_speed"]
        pinch_pt2 = venturi_item.mapToScene(dec_r.center())
        handled2 = venturi_item.handle_air_pinch(pinch_pt2)
        assert handled2 is True
        assert venturi_item._inlet_speed_ms == initial_speed

    def test_air_pinch_throat_controls(self, venturi_item):
        """Touchless air pinch on narrow/widen throat modifies D2."""
        initial_d2 = venturi_item._throat_dia_mm
        narrow_r = venturi_item._btn_rects["narrow_throat"]
        pinch_pt = venturi_item.mapToScene(narrow_r.center())

        handled = venturi_item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert venturi_item._throat_dia_mm == initial_d2 - 5.0

    def test_air_pinch_close(self, venturi_item):
        """Air pinch on close button hides item."""
        close_r = venturi_item._btn_rects["close"]
        pinch_pt = venturi_item.mapToScene(close_r.center())

        venturi_item.setVisible(True)
        handled = venturi_item.handle_air_pinch(pinch_pt)
        assert handled is True
        assert venturi_item.isVisible() is False


class TestVectorDiagramAndBoardIntegration:
    """Verifies vector diagram stamping, LessonBoard lifecycle, and MainWindow slots."""

    def test_physics_tools_create_fluid_venturi_diagram(self):
        """Verify PhysicsTools.create_fluid_venturi_diagram generates styled vector items."""
        items = PhysicsTools.create_fluid_venturi_diagram(
            center=(400.0, 300.0),
            preset_code="VENTURI_METER",
            inlet_dia_mm=100.0,
            throat_dia_mm=45.0,
            inlet_speed_ms=8.0,
        )
        assert len(items) >= 12, "Diagram must generate background, tube walls, streamlines, vectors, manometer, and info"

        # Check background card exists
        bg_items = [it for it in items if isinstance(it, QGraphicsRectItem)]
        assert len(bg_items) >= 3

        # Check text items
        text_items = [it for it in items if isinstance(it, QGraphicsTextItem)]
        all_text = " ".join(t.toPlainText() for t in text_items)
        assert "CƠ HỌC CHẤT LƯU" in all_text
        assert "v₁=" in all_text
        assert "v₂=" in all_text
        assert "Δh =" in all_text
        assert "Bernoulli" in all_text

    def test_stamp_vector_diagram_to_board(self, board, venturi_item):
        """Verify stamping vector diagram to LessonBoard."""
        board.addItem(venturi_item)
        init_items = len(board.items())

        venturi_item.stamp_to_board()
        new_items = len(board.items())
        assert new_items > init_items + 10

    def test_lesson_board_lifecycle(self, board):
        """Verify LessonBoard show/toggle/get methods for fluid venturi."""
        assert board.is_fluid_venturi_visible() is False
        assert board.get_fluid_venturi() is None

        # Show
        item = board.show_fluid_venturi(True)
        assert item is not None
        assert board.is_fluid_venturi_visible() is True
        assert board.get_fluid_venturi() == item
        assert board.get_fluid_venturi_item() == item

        # Toggle to hide
        assert board.toggle_fluid_venturi() is False
        assert board.is_fluid_venturi_visible() is False

        # Toggle to show
        assert board.toggle_fluid_venturi() is True
        assert board.is_fluid_venturi_visible() is True

    def test_main_window_toggle_and_action(self, qapp):
        """Verify MainWindow slot and UI trigger for fluid venturi."""
        win = MainWindow(camera_index=-1)
        win.show()

        try:
            assert win._on_toggle_fluid_venturi() is True
            assert win._scene.is_fluid_venturi_visible() is True

            assert win._on_toggle_fluid_venturi() is False
            assert win._scene.is_fluid_venturi_visible() is False
        finally:
            win.close()
