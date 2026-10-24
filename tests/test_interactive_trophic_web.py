"""
Unit tests for Interactive Biology Ecosystem Trophic Food Web & Lotka-Volterra Simulator.
"""
from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication, QGraphicsScene

from src.ui.interactive_trophic_web import InteractiveTrophicWebItem
from src.ui.stem_tools import BiologyTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def scene(qapp):
    return QGraphicsScene()


@pytest.fixture
def trophic_sim(scene):
    item = InteractiveTrophicWebItem(
        pos=(200.0, 150.0),
        width=720.0,
        height=470.0,
        alpha=0.80,
        beta=0.02,
        gamma=0.40,
        delta=0.005,
        prey_init=100.0,
        predator_init=30.0,
    )
    scene.addItem(item)
    yield item
    item.close()


class TestInteractiveTrophicWeb:
    """Test suite for InteractiveTrophicWebItem."""

    def test_trophic_web_init_and_properties(self, trophic_sim: InteractiveTrophicWebItem):
        assert trophic_sim.alpha == pytest.approx(0.80, rel=1e-3)
        assert trophic_sim.beta == pytest.approx(0.02, rel=1e-3)
        assert trophic_sim.gamma == pytest.approx(0.40, rel=1e-3)
        assert trophic_sim.delta == pytest.approx(0.005, rel=1e-3)
        assert trophic_sim.prey_population == pytest.approx(100.0, rel=1e-3)
        assert trophic_sim.predator_population == pytest.approx(30.0, rel=1e-3)
        assert trophic_sim.is_running is True

        # Test stable fixed point (x*, y*) = (gamma / delta, alpha / beta)
        # x* = 0.40 / 0.005 = 80.0
        # y* = 0.80 / 0.02 = 40.0
        x_star, y_star = trophic_sim.equilibrium_fixed_point
        assert x_star == pytest.approx(80.0, rel=1e-3)
        assert y_star == pytest.approx(40.0, rel=1e-3)

        # Conserved Liapunov function V(x, y)
        v0 = trophic_sim.conserved_quantity
        assert isinstance(v0, float)
        assert math.isfinite(v0)

    def test_lotka_volterra_rk4_dynamics(self, trophic_sim: InteractiveTrophicWebItem):
        # Initial state
        x0 = trophic_sim.prey_population
        y0 = trophic_sim.predator_population

        # Step simulation several times
        for _ in range(50):
            trophic_sim.step_simulation(0.04)

        x1 = trophic_sim.prey_population
        y1 = trophic_sim.predator_population

        # Populations must remain strictly positive and within physical boundaries
        assert x1 > 0.0
        assert y1 > 0.0
        assert len(trophic_sim._history) > 1

        # Must have evolved away from initial point
        assert (x1 != x0) or (y1 != y0)

    def test_conserved_quantity_approximate_invariance(self, trophic_sim: InteractiveTrophicWebItem):
        # For small dt, the first integral V(x, y) should stay approximately constant
        v_initial = trophic_sim.conserved_quantity
        for _ in range(10):
            trophic_sim.step_simulation(0.01)

        v_final = trophic_sim.conserved_quantity
        # With high-order RK4 and dt=0.01, drift should be minimal (< 2%)
        assert abs(v_final - v_initial) / abs(v_initial) < 0.02

    def test_ecological_perturbations_drought_and_disease(self, trophic_sim: InteractiveTrophicWebItem):
        # 1. Drought cuts prey by 50%
        pre_drought_x = trophic_sim.prey_population
        trophic_sim.trigger_drought()
        assert trophic_sim.prey_population == pytest.approx(pre_drought_x * 0.5, rel=1e-2)
        assert "HẠN HÁN" in (trophic_sim._shock_alert or "")

        # 2. Disease cuts predators by 60% (retaining 40%)
        pre_disease_y = trophic_sim.predator_population
        trophic_sim.trigger_disease()
        assert trophic_sim.predator_population == pytest.approx(pre_disease_y * 0.4, rel=1e-2)
        assert "DỊCH BỆNH" in (trophic_sim._shock_alert or "")

        # 3. Add prey and predators
        cur_x = trophic_sim.prey_population
        trophic_sim.add_prey(30.0)
        assert trophic_sim.prey_population == pytest.approx(cur_x + 30.0, rel=1e-2)

        cur_y = trophic_sim.predator_population
        trophic_sim.add_predators(20.0)
        assert trophic_sim.predator_population == pytest.approx(cur_y + 20.0, rel=1e-2)

        # 4. Reset restores initial baseline
        trophic_sim.reset_simulation()
        assert trophic_sim.prey_population == pytest.approx(100.0, rel=1e-3)
        assert trophic_sim.predator_population == pytest.approx(30.0, rel=1e-3)

    def test_parameter_tuning_and_pyramid_toggle(self, trophic_sim: InteractiveTrophicWebItem):
        trophic_sim.set_alpha(1.2)
        assert trophic_sim.alpha == pytest.approx(1.2, rel=1e-3)

        trophic_sim.set_beta(0.035)
        assert trophic_sim.beta == pytest.approx(0.035, rel=1e-3)

        assert trophic_sim._show_pyramid is False
        trophic_sim.toggle_pyramid_view()
        assert trophic_sim._show_pyramid is True
        trophic_sim.toggle_pyramid_view()
        assert trophic_sim._show_pyramid is False

    def test_touchless_air_gestures(self, trophic_sim: InteractiveTrophicWebItem):
        # Test air pinch on a species node (e.g. producer)
        producer_node = trophic_sim._nodes["producer"]
        scene_pos = trophic_sim.mapToScene(producer_node.pos)

        # Pinch press
        handled = trophic_sim.handle_air_pinch(scene_pos, is_pressed=True)
        assert handled is True
        assert trophic_sim._active_drag_node == "producer"

        # Drag movement
        drag_scene_pos = trophic_sim.mapToScene(producer_node.pos + QPointF(20.0, 15.0))
        handled_move = trophic_sim.handle_air_pinch(drag_scene_pos, is_pressed=True)
        assert handled_move is True

        # Pinch release
        handled_rel = trophic_sim.handle_air_pinch(drag_scene_pos, is_pressed=False)
        assert handled_rel is True
        assert trophic_sim._active_drag_node is None

        # Test air tap on buttons
        run_btn_pos = trophic_sim.mapToScene(trophic_sim._rect_btn_run.center())
        was_running = trophic_sim.is_running
        assert trophic_sim.handle_air_tap(run_btn_pos) is True
        assert trophic_sim.is_running != was_running

        drought_btn_pos = trophic_sim.mapToScene(trophic_sim._rect_btn_drought.center())
        assert trophic_sim.handle_air_tap(drought_btn_pos) is True
        assert "HẠN HÁN" in (trophic_sim._shock_alert or "")

    def test_stem_tools_create_trophic_pyramid_and_stamping(self, qapp):
        items = BiologyTools.create_trophic_pyramid(
            center=(400.0, 300.0),
            base_width=380.0,
            total_height=260.0,
            efficiency=0.10,
            producer_energy=10000.0,
        )
        assert len(items) >= 4  # 4 trapezoid tiers + text labels + arrows

        from src.ui.lesson_board import LessonBoard
        board = LessonBoard()
        trophic_sim = InteractiveTrophicWebItem(pos=(200.0, 150.0))
        board.addItem(trophic_sim)

        # Verify stamping signal & items added to board
        stamp_called = False

        def on_stamp():
            nonlocal stamp_called
            stamp_called = True

        trophic_sim.stamp_requested.connect(on_stamp)
        stamped_items = trophic_sim.stamp_to_canvas(board)
        assert stamp_called is True
        assert len(stamped_items) >= 4

    def test_lessonboard_and_main_window_integration(self, qapp):
        from src.ui.lesson_board import LessonBoard

        board = LessonBoard()

        # Verify show and toggle methods on LessonBoard
        assert board.get_trophic_web() is None
        item = board.show_trophic_web(True, pos=(300.0, 200.0))
        assert item is not None
        assert board.get_trophic_web() is item
        assert item.pos() == QPointF(300.0, 200.0)

        # Calling again returns same instance
        assert board.show_trophic_web() is item

        # Toggle hides
        board.toggle_trophic_web()
        assert board.get_trophic_web() is None
        assert board.is_trophic_web_visible() is False

        # Toggle shows again
        board.toggle_trophic_web()
        assert board.get_trophic_web() is not None
        assert board.is_trophic_web_visible() is True

        # Check exclusion from eraser/selection
        items_in_rect = board.get_items_in_rect(board.sceneRect())
        assert board.get_trophic_web() not in items_in_rect
