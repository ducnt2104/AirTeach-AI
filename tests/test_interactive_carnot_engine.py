"""
Unit Tests for Interactive Thermodynamics Carnot Heat Engine & PV / TS Cycle Simulator
========================================================================================
Verifies:
  1. Authentic thermodynamic constants (Universal gas constant R, diatomic gamma = 1.40).
  2. Four reversible stages and Carnot symmetry (V3 / V4 = V2 / V1 = r).
  3. Poisson adiabatic expansion relation T_H * V2^(gamma-1) = T_C * V3^(gamma-1).
  4. Energy balance: Q_H, |Q_C|, W_net = Q_H - |Q_C|, and Carnot efficiency eta = 1 - T_C/T_H.
  5. Reverse cycle: Refrigerator and heat pump coefficient of performance (COP).
  6. Curriculum presets (STANDARD_STEAM, HIGH_EFFICIENCY, REVERSE_REFRIGERATOR, LOW_DELTA_T).
  7. Animation phase progression, cycle telemetry, and air gestures (pinch/tap).
  8. LessonBoard lifecycle, visibility toggles, selection exclusion, and vector diagram stamping.
"""
import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from src.ui.interactive_carnot_engine import (
    InteractiveCarnotEngineItem,
    GAS_CONSTANT_R,
    DEFAULT_MOLES_N,
    GAMMA_DIATOMIC,
    PRESETS,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import PhysicsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def board(qapp):
    return LessonBoard()


@pytest.fixture
def item(board):
    it = InteractiveCarnotEngineItem(pos=(320.0, 160.0))
    board.addItem(it)
    yield it
    it.cleanup()
    if it.scene():
        board.removeItem(it)


class TestCarnotEngineThermodynamics:
    """Verifies fundamental laws of thermodynamics and Carnot cycle mechanics."""

    def test_thermodynamic_constants(self):
        assert abs(GAS_CONSTANT_R - 8.314462618) < 1e-6
        assert abs(GAMMA_DIATOMIC - 1.40) < 1e-5
        assert DEFAULT_MOLES_N == 1.0

    def test_standard_steam_efficiency_and_work(self, item):
        """T_H = 600 K, T_C = 300 K, r = 2.0. Expected eta = 50.0%."""
        item.set_th(600.0)
        item.set_tc(300.0)
        item.set_ratio(2.0)

        eta = item.get_carnot_efficiency()
        assert abs(eta - 0.50) < 1e-5

        # Entropy change: ΔS = n * R * ln(r)
        expected_ds = 1.0 * GAS_CONSTANT_R * math.log(2.0)
        assert abs(item.get_entropy_change_ds() - expected_ds) < 1e-4

        # Heat absorbed: Q_H = T_H * ΔS
        qh = item.get_heat_absorbed_qh()
        assert abs(qh - (600.0 * expected_ds)) < 0.1

        # Heat rejected: |Q_C| = T_C * ΔS
        qc = item.get_heat_rejected_qc_magnitude()
        assert abs(qc - (300.0 * expected_ds)) < 0.1

        # Net work: W_net = Q_H - |Q_C|
        wnet = item.get_net_work_wnet()
        assert abs(wnet - (qh - qc)) < 1e-5
        assert abs(wnet - (300.0 * expected_ds)) < 0.1

        # Ratio W_net / Q_H must equal efficiency
        assert abs((wnet / qh) - eta) < 1e-5

    def test_carnot_symmetry_and_adiabatic_relations(self, item):
        """Checks Poisson relation and V3 / V4 = V2 / V1 = r."""
        item.set_th(750.0)
        item.set_tc(300.0)
        item.set_ratio(2.2)

        states = item.get_state_points()
        v1, p1 = states["1"]
        v2, p2 = states["2"]
        v3, p3 = states["3"]
        v4, p4 = states["4"]

        # V2 / V1 = r
        assert abs((v2 / v1) - 2.2) < 1e-4

        # V3 / V4 must also equal r = 2.2
        assert abs((v3 / v4) - 2.2) < 1e-4

        # Poisson relation for adiabatic expansion (2 -> 3):
        # T_H * V2^(gamma-1) == T_C * V3^(gamma-1)
        lhs = 750.0 * math.pow(v2, 1.4 - 1.0)
        rhs = 300.0 * math.pow(v3, 1.4 - 1.0)
        assert abs(lhs - rhs) / lhs < 1e-4

        # Poisson relation for adiabatic compression (4 -> 1):
        # T_C * V4^(gamma-1) == T_H * V1^(gamma-1)
        lhs4 = 300.0 * math.pow(v4, 1.4 - 1.0)
        rhs1 = 750.0 * math.pow(v1, 1.4 - 1.0)
        assert abs(lhs4 - rhs1) / rhs1 < 1e-4

    def test_reverse_cycle_refrigerator_cop(self, item):
        """T_H = 310 K, T_C = 250 K. COP_ref = 250 / (310 - 250) = 4.167."""
        item.set_th(310.0)
        item.set_tc(250.0)

        cop_ref = item.get_carnot_cop_refrigerator()
        expected_ref = 250.0 / (310.0 - 250.0)
        assert abs(cop_ref - expected_ref) < 1e-4
        assert abs(cop_ref - 4.1667) < 0.01

        cop_pump = item.get_carnot_cop_heat_pump()
        assert abs(cop_pump - (cop_ref + 1.0)) < 1e-5

    def test_curriculum_presets(self, item):
        for key, p in PRESETS.items():
            item.apply_preset(key)
            assert abs(item.th_k - p["th_k"]) < 0.1
            assert abs(item.tc_k - p["tc_k"]) < 0.1
            assert abs(item.ratio_r - p["ratio_r"]) < 0.05
            assert item.get_carnot_efficiency() > 0.0


class TestCarnotEngineInteractionAndBoard:
    """Verifies gestures, UI modes, lesson board lifecycle, and vector stamping."""

    def test_air_pinch_and_tap_controls(self, item):
        # Mode toggle: P-V vs T-S
        tab_pv, tab_ts = item._get_mode_toggle_rects()

        # Switch to TS_DIAGRAM
        pt_ts = item.mapToScene(tab_ts.center())
        assert item.handle_air_tap(pt_ts) is True
        assert item._diagram_mode == "TS_DIAGRAM"

        # Switch back to PV_DIAGRAM
        pt_pv = item.mapToScene(tab_pv.center())
        assert item.handle_air_pinch(pt_pv) is True
        assert item._diagram_mode == "PV_DIAGRAM"

        # Toggle animation play/pause
        pp_rect = item._get_play_pause_button_rect()
        pt_pp = item.mapToScene(pp_rect.center())
        init_anim = item._is_animating
        assert item.handle_air_tap(pt_pp) is True
        assert item._is_animating != init_anim

        # Apply preset HIGH_EFFICIENCY
        for key, rect in item._get_preset_rects():
            if key == "HIGH_EFFICIENCY":
                pt_preset = item.mapToScene(rect.center())
                assert item.handle_air_tap(pt_preset) is True
                assert item.th_k == 900.0
                assert abs(item.get_carnot_efficiency() - 0.6667) < 0.01
                break

    def test_lesson_board_lifecycle(self, board):
        # Initial state
        assert board.is_carnot_engine_visible() is False
        assert board.get_carnot_engine() is None

        # Show
        ce_item = board.show_carnot_engine(True, (380.0, 180.0))
        assert ce_item is not None
        assert board.is_carnot_engine_visible() is True
        assert board.get_carnot_engine() == ce_item
        assert board.get_carnot_engine_item() == ce_item
        assert ce_item.x() == 380.0
        assert ce_item.y() == 180.0

        # Toggle to hide
        assert board.toggle_carnot_engine() is False
        assert board.is_carnot_engine_visible() is False

        # Toggle to show
        assert board.toggle_carnot_engine() is True
        assert board.is_carnot_engine_visible() is True

        # Selection exclusion check
        selected = board.get_items_in_rect(board.sceneRect())
        assert ce_item not in selected

    def test_vector_stamping_to_board(self, board, item):
        initial_count = len(board.items())

        item.stamp_to_board()
        assert len(board.items()) > initial_count

        # Verify PhysicsTools.create_carnot_engine_diagram directly
        diagram = PhysicsTools.create_carnot_engine_diagram(
            center=(500.0, 300.0),
            width=520.0,
            height=320.0,
            th_k=600.0,
            tc_k=300.0,
            ratio_r=2.0,
            preset="STANDARD_STEAM",
        )
        assert len(diagram) >= 8
        text_contents = [it.toPlainText() for it in diagram if hasattr(it, "toPlainText")]
        assert any("CARNOT" in t for t in text_contents)
        assert any("W_net" in t for t in text_contents)
        assert any("T_H" in t for t in text_contents)
