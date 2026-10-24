"""
Unit tests for InteractiveHPLCItem and ChemistryTools.create_hplc_diagram.
Verifies Van Deemter column efficiency equation H = A + B/u + C·u, optimal linear velocity,
theoretical plate count N = L/H, hydraulic backpressure ΔP, void volume dead time t_0,
capacity factor k', chromatographic resolution R_s = 2(t_R2 - t_R1)/(W1 + W2),
touchless air gestures, and lesson board integration.
"""

import sys
import math
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from src.ui.interactive_hplc import (
    InteractiveHPLCItem,
    HPLC_PRESETS,
)
from src.ui.stem_tools import ChemistryTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_hplc_initialization(qapp):
    item = InteractiveHPLCItem()
    assert item.active_preset.code == "REVERSE_PHASE_ANALGESICS"
    assert math.isclose(item.column_length_mm, 150.0, rel_tol=1e-4)
    assert math.isclose(item.particle_size_um, 5.0, rel_tol=1e-4)
    assert math.isclose(item.flow_rate_ml_min, 1.0, rel_tol=1e-4)
    assert len(item.active_preset.analytes) == 3
    assert item.boundingRect().width() == InteractiveHPLCItem.CARD_W
    assert item.boundingRect().height() == InteractiveHPLCItem.CARD_H


def test_van_deemter_equation_and_minimum_plate_height(qapp):
    """
    Verifies Van Deemter efficiency equation:
    H(u) = A + B/u + C·u
    At u_opt = √(B/C), H(u_opt) = H_min = A + 2√(B·C).
    For any u != u_opt, H(u) > H_min.
    """
    item = InteractiveHPLCItem()
    a, b, c = item.van_deemter_coefficients
    u_opt = item.optimal_linear_velocity_mm_s
    h_min_theoretical = a + 2.0 * math.sqrt(b * c)

    assert math.isclose(item.min_plate_height_um, h_min_theoretical, rel_tol=1e-5)

    # Test H at u_opt
    h_at_opt = a + (b / u_opt) + (c * u_opt)
    assert math.isclose(h_at_opt, h_min_theoretical, rel_tol=1e-5)

    # Test at velocities away from optimal: H must strictly exceed H_min
    for factor in [0.2, 0.5, 1.5, 2.5, 4.0]:
        test_u = u_opt * factor
        h_val = a + (b / test_u) + (c * test_u)
        assert h_val > h_min_theoretical


def test_theoretical_plates_and_column_dimensions(qapp):
    """
    Verifies theoretical plate count N = L / H.
    Doubling column length strictly doubles plate count N for identical packing and flow.
    """
    item = InteractiveHPLCItem()
    item._col_length_mm = 100.0
    n_100 = item.theoretical_plates_n

    item._col_length_mm = 200.0
    n_200 = item.theoretical_plates_n

    assert math.isclose(n_200, 2.0 * n_100, rel_tol=1e-5)


def test_sub2um_uhplc_van_deemter_flattening(qapp):
    """
    Verifies UHPLC sub-2μm particle advantages and trade-offs:
    1. C-term (mass transfer resistance) is proportional to d_p² -> much flatter curve at high u.
    2. Backpressure ΔP is inversely proportional to d_p² -> substantially higher pressure.
    """
    item = InteractiveHPLCItem()
    item.load_preset("REVERSE_PHASE_ANALGESICS")  # dp = 5.0 μm
    p_5um = item.column_backpressure_bar
    _, _, c_5um = item.van_deemter_coefficients

    item.load_preset("VAN_DEEMTER_UHPLC_SUB2UM")  # dp = 1.7 μm
    p_17um = item.column_backpressure_bar
    _, _, c_17um = item.van_deemter_coefficients

    # C-term should be reduced by (5.0 / 1.7)^2 ≈ 8.65x
    assert c_17um < c_5um / 6.0
    # Backpressure should be higher
    assert p_17um > p_5um


def test_chromatographic_resolution_baseline_criterion(qapp):
    """
    Verifies chromatographic resolution R_s:
    - Baseline resolution criterion: R_s >= 1.5
    - Co-elution failure: R_s < 1.0
    """
    item = InteractiveHPLCItem()
    # 1. Reverse-phase analgesics should have clean baseline separation (R_s > 1.5)
    item.load_preset("REVERSE_PHASE_ANALGESICS")
    assert item.chromatographic_resolution_rs >= 1.5

    # 2. Isomer co-elution should have insufficient resolution (R_s < 1.0)
    item.load_preset("ISOMER_CO_ELUTION_FAILURE")
    assert item.chromatographic_resolution_rs < 1.0


def test_all_five_presets_load(qapp):
    item = InteractiveHPLCItem()
    for preset_code, preset in HPLC_PRESETS.items():
        item.load_preset(preset_code)
        assert item.active_preset.code == preset_code
        assert item.column_length_mm == preset.column_length_mm
        assert item.particle_size_um == preset.particle_size_um
        assert item.flow_rate_ml_min == preset.flow_rate_ml_min
        assert item.theoretical_plates_n > 1000.0
        assert item.column_backpressure_bar > 0.0
        assert item.column_void_time_s > 0.0
        assert item.chromatographic_resolution_rs > 0.0


def test_parameter_steppers_and_toggles(qapp):
    item = InteractiveHPLCItem()
    init_f = item.flow_rate_ml_min
    item.step_flow_rate(0.4)
    assert math.isclose(item.flow_rate_ml_min, init_f + 0.4, rel_tol=1e-4)

    init_dp = item.particle_size_um
    item.step_particle_size(1.0)
    assert math.isclose(item.particle_size_um, init_dp + 1.0, rel_tol=1e-4)

    init_col = item.column_length_mm
    item.toggle_column_length()
    assert item.column_length_mm != init_col

    item.toggle_animation()
    assert item._is_running is False
    item.toggle_animation()
    assert item._is_running is True


def test_touchless_air_gestures_and_buttons(qapp):
    item = InteractiveHPLCItem()
    start_pos = item.pos()
    item.handle_air_drag(QPointF(30.0, 50.0))
    assert item.pos() == start_pos + QPointF(30.0, 50.0)

    # Air pinch on btn_f_up
    rect_f_up = item._buttons["btn_f_up"]
    scene_pt = item.mapToScene(rect_f_up.center())
    prev_f = item.flow_rate_ml_min
    handled = item.handle_air_pinch(scene_pt)
    assert handled is True
    assert item.flow_rate_ml_min > prev_f


def test_vector_diagram_generation_and_stamping(qapp):
    item = InteractiveHPLCItem()
    stamped = []
    item.stamp_requested.connect(lambda d: stamped.append(d))
    item.stamp_diagram()
    assert len(stamped) == 1
    assert "plates_n" in stamped[0]
    assert "backpressure_bar" in stamped[0]
    assert "resolution_rs" in stamped[0]

    items = ChemistryTools.create_hplc_diagram(
        center=(400.0, 300.0),
        width=680.0,
        height=380.0,
        preset_code="REVERSE_PHASE_ANALGESICS",
        col_length_mm=150.0,
        dp_um=5.0,
        flow_rate=1.0,
        plate_height_um=15.0,
        plates_n=10000.0,
        backpressure_bar=120.0,
        resolution_rs=2.2,
        void_time_s=90.0,
    )
    assert len(items) >= 15
    assert any("HPLC" in getattr(it, "toPlainText", lambda: "")() for it in items)


def test_lesson_board_hplc_lifecycle(qapp):
    board = LessonBoard()
    assert board.is_hplc_visible() is False
    assert board.get_hplc() is None

    # Show
    item = board.show_hplc(True)
    assert item is not None
    assert board.is_hplc_visible() is True
    assert board.get_hplc() is item
    assert board.get_hplc_item() is item

    # Toggle to hide
    visible = board.toggle_hplc()
    assert visible is False
    assert board.is_hplc_visible() is False
    assert board.get_hplc() is None
    assert board.get_hplc_item() is not None

    # Toggle to show
    visible = board.toggle_hplc()
    assert visible is True
    assert board.is_hplc_visible() is True
