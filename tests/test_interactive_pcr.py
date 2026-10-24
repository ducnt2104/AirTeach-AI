"""
Unit tests for InteractivePCRItem and BiologyTools.create_pcr_diagram.
Verifies 3-step thermal cycling, exponential DNA amplification N_n = N_0·(1+E)^n,
cycle threshold C_t, logistic plateau effect, annealing temperature penalties,
agarose gel electrophoresis band migration, touchless air gestures, and lesson board integration.
"""

import sys
import math
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from src.ui.interactive_pcr import (
    InteractivePCRItem,
    PCR_PRESETS,
)
from src.ui.stem_tools import BiologyTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_pcr_initialization(qapp):
    item = InteractivePCRItem()
    assert item.active_preset.code == "COVID19_RTPCR_DIAGNOSTICS"
    assert item.active_preset.amplicon_size_bp == 150
    assert item.cycle_number == 25
    assert math.isclose(item.annealing_temp_c, 58.0, rel_tol=1e-4)
    assert item.dna_copies > 1e7
    assert item.boundingRect().width() == InteractivePCRItem.CARD_W
    assert item.boundingRect().height() == InteractivePCRItem.CARD_H


def test_exponential_amplification_and_logistic_plateau(qapp):
    """
    Verifies exponential phase amplification and plateau behavior:
    - In early cycles (n <= 15), copies scale as N_0 * (1 + E)^n
    - Strictly monotonic: N(n+1) > N(n)
    - Upper bound clamped at MAX_COPIES_PLATEAU
    """
    item = InteractivePCRItem()
    item.load_preset("FORENSIC_STR_DNA_PROFILING")  # N_0 = 200, E = 0.95
    n0 = item.active_preset.initial_copies
    e = item.efficiency_factor

    # Early cycle check at n = 10
    c10 = item.calc_copies_at_cycle(10)
    expected_c10 = n0 * ((1.0 + e) ** 10)
    assert math.isclose(c10, expected_c10, rel_tol=0.05)

    # Monotonicity check
    prev_copies = 0.0
    for n in range(1, 35, 3):
        curr_copies = item.calc_copies_at_cycle(n)
        assert curr_copies > prev_copies
        prev_copies = curr_copies

    # Plateau saturation check
    c40 = item.calc_copies_at_cycle(40)
    assert c40 <= InteractivePCRItem.MAX_COPIES_PLATEAU


def test_cycle_threshold_ct_calculation(qapp):
    """
    Verifies Cycle Threshold C_t where copies reach DETECTION_THRESHOLD (10^7):
    High initial copy N_0 yields earlier C_t than low initial copy N_0.
    """
    item = InteractivePCRItem()

    # Covid (N_0 = 10,000) vs Forensic (N_0 = 200)
    item.load_preset("COVID19_RTPCR_DIAGNOSTICS")
    ct_covid = item.cycle_threshold_ct

    item.load_preset("FORENSIC_STR_DNA_PROFILING")
    ct_forensic = item.cycle_threshold_ct

    assert ct_covid < ct_forensic
    assert 8.0 < ct_covid < 14.0
    assert 14.0 < ct_forensic < 22.0


def test_annealing_temperature_efficiency_penalty(qapp):
    """
    Verifies that deviating from optimal annealing temperature reduces effective efficiency.
    """
    item = InteractivePCRItem()
    item.load_preset("COVID19_RTPCR_DIAGNOSTICS")  # Optimal Ta = 58°C
    e_opt = item.efficiency_factor

    # Deviate by +5°C
    item.step_annealing_temp(5.0)
    e_penalized = item.efficiency_factor

    assert e_penalized < e_opt
    assert e_penalized >= 0.50


def test_all_five_presets_load(qapp):
    item = InteractivePCRItem()
    for preset_code, preset in PCR_PRESETS.items():
        item.load_preset(preset_code)
        assert item.active_preset.code == preset_code
        assert item.active_preset.amplicon_size_bp == preset.amplicon_size_bp
        assert len(item.active_preset.ladder_bands) >= 5
        assert item.dna_copies > 0.0
        assert item.cycle_threshold_ct > 0.0


def test_parameter_steppers_and_phases(qapp):
    item = InteractivePCRItem()
    init_n = item.cycle_number
    item.step_cycles(5)
    assert item.cycle_number == init_n + 5

    init_ta = item.annealing_temp_c
    item.step_annealing_temp(-2.0)
    assert math.isclose(item.annealing_temp_c, init_ta - 2.0, rel_tol=1e-4)

    item.toggle_animation()
    assert item._is_running is False
    item.toggle_animation()
    assert item._is_running is True


def test_touchless_air_gestures_and_buttons(qapp):
    item = InteractivePCRItem()
    start_pos = item.pos()
    item.handle_air_drag(QPointF(25.0, 40.0))
    assert item.pos() == start_pos + QPointF(25.0, 40.0)

    # Air pinch on btn_n_up
    rect_n_up = item._buttons["btn_n_up"]
    scene_pt = item.mapToScene(rect_n_up.center())
    prev_n = item.cycle_number
    handled = item.handle_air_pinch(scene_pt)
    assert handled is True
    assert item.cycle_number > prev_n


def test_vector_diagram_generation_and_stamping(qapp):
    item = InteractivePCRItem()
    stamped = []
    item.stamp_requested.connect(lambda d: stamped.append(d))
    item.stamp_diagram()
    assert len(stamped) == 1
    assert "amplicon_bp" in stamped[0]
    assert "dna_copies" in stamped[0]
    assert "ct_threshold" in stamped[0]

    items = BiologyTools.create_pcr_diagram(
        center=(400.0, 300.0),
        width=680.0,
        height=380.0,
        preset_code="COVID19_RTPCR_DIAGNOSTICS",
        target_gene="Gen N nucleocapsid",
        amplicon_bp=150,
        cycle_n=25,
        dna_copies=2.5e8,
        ct_threshold=18.5,
        efficiency=0.98,
        annealing_temp_c=58.0,
    )
    assert len(items) >= 15
    assert any("PCR" in getattr(it, "toPlainText", lambda: "")() for it in items)


def test_lesson_board_pcr_lifecycle(qapp):
    board = LessonBoard()
    assert board.is_pcr_visible() is False
    assert board.get_pcr() is None

    # Show
    item = board.show_pcr(True)
    assert item is not None
    assert board.is_pcr_visible() is True
    assert board.get_pcr() is item
    assert board.get_pcr_item() is item

    # Toggle to hide
    visible = board.toggle_pcr()
    assert visible is False
    assert board.is_pcr_visible() is False
    assert board.get_pcr() is None
    assert board.get_pcr_item() is not None

    # Toggle to show
    visible = board.toggle_pcr()
    assert visible is True
    assert board.is_pcr_visible() is True
