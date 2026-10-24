"""
Unit tests for InteractiveHardyWeinbergItem and BiologyTools.create_hardy_weinberg_diagram.
Verifies Hardy-Weinberg equilibrium invariants (p + q = 1, p² + 2pq + q² = 1),
Wright-Fisher genetic drift, natural selection with differential fitness,
sickle-cell heterozygote advantage balanced polymorphism, inbreeding coefficient F,
Chi-square goodness-of-fit test, touchless air gestures, and lesson board integration.
"""

import sys
import math
import pytest
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QColor

from src.ui.interactive_hardy_weinberg import (
    InteractiveHardyWeinbergItem,
    HW_PRESETS,
)
from src.ui.stem_tools import BiologyTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_hardy_weinberg_initialization(qapp):
    item = InteractiveHardyWeinbergItem()
    assert item.active_preset.code == "CLASSIC_IDEAL_EQUILIBRIUM"
    assert math.isclose(item.p, 0.70, rel_tol=1e-4)
    assert math.isclose(item.q, 0.30, rel_tol=1e-4)

    # Expected frequencies
    f_aa, f_het, f_rec = item.expected_hwe_frequencies
    assert math.isclose(f_aa, 0.49, rel_tol=1e-4)
    assert math.isclose(f_het, 0.42, rel_tol=1e-4)
    assert math.isclose(f_rec, 0.09, rel_tol=1e-4)

    assert item.is_in_hwe() is True
    assert item.chi_square_statistic < 0.01

    assert item.boundingRect().width() == InteractiveHardyWeinbergItem.CARD_W
    assert item.boundingRect().height() == InteractiveHardyWeinbergItem.CARD_H


def test_hardy_weinberg_algebraic_invariants(qapp):
    item = InteractiveHardyWeinbergItem()
    for test_p in [0.1, 0.25, 0.5, 0.73, 0.95]:
        item.step_p(test_p - item.p)
        assert math.isclose(item.p + item.q, 1.0, rel_tol=1e-5)

        f_aa, f_het, f_rec = item.expected_hwe_frequencies
        assert math.isclose(f_aa + f_het + f_rec, 1.0, rel_tol=1e-5)

        # Inbreeding adjusted
        for f_coeff in [0.0, 0.125, 0.25, 0.5]:
            item._inbreeding_f = f_coeff
            adj_aa, adj_het, adj_rec = item.inbreeding_adjusted_frequencies
            assert math.isclose(adj_aa + adj_het + adj_rec, 1.0, rel_tol=1e-5)


def test_sickle_cell_overdominance_convergence(qapp):
    item = InteractiveHardyWeinbergItem()
    item.load_preset("SICKLE_CELL_OVERDOMINANCE")

    # Overdominance formula for equilibrium p:
    # s1 = 1 - w_AA = 1 - 0.88 = 0.12
    # s2 = 1 - w_aa = 1 - 0.14 = 0.86
    # p_eq = s2 / (s1 + s2) = 0.86 / 0.98 ≈ 0.8776
    p_eq_theoretical = (1.0 - 0.14) / ((1.0 - 0.88) + (1.0 - 0.14))

    # Advance 40 generations
    for _ in range(40):
        item.advance_generation()

    assert math.isclose(item.p, p_eq_theoretical, rel_tol=0.08)


def test_recessive_lethal_selection_dynamics(qapp):
    item = InteractiveHardyWeinbergItem()
    item.load_preset("RECESSIVE_LETHAL_SELECTION")

    init_q = item.q
    for _ in range(20):
        item.advance_generation()

    # Deleterious recessive frequency must decrease
    assert item.q < init_q
    # But does not vanish entirely due to heterozygous carriers (2pq)
    assert item.q > 0.01


def test_genetic_drift_finite_population(qapp):
    item = InteractiveHardyWeinbergItem()
    item.load_preset("GENETIC_DRIFT_BOTTLENECK")
    assert item._pop_size == 25

    # Run 30 generations, tracking variance
    p_samples = []
    for _ in range(30):
        item.advance_generation()
        p_samples.append(item.p)

    # In a finite bottleneck population of 25, allele frequencies fluctuate stochastically
    assert len(set(p_samples)) > 1


def test_inbreeding_homozygosity_inflation(qapp):
    item = InteractiveHardyWeinbergItem()
    item.load_preset("INBREEDING_CONSANGUINITY_F")
    assert item._inbreeding_f == 0.25

    p = item.p
    q = item.q
    f_aa, f_het, f_rec = item.inbreeding_adjusted_frequencies

    # Inbreeding inflates homozygotes by F*p*q
    assert f_aa > p**2
    assert f_rec > q**2
    # And deflates heterozygotes by (1 - F)
    assert f_het < 2.0 * p * q
    assert math.isclose(f_het, 2.0 * p * q * 0.75, rel_tol=1e-4)


def test_chi_square_goodness_of_fit(qapp):
    item = InteractiveHardyWeinbergItem()
    item.load_preset("CLASSIC_IDEAL_EQUILIBRIUM")
    assert item.is_in_hwe() is True
    assert item.chi_square_statistic < 3.841

    # Force artificial deviation: observed all AA
    item._observed_counts = (100, 0, 0)
    assert item.chi_square_statistic > 3.841
    assert item.is_in_hwe() is False


def test_all_five_presets_load(qapp):
    item = InteractiveHardyWeinbergItem()
    preset_keys = [
        "CLASSIC_IDEAL_EQUILIBRIUM",
        "SICKLE_CELL_OVERDOMINANCE",
        "RECESSIVE_LETHAL_SELECTION",
        "GENETIC_DRIFT_BOTTLENECK",
        "INBREEDING_CONSANGUINITY_F",
    ]
    for key in preset_keys:
        item.load_preset(key)
        assert item.active_preset.code == key
        assert len(item.active_preset.title_vi) > 0
        assert 0.0 < item.p < 1.0
        assert item.mean_fitness > 0.0


def test_parameter_steppers_and_resets(qapp):
    item = InteractiveHardyWeinbergItem()
    init_p = item.p
    item.step_p(0.05)
    assert math.isclose(item.p, init_p + 0.05, rel_tol=1e-4)

    # Advance and reset
    item.advance_generation()
    assert item._generation == 1
    item.reset_simulation()
    assert item._generation == 0

    # Toggle animation
    init_run = item._is_running
    item.toggle_animation()
    assert item._is_running != init_run


def test_touchless_air_gestures_and_buttons(qapp):
    item = InteractiveHardyWeinbergItem(pos=(100.0, 80.0))
    init_pos = item.pos()
    item.handle_air_drag(QPointF(25.0, -15.0))
    assert item.pos().x() == init_pos.x() + 25.0
    assert item.pos().y() == init_pos.y() - 15.0

    # Pinch on btn_step_gen
    btn_step = item._buttons.get("btn_step_gen")
    assert btn_step is not None
    init_gen = item._generation
    scene_pt = item.mapToScene(btn_step.center())
    handled = item.handle_air_pinch(scene_pt)
    assert handled is True
    assert item._generation == init_gen + 1


def test_vector_diagram_generation_and_stamping(qapp):
    items = BiologyTools.create_hardy_weinberg_diagram(
        center=(340.0, 190.0),
        width=680.0,
        height=380.0,
        preset_code="CLASSIC_IDEAL_EQUILIBRIUM",
        generation=0,
        p=0.7,
        q=0.3,
        f_aa=0.49,
        f_het=0.42,
        f_rec=0.09,
        pop_size=0,
        chi_square=0.0,
        is_hwe=True,
    )
    assert len(items) >= 12
    bg = items[0]
    assert bg.rect().width() == 680.0
    assert bg.rect().height() == 380.0

    # Test stamp signal emission
    item = InteractiveHardyWeinbergItem()
    stamped = []
    item.stamp_requested.connect(lambda data: stamped.append(data))
    item.stamp_diagram()
    assert len(stamped) == 1
    assert stamped[0]["preset_code"] == "CLASSIC_IDEAL_EQUILIBRIUM"
    assert "chi_square" in stamped[0]


def test_lesson_board_hardy_weinberg_lifecycle(qapp):
    board = LessonBoard()
    assert not board.is_hardy_weinberg_visible()
    assert board.get_hardy_weinberg() is None

    # Show item
    item = board.show_hardy_weinberg(True, pos=(120.0, 80.0))
    assert item is not None
    assert board.is_hardy_weinberg_visible()
    assert board.get_hardy_weinberg() == item
    assert board.get_hardy_weinberg_item() == item

    # Toggle off
    res = board.toggle_hardy_weinberg()
    assert not res
    assert not board.is_hardy_weinberg_visible()
    assert board.get_hardy_weinberg() is None
    assert board.get_hardy_weinberg_item() is not None

    # Toggle on
    res2 = board.toggle_hardy_weinberg()
    assert res2
    assert board.is_hardy_weinberg_visible()
