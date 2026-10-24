"""
AirTeach AI — Unit Tests for Interactive Analytical Chemistry:
Acid-Base Buffer Systems, Henderson-Hasselbalch & Van Slyke Capacity (Milestone 139 / Task I.126).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_buffer_system import (
    InteractiveBufferSystemItem,
    BufferPreset,
    BUFFER_PRESETS,
)
from src.ui.stem_tools import ChemistryTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_henderson_hasselbalch_equimolar_and_ratio():
    """Verifies theoretical pH = pKa + log10([A-] / [HA]) across various ratios."""
    # Acetic acid buffer: pKa = 4.76
    pka = 4.76

    # 1. Equimolar ratio = 1.0 => log10(1) = 0 => pH = pKa
    ph_equi = pka + math.log10(1.0)
    assert math.isclose(ph_equi, 4.76, abs_tol=1e-5)

    # 2. Ratio = 10.0 => pH = pKa + 1.0 = 5.76
    ph_10 = pka + math.log10(10.0)
    assert math.isclose(ph_10, 5.76, abs_tol=1e-5)

    # 3. Ratio = 0.1 => pH = pKa - 1.0 = 3.76
    ph_01 = pka + math.log10(0.1)
    assert math.isclose(ph_01, 3.76, abs_tol=1e-5)


def test_van_slyke_buffer_capacity_maximum_at_pka(qapp):
    """
    Verifies that Van Slyke capacity achieves its local/global maximum
    at pH = pKa with beta_max = 0.575 * C_total.
    """
    item = InteractiveBufferSystemItem()
    item.set_preset("ACETIC_ACID_ACETATE_BUFFER")

    pka = item.pKa  # 4.76
    c_tot = item._c_total_M  # 0.20 M

    # Capacity at peak (pH = pKa)
    beta_peak = item.calculate_buffer_capacity_at_pH(pka)
    theoretical_max = 0.575 * c_tot

    assert math.isclose(beta_peak, theoretical_max, rel_tol=0.02), (
        f"Peak capacity {beta_peak:.4f} must match theoretical 0.575*C_total = {theoretical_max:.4f}"
    )

    # Off-peak capacities (pKa - 1.0 and pKa + 1.0) must be significantly lower (~33% of max)
    beta_low = item.calculate_buffer_capacity_at_pH(pka - 1.0)
    beta_high = item.calculate_buffer_capacity_at_pH(pka + 1.0)

    assert beta_low < beta_peak * 0.45, "Capacity 1 unit below pKa should drop by > 55%"
    assert beta_high < beta_peak * 0.45, "Capacity 1 unit above pKa should drop by > 55%"


def test_all_curriculum_presets_validity():
    """Verifies all 5 buffer presets have authentic physical constants and parameters."""
    expected_codes = [
        "ACETIC_ACID_ACETATE_BUFFER",
        "BLOOD_CARBONATE_PHYSIOLOGY",
        "PHOSPHATE_CELLULAR_BUFFER",
        "AMMONIA_AMMONIUM_ALKALINE",
        "UNBUFFERED_WATER_COMPARISON",
    ]
    for code in expected_codes:
        assert code in BUFFER_PRESETS, f"Preset {code} must be defined"
        preset = BUFFER_PRESETS[code]
        assert 0.0 < preset.pKa <= 14.0, "pKa must be within 0-14"
        assert preset.default_c_total_M >= 0.0, "Total concentration must be non-negative"
        assert len(preset.description) > 30, "Preset description must be thorough"

    # Physiological Blood Carbonate Buffer verification:
    blood = BUFFER_PRESETS["BLOOD_CARBONATE_PHYSIOLOGY"]
    # Ratio = 20:1, pKa = 6.10 => pH = 6.10 + log10(20) = 7.401
    blood_ph = blood.pKa + math.log10(blood.default_base_acid_ratio)
    assert 7.35 <= blood_ph <= 7.45, f"Physiological blood pH must be ~7.40, got {blood_ph:.3f}"


def test_perturbation_acid_and_base_addition(qapp):
    """Verifies that adding HCl decreases pH slightly, and adding NaOH increases pH slightly."""
    item = InteractiveBufferSystemItem()
    item.set_preset("ACETIC_ACID_ACETATE_BUFFER")

    initial_ph = item.current_pH
    assert math.isclose(initial_ph, 4.76, abs_tol=0.01)

    # Add 5 mM strong acid (HCl)
    item.add_strong_acid_drop(5.0)
    ph_after_acid = item.current_pH
    assert ph_after_acid < initial_ph, "Adding acid must lower pH"
    assert (initial_ph - ph_after_acid) < 0.10, "Buffer must resist acid; delta pH should be small"

    # Add 15 mM strong base (NaOH) -> Net +10 mM base
    item.add_strong_base_drop(15.0)
    ph_after_base = item.current_pH
    assert ph_after_base > initial_ph, "Net base addition must raise pH"

    # Reset
    item.reset_perturbations()
    assert math.isclose(item.current_pH, initial_ph, abs_tol=0.01)


def test_buffer_vs_water_stress_test(qapp):
    """Demonstrates that buffer resists pH shift orders of magnitude better than pure water."""
    item_buf = InteractiveBufferSystemItem()
    item_buf.set_preset("ACETIC_ACID_ACETATE_BUFFER")

    item_water = InteractiveBufferSystemItem()
    item_water.set_preset("UNBUFFERED_WATER_COMPARISON")

    # Add 10 mM strong acid to both
    item_buf.add_strong_acid_drop(10.0)
    item_water.add_strong_acid_drop(10.0)

    delta_buf = abs(item_buf.current_pH - 4.76)
    delta_water = abs(item_water.current_pH - 7.00)

    assert delta_buf < 0.15, f"Buffer delta pH must be small: {delta_buf:.3f}"
    assert delta_water > 3.0, f"Water delta pH must be massive (> 3.0 units): {delta_water:.3f}"
    # Resistance ratio
    assert (delta_water / delta_buf) > 20.0, "Buffer resistance should be > 20x stronger than water"


def test_speciation_fractions(qapp):
    """Verifies speciation fractions alpha_HA + alpha_A- = 1.0 and crossover at pH = pKa."""
    item = InteractiveBufferSystemItem()
    item.set_preset("PHOSPHATE_CELLULAR_BUFFER")  # pKa = 7.20

    # At pH = pKa (equimolar)
    alpha_ha = item.speciation_fraction_HA
    alpha_a = item.speciation_fraction_A

    assert math.isclose(alpha_ha + alpha_a, 1.0, rel_tol=1e-5), "Fractions must sum to 1.0"
    assert math.isclose(alpha_ha, 0.50, abs_tol=0.02), "At pH = pKa, alpha_HA must be ~0.50"
    assert math.isclose(alpha_a, 0.50, abs_tol=0.02), "At pH = pKa, alpha_A- must be ~0.50"


def test_universal_indicator_chromophore_colors(qapp):
    """Verifies universal indicator color transitions from acidic red to neutral green to basic violet."""
    item = InteractiveBufferSystemItem()

    col_acidic = item.get_indicator_color(2.0)
    col_neutral = item.get_indicator_color(7.0)
    col_basic = item.get_indicator_color(12.0)

    # Acidic should be reddish (high red, low blue)
    assert col_acidic.red() > col_acidic.blue()
    # Neutral should be green
    assert col_neutral.green() > col_neutral.red()
    # Basic should be purple/blue (high blue)
    assert col_basic.blue() > col_basic.green()


def test_interactive_adjustments_and_timer(qapp):
    """Tests ratio and concentration adjustments and animation tick."""
    item = InteractiveBufferSystemItem()
    item.set_preset("AMMONIA_AMMONIUM_ALKALINE")

    init_ratio = item._base_acid_ratio
    item.adjust_ratio(+0.4)
    assert item._base_acid_ratio == init_ratio + 0.4

    init_c = item._c_total_M
    item.adjust_total_concentration(+0.10)
    assert math.isclose(item._c_total_M, init_c + 0.10, rel_tol=1e-3)

    # Simulation tick
    init_phase = item._wave_phase
    item._on_sim_tick()
    assert item._wave_phase != init_phase


def test_paint_render_pipeline_no_crash(qapp):
    """Verifies that paint() renders all panels without runtime errors."""
    item = InteractiveBufferSystemItem()
    img = QImage(960, 650, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)

    painter = QPainter(img)
    try:
        item.paint(painter, None, None)
    finally:
        painter.end()

    assert not img.isNull()


def test_stamp_diagram_generation_and_lesson_board_integration(qapp):
    """Verifies ChemistryTools.create_buffer_system_diagram and LessonBoard integration."""
    board = LessonBoard()

    # 1. Show and toggle
    assert board.is_buffer_system_visible() is False
    item = board.show_buffer_system(True)
    assert item is not None
    assert board.is_buffer_system_visible() is True
    assert board.get_buffer_system() is item

    board.toggle_buffer_system()
    assert board.is_buffer_system_visible() is False

    # 2. Vector diagram stamping
    stamped_items = ChemistryTools.create_buffer_system_diagram(
        center=(400.0, 300.0),
        preset_code="ACETIC_ACID_ACETATE_BUFFER",
        acid_name="CH₃COOH",
        base_name="CH₃COO⁻",
        pKa=4.76,
        c_total_M=0.20,
        ratio=1.0,
        current_pH=4.76,
        van_slyke_capacity=0.115,
        beta_max=0.115,
    )
    assert len(stamped_items) >= 20, "Stamped diagram must produce full set of vector items"
    for itm in stamped_items:
        board.addItem(itm)

    # 3. Items in scene
    items_in_scene = board.get_items_in_rect(board.sceneRect())
    assert len(items_in_scene) >= len(stamped_items)


def test_main_window_toggle_buffer_system(qapp):
    """Verifies that MainWindow._on_toggle_buffer_system toggles visibility on scene."""
    from src.ui.main_window import MainWindow
    win = MainWindow()
    try:
        assert win._scene.is_buffer_system_visible() is False
        is_vis = win._on_toggle_buffer_system()
        assert is_vis is True
        assert win._scene.is_buffer_system_visible() is True

        is_vis = win._on_toggle_buffer_system()
        assert is_vis is False
        assert win._scene.is_buffer_system_visible() is False
    finally:
        win.close()
