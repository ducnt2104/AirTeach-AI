"""
AirTeach AI – Tests for Interactive Nuclear Physics, Binding Energy & Reactions Simulator (Milestone 113)
Verifies Einstein's mass defect (Δm), mass-energy equivalence (E = Δm·c²), Weizsäcker SEMF liquid drop model,
binding energy curve (Eb/A), D-T fusion, U-235 fission, p-p solar chain, air gestures, and LessonBoard integration.
"""

from __future__ import annotations

import math
import pytest
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

from src.ui.interactive_nuclear_reactions import (
    InteractiveNuclearReactionsItem,
    PRESETS,
    KNOWN_ISOTOPES,
    KNOWN_ISOTOPE_DICT,
    U_TO_MEV,
    U_TO_KG,
    MEV_TO_JOULE,
)
from src.ui.lesson_board import LessonBoard
from src.ui.stem_tools import PhysicsTools


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_nuclear_initialization(qapp):
    item = InteractiveNuclearReactionsItem(pos=(140.0, 50.0))
    try:
        assert item.pos().x() == 140.0
        assert item.pos().y() == 50.0
        assert item.boundingRect() == QRectF(0.0, 0.0, 700.0, 440.0)
        assert item.active_preset.code == "BINDING_ENERGY_CURVE"
        assert item.selected_isotope_symbol == "56Fe"
        assert item.atomic_number_z == 26
        assert item.mass_number_a == 56
        assert item.neutron_number_n == 30
        assert abs(item.current_binding_energy_per_nucleon - 8.790) < 0.01
        assert item.current_total_binding_energy_mev > 490.0
        assert item.current_mass_defect_u > 0.50
        assert item.current_mass_defect_kg > 8e-28
        assert item.is_animating is True
        assert item.is_reacting is False
    finally:
        item.cleanup()


def test_weizsaecker_semf_physics(qapp):
    """Verifies the 5 mathematical terms of the Liquid Drop Model (SEMF)."""
    # Test Iron-56 (Z=26, A=56)
    e_vol, e_surf, e_coul, e_asym, e_pair, b_total = (
        InteractiveNuclearReactionsItem.calculate_semf_binding_energy(56, 26)
    )

    # 1. Volume term (+a_v * A) must be positive and large
    assert e_vol > 0.0
    assert abs(e_vol - (15.75 * 56)) < 1e-4

    # 2. Surface term (-a_s * A^(2/3)) must be negative
    assert e_surf < 0.0

    # 3. Coulomb term (-a_c * Z*(Z-1) / A^(1/3)) must be negative
    assert e_coul < 0.0

    # 4. Asymmetry term must be zero or negative
    assert e_asym <= 0.0

    # 5. Pairing term: Even-even nucleus (Z=26, N=30) must be positive
    assert e_pair > 0.0

    # Total binding energy must yield realistic Eb/A ~ 8.5 - 8.9 MeV
    eb_a = b_total / 56.0
    assert 8.4 <= eb_a <= 9.0

    # Test odd-odd nucleus has negative pairing term
    # Nitrogen-14 (Z=7, N=7, A=14)
    _, _, _, _, e_pair_oo, _ = (
        InteractiveNuclearReactionsItem.calculate_semf_binding_energy(14, 7)
    )
    assert e_pair_oo < 0.0

    # Test odd-A nucleus has zero pairing term
    # Carbon-13 (Z=6, N=7, A=13)
    _, _, _, _, e_pair_oa, _ = (
        InteractiveNuclearReactionsItem.calculate_semf_binding_energy(13, 6)
    )
    assert e_pair_oa == 0.0

    # Boundary test for hydrogen-1 (A=1)
    v1, s1, c1, a1, p1, b1 = (
        InteractiveNuclearReactionsItem.calculate_semf_binding_energy(1, 1)
    )
    assert b1 == 0.0


def test_mass_defect_and_einstein_equivalence(qapp):
    item = InteractiveNuclearReactionsItem()
    try:
        item.select_isotope("4He")
        assert item.mass_number_a == 4
        assert item.atomic_number_z == 2

        # Alpha particle Eb/A ≈ 7.074 MeV, Eb ≈ 28.3 MeV
        eb_tot = item.current_total_binding_energy_mev
        assert abs(eb_tot - 28.296) < 0.1

        # Δm = Eb / 931.494 u
        dm_u = item.current_mass_defect_u
        assert abs(dm_u - (eb_tot / U_TO_MEV)) < 1e-6
        assert 0.029 <= dm_u <= 0.032

        # Energy in Joules
        e_j = item.current_total_binding_energy_joules
        assert abs(e_j - (eb_tot * MEV_TO_JOULE)) < 1e-18
        assert e_j > 4.0e-12  # ~4.5 pJ
    finally:
        item.cleanup()


def test_all_curriculum_presets(qapp):
    item = InteractiveNuclearReactionsItem()
    try:
        preset_keys = [
            "BINDING_ENERGY_CURVE",
            "DEUTERIUM_TRITIUM_FUSION",
            "URANIUM_235_FISSION",
            "PROTON_PROTON_SOLAR_CHAIN",
            "SEMF_LIQUID_DROP_DECOMPOSITION",
        ]
        for key in preset_keys:
            item.load_preset(key)
            assert item.active_preset.code == key
            assert len(item.active_preset.title_vi) > 0
            assert len(item.active_preset.description) > 0
            assert item.is_reacting is False
    finally:
        item.cleanup()


def test_parameter_controls_and_steppers(qapp):
    item = InteractiveNuclearReactionsItem()
    try:
        # A and Z setters
        item.set_a_and_z(235, 92)
        assert item.mass_number_a == 235
        assert item.atomic_number_z == 92
        assert item.neutron_number_n == 143
        assert item.selected_isotope_symbol == "235U"

        # Plasma temp setter & clamping
        item.set_fusion_temp_mk(150.0)
        assert item.fusion_plasma_temp_mk == 150.0
        item.set_fusion_temp_mk(5.0)  # Below min 10
        assert item.fusion_plasma_temp_mk == 10.0
        item.set_fusion_temp_mk(500.0)  # Above max 300
        assert item.fusion_plasma_temp_mk == 300.0

        # Fission k factor setter & clamping
        item.set_fission_k_factor(1.05)
        assert abs(item.fission_k_factor - 1.05) < 1e-4
        item.set_fission_k_factor(0.5)  # Below min 0.8
        assert abs(item.fission_k_factor - 0.80) < 1e-4
        item.set_fission_k_factor(2.0)  # Above max 1.25
        assert abs(item.fission_k_factor - 1.25) < 1e-4
    finally:
        item.cleanup()


def test_reaction_trigger_and_animation(qapp):
    item = InteractiveNuclearReactionsItem()
    try:
        signals_received = []
        item.reaction_triggered.connect(lambda mode: signals_received.append(mode))

        item.load_preset("DEUTERIUM_TRITIUM_FUSION")
        assert item.is_reacting is False

        item.trigger_reaction()
        assert item.is_reacting is True
        assert len(signals_received) == 1
        assert signals_received[0] == "DEUTERIUM_TRITIUM_FUSION"

        # Advance tick
        for _ in range(60):
            item._on_tick()
        assert item._reaction_phase > 0.5

        # Reset
        item.reset_reaction()
        assert item.is_reacting is False
        assert item._reaction_phase == 0.0

        # Toggle animation
        assert item.is_animating is True
        item.toggle_animation()
        assert item.is_animating is False
        item.toggle_animation()
        assert item.is_animating is True
    finally:
        item.cleanup()


def test_touchless_air_gestures(qapp):
    item = InteractiveNuclearReactionsItem(pos=(0.0, 0.0))
    try:
        # Pre-render to build button rects cache
        img = QImage(700, 440, QImage.Format.Format_ARGB32)
        p = QPainter(img)
        item.paint(p, None, None)
        p.end()

        # 1. Preset tabs
        r_fus = item._btn_rects.get("preset_DEUTERIUM_TRITIUM_FUSION")
        assert r_fus is not None
        assert item.handle_air_pinch(r_fus.center()) is True
        assert item.active_preset.code == "DEUTERIUM_TRITIUM_FUSION"

        # 2. Trigger reaction
        r_trig = item._btn_rects.get("trigger")
        assert r_trig is not None
        assert item.handle_air_pinch(r_trig.center()) is True
        assert item.is_reacting is True

        # 3. Reset reaction
        r_rst = item._btn_rects.get("reset")
        assert r_rst is not None
        assert item.handle_air_pinch(r_rst.center()) is True
        assert item.is_reacting is False

        # 4. A steppers
        r_au = item._btn_rects.get("a_up")
        cur_a = item.mass_number_a
        assert item.handle_air_pinch(r_au.center()) is True
        assert item.mass_number_a == cur_a + 1

        # 5. Out of bounds returns False
        assert item.handle_air_pinch(QPointF(-100.0, -100.0)) is False
    finally:
        item.cleanup()


def test_paint_rendering_all_presets(qapp):
    item = InteractiveNuclearReactionsItem()
    try:
        img = QImage(700, 440, QImage.Format.Format_ARGB32)
        for preset_key in PRESETS:
            item.load_preset(preset_key)
            item.trigger_reaction()
            # Advance reaction phase midway
            item._reaction_phase = 0.5

            painter = QPainter(img)
            item.paint(painter, None, None)
            painter.end()
            assert not img.isNull()
    finally:
        item.cleanup()


def test_stem_tools_vector_diagram(qapp):
    presets_to_test = [
        ("BINDING_ENERGY_CURVE", "56Fe", 8.790, 0.528, 0.0),
        ("DEUTERIUM_TRITIUM_FUSION", "4He", 7.074, 0.030, 17.59),
        ("URANIUM_235_FISSION", "235U", 7.591, 1.915, 200.0),
    ]

    for p_code, iso, eb_a, dm, q in presets_to_test:
        items = PhysicsTools.create_nuclear_reactions_diagram(
            center=(400.0, 300.0),
            width=640.0,
            height=340.0,
            preset_code=p_code,
            selected_isotope=iso,
            eb_per_nucleon=eb_a,
            mass_defect_u=dm,
            q_value_mev=q,
        )
        assert len(items) >= 5
        # Ensure first item is background rect
        assert items[0].rect().width() == 640.0
        assert items[0].rect().height() == 340.0


def test_lesson_board_integration(qapp):
    board = LessonBoard()
    try:
        assert board.is_nuclear_reactions_visible() is False
        assert board.get_nuclear_reactions() is None

        # Toggle ON
        vis1 = board.toggle_nuclear_reactions()
        assert vis1 is True
        assert board.is_nuclear_reactions_visible() is True
        item = board.get_nuclear_reactions()
        assert item is not None
        assert isinstance(item, InteractiveNuclearReactionsItem)

        # Excluded from stroke selection
        excluded = board.get_items_in_rect(QRectF(0.0, 0.0, 1000.0, 1000.0))
        assert item not in excluded

        # Toggle OFF
        vis2 = board.toggle_nuclear_reactions()
        assert vis2 is False
        assert board.is_nuclear_reactions_visible() is False
        assert board.get_nuclear_reactions() is None
        assert board.get_nuclear_reactions_item() is not None
    finally:
        item = board.get_nuclear_reactions_item()
        if item is not None and hasattr(item, "cleanup"):
            item.cleanup()
