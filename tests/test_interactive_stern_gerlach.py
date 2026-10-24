"""
AirTeach AI — Unit Tests for Interactive Physics & Quantum Mechanics:
Stern-Gerlach Experiment, Space Quantization & Electron Spin 1/2 (Milestone 140 / Task I.127).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_stern_gerlach import (
    InteractiveSternGerlachItem,
    SternGerlachPreset,
    STERN_GERLACH_PRESETS,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_stern_gerlach_force_and_deflection(qapp):
    """Verifies theoretical magnetic deflection force F_z = mu_B * (dB_z / dz)."""
    item = InteractiveSternGerlachItem()
    item.set_preset("SINGLE_STAGE_SILVER_ATOMS")

    mu_b = item.BOHR_MAGNETON_J_PER_T  # 9.274010e-24 J/T
    grad = item._gradient_T_per_m       # 1200 T/m

    expected_force = mu_b * grad
    assert math.isclose(item.magnetic_deflection_force_N, expected_force, rel_tol=1e-5)
    assert 1.0e-20 < item.magnetic_deflection_force_N < 2.0e-20

    # Physical deflection on detector screen should be in the millimeter range
    defl_mm = item.beam_deflection_mm
    assert 0.5 <= defl_mm <= 10.0, f"Deflection {defl_mm:.2f} mm must be in observable millimeter range"


def test_quantum_probability_malus_spin_law(qapp):
    """Verifies quantum projection probabilities P(up) = cos^2(theta/2) and P(down) = sin^2(theta/2)."""
    item = InteractiveSternGerlachItem()

    # 1. theta = 0 deg => P(up) = 1.0, P(down) = 0.0
    item.set_magnet_angle(0.0)
    assert math.isclose(item.quantum_prob_spin_up, 1.0, abs_tol=1e-5)
    assert math.isclose(item.quantum_prob_spin_down, 0.0, abs_tol=1e-5)

    # 2. theta = 60 deg => P(up) = cos^2(30 deg) = 0.75, P(down) = sin^2(30 deg) = 0.25
    item.set_magnet_angle(60.0)
    assert math.isclose(item.quantum_prob_spin_up, 0.75, abs_tol=1e-5)
    assert math.isclose(item.quantum_prob_spin_down, 0.25, abs_tol=1e-5)

    # 3. theta = 90 deg => P(up) = 0.50, P(down) = 0.50
    item.set_magnet_angle(90.0)
    assert math.isclose(item.quantum_prob_spin_up, 0.50, abs_tol=1e-5)
    assert math.isclose(item.quantum_prob_spin_down, 0.50, abs_tol=1e-5)

    # 4. theta = 180 deg => P(up) = 0.0, P(down) = 1.0
    item.set_magnet_angle(180.0)
    assert math.isclose(item.quantum_prob_spin_up, 0.0, abs_tol=1e-5)
    assert math.isclose(item.quantum_prob_spin_down, 1.0, abs_tol=1e-5)

    # Probability conservation sum P(up) + P(down) == 1.0 for all angles
    for ang in range(0, 181, 15):
        item.set_magnet_angle(float(ang))
        assert math.isclose(item.quantum_prob_spin_up + item.quantum_prob_spin_down, 1.0, abs_tol=1e-5)


def test_all_curriculum_presets_validity():
    """Verifies that all 5 presets contain authentic quantum physics parameters."""
    expected_codes = [
        "SINGLE_STAGE_SILVER_ATOMS",
        "CLASSICAL_SMUDGE_VS_QUANTUM_SPLIT",
        "SEQUENTIAL_SGZ_SGX_SGZ_PARADOX",
        "ARBITRARY_MAGNET_ANGLE_THETA",
        "MAGNETIC_GRADIENT_DEFLECTION_SCALING",
    ]
    for code in expected_codes:
        assert code in STERN_GERLACH_PRESETS, f"Preset {code} must be defined"
        preset = STERN_GERLACH_PRESETS[code]
        assert preset.gradient_T_per_m >= 200.0, "Field gradient must be >= 200 T/m"
        assert preset.oven_temperature_K >= 300.0, "Oven temperature must be physical"
        assert len(preset.description) > 40, "Preset description must be detailed"


def test_deflection_scaling_with_gradient_and_temperature(qapp):
    """
    Verifies that deflection increases with higher gradient dB/dz
    and increases with lower oven temperature (slower atoms spend more time in field).
    """
    item = InteractiveSternGerlachItem()
    item.set_preset("SINGLE_STAGE_SILVER_ATOMS")

    initial_defl = item.beam_deflection_mm

    # Increase gradient from 1200 to 2200 T/m
    item.adjust_gradient(+1000.0)
    defl_high_grad = item.beam_deflection_mm
    assert defl_high_grad > initial_defl, "Higher gradient must produce larger deflection"

    # Reset gradient, lower temperature (atoms move slower, spend more time in B field)
    item.adjust_gradient(-1000.0)
    item.adjust_temperature(-400.0)
    defl_low_temp = item.beam_deflection_mm
    assert defl_low_temp > initial_defl, "Lower temperature (slower velocity) must increase deflection"


def test_classical_and_sequential_mode_toggles(qapp):
    """Tests toggling between classical continuous smudge and 3-stage sequential quantum paradox."""
    item = InteractiveSternGerlachItem()

    # Initial state
    assert item._is_classical_mode is False
    assert item._is_sequential_mode is False

    # Toggle classical mode
    item.toggle_classical_comparison()
    assert item._is_classical_mode is True

    item.toggle_classical_comparison()
    assert item._is_classical_mode is False

    # Toggle sequential mode
    item.toggle_sequential_mode()
    assert item._is_sequential_mode is True

    item.toggle_sequential_mode()
    assert item._is_sequential_mode is False


def test_particle_beam_animation_ticks(qapp):
    """Verifies that animation timer spawns and advances beam particles along trajectories."""
    item = InteractiveSternGerlachItem()
    item._particles.clear()

    # Run several ticks
    for _ in range(10):
        item._on_sim_tick()

    assert len(item._particles) > 0, "Particles must be spawned during simulation ticks"
    assert all(0.0 <= p["progress"] <= 1.05 for p in item._particles)


def test_paint_render_pipeline_no_crash(qapp):
    """Verifies that paint() renders both 1-stage and 3-stage sequential modes cleanly."""
    item = InteractiveSternGerlachItem()
    img = QImage(960, 650, QImage.Format.Format_ARGB32_Premultiplied)
    img.fill(Qt.GlobalColor.black)

    painter = QPainter(img)
    try:
        # Standard 1-stage
        item.paint(painter, None, None)

        # 3-stage sequential mode
        item.toggle_sequential_mode()
        item.paint(painter, None, None)

        # Classical smudge mode
        item.toggle_classical_comparison()
        item.paint(painter, None, None)
    finally:
        painter.end()

    assert not img.isNull()


def test_stamp_diagram_generation_and_lesson_board_integration(qapp):
    """Verifies PhysicsTools.create_stern_gerlach_diagram and LessonBoard integration."""
    board = LessonBoard()

    # 1. Show and toggle
    assert board.is_stern_gerlach_visible() is False
    item = board.show_stern_gerlach(True)
    assert item is not None
    assert board.is_stern_gerlach_visible() is True
    assert board.get_stern_gerlach() is item

    board.toggle_stern_gerlach()
    assert board.is_stern_gerlach_visible() is False

    # 2. Vector diagram stamping
    stamped_items = PhysicsTools.create_stern_gerlach_diagram(
        center=(400.0, 300.0),
        preset_code="SINGLE_STAGE_SILVER_ATOMS",
        gradient_T_per_m=1200.0,
        deflection_mm=2.15,
        p_up=1.0,
        p_down=1.0,
        is_classical=False,
    )
    assert len(stamped_items) >= 20, "Stamped diagram must produce full set of vector items"
    for itm in stamped_items:
        board.addItem(itm)

    # 3. Items in scene
    items_in_scene = board.get_items_in_rect(board.sceneRect())
    assert len(items_in_scene) >= len(stamped_items)


def test_main_window_toggle_stern_gerlach(qapp):
    """Verifies that MainWindow._on_toggle_stern_gerlach toggles visibility on scene."""
    from src.ui.main_window import MainWindow
    win = MainWindow()
    try:
        assert win._scene.is_stern_gerlach_visible() is False
        is_vis = win._on_toggle_stern_gerlach()
        assert is_vis is True
        assert win._scene.is_stern_gerlach_visible() is True

        is_vis = win._on_toggle_stern_gerlach()
        assert is_vis is False
        assert win._scene.is_stern_gerlach_visible() is False
    finally:
        win.close()
