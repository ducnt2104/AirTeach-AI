"""
AirTeach AI - Tests for Interactive Compton Scattering Simulation
================================================================
Validates relativistic photon-electron scattering, Compton shift formula,
energy-momentum conservation, Klein-Nishina cross section, and UI interactions.
"""
import math
import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QGraphicsScene, QStyleOptionGraphicsItem

from src.ui.interactive_compton_scattering import (
    COMPTON_PRESETS,
    COMPTON_WAVELENGTH_PM,
    ELECTRON_REST_ENERGY_KEV,
    InteractiveComptonScatteringItem,
)


@pytest.fixture
def compton_item(qapp):
    return InteractiveComptonScatteringItem(incident_energy_kev=511.0, scattering_angle_deg=90.0)


def test_compton_wavelength_shift_90deg(compton_item):
    """At 90 degrees, delta_lambda should exactly equal the Compton wavelength."""
    compton_item.set_scattering_angle_deg(90.0)
    phys = compton_item.compute_physics()
    assert math.isclose(phys["delta_lambda_pm"], COMPTON_WAVELENGTH_PM, rel_tol=1e-4)


def test_compton_wavelength_shift_180deg(compton_item):
    """At 180 degrees (backscatter), delta_lambda should equal 2 * lambda_C."""
    compton_item.set_scattering_angle_deg(180.0)
    phys = compton_item.compute_physics()
    expected = 2.0 * COMPTON_WAVELENGTH_PM
    assert math.isclose(phys["delta_lambda_pm"], expected, rel_tol=1e-4)


def test_compton_wavelength_shift_0deg(compton_item):
    """At 0 degrees, delta_lambda should equal 0."""
    compton_item.set_scattering_angle_deg(0.0)
    phys = compton_item.compute_physics()
    assert math.isclose(phys["delta_lambda_pm"], 0.0, abs_tol=1e-6)


def test_energy_conservation(compton_item):
    """Incident photon energy must equal scattered photon energy + electron kinetic energy."""
    for angle in [15.0, 45.0, 90.0, 135.0, 180.0]:
        for energy in [17.44, 100.0, 511.0, 1250.0]:
            compton_item.set_incident_energy_kev(energy)
            compton_item.set_scattering_angle_deg(angle)
            phys = compton_item.compute_physics()
            total_final = phys["energy_scatter_kev"] + phys["electron_kinetic_kev"]
            assert math.isclose(total_final, energy, rel_tol=1e-4)


def test_recoil_electron_angle(compton_item):
    """Test electron recoil angle phi calculation."""
    compton_item.set_incident_energy_kev(511.0)
    compton_item.set_scattering_angle_deg(90.0)
    phys = compton_item.compute_physics()
    # At 90 deg, theta/2 = 45 deg, tan(45) = 1.
    # cot phi = 1 + E / m_e = 1 + 511 / 510.999 ≈ 2.0
    # tan phi ≈ 0.5 -> phi ≈ 26.56 degrees
    assert 25.0 < phys["phi_deg"] < 28.0


def test_klein_nishina_cross_section(compton_item):
    """Verify Klein-Nishina differential cross-section factor is positive and finite."""
    for angle in [0.0, 30.0, 60.0, 90.0, 120.0, 180.0]:
        compton_item.set_scattering_angle_deg(angle)
        phys = compton_item.compute_physics()
        assert phys["kn_factor"] > 0.0
        assert math.isfinite(phys["kn_factor"])


def test_presets_application(compton_item):
    """Test applying all experimental presets."""
    for key, preset in COMPTON_PRESETS.items():
        compton_item.apply_preset(key)
        assert math.isclose(compton_item.incident_energy_kev, preset.incident_energy_kev)
        assert math.isclose(compton_item.scattering_angle_deg, preset.scattering_angle_deg)


def test_rendering_and_paint(qapp, compton_item):
    """Verify paint executes cleanly onto a QImage without throwing exceptions."""
    scene = QGraphicsScene()
    scene.addItem(compton_item)

    image = QImage(650, 450, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        opt = QStyleOptionGraphicsItem()
        compton_item.paint(painter, opt, None)
    finally:
        painter.end()

    # Advance animation
    compton_item.advance_animation(0.2)
    assert compton_item._anim_phase > 0.0
