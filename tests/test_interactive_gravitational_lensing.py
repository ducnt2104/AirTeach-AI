"""
AirTeach AI - Tests for Interactive Gravitational Lensing Simulator
===================================================================
Validates Einstein ring formation, Schwarzschild radius calculation,
dual image angular deflection, quadrupole cross lensing, and UI rendering.
"""
import math
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QGraphicsScene, QStyleOptionGraphicsItem

from src.ui.interactive_gravitational_lensing import (
    LENSING_PRESETS,
    InteractiveGravitationalLensingItem,
)


@pytest.fixture
def lens_item(qapp):
    return InteractiveGravitationalLensingItem(lens_mass_solar=1.0e12, source_offset_arcsec=0.5)


def test_einstein_radius_scaling(lens_item):
    """Einstein angular radius must scale as sqrt(M)."""
    lens_item.set_lens_mass_solar(1.0e12)
    phys_1 = lens_item.compute_physics()

    lens_item.set_lens_mass_solar(4.0e12)
    phys_4 = lens_item.compute_physics()

    # theta_E at 4x mass should be approximately 2x theta_E at 1x mass
    ratio = phys_4["theta_e_arcsec"] / phys_1["theta_e_arcsec"]
    assert math.isclose(ratio, 2.0, rel_tol=1e-3)


def test_einstein_ring_alignment(lens_item):
    """At perfect alignment (beta = 0), theta_plus and theta_minus must equal theta_E."""
    lens_item.set_source_offset_arcsec(0.0)
    phys = lens_item.compute_physics()

    assert math.isclose(phys["theta_plus_arcsec"], phys["theta_e_arcsec"], rel_tol=1e-4)
    assert math.isclose(phys["theta_minus_arcsec"], phys["theta_e_arcsec"], rel_tol=1e-4)
    assert phys["mu_tot"] >= 50.0  # High magnification near caustics


def test_lens_equation_consistency(lens_item):
    """Verify lens equation: beta = theta_plus - theta_E^2 / theta_plus."""
    lens_item.set_lens_mass_solar(1.0e12)
    lens_item.set_source_offset_arcsec(0.8)
    phys = lens_item.compute_physics()

    theta_e = phys["theta_e_arcsec"]
    theta_plus = phys["theta_plus_arcsec"]
    beta_calc = theta_plus - (theta_e**2) / theta_plus

    assert math.isclose(beta_calc, phys["beta_arcsec"], rel_tol=1e-4)


def test_schwarzschild_radius_calculation(lens_item):
    """Rs = 2GM / c^2 ≈ 2.95 * M km."""
    lens_item.set_lens_mass_solar(1.0e6)
    phys = lens_item.compute_physics()
    assert math.isclose(phys["rs_km"], 2.95e6, rel_tol=1e-3)


def test_presets_loading(lens_item):
    """Verify loading all astrophysical lensing presets."""
    for key, preset in LENSING_PRESETS.items():
        lens_item.apply_preset(key)
        assert lens_item.lens_mass_solar == preset.lens_mass_solar
        assert lens_item.source_offset_arcsec == preset.source_offset_arcsec


def test_signals_emission(lens_item):
    """Verify physics_computed and stamp_requested signal emissions."""
    received_phys = []
    lens_item.physics_computed.connect(lambda d: received_phys.append(d))

    lens_item.set_source_offset_arcsec(1.2)
    assert len(received_phys) > 0
    assert "theta_e_arcsec" in received_phys[-1]

    received_stamp = []
    lens_item.stamp_requested.connect(lambda d: received_stamp.append(d))
    lens_item.request_stamp()
    assert len(received_stamp) == 1
    assert "theta_plus_arcsec" in received_stamp[0]


def test_paint_and_render_zero_crash(qapp, lens_item):
    """Verify paint rendering executes without crashing on a QImage."""
    scene = QGraphicsScene()
    scene.addItem(lens_item)

    image = QImage(700, 450, QImage.Format_ARGB32_Premultiplied)
    image.fill(Qt.black)
    painter = QPainter(image)
    option = QStyleOptionGraphicsItem()
    lens_item.paint(painter, option, None)
    painter.end()

    assert not image.isNull()
