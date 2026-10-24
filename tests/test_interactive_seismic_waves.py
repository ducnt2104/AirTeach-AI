"""
AirTeach AI — Unit Tests for Interactive Earth Sciences & Geophysics:
Seismic Waves, P-Wave / S-Wave Travel-Time Curves & Earthquake Epicenter Triangulation Simulator (Milestone 144 / Task I.131).
"""

import math
import pytest
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QImage

from src.ui.interactive_seismic_waves import (
    InteractiveSeismicWavesItem,
    SeismicPreset,
    SEISMIC_PRESETS,
    DEFAULT_STATIONS,
)
from src.ui.stem_tools import PhysicsTools
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_seismic_presets_and_parameters():
    """Verifies that all 5 curriculum presets are properly defined with physical parameters."""
    expected_codes = [
        "TOHOKU_2011_MEGATHRUST",
        "SAN_ANDREAS_TRANSVERSE_SLIP",
        "HIMALAYAN_COLLISION_DEEP",
        "MOONQUAKE_APOLLO_SEISMIC",
        "EARTH_OUTER_CORE_SHADOW_ZONE",
    ]
    for code in expected_codes:
        assert code in SEISMIC_PRESETS
        preset = SEISMIC_PRESETS[code]
        assert preset.p_wave_speed_km_s > 0.0
        assert preset.s_wave_speed_km_s > 0.0
        assert preset.p_wave_speed_km_s > preset.s_wave_speed_km_s  # P-wave is always faster
        assert preset.magnitude_mw > 0.0
        assert preset.depth_km >= 0.0
        assert len(preset.title_vi) > 0
        assert len(preset.event_name) > 0


def test_p_and_s_wave_travel_times_and_sp_lag(qapp):
    """
    Verifies that:
    1. t_p = d / v_p
    2. t_s = d / v_s
    3. Δt_{S-P} = t_s - t_p = d * (v_p - v_s) / (v_p * v_s)
    4. Distance can be recovered via Omori formula: d = Δt_{S-P} * k.
    """
    item = InteractiveSeismicWavesItem()
    item.set_preset("TOHOKU_2011_MEGATHRUST")
    st = DEFAULT_STATIONS[0]

    d = item.station_distance_km(st)
    tp = item.p_arrival_time_s(st)
    ts = item.s_arrival_time_s(st)
    sp_lag = item.sp_time_lag_s(st)

    assert d > 0.0
    assert tp > 0.0
    assert ts > tp  # S wave arrives strictly after P wave
    assert math.isclose(sp_lag, ts - tp, rel_tol=1e-4)

    # Recover distance using Omori formula
    k = item.omori_factor_km_per_s
    recovered_d = sp_lag * k
    assert math.isclose(recovered_d, d, rel_tol=1e-4)


def test_3_station_distance_consistency(qapp):
    """Verifies that distance to 3 distinct stations reflects relative geometric positions."""
    item = InteractiveSeismicWavesItem()
    item.set_preset("TOHOKU_2011_MEGATHRUST")

    st_a = DEFAULT_STATIONS[0]
    st_b = DEFAULT_STATIONS[1]
    st_c = DEFAULT_STATIONS[2]

    dA = item.station_distance_km(st_a)
    dB = item.station_distance_km(st_b)
    dC = item.station_distance_km(st_c)

    assert dA > 0.0
    assert dB > 0.0
    assert dC > 0.0
    # The 3 stations have distinct coordinates, distances should be distinct
    assert not math.isclose(dA, dB, abs_tol=1.0) or not math.isclose(dB, dC, abs_tol=1.0)


def test_interactive_controls_and_buttons(qapp):
    """Verifies epicenter position adjustments, wave speeds, and resets."""
    item = InteractiveSeismicWavesItem()
    item.set_preset("SAN_ANDREAS_TRANSVERSE_SLIP")
    init_x = item._epicenter_x
    init_y = item._epicenter_y
    init_vp = item._vp

    # X adjustment
    item._on_button_clicked("btn_x_up")
    assert item._epicenter_x == init_x + 20.0
    item._on_button_clicked("btn_x_down")
    assert item._epicenter_x == init_x

    # Y adjustment
    item._on_button_clicked("btn_y_up")
    assert item._epicenter_y == init_y + 20.0
    item._on_button_clicked("btn_y_down")
    assert item._epicenter_y == init_y

    # vp adjustment
    item._on_button_clicked("btn_vp_up")
    assert item._vp == init_vp + 0.5
    item._on_button_clicked("btn_vp_down")
    assert item._vp == init_vp

    # Preset cycle
    item._on_button_clicked("btn_preset")
    assert item._active_preset_code != "SAN_ANDREAS_TRANSVERSE_SLIP"

    # Reset
    item._on_button_clicked("btn_reset")
    assert math.isclose(item._epicenter_x, item._preset.epicenter_x_km, rel_tol=1e-4)


def test_stamping_and_stem_tools_vector_generation(qapp):
    """Verifies that PhysicsTools.create_seismic_waves_diagram generates valid vector items."""
    diagram_items = PhysicsTools.create_seismic_waves_diagram(
        center=(400.0, 300.0),
        width=680.0,
        height=380.0,
        preset_code="TOHOKU_2011_MEGATHRUST",
        event_name="Trận động đất Tohoku-Oki 2011",
        epicenter_x=140.0,
        epicenter_y=-60.0,
        magnitude=9.1,
        vp=7.6,
        vs=4.3,
        depth_km=29.0,
        dist_a_km=300.0,
        sp_lag_a_s=30.0,
    )
    assert len(diagram_items) >= 20

    # Test stamp signal emission
    item = InteractiveSeismicWavesItem()
    emitted = []
    item.stamp_requested.connect(lambda d: emitted.append(d))
    item.request_stamp()
    assert len(emitted) == 1
    assert "epicenter_x" in emitted[0]
    assert "epicenter_y" in emitted[0]
    assert "magnitude" in emitted[0]
    assert "sp_lag_a_s" in emitted[0]


def test_lesson_board_integration(qapp):
    """Verifies lifecycle methods on LessonBoard."""
    board = LessonBoard()

    # Initially hidden
    assert not board.is_seismic_waves_visible()
    assert board.get_seismic_waves() is None

    # Show
    item = board.show_seismic_waves(True, pos=(110.0, 40.0))
    assert item is not None
    assert board.is_seismic_waves_visible()
    assert board.get_seismic_waves() is item

    # Toggle hide
    is_vis = board.toggle_seismic_waves()
    assert not is_vis
    assert not board.is_seismic_waves_visible()

    # Toggle show
    is_vis = board.toggle_seismic_waves()
    assert is_vis
    assert board.is_seismic_waves_visible()

    # Stamping
    cnt_before = len(board.items())
    board._on_stamp_seismic_waves({
        "preset_code": "TOHOKU_2011_MEGATHRUST",
        "event_name": "Trận động đất Tohoku-Oki 2011",
        "epicenter_x": 140.0,
        "epicenter_y": -60.0,
        "magnitude": 9.1,
        "vp": 7.6,
        "vs": 4.3,
        "depth_km": 29.0,
        "dist_a_km": 300.0,
        "sp_lag_a_s": 30.0,
    })
    assert len(board.items()) > cnt_before


def test_paint_render_pipeline(qapp):
    """Verifies that the item can paint onto a QImage without throwing exceptions."""
    item = InteractiveSeismicWavesItem()
    image = QImage(800, 700, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    item.paint(painter, None, None)
    painter.end()
    assert not image.isNull()
