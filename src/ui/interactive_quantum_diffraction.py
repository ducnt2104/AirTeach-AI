"""
AirTeach AI - Interactive Quantum Mechanics: Double-Slit Diffraction & Matter Waves
===================================================================================
Authentic quantum mechanical simulation based on de Broglie matter waves & Born rule:
- de Broglie Matter Wavelength:
    λ = h / p
    - Non-relativistic: p = sqrt(2 * m * E_k) => λ = h / sqrt(2 * m * E_k)
    - Relativistic (electrons at high voltage):
        p = sqrt(E_k * (E_k + 2 * m_0 * c²)) / c
        λ = h * c / sqrt(E_k * (E_k + 2 * m_0 * c²))
    - Photons (massless): p = E / c => λ = h * c / E
- Double-Slit Quantum Superposition & Born Rule:
    Path from slit 1 to point y: r_1 = sqrt(D² + (y - d/2)²)
    Path from slit 2 to point y: r_2 = sqrt(D² + (y + d/2)²)
    Phase difference: Δϕ = 2π * (r_2 - r_1) / λ ≈ 2π * d * y / (λ * D)
    Single-slit diffraction parameter: β = π * a * y / (λ * D)
- Coherent Probability Density (No Which-Way Detector):
    I_coherent(y) = I_0 * cos²(Δϕ / 2) * sinc²(β)
    Fringe separation: Δy = λ * D / d
- Which-Way (Welcher-Weg) Detector & Wavepacket Collapse:
    When a which-way detector tags particles at slit 1 or 2 with fidelity η ∈ [0, 1]:
    I_incoherent(y) = (1/2) * I_0 * [ sinc²(π * a * (y - d/2) / (λ * D))
                                   + sinc²(π * a * (y + d/2) / (λ * D)) ]
    Observed Intensity:
      I_total(y) = (1 - η) * I_coherent(y) + η * I_incoherent(y)
    Fringe Visibility (Michelson Contrast):
      V = (I_max - I_min) / (I_max + I_min) = 1 - η
- Single-Particle Stochastic Accumulation (Hitachi Tonomura 1989):
    Simulates individual electron/photon stochastic arrivals sampled from probability density
    P(y) = I(y) / ∫ I(u) du, displaying build-up from discrete dots into quantum fringes.
- 5 Curriculum Presets:
    1. ELECTRON_HITACHI: 50 keV electron beam, λ = 5.36 pm, single-particle accumulation.
    2. PHOTON_YOUNG: 532 nm green laser optical double-slit interference.
    3. WHICH_WAY_COLLAPSE: Slit detector causing wavepacket decoherence and fringe erasure.
    4. THERMAL_NEUTRONS: 0.025 eV thermal neutron matter-wave diffraction.
    5. FULLERENE_C60: 720 u massive Buckyball molecule quantum diffraction (Zeilinger 1999).
- Touchless webcam air gestures (air-tap, slider dragging), bimanual manipulation,
  canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

# Fundamental Physical Constants (SI Units)
PLANCK_H: float = 6.62607015e-34       # J·s
SPEED_C: float = 299792458.0           # m/s
E_CHARGE: float = 1.602176634e-19      # C
M_ELECTRON: float = 9.1093837e-31      # kg
M_NEUTRON: float = 1.674927498e-27     # kg
AMU: float = 1.66053906660e-27         # kg
M_C60: float = 60.0 * 12.011 * AMU     # kg (~1.196e-24 kg)


@dataclass
class QuantumPreset:
    """Preset configuration for educational quantum diffraction experiments."""
    key: str
    name: str
    desc: str
    particle_type: str        # 'electron', 'photon', 'neutron', 'c60'
    mass_kg: float
    energy_ev: float          # Kinetic energy (eV)
    wavelength_m: float       # de Broglie or optical wavelength (m)
    slit_d_um: float          # Slit separation d (μm)
    slit_a_um: float          # Slit width a (μm)
    screen_dist_m: float      # Distance to screen D (m)
    which_way_active: bool    # Detector turned on
    decoherence_rate: float   # η ∈ [0, 1]


PRESETS: List[QuantumPreset] = [
    QuantumPreset(
        key="electron_hitachi",
        name="Hitachi Electron (Tonomura 1989)",
        desc="Chùm electron 50 keV tích lũy từng hạt trên màn huỳnh quang, tạo hệ vân giao thoa",
        particle_type="electron",
        mass_kg=M_ELECTRON,
        energy_ev=50000.0,
        wavelength_m=5.355e-12,    # 5.36 pm
        slit_d_um=1.0,
        slit_a_um=0.20,
        screen_dist_m=1.2,
        which_way_active=False,
        decoherence_rate=0.0,
    ),
    QuantumPreset(
        key="photon_young",
        name="Giao thoa Quang học Young (532nm)",
        desc="Laser quang học bước sóng 532 nm (ánh sáng xanh lục) qua khe kép kinh điển của Thomas Young",
        particle_type="photon",
        mass_kg=0.0,
        energy_ev=2.33,
        wavelength_m=532.0e-9,     # 532 nm
        slit_d_um=250.0,
        slit_a_um=50.0,
        screen_dist_m=1.0,
        which_way_active=False,
        decoherence_rate=0.0,
    ),
    QuantumPreset(
        key="which_way_collapse",
        name="Bộ dò Which-Way & Sụp đổ sóng",
        desc="Đặt máy dò xác định đường đi ở khe 1, làm biến mất tính kết hợp pha và xóa sổ vân giao thoa",
        particle_type="electron",
        mass_kg=M_ELECTRON,
        energy_ev=50000.0,
        wavelength_m=5.355e-12,
        slit_d_um=1.0,
        slit_a_um=0.20,
        screen_dist_m=1.2,
        which_way_active=True,
        decoherence_rate=1.0,
    ),
    QuantumPreset(
        key="thermal_neutrons",
        name="Sóng vật chất Neutron nhiệt",
        desc="Chùm neutron lò phản ứng năng lượng 0.025 eV, bước sóng de Broglie 0.181 nm qua mạng tinh thể",
        particle_type="neutron",
        mass_kg=M_NEUTRON,
        energy_ev=0.025,
        wavelength_m=1.81e-10,     # 0.181 nm
        slit_d_um=25.0,
        slit_a_um=5.0,
        screen_dist_m=1.5,
        which_way_active=False,
        decoherence_rate=0.0,
    ),
    QuantumPreset(
        key="fullerene_c60",
        name="Đại phân tử Buckyball C60 (Zeilinger)",
        desc="Phân tử C60 nặng 720 amu bay ở vận tốc 220 m/s với bước sóng de Broglie 2.5 pm (Kỷ lục 1999)",
        particle_type="c60",
        mass_kg=M_C60,
        energy_ev=0.18,
        wavelength_m=2.5e-12,      # 2.5 pm
        slit_d_um=0.10,
        slit_a_um=0.02,
        screen_dist_m=1.2,
        which_way_active=False,
        decoherence_rate=0.0,
    ),
]


class InteractiveQuantumDiffractionItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic Quantum Mechanics, de Broglie matter wave diffraction,
    which-way wavepacket collapse, and single-particle stochastic accumulation.
    """
    preset_changed = Signal(str)
    property_changed = Signal(str, float)

    WIDTH: float = 920.0
    HEIGHT: float = 600.0

    # Max particle hit buffer size
    MAX_PARTICLES: int = 4000

    def __init__(
        self,
        board: Optional[LessonBoard] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board = board
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Simulation state
        self._preset_index: int = 0
        self._is_paused: bool = False
        self._sim_time: float = 0.0
        self._beam_flux_rate: float = 120.0  # particles per second

        # Quantum experiment parameters
        preset = PRESETS[0]
        self._particle_type: str = preset.particle_type
        self._mass: float = preset.mass_kg
        self._energy_ev: float = preset.energy_ev
        self._wavelength: float = preset.wavelength_m
        self._slit_d_um: float = preset.slit_d_um
        self._slit_a_um: float = preset.slit_a_um
        self._screen_dist: float = preset.screen_dist_m
        self._which_way: bool = preset.which_way_active
        self._decoherence: float = preset.decoherence_rate

        # Accumulated particle impacts on screen: list of normalized y positions in [-1.0, +1.0]
        self._particle_hits: List[float] = []

        # Sub-accumulator for fractional particle arrival
        self._particle_acc: float = 0.0

        # UI interaction state
        self._dragging_slider: Optional[str] = None

        # Calculate initial wavelength
        self._recalc_wavelength()

    @property
    def current_preset(self) -> QuantumPreset:
        return PRESETS[self._preset_index]

    def _recalc_wavelength(self) -> None:
        """Compute exact de Broglie wavelength λ = h / p using relativistic/classical mechanics."""
        if self._particle_type == "photon":
            energy_j = self._energy_ev * E_CHARGE
            self._wavelength = (PLANCK_H * SPEED_C) / energy_j if energy_j > 0 else 532e-9
        elif self._particle_type == "electron":
            # Relativistic electron de Broglie wavelength:
            # p = sqrt(E_k * (E_k + 2 * m_0 * c^2)) / c
            e_k_j = self._energy_ev * E_CHARGE
            m0_c2_j = M_ELECTRON * (SPEED_C**2)
            p = math.sqrt(e_k_j * (e_k_j + 2.0 * m0_c2_j)) / SPEED_C
            self._wavelength = PLANCK_H / p if p > 0 else 5.355e-12
        else:
            # Non-relativistic de Broglie: p = sqrt(2 * m * E_k)
            e_k_j = self._energy_ev * E_CHARGE
            p = math.sqrt(2.0 * self._mass * e_k_j)
            self._wavelength = PLANCK_H / p if p > 0 else 1e-10

    def set_preset(self, index: int) -> None:
        """Switch active quantum diffraction curriculum preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            p = PRESETS[index]
            self._particle_type = p.particle_type
            self._mass = p.mass_kg
            self._energy_ev = p.energy_ev
            self._slit_d_um = p.slit_d_um
            self._slit_a_um = p.slit_a_um
            self._screen_dist = p.screen_dist_m
            self._which_way = p.which_way_active
            self._decoherence = p.decoherence_rate
            self._recalc_wavelength()
            self._particle_hits.clear()
            self._sim_time = 0.0
            self.preset_changed.emit(p.key)
            self.update()

    def set_slit_separation(self, d_um: float) -> None:
        """Update slit separation distance d (μm)."""
        self._slit_d_um = max(0.01, min(d_um, 500.0))
        self._particle_hits.clear()
        self.property_changed.emit("slit_d_um", self._slit_d_um)
        self.update()

    def set_slit_width(self, a_um: float) -> None:
        """Update slit aperture width a (μm)."""
        self._slit_a_um = max(0.005, min(a_um, self._slit_d_um * 0.95))
        self._particle_hits.clear()
        self.property_changed.emit("slit_a_um", self._slit_a_um)
        self.update()

    def set_decoherence(self, eta: float) -> None:
        """Update which-way detector decoherence rate η ∈ [0, 1]."""
        self._decoherence = max(0.0, min(eta, 1.0))
        self._which_way = (self._decoherence > 0.0)
        self._particle_hits.clear()
        self.property_changed.emit("decoherence", self._decoherence)
        self.update()

    def toggle_which_way(self) -> None:
        """Toggle which-way detector on or off."""
        self._which_way = not self._which_way
        self._decoherence = 1.0 if self._which_way else 0.0
        self._particle_hits.clear()
        self.update()

    def toggle_pause(self) -> None:
        """Toggle simulation pause/resume state."""
        self._is_paused = not self._is_paused
        self.update()

    def clear_screen(self) -> None:
        """Clear accumulated particle impact dots on screen."""
        self._particle_hits.clear()
        self.update()

    def compute_intensity_at(self, y_norm: float) -> float:
        """
        Compute theoretical quantum probability density I(y) for normalized screen pos y_norm in [-1, 1].
        y_phys corresponds to physical coordinate in [-y_max, +y_max].
        """
        # Physical screen half-height: y_max = 5 * (lambda * D / d)
        fringe_spacing = (self._wavelength * self._screen_dist) / (self._slit_d_um * 1e-6)
        y_max = max(1e-6, 5.0 * fringe_spacing)
        y_phys = y_norm * y_max

        # Double slit phase difference: delta_phi = 2pi * d * y / (lambda * D)
        delta_phi = (2.0 * math.pi * (self._slit_d_um * 1e-6) * y_phys) / (self._wavelength * self._screen_dist)

        # Single slit diffraction parameter: beta = pi * a * y / (lambda * D)
        beta = (math.pi * (self._slit_a_um * 1e-6) * y_phys) / (self._wavelength * self._screen_dist)
        sinc_sq = (math.sin(beta) / beta)**2 if abs(beta) > 1e-6 else 1.0

        # Fully coherent intensity
        i_coherent = (math.cos(delta_phi * 0.5)**2) * sinc_sq

        # Incoherent (which-way / collapsed) intensity: sum of two single slits shifted by +/- d/2
        d_half = (self._slit_d_um * 1e-6) * 0.5
        b1 = (math.pi * (self._slit_a_um * 1e-6) * (y_phys - d_half)) / (self._wavelength * self._screen_dist)
        b2 = (math.pi * (self._slit_a_um * 1e-6) * (y_phys + d_half)) / (self._wavelength * self._screen_dist)
        sinc1 = (math.sin(b1) / b1)**2 if abs(b1) > 1e-6 else 1.0
        sinc2 = (math.sin(b2) / b2)**2 if abs(b2) > 1e-6 else 1.0
        i_incoherent = 0.5 * (sinc1 + sinc2)

        # Blend according to decoherence factor
        return (1.0 - self._decoherence) * i_coherent + self._decoherence * i_incoherent

    def sample_particle_arrival(self) -> float:
        """Rejection sampling of a single particle hit according to quantum Born probability distribution."""
        # Trial random positions in [-1, 1]
        for _ in range(50):
            cand_y = random.uniform(-1.0, 1.0)
            prob = self.compute_intensity_at(cand_y)
            if random.random() < prob:
                return cand_y
        return random.gauss(0.0, 0.35)

    def step_simulation(self, dt: float) -> None:
        """Advance quantum wave propagation and particle stochastic accumulation."""
        if self._is_paused or dt <= 0.0:
            return

        self._sim_time += dt

        # Accumulate particles
        self._particle_acc += self._beam_flux_rate * dt
        new_particles = int(self._particle_acc)
        self._particle_acc -= new_particles

        if new_particles > 0:
            count_to_add = min(new_particles, 150)
            for _ in range(count_to_add):
                if len(self._particle_hits) < self.MAX_PARTICLES:
                    self._particle_hits.append(self.sample_particle_arrival())
                else:
                    # Circular overwrite when buffer fills
                    replace_idx = random.randint(0, self.MAX_PARTICLES - 1)
                    self._particle_hits[replace_idx] = self.sample_particle_arrival()

        self.update()

    # -------------------------------------------------------------------------
    # Layout Geometry
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _wave_rect(self) -> QRectF:
        """Bounding rect of the 2D wave propagation & barrier chamber."""
        return QRectF(25.0, 115.0, 460.0, 360.0)

    def _screen_rect(self) -> QRectF:
        """Bounding rect of the phosphor detector screen & intensity profile."""
        return QRectF(500.0, 115.0, 395.0, 360.0)

    def _controls_rect(self) -> QRectF:
        """Bounding rect of bottom controls."""
        return QRectF(25.0, 490.0, 870.0, 95.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, QuantumPreset]]:
        rects = []
        btn_w = 166.0
        btn_h = 32.0
        start_x = 25.0
        y = 65.0
        spacing = 10.0
        for i, preset in enumerate(PRESETS):
            x = start_x + i * (btn_w + spacing)
            rects.append((QRectF(x, y, btn_w, btn_h), preset))
        return rects

    # -------------------------------------------------------------------------
    # Painting & Rendering
    # -------------------------------------------------------------------------

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 1. Background
        self._paint_background(painter)

        # 2. Header
        self._paint_header(painter)

        # 3. Preset Buttons
        self._paint_preset_buttons(painter)

        # 4. Wave Propagation & Barrier Chamber
        self._paint_wave_chamber(painter)

        # 5. Phosphor Detector & Intensity Profile
        self._paint_screen_detector(painter)

        # 6. Controls & Sliders
        self._paint_controls(painter)

        painter.restore()

    def _paint_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        bg_grad = QLinearGradient(0, 0, 0, self.HEIGHT)
        bg_grad.setColorAt(0.0, QColor(10, 15, 29, 250))
        bg_grad.setColorAt(1.0, QColor(15, 23, 42, 252))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor(168, 85, 247, 80), 1.5))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        # Title
        painter.setPen(QColor(241, 245, 249))
        font_title = QFont("Segoe UI", 13, QFont.Bold)
        painter.setFont(font_title)
        painter.drawText(28, 38, "CƠ HỌC LƯỢNG TỬ: GIAO THOA KHE KÉP & SÓNG VẬT CHẤT DE BROGLIE")

        # Subtitle
        painter.setPen(QColor(148, 163, 184))
        font_sub = QFont("Segoe UI", 9)
        painter.setFont(font_sub)
        painter.drawText(28, 54, "Lưỡng tính sóng-hạt, tích lũy ngẫu nhiên từng hạt Hitachi & sụp đổ hàm sóng Which-Way")

        # Status badge (Live / Paused)
        badge_rect = QRectF(self.WIDTH - 150.0, 22.0, 122.0, 26.0)
        if self._is_paused:
            painter.setBrush(QColor(239, 68, 68, 40))
            painter.setPen(QPen(QColor(239, 68, 68), 1.2))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)
            painter.setPen(QColor(252, 165, 165))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, "⏸ TẠM DỪNG")
        else:
            painter.setBrush(QColor(168, 85, 247, 40))
            painter.setPen(QPen(QColor(168, 85, 247), 1.2))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)
            painter.setPen(QColor(216, 180, 254))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, "⚛ ĐANG MÔ PHỎNG")

    def _paint_preset_buttons(self, painter: QPainter) -> None:
        font_btn = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(font_btn)
        for i, (btn_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == self._preset_index)
            if is_active:
                painter.setBrush(QColor(147, 51, 234, 220))
                painter.setPen(QPen(QColor(216, 180, 254), 1.2))
            else:
                painter.setBrush(QColor(30, 41, 59, 180))
                painter.setPen(QPen(QColor(71, 85, 105), 1.0))
            painter.drawRoundedRect(btn_rect, 6.0, 6.0)

            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            painter.drawText(btn_rect, Qt.AlignCenter, preset.name)

    def _paint_wave_chamber(self, painter: QPainter) -> None:
        """Render 2D wave propagation from particle gun through double slits with animated wavefronts."""
        w_rect = self._wave_rect()

        # Frame
        painter.setBrush(QColor(5, 8, 18))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(w_rect, 8.0, 8.0)

        painter.save()
        painter.setClipRect(w_rect)

        # Particle gun / source position on left
        source_x = w_rect.left() + 30.0
        source_y = w_rect.center().y()

        # Barrier position
        barrier_x = w_rect.left() + 200.0
        slit_spacing_px = 50.0
        slit1_y = source_y - slit_spacing_px * 0.5
        slit2_y = source_y + slit_spacing_px * 0.5

        # 1. Incoming spherical wavefronts before barrier
        time_phase = self._sim_time * 6.0
        painter.setPen(QPen(QColor(168, 85, 247, 80), 1.5))
        for r in range(15, 180, 20):
            wave_r = (r + time_phase * 15.0) % 180.0
            if wave_r > 5.0:
                painter.drawArc(
                    int(source_x - wave_r), int(source_y - wave_r),
                    int(wave_r * 2), int(wave_r * 2),
                    -70 * 16, 140 * 16
                )

        # 2. Double Slit Barrier Wall
        painter.setBrush(QColor(51, 65, 85))
        painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        wall_w = 12.0
        slit_w_px = 6.0

        # Top segment
        painter.drawRect(QRectF(barrier_x, w_rect.top(), wall_w, slit1_y - slit_w_px * 0.5 - w_rect.top()))
        # Center segment between slits
        painter.drawRect(QRectF(barrier_x, slit1_y + slit_w_px * 0.5, wall_w, (slit2_y - slit_w_px * 0.5) - (slit1_y + slit_w_px * 0.5)))
        # Bottom segment
        painter.drawRect(QRectF(barrier_x, slit2_y + slit_w_px * 0.5, wall_w, w_rect.bottom() - (slit2_y + slit_w_px * 0.5)))

        # 3. Wavefronts propagating past slits
        painter.setPen(QPen(QColor(192, 132, 252, 90), 1.2))
        for r in range(10, 260, 22):
            w_r = (r + time_phase * 15.0) % 260.0
            if w_r > 5.0:
                # Wavefront from Slit 1
                painter.drawArc(
                    int(barrier_x + wall_w - w_r), int(slit1_y - w_r),
                    int(w_r * 2), int(w_r * 2),
                    -75 * 16, 150 * 16
                )
                # Wavefront from Slit 2
                painter.drawArc(
                    int(barrier_x + wall_w - w_r), int(slit2_y - w_r),
                    int(w_r * 2), int(w_r * 2),
                    -75 * 16, 150 * 16
                )

        # 4. Which-Way Detector indicators (if active)
        if self._which_way:
            det_rect1 = QRectF(barrier_x - 18.0, slit1_y - 10.0, 16.0, 20.0)
            painter.setBrush(QColor(239, 68, 68, 200))
            painter.setPen(QPen(QColor(254, 202, 202), 1.2))
            painter.drawRoundedRect(det_rect1, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(det_rect1, Qt.AlignCenter, "DET")

        # 5. Particle Gun label
        gun_rect = QRectF(source_x - 20.0, source_y - 14.0, 32.0, 28.0)
        painter.setBrush(QColor(30, 41, 59))
        painter.setPen(QPen(QColor(147, 51, 234), 1.5))
        painter.drawRoundedRect(gun_rect, 4.0, 4.0)
        painter.setPen(QColor(216, 180, 254))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(gun_rect, Qt.AlignCenter, "GUN")

        painter.restore()

        # Chamber label
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(192, 132, 252))
        painter.drawText(int(w_rect.left() + 12), int(w_rect.top() + 20), "BUỒNG TRUYỀN SÓNG & KHE KÉP")

    def _paint_screen_detector(self, painter: QPainter) -> None:
        """Render phosphor accumulation screen with empirical hits and theoretical Born intensity curve."""
        s_rect = self._screen_rect()

        # Frame
        painter.setBrush(QColor(5, 8, 18))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(s_rect, 8.0, 8.0)

        # Subdivide into Phosphor hit column (left 140px) and Theoretical/Empirical Curve (right 240px)
        screen_bar = QRectF(s_rect.left() + 15.0, s_rect.top() + 35.0, 110.0, s_rect.height() - 50.0)
        curve_rect = QRectF(s_rect.left() + 145.0, s_rect.top() + 35.0, s_rect.width() - 160.0, s_rect.height() - 50.0)

        # Phosphor Screen Background
        painter.setBrush(QColor(10, 25, 20))
        painter.setPen(QPen(QColor(16, 185, 129, 100), 1.0))
        painter.drawRoundedRect(screen_bar, 4.0, 4.0)

        # Draw individual particle hits (phosphor green/cyan glow)
        painter.setPen(QPen(QColor(110, 231, 183, 190), 1.5))
        center_y = screen_bar.center().y()
        half_h = screen_bar.height() * 0.46

        for norm_y in self._particle_hits:
            py = center_y + norm_y * half_h
            px = screen_bar.left() + 10.0 + random.random() * (screen_bar.width() - 20.0)
            if screen_bar.top() <= py <= screen_bar.bottom():
                painter.drawPoint(QPointF(px, py))

        # Theoretical Born Curve background
        painter.setBrush(QColor(15, 23, 42, 180))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(curve_rect, 4.0, 4.0)

        # Plot theoretical curve I(y)
        curve_path = QPainterPath()
        num_pts = 80
        for i in range(num_pts):
            norm_y = -1.0 + 2.0 * (i / (num_pts - 1))
            intensity = self.compute_intensity_at(norm_y)
            py = center_y + norm_y * half_h
            px = curve_rect.left() + 10.0 + intensity * (curve_rect.width() - 25.0)
            if i == 0:
                curve_path.moveTo(px, py)
            else:
                curve_path.lineTo(px, py)

        painter.setPen(QPen(QColor(234, 179, 8), 1.8))
        painter.drawPath(curve_path)

        # Labels
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(52, 211, 153))
        painter.drawText(int(screen_bar.left()), int(s_rect.top() + 22), "MÀN HUỲNH QUANG")

        painter.setPen(QColor(250, 204, 21))
        painter.drawText(int(curve_rect.left()), int(s_rect.top() + 22), "CƯỜNG ĐỘ BORN |Ψ|²")

        # Live hit counter
        hit_count_str = f"Số hạt: {len(self._particle_hits)} / {self.MAX_PARTICLES}"
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(226, 232, 240))
        painter.drawText(QRectF(s_rect.right() - 170, s_rect.top() + 10, 155, 18), Qt.AlignRight, hit_count_str)

        # Telemetry info box at bottom of curve panel
        fringe_spacing_um = ((self._wavelength * self._screen_dist) / (self._slit_d_um * 1e-6)) * 1e6
        visibility = max(0.0, 1.0 - self._decoherence)
        info_lines = [
            f"λ de Broglie: {self._wavelength:.3e} m",
            f"Động năng E_k: {self._energy_ev:.2f} eV",
            f"Khoảng vân Δy: {fringe_spacing_um:.2f} μm",
            f"Độ tương phản V: {visibility * 100:.1f} %",
        ]
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(148, 163, 184))
        for idx, text in enumerate(info_lines):
            painter.drawText(int(curve_rect.left() + 8), int(curve_rect.bottom() - 48 + idx * 12), text)

    def _paint_controls(self, painter: QPainter) -> None:
        """Render interactive bottom sliders and action buttons."""
        ctrl_rect = self._controls_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(ctrl_rect, 8.0, 8.0)

        # 4 Sliders arranged horizontally
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        sliders = [
            ("slit_d", "Khoảng cách khe d", f"{self._slit_d_um:.2f} μm", self._slit_d_um, 0.05, 50.0),
            ("slit_a", "Độ rộng khe a", f"{self._slit_a_um:.2f} μm", self._slit_a_um, 0.01, 10.0),
            ("decoherence", "Độ sụp sóng η (Which-Way)", f"{self._decoherence * 100:.0f} %", self._decoherence, 0.0, 1.0),
            ("flux", "Cường độ chùm hạt", f"{self._beam_flux_rate:.0f} hạt/s", self._beam_flux_rate, 20.0, 500.0),
        ]

        for i, (key, label, val_text, val, min_v, max_v) in enumerate(sliders):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(int(cx), int(cy), label)

            painter.setPen(QColor(192, 132, 252))
            painter.drawText(QRectF(cx, cy - 12, col_w - 20, 14), Qt.AlignRight, val_text)

            # Track
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx, track_y, track_w, 6.0)
            painter.setBrush(QColor(30, 41, 59))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(track_rect, 3.0, 3.0)

            # Active fill
            norm = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.0
            fill_rect = QRectF(cx, track_y, track_w * norm, 6.0)
            painter.setBrush(QColor(147, 51, 234))
            painter.drawRoundedRect(fill_rect, 3.0, 3.0)

            # Thumb
            thumb_x = cx + track_w * norm
            thumb_rect = QRectF(thumb_x - 6.0, track_y - 4.0, 12.0, 14.0)
            painter.setBrush(QColor(241, 245, 249))
            painter.setPen(QPen(QColor(147, 51, 234), 1.2))
            painter.drawRoundedRect(thumb_rect, 3.0, 3.0)

        # Action Buttons
        action_y = ctrl_rect.top() + 62.0
        btn_which_rect = QRectF(ctrl_rect.left() + 20, action_y, 180, 24)
        painter.setBrush(QColor(239, 68, 68, 40) if self._which_way else QColor(30, 41, 59))
        painter.setPen(QPen(QColor(239, 68, 68) if self._which_way else QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(btn_which_rect, 4.0, 4.0)
        painter.setPen(QColor(254, 202, 202) if self._which_way else QColor(241, 245, 249))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(btn_which_rect, Qt.AlignCenter, "🔍 BỘ DÒ WHICH-WAY")

        btn_clear_rect = QRectF(ctrl_rect.left() + 215, action_y, 160, 24)
        painter.setBrush(QColor(30, 41, 59))
        painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(btn_clear_rect, 4.0, 4.0)
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(btn_clear_rect, Qt.AlignCenter, "🧹 XÓA MÀN HUỲNH QUANG")

    # -------------------------------------------------------------------------
    # Mouse & Touchless Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check Preset buttons
        for i, (btn_rect, _) in enumerate(self._preset_button_rects()):
            if btn_rect.contains(pos):
                self.set_preset(i)
                event.accept()
                return

        # Check Status badge click (pause / resume)
        badge_rect = QRectF(self.WIDTH - 150.0, 22.0, 122.0, 26.0)
        if badge_rect.contains(pos):
            self.toggle_pause()
            event.accept()
            return

        # Check Action buttons
        ctrl_rect = self._controls_rect()
        action_y = ctrl_rect.top() + 62.0
        btn_which_rect = QRectF(ctrl_rect.left() + 20, action_y, 180, 24)
        if btn_which_rect.contains(pos):
            self.toggle_which_way()
            event.accept()
            return

        btn_clear_rect = QRectF(ctrl_rect.left() + 215, action_y, 160, 24)
        if btn_clear_rect.contains(pos):
            self.clear_screen()
            event.accept()
            return

        # Check Sliders
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        for i, key in enumerate(["slit_d", "slit_a", "decoherence", "flux"]):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx - 5.0, track_y - 8.0, track_w + 10.0, 22.0)
            if track_rect.contains(pos):
                self._dragging_slider = key
                self._update_slider_from_pos(key, pos.x(), cx, track_w)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._dragging_slider:
            ctrl_rect = self._controls_rect()
            col_w = (ctrl_rect.width() - 40.0) / 4.0
            slider_keys = ["slit_d", "slit_a", "decoherence", "flux"]
            if self._dragging_slider in slider_keys:
                idx = slider_keys.index(self._dragging_slider)
                cx = ctrl_rect.left() + 20.0 + idx * col_w
                track_w = col_w - 25.0
                self._update_slider_from_pos(self._dragging_slider, pos.x(), cx, track_w)
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._dragging_slider = None
        super().mouseReleaseEvent(event)

    def _update_slider_from_pos(self, key: str, px: float, cx: float, track_w: float) -> None:
        norm = max(0.0, min(1.0, (px - cx) / track_w))
        if key == "slit_d":
            self.set_slit_separation(0.05 + norm * (50.0 - 0.05))
        elif key == "slit_a":
            self.set_slit_width(0.01 + norm * (10.0 - 0.01))
        elif key == "decoherence":
            self.set_decoherence(norm)
        elif key == "flux":
            self._beam_flux_rate = 20.0 + norm * (500.0 - 20.0)
            self.update()

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulator state for .airteach lesson format."""
        return {
            "type": "InteractiveQuantumDiffractionItem",
            "preset_key": self.current_preset.key,
            "particle_type": self._particle_type,
            "energy_ev": self._energy_ev,
            "slit_d_um": self._slit_d_um,
            "slit_a_um": self._slit_a_um,
            "screen_dist": self._screen_dist,
            "which_way": self._which_way,
            "decoherence": self._decoherence,
            "beam_flux_rate": self._beam_flux_rate,
            "particle_count": len(self._particle_hits),
            "is_paused": self._is_paused,
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveQuantumDiffractionItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "electron_hitachi")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "slit_d_um" in data:
            item.set_slit_separation(float(data["slit_d_um"]))
        if "slit_a_um" in data:
            item.set_slit_width(float(data["slit_a_um"]))
        if "decoherence" in data:
            item.set_decoherence(float(data["decoherence"]))
        if "beam_flux_rate" in data:
            item._beam_flux_rate = float(data["beam_flux_rate"])
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
