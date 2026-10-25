"""
AirTeach AI - Interactive Nuclear & Particle Physics: Wilson Cloud Chamber & Particle Tracks
=============================================================================================
Authentic nuclear radiation and ionizing particle detection based on Wilson Cloud Chamber physics:
- Supersaturated Vapor Condensation (Charles Wilson, 1911; Nobel Prize 1927):
    Supersaturated isopropanol vapor at critical supersaturation S > 4.
    Ionizing radiation creates electron-ion pairs along its path.
    Vapor molecules condense onto positive and negative ions, producing visible droplet tracks.
- Authentic Particle Track Morphologies:
    1. Alpha Particle (α = ⁴₂He²⁺, m ≈ 3.727 GeV/c², q = +2e, E ≈ 4-6 MeV):
       - Heavy, slow, high Linear Energy Transfer (LET ≈ 100 keV/μm).
       - Dense, thick, straight tracks with high droplet density.
       - Bragg peak ionization spike near track termination with occasional Rutherford nuclear kick.
    2. Beta Particle (β⁻ = e⁻ / β⁺ = e⁺, m ≈ 511 keV/c², q = ±1e, E ≈ 0.1-1.5 MeV):
       - Low mass, high velocity, low LET.
       - Thin, wispy, tortuous paths with frequent multiple Coulomb scattering kinks.
    3. Cosmic Ray Muon (μ⁻ / μ⁺, m ≈ 105.7 MeV/c², relativistic γ ≫ 1, p ≈ 3-4 GeV/c):
       - High momentum rigidity, minimum ionizing particle (MIP).
       - Extremely straight, crisp, piercing line crossing entire chamber.
    4. Gamma Secondary Compton Electron (γ photon interaction):
       - Neutral photon leaves no track; sudden appearance of ejected Compton electron track.
- Lorentz Force & Magnetic Spectrometry:
    F_Lorentz = q (v × B)
    Cyclotron / Larmor Radius:
      R = p_perp / (|q| · B) = (γ · m · v_perp) / (|q| · B)
    - Applied magnetic field B_z perpendicular to chamber plane.
    - Curvature radius directly measures particle momentum p.
    - Direction of curvature differentiates positive (α, β⁺) from negative (β⁻, μ⁻) charges.
- Radioactive Decay Kinetics & Statistics (Rutherford & Soddy 1902):
    N(t) = N_0 * exp(-λ * t) = N_0 * (1/2)^(t / T_1/2)
    Activity A(t) = λ * N(t) (measured in Becquerel / CPM).
- 5 Curriculum Presets:
    1. ALPHA_AMERICIUM: Am-241 / Po-210 alpha source (thick, straight tracks, Bragg peak).
    2. BETA_STRONTIUM: Sr-90 / Y-90 beta source (thin, wispy, zig-zag electron tracks).
    3. COSMIC_RAY_MUONS: Natural cosmic background radiation (relativistic straight tracks).
    4. LORENTZ_SPECTROMETER: Variable magnetic field B separating α, β⁻, and β⁺ by q/m and momentum.
    5. RADON_CASCADE: Thoron / Radon alpha-alpha double pulse decay chain (Rn-220 -> Po-216 -> Pb-212).
- Touchless webcam air gestures (air-tap, slider dragging), bimanual manipulation,
  canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

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


@dataclass
class IonTrack:
    """A single particle ionization track made of vapor droplets."""
    particle_type: str  # "alpha", "beta", "muon", "compton"
    charge: int  # +2 for alpha, -1 for beta-, +1 for beta+, -1 for muon
    energy_mev: float
    points: list[tuple[float, float, float]]  # (x_px, y_px, droplet_radius)
    birth_time_sec: float
    max_lifetime_sec: float = 2.5
    color: QColor = field(default_factory=lambda: QColor(240, 246, 255))
    alpha_opacity: float = 1.0


@dataclass(frozen=True)
class CloudChamberPreset:
    code: str
    name_vi: str
    source_type: str
    b_field_tesla: float
    decay_rate_cpm: float  # counts per minute
    supersaturation: float
    description: str


PRESETS: dict[str, CloudChamberPreset] = {
    "ALPHA_AMERICIUM": CloudChamberPreset(
        code="ALPHA_AMERICIUM",
        name_vi="1. Hạt Alpha (Nguồn Americium-241)",
        source_type="alpha",
        b_field_tesla=0.0,
        decay_rate_cpm=45.0,
        supersaturation=4.5,
        description="Hạt alpha (⁴He²⁺) ion hóa mạnh, tạo vệt sương dày đặc, thẳng tắp và đạt đỉnh ion hóa Bragg ở cuối quãng đường.",
    ),
    "BETA_STRONTIUM": CloudChamberPreset(
        code="BETA_STRONTIUM",
        name_vi="2. Hạt Beta (Nguồn Strontium-90)",
        source_type="beta",
        b_field_tesla=0.0,
        decay_rate_cpm=60.0,
        supersaturation=4.5,
        description="Hạt beta (electron tốc độ cao) khối lượng nhẹ, tán xạ nhiều lần với nguyên tử khí tạo vệt sương mảnh ngoằn ngoèo.",
    ),
    "COSMIC_RAY_MUONS": CloudChamberPreset(
        code="COSMIC_RAY_MUONS",
        name_vi="3. Tia Vũ trụ Muon (Cosmic Rays)",
        source_type="muon",
        b_field_tesla=0.0,
        decay_rate_cpm=30.0,
        supersaturation=4.8,
        description="Hạt Muon năng lượng cao (~4 GeV) xuyên qua bầu khí quyển xuống phòng thí nghiệm tạo đường rạch thẳng sắc bén.",
    ),
    "LORENTZ_SPECTROMETER": CloudChamberPreset(
        code="LORENTZ_SPECTROMETER",
        name_vi="4. Khối phổ Kính từ Lorentz (B-Field)",
        source_type="mixed",
        b_field_tesla=0.35,  # 0.35 Tesla
        decay_rate_cpm=60.0,
        supersaturation=4.5,
        description="Từ trường B hướng vuông góc uốn cong quỹ đạo hạt theo lực Lorentz F = q(v×B): alpha lệch nhẹ, beta lệch mạnh ngược chiều.",
    ),
    "RADON_CASCADE": CloudChamberPreset(
        code="RADON_CASCADE",
        name_vi="5. Chuỗi Phân rã Kép Radon (Rn-220)",
        source_type="cascade",
        b_field_tesla=0.0,
        decay_rate_cpm=40.0,
        supersaturation=4.5,
        description="Chuỗi phân rã alpha kép đặc trưng: ²²⁰Rn (T₁/₂ = 55.6s) phóng alpha thành ²¹⁶Po (T₁/₂ = 0.145s) phóng tiếp alpha thứ hai.",
    ),
}


class InteractiveCloudChamberItem(QGraphicsObject):
    """
    Interactive QGraphicsObject for Wilson Cloud Chamber Simulation,
    Alpha/Beta/Muon Ionization Tracks, and Lorentz Magnetic Spectrometry.
    """

    # Signals
    preset_changed = Signal(str)
    b_field_changed = Signal(float)
    rate_changed = Signal(float)
    closed = Signal()

    def __init__(
        self,
        board: Optional["LessonBoard"] = None,
        pos: tuple[float, float] | QPointF = (80.0, 80.0),
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board = board

        # Dimensions
        self._width: float = 940.0
        self._height: float = 650.0

        # Physical state
        self._current_preset: str = "ALPHA_AMERICIUM"
        self._b_field_tesla: float = 0.0  # Range: -0.5 to +0.5 Tesla
        self._decay_rate_cpm: float = 45.0  # Range: 5 to 150 CPM
        self._supersaturation: float = 4.5  # Range: 2.0 to 6.0
        self._source_type: str = "alpha"

        # Simulation clock and particles
        self._sim_time_sec: float = 0.0
        self._is_playing: bool = True
        self._tracks: list[IonTrack] = []
        self._total_counts: dict[str, int] = {"alpha": 0, "beta": 0, "muon": 0}
        self._max_active_tracks: int = 50

        # Background mist condensation noise particles
        self._bg_droplets: list[tuple[float, float, float, float]] = []  # (x, y, radius, alpha)
        self._init_chamber_background_mist()

        # UI interaction state
        self._active_slider: Optional[str] = None  # "b_field", "rate", "supersat"
        self._hovered_button: Optional[str] = None

        # Flags & setup
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        if isinstance(pos, QPointF):
            self.setPos(pos.x(), pos.y())
        else:
            self.setPos(pos[0], pos[1])

        # Random generator seeded for authentic reproducible distributions
        self._rng = random.Random(42)

    def _init_chamber_background_mist(self) -> None:
        """Generate subtle condensation background mist droplets in the active viewing chamber."""
        self._bg_droplets.clear()
        center_x = 300.0
        center_y = 350.0
        radius_max = 220.0

        for _ in range(120):
            r = math.sqrt(random.random()) * radius_max
            th = random.uniform(0, 2 * math.pi)
            x = center_x + r * math.cos(th)
            y = center_y + r * math.sin(th)
            rad = random.uniform(0.6, 1.8)
            alpha = random.uniform(0.08, 0.25)
            self._bg_droplets.append((x, y, rad, alpha))

    # -------------------------------------------------------------------------
    # Track Generation & Physics Simulation
    # -------------------------------------------------------------------------

    def spawn_particle_track(self, particle_type: Optional[str] = None) -> Optional[IonTrack]:
        """
        Generate authentic ionizing particle trajectory through supersaturated alcohol vapor:
        - Incorporates Lorentz deflection in magnetic field: R = p / (q * B).
        - Multiple Coulomb scattering for electrons (beta).
        - High linear energy transfer (LET) dense droplet column for alpha.
        - Relativistic minimum ionizing line for cosmic muons.
        """
        ptype = particle_type or self._source_type
        if ptype == "mixed":
            roll = self._rng.random()
            if roll < 0.45:
                ptype = "alpha"
            elif roll < 0.85:
                ptype = "beta"
            else:
                ptype = "muon"
        elif ptype == "cascade":
            ptype = "alpha"

        chamber_cx = 300.0
        chamber_cy = 350.0
        chamber_r = 215.0

        pts: list[tuple[float, float, float]] = []

        if ptype == "alpha":
            # Alpha track from radioactive needle source near bottom-left of viewport
            start_x = chamber_cx - 140.0
            start_y = chamber_cy + 90.0
            # Emitter angle aimed into chamber
            angle = math.radians(self._rng.uniform(-35.0, 45.0))
            energy_mev = self._rng.uniform(4.5, 5.8)  # Am-241 / Po-210 alpha
            range_px = energy_mev * 28.0  # ~130 to 160 px (~4 cm)
            num_steps = int(range_px / 2.0)

            cur_x = start_x
            cur_y = start_y
            cur_angle = angle

            # Lorentz deflection: curvature dtheta = dt * (q * B / m)
            # Alpha mass is large (4 u), so deflection is very small: dtheta ~ 0.001 * B
            b_deflect = self._b_field_tesla * 0.015

            for s in range(num_steps):
                step_len = 2.0
                cur_angle += b_deflect

                # End of range Rutherford kick
                if s > num_steps - 6:
                    cur_angle += math.radians(self._rng.uniform(-10.0, 10.0))

                cur_x += step_len * math.cos(cur_angle)
                cur_y -= step_len * math.sin(cur_angle)

                # Bragg Peak: droplets get larger and denser towards the end
                progress = s / float(num_steps)
                droplet_rad = 1.2 + 1.8 * (progress ** 2)
                pts.append((cur_x, cur_y, droplet_rad))

            track = IonTrack(
                particle_type="alpha",
                charge=+2,
                energy_mev=energy_mev,
                points=pts,
                birth_time_sec=self._sim_time_sec,
                max_lifetime_sec=2.2,
                color=QColor(255, 255, 255),
            )
            self._total_counts["alpha"] += 1

        elif ptype == "beta":
            # Beta track from radioactive source (Sr-90 beta-)
            start_x = chamber_cx - 140.0
            start_y = chamber_cy + 70.0
            angle = math.radians(self._rng.uniform(-20.0, 60.0))
            energy_mev = self._rng.uniform(0.3, 1.4)
            num_steps = self._rng.randint(45, 90)

            cur_x = start_x
            cur_y = start_y
            cur_angle = angle

            # Strong Lorentz deflection for light beta particle: R = p / (q*B)
            # Beta- has charge -1, curves opposite to alpha
            b_deflect = -self._b_field_tesla * (0.09 / max(0.2, energy_mev))

            for _ in range(num_steps):
                step_len = 2.5
                # Multiple Coulomb scattering random kicks
                cur_angle += b_deflect + math.radians(self._rng.uniform(-7.0, 7.0))
                cur_x += step_len * math.cos(cur_angle)
                cur_y -= step_len * math.sin(cur_angle)

                # Check if out of chamber viewport
                if math.hypot(cur_x - chamber_cx, cur_y - chamber_cy) > chamber_r:
                    break

                # Thin droplet size
                droplet_rad = self._rng.uniform(0.8, 1.3)
                pts.append((cur_x, cur_y, droplet_rad))

            track = IonTrack(
                particle_type="beta",
                charge=-1,
                energy_mev=energy_mev,
                points=pts,
                birth_time_sec=self._sim_time_sec,
                max_lifetime_sec=1.8,
                color=QColor(180, 220, 255),
            )
            self._total_counts["beta"] += 1

        elif ptype == "muon":
            # Cosmic ray muon entering from top of chamber
            start_x = chamber_cx + self._rng.uniform(-160.0, 160.0)
            start_y = chamber_cy - chamber_r + 5.0
            angle = math.radians(self._rng.uniform(-105.0, -75.0))  # Downward
            energy_mev = 3500.0  # 3.5 GeV MIP

            cur_x = start_x
            cur_y = start_y
            cur_angle = angle

            # Muon has enormous momentum rigidity (3-4 GeV/c), deflection is barely noticeable
            b_deflect = -self._b_field_tesla * 0.002

            for _ in range(160):
                step_len = 2.8
                cur_angle += b_deflect
                cur_x += step_len * math.cos(cur_angle)
                cur_y -= step_len * math.sin(cur_angle)

                if math.hypot(cur_x - chamber_cx, cur_y - chamber_cy) > chamber_r:
                    break

                droplet_rad = 1.0
                pts.append((cur_x, cur_y, droplet_rad))

            track = IonTrack(
                particle_type="muon",
                charge=-1,
                energy_mev=energy_mev,
                points=pts,
                birth_time_sec=self._sim_time_sec,
                max_lifetime_sec=1.5,
                color=QColor(100, 240, 255),
            )
            self._total_counts["muon"] += 1

        if pts:
            self._tracks.append(track)
            if len(self._tracks) > self._max_active_tracks:
                self._tracks.pop(0)
            return track

        return None

    def step_simulation(self, dt_sec: float = 0.033) -> None:
        """Advance cloud chamber simulation, spawning and decaying tracks."""
        if not self._is_playing:
            return

        self._sim_time_sec += dt_sec

        # Probability of spawning track in this timestep:
        # Rate in CPM => Rate per second = CPM / 60
        spawn_prob = (self._decay_rate_cpm / 60.0) * dt_sec
        if self._rng.random() < spawn_prob:
            self.spawn_particle_track()

        # Update and fade existing tracks
        surviving_tracks = []
        for trk in self._tracks:
            age = self._sim_time_sec - trk.birth_time_sec
            if age < trk.max_lifetime_sec:
                trk.alpha_opacity = max(0.0, 1.0 - (age / trk.max_lifetime_sec))
                surviving_tracks.append(trk)
        self._tracks = surviving_tracks

        self.update()

    def set_preset(self, preset_code: str) -> None:
        """Apply one of the 5 curriculum presets."""
        if preset_code not in PRESETS:
            logger.warning(f"Unknown cloud chamber preset: {preset_code}")
            return

        p = PRESETS[preset_code]
        self._current_preset = preset_code
        self._source_type = p.source_type
        self._b_field_tesla = p.b_field_tesla
        self._decay_rate_cpm = p.decay_rate_cpm
        self._supersaturation = p.supersaturation

        self._tracks.clear()
        self.preset_changed.emit(preset_code)
        self.b_field_changed.emit(self._b_field_tesla)
        self.rate_changed.emit(self._decay_rate_cpm)
        self.update()

    def set_b_field(self, b_tesla: float) -> None:
        """Set vertical magnetic field in Tesla [-0.5 to +0.5 T]."""
        self._b_field_tesla = max(-0.5, min(0.5, b_tesla))
        self.b_field_changed.emit(self._b_field_tesla)
        self.update()

    def set_decay_rate(self, cpm: float) -> None:
        """Set emission rate in Counts Per Minute [5 to 150 CPM]."""
        self._decay_rate_cpm = max(5.0, min(150.0, cpm))
        self.rate_changed.emit(self._decay_rate_cpm)
        self.update()

    def set_supersaturation(self, s: float) -> None:
        """Set chamber alcohol vapor supersaturation ratio [2.0 to 6.0]."""
        self._supersaturation = max(2.0, min(6.0, s))
        self.update()

    def toggle_play_pause(self) -> None:
        """Toggle simulation pause state."""
        self._is_playing = not self._is_playing
        self.update()

    def clear_tracks(self) -> None:
        """Clear all active condensation tracks."""
        self._tracks.clear()
        self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides & Rendering
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        # 1. Main outer card background
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0, 0, 0, self._height)
        grad.setColorAt(0.0, QColor(12, 18, 30, 248))
        grad.setColorAt(1.0, QColor(8, 12, 20, 252))
        painter.setPen(QPen(QColor(38, 65, 105, 200), 1.5))
        painter.setBrush(QBrush(grad))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

        # 2. Header bar with title and close button
        self._paint_header(painter)

        # 3. Preset selector bar (5 buttons)
        self._paint_preset_bar(painter)

        # 4. Wilson Cloud Chamber circular viewing viewport
        self._paint_cloud_chamber_viewport(painter)

        # 5. Right-side dashboard: Sliders, Metrics & Particle Physics readouts
        self._paint_control_dashboard(painter)

    def _paint_header(self, painter: QPainter) -> None:
        """Draw top banner with title, badges, and close button."""
        title_font = QFont("Segoe UI", 12, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(240, 246, 255)))
        painter.drawText(
            QRectF(20, 14, 600, 26),
            Qt.AlignLeft | Qt.AlignVCenter,
            "VẬT LÝ HẠT NHÂN: BUỒNG SƯƠNG WILSON & VỆT BỨC XẠ ION HÓA",
        )

        sub_font = QFont("Segoe UI", 8, QFont.Normal)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(130, 175, 230)))
        painter.drawText(
            QRectF(20, 36, 600, 16),
            Qt.AlignLeft | Qt.AlignVCenter,
            "Wilson Cloud Chamber · Alpha / Beta / Muon Tracks · Lorentz Spectrometer · Bragg Peak",
        )

        # Close button [X]
        close_rect = QRectF(self._width - 36, 12, 24, 24)
        is_hover = self._hovered_button == "close"
        painter.setPen(QPen(QColor(230, 80, 80) if is_hover else QColor(160, 60, 60), 1.2))
        painter.setBrush(QBrush(QColor(220, 60, 60, 80 if is_hover else 30)))
        painter.drawRoundedRect(close_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(close_rect, Qt.AlignCenter, "×")

    def _paint_preset_bar(self, painter: QPainter) -> None:
        """Draw the 5 preset tabs."""
        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0

        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            p = PRESETS[key]
            x = start_x + i * (btn_w + 8.0)
            rect = QRectF(x, y, btn_w, btn_h)

            is_active = (self._current_preset == key)
            is_hover = (self._hovered_button == f"preset_{key}")

            if is_active:
                painter.setPen(QPen(QColor(56, 189, 248), 1.6))
                painter.setBrush(QBrush(QColor(24, 76, 120, 220)))
            elif is_hover:
                painter.setPen(QPen(QColor(70, 130, 180), 1.2))
                painter.setBrush(QBrush(QColor(30, 50, 75, 180)))
            else:
                painter.setPen(QPen(QColor(40, 60, 90), 1.0))
                painter.setBrush(QBrush(QColor(20, 32, 50, 140)))

            painter.drawRoundedRect(rect, 6.0, 6.0)

            font = QFont("Segoe UI", 8, QFont.Bold if is_active else QFont.Normal)
            painter.setFont(font)
            painter.setPen(QPen(QColor(255, 255, 255) if is_active else QColor(180, 205, 230)))
            painter.drawText(rect, Qt.AlignCenter, p.name_vi)

    def _paint_cloud_chamber_viewport(self, painter: QPainter) -> None:
        """Draw circular cryogenic vapor chamber, black velvet base, and droplet tracks."""
        canvas_rect = QRectF(20, 96, 560, 470)

        # Outer frame
        painter.setPen(QPen(QColor(30, 48, 75, 160), 1.0))
        painter.setBrush(QBrush(QColor(6, 10, 18, 240)))
        painter.drawRoundedRect(canvas_rect, 10.0, 10.0)

        cx = 300.0
        cy = 345.0
        r_chamber = 210.0

        # Circular dark chamber with cryogenic rim
        rim_rect = QRectF(cx - r_chamber - 10, cy - r_chamber - 10, (r_chamber + 10) * 2, (r_chamber + 10) * 2)
        grad_rim = QRadialGradient(QPointF(cx, cy), r_chamber + 10)
        grad_rim.setColorAt(0.92, QColor(25, 45, 75))
        grad_rim.setColorAt(1.0, QColor(12, 22, 38))
        painter.setPen(QPen(QColor(56, 189, 248, 120), 2.0))
        painter.setBrush(QBrush(grad_rim))
        painter.drawEllipse(rim_rect)

        # Inner black velvet chamber base (T = -30°C to -40°C)
        chamber_circle = QRectF(cx - r_chamber, cy - r_chamber, r_chamber * 2, r_chamber * 2)
        grad_base = QRadialGradient(QPointF(cx - 30, cy - 30), r_chamber)
        grad_base.setColorAt(0.0, QColor(10, 16, 26))
        grad_base.setColorAt(0.85, QColor(4, 7, 12))
        grad_base.setColorAt(1.0, QColor(2, 4, 8))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(grad_base))
        painter.drawEllipse(chamber_circle)

        painter.save()
        clip_path = QPainterPath()
        clip_path.addEllipse(chamber_circle)
        painter.setClipPath(clip_path)

        # Draw ambient supersaturated vapor mist droplets
        for mx, my, mrad, malpha in self._bg_droplets:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(200, 225, 255, int(malpha * 255 * (self._supersaturation / 4.5)))))
            painter.drawEllipse(QPointF(mx, my), mrad, mrad)

        # Draw radioactive needle source holder (americium/strontium)
        needle_pt = QPointF(cx - 140.0, cy + 90.0)
        painter.setPen(QPen(QColor(180, 190, 205), 3.0))
        painter.drawLine(QPointF(cx - 190.0, cy + 130.0), needle_pt)
        painter.setPen(QPen(QColor(239, 68, 68), 2.0))
        painter.setBrush(QBrush(QColor(250, 204, 21)))
        painter.drawEllipse(needle_pt, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QPen(QColor(250, 204, 21)))
        painter.drawText(needle_pt.x() - 40, needle_pt.y() + 18, "NGUỒN PHÓNG XẠ")

        # Draw B-field indicator arrows if magnetic field is active
        if abs(self._b_field_tesla) > 0.01:
            b_sign_str = "⊙ B (Hướng ra)" if self._b_field_tesla > 0 else "⊗ B (Hướng vào)"
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QPen(QColor(56, 189, 248, 160)))
            painter.drawText(QRectF(cx - 70, cy - r_chamber + 15, 140, 20), Qt.AlignCenter, f"{b_sign_str} {abs(self._b_field_tesla):.2f} T")

        # ---------------------------------------------------------------------
        # Draw Condensation Droplet Tracks
        # ---------------------------------------------------------------------
        for trk in self._tracks:
            alpha_val = int(trk.alpha_opacity * 255)
            if alpha_val <= 0:
                continue

            base_color = trk.color
            pen_color = QColor(base_color.red(), base_color.green(), base_color.blue(), alpha_val)

            # Draw glowing halo around track
            for px, py, rad in trk.points:
                # Outer diffuse glow
                glow_col = QColor(base_color.red(), base_color.green(), base_color.blue(), int(alpha_val * 0.25))
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(glow_col))
                painter.drawEllipse(QPointF(px, py), rad * 2.2, rad * 2.2)

                # Core droplet
                painter.setBrush(QBrush(pen_color))
                painter.drawEllipse(QPointF(px, py), rad, rad)

        painter.restore()

        # Canvas bottom bar: Play / Pause, Inject Particle, Clear
        self._paint_canvas_bottom_bar(painter, canvas_rect)

    def _paint_canvas_bottom_bar(self, painter: QPainter, canvas_rect: QRectF) -> None:
        """Draw interactive playback and trigger buttons under canvas."""
        y_bar = canvas_rect.bottom() + 10
        btn_h = 28.0

        # Play / Pause
        play_rect = QRectF(canvas_rect.left(), y_bar, 96, btn_h)
        is_hover_play = self._hovered_button == "toggle_play"
        painter.setPen(QPen(QColor(56, 189, 248) if is_hover_play else QColor(40, 70, 105), 1.2))
        painter.setBrush(QBrush(QColor(24, 52, 85, 200 if is_hover_play else 140)))
        painter.drawRoundedRect(play_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(play_rect, Qt.AlignCenter, "⏸ TẠM DỪNG" if self._is_playing else "▶ TIẾP TỤC")

        # Inject Particle button
        inject_rect = QRectF(play_rect.right() + 8, y_bar, 110, btn_h)
        is_hover_inj = self._hovered_button == "inject_particle"
        painter.setPen(QPen(QColor(56, 189, 248) if is_hover_inj else QColor(40, 70, 105), 1.2))
        painter.setBrush(QBrush(QColor(24, 52, 85, 200 if is_hover_inj else 140)))
        painter.drawRoundedRect(inject_rect, 6.0, 6.0)
        painter.drawText(inject_rect, Qt.AlignCenter, "⚡ PHÁT HẠT")

        # Clear tracks button
        clear_rect = QRectF(inject_rect.right() + 8, y_bar, 88, btn_h)
        is_hover_clr = self._hovered_button == "clear_tracks"
        painter.setPen(QPen(QColor(56, 189, 248) if is_hover_clr else QColor(40, 70, 105), 1.2))
        painter.setBrush(QBrush(QColor(24, 52, 85, 200 if is_hover_clr else 140)))
        painter.drawRoundedRect(clear_rect, 6.0, 6.0)
        painter.drawText(clear_rect, Qt.AlignCenter, "↺ XÓA VỆT")

        # Status text
        info_rect = QRectF(clear_rect.right() + 12, y_bar, canvas_rect.right() - clear_rect.right() - 12, btn_h)
        painter.setFont(QFont("Segoe UI", 8, QFont.Normal))
        painter.setPen(QPen(QColor(160, 195, 235)))
        active_cnt = len(self._tracks)
        info_str = f"Vệt sương đang hiện: {active_cnt}  ·  T_base = -32°C  ·  R_chamber = 11.5 cm"
        painter.drawText(info_rect, Qt.AlignVCenter | Qt.AlignLeft, info_str)

    def _paint_control_dashboard(self, painter: QPainter) -> None:
        """Draw sliders, live nuclear physics counters, and particle telemetry."""
        dash_rect = QRectF(596, 96, 324, 534)
        painter.setPen(QPen(QColor(30, 48, 75, 160), 1.0))
        painter.setBrush(QBrush(QColor(10, 16, 28, 230)))
        painter.drawRoundedRect(dash_rect, 10.0, 10.0)

        y = dash_rect.top() + 14

        # Section 1: Dynamic Sliders
        header_font = QFont("Segoe UI", 9, QFont.Bold)
        painter.setFont(header_font)
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(QRectF(dash_rect.left() + 16, y, 290, 20), Qt.AlignLeft, "ĐIỀU KHIỂN BUỒNG SƯƠNG & TỪ TRƯỜNG")
        y += 26

        # Slider 1: Magnetic Field B
        self._paint_slider(
            painter,
            slider_id="b_field",
            label="Từ trường Lorentz B",
            val_text=f"{self._b_field_tesla:+.2f} T",
            min_val=-0.50,
            max_val=0.50,
            current_val=self._b_field_tesla,
            y_pos=y,
            dash_left=dash_rect.left(),
        )
        y += 54

        # Slider 2: Decay Emission Rate (CPM)
        self._paint_slider(
            painter,
            slider_id="rate",
            label="Tốc độ đếm phân rã (CPM)",
            val_text=f"{self._decay_rate_cpm:.0f} CPM ({self._decay_rate_cpm/60.0:.1f} Bq)",
            min_val=5.0,
            max_val=150.0,
            current_val=self._decay_rate_cpm,
            y_pos=y,
            dash_left=dash_rect.left(),
        )
        y += 54

        # Slider 3: Vapor Supersaturation S
        self._paint_slider(
            painter,
            slider_id="supersat",
            label="Độ quá bão hòa hơi cồn S",
            val_text=f"S = {self._supersaturation:.1f} (Ngưỡng ngưng tụ)",
            min_val=2.0,
            max_val=6.0,
            current_val=self._supersaturation,
            y_pos=y,
            dash_left=dash_rect.left(),
        )
        y += 56

        # Section 2: Nuclear Counting Statistics Box
        stats_rect = QRectF(dash_left := dash_rect.left() + 14, y, 296, 126)
        painter.setPen(QPen(QColor(40, 65, 100), 1.0))
        painter.setBrush(QBrush(QColor(16, 25, 42, 220)))
        painter.drawRoundedRect(stats_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(120, 180, 240)))
        painter.drawText(QRectF(stats_rect.left() + 10, stats_rect.top() + 8, 276, 18), "THỐNG KÊ ĐẾM BỨC XẠ ION HÓA:")

        metrics = [
            ("Hạt Alpha (⁴He²⁺):", f"{self._total_counts['alpha']} vệt (Dày, Thẳng, LET cao)"),
            ("Hạt Beta (e⁻):", f"{self._total_counts['beta']} vệt (Mảnh, Uốn cong)"),
            ("Tia Muon Vũ trụ (μ):", f"{self._total_counts['muon']} vệt (Xuyên thẳng cứng)"),
            ("Độ cong cyclotron R:", f"R = p/(qB) = {abs(0.5 / max(0.01, abs(self._b_field_tesla))):.1f} cm" if abs(self._b_field_tesla) > 0.01 else "Thẳng (B = 0)"),
        ]

        painter.setFont(QFont("Segoe UI", 7, QFont.Normal))
        for m_idx, (m_lbl, m_val) in enumerate(metrics):
            my = stats_rect.top() + 30 + m_idx * 21
            painter.setPen(QPen(QColor(180, 205, 230)))
            painter.drawText(QRectF(stats_rect.left() + 10, my, 130, 18), Qt.AlignLeft, m_lbl)
            painter.setPen(QPen(QColor(250, 204, 21) if "cyclotron" in m_lbl else QColor(74, 222, 128)))
            painter.drawText(QRectF(stats_rect.right() - 156, my, 146, 18), Qt.AlignRight, m_val)

        y += 140

        # Section 3: Educational Particle Recognition Guide Box
        guide_rect = QRectF(dash_left, y, 296, 154)
        painter.setPen(QPen(QColor(40, 65, 100), 1.0))
        painter.setBrush(QBrush(QColor(12, 20, 34, 230)))
        painter.drawRoundedRect(guide_rect, 8.0, 8.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(QRectF(guide_rect.left() + 10, guide_rect.top() + 8, 276, 18), "HƯỚNG DẪN NHẬN DIỆN VỆT SƯƠNG:")

        guide_items = [
            ("Alpha (α):", "Vệt trắng đục dày đặc ~4 cm, thẳng tắp, Bragg peak."),
            ("Beta (β⁻):", "Vệt mảnh ngoằn ngoèo do tán xạ electron, uốn theo B."),
            ("Muon (μ):", "Đường thẳng tắp sắc bén cắt ngang toàn bộ buồng sương."),
            ("Lorentz:", "B tách điện tích: hạt dương lệch phải, hạt âm lệch trái."),
        ]

        for g_idx, (g_tag, g_desc) in enumerate(guide_items):
            gy = guide_rect.top() + 30 + g_idx * 28
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QPen(QColor(250, 204, 21) if "Lorentz" in g_tag else QColor(100, 220, 255)))
            painter.drawText(QRectF(guide_rect.left() + 10, gy, 70, 16), g_tag)

            painter.setFont(QFont("Segoe UI", 7, QFont.Normal))
            painter.setPen(QPen(QColor(200, 220, 245)))
            painter.drawText(QRectF(guide_rect.left() + 80, gy, 206, 26), Qt.TextWordWrap, g_desc)

    def _paint_slider(
        self,
        painter: QPainter,
        slider_id: str,
        label: str,
        val_text: str,
        min_val: float,
        max_val: float,
        current_val: float,
        y_pos: float,
        dash_left: float,
    ) -> None:
        """Render a styled draggable slider track with knob."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Normal))
        painter.setPen(QPen(QColor(180, 205, 230)))
        painter.drawText(QRectF(dash_left + 16, y_pos, 160, 16), Qt.AlignLeft, label)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(QRectF(dash_left + 170, y_pos, 136, 16), Qt.AlignRight, val_text)

        # Track
        track_rect = QRectF(dash_left + 16, y_pos + 20, 292, 8)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(25, 40, 62)))
        painter.drawRoundedRect(track_rect, 4.0, 4.0)

        # Filled portion
        fraction = max(0.0, min(1.0, (current_val - min_val) / (max_val - min_val)))
        fill_w = max(6.0, fraction * track_rect.width())
        fill_rect = QRectF(track_rect.left(), track_rect.top(), fill_w, track_rect.height())
        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.drawRoundedRect(fill_rect, 4.0, 4.0)

        # Knob
        knob_x = track_rect.left() + fraction * track_rect.width()
        knob_pt = QPointF(knob_x, track_rect.center().y())
        is_active = (self._active_slider == slider_id)
        painter.setPen(QPen(QColor(255, 255, 255) if is_active else QColor(180, 220, 255), 1.5))
        painter.setBrush(QBrush(QColor(56, 189, 248) if is_active else QColor(30, 80, 140)))
        painter.drawEllipse(knob_pt, 7.0, 7.0)

    # -------------------------------------------------------------------------
    # Mouse & Touchless Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # 1. Close button
        close_rect = QRectF(self._width - 36, 12, 24, 24)
        if close_rect.contains(pos):
            self.closed.emit()
            if self.board and hasattr(self.board, "toggle_cloud_chamber"):
                self.board.toggle_cloud_chamber()
            event.accept()
            return

        # 2. Preset tabs
        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0
        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            rect = QRectF(start_x + i * (btn_w + 8.0), y, btn_w, btn_h)
            if rect.contains(pos):
                self.set_preset(key)
                event.accept()
                return

        # 3. Canvas bottom bar controls
        canvas_rect = QRectF(20, 96, 560, 470)
        y_bar = canvas_rect.bottom() + 10
        play_rect = QRectF(canvas_rect.left(), y_bar, 96, 28)
        if play_rect.contains(pos):
            self.toggle_play_pause()
            event.accept()
            return

        inject_rect = QRectF(play_rect.right() + 8, y_bar, 110, 28)
        if inject_rect.contains(pos):
            self.spawn_particle_track()
            event.accept()
            return

        clear_rect = QRectF(inject_rect.right() + 8, y_bar, 88, 28)
        if clear_rect.contains(pos):
            self.clear_tracks()
            event.accept()
            return

        # 4. Sliders
        dash_rect = QRectF(596, 96, 324, 534)
        dash_left = dash_rect.left()
        slider_w = 292.0

        b_track = QRectF(dash_left + 16, dash_rect.top() + 60, slider_w, 20)
        if b_track.contains(pos):
            self._active_slider = "b_field"
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, -0.50, 0.50, self.set_b_field)
            event.accept()
            return

        rate_track = QRectF(dash_left + 16, dash_rect.top() + 114, slider_w, 20)
        if rate_track.contains(pos):
            self._active_slider = "rate"
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 5.0, 150.0, self.set_decay_rate)
            event.accept()
            return

        sat_track = QRectF(dash_left + 16, dash_rect.top() + 168, slider_w, 20)
        if sat_track.contains(pos):
            self._active_slider = "supersat"
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 2.0, 6.0, self.set_supersaturation)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        dash_rect = QRectF(596, 96, 324, 534)
        dash_left = dash_rect.left()
        slider_w = 292.0

        if self._active_slider == "b_field":
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, -0.50, 0.50, self.set_b_field)
            event.accept()
            return
        elif self._active_slider == "rate":
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 5.0, 150.0, self.set_decay_rate)
            event.accept()
            return
        elif self._active_slider == "supersat":
            self._handle_slider_move(pos.x(), dash_left + 16, slider_w, 2.0, 6.0, self.set_supersaturation)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._active_slider = None
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event: Any) -> None:
        pos = event.pos()
        old_hover = self._hovered_button
        self._hovered_button = None

        close_rect = QRectF(self._width - 36, 12, 24, 24)
        if close_rect.contains(pos):
            self._hovered_button = "close"

        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0
        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            rect = QRectF(start_x + i * (btn_w + 8.0), y, btn_w, btn_h)
            if rect.contains(pos):
                self._hovered_button = f"preset_{key}"

        canvas_rect = QRectF(20, 96, 560, 470)
        y_bar = canvas_rect.bottom() + 10
        if QRectF(canvas_rect.left(), y_bar, 96, 28).contains(pos):
            self._hovered_button = "toggle_play"
        elif QRectF(canvas_rect.left() + 104, y_bar, 110, 28).contains(pos):
            self._hovered_button = "inject_particle"
        elif QRectF(canvas_rect.left() + 222, y_bar, 88, 28).contains(pos):
            self._hovered_button = "clear_tracks"

        if old_hover != self._hovered_button:
            self.update()

        super().hoverMoveEvent(event)

    def _handle_slider_move(
        self,
        mouse_x: float,
        track_left: float,
        track_width: float,
        min_val: float,
        max_val: float,
        setter_func: Any,
    ) -> None:
        fraction = max(0.0, min(1.0, (mouse_x - track_left) / track_width))
        val = min_val + fraction * (max_val - min_val)
        setter_func(val)

    # -------------------------------------------------------------------------
    # Touchless Webcam Air Gestures
    # -------------------------------------------------------------------------

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[Any] = None) -> bool:
        """Process air tap gesture on presets, buttons, or sliders."""
        pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(pos):
            return False

        # Close button
        if QRectF(self._width - 36, 12, 24, 24).contains(pos):
            self.closed.emit()
            if self.board and hasattr(self.board, "toggle_cloud_chamber"):
                self.board.toggle_cloud_chamber()
            return True

        # Presets
        btn_w = 176.0
        btn_h = 28.0
        start_x = 20.0
        y = 58.0
        preset_keys = list(PRESETS.keys())
        for i, key in enumerate(preset_keys):
            if QRectF(start_x + i * (btn_w + 8.0), y, btn_w, btn_h).contains(pos):
                self.set_preset(key)
                return True

        # Canvas buttons
        canvas_rect = QRectF(20, 96, 560, 470)
        y_bar = canvas_rect.bottom() + 10
        if QRectF(canvas_rect.left(), y_bar, 96, 28).contains(pos):
            self.toggle_play_pause()
            return True
        elif QRectF(canvas_rect.left() + 104, y_bar, 110, 28).contains(pos):
            self.spawn_particle_track()
            return True
        elif QRectF(canvas_rect.left() + 222, y_bar, 88, 28).contains(pos):
            self.clear_tracks()
            return True

        return True

    # -------------------------------------------------------------------------
    # Serialization & Canvas Stamping
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for .airteach packaging."""
        return {
            "type": "InteractiveCloudChamberItem",
            "pos": [self.x(), self.y()],
            "preset": self._current_preset,
            "b_field_tesla": self._b_field_tesla,
            "decay_rate_cpm": self._decay_rate_cpm,
            "supersaturation": self._supersaturation,
            "source_type": self._source_type,
            "counts": dict(self._total_counts),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        board: Optional["LessonBoard"] = None,
    ) -> "InteractiveCloudChamberItem":
        """Reconstruct from dictionary."""
        pos = data.get("pos", [80.0, 80.0])
        item = cls(board=board, pos=QPointF(pos[0], pos[1]))
        preset = data.get("preset", "ALPHA_AMERICIUM")
        item.set_preset(preset)
        if "b_field_tesla" in data:
            item.set_b_field(data["b_field_tesla"])
        if "decay_rate_cpm" in data:
            item.set_decay_rate(data["decay_rate_cpm"])
        if "supersaturation" in data:
            item.set_supersaturation(data["supersaturation"])
        if "counts" in data:
            item._total_counts.update(data["counts"])
        return item

    def stamp_to_canvas(self) -> "InteractiveCloudChamberItem":
        """Duplicate snapshot item onto lesson canvas."""
        clone = InteractiveCloudChamberItem(
            board=self.board,
            pos=(self.x() + 30.0, self.y() + 30.0),
        )
        clone.set_preset(self._current_preset)
        clone.set_b_field(self._b_field_tesla)
        clone.set_decay_rate(self._decay_rate_cpm)
        clone.set_supersaturation(self._supersaturation)
        clone._total_counts = dict(self._total_counts)
        if self.scene():
            self.scene().addItem(clone)
        return clone
