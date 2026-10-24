"""
AirTeach AI — Applied Electromagnetism & Modern Physics:
Mass Spectrometer, Wien Velocity Selector & Isotopic Deflection Simulator.

Simulates:
  - Wien Velocity Selector (cross-field filter E ⟂ B1):
      F_E = q * E,  F_B = q * v * B1
      Undeflected path condition: q * E = q * v * B1  ==>  v_0 = E / B1
  - Semicircular Magnetic Deflection Chamber (B2 ⟂ v):
      F_B = q * v * B2 = (m * v²) / r
      Deflection orbit radius: r = (m * v_0) / (q * B2) = (m * E) / (q * B1 * B2)
      Detector impact diameter: D = 2 * r = (2 * m * v_0) / (q * B2)
  - Mass-to-charge ratio determination (m/z):
      m / q = (r * B2) / v_0 = (r * B1 * B2) / E
  - Authentic isotopic abundance spectra:
      Chlorine (35Cl, 37Cl), Carbon (12C, 13C), Uranium (235U, 238U),
      Hydrogen/Deuterium (1H, 2H), Lithium (6Li, 7Li).
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QTimer
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QBrush,
    QFont,
    QPainterPath,
    QLinearGradient,
    QRadialGradient,
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


# Standard Physical Constants
AMU_KG: float = 1.66053906660e-27       # 1 atomic mass unit (u) in kg
E_CHARGE_C: float = 1.602176634e-19     # Elementary charge e in Coulombs


@dataclass(frozen=True)
class IsotopeData:
    """Represents a specific isotopic ion species."""
    symbol: str
    mass_u: float           # Mass in atomic mass units (u)
    charge_e: float         # Charge in elementary charge units (+1, +2, etc.)
    abundance_pct: float    # Natural abundance in percent [0, 100]
    color: QColor


@dataclass(frozen=True)
class MassSpectrometerPreset:
    """Curriculum scenario for Mass Spectrometry & Wien Filter."""
    code: str
    title_vi: str
    badge_label: str
    element_name: str
    default_electric_field_V_m: float   # E in V/m
    default_b1_Tesla: float             # B1 in Wien selector (Tesla)
    default_b2_Tesla: float             # B2 in deflection chamber (Tesla)
    isotopes: list[IsotopeData]
    description: str


SPECTROMETER_PRESETS: dict[str, MassSpectrometerPreset] = {
    "CHLORINE_ISOTOPE_35_37": MassSpectrometerPreset(
        code="CHLORINE_ISOTOPE_35_37",
        title_vi="Khối Phổ Đồng Vị Clo — Tỷ Lệ Tự Nhiên 3:1 (³⁵Cl⁺ & ³⁷Cl⁺)",
        badge_label="CHLORINE 3:1",
        element_name="Clo (Chlorine)",
        default_electric_field_V_m=50000.0,  # 50 kV/m
        default_b1_Tesla=0.20,               # v0 = 50000 / 0.20 = 2.5e5 m/s
        default_b2_Tesla=0.30,
        isotopes=[
            IsotopeData("³⁵Cl⁺", 34.969, 1.0, 75.78, QColor("#38bdf8")),
            IsotopeData("³⁷Cl⁺", 36.966, 1.0, 24.22, QColor("#f43f5e")),
        ],
        description="Thí nghiệm kinh điển của Francis Aston (giải Nobel Hóa học 1922): phát hiện các nguyên tử cùng điện tích hạt nhân nhưng có khối lượng khác nhau (đồng vị). Phổ khối clo cho hai vạch với tỷ lệ diện tích xấp xỉ 3:1.",
    ),
    "CARBON_12_13_GEOCHEMISTRY": MassSpectrometerPreset(
        code="CARBON_12_13_GEOCHEMISTRY",
        title_vi="Đồng Vị Carbon — Phân Tích Cổ Khí Hậu & Chu Trình Sinh Địa Hóa (¹²C & ¹³C)",
        badge_label="CARBON 12/13",
        element_name="Cacbon (Carbon)",
        default_electric_field_V_m=40000.0,
        default_b1_Tesla=0.20,               # v0 = 2.0e5 m/s
        default_b2_Tesla=0.35,
        isotopes=[
            IsotopeData("¹²C⁺", 12.000, 1.0, 98.93, QColor("#10b981")),
            IsotopeData("¹³C⁺", 13.003, 1.0, 1.07, QColor("#fbbf24")),
        ],
        description="Phân tích tỷ lệ đồng vị bền δ¹³C bằng khối phổ độ phân giải cao trong nghiên cứu cổ khí hậu, nguồn gốc dầu khí và cơ chế quang hợp thực vật C3/C4.",
    ),
    "URANIUM_235_238_CALUTRON": MassSpectrometerPreset(
        code="URANIUM_235_238_CALUTRON",
        title_vi="Làm Giàu Uranium Calutron — Dự Án Manhattan & Phân Tách Điện Từ",
        badge_label="CALUTRON 235/238",
        element_name="Uranium (U)",
        default_electric_field_V_m=60000.0,
        default_b1_Tesla=0.25,               # v0 = 2.4e5 m/s
        default_b2_Tesla=0.45,
        isotopes=[
            IsotopeData("²³⁵U⁺", 235.044, 1.0, 0.72, QColor("#a855f7")),
            IsotopeData("²³⁸U⁺", 238.051, 1.0, 99.28, QColor("#38bdf8")),
        ],
        description="Máy phân tách đồng vị điện từ Calutron do Ernest Lawrence thiết kế tại phòng thí nghiệm Oak Ridge (1943): tận dụng bán kính uốn cong khác biệt 1.28% trong từ trường để tách chiết 235U phục vụ năng lượng hạt nhân.",
    ),
    "HYDROGEN_DEUTERIUM_HEAVY_WATER": MassSpectrometerPreset(
        code="HYDROGEN_DEUTERIUM_HEAVY_WATER",
        title_vi="Hydro & Deuterium — Khối Lượng Gấp Đôi (¹H⁺ & ²H⁺ Nước Nặng)",
        badge_label="HYDROGEN / DEUTERIUM",
        element_name="Hydro (H) & Deuterium (D)",
        default_electric_field_V_m=30000.0,
        default_b1_Tesla=0.15,               # v0 = 2.0e5 m/s
        default_b2_Tesla=0.25,
        isotopes=[
            IsotopeData("¹H⁺", 1.008, 1.0, 99.985, QColor("#06b6d4")),
            IsotopeData("²H⁺ (D)", 2.014, 1.0, 0.015, QColor("#f59e0b")),
        ],
        description="Chênh lệch khối lượng 100% giữa hạt proton thông thường (1H) và hạt deuteron (2H = 1p + 1n) làm bán kính quỹ đạo trong buồng từ trường tăng gấp đôi chính xác r_D = 2 · r_H.",
    ),
    "LITHIUM_6_7_BATTERY_TECH": MassSpectrometerPreset(
        code="LITHIUM_6_7_BATTERY_TECH",
        title_vi="Đồng Vị Liti — Công Nghệ Pin Năng Lượng & Nhiệt Hạch (⁶Li & ⁷Li)",
        badge_label="LITHIUM 6/7",
        element_name="Liti (Lithium)",
        default_electric_field_V_m=35000.0,
        default_b1_Tesla=0.175,              # v0 = 2.0e5 m/s
        default_b2_Tesla=0.28,
        isotopes=[
            IsotopeData("⁶Li⁺", 6.015, 1.0, 7.59, QColor("#ec4899")),
            IsotopeData("⁷Li⁺", 7.016, 1.0, 92.41, QColor("#38bdf8")),
        ],
        description="Đồng vị 6Li có tiết diện bắt neutron nhiệt cực lớn ứng dụng sản xuất Tritium trong lò phản ứng nhiệt hạch Tokamak, trong khi 7Li được dùng làm chất điều hòa nước làm mát lò hạt nhân PWR.",
    ),
}


@dataclass
class AnimatedIonTracer:
    """Animated ion traversing the spectrometer."""
    iso_index: int
    progress: float     # Progress along trajectory [0.0, 1.0]
    speed: float
    color: QColor
    size: float


class InteractiveMassSpectrometerItem(QGraphicsObject):
    """
    Interactive PySide6 Graphics Item simulating the Wien velocity selector,
    magnetic deflection chamber, mass-to-charge (m/z) separation, and isotope detection.
    """

    spectrometry_computed = Signal(float, float, float)  # v0_m_s, r_iso0_mm, r_iso1_mm
    stamp_requested = Signal(dict)

    CARD_W = 740.0
    CARD_H = 650.0

    def __init__(self, pos: tuple[float, float] = (80.0, 40.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation state
        self._active_preset_code: str = "CHLORINE_ISOTOPE_35_37"
        self._preset: MassSpectrometerPreset = SPECTROMETER_PRESETS[self._active_preset_code]

        self._e_field_V_m: float = self._preset.default_electric_field_V_m
        self._b1_Tesla: float = self._preset.default_b1_Tesla
        self._b2_Tesla: float = self._preset.default_b2_Tesla

        # Ion tracer particles
        self._ions: list[AnimatedIonTracer] = []
        self._init_ion_tracers(num_ions=18)

        # Animation state
        self._sim_time: float = 0.0
        self._is_running: bool = True

        # UI & Buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Timer for 60fps real-time physics animation
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start(16)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # -------------------------------------------------------------------------
    # Physical & Electromagnetic Properties
    # -------------------------------------------------------------------------
    @property
    def selected_velocity_m_s(self) -> float:
        """
        Wien velocity selector filter speed:
        v_0 = E / B1
        """
        b1 = max(0.01, self._b1_Tesla)
        return self._e_field_V_m / b1

    def deflection_radius_m(self, iso: IsotopeData) -> float:
        """
        Deflection orbit radius inside magnetic chamber B2:
        r = (m * v_0) / (q * B2)
        """
        m_kg = iso.mass_u * AMU_KG
        q_c = iso.charge_e * E_CHARGE_C
        b2 = max(0.01, self._b2_Tesla)
        v0 = self.selected_velocity_m_s
        return (m_kg * v0) / (q_c * b2)

    def deflection_radius_mm(self, iso: IsotopeData) -> float:
        """Orbit radius r in millimeters."""
        return self.deflection_radius_m(iso) * 1000.0

    def detector_impact_distance_mm(self, iso: IsotopeData) -> float:
        """Detector separation diameter D = 2 * r in millimeters."""
        return 2.0 * self.deflection_radius_mm(iso)

    # -------------------------------------------------------------------------
    # Ion Tracer Lifecycle
    # -------------------------------------------------------------------------
    def _init_ion_tracers(self, num_ions: int = 18) -> None:
        """Initializes distributed ions in the spectrometer."""
        self._ions.clear()
        isos = self._preset.isotopes
        weights = [iso.abundance_pct for iso in isos]
        total_w = sum(weights) if sum(weights) > 0 else 1.0

        for _ in range(num_ions):
            # Select isotope weighted by abundance
            r_val = random.uniform(0.0, total_w)
            cum = 0.0
            chosen_idx = 0
            for idx, w in enumerate(weights):
                cum += w
                if r_val <= cum:
                    chosen_idx = idx
                    break

            prog = random.uniform(0.0, 1.0)
            spd = random.uniform(0.010, 0.016)
            iso = isos[chosen_idx]
            self._ions.append(AnimatedIonTracer(chosen_idx, prog, spd, iso.color, 3.5))

    def _on_sim_tick(self) -> None:
        """Updates ion tracer progress along the trajectory."""
        if not self._is_running:
            return

        dt = 0.016
        self._sim_time += dt
        isos = self._preset.isotopes

        for ion in self._ions:
            ion.progress += ion.speed
            if ion.progress > 1.0:
                # Re-spawn ion at nozzle
                ion.progress = 0.0
                weights = [iso.abundance_pct for iso in isos]
                r_val = random.uniform(0.0, sum(weights))
                cum = 0.0
                for idx, w in enumerate(weights):
                    cum += w
                    if r_val <= cum:
                        ion.iso_index = idx
                        break
                ion.color = isos[ion.iso_index].color

        self.update()

    # -------------------------------------------------------------------------
    # Controls & Presets
    # -------------------------------------------------------------------------
    def set_preset(self, code: str) -> None:
        """Switches to a curriculum scenario preset."""
        if code not in SPECTROMETER_PRESETS:
            return
        self._active_preset_code = code
        self._preset = SPECTROMETER_PRESETS[code]
        self._e_field_V_m = self._preset.default_electric_field_V_m
        self._b1_Tesla = self._preset.default_b1_Tesla
        self._b2_Tesla = self._preset.default_b2_Tesla
        self._init_ion_tracers()

        iso0 = self._preset.isotopes[0]
        iso1 = self._preset.isotopes[1] if len(self._preset.isotopes) > 1 else iso0
        self.spectrometry_computed.emit(
            self.selected_velocity_m_s,
            self.deflection_radius_mm(iso0),
            self.deflection_radius_mm(iso1),
        )
        self.update()

    def set_electric_field_V_m(self, e_val: float) -> None:
        """Sets Wien electric field E in V/m [5000, 200000]."""
        self._e_field_V_m = max(5000.0, min(200000.0, e_val))
        iso0 = self._preset.isotopes[0]
        iso1 = self._preset.isotopes[1] if len(self._preset.isotopes) > 1 else iso0
        self.spectrometry_computed.emit(
            self.selected_velocity_m_s,
            self.deflection_radius_mm(iso0),
            self.deflection_radius_mm(iso1),
        )
        self.update()

    def set_magnetic_field_2_Tesla(self, b2_val: float) -> None:
        """Sets deflection chamber B2 in Tesla [0.05, 2.0]."""
        self._b2_Tesla = max(0.05, min(2.0, b2_val))
        iso0 = self._preset.isotopes[0]
        iso1 = self._preset.isotopes[1] if len(self._preset.isotopes) > 1 else iso0
        self.spectrometry_computed.emit(
            self.selected_velocity_m_s,
            self.deflection_radius_mm(iso0),
            self.deflection_radius_mm(iso1),
        )
        self.update()

    def reset_defaults(self) -> None:
        """Resets parameters to active preset defaults."""
        self._e_field_V_m = self._preset.default_electric_field_V_m
        self._b1_Tesla = self._preset.default_b1_Tesla
        self._b2_Tesla = self._preset.default_b2_Tesla
        iso0 = self._preset.isotopes[0]
        iso1 = self._preset.isotopes[1] if len(self._preset.isotopes) > 1 else iso0
        self.spectrometry_computed.emit(
            self.selected_velocity_m_s,
            self.deflection_radius_mm(iso0),
            self.deflection_radius_mm(iso1),
        )
        self.update()

    # -------------------------------------------------------------------------
    # Layout Buttons
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Defines touch button bounding boxes at the bottom."""
        by = self.CARD_H - 46.0
        bw = 88.0
        bh = 32.0
        spacing = 10.0
        start_x = 18.0

        keys = [
            ("btn_preset", 130.0),
            ("btn_e_down", 72.0),
            ("btn_e_up", 72.0),
            ("btn_b_down", 72.0),
            ("btn_b_up", 72.0),
            ("btn_reset", 66.0),
            ("btn_stamp", 112.0),
        ]

        curr_x = start_x
        for key, width in keys:
            self._buttons[key] = QRectF(curr_x, by, width, bh)
            curr_x += width + spacing

    # -------------------------------------------------------------------------
    # Rendering Pipeline
    # -------------------------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Card Container
        card_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor("#080c14"))
        grad.setColorAt(1.0, QColor("#0f172a"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#1e293b"), 2.0))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Header
        self._paint_header(painter)

        # 3. Left Panel: Mass Spectrometer Apparatus & Trajectories
        self._paint_apparatus_and_trajectories(painter)

        # 4. Right Panel: Mass Spectrum Peak Intensity Chart (m/z)
        self._paint_mass_spectrum_chart(painter)

        # 5. Bottom Quantitative Metrics Panel
        self._paint_metrics_panel(painter)

        # 6. Bottom Touchless Buttons
        self._paint_buttons(painter)

    def _paint_header(self, painter: QPainter) -> None:
        # Badge
        badge_rect = QRectF(20.0, 16.0, 185.0, 24.0)
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"⚡ {self._preset.badge_label}")

        # Title
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(218.0, 33.0, self._preset.title_vi)

        # Subtitle
        v0 = self.selected_velocity_m_s
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            22.0,
            58.0,
            f"Nguyên tố: {self._preset.element_name} | Vận tốc chọn lọc v₀ = E/B₁ = {v0/1000.0:.1f} km/s (E = {self._e_field_V_m/1000.0:.1f} kV/m, B₁ = {self._b1_Tesla:.2f} T, B₂ = {self._b2_Tesla:.2f} T)",
        )

    def _paint_apparatus_and_trajectories(self, painter: QPainter) -> None:
        """Renders the physical spectrometer apparatus: nozzle -> Wien filter -> semicircular chamber."""
        px = 22.0
        py = 74.0
        pw = 430.0
        ph = 250.0

        # Background box
        painter.setBrush(QBrush(QColor("#050811")))
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 12.0, py + 20.0, "1. Ống Phóng Ion, Bộ Lọc Vận Tốc Wien & Buồng Lệch Từ B₂")

        # Layout coordinates
        # Ion Source at top-left
        src_x = px + 25.0
        src_y = py + 40.0
        src_w = 40.0
        src_h = 30.0

        painter.setBrush(QBrush(QColor("#334155")))
        painter.setPen(QPen(QColor("#64748b"), 1.0))
        painter.drawRect(QRectF(src_x, src_y, src_w, src_h))

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(QRectF(src_x, src_y, src_w, src_h), Qt.AlignmentFlag.AlignCenter, "NGUỒN\nION")

        # Wien Velocity Selector (Vertical section below nozzle)
        wien_x = src_x + src_w / 2.0 - 15.0
        wien_y = src_y + src_h + 8.0
        wien_w = 30.0
        wien_h = 65.0

        # Left plate (+), Right plate (-)
        painter.setBrush(QBrush(QColor("#f43f5e")))
        painter.drawRect(QRectF(wien_x - 3.0, wien_y, 3.0, wien_h))
        painter.setBrush(QBrush(QColor("#0ea5e9")))
        painter.drawRect(QRectF(wien_x + wien_w, wien_y, 3.0, wien_h))

        # E-field arrows
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor("#fbbf24")))
        painter.drawText(wien_x + 6.0, wien_y + wien_h / 2.0 - 6.0, "E ➔")
        painter.drawText(wien_x + 6.0, wien_y + wien_h / 2.0 + 8.0, "B₁ ⊙")

        # Collimation Slit (Exit into magnetic chamber)
        slit_y = wien_y + wien_h + 4.0
        slit_x = wien_x + wien_w / 2.0
        painter.setPen(QPen(QColor("#64748b"), 2.0))
        painter.drawLine(QPointF(px + 10.0, slit_y), QPointF(slit_x - 4.0, slit_y))
        painter.drawLine(QPointF(slit_x + 4.0, slit_y), QPointF(px + pw - 10.0, slit_y))

        # Magnetic Deflection Chamber B2 (below slit)
        # Semicircular field area with B2 out of screen symbols (⊙)
        chamb_y = slit_y + 2.0
        chamb_h = ph - (chamb_y - py) - 10.0
        painter.setBrush(QBrush(QColor(14, 165, 233, 15)))
        painter.setPen(QPen(QColor("#1e293b"), 1.0, Qt.PenStyle.DashLine))
        painter.drawRect(QRectF(px + 10.0, chamb_y, pw - 20.0, chamb_h))

        # B2 label
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(slit_x + 15.0, chamb_y + 20.0, "Buồng Từ Trường B₂ ⊙ (Vuông góc hướng ra)")

        # Detector Plane along slit_y to the right
        # Semicircles turn right and hit the detector at distance D = 2r from slit_x
        # We map deflection radius r in mm to pixels
        # Let maximum r be ~80 mm mapped to ~140 px
        iso_list = self._preset.isotopes
        max_r_mm = max(self.deflection_radius_mm(iso) for iso in iso_list)
        px_per_mm = 130.0 / max(1.0, max_r_mm)

        for iso in iso_list:
            r_mm = self.deflection_radius_mm(iso)
            r_px = r_mm * px_per_mm
            cx = slit_x + r_px
            cy = slit_y

            # Draw semicircular path
            semi_path = QPainterPath()
            semi_path.moveTo(slit_x, slit_y)
            # ArcTo(rect, startAngle, sweepAngle)
            # In Qt coords (y downwards):
            # start at (slit_x, slit_y) heading down, turning clockwise up to (slit_x + 2*r_px, slit_y)
            arc_rect = QRectF(slit_x, cy - r_px, 2.0 * r_px, 2.0 * r_px)
            semi_path.arcTo(arc_rect, 180.0, -180.0)

            painter.setPen(QPen(iso.color, 1.8))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(semi_path)

            # Impact point on detector
            impact_x = slit_x + 2.0 * r_px
            painter.setBrush(QBrush(iso.color))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.drawEllipse(QPointF(impact_x, slit_y), 4.0, 4.0)

            # Impact label
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.setPen(QPen(iso.color))
            painter.drawText(impact_x - 12.0, slit_y - 8.0, f"{iso.symbol}")

        # Animate Ion Tracers along the trajectory
        for ion in self._ions:
            iso = iso_list[ion.iso_index]
            r_px = self.deflection_radius_mm(iso) * px_per_mm

            # Phase 1: Moving down through Wien filter [0.0, 0.35]
            if ion.progress <= 0.35:
                t_frac = ion.progress / 0.35
                cur_x = slit_x
                cur_y = (src_y + src_h) + t_frac * (slit_y - (src_y + src_h))
            else:
                # Phase 2: Semicircular magnetic arc [0.35, 1.0]
                arc_frac = (ion.progress - 0.35) / 0.65
                # angle goes from pi (180 deg) down to 0 (0 deg)
                theta = math.pi * (1.0 - arc_frac)
                center_x = slit_x + r_px
                cur_x = center_x - r_px * math.cos(theta)
                cur_y = slit_y + r_px * math.sin(theta)

            painter.setBrush(QBrush(ion.color))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.drawEllipse(QPointF(cur_x, cur_y), ion.size / 2.0, ion.size / 2.0)

    def _paint_mass_spectrum_chart(self, painter: QPainter) -> None:
        """Renders the discrete mass spectrum bar chart (m/z vs relative abundance)."""
        px = 464.0
        py = 74.0
        pw = 254.0
        ph = 250.0

        # Background box
        painter.setBrush(QBrush(QColor("#050811")))
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 12.0, py + 20.0, "2. Phổ Khối Đồng Vị (m/z)")

        # Spectrum axes
        ox = px + 40.0
        oy = py + ph - 38.0
        ax_w = pw - 55.0
        ax_h = 160.0

        ax_pen = QPen(QColor("#64748b"), 1.2)
        painter.setPen(ax_pen)
        painter.drawLine(QPointF(ox, oy), QPointF(ox + ax_w + 6.0, oy))       # m/z axis
        painter.drawLine(QPointF(ox, oy), QPointF(ox, oy - ax_h - 6.0))       # Abundance % axis

        # Axis labels
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(ox + ax_w - 32.0, oy + 15.0, "m/z (u/e)")
        painter.drawText(ox - 30.0, oy - ax_h - 8.0, "Cường độ (%)")

        # 100% grid line
        painter.setPen(QPen(QColor("#1e293b"), 0.8, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(ox, oy - ax_h), QPointF(ox + ax_w, oy - ax_h))
        painter.drawText(ox - 28.0, oy - ax_h + 4.0, "100%")

        # Draw Isotope Spectral Peaks
        isos = self._preset.isotopes
        num_iso = len(isos)
        col_w = min(28.0, (ax_w - 30.0) / (num_iso * 1.5))

        for i, iso in enumerate(isos):
            bx = ox + 25.0 + i * (col_w + 45.0)
            bh = (iso.abundance_pct / 100.0) * ax_h
            by = oy - bh

            # Peak bar
            painter.setBrush(QBrush(iso.color))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.drawRoundedRect(QRectF(bx, by, col_w, bh), 3.0, 3.0)

            # Isotope name & m/z below
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.setPen(QPen(iso.color))
            painter.drawText(bx - 4.0, oy + 14.0, f"{iso.symbol}")
            painter.setFont(QFont("Consolas", 6))
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.drawText(bx - 6.0, oy + 26.0, f"{iso.mass_u:.1f}")

            # Percentage above peak
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(bx - 6.0, by - 6.0, f"{iso.abundance_pct:.1f}%")

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """Renders live quantitative parameters, deflection radii, and m/z formula."""
        px = 22.0
        py = 336.0
        pw = self.CARD_W - 44.0
        ph = 142.0

        # Background container
        painter.setBrush(QBrush(QColor("#09101c")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Header of metrics
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 14.0, py + 20.0, "3. Thông Số Vật Lý & Công Thức Bán Kính Lệch: r = (m·v₀) / (q·B₂) = (m·E) / (q·B₁·B₂)")

        # Column 1
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#e2e8f0")))
        col1_x = px + 14.0
        line1_y = py + 44.0
        dy = 22.0

        v0 = self.selected_velocity_m_s
        painter.drawText(col1_x, line1_y, f"• Vận tốc qua lọc Wien: v₀ = E/B₁ = {v0:,.0f} m/s ({v0/1000.0:.2f} km/s)")
        painter.drawText(col1_x, line1_y + dy, f"• Cường độ điện trường E: {self._e_field_V_m:,.0f} V/m ({self._e_field_V_m/1000.0:.1f} kV/m)")
        painter.drawText(col1_x, line1_y + 2 * dy, f"• Từ trường lọc Wien B₁: {self._b1_Tesla:.3f} T (Lực Fe = qE cân bằng Fb = qvB₁)")
        painter.drawText(col1_x, line1_y + 3 * dy, f"• Từ trường buồng lệch B₂: {self._b2_Tesla:.3f} T (Tạo quỹ đạo nửa đường tròn)")

        # Column 2: Deflection Radii for Isotopes
        col2_x = px + 360.0
        for idx, iso in enumerate(self._preset.isotopes[:3]):
            r_mm = self.deflection_radius_mm(iso)
            d_mm = self.detector_impact_distance_mm(iso)
            painter.drawText(
                col2_x,
                line1_y + idx * dy,
                f"• Đồng vị {iso.symbol} (m = {iso.mass_u:.3f} u): Bán kính r = {r_mm:.2f} mm ➔ Đường kính D = {d_mm:.2f} mm",
            )

        if len(self._preset.isotopes) >= 2:
            dr = abs(self.deflection_radius_mm(self._preset.isotopes[1]) - self.deflection_radius_mm(self._preset.isotopes[0]))
            painter.drawText(col2_x, line1_y + 3 * dy, f"• Độ phân giải tách biệt đỉnh: Δr = {dr:.2f} mm (ΔD = {2.0 * dr:.2f} mm)")

        # Explanatory Educational Text at bottom
        desc_rect = QRectF(px + 12.0, py + ph + 8.0, pw - 24.0, 95.0)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(desc_rect, Qt.TextFlag.TextWordWrap, self._preset.description)

    def _paint_buttons(self, painter: QPainter) -> None:
        """Renders interactive touch/mouse control buttons."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        labels = {
            "btn_preset": f"📋 Kịch Bản ({self._preset.badge_label})",
            "btn_e_down": "E - 10%",
            "btn_e_up": "E + 10%",
            "btn_b_down": "B₂ - 10%",
            "btn_b_up": "B₂ + 10%",
            "btn_reset": "🔄 Đặt Lại",
            "btn_stamp": "📌 Đóng Dấu Bảng",
        }

        for key, rect in self._buttons.items():
            is_hover = (key == self._hovered_btn)
            text = labels.get(key, key)

            if key == "btn_stamp":
                bg = QColor("#0284c7") if not is_hover else QColor("#0369a1")
                border = QColor("#38bdf8")
            elif key == "btn_preset":
                bg = QColor("#4338ca") if not is_hover else QColor("#3730a3")
                border = QColor("#818cf8")
            elif "up" in key:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#38bdf8")
            elif "down" in key:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#f43f5e")
            else:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#64748b")

            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.2))
            painter.drawRoundedRect(rect, 5.0, 5.0)

            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # -------------------------------------------------------------------------
    # Mouse & Touch Event Handling
    # -------------------------------------------------------------------------
    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_btn = self._hovered_btn
        self._hovered_btn = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._hovered_btn = key
                break
        if self._hovered_btn != old_btn:
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_btn is not None:
            self._hovered_btn = None
            self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            for key, rect in self._buttons.items():
                if rect.contains(pos):
                    self._on_button_clicked(key)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _on_button_clicked(self, key: str) -> None:
        """Dispatches button actions."""
        if key == "btn_preset":
            keys = list(SPECTROMETER_PRESETS.keys())
            idx = keys.index(self._active_preset_code)
            next_code = keys[(idx + 1) % len(keys)]
            self.set_preset(next_code)
        elif key == "btn_e_down":
            self.set_electric_field_V_m(self._e_field_V_m * 0.90)
        elif key == "btn_e_up":
            self.set_electric_field_V_m(self._e_field_V_m * 1.10)
        elif key == "btn_b_down":
            self.set_magnetic_field_2_Tesla(self._b2_Tesla * 0.90)
        elif key == "btn_b_up":
            self.set_magnetic_field_2_Tesla(self._b2_Tesla * 1.10)
        elif key == "btn_reset":
            self.reset_defaults()
        elif key == "btn_stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits signal to stamp vector diagram onto whiteboard canvas."""
        iso0 = self._preset.isotopes[0]
        iso1 = self._preset.isotopes[1] if len(self._preset.isotopes) > 1 else iso0
        data = {
            "preset_code": self._active_preset_code,
            "element_name": self._preset.element_name,
            "e_field_V_m": self._e_field_V_m,
            "b1_Tesla": self._b1_Tesla,
            "b2_Tesla": self._b2_Tesla,
            "velocity_m_s": self.selected_velocity_m_s,
            "iso0_symbol": iso0.symbol,
            "iso0_radius_mm": self.deflection_radius_mm(iso0),
            "iso1_symbol": iso1.symbol,
            "iso1_radius_mm": self.deflection_radius_mm(iso1),
        }
        self.stamp_requested.emit(data)
