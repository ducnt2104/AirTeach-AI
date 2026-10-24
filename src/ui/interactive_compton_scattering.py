"""
AirTeach AI - Interactive Quantum Mechanics: Compton Scattering & Relativistic Collision Simulator
==================================================================================================
Simulates authentic relativistic photon-electron scattering (Arthur H. Compton, 1923, Nobel 1927).
Validates quantum theory of light through photon-electron 4-momentum conservation.

Physics Model:
  1. Compton Wavelength Shift:
       Δλ = λ' - λ = λ_C * (1 - cos θ)
       where λ_C = h / (m_e * c) ≈ 2.4263102367e-12 m = 2.4263 pm
  2. Scattered Photon Energy:
       E' = E / [1 + (E / (m_e * c^2)) * (1 - cos θ)]
       where m_e * c^2 ≈ 510.99895 keV (electron rest mass)
  3. Recoil Electron Kinetic Energy:
       T_e = E - E'
  4. Recoil Electron Scattering Angle (φ):
       cot φ = [1 + E / (m_e * c^2)] * tan(θ / 2)
  5. Klein-Nishina Differential Cross-Section:
       dσ/dΩ = (r_e^2 / 2) * P(E, θ)^2 * [P(E, θ) + 1/P(E, θ) - sin^2 θ]
       where r_e = 2.81794e-15 m (classical electron radius)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject, QGraphicsSceneMouseEvent

# Physical constants
COMPTON_WAVELENGTH_PM = 2.4263102  # pm (10^-12 m)
ELECTRON_REST_ENERGY_KEV = 510.99895  # keV
HC_KEV_PM = 1239.84198  # keV * pm (hc product)
CLASSICAL_ELECTRON_RADIUS_FM = 2.81794  # fm (10^-15 m)


@dataclass(frozen=True)
class ComptonPreset:
    name: str
    description: str
    incident_energy_kev: float
    scattering_angle_deg: float
    target_material: str


COMPTON_PRESETS: Dict[str, ComptonPreset] = {
    "MOLYBDENUM_K_ALPHA_GRAPHITE": ComptonPreset(
        name="Moly Kα trên Than chì (Compton 1923)",
        description="Thí nghiệm lịch sử của Arthur Compton dùng tia X Molybdenum (17.4 keV)",
        incident_energy_kev=17.44,
        scattering_angle_deg=90.0,
        target_material="Carbon Graphite (e⁻ tự do)",
    ),
    "COBALT_60_GAMMA_HIGH_ENERGY": ComptonPreset(
        name="Tia Gamma Cobalt-60 (1.25 MeV)",
        description="Tia gamma năng lượng siêu cao: hiệu ứng Compton chiếm ưu thế tuyệt đối",
        incident_energy_kev=1250.0,
        scattering_angle_deg=60.0,
        target_material="Khối chì hấp thụ (Pb)",
    ),
    "COPPER_K_ALPHA_MOMENTUM": ComptonPreset(
        name="Tia X Đồng Kα (8.04 keV)",
        description="Độ dịch bước sóng nhỏ trong vùng tia X thông dụng",
        incident_energy_kev=8.04,
        scattering_angle_deg=135.0,
        target_material="Lá nhôm mỏng (Al)",
    ),
    "EXTREME_BACKSCATTER_180DEG": ComptonPreset(
        name="Tán xạ ngược cực đại (180° Backscatter)",
        description="Độ dịch bước sóng cực đại Δλ = 2*λ_C ≈ 4.85 pm, truyền động lượng tối đa",
        incident_energy_kev=511.0,
        scattering_angle_deg=180.0,
        target_material="Bia tán xạ ngược",
    ),
    "GRAZING_SCATTER_15DEG": ComptonPreset(
        name="Tán xạ góc hẹp (15° Grazing Angle)",
        description="Tán xạ lướt góc hẹp, độ dịch bước sóng rất nhỏ, photon giữ gần nguyên năng lượng",
        incident_energy_kev=100.0,
        scattering_angle_deg=15.0,
        target_material="Beryllium (Be)",
    ),
}


class InteractiveComptonScatteringItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Relativistic Compton Scattering.
    Allows real-time touchless air gesture rotation of scattering angle,
    incident photon energy adjustment, live Klein-Nishina polar cross-section,
    and kinetic energy transfer readout.
    """

    stamp_requested = Signal(dict)
    compton_computed = Signal(dict)

    def __init__(
        self,
        incident_energy_kev: float = 511.0,
        scattering_angle_deg: float = 60.0,
        pos: Optional[Tuple[float, float]] = None,
        parent: Optional[QGraphicsItem] = None,
    ):
        super().__init__(parent)
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        if pos is not None:
            self.setPos(pos[0], pos[1])

        self._incident_energy_kev = float(incident_energy_kev)
        self._scattering_angle_deg = float(scattering_angle_deg)
        self._target_material = "Carbon Graphite"

        self._width = 620.0
        self._height = 420.0

        # Animation state
        self._anim_phase: float = 0.0
        self._is_dragging_angle: bool = False
        self._is_dragging_energy: bool = False

        self._active_preset_name: str = "COBALT_60_GAMMA_HIGH_ENERGY"
        self._stamp_btn_rect = QRectF(0, 0, 0, 0)

    # ── Geometry & Bounding Rect ─────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(-self._width / 2, -self._height / 2, self._width, self._height)

    # ── Physics Calculations ────────────────────────────────────────────────

    @property
    def incident_energy_kev(self) -> float:
        return self._incident_energy_kev

    @property
    def scattering_angle_deg(self) -> float:
        return self._scattering_angle_deg

    def set_incident_energy_kev(self, energy: float) -> None:
        self._incident_energy_kev = max(5.0, min(5000.0, float(energy)))
        self.compton_computed.emit(self.compute_physics())
        self.update()

    def set_scattering_angle_deg(self, angle: float) -> None:
        self._scattering_angle_deg = max(0.0, min(180.0, float(angle)))
        self.compton_computed.emit(self.compute_physics())
        self.update()

    def compute_physics(self) -> Dict[str, float]:
        """Compute full relativistic Compton collision mechanics."""
        theta_rad = math.radians(self._scattering_angle_deg)
        e_in = self._incident_energy_kev
        m_e = ELECTRON_REST_ENERGY_KEV

        # Initial wavelength: λ = hc / E
        lambda_in_pm = HC_KEV_PM / e_in

        # Compton wavelength shift: Δλ = λ_C * (1 - cos θ)
        delta_lambda_pm = COMPTON_WAVELENGTH_PM * (1.0 - math.cos(theta_rad))

        # Scattered wavelength: λ' = λ + Δλ
        lambda_scatter_pm = lambda_in_pm + delta_lambda_pm

        # Scattered photon energy: E' = hc / λ' = E / [1 + (E/m_e)*(1 - cos θ)]
        energy_scatter_kev = e_in / (1.0 + (e_in / m_e) * (1.0 - math.cos(theta_rad)))

        # Recoil electron kinetic energy: T_e = E - E'
        electron_kinetic_kev = e_in - energy_scatter_kev

        # Recoil electron angle φ: cot φ = (1 + E/m_e) * tan(θ/2)
        half_theta = theta_rad / 2.0
        if math.sin(half_theta) == 0.0:
            phi_deg = 0.0
        else:
            cot_phi = (1.0 + e_in / m_e) * math.tan(half_theta)
            phi_rad = math.atan2(1.0, cot_phi)
            phi_deg = math.degrees(phi_rad)

        # Klein-Nishina relative differential cross-section
        p = energy_scatter_kev / e_in  # E'/E
        kn_factor = 0.5 * (p**2) * (p + 1.0 / p - (math.sin(theta_rad) ** 2))

        return {
            "lambda_in_pm": lambda_in_pm,
            "delta_lambda_pm": delta_lambda_pm,
            "lambda_scatter_pm": lambda_scatter_pm,
            "energy_scatter_kev": energy_scatter_kev,
            "electron_kinetic_kev": electron_kinetic_kev,
            "phi_deg": phi_deg,
            "kn_factor": kn_factor,
            "fraction_transferred": electron_kinetic_kev / e_in,
        }

    def apply_preset(self, preset_key: str) -> None:
        """Apply a known experimental preset."""
        preset = COMPTON_PRESETS.get(preset_key)
        if preset:
            self._incident_energy_kev = preset.incident_energy_kev
            self._scattering_angle_deg = preset.scattering_angle_deg
            self._target_material = preset.target_material
            self._active_preset_name = preset_key
            self.compton_computed.emit(self.compute_physics())
            self.update()

    def advance_animation(self, delta_phase: float = 0.1) -> None:
        """Advance wave propagation phase for animated rendering."""
        self._anim_phase = (self._anim_phase + delta_phase) % (2.0 * math.pi)
        self.update()

    def request_stamp(self) -> None:
        """Emit stamp request to permanently convert into vector diagram on board."""
        phys = self.compute_physics()
        data = {
            "center": (self.x(), self.y()),
            "incident_energy_kev": self._incident_energy_kev,
            "scattering_angle_deg": self._scattering_angle_deg,
            "preset_name": COMPTON_PRESETS.get(self._active_preset_name, COMPTON_PRESETS["COBALT_60_GAMMA_HIGH_ENERGY"]).name,
            "target_material": self._target_material,
            "delta_lambda_pm": phys["delta_lambda_pm"],
            "lambda_scatter_pm": phys["lambda_scatter_pm"],
            "energy_scatter_kev": phys["energy_scatter_kev"],
            "electron_kinetic_kev": phys["electron_kinetic_kev"],
            "phi_deg": phys["phi_deg"],
        }
        self.stamp_requested.emit(data)

    # ── Rendering ────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        rect = self.boundingRect()
        phys = self.compute_physics()

        # 1. Dark Technical Background
        bg_brush = QBrush(QColor(15, 20, 32, 245))
        painter.setBrush(bg_brush)
        painter.setPen(QPen(QColor(48, 64, 96), 1.5))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # 2. Header Panel
        self._draw_header(painter, rect, phys)

        # 3. Collision Chamber (Left 60% of width)
        chamber_rect = QRectF(rect.left() + 15, rect.top() + 55, 360, 290)
        self._draw_collision_chamber(painter, chamber_rect, phys)

        # 4. Klein-Nishina Polar Cross-Section & Readout (Right 40%)
        info_rect = QRectF(chamber_rect.right() + 15, rect.top() + 55, 215, 290)
        self._draw_polar_and_readout(painter, info_rect, phys)

        # 5. Interactive Bottom Sliders & Presets Bar
        control_rect = QRectF(rect.left() + 15, rect.bottom() - 65, rect.width() - 30, 55)
        self._draw_controls(painter, control_rect, phys)

    def _draw_header(self, painter: QPainter, rect: QRectF, phys: Dict[str, float]) -> None:
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(QColor(220, 230, 245))
        painter.drawText(
            QRectF(rect.left() + 15, rect.top() + 8, rect.width() - 30, 22),
            Qt.AlignLeft | Qt.AlignVCenter,
            "⚛ TÁN XẠ COMPTON — BẢO TOÀN ĐỘNG LƯỢNG 4-VECTOR RELATIVISTIC",
        )

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(130, 160, 200))
        sub_text = (
            f"Δλ = λ_C(1 - cos θ) = {phys['delta_lambda_pm']:.3f} pm | "
            f"λ_C = 2.426 pm | E = {self._incident_energy_kev:.1f} keV"
        )
        painter.drawText(
            QRectF(rect.left() + 15, rect.top() + 28, rect.width() - 30, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            sub_text,
        )

    def _draw_collision_chamber(
        self, painter: QPainter, rect: QRectF, phys: Dict[str, float]
    ) -> None:
        painter.setBrush(QBrush(QColor(10, 14, 24, 200)))
        painter.setPen(QPen(QColor(35, 48, 75), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        center_x = rect.left() + 120.0
        center_y = rect.center().y()
        center = QPointF(center_x, center_y)

        # Coordinate axes
        painter.setPen(QPen(QColor(60, 80, 110, 120), 1.0, Qt.DashLine))
        painter.drawLine(center_x - 100, center_y, center_x + 220, center_y)
        painter.drawLine(center_x, center_y - 120, center_x, center_y + 120)

        # 1. Incident Photon
        self._draw_wave_packet(
            painter,
            start=QPointF(center_x - 100, center_y),
            end=center,
            color=QColor(0, 220, 255),
            wavelength_visual=10.0,
            label=f"Photon tới\nE = {self._incident_energy_kev:.1f} keV",
        )

        # 2. Target Electron at rest (Origin)
        painter.setBrush(QBrush(QColor(255, 80, 120)))
        painter.setPen(QPen(QColor(255, 200, 220), 1.5))
        painter.drawEllipse(center, 7.0, 7.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(QRectF(center_x - 25, center_y + 10, 50, 15), Qt.AlignCenter, "e⁻ tự do")

        # 3. Scattered Photon at angle θ
        theta_rad = math.radians(self._scattering_angle_deg)
        scat_len = 160.0
        scat_end = QPointF(
            center_x + scat_len * math.cos(theta_rad),
            center_y - scat_len * math.sin(theta_rad),
        )
        self._draw_wave_packet(
            painter,
            start=center,
            end=scat_end,
            color=QColor(255, 200, 50),
            wavelength_visual=14.0,
            label=f"Photon tán xạ\nE' = {phys['energy_scatter_kev']:.1f} keV (θ = {self._scattering_angle_deg:.0f}°)",
        )

        # Arc for angle θ
        arc_rect = QRectF(center_x - 35, center_y - 35, 70, 70)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(255, 200, 50, 180), 1.5))
        painter.drawArc(arc_rect, 0, int(self._scattering_angle_deg * 16))

        # 4. Recoil Electron at angle -φ
        phi_rad = math.radians(phys["phi_deg"])
        recoil_len = 130.0
        recoil_end = QPointF(
            center_x + recoil_len * math.cos(phi_rad),
            center_y + recoil_len * math.sin(phi_rad),
        )
        painter.setPen(QPen(QColor(255, 80, 120), 2.0))
        painter.drawLine(center, recoil_end)
        painter.setBrush(QBrush(QColor(255, 80, 120)))
        painter.drawEllipse(recoil_end, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(255, 140, 170))
        painter.drawText(
            QRectF(recoil_end.x() + 5, recoil_end.y() - 10, 110, 25),
            Qt.AlignLeft,
            f"e⁻ giật lùi\nT_e = {phys['electron_kinetic_kev']:.1f} keV\nφ = {phys['phi_deg']:.1f}°",
        )

        # Arc for recoil angle φ
        phi_arc_rect = QRectF(center_x - 45, center_y - 45, 90, 90)
        painter.setPen(QPen(QColor(255, 80, 120, 180), 1.5))
        painter.drawArc(phi_arc_rect, 0, -int(phys["phi_deg"] * 16))

    def _draw_wave_packet(
        self,
        painter: QPainter,
        start: QPointF,
        end: QPointF,
        color: QColor,
        wavelength_visual: float,
        label: str,
    ) -> None:
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            return

        angle = math.atan2(dy, dx)
        perp_x = -math.sin(angle)
        perp_y = math.cos(angle)

        num_points = 50
        path = QPainterPath()
        path.moveTo(start)

        amp = 6.0
        for i in range(1, num_points + 1):
            t = i / num_points
            curr_dist = t * dist
            wave_val = amp * math.sin(
                (2.0 * math.pi * curr_dist / wavelength_visual) - self._anim_phase
            )
            envelope = math.sin(math.pi * t)
            offset = wave_val * envelope

            px = start.x() + t * dx + offset * perp_x
            py = start.y() + t * dy + offset * perp_y
            path.lineTo(px, py)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(color, 2.0))
        painter.drawPath(path)

        arrow_size = 7.0
        arrow_p1 = QPointF(
            end.x() - arrow_size * math.cos(angle - math.pi / 6),
            end.y() - arrow_size * math.sin(angle - math.pi / 6),
        )
        arrow_p2 = QPointF(
            end.x() - arrow_size * math.cos(angle + math.pi / 6),
            end.y() - arrow_size * math.sin(angle + math.pi / 6),
        )
        arrow_path = QPainterPath()
        arrow_path.moveTo(end)
        arrow_path.lineTo(arrow_p1)
        arrow_path.lineTo(arrow_p2)
        arrow_path.closeSubpath()
        painter.setBrush(QBrush(color))
        painter.drawPath(arrow_path)

        mid_x = (start.x() + end.x()) / 2.0 + 15 * perp_x
        mid_y = (start.y() + end.y()) / 2.0 + 15 * perp_y
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(color)
        painter.drawText(QRectF(mid_x - 60, mid_y - 20, 120, 25), Qt.AlignCenter, label)

    def _draw_polar_and_readout(
        self, painter: QPainter, rect: QRectF, phys: Dict[str, float]
    ) -> None:
        painter.setBrush(QBrush(QColor(18, 24, 38, 220)))
        painter.setPen(QPen(QColor(40, 56, 85), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(200, 220, 245))
        painter.drawText(
            QRectF(rect.left() + 8, rect.top() + 6, rect.width() - 16, 18),
            Qt.AlignLeft,
            "📊 THÔNG SỐ VẬT LÝ & TIẾT DIỆN",
        )

        lines = [
            ("Bước sóng photon tới:", f"{phys['lambda_in_pm']:.3f} pm"),
            ("Độ dịch Compton Δλ:", f"{phys['delta_lambda_pm']:.3f} pm"),
            ("Bước sóng photon tán xạ:", f"{phys['lambda_scatter_pm']:.3f} pm"),
            ("Năng lượng photon tán xạ:", f"{phys['energy_scatter_kev']:.1f} keV"),
            ("Động năng e⁻ giật lùi:", f"{phys['electron_kinetic_kev']:.1f} keV"),
            ("Góc tán xạ photon (θ):", f"{self._scattering_angle_deg:.1f}°"),
            ("Góc giật lùi e⁻ (φ):", f"{phys['phi_deg']:.1f}°"),
            ("Tỷ lệ truyền năng lượng:", f"{phys['fraction_transferred']*100:.1f} %"),
        ]

        y = rect.top() + 28
        painter.setFont(QFont("Segoe UI", 7))
        for label, val in lines:
            painter.setPen(QColor(140, 160, 190))
            painter.drawText(QRectF(rect.left() + 10, y, 120, 15), Qt.AlignLeft, label)
            painter.setPen(QColor(0, 225, 255) if "keV" in val or "%" in val else QColor(255, 230, 140))
            painter.drawText(QRectF(rect.right() - 85, y, 80, 15), Qt.AlignRight, val)
            y += 16

        plot_box = QRectF(rect.left() + 10, y + 5, rect.width() - 20, 110)
        painter.setBrush(QBrush(QColor(10, 14, 22)))
        painter.setPen(QPen(QColor(30, 42, 65), 1.0))
        painter.drawRoundedRect(plot_box, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
        painter.setPen(QColor(160, 180, 210))
        painter.drawText(
            QRectF(plot_box.left() + 5, plot_box.top() + 3, plot_box.width() - 10, 12),
            Qt.AlignLeft,
            "Tiết diện vi phân Klein-Nishina (dσ/dΩ)",
        )

        origin = QPointF(plot_box.left() + 35, plot_box.center().y() + 5)
        painter.setPen(QPen(QColor(50, 70, 100, 100), 1.0, Qt.DotLine))
        painter.drawLine(origin.x(), origin.y() - 40, origin.x(), origin.y() + 40)
        painter.drawLine(origin.x(), origin.y(), origin.x() + 140, origin.y())

        polar_path = QPainterPath()
        polar_path.moveTo(origin)
        e_in = self._incident_energy_kev
        m_e = ELECTRON_REST_ENERGY_KEV
        max_r = 120.0

        for a in range(-90, 91, 3):
            rad = math.radians(a)
            p_val = 1.0 / (1.0 + (e_in / m_e) * (1.0 - math.cos(rad)))
            ds = 0.5 * (p_val**2) * (p_val + 1.0 / p_val - math.sin(rad) ** 2)
            r = min(max_r, ds * 100.0)
            px = origin.x() + r * math.cos(rad)
            py = origin.y() - r * math.sin(rad)
            if a == -90:
                polar_path.moveTo(px, py)
            else:
                polar_path.lineTo(px, py)

        painter.setBrush(QBrush(QColor(0, 180, 240, 40)))
        painter.setPen(QPen(QColor(0, 200, 255), 1.5))
        painter.drawPath(polar_path)

        curr_rad = math.radians(min(90.0, self._scattering_angle_deg))
        curr_p = 1.0 / (1.0 + (e_in / m_e) * (1.0 - math.cos(curr_rad)))
        curr_ds = 0.5 * (curr_p**2) * (curr_p + 1.0 / curr_p - math.sin(curr_rad) ** 2)
        curr_r = min(max_r, curr_ds * 100.0)
        curr_pt = QPointF(
            origin.x() + curr_r * math.cos(curr_rad),
            origin.y() - curr_r * math.sin(curr_rad),
        )
        painter.setBrush(QBrush(QColor(255, 200, 50)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.0))
        painter.drawEllipse(curr_pt, 3.5, 3.5)

    def _draw_controls(self, painter: QPainter, rect: QRectF, phys: Dict[str, float]) -> None:
        painter.setBrush(QBrush(QColor(18, 25, 40, 230)))
        painter.setPen(QPen(QColor(45, 60, 90), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # Slider 1: Scattering Angle θ
        angle_track_rect = QRectF(rect.left() + 90, rect.top() + 12, 160, 10)
        painter.setBrush(QBrush(QColor(30, 40, 60)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(angle_track_rect, 5.0, 5.0)

        angle_fill_w = (self._scattering_angle_deg / 180.0) * angle_track_rect.width()
        fill_rect = QRectF(angle_track_rect.left(), angle_track_rect.top(), angle_fill_w, 10)
        painter.setBrush(QBrush(QColor(255, 180, 40)))
        painter.drawRoundedRect(fill_rect, 5.0, 5.0)

        handle_x = angle_track_rect.left() + angle_fill_w
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(255, 180, 40), 1.5))
        painter.drawEllipse(QPointF(handle_x, angle_track_rect.center().y()), 7.0, 7.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(200, 220, 240))
        painter.drawText(
            QRectF(rect.left() + 10, rect.top() + 8, 75, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"Góc θ: {self._scattering_angle_deg:.0f}°",
        )

        # Slider 2: Incident Energy E
        energy_track_rect = QRectF(rect.left() + 90, rect.top() + 32, 160, 10)
        painter.setBrush(QBrush(QColor(30, 40, 60)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(energy_track_rect, 5.0, 5.0)

        energy_norm = (self._incident_energy_kev - 10.0) / (2000.0 - 10.0)
        energy_norm = max(0.0, min(1.0, energy_norm))
        energy_fill_w = energy_norm * energy_track_rect.width()
        e_fill_rect = QRectF(energy_track_rect.left(), energy_track_rect.top(), energy_fill_w, 10)
        painter.setBrush(QBrush(QColor(0, 210, 255)))
        painter.drawRoundedRect(e_fill_rect, 5.0, 5.0)

        e_handle_x = energy_track_rect.left() + energy_fill_w
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 210, 255), 1.5))
        painter.drawEllipse(QPointF(e_handle_x, energy_track_rect.center().y()), 7.0, 7.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(200, 220, 240))
        painter.drawText(
            QRectF(rect.left() + 10, rect.top() + 28, 75, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"E: {self._incident_energy_kev:.0f} keV",
        )

        # Stamp Button
        self._stamp_btn_rect = QRectF(rect.right() - 145, rect.top() + 12, 135, 30)
        painter.setBrush(QBrush(QColor(37, 99, 235)))
        painter.setPen(QPen(QColor(96, 165, 250), 1.5))
        painter.drawRoundedRect(self._stamp_btn_rect, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(self._stamp_btn_rect, Qt.AlignCenter, "📌 Đóng dấu bài giảng")

    # ── Interactive Mouse & Gesture Events ───────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        rect = self.boundingRect()

        if self._stamp_btn_rect.contains(pos):
            self.request_stamp()
            event.accept()
            return

        angle_track_rect = QRectF(rect.left() + 85, rect.bottom() - 65 + 8, 170, 18)
        if angle_track_rect.contains(pos):
            self._is_dragging_angle = True
            self._update_angle_from_x(pos.x(), angle_track_rect)
            event.accept()
            return

        energy_track_rect = QRectF(rect.left() + 85, rect.bottom() - 65 + 28, 170, 18)
        if energy_track_rect.contains(pos):
            self._is_dragging_energy = True
            self._update_energy_from_x(pos.x(), energy_track_rect)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        rect = self.boundingRect()

        if self._is_dragging_angle:
            angle_track_rect = QRectF(rect.left() + 85, rect.bottom() - 65 + 8, 170, 18)
            self._update_angle_from_x(pos.x(), angle_track_rect)
            event.accept()
            return

        if self._is_dragging_energy:
            energy_track_rect = QRectF(rect.left() + 85, rect.bottom() - 65 + 28, 170, 18)
            self._update_energy_from_x(pos.x(), energy_track_rect)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._is_dragging_angle = False
        self._is_dragging_energy = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        keys = list(COMPTON_PRESETS.keys())
        try:
            curr_idx = keys.index(self._active_preset_name)
            next_idx = (curr_idx + 1) % len(keys)
        except ValueError:
            next_idx = 0
        self.apply_preset(keys[next_idx])
        event.accept()

    def _update_angle_from_x(self, x: float, track: QRectF) -> None:
        norm = (x - track.left()) / track.width()
        norm = max(0.0, min(1.0, norm))
        self.set_scattering_angle_deg(norm * 180.0)

    def _update_energy_from_x(self, x: float, track: QRectF) -> None:
        norm = (x - track.left()) / track.width()
        norm = max(0.0, min(1.0, norm))
        energy = 10.0 + norm * (2000.0 - 10.0)
        self.set_incident_energy_kev(energy)
