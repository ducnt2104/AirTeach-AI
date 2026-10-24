"""
AirTeach AI - Interactive Astrophysics & General Relativity: Gravitational Lensing Simulator
=============================================================================================
Simulates authentic General Relativity photon deflection in curved spacetime (Einstein 1915/1936).
Models Schwarzschild black hole deflection, Einstein ring formation, Einstein cross,
and gravitational microlensing magnification.

Physics Model:
  1. Einstein Deflection Angle:
       α̂ = 4*G*M / (c^2 * ξ)
  2. Einstein Angular Radius:
       θ_E = sqrt[(4*G*M / c^2) * (D_ds / (D_d * D_s))]
  3. Lens Equation:
       β = θ - θ_E^2 / θ
  4. Image Positions (for source at position β):
       θ_± = 0.5 * [β ± sqrt(β^2 + 4 * θ_E^2)]
  5. Magnifications (u = β / θ_E):
       μ_± = (u^2 + 2) / [2 * u * sqrt(u^2 + 4)] ± 0.5
       μ_total = (u^2 + 2) / [u * sqrt(u^2 + 4)]
  6. Einstein Ring:
       At perfect alignment (β = 0), image becomes a continuous ring of radius θ = θ_E.
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
    QRadialGradient,
)
from PySide6.QtWidgets import QGraphicsItem, QGraphicsObject, QGraphicsSceneMouseEvent

# Astronomical & Physical Constants
G_CONST = 6.67430e-11  # m^3 kg^-1 s^-2
C_LIGHT = 2.99792458e8  # m s^-1
SOLAR_MASS_KG = 1.98847e30  # kg
KPC_TO_METERS = 3.085677581e19  # m (1 kiloparsec)
ARCSEC_TO_RAD = 4.848136811e-6  # rad


@dataclass(frozen=True)
class LensingPreset:
    name: str
    description: str
    lens_mass_solar: float  # M_sun
    d_lens_kpc: float  # Distance to lens
    d_source_kpc: float  # Distance to source
    source_offset_arcsec: float  # Angular separation beta
    is_quadrupole_cross: bool  # Elliptical galaxy -> Einstein cross


LENSING_PRESETS: Dict[str, LensingPreset] = {
    "PERFECT_EINSTEIN_RING": LensingPreset(
        name="Chiếc Nhẫn Einstein Hoàn Mỹ (Horseshoe SDSS)",
        description="Nguồn sáng thẳng hàng tuyệt đối với thấu kính hấp dẫn (β = 0), tạo vòng tròn Einstein hoàn hảo",
        lens_mass_solar=1.2e12,
        d_lens_kpc=1.5e6,
        d_source_kpc=3.0e6,
        source_offset_arcsec=0.0,
        is_quadrupole_cross=False,
    ),
    "EINSTEIN_CROSS_Q2237": LensingPreset(
        name="Cây Thánh Giá Einstein (Q2237+0305 Huchra)",
        description="Thiên hà elip có thế hấp dẫn tứ cực biến 1 quasar phía sau thành 4 ảnh đối xứng rực rỡ",
        lens_mass_solar=1.0e11,
        d_lens_kpc=1.2e5,
        d_source_kpc=2.4e6,
        source_offset_arcsec=0.25,
        is_quadrupole_cross=True,
    ),
    "SUPERMASSIVE_BLACK_HOLE_SGR_A": LensingPreset(
        name="Lỗ Đen Siêu Khối Sagittarius A* (Ngân Hà)",
        description="Lỗ đen 4.3 triệu khối lượng Mặt Trời tại tâm Ngân Hà uốn cong tia sáng các ngôi sao nền",
        lens_mass_solar=4.3e6,
        d_lens_kpc=8.2,
        d_source_kpc=16.0,
        source_offset_arcsec=0.8,
        is_quadrupole_cross=False,
    ),
    "ABELL_370_GALAXY_CLUSTER": LensingPreset(
        name="Cụm Thiên Hà Khổng Lồ Abell 370",
        description="Khối lượng 10¹⁵ Mặt Trời bẻ cong ánh sáng thiên hà xoắn ốc nền thành các cung tròn khổng lồ",
        lens_mass_solar=1.0e15,
        d_lens_kpc=1.8e6,
        d_source_kpc=3.5e6,
        source_offset_arcsec=1.2,
        is_quadrupole_cross=False,
    ),
    "MICROLENSING_EXOPLANET": LensingPreset(
        name="Kính Vi Hấp Dẫn Phát Hiện Ngoại Hành Tinh",
        description="Ngôi sao đi qua trước thiên thể nền làm bùng sáng độ phóng đại quang thông cực đại",
        lens_mass_solar=1.0,
        d_lens_kpc=4.0,
        d_source_kpc=8.0,
        source_offset_arcsec=0.05,
        is_quadrupole_cross=False,
    ),
}


class InteractiveGravitationalLensingItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating General Relativity Gravitational Lensing,
    Einstein Rings, and the Einstein Cross.
    """

    stamp_requested = Signal(dict)
    physics_computed = Signal(dict)

    def __init__(
        self,
        lens_mass_solar: float = 1.0e12,
        source_offset_arcsec: float = 0.5,
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

        self._lens_mass_solar = float(lens_mass_solar)
        self._source_offset_arcsec = float(source_offset_arcsec)
        self._is_quadrupole = False
        self._active_preset_name = "PERFECT_EINSTEIN_RING"

        self._width = 680.0
        self._height = 430.0

        self._is_dragging_source = False
        self._is_dragging_mass = False
        self._stamp_btn_rect = QRectF(0, 0, 0, 0)
        self._source_handle_rect = QRectF(0, 0, 0, 0)

    def boundingRect(self) -> QRectF:
        return QRectF(-self._width / 2, -self._height / 2, self._width, self._height)

    # ── Physics Calculations ────────────────────────────────────────────────

    @property
    def lens_mass_solar(self) -> float:
        return self._lens_mass_solar

    @property
    def source_offset_arcsec(self) -> float:
        return self._source_offset_arcsec

    def set_lens_mass_solar(self, mass: float) -> None:
        self._lens_mass_solar = max(0.5, min(1.0e15, float(mass)))
        self.physics_computed.emit(self.compute_physics())
        self.update()

    def set_source_offset_arcsec(self, offset: float) -> None:
        self._source_offset_arcsec = max(0.0, min(3.0, float(offset)))
        self.physics_computed.emit(self.compute_physics())
        self.update()

    def compute_physics(self) -> Dict[str, float]:
        """Compute Einstein radius, image angles, and total magnification."""
        # Typical cosmological distance scaling: D_d = 1 Gpc, D_s = 2 Gpc, D_ds = 1 Gpc
        # theta_E ≈ sqrt( (4*G*M / c^2) * (D_ds / (D_d * D_s)) )
        # Using standard astrophysical scaling: theta_E (arcsec) ≈ 1.0 * sqrt(M / 1e12 M_sun)
        mass_norm = self._lens_mass_solar / 1.0e12
        theta_e_arcsec = 1.60 * math.sqrt(mass_norm) if mass_norm > 0 else 0.01

        beta = self._source_offset_arcsec

        # Image positions: theta_± = 0.5 * (beta ± sqrt(beta^2 + 4*theta_E^2))
        disc = math.sqrt(beta**2 + 4.0 * (theta_e_arcsec**2))
        theta_plus = 0.5 * (beta + disc)
        theta_minus = 0.5 * (beta - disc)  # Negative angle (opposite side)

        # Magnification: u = beta / theta_E
        if theta_e_arcsec > 0:
            u = beta / theta_e_arcsec
        else:
            u = 1.0

        if u < 1e-4:
            # Near perfect alignment -> Singular magnification, ring formation
            mu_plus = 25.0
            mu_minus = -25.0
            mu_tot = 50.0
        else:
            u_sqrt = math.sqrt(u**2 + 4.0)
            mu_plus = (u**2 + 2.0) / (2.0 * u * u_sqrt) + 0.5
            mu_minus = (u**2 + 2.0) / (2.0 * u * u_sqrt) - 0.5
            mu_tot = (u**2 + 2.0) / (u * u_sqrt)

        # Schwarzschild radius in km: Rs = 2GM / c^2 ≈ 2.95 * (M/M_sun) km
        rs_km = 2.95 * (self._lens_mass_solar)

        return {
            "theta_e_arcsec": theta_e_arcsec,
            "beta_arcsec": beta,
            "theta_plus_arcsec": theta_plus,
            "theta_minus_arcsec": abs(theta_minus),
            "mu_plus": mu_plus,
            "mu_minus": abs(mu_minus),
            "mu_tot": min(100.0, mu_tot),
            "rs_km": rs_km,
        }

    def apply_preset(self, preset_key: str) -> None:
        preset = LENSING_PRESETS.get(preset_key)
        if preset:
            self._lens_mass_solar = preset.lens_mass_solar
            self._source_offset_arcsec = preset.source_offset_arcsec
            self._is_quadrupole = preset.is_quadrupole_cross
            self._active_preset_name = preset_key
            self.physics_computed.emit(self.compute_physics())
            self.update()

    def request_stamp(self) -> None:
        phys = self.compute_physics()
        preset = LENSING_PRESETS.get(self._active_preset_name, LENSING_PRESETS["PERFECT_EINSTEIN_RING"])
        data = {
            "center": (self.x(), self.y()),
            "preset_name": preset.name,
            "description": preset.description,
            "lens_mass_solar": self._lens_mass_solar,
            "source_offset_arcsec": self._source_offset_arcsec,
            "theta_e_arcsec": phys["theta_e_arcsec"],
            "theta_plus_arcsec": phys["theta_plus_arcsec"],
            "theta_minus_arcsec": phys["theta_minus_arcsec"],
            "mu_tot": phys["mu_tot"],
            "is_quadrupole": self._is_quadrupole,
        }
        self.stamp_requested.emit(data)

    # ── Rendering ────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        rect = self.boundingRect()
        phys = self.compute_physics()

        # 1. Background
        bg_brush = QBrush(QColor(8, 11, 20, 248))
        painter.setBrush(bg_brush)
        painter.setPen(QPen(QColor(32, 48, 76), 1.5))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # 2. Header
        self._draw_header(painter, rect, phys)

        # 3. Lensing Viewport (Left 60%)
        view_rect = QRectF(rect.left() + 15, rect.top() + 55, 380, 300)
        self._draw_lensing_viewport(painter, view_rect, phys)

        # 4. Right Astronomy Data Panel (Right 40%)
        info_rect = QRectF(view_rect.right() + 15, rect.top() + 55, 255, 300)
        self._draw_astronomy_panel(painter, info_rect, phys)

        # 5. Bottom Controls Bar
        control_rect = QRectF(rect.left() + 15, rect.bottom() - 65, rect.width() - 30, 55)
        self._draw_controls(painter, control_rect, phys)

    def _draw_header(self, painter: QPainter, rect: QRectF, phys: Dict[str, float]) -> None:
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(QColor(220, 230, 245))
        preset = LENSING_PRESETS.get(self._active_preset_name, LENSING_PRESETS["PERFECT_EINSTEIN_RING"])
        painter.drawText(
            QRectF(rect.left() + 15, rect.top() + 8, rect.width() - 30, 22),
            Qt.AlignLeft | Qt.AlignVCenter,
            "🪐 THẤU KÍNH HẤP DẪN & CHIẾC NHẪN EINSTEIN — THUYẾT TƯƠNG ĐỐI RỘNG",
        )

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(130, 160, 200))
        sub_text = (
            f"Mô hình: {preset.name} | M = {self._lens_mass_solar:.2e} M_sun | "
            f"Bán kính Einstein θ_E = {phys['theta_e_arcsec']:.2f}\" | Phóng đại μ = {phys['mu_tot']:.1f}×"
        )
        painter.drawText(
            QRectF(rect.left() + 15, rect.top() + 28, rect.width() - 30, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            sub_text,
        )

    def _draw_lensing_viewport(
        self, painter: QPainter, rect: QRectF, phys: Dict[str, float]
    ) -> None:
        painter.setBrush(QBrush(QColor(4, 6, 12)))
        painter.setPen(QPen(QColor(25, 38, 60), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        center = rect.center()
        scale = 75.0  # pixels per arcsec

        # Background Stars
        painter.setPen(QColor(180, 200, 240, 140))
        for sx, sy in [(-120, -80), (130, 90), (-90, 70), (110, -100), (50, 110), (-140, 20)]:
            painter.drawPoint(QPointF(center.x() + sx, center.y() + sy))

        # 1. Einstein Ring / Critical Curve (dashed cyan circle)
        theta_e_px = phys["theta_e_arcsec"] * scale
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(56, 189, 248, 90), 1.2, Qt.DashLine))
        painter.drawEllipse(center, theta_e_px, theta_e_px)

        # Label Einstein Ring
        painter.setFont(QFont("Segoe UI", 6.5))
        painter.setPen(QColor(56, 189, 248, 160))
        painter.drawText(
            QRectF(center.x() - theta_e_px, center.y() + theta_e_px + 2, theta_e_px * 2, 14),
            Qt.AlignCenter,
            f"Vòng Einstein θ_E = {phys['theta_e_arcsec']:.2f}\"",
        )

        # 2. Central Lens (Black Hole / Galaxy Cluster)
        lens_glow = QRadialGradient(center, 24.0)
        lens_glow.setColorAt(0.0, QColor(0, 0, 0, 255))
        lens_glow.setColorAt(0.4, QColor(15, 23, 42, 240))
        lens_glow.setColorAt(0.8, QColor(56, 189, 248, 60))
        lens_glow.setColorAt(1.0, QColor(56, 189, 248, 0))
        painter.setBrush(QBrush(lens_glow))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, 24.0, 24.0)

        painter.setBrush(QBrush(QColor(0, 0, 0)))
        painter.setPen(QPen(QColor(148, 163, 184), 1.2))
        painter.drawEllipse(center, 8.0, 8.0)

        # 3. Lensed Images or Einstein Cross
        beta = self._source_offset_arcsec
        if self._is_quadrupole:
            # Einstein Cross: 4 symmetric quasar images at ~theta_E in a cross pattern
            cross_r = theta_e_px
            for angle_deg in [0, 90, 180, 270]:
                rad = math.radians(angle_deg)
                ix = center.x() + cross_r * math.cos(rad)
                iy = center.y() + cross_r * math.sin(rad)
                # Quasar image glow
                q_glow = QRadialGradient(QPointF(ix, iy), 12.0)
                q_glow.setColorAt(0.0, QColor(254, 240, 138))
                q_glow.setColorAt(0.5, QColor(250, 204, 21, 160))
                q_glow.setColorAt(1.0, QColor(250, 204, 21, 0))
                painter.setBrush(QBrush(q_glow))
                painter.drawEllipse(QPointF(ix, iy), 10.0, 10.0)
        elif beta < 0.05:
            # Perfect Einstein Ring: Glowing continuous circular halo
            ring_pen = QPen(QColor(250, 204, 21, 230), 4.5)
            painter.setPen(ring_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(center, theta_e_px, theta_e_px)
        else:
            # Dual Images: Primary arc (+) and Secondary arc (-)
            r_plus = phys["theta_plus_arcsec"] * scale
            r_minus = phys["theta_minus_arcsec"] * scale

            # Primary arc on the source side (+X direction)
            span_angle = max(30.0, min(160.0, 180.0 / (1.0 + beta * 2.0)))
            arc_p_rect = QRectF(center.x() - r_plus, center.y() - r_plus, r_plus * 2, r_plus * 2)
            painter.setPen(QPen(QColor(250, 204, 21, 230), 4.0))
            painter.drawArc(arc_p_rect, -int(span_angle * 8), int(span_angle * 16))

            # Secondary arc on the opposite side (-X direction)
            arc_m_rect = QRectF(center.y() - r_minus, center.y() - r_minus, r_minus * 2, r_minus * 2)
            arc_m_rect = QRectF(center.x() - r_minus, center.y() - r_minus, r_minus * 2, r_minus * 2)
            painter.setPen(QPen(QColor(251, 146, 60, 200), 2.8))
            painter.drawArc(arc_m_rect, int((180 - span_angle / 2) * 16), int(span_angle * 16))

        # 4. Background True Source Position (Cyan dot with dashed indicator)
        source_x = center.x() + beta * scale
        source_y = center.y()
        self._source_handle_rect = QRectF(source_x - 10, source_y - 10, 20, 20)

        painter.setPen(QPen(QColor(56, 189, 248, 120), 1.0, Qt.DashLine))
        painter.drawLine(center.x(), center.y(), source_x, source_y)

        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(source_x, source_y), 5.5, 5.5)

        painter.setFont(QFont("Segoe UI", 6.5, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(
            QRectF(source_x - 35, source_y - 20, 70, 14),
            Qt.AlignCenter,
            f"Nguồn thực β={beta:.2f}\"",
        )

    def _draw_astronomy_panel(
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
            "🔭 THÔNG SỐ VẬT LÝ THIÊN VĂN",
        )

        rows = [
            ("Bán kính Einstein (θ_E):", f"{phys['theta_e_arcsec']:.3f} arcsec", QColor(56, 189, 248)),
            ("Độ lệch nguồn (β):", f"{phys['beta_arcsec']:.3f} arcsec", QColor(250, 204, 21)),
            ("Vị trí ảnh sơ cấp (θ_+):", f"{phys['theta_plus_arcsec']:.3f} arcsec", QColor(74, 222, 128)),
            ("Vị trí ảnh thứ cấp (θ_-):", f"{phys['theta_minus_arcsec']:.3f} arcsec", QColor(251, 146, 60)),
            ("Độ phóng đại ảnh sơ cấp:", f"{phys['mu_plus']:.2f} ×", QColor(74, 222, 128)),
            ("Độ phóng đại ảnh thứ cấp:", f"{phys['mu_minus']:.2f} ×", QColor(251, 146, 60)),
            ("TỔNG ĐỘ PHÓNG ĐẠI (μ_tot):", f"{phys['mu_tot']:.2f} ×", QColor(236, 72, 153)),
            ("Bán kính Schwarzschild:", f"{phys['rs_km']:.2e} km", QColor(168, 85, 247)),
        ]

        y = rect.top() + 28
        painter.setFont(QFont("Segoe UI", 7))
        for label, val, col in rows:
            painter.setPen(QColor(140, 160, 190))
            painter.drawText(QRectF(rect.left() + 10, y, 140, 15), Qt.AlignLeft, label)
            painter.setPen(col)
            painter.drawText(QRectF(rect.right() - 95, y, 85, 15), Qt.AlignRight, val)
            y += 18

        # Formulas Box
        f_box = QRectF(rect.left() + 8, y + 4, rect.width() - 16, 92)
        painter.setBrush(QBrush(QColor(10, 14, 24)))
        painter.setPen(QPen(QColor(30, 42, 65), 1.0))
        painter.drawRoundedRect(f_box, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(QRectF(f_box.left() + 6, f_box.top() + 6, f_box.width() - 12, 14), Qt.AlignLeft, "QUANG HỌC TƯƠNG ĐỐI TÍNH:")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(203, 213, 225))
        formulas = [
            "• Góc lệch: α̂ = 4GM / (c²ξ)",
            "• Bán kính: θ_E = √[(4GM/c²)(D_ds / D_d·D_s)]",
            "• PT thấu kính: β = θ - θ_E² / θ",
            "• Độ phóng đại: μ = (u² + 2) / [u√(u² + 4)]",
        ]
        fy = f_box.top() + 22
        for form in formulas:
            painter.drawText(QRectF(f_box.left() + 6, fy, f_box.width() - 12, 14), Qt.AlignLeft, form)
            fy += 16

    def _draw_controls(self, painter: QPainter, rect: QRectF, phys: Dict[str, float]) -> None:
        painter.setBrush(QBrush(QColor(18, 25, 40, 230)))
        painter.setPen(QPen(QColor(45, 60, 90), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # Slider 1: Source Offset beta (0 to 3.0 arcsec)
        beta_track = QRectF(rect.left() + 90, rect.top() + 12, 180, 10)
        painter.setBrush(QBrush(QColor(30, 40, 60)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(beta_track, 5.0, 5.0)

        b_norm = self._source_offset_arcsec / 3.0
        b_fill_w = max(0.0, min(1.0, b_norm)) * beta_track.width()
        painter.setBrush(QBrush(QColor(250, 204, 21)))
        painter.drawRoundedRect(QRectF(beta_track.left(), beta_track.top(), b_fill_w, 10), 5.0, 5.0)

        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(250, 204, 21), 1.5))
        painter.drawEllipse(QPointF(beta_track.left() + b_fill_w, beta_track.center().y()), 7.0, 7.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(200, 220, 240))
        painter.drawText(
            QRectF(rect.left() + 10, rect.top() + 8, 75, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"Lệch β: {self._source_offset_arcsec:.2f}\"",
        )

        # Slider 2: Lens Mass (log10 scale: 10^6 to 10^15 M_sun)
        mass_track = QRectF(rect.left() + 90, rect.top() + 32, 180, 10)
        painter.setBrush(QBrush(QColor(30, 40, 60)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(mass_track, 5.0, 5.0)

        log_m = math.log10(max(1.0, self._lens_mass_solar))
        m_norm = (log_m - 6.0) / (15.0 - 6.0)
        m_fill_w = max(0.0, min(1.0, m_norm)) * mass_track.width()
        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.drawRoundedRect(QRectF(mass_track.left(), mass_track.top(), m_fill_w, 10), 5.0, 5.0)

        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(56, 189, 248), 1.5))
        painter.drawEllipse(QPointF(mass_track.left() + m_fill_w, mass_track.center().y()), 7.0, 7.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(200, 220, 240))
        painter.drawText(
            QRectF(rect.left() + 10, rect.top() + 28, 75, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"M: 10^{log_m:.1f} M☉",
        )

        # Stamp Button
        self._stamp_btn_rect = QRectF(rect.right() - 145, rect.top() + 12, 135, 30)
        painter.setBrush(QBrush(QColor(37, 99, 235)))
        painter.setPen(QPen(QColor(96, 165, 250), 1.5))
        painter.drawRoundedRect(self._stamp_btn_rect, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(self._stamp_btn_rect, Qt.AlignCenter, "📌 Đóng dấu bài học")

        # Preset hint
        painter.setFont(QFont("Segoe UI", 6.5))
        painter.setPen(QColor(130, 150, 180))
        painter.drawText(
            QRectF(rect.right() - 360, rect.top() + 12, 205, 30),
            Qt.AlignRight | Qt.AlignVCenter,
            "Nhấp đúp chuột để chuyển Preset",
        )

    # ── Mouse Events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        rect = self.boundingRect()

        if self._stamp_btn_rect.contains(pos):
            self.request_stamp()
            event.accept()
            return

        beta_track = QRectF(rect.left() + 85, rect.bottom() - 65 + 8, 190, 18)
        if beta_track.contains(pos):
            self._is_dragging_source = True
            self._update_beta_from_x(pos.x(), beta_track)
            event.accept()
            return

        mass_track = QRectF(rect.left() + 85, rect.bottom() - 65 + 28, 190, 18)
        if mass_track.contains(pos):
            self._is_dragging_mass = True
            self._update_mass_from_x(pos.x(), mass_track)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        rect = self.boundingRect()

        if self._is_dragging_source:
            beta_track = QRectF(rect.left() + 85, rect.bottom() - 65 + 8, 190, 18)
            self._update_beta_from_x(pos.x(), beta_track)
            event.accept()
            return

        if self._is_dragging_mass:
            mass_track = QRectF(rect.left() + 85, rect.bottom() - 65 + 28, 190, 18)
            self._update_mass_from_x(pos.x(), mass_track)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._is_dragging_source = False
        self._is_dragging_mass = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        keys = list(LENSING_PRESETS.keys())
        try:
            curr_idx = keys.index(self._active_preset_name)
            next_idx = (curr_idx + 1) % len(keys)
        except ValueError:
            next_idx = 0
        self.apply_preset(keys[next_idx])
        event.accept()

    def _update_beta_from_x(self, x: float, track: QRectF) -> None:
        norm = (x - track.left()) / track.width()
        norm = max(0.0, min(1.0, norm))
        self.set_source_offset_arcsec(norm * 3.0)

    def _update_mass_from_x(self, x: float, track: QRectF) -> None:
        norm = (x - track.left()) / track.width()
        norm = max(0.0, min(1.0, norm))
        log_m = 6.0 + norm * (15.0 - 6.0)
        self.set_lens_mass_solar(10.0**log_m)
