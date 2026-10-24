"""
AirTeach AI - Interactive Nuclear Physics: Binding Energy Curve & Semi-Empirical Mass Formula (SEMF)
=====================================================================================================
Simulates authentic nuclear binding energy, mass defect, and the liquid-drop model
via the Bethe-Weizsäcker Semi-Empirical Mass Formula (SEMF):
  B(A, Z) = a_v*A - a_s*A^(2/3) - a_c*Z*(Z-1)/A^(1/3) - a_a*(A-2Z)^2/A + δ(A, Z)

Key Physical Quantities:
  1. Volume Term: E_v = a_v * A (a_v ≈ 15.8 MeV)
  2. Surface Term: E_s = -a_s * A^(2/3) (a_s ≈ 18.3 MeV)
  3. Coulomb Repulsion: E_c = -a_c * Z*(Z-1) / A^(1/3) (a_c ≈ 0.714 MeV)
  4. Asymmetry Energy: E_a = -a_a * (A - 2Z)^2 / A (a_a ≈ 23.2 MeV)
  5. Pairing Energy: δ(A, Z) = +12.0/√A (even-even), 0 (odd A), -12.0/√A (odd-odd)
  6. Binding Energy per Nucleon: BE/A = B(A, Z) / A
  7. Mass Defect: Δm = B(A, Z) / (931.494 MeV/u)
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

# Bethe-Weizsäcker Constants (in MeV)
A_V = 15.8  # Volume
A_S = 18.3  # Surface
A_C = 0.714  # Coulomb
A_A = 23.2  # Asymmetry
A_P = 12.0  # Pairing coefficient
AMU_TO_MEV = 931.494061  # MeV / u


@dataclass(frozen=True)
class NuclidePreset:
    name: str
    symbol: str
    A: int
    Z: int
    description: str
    regime: str  # "fusion", "peak", "fission"


NUCLIDE_PRESETS: Dict[str, NuclidePreset] = {
    "IRON_56_MAX_STABILITY": NuclidePreset(
        name="Sắt-56 (Peak Stability)",
        symbol="⁵⁶₂₆Fe",
        A=56,
        Z=26,
        description="Đỉnh bền vững hạt nhân cao nhất vũ trụ (8.79 MeV/nucleon), tro tàn sao khổng lồ",
        regime="peak",
    ),
    "DEUTERIUM_TRITIUM_FUSION": NuclidePreset(
        name="Heli-4 (Sản phẩm Nhiệt Hạch D-T)",
        symbol="⁴₂He",
        A=4,
        Z=2,
        description="Vỏ ma thuật đôi bền vững, tỏa năng lượng khổng lồ 17.6 MeV khi nhiệt hạch D-T",
        regime="fusion",
    ),
    "URANIUM_235_NUCLEAR_FISSION": NuclidePreset(
        name="Urani-235 (Nhiên liệu Phân Hạch)",
        symbol="²³⁵₉₂U",
        A=235,
        Z=92,
        description="Hạt nhân nặng dễ phân hạch khi bắt neutron chậm, tỏa ~200 MeV năng lượng",
        regime="fission",
    ),
    "CARBON_12_STANDARD": NuclidePreset(
        name="Cacbon-12 (Chuẩn Đơn Vị u)",
        symbol="¹²₆C",
        A=12,
        Z=6,
        description="Nguyên tố cơ sở của sự sống và chuẩn định nghĩa đơn vị khối lượng nguyên tử",
        regime="fusion",
    ),
    "NICKEL_62_ABSOLUTE_MAX": NuclidePreset(
        name="Niken-62 (Năng lượng liên kết cực đại tuyệt đối)",
        symbol="⁶²₂₈Ni",
        A=62,
        Z=28,
        description="Hạt nhân có năng lượng liên kết trên mỗi nucleon lớn nhất tuyệt đối (8.7946 MeV)",
        regime="peak",
    ),
}


class InteractiveNuclearBindingEnergyItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating the Nuclear Binding Energy Curve
    and Bethe-Weizsäcker Semi-Empirical Mass Formula (SEMF).
    """

    stamp_requested = Signal(dict)
    physics_computed = Signal(dict)

    def __init__(
        self,
        A: int = 56,
        Z: int = 26,
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

        self._A = int(A)
        self._Z = int(Z)
        self._active_preset_name = "IRON_56_MAX_STABILITY"

        self._width = 680.0
        self._height = 430.0

        self._is_dragging_A = False
        self._is_dragging_Z = False
        self._stamp_btn_rect = QRectF(0, 0, 0, 0)

    def boundingRect(self) -> QRectF:
        return QRectF(-self._width / 2, -self._height / 2, self._width, self._height)

    # ── Physics Calculations ────────────────────────────────────────────────

    @property
    def A(self) -> int:
        return self._A

    @property
    def Z(self) -> int:
        return self._Z

    def set_A(self, a_val: int) -> None:
        self._A = max(1, min(260, int(a_val)))
        # Keep Z within realistic physical bounds (Z <= A)
        if self._Z > self._A:
            self._Z = max(1, self._A // 2)
        self.physics_computed.emit(self.compute_physics())
        self.update()

    def set_Z(self, z_val: int) -> None:
        self._Z = max(1, min(self._A, int(z_val)))
        self.physics_computed.emit(self.compute_physics())
        self.update()

    @staticmethod
    def calculate_semf(A: int, Z: int) -> Dict[str, float]:
        """Calculate the 5 terms of the Bethe-Weizsäcker formula."""
        if A <= 1:
            return {
                "e_vol": 0.0,
                "e_surf": 0.0,
                "e_coul": 0.0,
                "e_asym": 0.0,
                "e_pair": 0.0,
                "total_be_mev": 0.0,
                "be_per_nucleon_mev": 0.0,
                "mass_defect_u": 0.0,
            }

        N = A - Z
        # 1. Volume Term
        e_vol = A_V * A

        # 2. Surface Term
        e_surf = -A_S * (A ** (2.0 / 3.0))

        # 3. Coulomb Term
        e_coul = -A_C * Z * (Z - 1) / (A ** (1.0 / 3.0))

        # 4. Asymmetry Term
        e_asym = -A_A * ((A - 2 * Z) ** 2) / A

        # 5. Pairing Term
        if A % 2 == 1:
            e_pair = 0.0
        elif Z % 2 == 0 and N % 2 == 0:
            e_pair = A_P / math.sqrt(A)
        else:
            e_pair = -A_P / math.sqrt(A)

        total_be = e_vol + e_surf + e_coul + e_asym + e_pair
        total_be = max(0.0, total_be)
        be_per_nucleon = total_be / A
        mass_defect_u = total_be / AMU_TO_MEV

        return {
            "e_vol": e_vol,
            "e_surf": e_surf,
            "e_coul": e_coul,
            "e_asym": e_asym,
            "e_pair": e_pair,
            "total_be_mev": total_be,
            "be_per_nucleon_mev": be_per_nucleon,
            "mass_defect_u": mass_defect_u,
        }

    def compute_physics(self) -> Dict[str, float]:
        res = self.calculate_semf(self._A, self._Z)
        res["A"] = float(self._A)
        res["Z"] = float(self._Z)
        res["N"] = float(self._A - self._Z)
        return res

    def apply_preset(self, preset_key: str) -> None:
        preset = NUCLIDE_PRESETS.get(preset_key)
        if preset:
            self._A = preset.A
            self._Z = preset.Z
            self._active_preset_name = preset_key
            self.physics_computed.emit(self.compute_physics())
            self.update()

    def request_stamp(self) -> None:
        phys = self.compute_physics()
        preset = NUCLIDE_PRESETS.get(self._active_preset_name, NUCLIDE_PRESETS["IRON_56_MAX_STABILITY"])
        data = {
            "center": (self.x(), self.y()),
            "A": self._A,
            "Z": self._Z,
            "preset_name": preset.name,
            "symbol": preset.symbol,
            "description": preset.description,
            "total_be_mev": phys["total_be_mev"],
            "be_per_nucleon_mev": phys["be_per_nucleon_mev"],
            "mass_defect_u": phys["mass_defect_u"],
            "e_vol": phys["e_vol"],
            "e_surf": phys["e_surf"],
            "e_coul": phys["e_coul"],
            "e_asym": phys["e_asym"],
            "e_pair": phys["e_pair"],
        }
        self.stamp_requested.emit(data)

    # ── Rendering ────────────────────────────────────────────────────────────

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        rect = self.boundingRect()
        phys = self.compute_physics()

        # 1. Dark Technical Background
        bg_brush = QBrush(QColor(13, 19, 31, 248))
        painter.setBrush(bg_brush)
        painter.setPen(QPen(QColor(38, 54, 85), 1.5))
        painter.drawRoundedRect(rect, 10.0, 10.0)

        # 2. Header
        self._draw_header(painter, rect, phys)

        # 3. Main B/A Curve Plot (Left 60%)
        plot_rect = QRectF(rect.left() + 15, rect.top() + 55, 380, 300)
        self._draw_binding_curve(painter, plot_rect, phys)

        # 4. Right SEMF Energy Breakdown Panel (Right 40%)
        info_rect = QRectF(plot_rect.right() + 15, rect.top() + 55, 255, 300)
        self._draw_semf_breakdown(painter, info_rect, phys)

        # 5. Bottom Interactive Sliders Bar
        control_rect = QRectF(rect.left() + 15, rect.bottom() - 65, rect.width() - 30, 55)
        self._draw_controls(painter, control_rect, phys)

    def _draw_header(self, painter: QPainter, rect: QRectF, phys: Dict[str, float]) -> None:
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(QColor(220, 230, 245))
        preset = NUCLIDE_PRESETS.get(self._active_preset_name, NUCLIDE_PRESETS["IRON_56_MAX_STABILITY"])
        painter.drawText(
            QRectF(rect.left() + 15, rect.top() + 8, rect.width() - 30, 22),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"⚛ ĐƯỜNG CONG NĂNG LƯỢNG LIÊN KẾT HẠT NHÂN & CÔNG THỨC BETHE-WEIZSÄCKER (SEMF)",
        )

        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor(130, 160, 200))
        sub_text = (
            f"Nuclide: {preset.symbol} (A = {self._A}, Z = {self._Z}, N = {self._A - self._Z}) | "
            f"BE/A = {phys['be_per_nucleon_mev']:.3f} MeV/nucleon | Δm = {phys['mass_defect_u']:.4f} u"
        )
        painter.drawText(
            QRectF(rect.left() + 15, rect.top() + 28, rect.width() - 30, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            sub_text,
        )

    def _draw_binding_curve(self, painter: QPainter, rect: QRectF, phys: Dict[str, float]) -> None:
        painter.setBrush(QBrush(QColor(8, 12, 20, 220)))
        painter.setPen(QPen(QColor(30, 44, 70), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # Plot margins
        ox = rect.left() + 45.0
        oy = rect.bottom() - 35.0
        pw = rect.width() - 55.0
        ph = rect.height() - 50.0

        # Axis Lines
        painter.setPen(QPen(QColor(70, 90, 120), 1.2))
        painter.drawLine(ox, oy, ox + pw, oy)  # X: Mass Number A (0 to 250)
        painter.drawLine(ox, oy, ox, oy - ph)  # Y: BE/A (0 to 10 MeV)

        # Ticks & Grid
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QColor(110, 130, 160))
        for a_val in [50, 100, 150, 200, 250]:
            tx = ox + (a_val / 250.0) * pw
            painter.drawLine(tx, oy, tx, oy + 4)
            painter.drawText(QRectF(tx - 15, oy + 5, 30, 12), Qt.AlignCenter, str(a_val))
            painter.setPen(QPen(QColor(35, 50, 75, 80), 1.0, Qt.DotLine))
            painter.drawLine(tx, oy - ph, tx, oy)
            painter.setPen(QColor(110, 130, 160))

        for be_val in [2, 4, 6, 8, 10]:
            ty = oy - (be_val / 10.0) * ph
            painter.drawLine(ox - 4, ty, ox, ty)
            painter.drawText(QRectF(ox - 25, ty - 6, 20, 12), Qt.AlignRight | Qt.AlignVCenter, str(be_val))
            painter.setPen(QPen(QColor(35, 50, 75, 80), 1.0, Qt.DotLine))
            painter.drawLine(ox, ty, ox + pw, ty)
            painter.setPen(QColor(110, 130, 160))

        # Axis labels
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.drawText(QRectF(ox + pw - 90, oy + 16, 90, 14), Qt.AlignRight, "Số khối A →")
        painter.drawText(QRectF(ox - 40, oy - ph - 16, 120, 14), Qt.AlignLeft, "BE/A (MeV) ↑")

        # Fusion & Fission Zones Shading
        # Fusion Zone (A: 0 - 56)
        x_iron = ox + (56.0 / 250.0) * pw
        painter.setBrush(QBrush(QColor(16, 185, 129, 25)))
        painter.setPen(Qt.NoPen)
        painter.drawRect(QRectF(ox, oy - ph, x_iron - ox, ph))
        painter.setFont(QFont("Segoe UI", 6.5, QFont.Bold))
        painter.setPen(QColor(52, 211, 153))
        painter.drawText(QRectF(ox + 5, oy - ph + 5, 80, 14), Qt.AlignLeft, "VÙNG NHIỆT HẠCH")

        # Fission Zone (A: 56 - 250)
        painter.setBrush(QBrush(QColor(239, 68, 68, 20)))
        painter.drawRect(QRectF(x_iron, oy - ph, (ox + pw) - x_iron, ph))
        painter.setPen(QColor(248, 113, 113))
        painter.drawText(QRectF(x_iron + 30, oy - ph + 5, 80, 14), Qt.AlignLeft, "VÙNG PHÂN HẠCH")

        # Plot full SEMF curve along beta-stability valley: Z_stable ≈ A / (2 + 0.015 * A^(2/3))
        curve_path = QPainterPath()
        first = True

        for a_step in range(2, 255, 2):
            z_stable = int(round(a_step / (2.0 + 0.015 * (a_step ** (2.0 / 3.0)))))
            z_stable = max(1, z_stable)
            semf = self.calculate_semf(a_step, z_stable)
            be_a = semf["be_per_nucleon_mev"]

            px = ox + (a_step / 250.0) * pw
            py = oy - (be_a / 10.0) * ph

            if first:
                curve_path.moveTo(px, py)
                first = False
            else:
                curve_path.lineTo(px, py)

        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(56, 189, 248), 2.2))
        painter.drawPath(curve_path)

        # Highlight Iron-56 peak
        fe_x = ox + (56.0 / 250.0) * pw
        fe_y = oy - (8.79 / 10.0) * ph
        painter.setBrush(QBrush(QColor(250, 204, 21)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawEllipse(QPointF(fe_x, fe_y), 4.5, 4.5)
        painter.setFont(QFont("Segoe UI", 6.5, QFont.Bold))
        painter.setPen(QColor(250, 204, 21))
        painter.drawText(QRectF(fe_x - 30, fe_y - 18, 60, 14), Qt.AlignCenter, "⁵⁶Fe (8.79)")

        # Current Selected Nuclide Marker
        curr_be = phys["be_per_nucleon_mev"]
        curr_px = ox + (self._A / 250.0) * pw
        curr_py = oy - (curr_be / 10.0) * ph

        # Vertical indicator line
        painter.setPen(QPen(QColor(236, 72, 153, 180), 1.2, Qt.DashLine))
        painter.drawLine(curr_px, oy, curr_px, curr_py)

        # Glowing dot
        painter.setBrush(QBrush(QColor(236, 72, 153)))
        painter.setPen(QPen(QColor(255, 255, 255), 2.0))
        painter.drawEllipse(QPointF(curr_px, curr_py), 6.0, 6.0)

        # Tooltip tag
        preset = NUCLIDE_PRESETS.get(self._active_preset_name, NUCLIDE_PRESETS["IRON_56_MAX_STABILITY"])
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(255, 255, 255))
        tag_rect = QRectF(curr_px - 40, curr_py - 25, 80, 18)
        painter.setBrush(QBrush(QColor(236, 72, 153, 230)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(tag_rect, 3.0, 3.0)
        painter.setPen(QColor(255, 255, 255))
        painter.drawText(tag_rect, Qt.AlignCenter, f"{preset.symbol}: {curr_be:.2f}")

    def _draw_semf_breakdown(
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
            "🔬 CƠ CẤU 5 THÀNH PHẦN SEMF (MeV)",
        )

        terms = [
            ("Năng lượng Thể tích (a_v·A):", phys["e_vol"], QColor(56, 189, 248), True),
            ("Năng lượng Bề mặt (-a_s·A²/³):", phys["e_surf"], QColor(248, 113, 113), False),
            ("Đẩy Coulomb (-a_c·Z²/A¹/³):", phys["e_coul"], QColor(251, 146, 60), False),
            ("Bất đối xứng (-a_a·(A-2Z)²/A):", phys["e_asym"], QColor(168, 85, 247), False),
            ("Năng lượng Ghép đôi δ(A,Z):", phys["e_pair"], QColor(250, 204, 21), phys["e_pair"] >= 0),
        ]

        y = rect.top() + 28
        painter.setFont(QFont("Segoe UI", 7))
        for name, val, col, is_pos in terms:
            painter.setPen(QColor(140, 160, 190))
            painter.drawText(QRectF(rect.left() + 10, y, 145, 14), Qt.AlignLeft, name)
            painter.setPen(col)
            painter.drawText(
                QRectF(rect.right() - 85, y, 75, 14),
                Qt.AlignRight,
                f"{val:+.1f} MeV",
            )

            # Mini bar
            bar_w = min(180.0, (abs(val) / 2500.0) * 180.0)
            bar_rect = QRectF(rect.left() + 10, y + 14, max(4.0, bar_w), 5.0)
            painter.setBrush(QBrush(col))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(bar_rect, 2.0, 2.0)
            y += 24

        # Net Total
        painter.setPen(QPen(QColor(50, 70, 100), 1.0))
        painter.drawLine(rect.left() + 10, y + 2, rect.right() - 10, y + 2)
        y += 8

        painter.setFont(QFont("Segoe UI", 7.5, QFont.Bold))
        painter.setPen(QColor(220, 235, 255))
        painter.drawText(QRectF(rect.left() + 10, y, 130, 16), Qt.AlignLeft, "TỔNG NĂNG LƯỢNG B:")
        painter.setPen(QColor(52, 211, 153))
        painter.drawText(QRectF(rect.right() - 95, y, 85, 16), Qt.AlignRight, f"{phys['total_be_mev']:.1f} MeV")
        y += 18

        painter.setPen(QColor(220, 235, 255))
        painter.drawText(QRectF(rect.left() + 10, y, 130, 16), Qt.AlignLeft, "ĐỘ HỤT KHỐI Δm:")
        painter.setPen(QColor(251, 191, 36))
        painter.drawText(QRectF(rect.right() - 95, y, 85, 16), Qt.AlignRight, f"{phys['mass_defect_u']:.4f} u")
        y += 22

        # Physics Regime Note
        regime_box = QRectF(rect.left() + 8, y, rect.width() - 16, 55)
        painter.setBrush(QBrush(QColor(10, 14, 24)))
        painter.setPen(QPen(QColor(30, 42, 65), 1.0))
        painter.drawRoundedRect(regime_box, 4.0, 4.0)

        preset = NUCLIDE_PRESETS.get(self._active_preset_name, NUCLIDE_PRESETS["IRON_56_MAX_STABILITY"])
        painter.setFont(QFont("Segoe UI", 6.5))
        painter.setPen(QColor(180, 200, 225))
        painter.drawText(
            QRectF(regime_box.left() + 6, regime_box.top() + 4, regime_box.width() - 12, regime_box.height() - 8),
            Qt.AlignLeft | Qt.TextWordWrap,
            f"💡 {preset.description}",
        )

    def _draw_controls(self, painter: QPainter, rect: QRectF, phys: Dict[str, float]) -> None:
        painter.setBrush(QBrush(QColor(18, 25, 40, 230)))
        painter.setPen(QPen(QColor(45, 60, 90), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        # Slider 1: Mass Number A (1 to 260)
        a_track_rect = QRectF(rect.left() + 90, rect.top() + 12, 180, 10)
        painter.setBrush(QBrush(QColor(30, 40, 60)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(a_track_rect, 5.0, 5.0)

        a_norm = (self._A - 1.0) / (260.0 - 1.0)
        a_fill_w = max(0.0, min(1.0, a_norm)) * a_track_rect.width()
        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.drawRoundedRect(QRectF(a_track_rect.left(), a_track_rect.top(), a_fill_w, 10), 5.0, 5.0)

        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(56, 189, 248), 1.5))
        painter.drawEllipse(QPointF(a_track_rect.left() + a_fill_w, a_track_rect.center().y()), 7.0, 7.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(200, 220, 240))
        painter.drawText(
            QRectF(rect.left() + 10, rect.top() + 8, 75, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"Số khối A: {self._A}",
        )

        # Slider 2: Atomic Number Z (1 to A)
        z_track_rect = QRectF(rect.left() + 90, rect.top() + 32, 180, 10)
        painter.setBrush(QBrush(QColor(30, 40, 60)))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(z_track_rect, 5.0, 5.0)

        z_norm = (self._Z - 1.0) / max(1.0, float(self._A - 1.0))
        z_fill_w = max(0.0, min(1.0, z_norm)) * z_track_rect.width()
        painter.setBrush(QBrush(QColor(236, 72, 153)))
        painter.drawRoundedRect(QRectF(z_track_rect.left(), z_track_rect.top(), z_fill_w, 10), 5.0, 5.0)

        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(236, 72, 153), 1.5))
        painter.drawEllipse(QPointF(z_track_rect.left() + z_fill_w, z_track_rect.center().y()), 7.0, 7.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(200, 220, 240))
        painter.drawText(
            QRectF(rect.left() + 10, rect.top() + 28, 75, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"Điện tích Z: {self._Z}",
        )

        # Stamp Button
        self._stamp_btn_rect = QRectF(rect.right() - 145, rect.top() + 12, 135, 30)
        painter.setBrush(QBrush(QColor(16, 185, 129)))
        painter.setPen(QPen(QColor(52, 211, 153), 1.5))
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
            "Nhấp đúp chuột để chuyển Preset hạt nhân",
        )

    # ── Mouse Events ─────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        rect = self.boundingRect()

        if self._stamp_btn_rect.contains(pos):
            self.request_stamp()
            event.accept()
            return

        a_track = QRectF(rect.left() + 85, rect.bottom() - 65 + 8, 190, 18)
        if a_track.contains(pos):
            self._is_dragging_A = True
            self._update_A_from_x(pos.x(), a_track)
            event.accept()
            return

        z_track = QRectF(rect.left() + 85, rect.bottom() - 65 + 28, 190, 18)
        if z_track.contains(pos):
            self._is_dragging_Z = True
            self._update_Z_from_x(pos.x(), z_track)
            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        rect = self.boundingRect()

        if self._is_dragging_A:
            a_track = QRectF(rect.left() + 85, rect.bottom() - 65 + 8, 190, 18)
            self._update_A_from_x(pos.x(), a_track)
            event.accept()
            return

        if self._is_dragging_Z:
            z_track = QRectF(rect.left() + 85, rect.bottom() - 65 + 28, 190, 18)
            self._update_Z_from_x(pos.x(), z_track)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._is_dragging_A = False
        self._is_dragging_Z = False
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        keys = list(NUCLIDE_PRESETS.keys())
        try:
            curr_idx = keys.index(self._active_preset_name)
            next_idx = (curr_idx + 1) % len(keys)
        except ValueError:
            next_idx = 0
        self.apply_preset(keys[next_idx])
        event.accept()

    def _update_A_from_x(self, x: float, track: QRectF) -> None:
        norm = (x - track.left()) / track.width()
        norm = max(0.0, min(1.0, norm))
        val = int(round(1.0 + norm * (260.0 - 1.0)))
        self.set_A(val)

    def _update_Z_from_x(self, x: float, track: QRectF) -> None:
        norm = (x - track.left()) / track.width()
        norm = max(0.0, min(1.0, norm))
        val = int(round(1.0 + norm * (self._A - 1.0)))
        self.set_Z(val)
