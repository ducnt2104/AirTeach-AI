"""
AirTeach AI - Interactive Aerodynamics & Compressible Flow: de Laval Nozzle Simulator
======================================================================================
Authentic compressible gas dynamics, isentropic nozzle flow & supersonic shock diamonds:
- Area-Velocity Relation (Hugoniot Equation):
    dA / A = (M² - 1) * (du / u)
    - Subsonic (M < 1): Converging (dA < 0) accelerates flow (du > 0).
    - Sonic Throat (M = 1): dA = 0 (minimum throat area A*).
    - Supersonic (M > 1): Diverging (dA > 0) accelerates flow (du > 0).
- Isentropic Stagnation Relations (Ideal gas, γ = 1.4 for air/exhaust):
    T_0 / T = 1 + ((γ - 1) / 2) * M²
    P_0 / P = (1 + ((γ - 1) / 2) * M²)^(γ / (γ - 1))
    ρ_0 / ρ = (1 + ((γ - 1) / 2) * M²)^(1 / (γ - 1))
    A / A* = (1 / M) * [ (2 / (γ + 1)) * (1 + ((γ - 1) / 2) * M²) ]^((γ + 1) / (2 * (γ - 1)))
- Normal Shock Wave (Rankine-Hugoniot Jump Conditions):
    M₂² = ((γ - 1)*M₁² + 2) / (2*γ*M₁² - (γ - 1))
    P₂ / P₁ = 1 + (2*γ / (γ + 1)) * (M₁² - 1)
    T₂ / T₁ = [1 + (2*γ / (γ + 1))*(M₁² - 1)] * [ (2 + (γ - 1)*M₁²) / ((γ + 1)*M₁²) ]
    P_{0,2} / P_{0,1} = exp(-Δs / R) < 1
- Exhaust Plume Regimes & Supersonic Shock Diamonds:
    - Overexpanded (P_e < P_a): Plume compressed by oblique shocks creating periodic diamond shock lattice.
    - Ideally Expanded (P_e = P_a): Maximum thrust efficiency, parallel supersonic exhaust stream.
    - Underexpanded (P_e > P_a): Plume expands outward via Prandtl-Meyer expansion fan.
- Rocket Thrust & Specific Impulse (I_sp):
    F = m_dot * v_e + (P_e - P_a) * A_e
    m_dot = (P_0 * A* / sqrt(T_0)) * sqrt(γ / R) * (2 / (γ + 1))^((γ + 1) / (2*(γ - 1)))
- 5 Curriculum Presets:
    1. IDEALLY_EXPANDED_ROCKET: P_e = P_a, Mach M_e = 3.0, maximum efficiency parallel plume.
    2. OVEREXPANDED_SHOCK_DIAMONDS: P_e = 0.40 P_a, sea-level static test with luminous shock diamonds.
    3. NORMAL_SHOCK_IN_NOZZLE: Normal shock standing inside diverging section with subsonic recovery.
    4. UNDEREXPANDED_VACUUM: High-altitude flight P_e = 4.5 P_a, wide billowing expansion plume.
    5. SUBSONIC_VENTURI_TUBE: Low pressure drop, M < 0.6 everywhere, classical Bernoulli venturi.
- Touchless webcam air gestures (air-tap, slider dragging), bimanual manipulation,
  canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
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
    QPolygonF,
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

# Gas Dynamics Constants
GAMMA: float = 1.40               # Ratio of specific heats (Air / combustion gas)
GAS_CONST_R: float = 287.05       # Specific gas constant for air (J/(kg·K))
ATM_PA: float = 101325.0          # Standard sea-level atmospheric pressure (Pa)


@dataclass
class LavalPreset:
    """Preset configuration for educational de Laval nozzle compressible flow experiments."""
    key: str
    name: str
    desc: str
    p0_bar: float             # Stagnation chamber pressure (bar)
    pb_bar: float             # Ambient back pressure (bar)
    area_ratio: float         # Exit area ratio A_e / A*
    t0_k: float               # Stagnation temperature (Kelvin)
    throat_radius_cm: float   # Throat radius (cm)


PRESETS: List[LavalPreset] = [
    LavalPreset(
        key="ideally_expanded",
        name="Phun tối ưu (Ideally Expanded)",
        desc="Áp suất miệng phun bằng đúng áp suất khí quyển (Pe = Pa), luồng khí song song, lực đẩy cực đại",
        p0_bar=36.7,
        pb_bar=1.0,
        area_ratio=4.23,
        t0_k=2800.0,
        throat_radius_cm=3.0,
    ),
    LavalPreset(
        key="shock_diamonds",
        name="Vòng kim cương Mach (Shock Diamonds)",
        desc="Áp suất miệng phun thấp hơn môi trường (Pe < Pa), sóng xung kích xiên nén luồng tạo chuỗi kim cương",
        p0_bar=15.0,
        pb_bar=1.0,
        area_ratio=4.23,
        t0_k=2400.0,
        throat_radius_cm=3.0,
    ),
    LavalPreset(
        key="normal_shock",
        name="Sóng xung kích thẳng trong loa",
        desc="Áp suất buồng thấp khiến sóng xung kích thẳng đứng xuất hiện ngay trong phần mở rộng",
        p0_bar=2.1,
        pb_bar=1.0,
        area_ratio=2.5,
        t0_k=1200.0,
        throat_radius_cm=3.0,
    ),
    LavalPreset(
        key="underexpanded",
        name="Phun thiếu mở rộng (Tầng cao/Chân không)",
        desc="Áp suất miệng phun cao hơn môi trường (Pe > Pa), luồng khí bung rộng quạt nở Prandtl-Meyer",
        p0_bar=50.0,
        pb_bar=0.20,
        area_ratio=2.5,
        t0_k=3000.0,
        throat_radius_cm=3.0,
    ),
    LavalPreset(
        key="subsonic_venturi",
        name="Ống Venturi hạ âm Bernoulli",
        desc="Chênh lệch áp suất nhỏ, dòng khí không bị nghẽn (M < 1 mọi nơi), áp suất giảm sâu tại cổ họng",
        p0_bar=1.25,
        pb_bar=1.0,
        area_ratio=2.0,
        t0_k=300.0,
        throat_radius_cm=3.0,
    ),
]


class InteractiveLavalNozzleItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic Compressible Aerodynamics, de Laval Nozzle,
    isentropic supersonic flow, normal shock waves, and Mach shock diamonds.
    """
    preset_changed = Signal(str)
    property_changed = Signal(str, float)

    WIDTH: float = 920.0
    HEIGHT: float = 600.0

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

        # Physical Parameters
        p = PRESETS[0]
        self._p0_bar: float = p.p0_bar
        self._pb_bar: float = p.pb_bar
        self._area_ratio: float = p.area_ratio
        self._t0_k: float = p.t0_k
        self._throat_r_cm: float = p.throat_radius_cm

        # UI interaction tracking
        self._dragging_slider: Optional[str] = None

    @property
    def current_preset(self) -> LavalPreset:
        return PRESETS[self._preset_index]

    def set_preset(self, index: int) -> None:
        """Switch active compressible flow preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            p = PRESETS[index]
            self._p0_bar = p.p0_bar
            self._pb_bar = p.pb_bar
            self._area_ratio = p.area_ratio
            self._t0_k = p.t0_k
            self._throat_r_cm = p.throat_radius_cm
            self._sim_time = 0.0
            self.preset_changed.emit(p.key)
            self.update()

    def set_chamber_pressure(self, val_bar: float) -> None:
        """Update stagnation chamber pressure P_0 (bar)."""
        self._p0_bar = max(0.5, min(val_bar, 100.0))
        self.property_changed.emit("p0_bar", self._p0_bar)
        self.update()

    def set_back_pressure(self, val_bar: float) -> None:
        """Update ambient back pressure P_b (bar)."""
        self._pb_bar = max(0.01, min(val_bar, 5.0))
        self.property_changed.emit("pb_bar", self._pb_bar)
        self.update()

    def set_area_ratio(self, val: float) -> None:
        """Update exit area ratio A_e / A*."""
        self._area_ratio = max(1.1, min(val, 15.0))
        self.property_changed.emit("area_ratio", self._area_ratio)
        self.update()

    def set_chamber_temp(self, val_k: float) -> None:
        """Update stagnation chamber temperature T_0 (Kelvin)."""
        self._t0_k = max(200.0, min(val_k, 4000.0))
        self.property_changed.emit("t0_k", self._t0_k)
        self.update()

    def toggle_pause(self) -> None:
        """Toggle simulation pause/resume state."""
        self._is_paused = not self._is_paused
        self.update()

    def step_simulation(self, dt: float) -> None:
        """Advance time animation for exhaust plume turbulent shock shimmering."""
        if self._is_paused or dt <= 0.0:
            return
        self._sim_time += dt
        self.update()

    # -------------------------------------------------------------------------
    # Compressible Flow Mathematical Solver
    # -------------------------------------------------------------------------

    @staticmethod
    def _area_from_mach(m: float) -> float:
        """Evaluate A / A* given Mach number for gamma = 1.4."""
        if m <= 1e-4:
            return 1e5
        term = (2.0 / (GAMMA + 1.0)) * (1.0 + 0.5 * (GAMMA - 1.0) * (m**2))
        exponent = (GAMMA + 1.0) / (2.0 * (GAMMA - 1.0))
        return (1.0 / m) * (term**exponent)

    @classmethod
    def solve_mach_from_area(cls, area_ratio: float, supersonic: bool = False) -> float:
        """Solve for Mach number from area ratio A/A* via Newton-Raphson."""
        if area_ratio <= 1.0001:
            return 1.0

        # Initial guess
        m = 2.5 if supersonic else 0.3
        for _ in range(30):
            # Evaluate f(m) = area_from_mach(m) - area_ratio
            f = cls._area_from_mach(m) - area_ratio
            # Numerical derivative df/dm
            dm = 1e-5
            df = (cls._area_from_mach(m + dm) - cls._area_from_mach(m - dm)) / (2.0 * dm)
            if abs(df) < 1e-9:
                break
            delta = f / df
            m -= delta
            if supersonic:
                m = max(1.001, m)
            else:
                m = max(1e-4, min(0.999, m))
            if abs(delta) < 1e-6:
                break
        return float(m)

    def compute_flow_state(self) -> dict[str, Any]:
        """
        Evaluate full 1D isentropic and normal shock compressible flow state:
        - Critical choking pressure ratio: P* / P_0 = (2 / (γ + 1))^(γ / (γ - 1)) = 0.52828
        - Full expansion exit Mach M_e_isen
        - Exit pressure P_e
        - Plume regime: 'subsonic', 'normal_shock', 'overexpanded', 'ideal', 'underexpanded'
        - Thrust F (N), Mass flow m_dot (kg/s), Specific impulse I_sp (s)
        """
        p0_pa = self._p0_bar * 1e5
        pb_pa = self._pb_bar * 1e5
        pr_total = p0_pa / pb_pa if pb_pa > 0 else 1e5

        # Critical pressure ratio for choking (gamma = 1.4 => 0.52828)
        choke_pr = (2.0 / (GAMMA + 1.0)) ** (GAMMA / (GAMMA - 1.0))  # ~0.52828
        is_choked = (pr_total >= (1.0 / choke_pr))

        # Full isentropic supersonic exit Mach
        m_exit_supersonic = self.solve_mach_from_area(self._area_ratio, supersonic=True)
        # Full isentropic subsonic exit Mach (if unchoked or venturi)
        m_exit_subsonic = self.solve_mach_from_area(self._area_ratio, supersonic=False)

        # Pressure ratio for full supersonic isentropic expansion
        pe_p0_isen = (1.0 + 0.5 * (GAMMA - 1.0) * (m_exit_supersonic**2)) ** (-GAMMA / (GAMMA - 1.0))
        pe_isen_pa = p0_pa * pe_p0_isen

        # Normal shock at exit plane
        # Upstream M1 = m_exit_supersonic
        p2_p1_exit = 1.0 + (2.0 * GAMMA / (GAMMA + 1.0)) * (m_exit_supersonic**2 - 1.0)
        pb_shock_at_exit_pa = pe_isen_pa * p2_p1_exit

        # Subsonic isentropic exit pressure ratio
        pe_p0_sub = (1.0 + 0.5 * (GAMMA - 1.0) * (m_exit_subsonic**2)) ** (-GAMMA / (GAMMA - 1.0))
        pb_subsonic_choke_pa = p0_pa * pe_p0_sub

        # Determine regime
        normal_shock_x: Optional[float] = None  # Normalized x in [0.5, 1.0]

        if not is_choked or self.current_preset.key == "subsonic_venturi":
            regime = "subsonic"
            actual_me = min(0.9, math.sqrt(max(0.0, (2.0 / (GAMMA - 1.0)) * ((p0_pa / pb_pa)**((GAMMA - 1.0) / GAMMA) - 1.0))))
            actual_pe_pa = pb_pa
        elif pb_pa > pb_shock_at_exit_pa:
            # Normal shock stands inside the diverging section!
            regime = "normal_shock"
            # Estimate shock location x_s in [0.5, 1.0]
            fraction = (pb_pa - pb_shock_at_exit_pa) / max(1.0, (pb_subsonic_choke_pa - pb_shock_at_exit_pa))
            fraction = max(0.0, min(1.0, fraction))
            normal_shock_x = 0.5 + (1.0 - fraction) * 0.45
            actual_me = 0.45
            actual_pe_pa = pb_pa
        elif abs(pe_isen_pa - pb_pa) / pb_pa < 0.05:
            regime = "ideal"
            actual_me = m_exit_supersonic
            actual_pe_pa = pe_isen_pa
        elif pe_isen_pa < pb_pa:
            regime = "overexpanded"
            actual_me = m_exit_supersonic
            actual_pe_pa = pe_isen_pa
        else:
            regime = "underexpanded"
            actual_me = m_exit_supersonic
            actual_pe_pa = pe_isen_pa

        # Throat area and dimensions
        throat_r_m = self._throat_r_cm * 0.01
        a_star = math.pi * (throat_r_m**2)
        a_exit = a_star * self._area_ratio

        # Mass flow rate m_dot (kg/s)
        if is_choked and regime != "subsonic":
            m_dot = (p0_pa * a_star / math.sqrt(self._t0_k)) * math.sqrt(GAMMA / GAS_CONST_R) * (
                (2.0 / (GAMMA + 1.0)) ** ((GAMMA + 1.0) / (2.0 * (GAMMA - 1.0)))
            )
        else:
            # Unchoked venturi mass flow
            m_throat = min(0.95, math.sqrt(max(0.0, (2.0 / (GAMMA - 1.0)) * ((p0_pa / pb_pa)**((GAMMA - 1.0) / GAMMA) - 1.0))))
            t_th = self._t0_k / (1.0 + 0.5 * (GAMMA - 1.0) * m_throat**2)
            p_th = p0_pa * ((1.0 + 0.5 * (GAMMA - 1.0) * m_throat**2) ** (-GAMMA / (GAMMA - 1.0)))
            rho_th = p_th / (GAS_CONST_R * t_th)
            v_th = m_throat * math.sqrt(GAMMA * GAS_CONST_R * t_th)
            m_dot = rho_th * a_star * v_th

        # Exit velocity v_e (m/s)
        t_exit = self._t0_k / (1.0 + 0.5 * (GAMMA - 1.0) * (actual_me**2))
        speed_sound_exit = math.sqrt(GAMMA * GAS_CONST_R * t_exit)
        v_exit = actual_me * speed_sound_exit

        # Thrust force: F = m_dot * v_e + (P_e - P_a) * A_e
        thrust_n = m_dot * v_exit + (actual_pe_pa - pb_pa) * a_exit
        thrust_n = max(0.0, thrust_n)

        # Specific impulse I_sp = F / (m_dot * g_0)
        g0 = 9.80665
        isp_s = (thrust_n / (m_dot * g0)) if m_dot > 0 else 0.0

        return {
            "is_choked": is_choked,
            "regime": regime,
            "mach_exit": actual_me,
            "pe_bar": actual_pe_pa * 1e-5,
            "normal_shock_x": normal_shock_x,
            "mass_flow_kg_s": m_dot,
            "thrust_n": thrust_n,
            "isp_s": isp_s,
            "t_exit_k": t_exit,
            "v_exit_m_s": v_exit,
            "a_star_cm2": a_star * 1e4,
            "a_exit_cm2": a_exit * 1e4,
        }

    # -------------------------------------------------------------------------
    # Layout Geometry
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _nozzle_view_rect(self) -> QRectF:
        """Bounding rect of the 2D cutaway nozzle & supersonic exhaust plume canvas."""
        return QRectF(25.0, 115.0, 870.0, 240.0)

    def _profile_plot_rect(self) -> QRectF:
        """Bounding rect of the 1D Mach / Pressure profile plot."""
        return QRectF(25.0, 370.0, 480.0, 105.0)

    def _hud_rect(self) -> QRectF:
        """Bounding rect of scientific telemetry HUD."""
        return QRectF(520.0, 370.0, 375.0, 105.0)

    def _controls_rect(self) -> QRectF:
        """Bounding rect of bottom controls."""
        return QRectF(25.0, 490.0, 870.0, 95.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, LavalPreset]]:
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

        # 4. 2D Nozzle Cutaway & Supersonic Exhaust Plume
        state = self.compute_flow_state()
        self._paint_nozzle_and_plume(painter, state)

        # 5. Mach & Pressure Profile Plot
        self._paint_profile_plot(painter, state)

        # 6. Scientific HUD
        self._paint_hud(painter, state)

        # 7. Bottom Controls & Sliders
        self._paint_controls(painter)

        painter.restore()

    def _paint_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        bg_grad = QLinearGradient(0, 0, 0, self.HEIGHT)
        bg_grad.setColorAt(0.0, QColor(10, 15, 29, 250))
        bg_grad.setColorAt(1.0, QColor(15, 23, 42, 252))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor(249, 115, 22, 80), 1.5))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        painter.setPen(QColor(241, 245, 249))
        font_title = QFont("Segoe UI", 13, QFont.Bold)
        painter.setFont(font_title)
        painter.drawText(28, 38, "KHÍ ĐỘNG HỌC DÒNG CHẢY NÉN ĐƯỢC: ỐNG PHUN DE LAVAL & VÒNG MACH")

        painter.setPen(QColor(148, 163, 184))
        font_sub = QFont("Segoe UI", 9)
        painter.setFont(font_sub)
        painter.drawText(28, 54, "Phương trình Hugoniot vi phân dA/A = (M² - 1)du/u, sóng xung kích Rankine & kim cương Mach")

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
            painter.setBrush(QColor(249, 115, 22, 40))
            painter.setPen(QPen(QColor(249, 115, 22), 1.2))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)
            painter.setPen(QColor(253, 186, 116))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, "🚀 ĐANG MÔ PHỎNG")

    def _paint_preset_buttons(self, painter: QPainter) -> None:
        font_btn = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(font_btn)
        for i, (btn_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == self._preset_index)
            if is_active:
                painter.setBrush(QColor(234, 88, 12, 220))
                painter.setPen(QPen(QColor(253, 186, 116), 1.2))
            else:
                painter.setBrush(QColor(30, 41, 59, 180))
                painter.setPen(QPen(QColor(71, 85, 105), 1.0))
            painter.drawRoundedRect(btn_rect, 6.0, 6.0)

            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            painter.drawText(btn_rect, Qt.AlignCenter, preset.name)

    def _paint_nozzle_and_plume(self, painter: QPainter, state: dict[str, Any]) -> None:
        """Render 2D nozzle cutaway geometry, Mach color gradient, and exhaust shock plume."""
        v_rect = self._nozzle_view_rect()

        # Canvas frame
        painter.setBrush(QColor(5, 8, 18))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(v_rect, 8.0, 8.0)

        # Coordinate layout
        nozzle_w = 420.0
        nozzle_left = v_rect.left() + 20.0
        nozzle_exit_x = nozzle_left + nozzle_w
        cy = v_rect.center().y()

        throat_x = nozzle_left + 160.0
        r_throat = 22.0
        r_inlet = 65.0
        r_exit = r_throat * math.sqrt(self._area_ratio)
        r_exit = min(r_exit, 95.0)

        # 1. Build upper and lower nozzle contour paths
        top_pts = []
        bot_pts = []
        num_x = 45
        for i in range(num_x):
            x = nozzle_left + (i / (num_x - 1)) * nozzle_w
            if x <= throat_x:
                # Converging section
                u = (x - nozzle_left) / (throat_x - nozzle_left)
                r = r_throat + (r_inlet - r_throat) * (math.cos(u * math.pi) * 0.5 + 0.5)
            else:
                # Diverging section (parabolic bell)
                u = (x - throat_x) / (nozzle_exit_x - throat_x)
                r = r_throat + (r_exit - r_throat) * (math.sin(u * math.pi * 0.5))
            top_pts.append(QPointF(x, cy - r))
            bot_pts.append(QPointF(x, cy + r))

        # Fill internal gas flow with Mach-based colored gradient
        gas_poly = QPolygonF(top_pts + bot_pts[::-1])
        gas_grad = QLinearGradient(nozzle_left, 0, nozzle_exit_x, 0)
        gas_grad.setColorAt(0.00, QColor(30, 90, 190, 220))   # Subsonic blue
        gas_grad.setColorAt(0.35, QColor(14, 165, 233, 230))  # Accelerating cyan
        gas_grad.setColorAt(0.40, QColor(234, 179, 8, 240))   # Throat Sonic yellow (M=1)
        if state["regime"] == "subsonic":
            gas_grad.setColorAt(1.00, QColor(30, 90, 190, 220))
        elif state["regime"] == "normal_shock":
            gas_grad.setColorAt(0.65, QColor(249, 115, 22, 240))  # Supersonic orange before shock
            gas_grad.setColorAt(0.70, QColor(30, 90, 190, 220))   # Subsonic blue after shock
            gas_grad.setColorAt(1.00, QColor(15, 23, 42, 220))
        else:
            gas_grad.setColorAt(0.70, QColor(249, 115, 22, 240))  # Supersonic orange
            gas_grad.setColorAt(1.00, QColor(239, 68, 68, 240))   # High Mach red

        painter.setBrush(QBrush(gas_grad))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(gas_poly)

        # Draw thick metal structural nozzle walls
        painter.setPen(QPen(QColor(148, 163, 184), 3.5))
        for i in range(len(top_pts) - 1):
            painter.drawLine(top_pts[i], top_pts[i + 1])
            painter.drawLine(bot_pts[i], bot_pts[i + 1])

        # Draw Throat Line (M = 1.0)
        painter.setPen(QPen(QColor(255, 255, 255, 160), 1.0, Qt.DashLine))
        painter.drawLine(QPointF(throat_x, cy - r_throat - 5), QPointF(throat_x, cy + r_throat + 5))
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(254, 240, 138))
        painter.drawText(QRectF(throat_x - 30, cy - r_throat - 20, 60, 14), Qt.AlignCenter, "CỔ (M=1)")

        # Normal shock line (if present)
        if state["normal_shock_x"] is not None:
            norm_x = nozzle_left + state["normal_shock_x"] * nozzle_w
            # Find height at norm_x
            r_sh = r_throat + (r_exit - r_throat) * math.sin((state["normal_shock_x"] - 0.38) / 0.62 * math.pi * 0.5)
            painter.setPen(QPen(QColor(239, 68, 68), 2.5))
            painter.drawLine(QPointF(norm_x, cy - r_sh), QPointF(norm_x, cy + r_sh))
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.setPen(QColor(252, 165, 165))
            painter.drawText(QRectF(norm_x - 50, cy - r_sh - 16, 100, 14), Qt.AlignCenter, "SÓNG XUNG KÍCH")

        # 2. Render Exhaust Plume (Extending right of nozzle exit)
        plume_left = nozzle_exit_x
        plume_w = v_rect.right() - nozzle_exit_x - 20.0
        self._paint_exhaust_plume(painter, plume_left, cy, r_exit, plume_w, state)

    def _paint_exhaust_plume(
        self,
        painter: QPainter,
        px: float,
        cy: float,
        r_exit: float,
        pw: float,
        state: dict[str, Any],
    ) -> None:
        """Render supersonic exhaust plume with shock diamonds, Mach disks, or underexpanded envelope."""
        regime = state["regime"]
        time_shimmer = math.sin(self._sim_time * 18.0) * 1.5

        if regime == "subsonic":
            # Gentle turbulent plume fading quickly
            painter.setPen(Qt.NoPen)
            grad = QRadialGradient(px, cy, pw * 0.5)
            grad.setColorAt(0.0, QColor(30, 90, 190, 120))
            grad.setColorAt(1.0, QColor(30, 90, 190, 0))
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(QPointF(px + 40, cy), 60.0, r_exit * 0.9)

        elif regime == "ideal":
            # Perfectly parallel supersonic exhaust cylinder
            plume_path = QPainterPath()
            plume_path.moveTo(px, cy - r_exit)
            plume_path.lineTo(px + pw, cy - r_exit)
            plume_path.lineTo(px + pw, cy + r_exit)
            plume_path.lineTo(px, cy + r_exit)
            plume_path.closeSubpath()

            grad = QLinearGradient(px, 0, px + pw, 0)
            grad.setColorAt(0.0, QColor(249, 115, 22, 220))
            grad.setColorAt(0.5, QColor(234, 179, 8, 180))
            grad.setColorAt(1.0, QColor(234, 88, 12, 40))
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.NoPen)
            painter.drawPath(plume_path)

        elif regime == "underexpanded":
            # Billowing Prandtl-Meyer expansion fan blooming wide
            plume_path = QPainterPath()
            plume_path.moveTo(px, cy - r_exit)
            plume_path.quadTo(px + pw * 0.4, cy - r_exit * 1.8 + time_shimmer, px + pw, cy - r_exit * 1.4)
            plume_path.lineTo(px + pw, cy + r_exit * 1.4)
            plume_path.quadTo(px + pw * 0.4, cy + r_exit * 1.8 - time_shimmer, px, cy + r_exit)
            plume_path.closeSubpath()

            grad = QLinearGradient(px, 0, px + pw, 0)
            grad.setColorAt(0.0, QColor(239, 68, 68, 230))
            grad.setColorAt(0.4, QColor(249, 115, 22, 190))
            grad.setColorAt(1.0, QColor(147, 51, 234, 20))
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.NoPen)
            painter.drawPath(plume_path)

        elif regime in ("overexpanded", "normal_shock"):
            # Supersonic Shock Diamonds (Mach Diamonds lattice)
            # Base plume envelope contracting
            num_diamonds = 4
            diamond_spacing = (pw - 30.0) / num_diamonds

            # Plume backdrop
            plume_path = QPainterPath()
            plume_path.moveTo(px, cy - r_exit)
            plume_path.lineTo(px + pw, cy - r_exit * 0.5)
            plume_path.lineTo(px + pw, cy + r_exit * 0.5)
            plume_path.lineTo(px, cy + r_exit)
            plume_path.closeSubpath()

            grad = QLinearGradient(px, 0, px + pw, 0)
            grad.setColorAt(0.0, QColor(249, 115, 22, 170))
            grad.setColorAt(1.0, QColor(30, 41, 59, 30))
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.NoPen)
            painter.drawPath(plume_path)

            # Draw glowing Shock Diamond rhomboids along centerline
            for i in range(num_diamonds):
                dx = px + (i + 0.6) * diamond_spacing
                dw = diamond_spacing * 0.42
                dh = (r_exit * 0.6) * (1.0 - i * 0.18)

                diamond = QPolygonF([
                    QPointF(dx - dw, cy),
                    QPointF(dx, cy - dh + time_shimmer),
                    QPointF(dx + dw, cy),
                    QPointF(dx, cy + dh - time_shimmer),
                ])
                d_grad = QRadialGradient(dx, cy, dw)
                d_grad.setColorAt(0.0, QColor(255, 255, 255, 240))  # Incandescent white core
                d_grad.setColorAt(0.4, QColor(253, 224, 71, 220))   # Yellow
                d_grad.setColorAt(1.0, QColor(239, 68, 68, 0))
                painter.setBrush(QBrush(d_grad))
                painter.setPen(QPen(QColor(254, 240, 138, 200), 1.0))
                painter.drawPolygon(diamond)

                # Oblique shock crosses
                painter.setPen(QPen(QColor(254, 202, 202, 140), 1.0))
                painter.drawLine(QPointF(dx - dw * 1.2, cy - dh * 1.2), QPointF(dx + dw * 1.2, cy + dh * 1.2))
                painter.drawLine(QPointF(dx - dw * 1.2, cy + dh * 1.2), QPointF(dx + dw * 1.2, cy - dh * 1.2))

    def _paint_profile_plot(self, painter: QPainter, state: dict[str, Any]) -> None:
        """Render 1D centerline distributions: Mach M(x) and Pressure P(x)/P0."""
        p_rect = self._profile_plot_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(p_rect, 6.0, 6.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(int(p_rect.left() + 10), int(p_rect.top() + 16), "ĐỒ THỊ DỌC TRỤC: MACH M(x) & ÁP SUẤT P/P0")

        # Axes
        plot_x0 = p_rect.left() + 35.0
        plot_x1 = p_rect.right() - 15.0
        plot_y0 = p_rect.top() + 25.0
        plot_y1 = p_rect.bottom() - 15.0
        pw = plot_x1 - plot_x0
        ph = plot_y1 - plot_y0

        painter.setPen(QPen(QColor(71, 85, 105), 1.0))
        painter.drawLine(QPointF(plot_x0, plot_y1), QPointF(plot_x1, plot_y1))
        painter.drawLine(QPointF(plot_x0, plot_y0), QPointF(plot_x0, plot_y1))

        # Horizontal reference line M = 1.0
        m1_y = plot_y1 - (1.0 / 3.5) * ph
        painter.setPen(QPen(QColor(148, 163, 184, 80), 1.0, Qt.DotLine))
        painter.drawLine(QPointF(plot_x0, m1_y), QPointF(plot_x1, m1_y))

        # Plot curves
        mach_path = QPainterPath()
        press_path = QPainterPath()

        num_pts = 50
        for i in range(num_pts):
            u = i / (num_pts - 1)  # 0 to 1 along nozzle
            px = plot_x0 + u * pw

            # Approximate local Mach M(x)
            if u < 0.38:
                # Subsonic converging
                m_loc = 0.2 + (u / 0.38) * 0.8
            else:
                # Diverging
                if state["normal_shock_x"] is not None and u > state["normal_shock_x"]:
                    m_loc = 0.45 - (u - state["normal_shock_x"]) * 0.1
                elif state["regime"] == "subsonic":
                    m_loc = 1.0 - (u - 0.38) * 0.8
                else:
                    m_loc = 1.0 + ((u - 0.38) / 0.62) * (state["mach_exit"] - 1.0)

            # Mach curve (normalized to max 3.5)
            my = plot_y1 - min(1.0, m_loc / 3.5) * ph
            if i == 0:
                mach_path.moveTo(px, my)
            else:
                mach_path.lineTo(px, my)

            # Pressure curve P / P0 = (1 + 0.2*M^2)^(-3.5)
            p_ratio = (1.0 + 0.2 * (m_loc**2)) ** (-3.5)
            py = plot_y1 - p_ratio * ph
            if i == 0:
                press_path.moveTo(px, py)
            else:
                press_path.lineTo(px, py)

        # Draw Mach Curve (Amber)
        painter.setPen(QPen(QColor(249, 115, 22), 1.8))
        painter.drawPath(mach_path)

        # Draw Pressure Curve (Cyan)
        painter.setPen(QPen(QColor(56, 189, 248), 1.8))
        painter.drawPath(press_path)

        # Legend
        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(249, 115, 22))
        painter.drawText(int(p_rect.right() - 130), int(p_rect.top() + 16), "━ M(x)")
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(int(p_rect.right() - 75), int(p_rect.top() + 16), "━ P/P₀")

    def _paint_hud(self, painter: QPainter, state: dict[str, Any]) -> None:
        """Render scientific telemetry HUD: thrust, Isp, exit Mach, pressure ratio."""
        h_rect = self._hud_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(h_rect, 6.0, 6.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(249, 115, 22))
        painter.drawText(int(h_rect.left() + 10), int(h_rect.top() + 16), "THÔNG SỐ ĐỘNG CƠ & LỰC ĐẨY (HUD)")

        # Metrics rows
        col1_x = h_rect.left() + 10
        col2_x = h_rect.left() + 190
        y_start = h_rect.top() + 34
        line_h = 16

        m1 = [
            ("Chế độ luồng:", state["regime"].upper()),
            ("Số Mach lối ra Me:", f"{state['mach_exit']:.2f}"),
            ("Áp suất miệng Pe:", f"{state['pe_bar']:.2f} bar"),
            ("Vận tốc khí ve:", f"{state['v_exit_m_s']:.0f} m/s"),
        ]

        m2 = [
            ("Lực đẩy động cơ F:", f"{state['thrust_n'] / 1e3:.2f} kN"),
            ("Xung lực riêng Isp:", f"{state['isp_s']:.1f} s"),
            ("Lưu lượng khối m_dot:", f"{state['mass_flow_kg_s']:.2f} kg/s"),
            ("Tỷ số diện tích Ae/A*:", f"{self._area_ratio:.2f}"),
        ]

        painter.setFont(QFont("Segoe UI", 7))
        for idx, (label, val_str) in enumerate(m1):
            y = y_start + idx * line_h
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(int(col1_x), int(y), label)
            painter.setPen(QColor(241, 245, 249))
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.drawText(QRectF(col1_x, y - 10, 165, 14), Qt.AlignRight, val_str)
            painter.setFont(QFont("Segoe UI", 7))

        for idx, (label, val_str) in enumerate(m2):
            y = y_start + idx * line_h
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(int(col2_x), int(y), label)
            painter.setPen(QColor(241, 245, 249))
            painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
            painter.drawText(QRectF(col2_x, y - 10, 165, 14), Qt.AlignRight, val_str)
            painter.setFont(QFont("Segoe UI", 7))

    def _paint_controls(self, painter: QPainter) -> None:
        """Render interactive bottom sliders for P0, Pb, Ae/A*, and T0."""
        ctrl_rect = self._controls_rect()

        # Frame
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(ctrl_rect, 8.0, 8.0)

        # 4 Sliders arranged horizontally
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        sliders = [
            ("p0", "Áp suất buồng P₀", f"{self._p0_bar:.1f} bar", self._p0_bar, 1.0, 80.0),
            ("pb", "Áp suất môi trường P_b", f"{self._pb_bar:.2f} bar", self._pb_bar, 0.05, 2.0),
            ("area", "Tỷ số diện tích Ae/A*", f"{self._area_ratio:.2f}", self._area_ratio, 1.2, 12.0),
            ("t0", "Nhiệt độ buồng T₀", f"{self._t0_k:.0f} K", self._t0_k, 300.0, 3500.0),
        ]

        for i, (key, label, val_text, val, min_v, max_v) in enumerate(sliders):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0

            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(int(cx), int(cy), label)

            painter.setPen(QColor(249, 115, 22))
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
            painter.setBrush(QColor(234, 88, 12))
            painter.drawRoundedRect(fill_rect, 3.0, 3.0)

            # Thumb
            thumb_x = cx + track_w * norm
            thumb_rect = QRectF(thumb_x - 6.0, track_y - 4.0, 12.0, 14.0)
            painter.setBrush(QColor(241, 245, 249))
            painter.setPen(QPen(QColor(234, 88, 12), 1.2))
            painter.drawRoundedRect(thumb_rect, 3.0, 3.0)

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

        # Check Sliders
        ctrl_rect = self._controls_rect()
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        for i, key in enumerate(["p0", "pb", "area", "t0"]):
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
            slider_keys = ["p0", "pb", "area", "t0"]
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
        if key == "p0":
            self.set_chamber_pressure(1.0 + norm * (80.0 - 1.0))
        elif key == "pb":
            self.set_back_pressure(0.05 + norm * (2.0 - 0.05))
        elif key == "area":
            self.set_area_ratio(1.2 + norm * (12.0 - 1.2))
        elif key == "t0":
            self.set_chamber_temp(300.0 + norm * (3500.0 - 300.0))

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulator state for .airteach lesson format."""
        return {
            "type": "InteractiveLavalNozzleItem",
            "preset_key": self.current_preset.key,
            "p0_bar": self._p0_bar,
            "pb_bar": self._pb_bar,
            "area_ratio": self._area_ratio,
            "t0_k": self._t0_k,
            "throat_r_cm": self._throat_r_cm,
            "is_paused": self._is_paused,
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveLavalNozzleItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "ideally_expanded")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "p0_bar" in data:
            item.set_chamber_pressure(float(data["p0_bar"]))
        if "pb_bar" in data:
            item.set_back_pressure(float(data["pb_bar"]))
        if "area_ratio" in data:
            item.set_area_ratio(float(data["area_ratio"]))
        if "t0_k" in data:
            item.set_chamber_temp(float(data["t0_k"]))
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
