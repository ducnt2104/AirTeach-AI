"""
AirTeach AI - Interactive Quantum Physics Bohr Atomic Model & Hydrogen Spectral Line Series Simulator
====================================================================================================
Provides an authentic quantum physics simulation of Niels Bohr's atomic model for Hydrogen:
  - Bohr's Postulates:
      1. Stationary Orbits (Quỹ đạo dừng):
          Electron revolves only on stable circular orbits without radiating:
          r_n = n² * r₀,  where r₀ = 0.529177 Å (Bohr radius, 52.92 pm)
          Orbits: n=1 (K), n=2 (L), n=3 (M), n=4 (N), n=5 (O), n=6 (P)
      2. Quantized Energy Levels (Mức năng lượng dừng):
          E_n = -E₀ / n² = -13.605693 eV / n²
          Ionization energy from ground state: E_ion = 13.61 eV
      3. Bohr Frequency Condition (Phát xạ / Hấp thụ Photon):
          Photon energy: ε = |E_m - E_n| = h * f = (h * c) / λ
          Wavelength: λ_mn = (h * c) / |E_m - E_n| = 1239.841984 / ΔE(eV) nm
          Rydberg formula: 1/λ = R_H * |1/n² - 1/m²|, R_H = 1.097373e7 m⁻¹
  - Authentic Spectral Series of Hydrogen:
      - Lyman Series (Dãy Lyman, n = 1, m ≥ 2):
          Ultraviolet region (Vùng Tử ngoại, λ < 380 nm)
          L_α (2 → 1): 121.57 nm, L_β (3 → 1): 102.57 nm, L_limit: 91.18 nm
      - Balmer Series (Dãy Ban-me, n = 2, m ≥ 3):
          Visible optical spectrum (Vùng Ánh sáng nhìn thấy & Gần tử ngoại):
          H_α (3 → 2): 656.28 nm (Đỏ rực / Crimson Red)
          H_β (4 → 2): 486.13 nm (Lam / Cyan)
          H_γ (5 → 2): 434.05 nm (Chàm / Indigo)
          H_δ (6 → 2): 410.17 nm (Tím / Violet)
      - Paschen Series (Dãy Pas-sen, n = 3, m ≥ 4):
          Near-Infrared region (Vùng Hồng ngoại gần, λ > 780 nm)
          P_α (4 → 3): 1875.1 nm, P_β (5 → 3): 1281.8 nm
      - Brackett Series (Dãy Bơ-ra-két, n = 4, m ≥ 5):
          Mid-Infrared region (Hồng ngoại trung bình, λ > 2500 nm)
          B_α (5 → 4): 4051.2 nm
  - Dynamic Visuals:
      - Concentric planetary orbits with central proton nucleus (+e).
      - Orbiting electron particle with velocity v_n = v₁ / n (v₁ ≈ 2188 km/s).
      - Radiating wavy photon wave packet escaping radially during quantum jump!
      - Right Pane: Discrete energy ladder diagram and authentic continuous/emission spectrum bar.
  - Air gestures (air tap / air pinch) & vector stamping to LessonBoard.
"""
from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
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

from src.ui.stem_tools import PhysicsTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

# ── Quantum Physical Constants ────────────────────────────────────────────────
RYDBERG_ENERGY_EV = 13.605693009    # eV (Ground state binding energy)
BOHR_RADIUS_ANGSTROM = 0.52917721   # Å (1 Å = 10^-10 m)
HC_EV_NM = 1239.841984              # eV·nm (h*c in eV·nm)
RYDBERG_CONSTANT_M1 = 1.0973731568e7  # m^-1
VELOCITY_V1_KMS = 2187.69           # km/s (Speed in first orbit n=1)

ORBIT_NAMES: Dict[int, str] = {
    1: "K",
    2: "L",
    3: "M",
    4: "N",
    5: "O",
    6: "P",
}

PRESETS: Dict[str, Dict[str, Any]] = {
    "BALMER_ALPHA": {
        "label": "Balmer H_α (Đỏ 656nm)",
        "m": 3,
        "n": 2,
        "desc": "Chuyển mức 3 → 2 (M → L): Phát xạ vạch đỏ H_α bước sóng 656.3 nm.",
    },
    "BALMER_BETA": {
        "label": "Balmer H_β (Lam 486nm)",
        "m": 4,
        "n": 2,
        "desc": "Chuyển mức 4 → 2 (N → L): Phát xạ vạch lam H_β bước sóng 486.1 nm.",
    },
    "LYMAN_ALPHA": {
        "label": "Lyman L_α (UV 122nm)",
        "m": 2,
        "n": 1,
        "desc": "Chuyển mức 2 → 1 (L → K): Phát xạ photon tử ngoại sâu L_α bước sóng 121.6 nm.",
    },
    "PASCHEN_ALPHA": {
        "label": "Paschen P_α (Hồng ngoại)",
        "m": 4,
        "n": 3,
        "desc": "Chuyển mức 4 → 3 (N → M): Phát xạ photon hồng ngoại gần P_α bước sóng 1875 nm.",
    },
}


def wavelength_to_spectral_color(wavelength_nm: float) -> Tuple[QColor, str, str]:
    """
    Classifies photon wavelength into (color, spectral_band, line_name).
    """
    wl = float(wavelength_nm)
    if wl < 380.0:
        return QColor("#a855f7"), "Vùng Tử ngoại (Ultraviolet)", "Tia Tử ngoại (UV)"
    elif 380.0 <= wl < 420.0:
        return QColor("#8b5cf6"), "Vùng Ánh sáng nhìn thấy (Tím)", "Vạch Tím H_δ (410.2 nm)"
    elif 420.0 <= wl < 450.0:
        return QColor("#6366f1"), "Vùng Ánh sáng nhìn thấy (Chàm)", "Vạch Chàm H_γ (434.1 nm)"
    elif 450.0 <= wl < 500.0:
        return QColor("#06b6d4"), "Vùng Ánh sáng nhìn thấy (Lam)", "Vạch Lam H_β (486.1 nm)"
    elif 500.0 <= wl < 580.0:
        return QColor("#10b981"), "Vùng Ánh sáng nhìn thấy (Lục)", "Quang phổ Lục"
    elif 580.0 <= wl < 630.0:
        return QColor("#f59e0b"), "Vùng Ánh sáng nhìn thấy (Vàng/Cam)", "Quang phổ Cam"
    elif 630.0 <= wl <= 750.0:
        return QColor("#ef4444"), "Vùng Ánh sáng nhìn thấy (Đỏ)", "Vạch Đỏ H_α (656.3 nm)"
    else:
        return QColor("#dc2626"), "Vùng Hồng ngoại (Infrared)", "Tia Hồng ngoại (IR)"


class InteractiveBohrAtomItem(QGraphicsObject):
    """
    Comprehensive Interactive Quantum Bohr Atom & Hydrogen Spectral Lines Item for LessonBoard.
    Visualizes electron orbits n = 1..6, quantum jumps m -> n, radiation wave packets,
    energy level ladder, and authentic hydrogen optical emission line spectrum.
    """

    parameter_changed = Signal()
    quantum_jump_triggered = Signal(int, int)
    closed = Signal()

    def __init__(self, pos: tuple[float, float] = (320.0, 160.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])

        self._width = 600.0
        self._height = 430.0

        # Quantum Level States
        self._level_m: int = 3   # Initial excited state (m = 1..6)
        self._level_n: int = 2   # Final lower state (n = 1..6)
        self._current_electron_orbit: int = 3
        self._current_preset: str = "BALMER_ALPHA"
        self._diagram_mode: str = "ORBITS_AND_LEVELS"  # or "EMISSION_SPECTRUM"

        # Orbit Animation & Photon Packet
        self._electron_angle_rad: float = 0.0
        self._photon_packets: List[Dict[str, Any]] = []
        self._is_animating: bool = True
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(25)  # 40 FPS
        self._anim_timer.timeout.connect(self._on_animation_step)
        self._anim_timer.start()

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(35.0)

    # ── Geometry & Bounds ──────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self._width, self._height)

    # ── Quantum Mechanics Formulas & Calculations ──────────────────────────────

    def get_orbit_radius_angstrom(self, n: int) -> float:
        """r_n = n² * r₀ (in Ångströms)."""
        return (n ** 2) * BOHR_RADIUS_ANGSTROM

    def get_energy_level_ev(self, n: int) -> float:
        """E_n = -13.605693 / n² (in eV)."""
        if n <= 0:
            return -RYDBERG_ENERGY_EV
        return -RYDBERG_ENERGY_EV / (n ** 2)

    def get_electron_velocity_kms(self, n: int) -> float:
        """v_n = v₁ / n (in km/s)."""
        if n <= 0:
            return VELOCITY_V1_KMS
        return VELOCITY_V1_KMS / float(n)

    def get_transition_energy_ev(self) -> float:
        """ΔE = |E_m - E_n| in eV."""
        e_m = self.get_energy_level_ev(self._level_m)
        e_n = self.get_energy_level_ev(self._level_n)
        return abs(e_m - e_n)

    def get_transition_wavelength_nm(self) -> float:
        """λ = hc / ΔE (in nm)."""
        delta_e = self.get_transition_energy_ev()
        if delta_e <= 0.001:
            return 9999.0
        return HC_EV_NM / delta_e

    def get_transition_frequency_hz(self) -> float:
        """f = c / λ (in Hz)."""
        wl_m = self.get_transition_wavelength_nm() * 1e-9
        if wl_m <= 0:
            return 0.0
        return 299792458.0 / wl_m

    def get_series_name(self) -> str:
        """Identifies spectral series from lower level: 1->Lyman, 2->Balmer, 3->Paschen, 4->Brackett."""
        lower_n = min(self._level_m, self._level_n)
        if lower_n == 1:
            return "Dãy Lyman (Tử ngoại - UV)"
        elif lower_n == 2:
            return "Dãy Balmer (Khả kiến & Gần UV)"
        elif lower_n == 3:
            return "Dãy Paschen (Hồng ngoại gần - IR)"
        elif lower_n == 4:
            return "Dãy Brackett (Hồng ngoại trung bình)"
        elif lower_n == 5:
            return "Dãy Pfund (Hồng ngoại xa)"
        else:
            return f"Dãy Humphreys (n = {lower_n})"

    def trigger_quantum_jump(self) -> None:
        """Initiates an animated quantum transition from level_m to level_n and emits a photon packet."""
        wl = self.get_transition_wavelength_nm()
        color, band, line_nm = wavelength_to_spectral_color(wl)

        # Spawn radiating wavy photon packet
        orbit_r_px = 25.0 + self._level_m * 14.0
        angle = self._electron_angle_rad

        self._photon_packets.append({
            "x": orbit_r_px * math.cos(angle),
            "y": orbit_r_px * math.sin(angle),
            "vx": 4.5 * math.cos(angle),
            "vy": 4.5 * math.sin(angle),
            "color": color,
            "wl_nm": wl,
            "age": 0,
            "max_age": 70,
        })

        # Switch current active electron orbit to target level n
        self._current_electron_orbit = self._level_n
        self.quantum_jump_triggered.emit(self._level_m, self._level_n)
        self.parameter_changed.emit()
        self.update()

    # ── Getters & Setters ─────────────────────────────────────────────────────

    @property
    def level_m(self) -> int:
        return self._level_m

    def set_level_m(self, val: int) -> None:
        clamped = max(1, min(6, int(val)))
        if clamped != self._level_m:
            self._level_m = clamped
            self._current_electron_orbit = clamped
            self.parameter_changed.emit()
            self.update()

    @property
    def level_n(self) -> int:
        return self._level_n

    def set_level_n(self, val: int) -> None:
        clamped = max(1, min(6, int(val)))
        if clamped != self._level_n:
            self._level_n = clamped
            self.parameter_changed.emit()
            self.update()

    def set_levels(self, m: int, n: int) -> None:
        self.set_level_m(m)
        self.set_level_n(n)

    def set_diagram_mode(self, mode: str) -> None:
        if mode in ("ORBITS_AND_LEVELS", "EMISSION_SPECTRUM") and mode != self._diagram_mode:
            self._diagram_mode = mode
            self.update()

    def apply_preset(self, preset_key: str) -> None:
        if preset_key in PRESETS:
            p = PRESETS[preset_key]
            self._current_preset = preset_key
            self.set_levels(p["m"], p["n"])
            self.trigger_quantum_jump()

    # ── Animation Step ────────────────────────────────────────────────────────

    def _on_animation_step(self) -> None:
        if self._is_animating:
            # Orbital angular velocity omega ~ 1 / n^3
            n_curr = max(1, self._current_electron_orbit)
            speed = 0.12 / math.pow(n_curr, 1.2)
            self._electron_angle_rad = (self._electron_angle_rad + speed) % (2.0 * math.pi)

            # Update photon packets
            alive: List[Dict[str, Any]] = []
            for pkt in self._photon_packets:
                pkt["x"] += pkt["vx"]
                pkt["y"] += pkt["vy"]
                pkt["age"] += 1
                if pkt["age"] < pkt["max_age"]:
                    alive.append(pkt)
            self._photon_packets = alive

            self.update()

    def cleanup(self) -> None:
        """Stop animation timer."""
        if hasattr(self, "_anim_timer") and self._anim_timer.isActive():
            self._anim_timer.stop()
        self._photon_packets.clear()

    def close(self) -> None:
        self.cleanup()
        self.setVisible(False)
        self.closed.emit()

    # ── UI Regions & Layout ───────────────────────────────────────────────────

    def _get_close_button_rect(self) -> QRectF:
        return QRectF(self._width - 32.0, 10.0, 22.0, 22.0)

    def _get_stamp_button_rect(self) -> QRectF:
        return QRectF(self._width - 150.0, 10.0, 110.0, 24.0)

    def _get_preset_rects(self) -> List[Tuple[str, QRectF]]:
        keys = list(PRESETS.keys())
        rects = []
        x_start = 14.0
        y_top = 42.0
        w = 138.0
        h = 24.0
        for i, k in enumerate(keys):
            rects.append((k, QRectF(x_start + i * (w + 6.0), y_top, w, h)))
        return rects

    def _get_mode_toggle_rects(self) -> Tuple[QRectF, QRectF]:
        x = 315.0
        y = 74.0
        w = 130.0
        h = 22.0
        return (QRectF(x, y, w, h), QRectF(x + w + 6.0, y, w, h))

    def _get_jump_button_rect(self) -> QRectF:
        return QRectF(500.0, 380.0, 86.0, 26.0)

    def _get_level_chips(self) -> Tuple[List[Tuple[int, QRectF]], List[Tuple[int, QRectF]]]:
        """Returns ([chips for level m], [chips for level n])."""
        m_chips = []
        n_chips = []
        # Level m chips (Initial state)
        for i in range(1, 7):
            m_chips.append((i, QRectF(75.0 + (i - 1) * 26.0, 372.0, 22.0, 20.0)))
        # Level n chips (Final state)
        for i in range(1, 7):
            n_chips.append((i, QRectF(75.0 + (i - 1) * 26.0, 396.0, 22.0, 20.0)))
        return m_chips, n_chips

    # ── Painting & Rendering ──────────────────────────────────────────────────

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Card
        card_rect = self.boundingRect()
        bg_grad = QLinearGradient(0, 0, 0, self._height)
        bg_grad.setColorAt(0.0, QColor("#090d16"))
        bg_grad.setColorAt(1.0, QColor("#050811"))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor("#1e293b"), 1.6))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Header Bar
        self._paint_header(painter)

        # 3. Presets & Mode Tabs
        self._paint_selectors(painter)

        # 4. Left Pane: Concentric Planetary Bohr Orbits & Nucleus
        self._paint_bohr_orbits(painter)

        # 5. Right Pane: Energy Level Ladder or Emission Spectrum
        self._paint_diagnostic_pane(painter)

        # 6. Bottom Controls & Level Selectors
        self._paint_controls(painter)

    def _paint_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(16, 26, "⚛️ MÔ HÌNH NGUYÊN TỬ BOHR & QUANG PHỔ HYDRO (NIELS BOHR 1913)")

        # Stamp Button
        s_rect = self._get_stamp_button_rect()
        painter.setBrush(QBrush(QColor("#7c3aed")))
        painter.setPen(QPen(QColor("#a78bfa"), 1.2))
        painter.drawRoundedRect(s_rect, 6.0, 6.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(s_rect, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu bảng")

        # Close Button
        c_rect = self._get_close_button_rect()
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(c_rect, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(c_rect, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_selectors(self, painter: QPainter) -> None:
        # 1. Preset Buttons
        for key, rect in self._get_preset_rects():
            p_data = PRESETS[key]
            is_active = (key == self._current_preset)
            bg = QColor("#6d28d9") if is_active else QColor("#1e293b")
            border = QColor("#a78bfa") if is_active else QColor("#334155")
            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.2 if is_active else 1.0))
            painter.drawRoundedRect(rect, 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#cbd5e1"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, p_data["label"])

        # 2. Mode Toggle Tabs
        tab_orb, tab_spec = self._get_mode_toggle_rects()
        for tab_rect, mode_id, label in [
            (tab_orb, "ORBITS_AND_LEVELS", "Quỹ đạo & Mức Năng lượng"),
            (tab_spec, "EMISSION_SPECTRUM", "Vạch Quang phổ Hydro"),
        ]:
            is_on = (self._diagram_mode == mode_id)
            painter.setBrush(QBrush(QColor("#5b21b6") if is_on else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#8b5cf6") if is_on else QColor("#334155"), 1.2 if is_on else 1.0))
            painter.drawRoundedRect(tab_rect, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_on else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_on else QColor("#94a3b8"))
            painter.drawText(tab_rect, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_bohr_orbits(self, painter: QPainter) -> None:
        """Paints concentric circular quantum orbits n = 1..6 and moving electron."""
        x_left = 14.0
        y_top = 104.0
        w = 260.0
        h = 246.0

        painter.setBrush(QBrush(QColor("#030712")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(QRectF(x_left, y_top, w, h), 8.0, 8.0)

        # Center of atom
        cx = x_left + w / 2.0
        cy = y_top + h / 2.0 - 12.0

        # Draw concentric orbits (n = 1..6)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Normal))
        for n in range(1, 7):
            r_px = 25.0 + n * 14.0
            is_active_orbit = (n == self._current_electron_orbit)
            is_target_orbit = (n == self._level_n)

            if is_active_orbit:
                pen = QPen(QColor("#38bdf8"), 1.4)
            elif is_target_orbit:
                pen = QPen(QColor("#f59e0b"), 1.2, Qt.PenStyle.DashLine)
            else:
                pen = QPen(QColor("#334155"), 1.0, Qt.PenStyle.DotLine)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(pen)
            painter.drawEllipse(QPointF(cx, cy), r_px, r_px)

            # Orbit label: K, L, M, N, O, P
            painter.setPen(QColor("#64748b"))
            painter.drawText(int(cx + r_px - 8), int(cy - 2), ORBIT_NAMES.get(n, str(n)))

        # Proton Nucleus (+e) at Center
        p_glow = QRadialGradient(cx, cy, 14.0)
        p_glow.setColorAt(0.0, QColor("#ef4444"))
        p_glow.setColorAt(1.0, QColor(239, 68, 68, 0))
        painter.setBrush(QBrush(p_glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(cx, cy), 14.0, 14.0)

        painter.setBrush(QBrush(QColor("#dc2626")))
        painter.setPen(QPen(QColor("#fca5a5"), 1.0))
        painter.drawEllipse(QPointF(cx, cy), 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(int(cx - 4), int(cy + 3), "+")

        # Revolving Electron (-e) on active orbit
        active_r_px = 25.0 + self._current_electron_orbit * 14.0
        ex = cx + active_r_px * math.cos(self._electron_angle_rad)
        ey = cy + active_r_px * math.sin(self._electron_angle_rad)

        # Electron Glow & Sphere
        e_glow = QRadialGradient(ex, ey, 9.0)
        e_glow.setColorAt(0.0, QColor(0, 245, 255, 180))
        e_glow.setColorAt(1.0, QColor(0, 245, 255, 0))
        painter.setBrush(QBrush(e_glow))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(ex, ey), 9.0, 9.0)

        painter.setBrush(QBrush(QColor("#00f5ff")))
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.drawEllipse(QPointF(ex, ey), 4.0, 4.0)

        # Draw Flying Photon Packets
        for pkt in self._photon_packets:
            px = cx + pkt["x"]
            py = cy + pkt["y"]
            pkt_color = pkt["color"]
            painter.setBrush(QBrush(pkt_color))
            painter.setPen(QPen(QColor("#ffffff"), 1.0))
            painter.drawEllipse(QPointF(px, py), 4.5, 4.5)

            # Radiating wavy pulse
            wave_path = QPainterPath()
            wave_path.moveTo(px, py)
            wave_path.quadTo(px + 6, py - 4, px + 12, py)
            wave_path.quadTo(px + 18, py + 4, px + 24, py)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(pkt_color, 1.5))
            painter.drawPath(wave_path)

        # Telemetry Banner at bottom of left pane
        tele_rect = QRectF(x_left + 10.0, y_top + h - 32.0, w - 20.0, 22.0)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(tele_rect, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        v_kms = self.get_electron_velocity_kms(self._current_electron_orbit)
        r_ang = self.get_orbit_radius_angstrom(self._current_electron_orbit)
        painter.drawText(
            tele_rect,
            Qt.AlignmentFlag.AlignCenter,
            f"Quỹ đạo {ORBIT_NAMES.get(self._current_electron_orbit, '')} (n={self._current_electron_orbit}): r = {r_ang:.2f}Å | v = {v_kms:.0f}km/s",
        )

    def _paint_diagnostic_pane(self, painter: QPainter) -> None:
        """Paints either Energy Level Ladder or Optical Emission Spectrum Bar."""
        x_left = 284.0
        y_top = 104.0
        w = 302.0
        h = 246.0

        painter.setBrush(QBrush(QColor("#030712")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(QRectF(x_left, y_top, w, h), 8.0, 8.0)

        if self._diagram_mode == "ORBITS_AND_LEVELS":
            self._paint_energy_ladder(painter, x_left, y_top, w, h)
        else:
            self._paint_emission_spectrum_bar(painter, x_left, y_top, w, h)

    def _paint_energy_ladder(self, painter: QPainter, x_left: float, y_top: float, w: float, h: float) -> None:
        """Draws discrete Bohr energy levels E_n = -13.6/n^2 and transition arrow."""
        plot_x = x_left + 45.0
        plot_y = y_top + 16.0
        plot_w = w - 60.0
        plot_h = h - 70.0
        bottom_y = plot_y + plot_h

        # Axis: Energy in eV from -13.6 eV up to 0 eV
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawLine(int(plot_x), int(plot_y), int(plot_x), int(bottom_y))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(int(plot_x - 38), int(plot_y + 10), "E (eV)")

        def ev_to_y(e_val: float) -> float:
            # -13.6 at bottom, 0.0 at top
            frac = (e_val - (-13.606)) / (0.0 - (-13.606))
            frac = max(0.0, min(1.0, frac))
            return bottom_y - frac * (plot_h - 15.0)

        # Draw levels n = 1..6 and n = inf
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        for n in range(1, 7):
            en = self.get_energy_level_ev(n)
            ly = ev_to_y(en)

            is_m = (n == self._level_m)
            is_n = (n == self._level_n)

            if is_m or is_n:
                pen = QPen(QColor("#38bdf8") if is_m else QColor("#f59e0b"), 1.8)
            else:
                pen = QPen(QColor("#475569"), 1.0)

            painter.setPen(pen)
            painter.drawLine(int(plot_x), int(ly), int(plot_x + plot_w), int(ly))

            # Label on right
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(int(plot_x + plot_w + 4), int(ly + 4), f"n={n}")
            painter.drawText(int(plot_x - 36), int(ly + 4), f"{en:.2f}")

        # Zero energy level (Ionization)
        y_inf = ev_to_y(0.0)
        painter.setPen(QPen(QColor("#94a3b8"), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(int(plot_x), int(y_inf), int(plot_x + plot_w), int(y_inf))
        painter.drawText(int(plot_x + plot_w + 4), int(y_inf + 4), "n=∞")
        painter.drawText(int(plot_x - 36), int(y_inf + 4), "0.00")

        # Transition arrow from level m to level n
        ym = ev_to_y(self.get_energy_level_ev(self._level_m))
        yn = ev_to_y(self.get_energy_level_ev(self._level_n))
        arr_x = plot_x + plot_w * 0.55

        wl = self.get_transition_wavelength_nm()
        arr_color, band_str, _ = wavelength_to_spectral_color(wl)

        painter.setPen(QPen(arr_color, 2.2))
        painter.drawLine(int(arr_x), int(ym), int(arr_x), int(yn))
        # Arrowhead pointing to level n
        dir_sign = 1 if yn > ym else -1
        painter.drawLine(int(arr_x), int(yn), int(arr_x - 4), int(yn - 6 * dir_sign))
        painter.drawLine(int(arr_x), int(yn), int(arr_x + 4), int(yn - 6 * dir_sign))

        # Metrics text below graph
        delta_e = self.get_transition_energy_ev()
        stat_y = y_top + h - 38.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        painter.setPen(QColor("#cbd5e1"))
        lines = [
            f"• Năng lượng photon: ε = |E_{self._level_m} - E_{self._level_n}| = {delta_e:.3f} eV",
            f"• Bước sóng: λ = {wl:.1f} nm  |  {self.get_series_name()}",
        ]
        painter.drawText(int(x_left + 10), int(stat_y), lines[0])
        painter.drawText(int(x_left + 10), int(stat_y + 16), lines[1])

    def _paint_emission_spectrum_bar(self, painter: QPainter, x_left: float, y_top: float, w: float, h: float) -> None:
        """Draws authentic visible emission spectrum bar of Hydrogen (Balmer series)."""
        spec_x = x_left + 24.0
        spec_y = y_top + 45.0
        spec_w = w - 48.0
        spec_h = 45.0

        # Dark optical band
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#475569"), 1.2))
        painter.drawRoundedRect(QRectF(spec_x, spec_y, spec_w, spec_h), 4.0, 4.0)

        # Wavelength range 380 nm to 750 nm
        wl_min = 380.0
        wl_max = 750.0

        def wl_to_px(wl_val: float) -> float:
            return spec_x + ((wl_val - wl_min) / (wl_max - wl_min)) * spec_w

        # Four Authentic Balmer Lines:
        # H_alpha (656.3 nm - Red), H_beta (486.1 nm - Cyan), H_gamma (434.1 nm - Indigo), H_delta (410.2 nm - Violet)
        balmer_lines = [
            (656.28, QColor("#ef4444"), "H_α (656.3 nm)", "3 → 2"),
            (486.13, QColor("#06b6d4"), "H_β (486.1 nm)", "4 → 2"),
            (434.05, QColor("#6366f1"), "H_γ (434.1 nm)", "5 → 2"),
            (410.17, QColor("#a855f7"), "H_δ (410.2 nm)", "6 → 2"),
        ]

        for wl_line, col, name, trans in balmer_lines:
            lx = wl_to_px(wl_line)
            # Glowing bright spectral line
            glow_pen = QPen(QColor(col.red(), col.green(), col.blue(), 100), 5.0)
            painter.setPen(glow_pen)
            painter.drawLine(int(lx), int(spec_y + 2), int(lx), int(spec_y + spec_h - 2))

            painter.setPen(QPen(col, 2.0))
            painter.drawLine(int(lx), int(spec_y + 2), int(lx), int(spec_y + spec_h - 2))

            # Text label above and below
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(col)
            painter.drawText(int(lx - 12), int(spec_y - 6), name.split()[0])
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Normal))
            painter.drawText(int(lx - 10), int(spec_y + spec_h + 14), trans)

        # Current transition line marker
        curr_wl = self.get_transition_wavelength_nm()
        if wl_min <= curr_wl <= wl_max:
            curr_x = wl_to_px(curr_wl)
            painter.setPen(QPen(QColor("#ffffff"), 1.5, Qt.PenStyle.DashLine))
            painter.drawLine(int(curr_x), int(spec_y - 14), int(curr_x), int(spec_y + spec_h + 20))
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawEllipse(QPointF(curr_x, spec_y + spec_h / 2.0), 3.0, 3.0)

        # Explanatory text
        stat_y = y_top + h - 50.0
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Normal))
        painter.setPen(QColor("#cbd5e1"))
        lines = [
            "• Dãy Balmer tạo nên 4 vạch phát xạ nhìn thấy đặc trưng của Hydro.",
            f"• Vạch đang chọn: λ = {curr_wl:.1f} nm ({self.get_series_name()})",
            f"• Công thức Rydberg: 1/λ = R_H (1/{self._level_n}² - 1/{self._level_m}²)",
        ]
        ty = stat_y
        for ln in lines:
            painter.drawText(int(x_left + 10), int(ty), ln)
            ty += 15.0

    def _paint_controls(self, painter: QPainter) -> None:
        """Paints level selector chips m (initial), n (final), and Quantum Jump button."""
        m_chips, n_chips = self._get_level_chips()

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(16, 386, "Mức đầu m:")

        painter.setPen(QColor("#f59e0b"))
        painter.drawText(16, 410, "Mức cuối n:")

        # Level m chips
        for num, rect in m_chips:
            is_active = (num == self._level_m)
            painter.setBrush(QBrush(QColor("#0284c7") if is_active else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#334155"), 1.2 if is_active else 1.0))
            painter.drawRoundedRect(rect, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(num))

        # Level n chips
        for num, rect in n_chips:
            is_active = (num == self._level_n)
            painter.setBrush(QBrush(QColor("#d97706") if is_active else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#f59e0b") if is_active else QColor("#334155"), 1.2 if is_active else 1.0))
            painter.drawRoundedRect(rect, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(num))

        # Quantum Jump Trigger Button
        j_rect = self._get_jump_button_rect()
        painter.setBrush(QBrush(QColor("#7c3aed")))
        painter.setPen(QPen(QColor("#a78bfa"), 1.2))
        painter.drawRoundedRect(j_rect, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(j_rect, Qt.AlignmentFlag.AlignCenter, "⚡ Nhảy mức")

    # ── Mouse & Touchless Air Gesture Handlers ────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        loc = event.pos()

        # Close
        if self._get_close_button_rect().contains(loc):
            self.close()
            event.accept()
            return

        # Stamp
        if self._get_stamp_button_rect().contains(loc):
            self.stamp_to_board()
            event.accept()
            return

        # Presets
        for key, rect in self._get_preset_rects():
            if rect.contains(loc):
                self.apply_preset(key)
                event.accept()
                return

        # Mode toggles
        tab_orb, tab_spec = self._get_mode_toggle_rects()
        if tab_orb.contains(loc):
            self.set_diagram_mode("ORBITS_AND_LEVELS")
            event.accept()
            return
        elif tab_spec.contains(loc):
            self.set_diagram_mode("EMISSION_SPECTRUM")
            event.accept()
            return

        # Jump Button
        if self._get_jump_button_rect().contains(loc):
            self.trigger_quantum_jump()
            event.accept()
            return

        # Chips for m and n
        m_chips, n_chips = self._get_level_chips()
        for num, rect in m_chips:
            if rect.contains(loc):
                self.set_level_m(num)
                event.accept()
                return
        for num, rect in n_chips:
            if rect.contains(loc):
                self.set_level_n(num)
                event.accept()
                return

        super().mousePressEvent(event)

    def handle_air_pinch(self, scene_pt: QPointF) -> bool:
        loc = self.mapFromScene(scene_pt)

        # Close
        if self._get_close_button_rect().contains(loc):
            self.close()
            return True

        # Stamp
        if self._get_stamp_button_rect().contains(loc):
            self.stamp_to_board()
            return True

        # Presets
        for key, rect in self._get_preset_rects():
            if rect.contains(loc):
                self.apply_preset(key)
                return True

        # Mode toggles
        tab_orb, tab_spec = self._get_mode_toggle_rects()
        if tab_orb.contains(loc):
            self.set_diagram_mode("ORBITS_AND_LEVELS")
            return True
        elif tab_spec.contains(loc):
            self.set_diagram_mode("EMISSION_SPECTRUM")
            return True

        # Jump Button
        if self._get_jump_button_rect().contains(loc):
            self.trigger_quantum_jump()
            return True

        # Level chips
        m_chips, n_chips = self._get_level_chips()
        for num, rect in m_chips:
            if rect.adjusted(-4, -4, 4, 4).contains(loc):
                self.set_level_m(num)
                return True
        for num, rect in n_chips:
            if rect.adjusted(-4, -4, 4, 4).contains(loc):
                self.set_level_n(num)
                return True

        return False

    def handle_air_tap(self, scene_pt: QPointF) -> bool:
        return self.handle_air_pinch(scene_pt)

    # ── LessonBoard Stamping ──────────────────────────────────────────────────

    def stamp_to_board(self) -> None:
        """Stamps permanent vector diagram of Bohr atom & Hydrogen spectrum onto LessonBoard."""
        board = self.scene()
        if board is None:
            return

        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        diagram_items = PhysicsTools.create_bohr_atom_diagram(
            center=(cx, cy + 30.0),
            width=520.0,
            height=320.0,
            level_m=self._level_m,
            level_n=self._level_n,
            preset=self._current_preset,
        )
        for it in diagram_items:
            board.addItem(it)

        if hasattr(board, "commit_state_action"):
            board.commit_state_action("Stamp Bohr Atom Spectrum Diagram")
        logger.info(f"Bohr Atom diagram stamped to LessonBoard at ({cx:.1f}, {cy:.1f}).")
