"""
AirTeach AI – Interactive Quantum Mechanics: Particle in a 1D/2D Infinite Potential Well (Particle in a Box)
============================================================================================================
Provides an authentic quantum physics simulation of the fundamental infinite square potential well:
  - Time-Independent Schrödinger Equation (TISE):
      - (hbar^2 / (2m)) * d^2ψ/dx^2 + V(x)*ψ = E*ψ
      For infinite well V(x) = 0 (0 <= x <= L), V(x) = ∞ elsewhere.
      Boundary conditions: ψ(0) = 0 and ψ(L) = 0.
  - Quantized Stationary Wavefunctions:
      ψ_n(x) = sqrt(2/L) * sin(n * pi * x / L) for n = 1, 2, 3, 4, ...
      De Broglie wavelength: λ_n = 2L / n
      Nodes count: (n - 1) internal nodes where probability density vanishes!
  - Born Statistical Interpretation & Probability Density:
      P_n(x) = |ψ_n(x)|^2 = (2/L) * sin^2(n * pi * x / L)
      Normalization: integral_0^L |ψ_n(x)|^2 dx = 1.0
  - Quantized Energy Levels & Zero-Point Energy:
      E_n = n^2 * h^2 / (8 * m * L^2) = n^2 * E_1
      Zero-point energy: E_1 > 0 (particle can NEVER be at absolute rest).
  - Heisenberg Uncertainty Principle:
      Δx = L * sqrt(1/12 - 1/(2 * n^2 * pi^2))
      Δp = n * pi * hbar / L
      Δx * Δp = (hbar / 2) * sqrt((n^2 * pi^2 / 3) - 2) >= hbar / 2
  - Time-Dependent Quantum Superposition & Quantum Beats:
      Ψ(x, t) = (1/sqrt(2)) * [ψ_1(x)*e^(-i E_1 t/hbar) + ψ_2(x)*e^(-i E_2 t/hbar)]
      |Ψ(x, t)|^2 oscillates across the well at Bohr transition frequency ω_21 = (E_2 - E_1) / hbar.
  - 5 Curriculum Presets:
      1. GROUND_STATE_N1: n = 1 Ground state with zero-point energy E_1, single central antinode.
      2. EXCITED_STATES_N2_N3_N4: Higher quantized eigenstates with multiple nodes.
      3. SUPERPOSITION_QUANTUM_BEATS: Coherent quantum oscillation at transition frequency ω_21.
      4. HEISENBERG_UNCERTAINTY_VERIFIER: Real-time verification of Δx · Δp >= hbar / 2.
      5. DEGENERATE_2D_QUANTUM_BOX: 2D potential box (L_x × L_y) and geometric energy degeneracy E_(1,2) = E_(2,1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)


# Physical Constants (CODATA 2022)
H_PLANCK = 6.62607015e-34            # Planck constant (J·s)
H_BAR = 1.054571817e-34              # Reduced Planck constant (J·s)
M_ELECTRON = 9.1093837015e-31        # Electron rest mass (kg)
M_PROTON = 1.67262192369e-27         # Proton rest mass (kg)
EV_TO_JOULE = 1.602176634e-19        # 1 eV in Joules


@dataclass(frozen=True)
class QuantumPreset:
    code: str
    title_vi: str
    badge_label: str
    n_quantum: int
    well_width_nm: float
    description: str


PRESETS: dict[str, QuantumPreset] = {
    "GROUND_STATE_N1": QuantumPreset(
        code="GROUND_STATE_N1",
        title_vi="Trạng thái cơ bản n = 1 (Năng lượng điểm không E₁)",
        badge_label="CƠ BẢN n=1",
        n_quantum=1,
        well_width_nm=1.00,
        description="n = 1: Năng lượng điểm không E₁ > 0. Hạt không bao giờ đứng yên hoàn toàn do nguyên lý bất định Heisenberg.",
    ),
    "EXCITED_STATES_N2_N3_N4": QuantumPreset(
        code="EXCITED_STATES_N2_N3_N4",
        title_vi="Các trạng thái kích thích n = 2, 3, 4 (Nút sóng lượng tử)",
        badge_label="KÍCH THÍCH",
        n_quantum=2,
        well_width_nm=1.00,
        description="Năng lượng tỷ lệ bậc hai E_n = n²·E₁. Xuất hiện (n-1) nút sóng nơi xác suất tìm thấy hạt triệt tiêu bằng 0.",
    ),
    "SUPERPOSITION_QUANTUM_BEATS": QuantumPreset(
        code="SUPERPOSITION_QUANTUM_BEATS",
        title_vi="Chồng chập Lượng tử & Dao động Phách (Quantum Beats)",
        badge_label="CHỒNG CHẬP",
        n_quantum=1,
        well_width_nm=1.00,
        description="Trạng thái chồng chập Ψ = (ψ₁ + ψ₂)/√2 dao động qua lại trong giếng với tần số chuyển mức Bohr ω₂₁ = (E₂-E₁)/ℏ.",
    ),
    "HEISENBERG_UNCERTAINTY_VERIFIER": QuantumPreset(
        code="HEISENBERG_UNCERTAINTY_VERIFIER",
        title_vi="Xác thực Hệ thức Bất định Heisenberg (Δx·Δp ≥ ℏ/2)",
        badge_label="BẤT ĐỊNH",
        n_quantum=1,
        well_width_nm=1.00,
        description="Độ bất định tọa độ Δx và xung lượng Δp tuân thủ nghiêm ngặt hệ thức bất định cơ bản của cơ học lượng tử.",
    ),
    "DEGENERATE_2D_QUANTUM_BOX": QuantumPreset(
        code="DEGENERATE_2D_QUANTUM_BOX",
        title_vi="Giếng thế 2D & Suy biến năng lượng E_(1,2) = E_(2,1)",
        badge_label="GIẾNG 2D",
        n_quantum=2,
        well_width_nm=1.00,
        description="Trong giếng 2D hình vuông, hai trạng thái khác biệt (1,2) và (2,1) có mức năng lượng hoàn toàn bằng nhau (Suy biến bậc 2).",
    ),
}


class InteractiveQuantumWellItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Quantum Mechanics: Particle in a 1D/2D Infinite Potential Box,
    Schrödinger wavefunctions ψ_n(x), Born probability density |ψ_n(x)|^2, and Heisenberg uncertainty.
    """

    WIDTH = 700.0
    HEIGHT = 440.0

    stamp_requested = Signal()
    quantum_number_changed = Signal(int)

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (140.0, 50.0),
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)

        if isinstance(pos, QPointF):
            self.setPos(pos)
        else:
            self.setPos(QPointF(float(pos[0]), float(pos[1])))

        self.setZValue(25.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )

        # Simulation Preset & Physical Parameters
        self._active_preset_code: str = "GROUND_STATE_N1"
        self._preset: QuantumPreset = PRESETS[self._active_preset_code]

        self._n: int = 1                         # Principal quantum number (n = 1, 2, 3, 4, 5)
        self._nx: int = 1                        # 2D x quantum number
        self._ny: int = 2                        # 2D y quantum number
        self._well_width_nm: float = 1.00        # Well width L in nanometers (10^-9 m)
        self._particle_mass_kg: float = M_ELECTRON # Default: Electron
        self._particle_name: str = "Electron (e⁻)"

        # Display Toggles: 0 = Both, 1 = Wavefunction ψ only, 2 = Probability Density |ψ|² only
        self._display_mode: int = 0
        self._is_superposition: bool = False

        # Animation State
        self._anim_time_fs: float = 0.0          # Time in femtoseconds (10^-15 s)
        self._anim_phase: float = 0.0
        self._is_animating: bool = True

        # Clickable button rects cache
        self._btn_rects: dict[str, QRectF] = {}

        # 30 FPS Animation Timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Physics Calculations (Zero Mock Data)
    # -------------------------------------------------------------------------
    @property
    def well_width_m(self) -> float:
        """Well width L in meters."""
        return self._well_width_nm * 1e-9

    @property
    def ground_state_energy_joules(self) -> float:
        """Zero-point ground state energy E_1 = h^2 / (8 * m * L^2) in Joules."""
        return (H_PLANCK ** 2) / (8.0 * self._particle_mass_kg * (self.well_width_m ** 2))

    @property
    def ground_state_energy_ev(self) -> float:
        """E_1 in electron-volts (eV)."""
        return self.ground_state_energy_joules / EV_TO_JOULE

    @property
    def current_energy_joules(self) -> float:
        """E_n = n^2 * E_1 in Joules."""
        return (self._n ** 2) * self.ground_state_energy_joules

    @property
    def current_energy_ev(self) -> float:
        """E_n in electron-volts (eV)."""
        return (self._n ** 2) * self.ground_state_energy_ev

    @property
    def de_broglie_wavelength_nm(self) -> float:
        """De Broglie wavelength inside well: λ_n = 2L / n."""
        return (2.0 * self._well_width_nm) / float(self._n)

    @property
    def node_count(self) -> int:
        """Number of interior nodes where wavefunction and probability vanish."""
        return self._n - 1

    @property
    def bohr_transition_freq_rad_per_fs(self) -> float:
        """Angular frequency ω_21 = (E_2 - E_1) / hbar in rad/fs for superposition beats."""
        delta_e = 3.0 * self.ground_state_energy_joules
        omega_s = delta_e / H_BAR
        return omega_s * 1e-15  # rad per femtosecond

    @property
    def heisenberg_uncertainty_product(self) -> float:
        """Ratio (Δx * Δp) / (hbar / 2) >= 1.0."""
        n_sq = float(self._n ** 2)
        val = math.sqrt((n_sq * (math.pi ** 2) / 3.0) - 2.0)
        return val

    def wavefunction_psi(self, x_norm: float, n: int) -> float:
        """
        Normalized 1D spatial wavefunction:
        ψ_n(x) = sqrt(2/L) * sin(n * pi * x / L).
        x_norm in [0.0, 1.0].
        """
        if x_norm <= 0.0 or x_norm >= 1.0:
            return 0.0
        norm_factor = math.sqrt(2.0 / self.well_width_m)
        val = norm_factor * math.sin(float(n) * math.pi * x_norm)
        return 0.0 if abs(val) < 1e-10 else val

    def probability_density(self, x_norm: float, n: int) -> float:
        """Probability density P(x) = |ψ_n(x)|^2."""
        psi = self.wavefunction_psi(x_norm, n)
        return psi ** 2

    def superposition_density(self, x_norm: float, t_fs: float) -> float:
        """
        Time-dependent superposition |Ψ(x, t)|^2 for state (ψ_1 + ψ_2) / sqrt(2):
        P(x, t) = 0.5 * [|ψ_1|^2 + |ψ_2|^2 + 2 * ψ_1 * ψ_2 * cos(ω_21 * t)].
        """
        if x_norm <= 0.0 or x_norm >= 1.0:
            return 0.0
        psi1 = self.wavefunction_psi(x_norm, 1)
        psi2 = self.wavefunction_psi(x_norm, 2)
        omega = self.bohr_transition_freq_rad_per_fs
        cos_term = math.cos(omega * t_fs)
        density = 0.5 * (psi1 ** 2 + psi2 ** 2 + 2.0 * psi1 * psi2 * cos_term)
        return max(0.0, density)

    # -------------------------------------------------------------------------
    # Getters & Setters
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> QuantumPreset:
        return self._preset

    @property
    def quantum_number_n(self) -> int:
        return self._n

    @property
    def well_width_nm(self) -> float:
        return self._well_width_nm

    @property
    def display_mode(self) -> int:
        return self._display_mode

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    def load_preset(self, preset_code: str) -> None:
        """Loads curriculum preset and sets appropriate quantum state."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._n = self._preset.n_quantum
        self._well_width_nm = self._preset.well_width_nm
        self._is_superposition = (preset_code == "SUPERPOSITION_QUANTUM_BEATS")
        self.quantum_number_changed.emit(self._n)
        self.update()

    def set_quantum_number(self, n: int) -> None:
        """Sets principal quantum number clamped to [1, 5]."""
        self._n = max(1, min(5, int(n)))
        self.quantum_number_changed.emit(self._n)
        self.update()

    def set_well_width_nm(self, width_nm: float) -> None:
        self._well_width_nm = max(0.5, min(3.0, round(width_nm, 2)))
        self.update()

    def set_particle_type(self, particle: str) -> None:
        if particle.lower() == "proton":
            self._particle_mass_kg = M_PROTON
            self._particle_name = "Proton (p⁺)"
        else:
            self._particle_mass_kg = M_ELECTRON
            self._particle_name = "Electron (e⁻)"
        self.update()

    def cycle_display_mode(self) -> None:
        """Cycles between: 0 (Both), 1 (ψ only), 2 (|ψ|² only)."""
        self._display_mode = (self._display_mode + 1) % 3
        self.update()

    def toggle_superposition(self) -> None:
        self._is_superposition = not self._is_superposition
        self.update()

    def toggle_animation(self) -> None:
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def stamp_to_board(self) -> None:
        self.stamp_requested.emit()

    def cleanup(self) -> None:
        if self._timer.isActive():
            self._timer.stop()

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_tick(self) -> None:
        self._anim_phase += 0.08
        if self._anim_phase >= 2.0 * math.pi:
            self._anim_phase -= 2.0 * math.pi

        # Advance femtosecond time for quantum beats
        dt_fs = 0.25
        self._anim_time_fs += dt_fs

        self.update()

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        if painter is None:
            return

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Box
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#070b14")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header & Preset Tabs
        self._draw_header(painter)

        # 3. Left Simulation Viewport (Potential Well & Quantum Curves)
        left_w = 445.0
        view_rect = QRectF(14.0, 48.0, left_w, 310.0)
        self._draw_viewport(painter, view_rect)

        # 4. Right Control & Quantum Readout Panel
        right_rect = QRectF(view_rect.right() + 10.0, 48.0, self.WIDTH - view_rect.right() - 24.0, 310.0)
        self._draw_control_panel(painter, right_rect)

        # 5. Bottom Physics Theory Banner
        bot_rect = QRectF(14.0, view_rect.bottom() + 10.0, self.WIDTH - 28.0, self.HEIGHT - view_rect.bottom() - 20.0)
        self._draw_theory_banner(painter, bot_rect)

        painter.restore()

    # -------------------------------------------------------------------------
    # Sub-Drawing Modules
    # -------------------------------------------------------------------------
    def _draw_header(self, painter: QPainter) -> None:
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QPointF(16.0, 24.0), "CƠ HỌC LƯỢNG TỬ: GIẾNG THẾ VÔ HẠN (PARTICLE IN A BOX)")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(16.0, 38.0), "Hàm sóng ψ_n(x) • Mật độ xác suất |ψ|² • Mức năng lượng E_n=n²E₁ • Bất định Heisenberg")

        # Preset Tabs
        px = 440.0
        py = 10.0
        pw = 42.0
        ph = 24.0

        for code, preset in PRESETS.items():
            btn_r = QRectF(px, py, pw, ph)
            self._btn_rects[f"preset_{code}"] = btn_r

            is_sel = (code == self._active_preset_code)
            painter.setBrush(QBrush(QColor("#0284c7") if is_sel else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_sel else QColor("#334155"), 1.0))
            painter.drawRoundedRect(btn_r, 4.0, 4.0)

            lbl = preset.badge_label.split()[0]
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_sel else QColor("#94a3b8"))
            painter.drawText(btn_r, Qt.AlignmentFlag.AlignCenter, lbl)
            px += pw + 4.0

        # Stamp Button
        stamp_r = QRectF(self.WIDTH - 52.0, 10.0, 38.0, 24.0)
        self._btn_rects["stamp"] = stamp_r
        painter.setBrush(QBrush(QColor("#059669")))
        painter.setPen(QPen(QColor("#10b981"), 1.0))
        painter.drawRoundedRect(stamp_r, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(stamp_r, Qt.AlignmentFlag.AlignCenter, "Ghim")

    def _draw_viewport(self, painter: QPainter, rect: QRectF) -> None:
        painter.save()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setClipRect(rect)

        if self._active_preset_code == "DEGENERATE_2D_QUANTUM_BOX":
            self._draw_2d_quantum_box(painter, rect)
            painter.restore()
            return

        # 1. Infinite Barrier Walls V(x) = ∞
        wall_w = 40.0
        well_left = rect.left() + 50.0
        well_right = rect.right() - 50.0
        well_w = well_right - well_left
        cy = rect.center().y() + 15.0

        # Left Wall V=∞
        left_wall = QRectF(rect.left(), rect.top() + 35.0, 50.0, rect.height() - 45.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.5))
        painter.drawRect(left_wall)

        # Right Wall V=∞
        right_wall = QRectF(well_right, rect.top() + 35.0, 50.0, rect.height() - 45.0)
        painter.drawRect(right_wall)

        # Wall hatch marks
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(QRectF(rect.left() + 4.0, cy - 20.0, 42.0, 40.0), Qt.AlignmentFlag.AlignCenter, "V = ∞\n(Vách)")
        painter.drawText(QRectF(well_right + 4.0, cy - 20.0, 42.0, 40.0), Qt.AlignmentFlag.AlignCenter, "V = ∞\n(Vách)")

        # Center potential region V(x) = 0
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QColor("#334155"))
        painter.drawText(QPointF(well_left + well_w / 2.0 - 45.0, cy + 85.0), "Vùng thế V(x) = 0")

        # Zero baseline
        painter.setPen(QPen(QColor("#334155"), 1.0, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(well_left, cy), QPointF(well_right, cy))

        # Boundary coordinate labels (x = 0, x = L)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QPointF(well_left - 15.0, cy + 22.0), "x = 0")
        painter.drawText(QPointF(well_right - 10.0, cy + 22.0), f"x = L ({self._well_width_nm:.1f}nm)")

        # 2. Render Curves
        n_steps = 140
        amp_psi = 70.0
        amp_prob = 100.0

        is_super = self._is_superposition
        disp = self._display_mode  # 0: Both, 1: ψ, 2: |ψ|²

        # A. Probability Density |ψ|² (Golden filled shaded area)
        if disp in [0, 2]:
            prob_path = QPainterPath()
            prob_path.moveTo(well_left, cy)
            norm_max = math.sqrt(2.0 / self.well_width_m)

            for i in range(n_steps + 1):
                xn = i / float(n_steps)
                px = well_left + xn * well_w
                if is_super:
                    p_val = self.superposition_density(xn, self._anim_time_fs)
                    py = cy - (p_val / (norm_max ** 2)) * amp_prob
                else:
                    p_val = self.probability_density(xn, self._n)
                    py = cy - (p_val / (norm_max ** 2)) * amp_prob
                prob_path.lineTo(px, py)

            prob_path.lineTo(well_right, cy)
            prob_path.closeSubpath()

            painter.setBrush(QBrush(QColor(251, 191, 36, 45)))
            painter.setPen(QPen(QColor("#fbbf24"), 1.6))
            painter.drawPath(prob_path)

        # B. Wavefunction ψ(x) (Cyan curve)
        if disp in [0, 1] and not is_super:
            psi_path = QPainterPath()
            norm_max = math.sqrt(2.0 / self.well_width_m)

            for i in range(n_steps + 1):
                xn = i / float(n_steps)
                px = well_left + xn * well_w
                psi_val = self.wavefunction_psi(xn, self._n)
                py = cy - (psi_val / norm_max) * amp_psi
                if i == 0:
                    psi_path.moveTo(px, py)
                else:
                    psi_path.lineTo(px, py)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#38bdf8"), 2.2))
            painter.drawPath(psi_path)

        # 3. Status Callouts
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        mode_label = "CHỒNG CHẬP Ψ = (ψ₁ + ψ₂)/√2 (DAO ĐỘNG PHÁCH)" if is_super else f"TRẠNG THÁI n = {self._n} • E_{self._n} = {self.current_energy_ev:.3f} eV ({self.current_energy_joules*1e20:.2f} × 10⁻²⁰ J)"
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 20.0), mode_label)

        # Legend badge
        leg_r = QRectF(rect.right() - 170.0, rect.top() + 8.0, 160.0, 22.0)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#334155"), 0.8))
        painter.drawRoundedRect(leg_r, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(leg_r.left() + 8.0, leg_r.top() + 14.0), "━ ψ(x)")
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QPointF(leg_r.left() + 65.0, leg_r.top() + 14.0), "■ |ψ(x)|² Xác suất")

        painter.restore()

    def _draw_2d_quantum_box(self, painter: QPainter, rect: QRectF) -> None:
        """Visualizes 2D square quantum well (nx, ny) contour and geometric degeneracy."""
        cx = rect.center().x()
        cy = rect.center().y() + 10.0
        sz = 180.0

        box_rect = QRectF(cx - sz / 2.0, cy - sz / 2.0, sz, sz)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.drawRect(box_rect)

        # 2D Probability Grid Heatmap
        grid_n = 24
        cell_sz = sz / float(grid_n)

        nx, ny = self._nx, self._ny

        for ix in range(grid_n):
            xn = (ix + 0.5) / float(grid_n)
            for iy in range(grid_n):
                yn = (iy + 0.5) / float(grid_n)
                # 2D Wavefunction: ψ(x, y) = 2/L * sin(nx * pi * x) * sin(ny * pi * y)
                psi_2d = math.sin(nx * math.pi * xn) * math.sin(ny * math.pi * yn)
                prob_2d = psi_2d ** 2

                # Color map: deep navy (0) to bright cyan/yellow (1)
                alpha = int(240 * prob_2d)
                col = QColor(251, 191, 36, alpha)
                painter.setBrush(QBrush(col))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRect(QRectF(box_rect.left() + ix * cell_sz, box_rect.top() + iy * cell_sz, cell_sz + 0.5, cell_sz + 0.5))

        # Header in 2D mode
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        e_2d = (nx ** 2 + ny ** 2) * self.ground_state_energy_ev
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 20.0), f"GIẾNG THẾ 2D: TRẠNG THÁI ({nx}, {ny}) • NĂNG LƯỢNG E = {e_2d:.2f} eV (SUY BIẾN VỚI ({ny}, {nx}))")

    # -------------------------------------------------------------------------
    # Right Control & Readout Panel
    # -------------------------------------------------------------------------
    def _draw_control_panel(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)

        ox = rect.left() + 10.0
        w = rect.width() - 20.0
        y = rect.top() + 14.0

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(ox, y), "THÔNG SỐ LƯỢNG TỬ:")

        # Quantum Number n Card
        y += 18.0
        n_card = QRectF(ox, y, w, 32.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(n_card, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QRectF(ox + 8.0, y, 70.0, 32.0), Qt.AlignmentFlag.AlignVCenter, f"n = {self._n}")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#e2e8f0"))
        state_name = "Cơ bản" if self._n == 1 else f"Kích thích {self._n-1}"
        painter.drawText(QRectF(ox + 80.0, y + 2.0, w - 85.0, 28.0), Qt.AlignmentFlag.AlignVCenter, f"{state_name} ({self._particle_name})\nL = {self._well_width_nm:.2f} nm")

        # Quantum Number n Steppers
        y += 38.0
        n_btns = []
        btn_w = (w - 16.0) / 5.0
        for i in range(1, 6):
            b_r = QRectF(ox + (i - 1) * (btn_w + 4.0), y, btn_w, 20.0)
            self._btn_rects[f"set_n_{i}"] = b_r
            is_sel = (self._n == i)
            painter.setBrush(QBrush(QColor("#0284c7") if is_sel else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_sel else QColor("#475569"), 1.0))
            painter.drawRoundedRect(b_r, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_sel else QColor("#94a3b8"))
            painter.drawText(b_r, Qt.AlignmentFlag.AlignCenter, f"n={i}")

        # Quantitative Measurements Readout
        y += 28.0
        e_ev = self.current_energy_ev
        e_j = self.current_energy_joules
        lam = self.de_broglie_wavelength_nm
        e1_ev = self.ground_state_energy_ev
        nodes = self.node_count
        heis = self.heisenberg_uncertainty_product

        rows = [
            (f"• Năng lượng E_{self._n}: {e_ev:.3f} eV", QColor("#10b981")),
            (f"• Năng lượng (J): {e_j*1e20:.2f} × 10⁻²⁰ J", QColor("#38bdf8")),
            (f"• Năng lượng gốc E₁: {e1_ev:.3f} eV", QColor("#fbbf24")),
            (f"• Bước sóng De Broglie: {lam:.2f} nm", QColor("#cbd5e1")),
            (f"• Số nút sóng bên trong: {nodes} nút", QColor("#f43f5e")),
            (f"• Bất định Δx·Δp: {heis:.2f} · (ℏ/2) ≥ 1", QColor("#10b981")),
            (f"• Tỷ lệ E_{self._n} / E₁: {self._n**2}× bậc hai", QColor("#94a3b8")),
        ]

        for txt, col in rows:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(col)
            painter.drawText(QPointF(ox, y), txt)
            y += 16.0

        # Action Buttons: Toggle Curve Display, Toggle Superposition
        y = rect.bottom() - 50.0
        disp_r = QRectF(ox, y, w / 2.0 - 4.0, 22.0)
        super_r = QRectF(ox + w / 2.0 + 4.0, y, w / 2.0 - 4.0, 22.0)
        self._btn_rects["toggle_disp"] = disp_r
        self._btn_rects["toggle_super"] = super_r

        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(disp_r, 4.0, 4.0)
        painter.drawRoundedRect(super_r, 4.0, 4.0)

        disp_names = ["Hiện cả hai", "Chỉ ψ(x)", "Chỉ |ψ|²"]
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(disp_r, Qt.AlignmentFlag.AlignCenter, disp_names[self._display_mode])
        painter.drawText(super_r, Qt.AlignmentFlag.AlignCenter, "🌊 Chồng chập" if not self._is_superposition else "Đơn mức")

        # Well width steppers
        y += 26.0
        w_m = QRectF(ox, y, 40.0, 18.0)
        w_p = QRectF(ox + 46.0, y, 40.0, 18.0)
        part_r = QRectF(ox + 96.0, y, w - 96.0, 18.0)
        self._btn_rects["w_m"] = w_m
        self._btn_rects["w_p"] = w_p
        self._btn_rects["toggle_particle"] = part_r

        for b_r, b_txt in [(w_m, "-0.2nm"), (w_p, "+0.2nm")]:
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(QPen(QColor("#475569"), 1.0))
            painter.drawRoundedRect(b_r, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(b_r, Qt.AlignmentFlag.AlignCenter, b_txt)

        painter.setBrush(QBrush(QColor("#0369a1")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(part_r, 3.0, 3.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(part_r, Qt.AlignmentFlag.AlignCenter, "Đổi e⁻ ⇋ p⁺")

    # -------------------------------------------------------------------------
    # Bottom Physics Theory Banner
    # -------------------------------------------------------------------------
    def _draw_theory_banner(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 10.0, rect.top() + 16.0), "NGUYÊN LÝ PHƯƠNG TRÌNH SCHRÖDINGER & LƯỢNG TỬ HÓA NĂNG LƯỢNG (1926):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        desc = (
            f"• Phương trình Schrödinger độc lập thời gian dẫn đến hàm sóng dừng dạng sin: ψ_n(x) = √(2/L)·sin(nπx/L).\n"
            f"• Mức năng lượng bị gián đoạn hóa E_n = n²·h²/(8mL²). Sự tồn tại của năng lượng điểm không E₁ > 0 là minh chứng trực tiếp cho hệ thức bất định Heisenberg Δx·Δp ≥ ℏ/2.\n"
            f"• Mô tả chế độ: {self._preset.description}"
        )
        painter.drawText(
            QRectF(rect.left() + 10.0, rect.top() + 22.0, rect.width() - 20.0, 38.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            desc,
        )

    # -------------------------------------------------------------------------
    # Touchless Gesture & Mouse Interaction
    # -------------------------------------------------------------------------
    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air gesture pinch handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset tabs
        for code in PRESETS:
            r = self._btn_rects.get(f"preset_{code}")
            if r and r.contains(local_pos):
                self.load_preset(code)
                return True

        # 2. Quantum number n buttons (1..5)
        for i in range(1, 6):
            r = self._btn_rects.get(f"set_n_{i}")
            if r and r.contains(local_pos):
                self.set_quantum_number(i)
                return True

        # 3. Display mode toggle
        if self._btn_rects.get("toggle_disp") and self._btn_rects["toggle_disp"].contains(local_pos):
            self.cycle_display_mode()
            return True

        # 4. Superposition toggle
        if self._btn_rects.get("toggle_super") and self._btn_rects["toggle_super"].contains(local_pos):
            self.toggle_superposition()
            return True

        # 5. Well width steppers
        if self._btn_rects.get("w_m") and self._btn_rects["w_m"].contains(local_pos):
            self.set_well_width_nm(self._well_width_nm - 0.20)
            return True
        if self._btn_rects.get("w_p") and self._btn_rects["w_p"].contains(local_pos):
            self.set_well_width_nm(self._well_width_nm + 0.20)
            return True

        # 6. Particle toggle
        if self._btn_rects.get("toggle_particle") and self._btn_rects["toggle_particle"].contains(local_pos):
            new_p = "proton" if self._particle_mass_kg == M_ELECTRON else "electron"
            self.set_particle_type(new_p)
            return True

        # 7. Stamp button
        if self._btn_rects.get("stamp") and self._btn_rects["stamp"].contains(local_pos):
            self.stamp_to_board()
            return True

        return False

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        loc = event.pos()
        if self.handle_air_pinch(self.mapToScene(loc)):
            event.accept()
            return
        super().mousePressEvent(event)
