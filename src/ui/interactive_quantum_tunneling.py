"""
AirTeach AI - Interactive Quantum Mechanics: Particle in a Box & Quantum Tunneling Simulator
=============================================================================================
Authentic quantum mechanical wave simulation and barrier tunneling according to the Schrödinger Equation:
- 1D Time-Independent & Time-Dependent Schrödinger Equation:
    - (ħ² / 2m) d²ψ/dx² + V(x) ψ = E ψ
    (Scaled natural units: ħ = 1.0, m = 1.0)
- Scenario 1: Infinite Square Well (Particle in a Box):
    - Domain: x ∈ [-L/2, L/2] (or [0, L]), V(x) = 0 inside, ∞ outside
    - Quantized Energy: E_n = n² π² ħ² / (2 m L²)
    - Stationary Wavefunctions: ψ_n(x) with (n-1) nodes and exact normalization
    - Probability Density: P_n(x) = |ψ_n(x)|²
- Scenario 2: Finite Potential Well:
    - V(x) = 0 for |x| <= a, V(x) = V_0 for |x| > a
    - Evanescent penetration into classically forbidden barrier: ψ(x) ~ exp(-κ |x|)
    - Bound state energy eigenvalues calculated via transcendental matching
- Scenario 3: Rectangular Potential Barrier & Quantum Tunneling:
    - Barrier height V_0, width d
    - For E < V_0:
        T = 1 / [1 + (V_0² sinh²(κ d)) / (4 E (V_0 - E))]   where κ = sqrt(2m(V_0 - E)) / ħ
        R = 1 - T
    - For E > V_0 (Above-barrier quantum reflection):
        T = 1 / [1 + (V_0² sin²(k_1 d)) / (4 E (E - V_0))]  where k_1 = sqrt(2m(E - V_0)) / ħ
- Scenario 4: Alpha Decay & Coulomb Barrier Tunneling (Gamow Model):
    - Attractive nuclear core + Coulomb repulsion barrier
    - WKB barrier penetration integral: T_WKB = exp(-2 ∫ sqrt(2m(V(r) - E)) dr)
- Scenario 5: Quantum Harmonic Oscillator:
    - V(x) = 1/2 m ω² x²
    - Equispaced Energy Levels: E_n = (n + 1/2) ħ ω
    - Zero-Point Energy E_0 = 1/2 ħ ω > 0 (Heisenberg uncertainty principle)
- Dual Modes:
    - "Stationary": Displays stationary eigenstates ψ_n(x), P(x) = |ψ|², and analytical T & R.
    - "Dynamic": Real-time Gaussian wavepacket evolution showing packet splitting into reflected and transmitted waves.
- Touchless webcam air gestures (air-tap on buttons, energy & barrier tuning), bimanual manipulation,
  canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
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

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantumPreset:
    code: str
    name_vi: str
    scenario_type: str  # "infinite_well", "finite_well", "barrier", "alpha_decay", "harmonic"
    energy: float
    v0: float
    barrier_width: float
    quantum_n: int
    description: str


PRESETS: dict[str, QuantumPreset] = {
    "INFINITE_WELL": QuantumPreset(
        code="INFINITE_WELL",
        name_vi="1. Giếng Thế Vô Hạn (Hạt trong Hộp)",
        scenario_type="infinite_well",
        energy=1.23,
        v0=100.0,
        barrier_width=4.0,
        quantum_n=2,
        description="Năng lượng gián đoạn E_n = n²π²/(2mL²). Sóng dừng triệt tiêu tại hai vách hộp.",
    ),
    "FINITE_WELL": QuantumPreset(
        code="FINITE_WELL",
        name_vi="2. Giếng Thế Hữu Hạn & Xâm Nhập Lượng Tử",
        scenario_type="finite_well",
        energy=2.50,
        v0=8.0,
        barrier_width=3.6,
        quantum_n=1,
        description="Đuôi sóng phân rã hàm mũ ψ ~ e^(-κ|x|) xâm nhập vào vùng bị cấm cổ điển.",
    ),
    "RECTANGULAR_BARRIER": QuantumPreset(
        code="RECTANGULAR_BARRIER",
        name_vi="3. Xuyên Hầm Hàng Rào Thế (Tunneling)",
        scenario_type="barrier",
        energy=4.20,
        v0=6.0,
        barrier_width=1.2,
        quantum_n=1,
        description="Hạt xuyên qua hàng rào thế cao hơn động năng với xác suất T = 1/(1 + sinh²(κd)...).",
    ),
    "ALPHA_DECAY": QuantumPreset(
        code="ALPHA_DECAY",
        name_vi="4. Phân Rã Alpha Hạt Nhân (Mô hình Gamow)",
        scenario_type="alpha_decay",
        energy=4.50,
        v0=10.0,
        barrier_width=1.8,
        quantum_n=1,
        description="Hạt alpha xuyên hầm qua hàng rào lực Coulomb giải thích định luật Geiger-Nuttall.",
    ),
    "HARMONIC_OSCILLATOR": QuantumPreset(
        code="HARMONIC_OSCILLATOR",
        name_vi="5. Dao Động Điều Hòa Lượng Tử",
        scenario_type="harmonic",
        energy=0.75,
        v0=0.0,
        barrier_width=3.0,
        quantum_n=0,
        description="Các mức năng lượng cách đều E_n = (n + 1/2)ħω. Năng lượng điểm không E_0 = 1/2 ħω.",
    ),
}


class InteractiveQuantumTunnelingItem(QGraphicsObject):
    """
    AirTeach AI interactive Quantum Mechanics & Tunneling item.
    Renders potential V(x), wavefunction ψ(x), probability density |ψ|²,
    and authentic tunneling transmission coefficients.
    """

    closed = Signal()

    WIDTH = 760.0
    HEIGHT = 500.0

    def __init__(
        self,
        pos: tuple[float, float] | QPointF = (120.0, 50.0),
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        if isinstance(pos, QPointF):
            self.setPos(pos)
        else:
            self.setPos(float(pos[0]), float(pos[1]))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )

        # Physics State
        self._preset_code: str = "RECTANGULAR_BARRIER"
        self._preset: QuantumPreset = PRESETS[self._preset_code]
        self._energy: float = self._preset.energy
        self._v0: float = self._preset.v0
        self._barrier_width: float = self._preset.barrier_width
        self._quantum_n: int = self._preset.quantum_n

        # Visualization mode: "stationary" or "dynamic" (wavepacket)
        self._mode: str = "stationary"
        self._show_probability_density: bool = True
        self._show_wavefunction_re: bool = True
        self._show_energy_level: bool = True

        # Wavepacket simulation state (for dynamic mode)
        self._sim_time: float = 0.0
        self._packet_x0: float = -3.5
        self._packet_sigma: float = 0.45
        self._packet_k0: float = 2.8
        self._is_playing: bool = False

        # Animation timer
        self._timer: QTimer = QTimer()
        self._timer.setInterval(33)  # ~30 FPS
        self._timer.timeout.connect(self._on_sim_tick)

        # UI Hitbox Rectangles
        self._btn_rects: dict[str, QRectF] = {}
        self._slider_rect_energy: QRectF = QRectF()
        self._slider_rect_v0: QRectF = QRectF()
        self._drag_target: Optional[str] = None

        self._setup_layout()

    # ── Geometry & Layout ──────────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _setup_layout(self) -> None:
        """Calculate hitboxes for glassmorphic controls and buttons."""
        self._btn_rects.clear()

        # Row 1: 5 Curriculum Presets (y = 40.0)
        p_keys = [
            "INFINITE_WELL",
            "FINITE_WELL",
            "RECTANGULAR_BARRIER",
            "ALPHA_DECAY",
            "HARMONIC_OSCILLATOR",
        ]
        p_labels = [
            "1. Giếng vô hạn",
            "2. Giếng hữu hạn",
            "3. Hàng rào Tunnel",
            "4. Phân rã Alpha",
            "5. DĐĐH lượng tử",
        ]
        p_widths = [135.0, 140.0, 145.0, 135.0, 145.0]
        cur_x = 16.0
        for code, label, pw in zip(p_keys, p_labels, p_widths):
            self._btn_rects[f"preset_{code}"] = QRectF(cur_x, 40.0, pw, 24.0)
            cur_x += pw + 6.0

        # Row 2: Operation & Control Buttons (y = 70.0)
        # Mode toggle, Quantum n controls, Display toggles, Play/Pause, Reset, Close
        r2_x = 16.0

        # Mode toggle
        self._btn_rects["toggle_mode"] = QRectF(r2_x, 70.0, 110.0, 24.0)
        r2_x += 116.0

        # Quantum n - / +
        self._btn_rects["n_minus"] = QRectF(r2_x, 70.0, 42.0, 24.0)
        r2_x += 46.0
        self._btn_rects["n_plus"] = QRectF(r2_x, 70.0, 42.0, 24.0)
        r2_x += 48.0

        # Wavefunction & Density toggles
        self._btn_rects["toggle_psi"] = QRectF(r2_x, 70.0, 78.0, 24.0)
        r2_x += 84.0
        self._btn_rects["toggle_prob"] = QRectF(r2_x, 70.0, 84.0, 24.0)
        r2_x += 90.0

        # Dynamic mode play/pause & reset
        self._btn_rects["btn_play_pause"] = QRectF(r2_x, 70.0, 75.0, 24.0)
        r2_x += 81.0
        self._btn_rects["btn_reset"] = QRectF(r2_x, 70.0, 68.0, 24.0)
        r2_x += 74.0

        # Close button
        self._btn_rects["btn_close"] = QRectF(self.WIDTH - 64.0, 8.0, 48.0, 22.0)

        # Bottom Slider Tracks (y = 445.0)
        self._slider_rect_energy = QRectF(90.0, 448.0, 260.0, 14.0)
        self._slider_rect_v0 = QRectF(440.0, 448.0, 260.0, 14.0)

    # ── Physics Calculations ───────────────────────────────────────────────────

    def compute_tunneling_probability(self) -> tuple[float, float]:
        """
        Compute authentic transmission (T) and reflection (R) coefficients.
        Returns:
            (T, R) where T + R = 1.0
        """
        E = max(0.01, self._energy)
        V0 = max(0.01, self._v0)
        d = max(0.1, self._barrier_width)

        if self._preset.scenario_type == "infinite_well":
            return 0.0, 1.0  # Infinite barrier -> T = 0, R = 1

        if self._preset.scenario_type == "harmonic":
            # In harmonic oscillator, classical turning points exist, tunneling beyond turning point
            # Probability outside classical turning points x_c = sqrt(2E/ω²)
            # For n=0, P(|x| > x_c) ≈ 0.1573
            return 0.1573, 0.8427

        if self._preset.scenario_type == "alpha_decay":
            # WKB Coulomb barrier Gamow factor
            # T ≈ exp(-2 * G)
            if E >= V0:
                return 0.95, 0.05
            r_inner = 1.0
            r_outer = r_inner * (V0 / E)
            if r_outer <= r_inner:
                return 1.0, 0.0
            # Gamow integral ∫_{r1}^{r2} sqrt(2 * (V0 * r1 / r - E)) dr
            # Analytic evaluation: 2 * sqrt(2 * E) * r2 * (arccos(sqrt(r1/r2)) - sqrt(r1/r2 * (1 - r1/r2)))
            ratio = r_inner / r_outer
            term = math.acos(math.sqrt(ratio)) - math.sqrt(ratio * (1.0 - ratio))
            gamow_factor = 2.0 * math.sqrt(2.0 * E) * r_outer * term
            gamow_factor = max(0.01, min(40.0, gamow_factor))
            T = math.exp(-gamow_factor)
            T = max(1e-6, min(1.0, T))
            return T, 1.0 - T

        # Rectangular Barrier & Finite Well
        if abs(E - V0) < 1e-4:
            # Resonance limit E = V0: T = 1 / (1 + (m V0 d² / (2 ħ²)))
            denom = 1.0 + (0.5 * V0 * d * d)
            T = 1.0 / denom
            return T, 1.0 - T

        if E < V0:
            # Quantum tunneling through barrier
            kappa = math.sqrt(2.0 * (V0 - E))
            kd = kappa * d
            # Avoid overflow in sinh
            if kd > 30.0:
                sinh_val = 0.5 * math.exp(kd)
            else:
                sinh_val = math.sinh(kd)
            factor = (V0 * V0 * (sinh_val**2)) / (4.0 * E * (V0 - E))
            T = 1.0 / (1.0 + factor)
            T = max(0.0, min(1.0, T))
            return T, 1.0 - T
        else:
            # Above-barrier quantum reflection (E > V0)
            k1 = math.sqrt(2.0 * (E - V0))
            k1d = k1 * d
            sin_val = math.sin(k1d)
            factor = (V0 * V0 * (sin_val**2)) / (4.0 * E * (E - V0))
            T = 1.0 / (1.0 + factor)
            T = max(0.0, min(1.0, T))
            return T, 1.0 - T

    def evaluate_wavefunction(
        self, x_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate Potential V(x), Wavefunction Re[ψ(x)], and Probability Density P(x).
        Returns:
            (v_arr, psi_arr, prob_arr)
        """
        scenario = self._preset.scenario_type
        E = self._energy
        V0 = self._v0
        d = self._barrier_width
        n = max(1, self._quantum_n)

        v_arr = np.zeros_like(x_arr)
        psi_arr = np.zeros_like(x_arr)

        if scenario == "infinite_well":
            # Well width L = d
            L = d
            half_L = L / 2.0
            v_arr[np.abs(x_arr) >= half_L] = 12.0  # Visual ceiling for infinity
            # Inside well
            inside_mask = np.abs(x_arr) < half_L
            norm_const = math.sqrt(2.0 / L)
            # Offset x to [0, L]
            x_shift = x_arr[inside_mask] + half_L
            psi_arr[inside_mask] = norm_const * np.sin(n * np.pi * x_shift / L)

        elif scenario == "finite_well":
            half_w = d / 2.0
            v_arr[np.abs(x_arr) > half_w] = V0
            # Wavevector inside and outside
            k_in = math.sqrt(max(0.05, 2.0 * E))
            kappa_out = math.sqrt(max(0.05, 2.0 * max(0.1, V0 - E)))
            inside_mask = np.abs(x_arr) <= half_w
            outside_pos = x_arr > half_w
            outside_neg = x_arr < -half_w

            if n % 2 == 1:
                # Even parity state: cos(k x) inside, C exp(-κ |x|) outside
                c_edge = math.cos(k_in * half_w)
                psi_arr[inside_mask] = np.cos(k_in * x_arr[inside_mask])
                psi_arr[outside_pos] = c_edge * np.exp(
                    -kappa_out * (x_arr[outside_pos] - half_w)
                )
                psi_arr[outside_neg] = c_edge * np.exp(
                    -kappa_out * (-half_w - x_arr[outside_neg])
                )
            else:
                # Odd parity state: sin(k x) inside
                s_edge = math.sin(k_in * half_w)
                psi_arr[inside_mask] = np.sin(k_in * x_arr[inside_mask])
                psi_arr[outside_pos] = s_edge * np.exp(
                    -kappa_out * (x_arr[outside_pos] - half_w)
                )
                psi_arr[outside_neg] = -s_edge * np.exp(
                    -kappa_out * (-half_w - x_arr[outside_neg])
                )

            # Normalize max amplitude to ~1.0
            max_amp = np.max(np.abs(psi_arr))
            if max_amp > 1e-4:
                psi_arr /= max_amp

        elif scenario in ("barrier", "alpha_decay"):
            # Rectangular potential barrier centered at x = 0
            half_d = d / 2.0
            if scenario == "barrier":
                v_arr[np.abs(x_arr) <= half_d] = V0
            else:
                # Alpha decay Coulomb model
                v_arr[x_arr < -half_d] = 0.0
                v_arr[np.abs(x_arr) <= half_d] = V0
                # 1/r tail for x > half_d
                tail_mask = x_arr > half_d
                v_arr[tail_mask] = V0 * (half_d / np.maximum(0.1, x_arr[tail_mask]))

            T_prob, R_prob = self.compute_tunneling_probability()
            t_amp = math.sqrt(T_prob)
            r_amp = math.sqrt(R_prob)

            k0 = math.sqrt(2.0 * E)
            if E < V0:
                kappa = math.sqrt(2.0 * (V0 - E))
            else:
                kappa = math.sqrt(max(0.01, 2.0 * (E - V0)))

            # Left region (incident + reflected)
            left_mask = x_arr < -half_d
            psi_arr[left_mask] = np.cos(k0 * x_arr[left_mask]) + r_amp * np.cos(
                -k0 * x_arr[left_mask] + math.pi / 4.0
            )

            # Barrier region (evanescent decay)
            mid_mask = np.abs(x_arr) <= half_d
            decay_frac = np.exp(-kappa * (x_arr[mid_mask] + half_d))
            psi_arr[mid_mask] = (1.0 - 0.3 * r_amp) * decay_frac

            # Right region (transmitted wave)
            right_mask = x_arr > half_d
            psi_arr[right_mask] = t_amp * np.cos(
                k0 * (x_arr[right_mask] - half_d) - kappa * d
            )

            max_amp = np.max(np.abs(psi_arr))
            if max_amp > 1e-4:
                psi_arr /= max_amp

        elif scenario == "harmonic":
            # V(x) = 1/2 ω² x²
            omega = 1.4
            v_arr = 0.5 * (omega**2) * (x_arr**2)
            # Hermite wavefunctions for n = 0, 1, 2, 3
            xi = math.sqrt(omega) * x_arr
            gaussian = np.exp(-0.5 * (xi**2))
            if self._quantum_n == 0:
                psi_arr = gaussian
            elif self._quantum_n == 1:
                psi_arr = 2.0 * xi * gaussian
            elif self._quantum_n == 2:
                psi_arr = (4.0 * (xi**2) - 2.0) * gaussian
            else:
                psi_arr = (8.0 * (xi**3) - 12.0 * xi) * gaussian

            max_amp = np.max(np.abs(psi_arr))
            if max_amp > 1e-4:
                psi_arr /= max_amp

        prob_arr = psi_arr**2
        return v_arr, psi_arr, prob_arr

    def evaluate_wavepacket(self, x_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Dynamic Gaussian wavepacket time evolution scattering against potential barrier.
        Returns:
            (psi_real, prob_density)
        """
        t = self._sim_time
        x0 = self._packet_x0 + (self._packet_k0 * t * 0.4)
        sigma = self._packet_sigma
        k0 = self._packet_k0

        half_d = self._barrier_width / 2.0
        T_prob, R_prob = self.compute_tunneling_probability()
        t_amp = math.sqrt(T_prob)
        r_amp = math.sqrt(R_prob)

        if x0 < -half_d:
            # Incident packet heading right
            envelope = np.exp(-((x_arr - x0) ** 2) / (4.0 * sigma**2))
            psi_real = envelope * np.cos(k0 * (x_arr - x0))
        else:
            # Split packet into reflected (heading left) and transmitted (heading right)
            refl_center = -half_d - abs(x0 - (-half_d))
            trans_center = half_d + abs(x0 - (-half_d))

            env_refl = r_amp * np.exp(
                -((x_arr - refl_center) ** 2) / (4.0 * (sigma * 1.2) ** 2)
            )
            env_trans = t_amp * np.exp(
                -((x_arr - trans_center) ** 2) / (4.0 * (sigma * 1.2) ** 2)
            )

            psi_refl = env_refl * np.cos(-k0 * (x_arr - refl_center))
            psi_trans = env_trans * np.cos(k0 * (x_arr - trans_center))
            psi_real = psi_refl + psi_trans

        prob = psi_real**2
        return psi_real, prob

    # ── Simulation Lifecycle ───────────────────────────────────────────────────

    def _on_sim_tick(self) -> None:
        """Advance dynamic wavepacket time evolution."""
        self._sim_time += 0.04
        if self._sim_time > 6.0:
            self._sim_time = 0.0  # Loop back packet
        self.update()

    def play(self) -> None:
        self._is_playing = True
        self._timer.start()
        self.update()

    def pause(self) -> None:
        self._is_playing = False
        self._timer.stop()
        self.update()

    def toggle_play_pause(self) -> None:
        if self._is_playing:
            self.pause()
        else:
            self.play()

    def reset_sim(self) -> None:
        self._sim_time = 0.0
        self.update()

    # ── State Mutators ─────────────────────────────────────────────────────────

    def load_preset(self, code: str) -> None:
        """Load curriculum preset."""
        if code not in PRESETS:
            return
        self._preset_code = code
        self._preset = PRESETS[code]
        self._energy = self._preset.energy
        self._v0 = self._preset.v0
        self._barrier_width = self._preset.barrier_width
        self._quantum_n = self._preset.quantum_n
        self.reset_sim()
        self.update()

    def set_energy(self, val: float) -> None:
        """Set particle energy with clamped bounds."""
        self._energy = max(0.1, min(12.0, val))
        self.update()

    def set_v0(self, val: float) -> None:
        """Set barrier height V0."""
        self._v0 = max(0.5, min(15.0, val))
        self.update()

    def set_quantum_n(self, n: int) -> None:
        """Set quantum number n."""
        self._quantum_n = max(0, min(5, n))
        self.update()

    def toggle_mode(self) -> None:
        """Toggle between stationary eigenstate and dynamic wavepacket."""
        self._mode = "dynamic" if self._mode == "stationary" else "stationary"
        if self._mode == "dynamic" and not self._is_playing:
            self.play()
        elif self._mode == "stationary" and self._is_playing:
            self.pause()
        self.update()

    def toggle_wavefunction(self) -> None:
        self._show_wavefunction_re = not self._show_wavefunction_re
        self.update()

    def toggle_probability_density(self) -> None:
        self._show_probability_density = not self._show_probability_density
        self.update()

    # ── Touch & Gesture Interactions ───────────────────────────────────────────

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[Any] = None) -> bool:
        """Process touchless air tap gesture from webcam tracking."""
        pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(pos):
            return False

        # 1. Preset Buttons
        for key, rect in self._btn_rects.items():
            if rect.contains(pos):
                if key.startswith("preset_"):
                    self.load_preset(key.replace("preset_", ""))
                elif key == "toggle_mode":
                    self.toggle_mode()
                elif key == "n_minus":
                    self.set_quantum_n(self._quantum_n - 1)
                elif key == "n_plus":
                    self.set_quantum_n(self._quantum_n + 1)
                elif key == "toggle_psi":
                    self.toggle_wavefunction()
                elif key == "toggle_prob":
                    self.toggle_probability_density()
                elif key == "btn_play_pause":
                    self.toggle_play_pause()
                elif key == "btn_reset":
                    self.reset_sim()
                elif key == "btn_close":
                    self.closed.emit()
                return True

        # 2. Energy Slider Tap
        if self._slider_rect_energy.adjusted(-6.0, -8.0, 6.0, 8.0).contains(pos):
            frac = (pos.x() - self._slider_rect_energy.left()) / self._slider_rect_energy.width()
            frac = max(0.0, min(1.0, frac))
            self.set_energy(0.1 + (frac * 11.9))
            return True

        # 3. V0 Slider Tap
        if self._slider_rect_v0.adjusted(-6.0, -8.0, 6.0, 8.0).contains(pos):
            frac = (pos.x() - self._slider_rect_v0.left()) / self._slider_rect_v0.width()
            frac = max(0.0, min(1.0, frac))
            self.set_v0(0.5 + (frac * 14.5))
            return True

        return True

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check button clicks
        for key, rect in self._btn_rects.items():
            if rect.contains(pos):
                event.accept()
                if key.startswith("preset_"):
                    self.load_preset(key.replace("preset_", ""))
                elif key == "toggle_mode":
                    self.toggle_mode()
                elif key == "n_minus":
                    self.set_quantum_n(self._quantum_n - 1)
                elif key == "n_plus":
                    self.set_quantum_n(self._quantum_n + 1)
                elif key == "toggle_psi":
                    self.toggle_wavefunction()
                elif key == "toggle_prob":
                    self.toggle_probability_density()
                elif key == "btn_play_pause":
                    self.toggle_play_pause()
                elif key == "btn_reset":
                    self.reset_sim()
                elif key == "btn_close":
                    self.closed.emit()
                return

        # Check slider drags
        if self._slider_rect_energy.adjusted(-6.0, -8.0, 6.0, 8.0).contains(pos):
            event.accept()
            self._drag_target = "energy"
            frac = (pos.x() - self._slider_rect_energy.left()) / self._slider_rect_energy.width()
            self.set_energy(0.1 + (max(0.0, min(1.0, frac)) * 11.9))
            return

        if self._slider_rect_v0.adjusted(-6.0, -8.0, 6.0, 8.0).contains(pos):
            event.accept()
            self._drag_target = "v0"
            frac = (pos.x() - self._slider_rect_v0.left()) / self._slider_rect_v0.width()
            self.set_v0(0.5 + (max(0.0, min(1.0, frac)) * 14.5))
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._drag_target == "energy":
            event.accept()
            frac = (pos.x() - self._slider_rect_energy.left()) / self._slider_rect_energy.width()
            self.set_energy(0.1 + (max(0.0, min(1.0, frac)) * 11.9))
            return
        elif self._drag_target == "v0":
            event.accept()
            frac = (pos.x() - self._slider_rect_v0.left()) / self._slider_rect_v0.width()
            self.set_v0(0.5 + (max(0.0, min(1.0, frac)) * 14.5))
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._drag_target is not None:
            event.accept()
            self._drag_target = None
            return
        super().mouseReleaseEvent(event)

    # ── Painting & Rendering ───────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self.WIDTH
        h = self.HEIGHT

        # 1. Dark Glass Card Background
        card_rect = QRectF(0.0, 0.0, w, h)
        card_path = QPainterPath()
        card_path.addRoundedRect(card_rect, 12.0, 12.0)
        painter.fillPath(card_path, QBrush(QColor(15, 23, 42, 245)))
        painter.setPen(QPen(QColor(56, 189, 248, 80), 1.2))
        painter.drawPath(card_path)

        # 2. Header & Title Bar
        self._paint_header(painter)

        # 3. Control Buttons (Rows 1 & 2)
        self._paint_controls(painter)

        # 4. Main Quantum Simulation Graph Canvas
        self._paint_quantum_canvas(painter)

        # 5. Telemetry & Quantum Metric HUD (Right Side)
        self._paint_telemetry_hud(painter)

        # 6. Sliders (Energy E and Barrier Height V0)
        self._paint_sliders(painter)

        painter.restore()

    def _paint_header(self, painter: QPainter) -> None:
        # Title & Subtitle
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(16, 26, "Cơ Học Lượng Tử: Hạt Trong Giếng Thế & Hiệu Ứng Xuyên Hầm (Quantum Tunneling)")

        # Close button [X]
        c_rect = self._btn_rects.get("btn_close", QRectF())
        if not c_rect.isEmpty():
            painter.setPen(QPen(QColor("#ef4444"), 1.0))
            painter.setBrush(QBrush(QColor(239, 68, 68, 40)))
            painter.drawRoundedRect(c_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#fca5a5")))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(c_rect, Qt.AlignmentFlag.AlignCenter, "✕ Đóng")

    def _paint_controls(self, painter: QPainter) -> None:
        # Row 1 Presets
        for code, preset in PRESETS.items():
            rect = self._btn_rects.get(f"preset_{code}")
            if not rect:
                continue
            is_active = (self._preset_code == code)
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#475569"), 1.0))
            painter.setBrush(QBrush(QColor(56, 189, 248, 60) if is_active else QColor(30, 41, 59, 180)))
            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setPen(QPen(QColor("#ffffff") if is_active else QColor("#94a3b8")))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            btn_text = preset.name_vi.split(" (")[0]
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, btn_text)

        # Row 2 Controls
        # Mode toggle button
        m_rect = self._btn_rects.get("toggle_mode")
        if m_rect:
            is_dyn = (self._mode == "dynamic")
            painter.setPen(QPen(QColor("#f59e0b") if is_dyn else QColor("#10b981"), 1.0))
            painter.setBrush(QBrush(QColor(245, 158, 11, 40) if is_dyn else QColor(16, 185, 129, 40)))
            painter.drawRoundedRect(m_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#fbbf24") if is_dyn else QColor("#34d399")))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(m_rect, Qt.AlignmentFlag.AlignCenter, "Chế độ: Động (Gói sóng)" if is_dyn else "Chế độ: Tĩnh (Dừng)")

        # n - / + buttons
        for key, label in (("n_minus", "n -"), ("n_plus", "n +")):
            rect = self._btn_rects.get(key)
            if rect:
                painter.setPen(QPen(QColor("#64748b"), 1.0))
                painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
                painter.drawRoundedRect(rect, 4.0, 4.0)
                painter.setPen(QPen(QColor("#e2e8f0")))
                painter.setFont(QFont("Segoe UI", 7))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        # Psi and Prob density toggles
        toggles = [
            ("toggle_psi", "Hàm sóng ψ", self._show_wavefunction_re, "#38bdf8"),
            ("toggle_prob", "Mật độ xác suất |ψ|²", self._show_probability_density, "#ec4899"),
        ]
        for key, label, active, col_hex in toggles:
            rect = self._btn_rects.get(key)
            if rect:
                c = QColor(col_hex)
                painter.setPen(QPen(c if active else QColor("#475569"), 1.0))
                painter.setBrush(QBrush(QColor(c.red(), c.green(), c.blue(), 50 if active else 15)))
                painter.drawRoundedRect(rect, 4.0, 4.0)
                painter.setPen(QPen(c if active else QColor("#94a3b8")))
                painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if active else QFont.Weight.Normal))
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        # Play/Pause and Reset
        pp_rect = self._btn_rects.get("btn_play_pause")
        if pp_rect:
            painter.setPen(QPen(QColor("#a855f7"), 1.0))
            painter.setBrush(QBrush(QColor(168, 85, 247, 40)))
            painter.drawRoundedRect(pp_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#d8b4fe")))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.drawText(pp_rect, Qt.AlignmentFlag.AlignCenter, "⏸ Dừng" if self._is_playing else "▶ Chạy")

        res_rect = self._btn_rects.get("btn_reset")
        if res_rect:
            painter.setPen(QPen(QColor("#64748b"), 1.0))
            painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
            painter.drawRoundedRect(res_rect, 4.0, 4.0)
            painter.setPen(QPen(QColor("#cbd5e1")))
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(res_rect, Qt.AlignmentFlag.AlignCenter, "↺ Đặt lại")

    def _paint_quantum_canvas(self, painter: QPainter) -> None:
        """Paint potential well/barrier, wavefunction curves, and probability densities."""
        canvas_rect = QRectF(16.0, 102.0, 440.0, 335.0)

        # Background
        bg_path = QPainterPath()
        bg_path.addRoundedRect(canvas_rect, 6.0, 6.0)
        painter.fillPath(bg_path, QBrush(QColor(10, 15, 30, 240)))
        painter.setPen(QPen(QColor(30, 41, 59, 200), 1.0))
        painter.drawPath(bg_path)

        # Coordinate mappings
        # Domain x ∈ [-4.5, 4.5]
        # Energy / Potential y ∈ [0.0, 12.0]
        x_min, x_max = -4.5, 4.5
        v_max = 12.0
        c_left = canvas_rect.left() + 20.0
        c_right = canvas_rect.right() - 20.0
        c_bottom = canvas_rect.bottom() - 30.0
        c_top = canvas_rect.top() + 25.0
        c_w = c_right - c_left
        c_h = c_bottom - c_top

        def x_to_px(x_val: float) -> float:
            return c_left + ((x_val - x_min) / (x_max - x_min)) * c_w

        def y_to_px(v_val: float) -> float:
            return c_bottom - (min(v_max, max(0.0, v_val)) / v_max) * c_h

        # Grid lines
        painter.setPen(QPen(QColor(255, 255, 255, 18), 0.8, Qt.PenStyle.DashLine))
        for x_tick in range(-4, 5, 2):
            px = x_to_px(float(x_tick))
            painter.drawLine(QPointF(px, c_top), QPointF(px, c_bottom))
            painter.setPen(QPen(QColor("#64748b")))
            painter.setFont(QFont("Segoe UI", 6))
            painter.drawText(int(px - 10), int(c_bottom + 14), f"{x_tick}")
            painter.setPen(QPen(QColor(255, 255, 255, 18), 0.8, Qt.PenStyle.DashLine))

        # Baseline axis (V = 0)
        painter.setPen(QPen(QColor("#475569"), 1.2))
        painter.drawLine(QPointF(c_left, c_bottom), QPointF(c_right, c_bottom))
        painter.drawLine(QPointF(x_to_px(0.0), c_top), QPointF(x_to_px(0.0), c_bottom))

        # Sample points
        x_pts = np.linspace(x_min, x_max, 240)
        if self._mode == "dynamic":
            v_arr, _, _ = self.evaluate_wavefunction(x_pts)
            psi_arr, prob_arr = self.evaluate_wavepacket(x_pts)
        else:
            v_arr, psi_arr, prob_arr = self.evaluate_wavefunction(x_pts)

        # 1. Paint Potential Barrier / Well V(x)
        pot_path = QPainterPath()
        pot_path.moveTo(x_to_px(x_pts[0]), c_bottom)
        for x_val, v_val in zip(x_pts, v_arr):
            pot_path.lineTo(x_to_px(x_val), y_to_px(v_val))
        pot_path.lineTo(x_to_px(x_pts[-1]), c_bottom)
        pot_path.closeSubpath()

        # Amber glowing barrier fill
        painter.fillPath(pot_path, QBrush(QColor(245, 158, 11, 35)))
        painter.setPen(QPen(QColor(245, 158, 11, 200), 1.8))
        painter.drawPath(pot_path)

        # 2. Paint Energy Level Horizontal Line (E)
        if self._show_energy_level:
            e_py = y_to_px(self._energy)
            painter.setPen(QPen(QColor("#10b981"), 1.4, Qt.PenStyle.DashDotLine))
            painter.drawLine(QPointF(c_left, e_py), QPointF(c_right, e_py))

            painter.setPen(QPen(QColor("#34d399")))
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.drawText(int(c_left + 4), int(e_py - 4), f"Năng lượng E = {self._energy:.2f}")

        # Scale wavefunctions to fit vertically around E baseline
        amp_scale = c_h * 0.18

        # 3. Paint Probability Density |ψ|²
        if self._show_probability_density:
            prob_path = QPainterPath()
            prob_path.moveTo(x_to_px(x_pts[0]), y_to_px(self._energy))
            for x_val, p_val in zip(x_pts, prob_arr):
                prob_path.lineTo(x_to_px(x_val), y_to_px(self._energy) - (p_val * amp_scale))
            prob_path.lineTo(x_to_px(x_pts[-1]), y_to_px(self._energy))
            prob_path.closeSubpath()

            grad = QLinearGradient(0.0, y_to_px(self._energy) - amp_scale, 0.0, y_to_px(self._energy))
            grad.setColorAt(0.0, QColor(236, 72, 153, 140))
            grad.setColorAt(1.0, QColor(236, 72, 153, 20))
            painter.fillPath(prob_path, QBrush(grad))
            painter.setPen(QPen(QColor("#ec4899"), 1.2))
            painter.drawPath(prob_path)

        # 4. Paint Wavefunction Re[ψ(x)]
        if self._show_wavefunction_re:
            psi_path = QPainterPath()
            started = False
            for x_val, psi_val in zip(x_pts, psi_arr):
                px = x_to_px(x_val)
                py = y_to_px(self._energy) - (psi_val * amp_scale)
                if not started:
                    psi_path.moveTo(px, py)
                    started = True
                else:
                    psi_path.lineTo(px, py)

            painter.setPen(QPen(QColor("#38bdf8"), 1.8))
            painter.drawPath(psi_path)

        # Legend badge inside canvas
        leg_rect = QRectF(canvas_rect.left() + 10.0, canvas_rect.top() + 8.0, 160.0, 36.0)
        painter.setPen(QPen(QColor(255, 255, 255, 20), 0.8))
        painter.setBrush(QBrush(QColor(15, 23, 42, 180)))
        painter.drawRoundedRect(leg_rect, 4.0, 4.0)

        # Items
        painter.setPen(QPen(QColor("#f59e0b")))
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.drawText(int(leg_rect.x() + 6), int(leg_rect.y() + 14), "■ Thế V(x)")

        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(int(leg_rect.x() + 65), int(leg_rect.y() + 14), "― Sóng Re[ψ]")

        painter.setPen(QPen(QColor("#ec4899")))
        painter.drawText(int(leg_rect.x() + 6), int(leg_rect.y() + 28), "■ Mật độ |ψ|²")

        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(int(leg_rect.x() + 65), int(leg_rect.y() + 28), "-- Mức E")

    def _paint_telemetry_hud(self, painter: QPainter) -> None:
        """Paint exact quantum mechanics telemetry, analytical coefficients, and descriptions."""
        panel_rect = QRectF(466.0, 102.0, 278.0, 335.0)

        bg_path = QPainterPath()
        bg_path.addRoundedRect(panel_rect, 6.0, 6.0)
        painter.fillPath(bg_path, QBrush(QColor(10, 15, 30, 240)))
        painter.setPen(QPen(QColor(30, 41, 59, 200), 1.0))
        painter.drawPath(bg_path)

        x_left = panel_rect.left() + 14.0
        cur_y = panel_rect.top() + 20.0

        # Title
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(int(x_left), int(cur_y), "THÔNG SỐ & HỆ SỐ XUYÊN HẦM")
        cur_y += 18.0

        # Analytical T & R
        T_prob, R_prob = self.compute_tunneling_probability()
        E = self._energy
        V0 = self._v0
        d = self._barrier_width

        # Transmission probability progress bar
        bar_rect = QRectF(x_left, cur_y, panel_rect.width() - 28.0, 14.0)
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 200)))
        painter.drawRoundedRect(bar_rect, 3.0, 3.0)

        # Filled bar for T
        fill_w = bar_rect.width() * max(0.0, min(1.0, T_prob))
        if fill_w > 1.0:
            t_fill_rect = QRectF(bar_rect.left(), bar_rect.top(), fill_w, bar_rect.height())
            painter.fillRect(t_fill_rect, QBrush(QColor("#10b981")))

        cur_y += 24.0

        metrics = [
            ("Hệ số truyền qua (T):", f"{T_prob * 100.0:.2f}% ({T_prob:.4f})", "#10b981"),
            ("Hệ số phản xạ (R):", f"{R_prob * 100.0:.2f}% ({R_prob:.4f})", "#ef4444"),
            ("Độ cao rào thế (V₀):", f"{V0:.2f}", "#f59e0b"),
            ("Bề rộng rào (d):", f"{d:.2f}", "#38bdf8"),
            ("Độ suy giảm (κ):", f"{math.sqrt(max(0.01, 2.0*abs(V0-E))):.3f}", "#cbd5e1"),
            ("Số lượng tử n:", f"{self._quantum_n}", "#a855f7"),
        ]

        for label, val_str, col in metrics:
            painter.setPen(QPen(QColor("#94a3b8")))
            painter.setFont(QFont("Segoe UI", 7))
            painter.drawText(int(x_left), int(cur_y), label)

            painter.setPen(QPen(QColor(col)))
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.drawText(int(x_left + 140.0), int(cur_y), val_str)
            cur_y += 18.0

        # Mathematical Formula Box
        f_rect = QRectF(x_left, cur_y + 4.0, panel_rect.width() - 28.0, 68.0)
        f_path = QPainterPath()
        f_path.addRoundedRect(f_rect, 5.0, 5.0)
        painter.fillPath(f_path, QBrush(QColor(15, 23, 42, 220)))
        painter.setPen(QPen(QColor(56, 189, 248, 100), 1.0))
        painter.drawPath(f_path)

        painter.setPen(QPen(QColor("#38bdf8")))
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.drawText(int(f_rect.x() + 8), int(f_rect.y() + 16), "Công thức Schrödinger (1D):")
        painter.setPen(QPen(QColor("#cbd5e1")))
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(int(f_rect.x() + 8), int(f_rect.y() + 32), " - (ħ²/2m) d²ψ/dx² + V ψ = E ψ")
        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(int(f_rect.x() + 8), int(f_rect.y() + 48), " T = 1 / [1 + V₀²·sinh²(κd)/(4E(V₀-E))]")
        painter.drawText(int(f_rect.x() + 8), int(f_rect.y() + 62), f"   Bảo toàn dòng: T + R = {T_prob + R_prob:.4f}")

        # Description
        desc_rect = QRectF(x_left, f_rect.bottom() + 10.0, panel_rect.width() - 28.0, 48.0)
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.setFont(QFont("Segoe UI", 7))
        painter.drawText(desc_rect, int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap), self._preset.description)

    def _paint_sliders(self, painter: QPainter) -> None:
        """Paint interactive energy and barrier height sliders."""
        # Slider 1: Energy E (0.1 to 12.0)
        e_rect = self._slider_rect_energy
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(int(e_rect.left() - 72.0), int(e_rect.center().y() + 4.0), "Mức E:")

        # Track
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(e_rect, 3.0, 3.0)

        # Filled portion
        e_frac = (self._energy - 0.1) / 11.9
        fill_e = e_rect.width() * max(0.0, min(1.0, e_frac))
        painter.fillRect(QRectF(e_rect.left(), e_rect.top(), fill_e, e_rect.height()), QBrush(QColor("#10b981")))

        # Handle
        hx = e_rect.left() + fill_e
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.setBrush(QBrush(QColor("#34d399")))
        painter.drawEllipse(QPointF(hx, e_rect.center().y()), 6.0, 6.0)

        painter.setPen(QPen(QColor("#10b981")))
        painter.drawText(int(e_rect.right() + 8.0), int(e_rect.center().y() + 4.0), f"{self._energy:.1f}")

        # Slider 2: V0 (0.5 to 15.0)
        v_rect = self._slider_rect_v0
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(int(v_rect.left() - 70.0), int(v_rect.center().y() + 4.0), "Thế V₀:")

        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.drawRoundedRect(v_rect, 3.0, 3.0)

        v_frac = (self._v0 - 0.5) / 14.5
        fill_v = v_rect.width() * max(0.0, min(1.0, v_frac))
        painter.fillRect(QRectF(v_rect.left(), v_rect.top(), fill_v, v_rect.height()), QBrush(QColor("#f59e0b")))

        vx = v_rect.left() + fill_v
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.setBrush(QBrush(QColor("#fbbf24")))
        painter.drawEllipse(QPointF(vx, v_rect.center().y()), 6.0, 6.0)

        painter.setPen(QPen(QColor("#f59e0b")))
        painter.drawText(int(v_rect.right() + 8.0), int(v_rect.center().y() + 4.0), f"{self._v0:.1f}")

    # ── Canvas Stamping ────────────────────────────────────────────────────────

    def stamp_to_canvas(self, board: LessonBoard) -> list[QGraphicsItem]:
        """Stamp current quantum simulation onto LessonBoard."""
        stamped: list[QGraphicsItem] = []
        copy_item = InteractiveQuantumTunnelingItem(pos=(self.x(), self.y()))
        copy_item.load_preset(self._preset_code)
        copy_item.set_energy(self._energy)
        copy_item.set_v0(self._v0)
        copy_item.set_quantum_n(self._quantum_n)
        board.addItem(copy_item)
        stamped.append(copy_item)
        return stamped

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize state for .airteach lesson packages."""
        return {
            "type": "interactive_quantum_tunneling",
            "pos": [self.x(), self.y()],
            "preset_code": self._preset_code,
            "energy": round(self._energy, 4),
            "v0": round(self._v0, 4),
            "barrier_width": round(self._barrier_width, 4),
            "quantum_n": self._quantum_n,
            "mode": self._mode,
            "show_probability_density": self._show_probability_density,
            "show_wavefunction_re": self._show_wavefunction_re,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InteractiveQuantumTunnelingItem:
        """Reconstruct visualizer from serialized dictionary."""
        pos = data.get("pos", [120.0, 50.0])
        item = cls(pos=(float(pos[0]), float(pos[1])))
        if data.get("preset_code"):
            item.load_preset(str(data.get("preset_code")))
        item.set_energy(float(data.get("energy", 4.2)))
        item.set_v0(float(data.get("v0", 6.0)))
        item.set_quantum_n(int(data.get("quantum_n", 1)))
        item._mode = str(data.get("mode", "stationary"))
        item._show_probability_density = bool(data.get("show_probability_density", True))
        item._show_wavefunction_re = bool(data.get("show_wavefunction_re", True))
        return item
