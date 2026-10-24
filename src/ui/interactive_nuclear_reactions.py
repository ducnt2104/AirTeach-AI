"""
AirTeach AI – Interactive Nuclear Physics, Binding Energy, Mass Defect, SEMF & Nuclear Reactions Simulator
==========================================================================================================
Provides an authentic nuclear physics simulation:
  - Einstein's Mass Defect & Mass-Energy Equivalence:
      Δm = [Z * m_p + (A - Z) * m_n] - M_nucleus
      E_b = Δm * c^2  (where 1 u * c^2 = 931.494102 MeV)
      E_b / A: Nuclear binding energy per nucleon.
  - Weizsäcker Semi-Empirical Mass Formula (SEMF / Liquid Drop Model):
      B(A, Z) = a_v*A - a_s*A^(2/3) - a_c*Z*(Z-1)/A^(1/3) - a_a*(A-2Z)^2/A + δ(A, Z)
      Where:
        a_v = 15.75 MeV (Volume term: strong force saturation)
        a_s = 17.80 MeV (Surface tension term: deficit of surface neighbors)
        a_c = 0.711 MeV (Coulomb electrostatic repulsion among Z protons)
        a_a = 23.70 MeV (Asymmetry term: Pauli exclusion principle favoring N ≈ Z)
        a_p = 11.18 MeV (Pairing term: +δ for even-even, 0 for odd-A, -δ for odd-odd)
  - Landmark Nuclear Binding Energy Curve:
      From Hydrogen (Eb/A = 0) through Deuterium (1.11 MeV), Helium-4 (7.07 MeV),
      up to Iron-56 (8.79 MeV) and Nickel-62 (8.80 MeV peak), descending to Uranium-238 (7.57 MeV).
      Explains thermonuclear fusion (A < 56) and nuclear fission (A > 200).
  - Curriculum Presets:
      1. BINDING_ENERGY_CURVE: Macroscopic Eb/A vs A curve with fusion/fission regions and isotope selection.
      2. DEUTERIUM_TRITIUM_FUSION: Thermonuclear reaction 2H + 3H -> 4He (3.52 MeV) + n (14.07 MeV) + 17.59 MeV.
      3. URANIUM_235_FISSION: Induced fission 235U + n -> [236U]* -> 141Ba + 92Kr + 3n + 200 MeV.
      4. PROTON_PROTON_SOLAR_CHAIN: Solar core p-p fusion chain (4p -> 4He + 2e+ + 2nu_e + 2gamma + 26.73 MeV).
      5. SEMF_LIQUID_DROP_DECOMPOSITION: Decomposition of the 5 Weizsäcker terms for any chosen nucleus.
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


# Physical Constants
U_TO_MEV = 931.49410242              # 1 u in MeV/c^2
U_TO_KG = 1.66053906660e-27          # 1 u in kg
MEV_TO_JOULE = 1.602176634e-13       # 1 MeV in Joules
PROTON_MASS_U = 1.007276466621       # Proton rest mass in u
NEUTRON_MASS_U = 1.00866491588       # Neutron rest mass in u
ELECTRON_MASS_U = 0.0005485799       # Electron rest mass in u

# Weizsäcker Liquid Drop Model Parameters (in MeV)
SEMF_AV = 15.75                      # Volume term
SEMF_AS = 17.80                      # Surface term
SEMF_AC = 0.711                      # Coulomb repulsion term
SEMF_AA = 23.70                      # Asymmetry term
SEMF_AP = 11.18                      # Pairing term constant


@dataclass(frozen=True)
class KnownIsotope:
    symbol: str
    name_vi: str
    z: int
    a: int
    eb_per_nucleon_mev: float        # Measured binding energy per nucleon (MeV)
    color: str


# Accurate experimental reference nuclides
KNOWN_ISOTOPES: list[KnownIsotope] = [
    KnownIsotope("1H", "Hiđrô-1 (Proton)", 1, 1, 0.000, "#94a3b8"),
    KnownIsotope("2H", "Đơteri (Deuterium)", 1, 2, 1.112, "#38bdf8"),
    KnownIsotope("3H", "Triti (Tritium)", 1, 3, 2.827, "#06b6d4"),
    KnownIsotope("3He", "Heli-3", 2, 3, 2.573, "#0ea5e9"),
    KnownIsotope("4He", "Heli-4 (Hạt Alpha)", 2, 4, 7.074, "#10b981"),
    KnownIsotope("6Li", "Liti-6", 3, 6, 5.332, "#84cc16"),
    KnownIsotope("7Li", "Liti-7", 3, 7, 5.606, "#a3e635"),
    KnownIsotope("12C", "Cacbon-12", 6, 12, 7.680, "#f59e0b"),
    KnownIsotope("14N", "Nitơ-14", 7, 14, 7.476, "#fbbf24"),
    KnownIsotope("16O", "Oxy-16", 8, 16, 7.976, "#f97316"),
    KnownIsotope("40Ca", "Canxi-40", 20, 40, 8.551, "#ec4899"),
    KnownIsotope("56Fe", "Sắt-56 (Đỉnh ổn định)", 26, 56, 8.790, "#ef4444"),
    KnownIsotope("62Ni", "Niken-62 (Cực đại bền vững)", 28, 62, 8.795, "#dc2626"),
    KnownIsotope("84Kr", "Krypton-84", 36, 84, 8.717, "#a855f7"),
    KnownIsotope("92Kr", "Krypton-92 (Mảnh phân hạch)", 36, 92, 8.531, "#c084fc"),
    KnownIsotope("120Sn", "Thiếc-120", 50, 120, 8.505, "#6366f1"),
    KnownIsotope("141Ba", "Bari-141 (Mảnh phân hạch)", 56, 141, 8.325, "#818cf8"),
    KnownIsotope("197Au", "Vàng-197", 79, 197, 7.916, "#eab308"),
    KnownIsotope("208Pb", "Chì-208 (Kỳ diệu kép)", 82, 208, 7.867, "#14b8a6"),
    KnownIsotope("235U", "Urani-235 (Phân hạch)", 92, 235, 7.591, "#f43f5e"),
    KnownIsotope("238U", "Urani-238", 92, 238, 7.570, "#e11d48"),
    KnownIsotope("239Pu", "Plutoni-239", 94, 239, 7.560, "#be123c"),
]

KNOWN_ISOTOPE_DICT: dict[str, KnownIsotope] = {iso.symbol: iso for iso in KNOWN_ISOTOPES}


@dataclass(frozen=True)
class NuclearPreset:
    code: str
    title_vi: str
    badge_label: str
    description: str


PRESETS: dict[str, NuclearPreset] = {
    "BINDING_ENERGY_CURVE": NuclearPreset(
        code="BINDING_ENERGY_CURVE",
        title_vi="Đường cong Năng lượng liên kết riêng (Eb/A)",
        badge_label="ĐỒ THỊ Eb/A",
        description="Đồ thị Eb/A theo số khối A; vùng nhiệt hạch A<56 và phân hạch A>200 hướng tới đỉnh sắt 56Fe.",
    ),
    "DEUTERIUM_TRITIUM_FUSION": NuclearPreset(
        code="DEUTERIUM_TRITIUM_FUSION",
        title_vi="Phản ứng Nhiệt hạch D-T (Thermonuclear Fusion)",
        badge_label="NHIỆT HẠCH D-T",
        description="Tổng hợp hạt nhân 2H + 3H -> 4He (3.52 MeV) + n (14.07 MeV) tỏa 17.59 MeV năng lượng.",
    ),
    "URANIUM_235_FISSION": NuclearPreset(
        code="URANIUM_235_FISSION",
        title_vi="Phản ứng Phân hạch Dây chuyền Urani-235",
        badge_label="PHÂN HẠCH U-235",
        description="Neutron nhiệt kích hoạt 235U dao động dạng giọt lỏng, tách thành 141Ba + 92Kr + 3n tỏa ~200 MeV.",
    ),
    "PROTON_PROTON_SOLAR_CHAIN": NuclearPreset(
        code="PROTON_PROTON_SOLAR_CHAIN",
        title_vi="Chu trình Nhiệt hạch Proton-Proton trong lòng Mặt Trời",
        badge_label="P-P MẶT TRỜI",
        description="4 proton tổng hợp thành 4He + 2e+ + 2ν_e + 2γ tỏa 26.73 MeV, nguồn năng lượng duy trì sự sống.",
    ),
    "SEMF_LIQUID_DROP_DECOMPOSITION": NuclearPreset(
        code="SEMF_LIQUID_DROP_DECOMPOSITION",
        title_vi="Mô hình Giọt chất lỏng Weizsäcker (SEMF 5 thành phần)",
        badge_label="MÔ HÌNH SEMF",
        description="Phân tích 5 số hạng: Thể tích, Bề mặt, Đẩy tĩnh điện Coulomb, Bất đối xứng và Ghép đôi spin.",
    ),
}


class InteractiveNuclearReactionsItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating Nuclear Physics:
    Binding Energy per Nucleon (Eb/A), Mass Defect (Δm), Semi-Empirical Mass Formula (SEMF),
    Nuclear Fission & Thermonuclear Fusion Reactions.
    """

    WIDTH = 700.0
    HEIGHT = 440.0

    stamp_requested = Signal()
    reaction_triggered = Signal(str)

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

        # Simulation Preset & Selected Nuclide
        self._active_preset_code: str = "BINDING_ENERGY_CURVE"
        self._preset: NuclearPreset = PRESETS[self._active_preset_code]

        # Current Isotope in Focus
        self._selected_isotope_symbol: str = "56Fe"
        self._z: int = 26
        self._a: int = 56

        # Reaction Parameters
        self._fusion_plasma_temp_mk: float = 120.0  # Million Kelvin (120 MK)
        self._fission_k_factor: float = 1.005       # Criticality multiplication factor (k ~ 1.0)
        self._reaction_phase: float = 0.0          # Progress of current animated reaction (0.0 .. 1.0)
        self._is_reacting: bool = False             # Whether reaction animation is currently playing
        self._reaction_step: int = 0               # Discrete reaction stage

        # General Animation
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
    @staticmethod
    def calculate_semf_binding_energy(a: int, z: int) -> tuple[float, float, float, float, float, float]:
        """
        Calculates binding energy terms using Weizsäcker Semi-Empirical Mass Formula (SEMF).
        Returns: (E_vol, E_surf, E_coul, E_asym, E_pair, B_total) in MeV.
        """
        if a <= 0 or z <= 0 or z > a:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if a == 1:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        n = a - z

        # 1. Volume term (+a_v * A)
        e_vol = SEMF_AV * a

        # 2. Surface term (-a_s * A^(2/3))
        e_surf = -SEMF_AS * (a ** (2.0 / 3.0))

        # 3. Coulomb repulsion (-a_c * Z*(Z-1) / A^(1/3))
        e_coul = -SEMF_AC * (z * (z - 1)) / (a ** (1.0 / 3.0))

        # 4. Asymmetry term (-a_a * (A - 2Z)^2 / A)
        e_asym = -SEMF_AA * ((a - 2.0 * z) ** 2) / float(a)

        # 5. Pairing term δ(A, Z)
        if a % 2 != 0:
            # Odd A (either Z even N odd, or Z odd N even)
            e_pair = 0.0
        elif z % 2 == 0 and n % 2 == 0:
            # Even-even nucleus: extra stable
            e_pair = SEMF_AP / math.sqrt(float(a))
        else:
            # Odd-odd nucleus: least stable
            e_pair = -SEMF_AP / math.sqrt(float(a))

        b_total = max(0.0, e_vol + e_surf + e_coul + e_asym + e_pair)
        return e_vol, e_surf, e_coul, e_asym, e_pair, b_total

    def get_binding_energy_per_nucleon(self, a: int, z: int) -> float:
        """Returns binding energy per nucleon (Eb/A) in MeV/nucleon."""
        if a <= 1:
            return 0.0
        # Check if known exact experimental isotope
        for iso in KNOWN_ISOTOPES:
            if iso.a == a and iso.z == z:
                return iso.eb_per_nucleon_mev
        # Otherwise compute from SEMF
        _, _, _, _, _, b_total = self.calculate_semf_binding_energy(a, z)
        return b_total / float(a)

    @property
    def current_binding_energy_per_nucleon(self) -> float:
        """Binding energy per nucleon of current selected isotope."""
        return self.get_binding_energy_per_nucleon(self._a, self._z)

    @property
    def current_total_binding_energy_mev(self) -> float:
        """Total binding energy of current selected isotope in MeV."""
        return self.current_binding_energy_per_nucleon * float(self._a)

    @property
    def current_mass_defect_u(self) -> float:
        """Mass defect Δm = Eb / (c^2) in atomic mass units (u)."""
        return self.current_total_binding_energy_mev / U_TO_MEV

    @property
    def current_mass_defect_kg(self) -> float:
        """Mass defect in kilograms."""
        return self.current_mass_defect_u * U_TO_KG

    @property
    def current_total_binding_energy_joules(self) -> float:
        """Total binding energy in Joules."""
        return self.current_total_binding_energy_mev * MEV_TO_JOULE

    # -------------------------------------------------------------------------
    # Getters & Setters
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> NuclearPreset:
        return self._preset

    @property
    def mass_number_a(self) -> int:
        return self._a

    @property
    def atomic_number_z(self) -> int:
        return self._z

    @property
    def neutron_number_n(self) -> int:
        return self._a - self._z

    @property
    def selected_isotope_symbol(self) -> str:
        return self._selected_isotope_symbol

    @property
    def fusion_plasma_temp_mk(self) -> float:
        return self._fusion_plasma_temp_mk

    @property
    def fission_k_factor(self) -> float:
        return self._fission_k_factor

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    @property
    def is_reacting(self) -> bool:
        return self._is_reacting

    def load_preset(self, preset_code: str) -> None:
        """Switch simulation mode to specified preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._reaction_phase = 0.0
        self._reaction_step = 0
        self._is_reacting = False

        if preset_code == "BINDING_ENERGY_CURVE":
            self.select_isotope("56Fe")
        elif preset_code == "DEUTERIUM_TRITIUM_FUSION":
            self.select_isotope("4He")
        elif preset_code == "URANIUM_235_FISSION":
            self.select_isotope("235U")
        elif preset_code == "PROTON_PROTON_SOLAR_CHAIN":
            self.select_isotope("4He")
        elif preset_code == "SEMF_LIQUID_DROP_DECOMPOSITION":
            self.select_isotope("208Pb")

        self.update()

    def select_isotope(self, symbol: str) -> None:
        """Selects an isotope from the known catalog or sets corresponding A, Z."""
        if symbol in KNOWN_ISOTOPE_DICT:
            iso = KNOWN_ISOTOPE_DICT[symbol]
            self._selected_isotope_symbol = iso.symbol
            self._z = iso.z
            self._a = iso.a
        self.update()

    def set_a_and_z(self, a: int, z: int) -> None:
        """Directly sets mass number A and atomic number Z."""
        self._a = max(1, min(240, a))
        self._z = max(1, min(self._a, z))
        # Match symbol if known
        matched = False
        for iso in KNOWN_ISOTOPES:
            if iso.a == self._a and iso.z == self._z:
                self._selected_isotope_symbol = iso.symbol
                matched = True
                break
        if not matched:
            self._selected_isotope_symbol = f"{self._a}X"
        self.update()

    def set_fusion_temp_mk(self, temp_mk: float) -> None:
        self._fusion_plasma_temp_mk = max(10.0, min(300.0, temp_mk))
        self.update()

    def set_fission_k_factor(self, k: float) -> None:
        self._fission_k_factor = max(0.80, min(1.25, k))
        self.update()

    def trigger_reaction(self) -> None:
        """Triggers dynamic reaction animation."""
        self._is_reacting = True
        self._reaction_phase = 0.0
        self._reaction_step = 0
        self.reaction_triggered.emit(self._active_preset_code)
        self.update()

    def toggle_animation(self) -> None:
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def reset_reaction(self) -> None:
        self._reaction_phase = 0.0
        self._reaction_step = 0
        self._is_reacting = False
        self.update()

    def stamp_to_board(self) -> None:
        """Requests stamping permanent vector diagram to LessonBoard."""
        self.stamp_requested.emit()

    def cleanup(self) -> None:
        """Stop internal timer and release resources."""
        if self._timer.isActive():
            self._timer.stop()

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_tick(self) -> None:
        self._anim_phase += 0.06
        if self._anim_phase >= 2.0 * math.pi:
            self._anim_phase -= 2.0 * math.pi

        if self._is_reacting:
            self._reaction_phase += 0.02
            if self._reaction_phase >= 1.0:
                self._reaction_phase = 1.0
                self._is_reacting = False

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

        # 2. Top Header & Presets
        self._draw_header(painter)

        # 3. Left Simulation Viewport
        left_w = 445.0
        view_rect = QRectF(14.0, 48.0, left_w, 310.0)
        self._draw_viewport(painter, view_rect)

        # 4. Right Control & Readout Panel
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
        # Title
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QPointF(16.0, 24.0), "VẬT LÝ HẠT NHÂN & NĂNG LƯỢNG LIÊN KẾT (NUCLEAR REACTIONS)")

        # Sub-title
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(16.0, 38.0), f"Độ hụt khối Δm • Năng lượng liên kết riêng Eb/A • SEMF Giọt chất lỏng • Phân hạch & Nhiệt hạch")

        # Preset Buttons
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

            # Compact label
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
        """Draws the main interactive visualizer according to the active preset."""
        painter.save()
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(rect, 8.0, 8.0)
        painter.setClipRect(rect)

        if self._active_preset_code == "BINDING_ENERGY_CURVE":
            self._draw_binding_energy_curve(painter, rect)
        elif self._active_preset_code == "DEUTERIUM_TRITIUM_FUSION":
            self._draw_fusion_simulation(painter, rect)
        elif self._active_preset_code == "URANIUM_235_FISSION":
            self._draw_fission_simulation(painter, rect)
        elif self._active_preset_code == "PROTON_PROTON_SOLAR_CHAIN":
            self._draw_pp_chain_simulation(painter, rect)
        elif self._active_preset_code == "SEMF_LIQUID_DROP_DECOMPOSITION":
            self._draw_semf_decomposition(painter, rect)

        painter.restore()

    def _draw_binding_energy_curve(self, painter: QPainter, rect: QRectF) -> None:
        """Draws the authentic Eb/A vs A macroscopic binding energy curve."""
        ox = rect.left() + 45.0
        oy = rect.bottom() - 35.0
        plot_w = rect.width() - 65.0
        plot_h = rect.height() - 65.0

        # Background regions: Fusion (A < 56) in subtle teal/green, Fission (A > 200) in subtle red
        x_iron = ox + (56.0 / 240.0) * plot_w
        x_fiss = ox + (200.0 / 240.0) * plot_w

        fusion_rect = QRectF(ox, rect.top() + 25.0, x_iron - ox, plot_h + 10.0)
        painter.setBrush(QBrush(QColor(16, 185, 129, 20)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRect(fusion_rect)

        fission_rect = QRectF(x_fiss, rect.top() + 25.0, (ox + plot_w) - x_fiss, plot_h + 10.0)
        painter.setBrush(QBrush(QColor(239, 68, 68, 20)))
        painter.drawRect(fission_rect)

        # Region labels
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(QRectF(ox + 8.0, rect.top() + 28.0, 140.0, 20.0), Qt.AlignmentFlag.AlignLeft, "VÙNG NHIỆT HẠCH (FUSION)")

        painter.setPen(QColor("#ef4444"))
        painter.drawText(QRectF(x_fiss - 10.0, rect.top() + 28.0, 100.0, 20.0), Qt.AlignmentFlag.AlignRight, "VÙNG PHÂN HẠCH")

        # Grid lines
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor("#1e293b"), 0.8, Qt.PenStyle.DashLine))

        # Y axis (0 to 10 MeV/nucleon)
        for eb_tick in range(0, 11, 2):
            y_tick = oy - (eb_tick / 10.0) * plot_h
            painter.drawLine(QPointF(ox, y_tick), QPointF(ox + plot_w, y_tick))
            painter.setPen(QColor("#64748b"))
            painter.drawText(QRectF(rect.left() + 8.0, y_tick - 7.0, 32.0, 14.0), Qt.AlignmentFlag.AlignRight, f"{eb_tick}")
            painter.setPen(QPen(QColor("#1e293b"), 0.8, Qt.PenStyle.DashLine))

        # X axis (0 to 240)
        for a_tick in [1, 20, 56, 100, 150, 200, 240]:
            x_tick = ox + (a_tick / 240.0) * plot_w
            painter.drawLine(QPointF(x_tick, oy), QPointF(x_tick, oy - plot_h))
            painter.setPen(QColor("#64748b"))
            painter.drawText(QRectF(x_tick - 15.0, oy + 6.0, 30.0, 14.0), Qt.AlignmentFlag.AlignCenter, f"{a_tick}")
            painter.setPen(QPen(QColor("#1e293b"), 0.8, Qt.PenStyle.DashLine))

        # Solid Axes
        painter.setPen(QPen(QColor("#94a3b8"), 1.5))
        painter.drawLine(QPointF(ox, oy), QPointF(ox + plot_w, oy))
        painter.drawLine(QPointF(ox, oy), QPointF(ox, oy - plot_h))

        # Axis Titles
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(QPointF(ox + plot_w - 90.0, oy + 24.0), "Số khối A (Nucleon)")
        painter.drawText(QPointF(ox - 35.0, oy - plot_h - 6.0), "Eb/A (MeV/nucl)")

        # 1. Theoretical Liquid Drop SEMF Smooth Curve
        semf_path = QPainterPath()
        first_pt = True
        for a_val in range(12, 241, 2):
            # Beta-stability valley empirical approximation: Z ≈ A / (2 + 0.015 * A^(2/3))
            z_val = max(1, int(round(a_val / (2.0 + 0.015 * (a_val ** (2.0 / 3.0))))))
            _, _, _, _, _, b_tot = self.calculate_semf_binding_energy(a_val, z_val)
            eb_a = b_tot / float(a_val)
            px = ox + (a_val / 240.0) * plot_w
            py = oy - (eb_a / 10.0) * plot_h
            if first_pt:
                semf_path.moveTo(px, py)
                first_pt = False
            else:
                semf_path.lineTo(px, py)

        painter.setPen(QPen(QColor("#38bdf8"), 1.8))
        painter.drawPath(semf_path)

        # 2. Plot Real Known Isotopes (Points & Labels)
        for iso in KNOWN_ISOTOPES:
            px = ox + (iso.a / 240.0) * plot_w
            py = oy - (iso.eb_per_nucleon_mev / 10.0) * plot_h

            is_active = (iso.symbol == self._selected_isotope_symbol)

            # Marker
            radius = 5.0 if is_active else 3.2
            painter.setBrush(QBrush(QColor("#fbbf24") if is_active else QColor(iso.color)))
            painter.setPen(QPen(QColor("#ffffff") if is_active else QColor("#000000"), 1.0))
            painter.drawEllipse(QPointF(px, py), radius, radius)

            # Store hit rect for interactive click
            self._btn_rects[f"iso_{iso.symbol}"] = QRectF(px - 7.0, py - 7.0, 14.0, 14.0)

            # Label prominent peaks
            if iso.symbol in ["2H", "4He", "12C", "56Fe", "62Ni", "208Pb", "235U", "238U"]:
                painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
                painter.setPen(QColor("#f8fafc") if is_active else QColor("#cbd5e1"))
                offset_y = -8.0 if iso.symbol != "2H" else 10.0
                painter.drawText(QPointF(px - 10.0, py + offset_y), iso.symbol)

        # 3. Selected Isotope Pointer Ring
        sel_iso = KNOWN_ISOTOPE_DICT.get(self._selected_isotope_symbol)
        sel_eb_a = sel_iso.eb_per_nucleon_mev if sel_iso else self.current_binding_energy_per_nucleon
        cur_x = ox + (self._a / 240.0) * plot_w
        cur_y = oy - (sel_eb_a / 10.0) * plot_h

        pulse_r = 7.0 + 2.0 * math.sin(self._anim_phase * 2.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#fbbf24"), 1.8, Qt.PenStyle.DashLine))
        painter.drawEllipse(QPointF(cur_x, cur_y), pulse_r, pulse_r)

    def _draw_fusion_simulation(self, painter: QPainter, rect: QRectF) -> None:
        """Visualizes the D-T thermonuclear fusion process and energy release."""
        cx = rect.center().x()
        cy = rect.center().y() - 10.0

        p = self._reaction_phase

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 22.0), "PHẢN ỨNG NHIỆT HẠCH D-T:  ²₁H + ³₁H ➔ ⁴₂He + ¹₀n + 17.59 MeV")

        # Step indicator
        step_txt = "Tiếp cận rào thế Coulomb" if p < 0.35 else ("Hạt nhân hợp phần [⁵He]* kích thích" if p < 0.65 else "Bắn tách Alpha (3.52 MeV) & Neutron (14.07 MeV)")
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 38.0), f"Giai đoạn: {step_txt} (T plasma = {self._fusion_plasma_temp_mk:.0f} Triệu K)")

        if p < 0.35:
            # Stage 1: Approaching reactants Deuterium and Tritium
            sub_p = p / 0.35
            d_x = cx - 140.0 + sub_p * 100.0
            t_x = cx + 140.0 - sub_p * 100.0

            # Deuterium (1 p red, 1 n blue)
            self._draw_deuterium(painter, QPointF(d_x, cy))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#38bdf8"))
            painter.drawText(QPointF(d_x - 15.0, cy + 32.0), "Đơteri (²H)")

            # Tritium (1 p red, 2 n blue)
            self._draw_tritium(painter, QPointF(t_x, cy))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#06b6d4"))
            painter.drawText(QPointF(t_x - 15.0, cy + 32.0), "Triti (³H)")

            # Coulomb barrier energy curve
            self._draw_coulomb_barrier_mini(painter, QRectF(rect.left() + 14.0, rect.bottom() - 65.0, 160.0, 50.0))

        elif p < 0.65:
            # Stage 2: Fusion spark / compound nucleus [5He]*
            sub_p = (p - 0.35) / 0.30
            glow_r = 25.0 + 15.0 * math.sin(sub_p * math.pi)

            # Glowing fusion flash
            painter.setBrush(QBrush(QColor(251, 191, 36, int(180 * math.sin(sub_p * math.pi)))))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(cx, cy), glow_r * 1.5, glow_r * 1.5)

            # Intermediate compound nucleus [5He]*
            self._draw_compound_5he(painter, QPointF(cx, cy))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#f59e0b"))
            painter.drawText(QPointF(cx - 50.0, cy - 35.0), "[⁵₂He]* Hợp phần")

        else:
            # Stage 3: High speed products Alpha + Neutron flying apart
            sub_p = (p - 0.65) / 0.35
            alpha_x = cx - 40.0 - sub_p * 120.0
            alpha_y = cy - sub_p * 30.0
            n_x = cx + 40.0 + sub_p * 140.0
            n_y = cy + sub_p * 25.0

            # Alpha particle (4He: 2p + 2n)
            self._draw_alpha_particle(painter, QPointF(alpha_x, alpha_y))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#10b981"))
            painter.drawText(QPointF(alpha_x - 25.0, alpha_y + 35.0), "Hạt α (⁴₂He)\nE_k = 3.52 MeV")

            # Fast neutron (1n)
            self._draw_single_neutron(painter, QPointF(n_x, n_y))
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#f43f5e"))
            painter.drawText(QPointF(n_x - 20.0, n_y + 25.0), "Neutron nhanh (¹₀n)\nE_k = 14.07 MeV")

            # Energy output summary badge
            q_badge = QRectF(cx - 75.0, rect.bottom() - 45.0, 150.0, 32.0)
            painter.setBrush(QBrush(QColor("#064e3b")))
            painter.setPen(QPen(QColor("#10b981"), 1.2))
            painter.drawRoundedRect(q_badge, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(q_badge, Qt.AlignmentFlag.AlignCenter, "NĂNG LƯỢNG TỎA RA: Q = 17.59 MeV\n(Mỗi phản ứng: ~2.82 pJ)")

    def _draw_fission_simulation(self, painter: QPainter, rect: QRectF) -> None:
        """Visualizes thermal neutron-induced U-235 fission liquid drop deformation and chain branching."""
        cx = rect.center().x()
        cy = rect.center().y() - 10.0

        p = self._reaction_phase

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#f43f5e"))
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 22.0), "PHÂN HẠCH DÂY CHUYỀN:  ²³⁵₉₂U + ¹₀n ➔ [²³⁶₉₂U]* ➔ ¹⁴¹₅₆Ba + ⁹²₃₆Kr + 3 ¹₀n + 200 MeV")

        # Criticality Status
        k_str = f"Hệ số nhân neutron k = {self._fission_k_factor:.3f} ("
        if self._fission_k_factor < 0.995:
            k_str += "Dưới hạn - Chuỗi tắt dần)"
            k_col = QColor("#38bdf8")
        elif self._fission_k_factor <= 1.005:
            k_str += "Tới hạn - Lò phản ứng ổn định)"
            k_col = QColor("#10b981")
        else:
            k_str += "Vượt hạn - Phản ứng bùng nổ)"
            k_col = QColor("#ef4444")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(k_col)
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 38.0), k_str)

        if p < 0.25:
            # Stage 1: Incident thermal neutron approaching Uranium-235
            sub_p = p / 0.25
            n_x = cx - 180.0 + sub_p * 130.0

            # Incident neutron
            self._draw_single_neutron(painter, QPointF(n_x, cy))
            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QColor("#38bdf8"))
            painter.drawText(QPointF(n_x - 20.0, cy - 12.0), "Neutron nhiệt (0.025 eV)")

            # Spherical Uranium-235 nucleus
            self._draw_uranium_sphere(painter, QPointF(cx, cy), 32.0)
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#f43f5e"))
            painter.drawText(QPointF(cx - 20.0, cy + 45.0), "²³⁵₉₂U")

        elif p < 0.55:
            # Stage 2: Liquid drop oscillation and neck deformation (dumb-bell shape)
            sub_p = (p - 0.25) / 0.30
            elongation = 32.0 + sub_p * 28.0
            neck_r = max(6.0, 20.0 - sub_p * 16.0)

            # Dumbbell liquid drop shape
            self._draw_dumbbell_nucleus(painter, QPointF(cx, cy), elongation, neck_r)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#fbbf24"))
            painter.drawText(QPointF(cx - 60.0, cy - 35.0), "[²³⁶₉₂U]* Dao động thắt cổ chai")

        else:
            # Stage 3: Scission into Ba-141 and Kr-92 + 3 flying prompt neutrons
            sub_p = (p - 0.55) / 0.45
            ba_x = cx - 60.0 - sub_p * 100.0
            kr_x = cx + 60.0 + sub_p * 100.0

            # Barium-141 (heavier fragment)
            self._draw_fragment(painter, QPointF(ba_x, cy), 24.0, "¹⁴¹₅₆Ba", "#818cf8")

            # Krypton-92 (lighter fragment)
            self._draw_fragment(painter, QPointF(kr_x, cy), 18.0, "⁹²₃₆Kr", "#c084fc")

            # 3 Flying Prompt Neutrons
            n1_pos = QPointF(cx + (sub_p * 80.0) * math.cos(math.pi * 0.3), cy - sub_p * 70.0)
            n2_pos = QPointF(cx, cy + sub_p * 80.0)
            n3_pos = QPointF(cx - (sub_p * 80.0) * math.cos(math.pi * 0.3), cy - sub_p * 70.0)

            for n_pos in [n1_pos, n2_pos, n3_pos]:
                self._draw_single_neutron(painter, n_pos)

            # Energy breakdown card
            e_r = QRectF(cx - 100.0, rect.bottom() - 48.0, 200.0, 36.0)
            painter.setBrush(QBrush(QColor("#1e1b4b")))
            painter.setPen(QPen(QColor("#6366f1"), 1.0))
            painter.drawRoundedRect(e_r, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(e_r, Qt.AlignmentFlag.AlignCenter, "NĂNG LƯỢNG PHÂN HẠCH: Q ≈ 200 MeV\nĐộng năng mảnh: 168 MeV • 3 Neutron: 5 MeV • Tia γ & β: 27 MeV")

    def _draw_pp_chain_simulation(self, painter: QPainter, rect: QRectF) -> None:
        """Visualizes the Sun's core Proton-Proton fusion chain."""
        cx = rect.center().x()
        cy = rect.center().y() - 10.0

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 22.0), "CHU TRÌNH PROTON-PROTON (P-P CHAIN) LÒ LUYỆN NGUYÊN TỐ MẶT TRỜI")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(QPointF(rect.left() + 14.0, rect.top() + 38.0), "Nhiệt độ lõi: 15.7 triệu K • Áp suất: 250 tỷ atm • Tổng hợp 600 triệu tấn Hiđrô/giây")

        # 3 Step Visual Pipeline
        step_boxes = [
            ("Bước 1: Tương tác yếu", "p + p ➔ d + e⁺ + ν_e\n(+0.42 MeV)", "#38bdf8", rect.left() + 18.0),
            ("Bước 2: Bắt bức xạ", "d + p ➔ ³He + γ\n(+5.49 MeV)", "#06b6d4", rect.left() + 155.0),
            ("Bước 3: Tổng hợp Alpha", "³He + ³He ➔ ⁴He + 2p\n(+12.86 MeV)", "#10b981", rect.left() + 295.0),
        ]

        for title, formula, col_hex, bx in step_boxes:
            b_rect = QRectF(bx, cy - 50.0, 125.0, 100.0)
            painter.setBrush(QBrush(QColor("#0f172a")))
            painter.setPen(QPen(QColor(col_hex), 1.2))
            painter.drawRoundedRect(b_rect, 6.0, 6.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor(col_hex))
            painter.drawText(QRectF(bx + 6.0, cy - 44.0, 113.0, 16.0), Qt.AlignmentFlag.AlignCenter, title)

            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QColor("#f8fafc"))
            painter.drawText(QRectF(bx + 6.0, cy - 22.0, 113.0, 40.0), Qt.AlignmentFlag.AlignCenter, formula)

            # Miniature particle icon inside box
            if "Bước 1" in title:
                self._draw_single_proton(painter, QPointF(bx + 40.0, cy + 30.0))
                self._draw_single_proton(painter, QPointF(bx + 85.0, cy + 30.0))
            elif "Bước 2" in title:
                self._draw_deuterium(painter, QPointF(bx + 40.0, cy + 30.0), r=6.0)
                self._draw_single_proton(painter, QPointF(bx + 85.0, cy + 30.0))
            else:
                self._draw_alpha_particle(painter, QPointF(bx + 62.0, cy + 30.0), r=7.0)

        # Net reaction banner
        net_r = QRectF(rect.left() + 18.0, rect.bottom() - 48.0, rect.width() - 36.0, 36.0)
        painter.setBrush(QBrush(QColor("#451a03")))
        painter.setPen(QPen(QColor("#f59e0b"), 1.2))
        painter.drawRoundedRect(net_r, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(net_r, Qt.AlignmentFlag.AlignCenter, "PHƯƠNG TRÌNH TỔNG QUÁT:  4 ¹₁p ➔ ⁴₂He + 2 e⁺ + 2 ν_e + 2 γ + 26.73 MeV\nHiệu suất chuyển hóa khối lượng thành năng lượng: η = 0.71% (Δm/m = 0.0071)")

    def _draw_semf_decomposition(self, painter: QPainter, rect: QRectF) -> None:
        """Visualizes the 5 Weizsäcker SEMF terms (Liquid Drop Model) as comparative bars."""
        ox = rect.left() + 20.0
        oy = rect.top() + 45.0

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(ox, rect.top() + 22.0), f"PHÂN TÍCH 5 THÀNH PHẦN SEMF WEIZSÄCKER: HẠT NHÂN {self._selected_isotope_symbol} (A={self._a}, Z={self._z})")

        e_vol, e_surf, e_coul, e_asym, e_pair, b_total = self.calculate_semf_binding_energy(self._a, self._z)

        terms = [
            ("1. Năng lượng Thể tích (+a_v*A):", e_vol, "#0ea5e9", "Tương tác mạnh bão hòa giữa các nucleon lân cận"),
            ("2. Năng lượng Bề mặt (-a_s*A^(2/3)):", e_surf, "#f59e0b", "Sự thiếu hụt liên kết của các nucleon ở mặt ngoài giọt"),
            ("3. Lực đẩy Coulomb (-a_c*Z*(Z-1)/A^(1/3)):", e_coul, "#ef4444", "Lực đẩy tĩnh điện tầm xa giữa Z hạt proton"),
            ("4. Năng lượng Bất đối xứng (-a_a*(A-2Z)^2/A):", e_asym, "#a855f7", "Nguyên lý loại trừ Pauli ưu tiên N ≈ Z"),
            ("5. Năng lượng Ghép đôi spin (±δ):", e_pair, "#10b981", "Độ bền tăng khi Z chẵn, N chẵn (even-even)"),
        ]

        bar_y = oy + 10.0
        max_scale = max(100.0, abs(e_vol))

        for name, val, col_hex, desc in terms:
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor(col_hex))
            painter.drawText(QPointF(ox, bar_y + 10.0), f"{name} {val:+.2f} MeV")

            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(QPointF(ox + 220.0, bar_y + 10.0), desc)

            # Bar track
            bar_w = 380.0
            mid_x = ox + 150.0
            len_px = (val / max_scale) * 180.0

            painter.setPen(QPen(QColor("#1e293b"), 1.0))
            painter.drawLine(QPointF(ox, bar_y + 20.0), QPointF(ox + bar_w, bar_y + 20.0))

            # Value bar
            b_rect = QRectF(mid_x, bar_y + 16.0, len_px, 8.0) if len_px >= 0 else QRectF(mid_x + len_px, bar_y + 16.0, -len_px, 8.0)
            painter.setBrush(QBrush(QColor(col_hex)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(b_rect, 3.0, 3.0)

            bar_y += 38.0

        # Total SEMF Result Card
        tot_rect = QRectF(ox, rect.bottom() - 50.0, rect.width() - 40.0, 38.0)
        painter.setBrush(QBrush(QColor("#0369a1")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.2))
        painter.drawRoundedRect(tot_rect, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(
            tot_rect,
            Qt.AlignmentFlag.AlignCenter,
            f"TỔNG NĂNG LƯỢNG LIÊN KẾT B(A, Z) = {b_total:.2f} MeV   |   LIÊN KẾT RIÊNG: Eb/A = {b_total/self._a:.3f} MeV/nucleon\n"
            f"Độ hụt khối lý thuyết: Δm = {b_total/U_TO_MEV:.5f} u ({b_total/U_TO_MEV * U_TO_KG * 1e27:.3f} × 10⁻²⁷ kg)"
        )

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

        # Header: Selected Isotope Card
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(ox, y), "THÔNG SỐ HẠT NHÂN:")

        y += 18.0
        iso_card = QRectF(ox, y, w, 32.0)
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(iso_card, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(QRectF(ox + 8.0, y, 70.0, 32.0), Qt.AlignmentFlag.AlignVCenter, self._selected_isotope_symbol)

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(QRectF(ox + 80.0, y + 2.0, w - 85.0, 28.0), Qt.AlignmentFlag.AlignVCenter, f"Z={self._z} (p) | N={self.neutron_number_n} (n)\nA={self._a} (Nucleon)")

        # Steppers for A and Z
        y += 38.0
        btn_a_down = QRectF(ox, y, 32.0, 20.0)
        btn_a_up = QRectF(ox + 36.0, y, 32.0, 20.0)
        btn_z_down = QRectF(ox + 100.0, y, 32.0, 20.0)
        btn_z_up = QRectF(ox + 136.0, y, 32.0, 20.0)

        self._btn_rects["a_down"] = btn_a_down
        self._btn_rects["a_up"] = btn_a_up
        self._btn_rects["z_down"] = btn_z_down
        self._btn_rects["z_up"] = btn_z_up

        for b_r, b_txt in [(btn_a_down, "-A"), (btn_a_up, "+A"), (btn_z_down, "-Z"), (btn_z_up, "+Z")]:
            painter.setBrush(QBrush(QColor("#1e293b")))
            painter.setPen(QPen(QColor("#475569"), 1.0))
            painter.drawRoundedRect(b_r, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(b_r, Qt.AlignmentFlag.AlignCenter, b_txt)

        # Physics Measurements Readout
        y += 28.0
        eb_a = self.current_binding_energy_per_nucleon
        eb_tot = self.current_total_binding_energy_mev
        dm_u = self.current_mass_defect_u
        dm_kg = self.current_mass_defect_kg

        data_rows = [
            (f"• Eb/A: {eb_a:.3f} MeV/nucl", QColor("#10b981")),
            (f"• Tổng Eb: {eb_tot:.2f} MeV", QColor("#38bdf8")),
            (f"• Độ hụt khối Δm: {dm_u:.5f} u", QColor("#f59e0b")),
            (f"• Khối lượng thiếu: {dm_kg*1e27:.3f} ×10⁻²⁷ kg", QColor("#94a3b8")),
            (f"• Năng lượng nghỉ: {eb_tot * MEV_TO_JOULE * 1e12:.2f} pJ", QColor("#cbd5e1")),
        ]

        for txt, col in data_rows:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(col)
            painter.drawText(QPointF(ox, y), txt)
            y += 16.0

        # Mode Specific Controls
        y += 4.0
        if self._active_preset_code == "DEUTERIUM_TRITIUM_FUSION":
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#38bdf8"))
            painter.drawText(QPointF(ox, y), f"Nhiệt độ Plasma: {self._fusion_plasma_temp_mk:.0f} MK")
            y += 8.0
            t_down = QRectF(ox, y, 40.0, 20.0)
            t_up = QRectF(ox + 46.0, y, 40.0, 20.0)
            self._btn_rects["temp_down"] = t_down
            self._btn_rects["temp_up"] = t_up
            for b_r, b_txt in [(t_down, "-20M"), (t_up, "+20M")]:
                painter.setBrush(QBrush(QColor("#1e293b")))
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.drawRoundedRect(b_r, 3.0, 3.0)
                painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
                painter.setPen(QColor("#ffffff"))
                painter.drawText(b_r, Qt.AlignmentFlag.AlignCenter, b_txt)
            y += 26.0

        elif self._active_preset_code == "URANIUM_235_FISSION":
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#f43f5e"))
            painter.drawText(QPointF(ox, y), f"Hệ số nhân k: {self._fission_k_factor:.3f}")
            y += 8.0
            k_down = QRectF(ox, y, 40.0, 20.0)
            k_up = QRectF(ox + 46.0, y, 40.0, 20.0)
            self._btn_rects["k_down"] = k_down
            self._btn_rects["k_up"] = k_up
            for b_r, b_txt in [(k_down, "-0.05"), (k_up, "+0.05")]:
                painter.setBrush(QBrush(QColor("#1e293b")))
                painter.setPen(QPen(QColor("#475569"), 1.0))
                painter.drawRoundedRect(b_r, 3.0, 3.0)
                painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
                painter.setPen(QColor("#ffffff"))
                painter.drawText(b_r, Qt.AlignmentFlag.AlignCenter, b_txt)
            y += 26.0

        # Action Buttons: Trigger, Play/Pause, Reset
        trig_r = QRectF(ox, rect.bottom() - 48.0, w, 22.0)
        self._btn_rects["trigger"] = trig_r
        painter.setBrush(QBrush(QColor("#dc2626") if self._is_reacting else QColor("#0284c7")))
        painter.setPen(QPen(QColor("#f8fafc"), 1.0))
        painter.drawRoundedRect(trig_r, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(trig_r, Qt.AlignmentFlag.AlignCenter, "💥 KÍCH HOẠT PHẢN ỨNG" if not self._is_reacting else "ĐANG PHẢN ỨNG...")

        rst_r = QRectF(ox, rect.bottom() - 22.0, w / 2.0 - 4.0, 18.0)
        anim_r = QRectF(ox + w / 2.0 + 4.0, rect.bottom() - 22.0, w / 2.0 - 4.0, 18.0)
        self._btn_rects["reset"] = rst_r
        self._btn_rects["toggle_anim"] = anim_r

        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(rst_r, 3.0, 3.0)
        painter.drawRoundedRect(anim_r, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(rst_r, Qt.AlignmentFlag.AlignCenter, "Đặt lại")
        painter.drawText(anim_r, Qt.AlignmentFlag.AlignCenter, "Dừng" if self._is_animating else "Chạy")

    # -------------------------------------------------------------------------
    # Bottom Physics Theory Banner
    # -------------------------------------------------------------------------
    def _draw_theory_banner(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 10.0, rect.top() + 16.0), "NGUYÊN LÝ VẬT LÝ HẠT NHÂN & NĂNG LƯỢNG LIÊN KẾT EINSTEIN (1905):")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        desc = (
            f"• Định luật Einstein: E = Δm·c² (1 u = 931.494 MeV). Độ hụt khối Δm = [Z·m_p + N·m_n] - M_hạt_nhân.\n"
            f"• Năng lượng liên kết riêng Eb/A đạt cực đại ở Fe-56/Ni-62 (~8.8 MeV/nucl). "
            f"Hạt nhân nhẹ kết hợp (Nhiệt hạch) hoặc hạt nhân nặng phân tách (Phân hạch) đều chuyển dịch về phía đỉnh bền vững, giải phóng năng lượng khổng lồ.\n"
            f"• Mô tả chế độ: {self._preset.description}"
        )
        painter.drawText(
            QRectF(rect.left() + 10.0, rect.top() + 22.0, rect.width() - 20.0, 38.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            desc,
        )

    # -------------------------------------------------------------------------
    # Visual Particle Rendering Helpers
    # -------------------------------------------------------------------------
    def _draw_single_proton(self, painter: QPainter, pos: QPointF, r: float = 8.0) -> None:
        painter.setBrush(QBrush(QColor("#ef4444")))
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.drawEllipse(pos, r, r)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.drawText(QRectF(pos.x() - r, pos.y() - r, 2 * r, 2 * r), Qt.AlignmentFlag.AlignCenter, "+")

    def _draw_single_neutron(self, painter: QPainter, pos: QPointF, r: float = 8.0) -> None:
        painter.setBrush(QBrush(QColor("#3b82f6")))
        painter.setPen(QPen(QColor("#ffffff"), 1.0))
        painter.drawEllipse(pos, r, r)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.drawText(QRectF(pos.x() - r, pos.y() - r, 2 * r, 2 * r), Qt.AlignmentFlag.AlignCenter, "n")

    def _draw_deuterium(self, painter: QPainter, pos: QPointF, r: float = 8.0) -> None:
        self._draw_single_proton(painter, QPointF(pos.x() - r * 0.7, pos.y()), r)
        self._draw_single_neutron(painter, QPointF(pos.x() + r * 0.7, pos.y()), r)

    def _draw_tritium(self, painter: QPainter, pos: QPointF, r: float = 8.0) -> None:
        self._draw_single_proton(painter, QPointF(pos.x(), pos.y() - r * 0.8), r)
        self._draw_single_neutron(painter, QPointF(pos.x() - r * 0.8, pos.y() + r * 0.6), r)
        self._draw_single_neutron(painter, QPointF(pos.x() + r * 0.8, pos.y() + r * 0.6), r)

    def _draw_alpha_particle(self, painter: QPainter, pos: QPointF, r: float = 8.0) -> None:
        self._draw_single_proton(painter, QPointF(pos.x() - r * 0.6, pos.y() - r * 0.6), r)
        self._draw_single_neutron(painter, QPointF(pos.x() + r * 0.6, pos.y() - r * 0.6), r)
        self._draw_single_neutron(painter, QPointF(pos.x() - r * 0.6, pos.y() + r * 0.6), r)
        self._draw_single_proton(painter, QPointF(pos.x() + r * 0.6, pos.y() + r * 0.6), r)

    def _draw_compound_5he(self, painter: QPainter, pos: QPointF) -> None:
        for angle_deg in [0, 72, 144, 216, 288]:
            rad = math.radians(angle_deg + self._anim_phase * 50.0)
            pt = QPointF(pos.x() + 12.0 * math.cos(rad), pos.y() + 12.0 * math.sin(rad))
            if angle_deg in [0, 144]:
                self._draw_single_proton(painter, pt, 7.0)
            else:
                self._draw_single_neutron(painter, pt, 7.0)

    def _draw_uranium_sphere(self, painter: QPainter, pos: QPointF, radius: float) -> None:
        painter.setBrush(QBrush(QColor("#7f1d1d")))
        painter.setPen(QPen(QColor("#ef4444"), 1.8))
        painter.drawEllipse(pos, radius, radius)

        # Micro nucleons pattern
        painter.setPen(Qt.PenStyle.NoPen)
        for i in range(12):
            ang = i * (2.0 * math.pi / 12.0)
            dist = radius * 0.6
            pt = QPointF(pos.x() + dist * math.cos(ang), pos.y() + dist * math.sin(ang))
            col = QColor("#ef4444") if i % 2 == 0 else QColor("#3b82f6")
            painter.setBrush(QBrush(col))
            painter.drawEllipse(pt, 4.0, 4.0)

    def _draw_dumbbell_nucleus(self, painter: QPainter, pos: QPointF, elongation: float, neck_r: float) -> None:
        p1 = QPointF(pos.x() - elongation, pos.y())
        p2 = QPointF(pos.x() + elongation, pos.y())

        painter.setBrush(QBrush(QColor("#991b1b")))
        painter.setPen(QPen(QColor("#f87171"), 1.5))
        painter.drawEllipse(p1, 24.0, 24.0)
        painter.drawEllipse(p2, 20.0, 20.0)

        # Neck connecting them
        neck_rect = QRectF(pos.x() - elongation * 0.4, pos.y() - neck_r, elongation * 0.8, 2 * neck_r)
        painter.setBrush(QBrush(QColor("#b91c1c")))
        painter.drawRoundedRect(neck_rect, neck_r, neck_r)

    def _draw_fragment(self, painter: QPainter, pos: QPointF, radius: float, label: str, col_hex: str) -> None:
        painter.setBrush(QBrush(QColor(col_hex)))
        painter.setPen(QPen(QColor("#ffffff"), 1.2))
        painter.drawEllipse(pos, radius, radius)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRectF(pos.x() - radius, pos.y() - radius, 2 * radius, 2 * radius), Qt.AlignmentFlag.AlignCenter, label)

    def _draw_coulomb_barrier_mini(self, painter: QPainter, rect: QRectF) -> None:
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#334155"), 0.8))
        painter.drawRoundedRect(rect, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 5, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(rect.left() + 4.0, rect.top() + 10.0), "RÀO THẾ COULOMB V_C:")

        # Draw barrier peak
        path = QPainterPath()
        path.moveTo(rect.left() + 6.0, rect.bottom() - 6.0)
        path.quadTo(rect.left() + rect.width() * 0.5, rect.top() + 14.0, rect.right() - 6.0, rect.bottom() - 6.0)
        painter.setPen(QPen(QColor("#ef4444"), 1.2))
        painter.drawPath(path)

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

        # 2. Known isotopes clicking on binding energy curve
        for iso in KNOWN_ISOTOPES:
            r = self._btn_rects.get(f"iso_{iso.symbol}")
            if r and r.contains(local_pos):
                self.select_isotope(iso.symbol)
                return True

        # 3. Mass number steppers
        if self._btn_rects.get("a_down") and self._btn_rects["a_down"].contains(local_pos):
            self.set_a_and_z(self._a - 1, self._z)
            return True
        if self._btn_rects.get("a_up") and self._btn_rects["a_up"].contains(local_pos):
            self.set_a_and_z(self._a + 1, self._z)
            return True
        if self._btn_rects.get("z_down") and self._btn_rects["z_down"].contains(local_pos):
            self.set_a_and_z(self._a, self._z - 1)
            return True
        if self._btn_rects.get("z_up") and self._btn_rects["z_up"].contains(local_pos):
            self.set_a_and_z(self._a, self._z + 1)
            return True

        # 4. Temperature steppers
        if self._btn_rects.get("temp_down") and self._btn_rects["temp_down"].contains(local_pos):
            self.set_fusion_temp_mk(self._fusion_plasma_temp_mk - 20.0)
            return True
        if self._btn_rects.get("temp_up") and self._btn_rects["temp_up"].contains(local_pos):
            self.set_fusion_temp_mk(self._fusion_plasma_temp_mk + 20.0)
            return True

        # 5. Fission k steppers
        if self._btn_rects.get("k_down") and self._btn_rects["k_down"].contains(local_pos):
            self.set_fission_k_factor(self._fission_k_factor - 0.05)
            return True
        if self._btn_rects.get("k_up") and self._btn_rects["k_up"].contains(local_pos):
            self.set_fission_k_factor(self._fission_k_factor + 0.05)
            return True

        # 6. Action buttons
        if self._btn_rects.get("trigger") and self._btn_rects["trigger"].contains(local_pos):
            self.trigger_reaction()
            return True
        if self._btn_rects.get("reset") and self._btn_rects["reset"].contains(local_pos):
            self.reset_reaction()
            return True
        if self._btn_rects.get("toggle_anim") and self._btn_rects["toggle_anim"].contains(local_pos):
            self.toggle_animation()
            return True
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
