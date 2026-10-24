"""
AirTeach AI - Interactive Chemistry & Physics GC-MS (Gas Chromatography - Mass Spectrometry) Simulator
========================================================================================================
Provides an authentic analytical chemistry simulation of GC-MS chromatography and mass spectrometry:
  - Theoretical Foundations (Zero Mock Data):
      1. Gas Chromatography (GC - Sắc Ký Khí):
           - Mobile phase: Inert carrier gas (He / N₂) at linear velocity u (cm/s).
           - Capillary column (DB-5MS, L = 30m, 0.25mm ID, 0.25μm df).
           - Van Deemter Equation for plate height H:
               H(u) = A + B/u + (C_s + C_m)·u
               Number of theoretical plates: N = L / H = 5.545 · (t_R / w_{1/2})²
           - Temperature programming: T(t) = T_init + β · t
           - Chromatogram: Total Ion Current (TIC) Gaussian peaks:
               I(t) = Σ [ A_i / (σ_i · sqrt(2π)) ] · exp(-(t - t_{R,i})² / (2·σ_i²))
           - Chromatographic Resolution: R_s = 2 · (t_{R,2} - t_{R,1}) / (w_1 + w_2) ≥ 1.5 (Baseline separation)
      2. Electron Ionization Mass Spectrometry (EI-MS 70 eV - Khối Phổ Ion Hóa Điện Tử):
           - 70 eV electron bombardment generating radical cations:
               M + e⁻ (70 eV) → M⁺• + 2 e⁻
           - Unimolecular fragmentation pathways:
               α-cleavage, inductive cleavage, McLafferty rearrangement, tropylium ion formation (m/z 91).
           - Normalized stick mass spectrum: Relative abundance (%) vs mass-to-charge ratio (m/z).
           - Isotope cluster patterns (¹³C at 1.1%/C, ³⁵Cl/³⁷Cl at 3:1, ⁷⁹Br/⁸¹Br at 1:1).
  - 5 Authentic Analytical Curriculum Presets:
      - ESSENTIAL_OIL_TERPENES: Phân tích tinh dầu thiên nhiên (α-Pinene, Limonene, Linalool, Menthol, Caryophyllene)
      - FORENSIC_BLOOD_ALCOHOLS: Giám định pháp y cồn trong máu (Methanol, Ethanol, Isopropanol, n-Propanol chuẩn nội)
      - ENVIRONMENTAL_POPS_PCBS: Độc chất môi trường Clo hóa POPs/PCB với chùm vạch đồng vị Cl đặc trưng
      - DOPING_ANABOLIC_STEROIDS: Xét nghiệm doping thể thao (Testosterone vs Epitestosterone T/E ratio)
      - PETROCHEMICAL_GASOLINE_BTEX: Hóa dầu và khí thải (Benzen, Toluen, Etylbenzen, m/p-Xylen, o-Xylen)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional, Union, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
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

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GCMSPeak:
    """Represents a chemical compound peak in GC and its EI-MS 70eV fragmentation spectrum."""

    name_vi: str
    formula: str
    retention_time_min: float  # Base retention time at standard ramp
    peak_area: float  # Relative area / abundance
    molecular_weight: int  # Molecular ion m/z
    base_peak_mz: int  # m/z of 100% intensity peak
    # Characteristic mass fragments: list of (m/z, relative_abundance_pct, fragment_identity)
    fragments: tuple[tuple[int, float, str], ...]


@dataclass(frozen=True)
class GCMSPreset:
    """Curriculum analytical preset for GC-MS workstation."""

    code: str
    title_vi: str
    carrier_gas: str  # "Helium" or "Nitrogen"
    column_type: str  # e.g. "DB-5MS (30m × 0.25mm × 0.25μm)"
    t_init_c: float  # Initial oven temp
    ramp_rate_c_min: float  # Heating ramp rate β (°C/min)
    t_final_c: float  # Final oven temp
    flow_rate_ml_min: float  # Carrier gas flow rate
    compounds: tuple[GCMSPeak, ...]
    description_vi: str
    formula_vi: str


PRESETS: dict[str, GCMSPreset] = {
    "ESSENTIAL_OIL_TERPENES": GCMSPreset(
        code="ESSENTIAL_OIL_TERPENES",
        title_vi="Tinh Dầu Thiên Nhiên (Terpenes: Bạc Hà & Cam Chanh)",
        carrier_gas="Helium",
        column_type="DB-5MS (30m × 0.25mm × 0.25μm)",
        t_init_c=60.0,
        ramp_rate_c_min=10.0,
        t_final_c=240.0,
        flow_rate_ml_min=1.2,
        compounds=(
            GCMSPeak("α-Pinene", "C₁₀H₁₆", 3.2, 18.0, 136, 93, ((93, 100.0, "C₇H₉⁺ base"), (91, 45.0, "C₇H₇⁺"), (77, 30.0, "C₆H₅⁺"), (121, 20.0, "[M-15]⁺"), (136, 12.0, "M⁺•"))),
            GCMSPeak("Limonene", "C₁₀H₁₆", 5.1, 45.0, 136, 68, ((68, 100.0, "C₅H₈⁺ retro-DA"), (93, 75.0, "C₇H₉⁺"), (79, 35.0, "C₆H₇⁺"), (121, 25.0, "[M-15]⁺"), (136, 22.0, "M⁺•"))),
            GCMSPeak("Linalool", "C₁₀H₁₈O", 6.8, 22.0, 154, 71, ((71, 100.0, "C₄H₇O⁺"), (93, 65.0, "C₇H₉⁺"), (55, 50.0, "C₄H₇⁺"), (121, 28.0, "[M-H₂O-CH₃]⁺"), (136, 15.0, "[M-H₂O]⁺•"))),
            GCMSPeak("Menthol", "C₁₀H₂₀O", 8.4, 38.0, 156, 71, ((71, 100.0, "C₄H₇O⁺"), (81, 70.0, "C₆H₉⁺"), (95, 60.0, "C₇H₁₁⁺"), (138, 18.0, "[M-H₂O]⁺•"), (156, 5.0, "M⁺•"))),
            GCMSPeak("β-Caryophyllene", "C₁₅H₂₄", 11.2, 28.0, 204, 93, ((93, 100.0, "C₇H₉⁺"), (79, 78.0, "C₆H₇⁺"), (133, 55.0, "C₁₀H₁₃⁺"), (161, 38.0, "[M-43]⁺"), (204, 15.0, "M⁺•"))),
        ),
        description_vi="Phân tích định tính và định lượng hỗn hợp tinh dầu bay hơi. Tách rời monoterpene (C10) và sesquiterpene (C15) theo độ sôi và độ phân cực.",
        formula_vi="N = 5.545 · (t_R / w_{1/2})²  |  EI-MS 70 eV M⁺• → mảnh phân rã đặc trưng",
    ),
    "FORENSIC_BLOOD_ALCOHOLS": GCMSPreset(
        code="FORENSIC_BLOOD_ALCOHOLS",
        title_vi="Giám Định Pháp Y: Nồng Độ Cồn Trong Máu & Độc Chất (Headspace GC-MS)",
        carrier_gas="Helium",
        column_type="DB-ALC1 (30m × 0.32mm × 1.8μm)",
        t_init_c=40.0,
        ramp_rate_c_min=8.0,
        t_final_c=120.0,
        flow_rate_ml_min=1.5,
        compounds=(
            GCMSPeak("Methanol (Cồn độc)", "CH₄O", 1.8, 15.0, 32, 31, ((31, 100.0, "[M-H]⁺ CH₂OH⁺"), (29, 65.0, "CHO⁺"), (32, 42.0, "M⁺•"))),
            GCMSPeak("Ethanol (Cồn uống)", "C₂H₆O", 2.9, 65.0, 46, 31, ((31, 100.0, "CH₂OH⁺ α-cleavage"), (45, 48.0, "[M-H]⁺"), (29, 35.0, "C₂H₅⁺"), (46, 18.0, "M⁺•"))),
            GCMSPeak("Acetone", "C₃H₆O", 3.7, 20.0, 58, 43, ((43, 100.0, "CH₃CO⁺ acylium"), (58, 35.0, "M⁺•"), (15, 18.0, "CH₃⁺"))),
            GCMSPeak("Isopropanol", "C₃H₈O", 4.5, 25.0, 60, 45, ((45, 100.0, "CH₃CH=OH⁺"), (43, 22.0, "C₃H₇⁺"), (59, 10.0, "[M-H]⁺"), (60, 4.0, "M⁺•"))),
            GCMSPeak("n-Propanol (Chuẩn nội IS)", "C₃H₈O", 5.6, 50.0, 60, 31, ((31, 100.0, "CH₂OH⁺"), (59, 25.0, "[M-H]⁺"), (42, 18.0, "C₃H₆⁺•"), (60, 8.0, "M⁺•"))),
        ),
        description_vi="Quy trình chuẩn trong giám định vi phạm nồng độ cồn và ngộ độc rượu giả methanol bằng kỹ thuật tiêm mẫu không gian hơi (Headspace GC-MS).",
        formula_vi="C_ethanol = (A_EtOH / A_IS) · (C_IS / RF)  |  Độ phân giải R_s ≥ 2.0",
    ),
    "ENVIRONMENTAL_POPS_PCBS": GCMSPreset(
        code="ENVIRONMENTAL_POPS_PCBS",
        title_vi="Độc Chất Môi Trường: Hợp Chất Clo Hóa POPs & PCB (Chùm Vạch Đồng Vị Cl)",
        carrier_gas="Helium",
        column_type="DB-5MS (60m × 0.25mm × 0.25μm)",
        t_init_c=100.0,
        ramp_rate_c_min=15.0,
        t_final_c=300.0,
        flow_rate_ml_min=1.0,
        compounds=(
            GCMSPeak("1,4-Dichlorobenzene", "C₆H₄Cl₂", 4.2, 25.0, 146, 146, ((146, 100.0, "M⁺• (³⁵Cl₂)"), (148, 64.0, "[M+2]⁺• (³⁵Cl³⁷Cl)"), (111, 42.0, "[M-Cl]⁺"), (75, 25.0, "C₆H₃⁺"))),
            GCMSPeak("Lindane (γ-HCH)", "C₆H₆Cl₆", 8.6, 32.0, 290, 181, ((181, 100.0, "C₆H₆Cl₃⁺"), (183, 96.0, "[M+2]"), (217, 45.0, "C₆H₅Cl₄⁺"), (109, 38.0, "C₆H₅Cl⁺"), (290, 5.0, "M⁺•"))),
            GCMSPeak("PCB-28 (Trichlorobiphenyl)", "C₁₂H₇Cl₃", 11.5, 30.0, 256, 256, ((256, 100.0, "M⁺• (³⁵Cl₃)"), (258, 98.0, "[M+2]⁺• (³⁵Cl₂³⁷Cl)"), (260, 32.0, "[M+4]⁺•"), (186, 45.0, "[M-2Cl]⁺•"))),
            GCMSPeak("PCB-52 (Tetrachlorobiphenyl)", "C₁₂H₆Cl₄", 13.8, 35.0, 290, 292, ((292, 100.0, "[M+2]⁺•"), (290, 77.0, "M⁺• (³⁵Cl₄)"), (294, 48.0, "[M+4]⁺•"), (220, 35.0, "[M-2Cl]⁺•"))),
            GCMSPeak("p,p'-DDT", "C₁₄H₉Cl₅", 16.4, 40.0, 352, 235, ((235, 100.0, "[(p-ClC₆H₄)₂CH]⁺"), (237, 65.0, "[M+2]"), (165, 38.0, "Fluorene ion"), (178, 25.0, "C₁₄H₁₀⁺"), (352, 3.0, "M⁺•"))),
        ),
        description_vi="Xác định vết thuốc trừ sâu và hóa chất tích lũy sinh học POPs trong đất nước. Chùm vạch đồng vị Clo ³⁵Cl/³⁷Cl (3:1) là vân tay nhận diện đặc trưng.",
        formula_vi="Tỉ lệ chùm vạch Clo: (3a + b)ⁿ  (³⁵Cl : ³⁷Cl ≈ 3 : 1)  |  SIM Mode quan trắc vết",
    ),
    "DOPING_ANABOLIC_STEROIDS": GCMSPreset(
        code="DOPING_ANABOLIC_STEROIDS",
        title_vi="Kiểm Tra Doping Thể Thao: Steroid Đồng Hóa (WADA Screening)",
        carrier_gas="Helium",
        column_type="HP-1MS (25m × 0.20mm × 0.11μm)",
        t_init_c=150.0,
        ramp_rate_c_min=12.0,
        t_final_c=310.0,
        flow_rate_ml_min=1.0,
        compounds=(
            GCMSPeak("Androsterone (Deri-TMS)", "C₁₉H₃₀O₂-(TMS)₂", 7.8, 30.0, 434, 269, ((269, 100.0, "[M-TMSO-TMS]⁺"), (434, 45.0, "M⁺•"), (73, 85.0, "Si(CH₃)₃⁺"), (344, 25.0, "[M-HOTMS]⁺•"))),
            GCMSPeak("Etiocholanolone (Deri-TMS)", "C₁₉H₃₀O₂-(TMS)₂", 8.2, 28.0, 434, 269, ((269, 100.0, "[M-TMSO-TMS]⁺"), (73, 90.0, "Si(CH₃)₃⁺"), (434, 40.0, "M⁺•"), (344, 20.0, "[M-HOTMS]⁺•"))),
            GCMSPeak("Epitestosterone-bis-TMS", "C₁₉H₂₈O₂-(TMS)₂", 9.5, 12.0, 432, 432, ((432, 100.0, "M⁺• steroid ring"), (73, 95.0, "Si(CH₃)₃⁺ TMS"), (209, 45.0, "A-B ring fragment"), (327, 28.0, "[M-105]⁺"))),
            GCMSPeak("Testosterone-bis-TMS", "C₁₉H₂₈O₂-(TMS)₂", 10.1, 48.0, 432, 432, ((432, 100.0, "M⁺• steroid ring"), (73, 95.0, "Si(CH₃)₃⁺ TMS"), (209, 50.0, "A-B ring fragment"), (327, 30.0, "[M-105]⁺"))),
            GCMSPeak("19-Nortestosterone (Nandrolone)", "C₁₈H₂₆O₂-(TMS)₂", 11.8, 15.0, 418, 418, ((418, 100.0, "M⁺•"), (73, 88.0, "Si(CH₃)₃⁺"), (240, 40.0, "Ring fragment"), (313, 22.0, "[M-105]⁺"))),
        ),
        description_vi="Phương pháp tiêu chuẩn của Cơ quan Phòng chống Doping Thế giới (WADA). Dẫn xuất hóa TMS để tăng độ bay hơi, phân tích tỉ số T/E (ngưỡng gian lận > 4.0).",
        formula_vi="Tỉ số T/E = Area(Testosterone) / Area(Epitestosterone) ≤ 4.0 (Ngưỡng sinh học an toàn)",
    ),
    "PETROCHEMICAL_GASOLINE_BTEX": GCMSPreset(
        code="PETROCHEMICAL_GASOLINE_BTEX",
        title_vi="Hóa Dầu & Xăng Dầu: Phân Tích Hydrocarbon Thơm BTEX",
        carrier_gas="Nitrogen",
        column_type="DB-WAX (30m × 0.25mm × 0.25μm)",
        t_init_c=45.0,
        ramp_rate_c_min=6.0,
        t_final_c=180.0,
        flow_rate_ml_min=1.1,
        compounds=(
            GCMSPeak("Benzene (C₆H₆)", "C₆H₆", 2.5, 30.0, 78, 78, ((78, 100.0, "M⁺• thơm bền vững"), (77, 25.0, "[M-H]⁺ phenyl"), (52, 22.0, "C₄H₄⁺•"), (39, 15.0, "C₃H₃⁺"))),
            GCMSPeak("Toluene (C₇H₈)", "C₇H₈", 4.1, 55.0, 92, 91, ((91, 100.0, "C₇H₇⁺ Tropylium thơm bền"), (92, 75.0, "M⁺•"), (65, 20.0, "C₅H₅⁺ cyclopentadienyl"), (39, 12.0, "C₃H₃⁺"))),
            GCMSPeak("Ethylbenzene (C₈H₁₀)", "C₈H₁₀", 6.2, 25.0, 106, 91, ((91, 100.0, "C₇H₇⁺ Tropylium (M-CH₃)"), (106, 35.0, "M⁺•"), (65, 15.0, "C₅H₅⁺"), (77, 10.0, "C₆H₅⁺"))),
            GCMSPeak("m/p-Xylene (C₈H₁₀)", "C₈H₁₀", 6.9, 60.0, 106, 91, ((91, 100.0, "CH₃-C₆H₄-CH₂⁺"), (106, 65.0, "M⁺•"), (105, 30.0, "[M-H]⁺"), (77, 12.0, "C₆H₅⁺"))),
            GCMSPeak("o-Xylene (C₈H₁₀)", "C₈H₁₀", 7.6, 28.0, 106, 91, ((91, 100.0, "C₇H₇⁺ Tropylium"), (106, 68.0, "M⁺•"), (105, 32.0, "[M-H]⁺"), (77, 14.0, "C₆H₅⁺"))),
        ),
        description_vi="Đánh giá hàm lượng hợp chất thơm gây ô nhiễm trong xăng dầu và khí thải đô thị. Tropylium m/z 91 là mảnh ion vòng 7 cạnh thơm đặc trưng cho gốc benzyl.",
        formula_vi="C₇H₈ + e⁻ → C₇H₇⁺ (Tropylium, m/z 91, 100% Base Peak) + H• + 2e⁻",
    ),
}


class InteractiveGCMSItem(QGraphicsObject):
    """
    Interactive PySide6 QGraphicsItem simulating GC-MS Analytical Workstation
    with real-time Gas Chromatogram (TIC) and EI-MS 70eV Fragmentation Spectrum.
    """

    WIDTH = 740.0
    HEIGHT = 510.0

    preset_changed = Signal(str)
    peak_selected = Signal(str)
    closed = Signal()

    def __init__(self, pos: Union[tuple[float, float], QPointF] = (200.0, 80.0)) -> None:
        super().__init__()
        if isinstance(pos, QPointF):
            self.setPos(pos)
        elif isinstance(pos, (tuple, list)) and len(pos) >= 2:
            self.setPos(float(pos[0]), float(pos[1]))
        else:
            self.setPos(200.0, 80.0)

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(25.0)

        # Active Preset and Physical State
        self._active_preset_code: str = "ESSENTIAL_OIL_TERPENES"
        self._preset: GCMSPreset = PRESETS[self._active_preset_code]

        self._ramp_rate: float = self._preset.ramp_rate_c_min
        self._flow_rate: float = self._preset.flow_rate_ml_min
        self._oven_temp_c: float = self._preset.t_init_c

        # Active Selected Compound Index for MS stick spectrum
        self._selected_peak_idx: int = 1  # Default to 2nd peak (e.g. Limonene)

        # Simulation & Animation State
        self._sim_time_min: float = 0.0
        self._is_animating: bool = True
        self._injection_slug_x: float = 0.0

        # Animation timer (~30 FPS)
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start()

        # UI interaction buttons mapping
        self._hovered_button: Optional[str] = None
        self._btn_rects: dict[str, QRectF] = {}
        self._peak_hit_rects: list[tuple[QRectF, int]] = []
        self._init_button_rects()

    def _init_button_rects(self) -> None:
        """Initialize click and air-pinch bounding boxes."""
        w, h = self.WIDTH, self.HEIGHT
        self._btn_rects.clear()

        # Header close button
        self._btn_rects["close"] = QRectF(w - 36.0, 10.0, 26.0, 26.0)

        # 5 Preset Buttons (top toolbar)
        p_keys = list(PRESETS.keys())
        p_w = (w - 28.0) / len(p_keys)
        for i, k in enumerate(p_keys):
            self._btn_rects[f"preset_{k}"] = QRectF(14.0 + i * p_w, 44.0, p_w - 4.0, 24.0)

        # Bottom toolbar buttons
        y_bot = h - 42.0
        self._btn_rects["play_pause"] = QRectF(16.0, y_bot, 95.0, 30.0)
        self._btn_rects["dec_ramp"] = QRectF(118.0, y_bot, 32.0, 30.0)
        self._btn_rects["inc_ramp"] = QRectF(152.0, y_bot, 32.0, 30.0)
        self._btn_rects["dec_flow"] = QRectF(258.0, y_bot, 32.0, 30.0)
        self._btn_rects["inc_flow"] = QRectF(292.0, y_bot, 32.0, 30.0)
        self._btn_rects["reset_sim"] = QRectF(398.0, y_bot, 85.0, 30.0)
        self._btn_rects["stamp"] = QRectF(w - 128.0, y_bot, 112.0, 30.0)

    # -------------------------------------------------------------------------
    # Public Properties
    # -------------------------------------------------------------------------
    @property
    def active_preset_code(self) -> str:
        return self._active_preset_code

    @property
    def preset(self) -> GCMSPreset:
        return self._preset

    @property
    def ramp_rate(self) -> float:
        return self._ramp_rate

    @property
    def flow_rate(self) -> float:
        return self._flow_rate

    @property
    def oven_temp_c(self) -> float:
        return self._oven_temp_c

    @property
    def selected_compound(self) -> GCMSPeak:
        idx = max(0, min(len(self._preset.compounds) - 1, self._selected_peak_idx))
        return self._preset.compounds[idx]

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    # -------------------------------------------------------------------------
    # Public Controller Actions
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> bool:
        """Load an authentic analytical chemistry GC-MS preset."""
        if preset_code not in PRESETS:
            return False

        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._ramp_rate = self._preset.ramp_rate_c_min
        self._flow_rate = self._preset.flow_rate_ml_min
        self._oven_temp_c = self._preset.t_init_c
        self._selected_peak_idx = 0
        self._sim_time_min = 0.0
        self.preset_changed.emit(preset_code)
        self.update()
        return True

    def select_peak(self, index: int) -> bool:
        """Select chromatographic peak to inspect its EI-MS spectrum."""
        if 0 <= index < len(self._preset.compounds):
            self._selected_peak_idx = index
            self.peak_selected.emit(self.selected_compound.name_vi)
            self.update()
            return True
        return False

    def set_ramp_rate(self, ramp_c_min: float) -> None:
        """Set GC oven heating rate (clamped between 2.0 and 30.0 °C/min)."""
        self._ramp_rate = max(2.0, min(30.0, float(ramp_c_min)))
        self.update()

    def set_flow_rate(self, flow_ml_min: float) -> None:
        """Set carrier gas flow rate (clamped between 0.5 and 4.0 mL/min)."""
        self._flow_rate = max(0.5, min(4.0, float(flow_ml_min)))
        self.update()

    def reset_run(self) -> None:
        """Reset chromatographic run to t = 0 min."""
        self._sim_time_min = 0.0
        self._oven_temp_c = self._preset.t_init_c
        self.update()

    def toggle_animation(self) -> bool:
        """Toggle live chromatography progression."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def stamp_to_board(self) -> None:
        """Stamps active GC-MS report and mass spectrum to LessonBoard scene."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import ChemistryTools

        center = (self.pos().x() + 80.0, self.pos().y() + self.HEIGHT + 40.0)
        items = ChemistryTools.create_gcms_diagram(
            center=center,
            preset_code=self._active_preset_code,
            selected_peak_idx=self._selected_peak_idx,
            ramp_rate=self._ramp_rate,
            flow_rate=self._flow_rate,
        )
        for itm in items:
            scene.addItem(itm)

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air pinch gesture handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Check peak clicks on chromatogram
        for rect, idx in self._peak_hit_rects:
            if rect.contains(local_pos):
                self.select_peak(idx)
                return True

        # 2. Preset buttons
        for k in PRESETS:
            r = self._btn_rects.get(f"preset_{k}")
            if r and r.contains(local_pos):
                self.load_preset(k)
                return True

        # 3. Control deck buttons
        if self._btn_rects["close"].contains(local_pos):
            self.closed.emit()
            self.setVisible(False)
            return True

        if self._btn_rects["play_pause"].contains(local_pos):
            self.toggle_animation()
            return True

        if self._btn_rects["dec_ramp"].contains(local_pos):
            self.set_ramp_rate(self._ramp_rate - 2.0)
            return True
        if self._btn_rects["inc_ramp"].contains(local_pos):
            self.set_ramp_rate(self._ramp_rate + 2.0)
            return True

        if self._btn_rects["dec_flow"].contains(local_pos):
            self.set_flow_rate(self._flow_rate - 0.2)
            return True
        if self._btn_rects["inc_flow"].contains(local_pos):
            self.set_flow_rate(self._flow_rate + 0.2)
            return True

        if self._btn_rects["reset_sim"].contains(local_pos):
            self.reset_run()
            return True

        if self._btn_rects["stamp"].contains(local_pos):
            self.stamp_to_board()
            return True

        return True

    # -------------------------------------------------------------------------
    # Animation Tick
    # -------------------------------------------------------------------------
    def _on_animation_tick(self) -> None:
        """Advance chromatographic retention progression and oven temperature."""
        if not self._is_animating:
            return

        dt_min = 0.035 * (self._flow_rate / 1.0)
        self._sim_time_min += dt_min

        # Oven temperature heating
        t_target = min(self._preset.t_final_c, self._preset.t_init_c + self._ramp_rate * self._sim_time_min)
        self._oven_temp_c = t_target

        # Max run time (auto loop after completion)
        max_tr = max(p.retention_time_min for p in self._preset.compounds) + 3.0
        if self._sim_time_min > max_tr:
            self._sim_time_min = 0.0

        self.update()

    # -------------------------------------------------------------------------
    # Rendering & Painting
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#080d1a")))
        pen_border = QPen(QColor("#0ea5e9"), 2.0) if self.isSelected() else QPen(QColor("#1e293b"), 1.6)
        painter.setPen(pen_border)
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

        # 2. Header Bar
        self._paint_header(painter)

        # 3. Preset Selector Buttons
        self._paint_preset_toolbar(painter)

        # 4. Upper Viewport: Gas Chromatogram (TIC - Sắc ký đồ)
        self._paint_chromatogram_viewport(painter)

        # 5. Lower Viewport: EI-MS 70eV Mass Spectrum (Khối phổ vạch)
        self._paint_mass_spectrum_viewport(painter)

        # 6. Bottom Inspector Deck & Controls
        self._paint_bottom_toolbar(painter)

    def _paint_header(self, painter: QPainter) -> None:
        """Header banner and close button."""
        badge_rect = QRectF(14.0, 12.0, 24.0, 24.0)
        painter.setBrush(QBrush(QColor("#0ea5e9")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(badge_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "🔬")

        # Title
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(46.0, 29.0, "TRẠM PHÂN TÍCH SẮC KÝ KHÍ - KHỐI PHỔ (GC-MS 70 eV WORKSTATION)")

        # Subtitle
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(490.0, 29.0, f"Cột: {self._preset.column_type[:18]}… | Khí: {self._preset.carrier_gas}")

        # Close button
        r_cl = self._btn_rects["close"]
        hov = (self._hovered_button == "close")
        painter.setBrush(QBrush(QColor("#dc2626") if hov else QColor("#334155")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(r_cl, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_cl, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_preset_toolbar(self, painter: QPainter) -> None:
        """Preset buttons with active indicators."""
        labels = {
            "ESSENTIAL_OIL_TERPENES": "🌿 Tinh Dầu Terpenes",
            "FORENSIC_BLOOD_ALCOHOLS": "🩸 Pháp Y Cồn Máu",
            "ENVIRONMENTAL_POPS_PCBS": "☣️ Độc Chất Môi Trường",
            "DOPING_ANABOLIC_STEROIDS": "🏅 WADA Doping Steroid",
            "PETROCHEMICAL_GASOLINE_BTEX": "⛽ Xăng Dầu & BTEX",
        }
        for code, r in self._btn_rects.items():
            if not code.startswith("preset_"):
                continue
            pk = code.replace("preset_", "")
            is_act = (pk == self._active_preset_code)
            is_hov = (self._hovered_button == code)

            col_bg = QColor("#0369a1") if is_act else (QColor("#1e293b") if is_hov else QColor("#0f172a"))
            col_border = QColor("#38bdf8") if is_act else QColor("#334155")
            painter.setBrush(QBrush(col_bg))
            painter.setPen(QPen(col_border, 1.2 if is_act else 0.8))
            painter.drawRoundedRect(r, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_act else QFont.Weight.Normal))
            painter.setPen(QColor("#7dd3fc") if is_act else QColor("#cbd5e1"))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, labels.get(pk, pk))

    def _paint_chromatogram_viewport(self, painter: QPainter) -> None:
        """Upper viewport: Total Ion Current (TIC) Chromatogram with interactive peak buttons."""
        vp_rect = QRectF(14.0, 76.0, 420.0, 205.0)
        painter.setBrush(QBrush(QColor("#030712")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(vp_rect, 8.0, 8.0)

        # Labels
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(24.0, 94.0, "SẮC KÝ ĐỒ TỔNG ION (TIC CHROMATOGRAM)")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(260.0, 94.0, f"Lò sấy: {self._oven_temp_c:.1f}°C  |  t = {self._sim_time_min:.2f} phút")

        # Chromatogram Canvas coordinates
        c_x0 = 45.0
        c_y0 = 255.0
        c_w = 370.0
        c_h = 145.0

        # Baseline & Axes
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawLine(int(c_x0), int(c_y0), int(c_x0 + c_w), int(c_y0))
        painter.drawLine(int(c_x0), int(c_y0), int(c_x0), int(c_y0 - c_h))

        # Time ticks
        max_t = max(p.retention_time_min for p in self._preset.compounds) + 2.0
        scale_x = c_w / max_t
        for t_mark in range(0, int(max_t) + 1, 2):
            tx = c_x0 + t_mark * scale_x
            painter.drawLine(int(tx), int(c_y0), int(tx), int(c_y0 + 3))
            painter.drawText(int(tx - 6), int(c_y0 + 14), f"{t_mark}m")

        # Plot Gaussian Peaks
        self._peak_hit_rects.clear()
        tic_path = QPainterPath()
        tic_path.moveTo(c_x0, c_y0)

        pts_count = 150
        for step in range(pts_count + 1):
            t_curr = (step / pts_count) * max_t
            # Sum of Gaussians
            intensity = 0.0
            for p in self._preset.compounds:
                # Flow rate adjusts actual retention time: t_R ~ 1 / flow
                actual_tr = p.retention_time_min * (1.2 / self._flow_rate) * (10.0 / self._ramp_rate) ** 0.15
                sigma = 0.15 + (actual_tr * 0.02)
                intensity += p.peak_area * math.exp(-((t_curr - actual_tr) ** 2) / (2.0 * (sigma ** 2)))

            px = c_x0 + t_curr * scale_x
            py = c_y0 - min(c_h - 10.0, intensity * 2.2)
            tic_path.lineTo(px, py)

        # Draw TIC curve
        painter.setPen(QPen(QColor("#0ea5e9"), 1.8))
        painter.drawPath(tic_path)

        # Draw Peak Markers & Hit Rectangles
        for idx, p in enumerate(self._preset.compounds):
            actual_tr = p.retention_time_min * (1.2 / self._flow_rate) * (10.0 / self._ramp_rate) ** 0.15
            peak_px = c_x0 + actual_tr * scale_x
            peak_py = c_y0 - min(c_h - 10.0, p.peak_area * 2.2)

            is_sel = (idx == self._selected_peak_idx)
            # Marker dot
            painter.setBrush(QBrush(QColor("#f59e0b") if is_sel else QColor("#0284c7")))
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            painter.drawEllipse(QPointF(peak_px, peak_py), 4.5 if is_sel else 3.5, 4.5 if is_sel else 3.5)

            # Hit rect for air pinch / mouse click
            hit_r = QRectF(peak_px - 14.0, peak_py - 24.0, 28.0, 36.0)
            self._peak_hit_rects.append((hit_r, idx))

            # Peak label
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_sel else QFont.Weight.Normal))
            painter.setPen(QColor("#fbbf24") if is_sel else QColor("#94a3b8"))
            painter.drawText(QRectF(peak_px - 25.0, peak_py - 18.0, 50.0, 14.0), Qt.AlignmentFlag.AlignCenter, f"#{idx+1}")

        # Current Chromatographic Run Time Indicator
        cur_x = c_x0 + min(self._sim_time_min, max_t) * scale_x
        painter.setPen(QPen(QColor("#f43f5e"), 1.2, Qt.PenStyle.DashLine))
        painter.drawLine(int(cur_x), int(c_y0 - c_h + 10), int(cur_x), int(c_y0))

    def _paint_mass_spectrum_viewport(self, painter: QPainter) -> None:
        """Right / Lower Viewport: EI-MS 70eV Stick Mass Spectrum for the selected compound."""
        vp_rect = QRectF(446.0, 76.0, 280.0, 205.0)
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(vp_rect, 8.0, 8.0)

        sel_cmp = self.selected_compound

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(456.0, 94.0, f"KHỐI PHỔ EI-MS: {sel_cmp.name_vi.upper()}")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(456.0, 108.0, f"CTPT: {sel_cmp.formula} | M⁺• = {sel_cmp.molecular_weight} Da | Base = {sel_cmp.base_peak_mz}")

        # Mass Spectrum Canvas
        ms_x0 = 480.0
        ms_y0 = 255.0
        ms_w = 230.0
        ms_h = 135.0

        # Baseline & Axes
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawLine(int(ms_x0), int(ms_y0), int(ms_x0 + ms_w), int(ms_y0))
        painter.drawLine(int(ms_x0), int(ms_y0), int(ms_x0), int(ms_y0 - ms_h))

        # Y-axis ticks (% relative intensity: 0, 50, 100%)
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QColor("#64748b"))
        painter.drawText(int(ms_x0 - 24), int(ms_y0 - ms_h + 8), "100%")
        painter.drawText(int(ms_x0 - 20), int(ms_y0 - ms_h / 2 + 4), "50%")
        painter.drawText(int(ms_x0 - 15), int(ms_y0 + 2), "0%")

        # Find max m/z for scaling
        max_mz = max(sel_cmp.molecular_weight * 1.12, 100.0)
        scale_mz = ms_w / max_mz

        # Draw Mass Sticks
        for mz, rel_pct, frag_name in sel_cmp.fragments:
            sx = ms_x0 + mz * scale_mz
            sy = ms_y0 - (rel_pct / 100.0) * (ms_h - 15.0)

            is_base = (mz == sel_cmp.base_peak_mz)
            is_parent = (mz == sel_cmp.molecular_weight)

            col = QColor("#f43f5e") if is_base else (QColor("#38bdf8") if is_parent else QColor("#10b981"))
            painter.setPen(QPen(col, 2.0 if (is_base or is_parent) else 1.4))
            painter.drawLine(int(sx), int(ms_y0), int(sx), int(sy))

            # Peak label (m/z value)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold if is_base else QFont.Weight.Normal))
            painter.setPen(col)
            painter.drawText(int(sx - 10), int(sy - 3), str(mz))

    def _paint_bottom_toolbar(self, painter: QPainter) -> None:
        """Bottom Inspector deck: Compound theoretical details, formula, and stepper controls."""
        y_hud = self.HEIGHT - 130.0
        hud_rect = QRectF(14.0, y_hud, self.WIDTH - 28.0, 80.0)
        painter.setBrush(QBrush(QColor("#050b14")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(hud_rect, 8.0, 8.0)

        sel = self.selected_compound
        actual_tr = sel.retention_time_min * (1.2 / self._flow_rate) * (10.0 / self._ramp_rate) ** 0.15

        # Compound Details Grid
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(24.0, y_hud + 18.0, f"Đang chọn chất #{self._selected_peak_idx + 1}: {sel.name_vi} ({sel.formula})")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(24.0, y_hud + 36.0, f"Thời gian lưu t_R: {actual_tr:.2f} phút  |  Diện tích đỉnh: {sel.peak_area:.1f}%  |  M⁺•: {sel.molecular_weight} Da  |  Base: {sel.base_peak_mz}")

        # Fragment breakdown
        frag_str = "  ".join([f"{mz} ({pct:.0f}%): {ident}" for mz, pct, ident in sel.fragments[:3]])
        painter.setPen(QColor("#10b981"))
        painter.drawText(24.0, y_hud + 54.0, f"Mảnh ion chính: {frag_str}")

        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(24.0, y_hud + 70.0, f"Công thức & Lý thuyết: {self._preset.formula_vi}")

        # Bottom Buttons
        y_bot = self.HEIGHT - 42.0

        # Play / Pause
        r_play = self._btn_rects["play_pause"]
        is_hov_play = (self._hovered_button == "play_pause")
        painter.setBrush(QBrush(QColor("#0284c7") if is_hov_play else QColor("#0369a1")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(r_play, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        txt_play = "⏸ Tạm Dừng" if self._is_animating else "▶ Tiếp Tục"
        painter.drawText(r_play, Qt.AlignmentFlag.AlignCenter, txt_play)

        # Ramp Rate Stepper
        self._paint_stepper(painter, "dec_ramp", "inc_ramp", f"Tốc độ gia nhiệt: {self._ramp_rate:.0f}°C/min", 118.0)

        # Flow Rate Stepper
        self._paint_stepper(painter, "dec_flow", "inc_flow", f"Lưu lượng khí He: {self._flow_rate:.1f} mL/min", 258.0)

        # Reset button
        r_rst = self._btn_rects["reset_sim"]
        is_hov_rst = (self._hovered_button == "reset_sim")
        painter.setBrush(QBrush(QColor("#334155") if is_hov_rst else QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_rst, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_rst, Qt.AlignmentFlag.AlignCenter, "🔄 Đặt Lại")

        # Stamp Button
        r_stamp = self._btn_rects["stamp"]
        is_hov_st = (self._hovered_button == "stamp")
        painter.setBrush(QBrush(QColor("#059669") if is_hov_st else QColor("#047857")))
        painter.setPen(QPen(QColor("#34d399"), 1.2))
        painter.drawRoundedRect(r_stamp, 5.0, 5.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng Dấu Bảng")

    def _paint_stepper(self, painter: QPainter, btn_dec: str, btn_inc: str, label: str, x_start: float) -> None:
        """Helper to render decrement / increment stepper buttons."""
        r_dec = self._btn_rects[btn_dec]
        r_inc = self._btn_rects[btn_inc]

        hov_d = (self._hovered_button == btn_dec)
        painter.setBrush(QBrush(QColor("#334155") if hov_d else QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_dec, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_dec, Qt.AlignmentFlag.AlignCenter, "−")

        hov_i = (self._hovered_button == btn_inc)
        painter.setBrush(QBrush(QColor("#334155") if hov_i else QColor("#1e293b")))
        painter.drawRoundedRect(r_inc, 4.0, 4.0)
        painter.drawText(r_inc, Qt.AlignmentFlag.AlignCenter, "+")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(int(x_start), int(r_dec.top() - 4.0), label)

    # -------------------------------------------------------------------------
    # Mouse & Touch Event Handlers
    # -------------------------------------------------------------------------
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check peak click on chromatogram
        for rect, idx in self._peak_hit_rects:
            if rect.contains(pos):
                self.select_peak(idx)
                event.accept()
                return

        for key, r in self._btn_rects.items():
            if r.contains(pos):
                if key == "close":
                    self.closed.emit()
                    self.setVisible(False)
                    event.accept()
                    return
                elif key.startswith("preset_"):
                    self.load_preset(key.replace("preset_", ""))
                    event.accept()
                    return
                elif key == "play_pause":
                    self.toggle_animation()
                    event.accept()
                    return
                elif key == "dec_ramp":
                    self.set_ramp_rate(self._ramp_rate - 2.0)
                    event.accept()
                    return
                elif key == "inc_ramp":
                    self.set_ramp_rate(self._ramp_rate + 2.0)
                    event.accept()
                    return
                elif key == "dec_flow":
                    self.set_flow_rate(self._flow_rate - 0.2)
                    event.accept()
                    return
                elif key == "inc_flow":
                    self.set_flow_rate(self._flow_rate + 0.2)
                    event.accept()
                    return
                elif key == "reset_sim":
                    self.reset_run()
                    event.accept()
                    return
                elif key == "stamp":
                    self.stamp_to_board()
                    event.accept()
                    return
        super().mousePressEvent(event)

    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        hovered = None
        for key, r in self._btn_rects.items():
            if r.contains(pos):
                hovered = key
                break
        if hovered != self._hovered_button:
            self._hovered_button = hovered
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_button is not None:
            self._hovered_button = None
            self.update()
        super().hoverLeaveEvent(event)

    def cleanup(self) -> None:
        """Stops background animation timer and frees resources."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()
