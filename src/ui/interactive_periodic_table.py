"""
AirTeach AI — Interactive Periodic Table of Elements & Bohr Shell / Electron Configuration Visualizer
Authentic chemical database, Aufbau subshell orbital filling, Bohr electron shells,
Pauling electronegativity trends, and touchless air gesture integration.
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
    QStyleOptionGraphicsItem,
    QWidget,
)


@dataclass(frozen=True)
class ElementData:
    z: int
    symbol: str
    name_vi: str
    name_en: str
    atomic_mass: float
    period: int
    group: int
    category: str
    electron_config: str
    bohr_shells: tuple[int, ...]
    electronegativity: Optional[float]
    ionization_energy_kj: float
    atomic_radius_pm: int
    cpk_hex: str


# Authentic IUPAC chemical data for elements Z=1 to 36 + key heavy elements
ELEMENT_DB: dict[int, ElementData] = {
    1: ElementData(1, "H", "Hiđrô", "Hydrogen", 1.008, 1, 1, "Phi kim phản ứng", "1s¹", (1,), 2.20, 1312.0, 53, "#ffffff"),
    2: ElementData(2, "He", "Heli", "Helium", 4.003, 1, 18, "Khí hiếm", "1s²", (2,), None, 2372.3, 31, "#d9ffff"),
    3: ElementData(3, "Li", "Liti", "Lithium", 6.94, 2, 1, "Kim loại kiềm", "[He] 2s¹", (2, 1), 0.98, 520.2, 167, "#cc80ff"),
    4: ElementData(4, "Be", "Beri", "Beryllium", 9.012, 2, 2, "Kim loại kiềm thổ", "[He] 2s²", (2, 2), 1.57, 899.5, 112, "#c2ff00"),
    5: ElementData(5, "B", "Bo", "Boron", 10.81, 2, 13, "Á kim", "[He] 2s² 2p¹", (2, 3), 2.04, 800.6, 87, "#ffb5b5"),
    6: ElementData(6, "C", "Cacbon", "Carbon", 12.011, 2, 14, "Phi kim phản ứng", "[He] 2s² 2p²", (2, 4), 2.55, 1086.5, 67, "#909090"),
    7: ElementData(7, "N", "Nitơ", "Nitrogen", 14.007, 2, 15, "Phi kim phản ứng", "[He] 2s² 2p³", (2, 5), 3.04, 1402.3, 56, "#3050f8"),
    8: ElementData(8, "O", "Oxy", "Oxygen", 15.999, 2, 16, "Phi kim phản ứng", "[He] 2s² 2p⁴", (2, 6), 3.44, 1313.9, 48, "#ff0d0d"),
    9: ElementData(9, "F", "Flo", "Fluorine", 18.998, 2, 17, "Halogen", "[He] 2s² 2p⁵", (2, 7), 3.98, 1681.0, 42, "#90e050"),
    10: ElementData(10, "Ne", "Neon", "Neon", 20.180, 2, 18, "Khí hiếm", "[He] 2s² 2p⁶", (2, 8), None, 2080.7, 38, "#b3e3f5"),
    11: ElementData(11, "Na", "Natri", "Sodium", 22.990, 3, 1, "Kim loại kiềm", "[Ne] 3s¹", (2, 8, 1), 0.93, 495.8, 190, "#ab5cf2"),
    12: ElementData(12, "Mg", "Magie", "Magnesium", 24.305, 3, 2, "Kim loại kiềm thổ", "[Ne] 3s²", (2, 8, 2), 1.31, 737.7, 145, "#8aff00"),
    13: ElementData(13, "Al", "Nhôm", "Aluminium", 26.982, 3, 13, "Kim loại hậu chuyển tiếp", "[Ne] 3s² 3p¹", (2, 8, 3), 1.61, 577.5, 118, "#bfa6a6"),
    14: ElementData(14, "Si", "Silic", "Silicon", 28.085, 3, 14, "Á kim", "[Ne] 3s² 3p²", (2, 8, 4), 1.90, 786.5, 111, "#f0c8a0"),
    15: ElementData(15, "P", "Photpho", "Phosphorus", 30.974, 3, 15, "Phi kim phản ứng", "[Ne] 3s² 3p³", (2, 8, 5), 2.19, 1011.8, 98, "#ff8000"),
    16: ElementData(16, "S", "Lưu huỳnh", "Sulfur", 32.06, 3, 16, "Phi kim phản ứng", "[Ne] 3s² 3p⁴", (2, 8, 6), 2.58, 999.6, 88, "#ffff30"),
    17: ElementData(17, "Cl", "Clo", "Chlorine", 35.45, 3, 17, "Halogen", "[Ne] 3s² 3p⁵", (2, 8, 7), 3.16, 1251.2, 79, "#1ff01f"),
    18: ElementData(18, "Ar", "Argon", "Argon", 39.95, 3, 18, "Khí hiếm", "[Ne] 3s² 3p⁶", (2, 8, 8), None, 1520.6, 71, "#80d1e3"),
    19: ElementData(19, "K", "Kali", "Potassium", 39.098, 4, 1, "Kim loại kiềm", "[Ar] 4s¹", (2, 8, 8, 1), 0.82, 418.8, 243, "#8f40d4"),
    20: ElementData(20, "Ca", "Canxi", "Calcium", 40.078, 4, 2, "Kim loại kiềm thổ", "[Ar] 4s²", (2, 8, 8, 2), 1.00, 589.8, 194, "#3dff00"),
    21: ElementData(21, "Sc", "Scandi", "Scandium", 44.956, 4, 3, "Kim loại chuyển tiếp", "[Ar] 3d¹ 4s²", (2, 8, 9, 2), 1.36, 633.1, 184, "#e6e6e6"),
    22: ElementData(22, "Ti", "Titan", "Titanium", 47.867, 4, 4, "Kim loại chuyển tiếp", "[Ar] 3d² 4s²", (2, 8, 10, 2), 1.54, 658.8, 176, "#bfc2c7"),
    23: ElementData(23, "V", "Vanadi", "Vanadium", 50.942, 4, 5, "Kim loại chuyển tiếp", "[Ar] 3d³ 4s²", (2, 8, 11, 2), 1.63, 650.9, 171, "#a6a6ab"),
    24: ElementData(24, "Cr", "Crom", "Chromium", 51.996, 4, 6, "Kim loại chuyển tiếp", "[Ar] 3d⁵ 4s¹", (2, 8, 13, 1), 1.66, 652.9, 166, "#8a99c7"),
    25: ElementData(25, "Mn", "Mangan", "Manganese", 54.938, 4, 7, "Kim loại chuyển tiếp", "[Ar] 3d⁵ 4s²", (2, 8, 13, 2), 1.55, 717.3, 161, "#9c7ac7"),
    26: ElementData(26, "Fe", "Sắt", "Iron", 55.845, 4, 8, "Kim loại chuyển tiếp", "[Ar] 3d⁶ 4s²", (2, 8, 14, 2), 1.83, 762.5, 156, "#e06633"),
    27: ElementData(27, "Co", "Coban", "Cobalt", 58.933, 4, 9, "Kim loại chuyển tiếp", "[Ar] 3d⁷ 4s²", (2, 8, 15, 2), 1.88, 760.4, 152, "#f090a0"),
    28: ElementData(28, "Ni", "Niken", "Nickel", 58.693, 4, 10, "Kim loại chuyển tiếp", "[Ar] 3d⁸ 4s²", (2, 8, 16, 2), 1.91, 737.1, 149, "#50d050"),
    29: ElementData(29, "Cu", "Đồng", "Copper", 63.546, 4, 11, "Kim loại chuyển tiếp", "[Ar] 3d¹⁰ 4s¹", (2, 8, 18, 1), 1.90, 745.5, 145, "#c88033"),
    30: ElementData(30, "Zn", "Kẽm", "Zinc", 65.38, 4, 12, "Kim loại chuyển tiếp", "[Ar] 3d¹⁰ 4s²", (2, 8, 18, 2), 1.65, 906.4, 142, "#7d80b0"),
    31: ElementData(31, "Ga", "Gali", "Gallium", 69.723, 4, 13, "Kim loại hậu chuyển tiếp", "[Ar] 3d¹⁰ 4s² 4p¹", (2, 8, 18, 3), 1.81, 578.8, 136, "#c28f8f"),
    32: ElementData(32, "Ge", "Gecmani", "Germanium", 72.630, 4, 14, "Á kim", "[Ar] 3d¹⁰ 4s² 4p²", (2, 8, 18, 4), 2.01, 762.0, 125, "#668f8f"),
    33: ElementData(33, "As", "Asen", "Arsenic", 74.922, 4, 15, "Á kim", "[Ar] 3d¹⁰ 4s² 4p³", (2, 8, 18, 5), 2.18, 947.0, 114, "#bd80e3"),
    34: ElementData(34, "Se", "Selen", "Selenium", 78.971, 4, 16, "Phi kim phản ứng", "[Ar] 3d¹⁰ 4s² 4p⁴", (2, 8, 18, 6), 2.55, 941.0, 103, "#ffa100"),
    35: ElementData(35, "Br", "Brom", "Bromine", 79.904, 4, 17, "Halogen", "[Ar] 3d¹⁰ 4s² 4p⁵", (2, 8, 18, 7), 2.96, 1139.9, 94, "#a62929"),
    36: ElementData(36, "Kr", "Kripton", "Krypton", 83.798, 4, 18, "Khí hiếm", "[Ar] 3d¹⁰ 4s² 4p⁶", (2, 8, 18, 8), 3.00, 1350.8, 88, "#5cb8d1"),
}

# Group positions for mini periodic grid (Period 1 to 4)
PERIODIC_COORDS: dict[int, tuple[int, int]] = {
    # Period 1
    1: (0, 0), 2: (0, 17),
    # Period 2
    3: (1, 0), 4: (1, 1), 5: (1, 12), 6: (1, 13), 7: (1, 14), 8: (1, 15), 9: (1, 16), 10: (1, 17),
    # Period 3
    11: (2, 0), 12: (2, 1), 13: (2, 12), 14: (2, 13), 15: (2, 14), 16: (2, 15), 17: (2, 16), 18: (2, 17),
    # Period 4
    19: (3, 0), 20: (3, 1), 21: (3, 2), 22: (3, 3), 23: (3, 4), 24: (3, 5), 25: (3, 6), 26: (3, 7),
    27: (3, 8), 28: (3, 9), 29: (3, 10), 30: (3, 11), 31: (3, 12), 32: (3, 13), 33: (3, 14), 34: (3, 15),
    35: (3, 16), 36: (3, 17),
}


@dataclass(frozen=True)
class PeriodicPreset:
    code: str
    title_vi: str
    target_z: int
    view_mode: str  # STANDARD, ELECTRONEGATIVITY, BOHR_SHELLS
    description: str


PRESETS: dict[str, PeriodicPreset] = {
    "CHONPS_LIFE_ELEMENTS": PeriodicPreset(
        code="CHONPS_LIFE_ELEMENTS",
        title_vi="Nguyên tố sinh học (C-H-O-N-P-S)",
        target_z=6,  # Carbon
        view_mode="BOHR_SHELLS",
        description="6 nguyên tố cốt lõi chiếm hơn 98% khối lượng mọi sinh vật sống trên Trái Đất.",
    ),
    "ELECTRONEGATIVITY_TRENDS": PeriodicPreset(
        code="ELECTRONEGATIVITY_TRENDS",
        title_vi="Độ âm điện Pauling (Xu hướng tuần hoàn)",
        target_z=9,  # Fluorine (highest 3.98)
        view_mode="ELECTRONEGATIVITY",
        description="Độ âm điện tăng dần từ trái sang phải qua chu kỳ và giảm dần từ trên xuống dưới trong nhóm.",
    ),
    "NOBLE_GAS_OCTET_STABILITY": PeriodicPreset(
        code="NOBLE_GAS_OCTET_STABILITY",
        title_vi="Quy tắc Bát tử Khí hiếm (Octet Rule)",
        target_z=10,  # Neon
        view_mode="BOHR_SHELLS",
        description="Lớp vỏ ngoài cùng bão hòa 8 electron (s²p⁶) mang lại tính trơ hóa học bền vững tuyệt đối.",
    ),
    "TRANSITION_METALS_AUFBAU": PeriodicPreset(
        code="TRANSITION_METALS_AUFBAU",
        title_vi="Phân lớp 3d Kim loại chuyển tiếp (Cr/Cu)",
        target_z=24,  # Chromium (3d5 4s1 anomaly)
        view_mode="STANDARD",
        description="Quy tắc phân lớp d bán bão hòa (Cr: 3d⁵ 4s¹) và bão hòa (Cu: 3d¹⁰ 4s¹) phá vỡ trật tự Aufbau.",
    ),
    "ALKALI_VS_HALOGENS": PeriodicPreset(
        code="ALKALI_VS_HALOGENS",
        title_vi="Liên kết Ion: Kim loại kiềm & Halogen (NaCl)",
        target_z=11,  # Sodium
        view_mode="STANDARD",
        description="Natri (1e hóa trị) nhường electron cho Clo (7e hóa trị) tạo cặp ion Na⁺ và Cl⁻ bền vững.",
    ),
}


class InteractivePeriodicTableItem(QGraphicsObject):
    """
    Interactive QGraphicsObject for exploring the IUPAC Periodic Table of Elements,
    Aufbau subshell electron configurations, and Bohr orbital electron shells.
    """

    WIDTH = 640.0
    HEIGHT = 380.0

    stamp_requested = Signal()
    element_selected = Signal(int)  # Atomic number Z

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (180.0, 100.0),
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

        # State Variables
        self._active_preset_code: str = "CHONPS_LIFE_ELEMENTS"
        self._preset: PeriodicPreset = PRESETS[self._active_preset_code]

        self._selected_z: int = self._preset.target_z
        self._view_mode: str = self._preset.view_mode  # STANDARD, ELECTRONEGATIVITY, BOHR_SHELLS

        # Bohr orbital electron animation
        self._electron_orbit_angle: float = 0.0
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(40)  # 25 FPS
        self._anim_timer.timeout.connect(self._on_orbit_tick)
        self._anim_timer.start()

        # UI interaction button rects
        self._btn_rects: dict[str, QRectF] = {}
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Precomputes button rects for click and air pinch interaction."""
        btn_y = 44.0
        btn_h = 22.0
        cur_x = 14.0
        preset_names = [
            ("CHONPS_LIFE_ELEMENTS", "CHONPS"),
            ("ELECTRONEGATIVITY_TRENDS", "Độ âm điện"),
            ("NOBLE_GAS_OCTET_STABILITY", "Khí hiếm"),
            ("TRANSITION_METALS_AUFBAU", "Phân lớp 3d"),
            ("ALKALI_VS_HALOGENS", "Liên kết Ion"),
        ]
        for p_code, p_lbl in preset_names:
            pw = 74.0
            self._btn_rects[f"preset_{p_code}"] = QRectF(cur_x, btn_y, pw, btn_h)
            cur_x += pw + 5.0

        # Controls on top right: Prev Z, Next Z, View Mode, Stamp
        r_prev = QRectF(cur_x + 10.0, btn_y, 28.0, btn_h)
        self._btn_rects["prev_z"] = r_prev
        r_next = QRectF(r_prev.right() + 4.0, btn_y, 28.0, btn_h)
        self._btn_rects["next_z"] = r_next

        r_mode = QRectF(r_next.right() + 6.0, btn_y, 44.0, btn_h)
        self._btn_rects["toggle_mode"] = r_mode

        r_stamp = QRectF(r_mode.right() + 6.0, btn_y, 48.0, btn_h)
        self._btn_rects["stamp"] = r_stamp

        # Mini Periodic Table Grid bounds (Period 1 to 4, 18 columns)
        grid_x = 18.0
        grid_y = 75.0
        col_w = 17.5
        row_h = 19.0
        for z, (pr, gc) in PERIODIC_COORDS.items():
            r_elem = QRectF(grid_x + gc * col_w, grid_y + pr * row_h, col_w - 1.5, row_h - 1.5)
            self._btn_rects[f"elem_{z}"] = r_elem

    def cleanup(self) -> None:
        """Stops animation timer to avoid memory leaks."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()

    # -------------------------------------------------------------------------
    # Properties & Setters
    # -------------------------------------------------------------------------
    @property
    def selected_z(self) -> int:
        return self._selected_z

    @property
    def current_element(self) -> ElementData:
        return ELEMENT_DB.get(self._selected_z, ELEMENT_DB[1])

    @property
    def active_preset(self) -> PeriodicPreset:
        return self._preset

    @property
    def view_mode(self) -> str:
        return self._view_mode

    def load_preset(self, preset_code: str) -> None:
        """Loads a predefined periodic preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._selected_z = self._preset.target_z
        self._view_mode = self._preset.view_mode
        self.element_selected.emit(self._selected_z)
        self.update()

    def set_element_z(self, z: int) -> None:
        """Selects element by atomic number Z (clamped to 1..36)."""
        self._selected_z = max(1, min(36, int(z)))
        self.element_selected.emit(self._selected_z)
        self.update()

    def step_z(self, delta: int) -> None:
        """Steps atomic number Z by delta (+1 or -1)."""
        new_z = self._selected_z + delta
        if new_z < 1:
            new_z = 36
        elif new_z > 36:
            new_z = 1
        self.set_element_z(new_z)

    def toggle_view_mode(self) -> str:
        """Cycles view mode between STANDARD, ELECTRONEGATIVITY, and BOHR_SHELLS."""
        modes = ["STANDARD", "ELECTRONEGATIVITY", "BOHR_SHELLS"]
        cur_idx = modes.index(self._view_mode) if self._view_mode in modes else 0
        self._view_mode = modes[(cur_idx + 1) % len(modes)]
        self.update()
        return self._view_mode

    def _on_orbit_tick(self) -> None:
        """Animation tick updating orbital rotation angle."""
        self._electron_orbit_angle = (self._electron_orbit_angle + 3.0) % 360.0
        self.update()

    # -------------------------------------------------------------------------
    # Touchless Air Gesture Handling & Stamping
    # -------------------------------------------------------------------------
    def stamp_to_board(self) -> None:
        """Stamps active Periodic Table element diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import ChemistryTools

        center = (self.pos().x() + self.WIDTH / 2.0, self.pos().y() + self.HEIGHT + 40.0)
        items = ChemistryTools.create_periodic_table_diagram(
            center=center,
            atomic_z=self._selected_z,
            preset_code=self._active_preset_code,
            view_mode=self._view_mode,
        )
        for itm in items:
            scene.addItem(itm)
        self.stamp_requested.emit()

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air pinch gesture handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset buttons
        for k in PRESETS:
            r = self._btn_rects.get(f"preset_{k}")
            if r and r.contains(local_pos):
                self.load_preset(k)
                return True

        # 2. Stepper buttons
        r_prev = self._btn_rects.get("prev_z")
        if r_prev and r_prev.contains(local_pos):
            self.step_z(-1)
            return True

        r_next = self._btn_rects.get("next_z")
        if r_next and r_next.contains(local_pos):
            self.step_z(+1)
            return True

        # 3. View mode toggle
        r_mode = self._btn_rects.get("toggle_mode")
        if r_mode and r_mode.contains(local_pos):
            self.toggle_view_mode()
            return True

        # 4. Stamp
        r_stamp = self._btn_rects.get("stamp")
        if r_stamp and r_stamp.contains(local_pos):
            self.stamp_to_board()
            return True

        # 5. Direct element tile clicks in periodic grid
        for z in range(1, 37):
            r_elem = self._btn_rects.get(f"elem_{z}")
            if r_elem and r_elem.contains(local_pos):
                self.set_element_z(z)
                return True

        return True

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
        if painter is None:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # 1. Background Card
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header Banner
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(0.0, 0.0, self.WIDTH, 36.0), 10.0, 10.0)
        painter.drawRect(QRectF(0.0, 26.0, self.WIDTH, 10.0))

        elem = self.current_element
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(14.0, 0.0, self.WIDTH - 28.0, 36.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"⚛️ BẢNG TUẦN HOÀN & CẤU HÌNH ELECTRON: Z={elem.z} ({elem.symbol} — {elem.name_vi})",
        )

        # 3. Preset Buttons
        preset_names = [
            ("CHONPS_LIFE_ELEMENTS", "CHONPS"),
            ("ELECTRONEGATIVITY_TRENDS", "Độ âm điện"),
            ("NOBLE_GAS_OCTET_STABILITY", "Khí hiếm"),
            ("TRANSITION_METALS_AUFBAU", "Phân lớp 3d"),
            ("ALKALI_VS_HALOGENS", "Liên kết Ion"),
        ]
        for p_code, p_lbl in preset_names:
            r_btn = self._btn_rects[f"preset_{p_code}"]
            is_active = self._active_preset_code == p_code

            painter.setBrush(QBrush(QColor("#0284c7") if is_active else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#38bdf8") if is_active else QColor("#334155"), 1.0))
            painter.drawRoundedRect(r_btn, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(r_btn, Qt.AlignmentFlag.AlignCenter, p_lbl)

        # Controls
        r_prev = self._btn_rects["prev_z"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_prev, 4.0, 4.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(r_prev, Qt.AlignmentFlag.AlignCenter, "◀ Z")

        r_next = self._btn_rects["next_z"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_next, 4.0, 4.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(r_next, Qt.AlignmentFlag.AlignCenter, "Z ▶")

        r_mode = self._btn_rects["toggle_mode"]
        painter.setBrush(QBrush(QColor("#0f766e")))
        painter.setPen(QPen(QColor("#14b8a6"), 1.0))
        painter.drawRoundedRect(r_mode, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        mode_lbl = "Chuẩn" if self._view_mode == "STANDARD" else ("Âm điện" if self._view_mode == "ELECTRONEGATIVITY" else "Bohr")
        painter.drawText(r_mode, Qt.AlignmentFlag.AlignCenter, mode_lbl)

        r_stamp = self._btn_rects["stamp"]
        painter.setBrush(QBrush(QColor("#7c3aed")))
        painter.setPen(QPen(QColor("#a78bfa"), 1.0))
        painter.drawRoundedRect(r_stamp, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng")

        # ---------------------------------------------------------------------
        # 4. Mini Periodic Table Grid (Top Section)
        # ---------------------------------------------------------------------
        for z, el in ELEMENT_DB.items():
            r_elem = self._btn_rects.get(f"elem_{z}")
            if not r_elem:
                continue

            is_selected = z == self._selected_z

            # Background color by view mode
            if self._view_mode == "ELECTRONEGATIVITY":
                en = el.electronegativity or 0.0
                # Interpolate from purple (0.8) to red (4.0)
                frac = max(0.0, min(1.0, (en - 0.7) / 3.3))
                tile_bg = QColor(int(frac * 240), int((1.0 - frac) * 160), int((1.0 - frac) * 240))
            else:
                # Default CPK or dark tint
                tile_bg = QColor(el.cpk_hex).darker(160)

            painter.setBrush(QBrush(QColor("#fbbf24") if is_selected else tile_bg))
            painter.setPen(QPen(QColor("#ffffff") if is_selected else QColor("#334155"), 1.5 if is_selected else 0.5))
            painter.drawRoundedRect(r_elem, 2.0, 2.0)

            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold if is_selected else QFont.Weight.Normal))
            painter.setPen(QColor("#000000") if is_selected else QColor("#f8fafc"))
            painter.drawText(r_elem, Qt.AlignmentFlag.AlignCenter, el.symbol)

        # ---------------------------------------------------------------------
        # 5. Selected Element Spotlight Tile & Chemical Properties (Bottom Left)
        # ---------------------------------------------------------------------
        spot_x = 18.0
        spot_y = 160.0
        spot_w = 280.0
        spot_h = 205.0

        spot_rect = QRectF(spot_x, spot_y, spot_w, spot_h)
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(spot_rect, 6.0, 6.0)

        # Big Chemical Symbol Box (64x64)
        box_sz = 68.0
        box_rect = QRectF(spot_x + 12.0, spot_y + 12.0, box_sz, box_sz)
        painter.setBrush(QBrush(QColor(elem.cpk_hex).darker(140)))
        painter.setPen(QPen(QColor("#38bdf8"), 2.0))
        painter.drawRoundedRect(box_rect, 6.0, 6.0)

        # Atomic number top-left
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(QPointF(box_rect.x() + 6.0, box_rect.y() + 14.0), str(elem.z))

        # Atomic mass top-right
        painter.setFont(QFont("Consolas", 7))
        painter.drawText(
            QRectF(box_rect.x(), box_rect.y() + 4.0, box_sz - 6.0, 14.0),
            Qt.AlignmentFlag.AlignRight,
            f"{elem.atomic_mass:.2f}",
        )

        # Giant Symbol in center
        painter.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(
            QRectF(box_rect.x(), box_rect.y() + 14.0, box_sz, 36.0),
            Qt.AlignmentFlag.AlignCenter,
            elem.symbol,
        )

        # Element name in Vietnamese below symbol
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(
            QRectF(box_rect.x(), box_rect.bottom() - 16.0, box_sz, 14.0),
            Qt.AlignmentFlag.AlignCenter,
            elem.name_vi,
        )

        # Property detail lines
        prop_x = spot_x + box_sz + 24.0
        prop_lines = [
            (f"{elem.name_vi} ({elem.name_en})", QColor("#f8fafc"), True, 9),
            (f"• Nhóm: {elem.group} | Chu kỳ: {elem.period}", QColor("#38bdf8"), False, 7),
            (f"• Phân loại: {elem.category}", QColor("#fbbf24"), False, 7),
            (f"• Cấu hình e: {elem.electron_config}", QColor("#34d399"), True, 8),
            (f"• Lớp electron (Bohr): {', '.join(str(s) for s in elem.bohr_shells)}", QColor("#a78bfa"), True, 7),
            (f"• Độ âm điện: {elem.electronegativity if elem.electronegativity else 'N/A'}", QColor("#f472b6"), False, 7),
            (f"• Năng lượng ion hóa: {elem.ionization_energy_kj:.1f} kJ/mol", QColor("#94a3b8"), False, 7),
            (f"• Bán kính nguyên tử: {elem.atomic_radius_pm} pm", QColor("#94a3b8"), False, 7),
        ]
        py = spot_y + 16.0
        for p_txt, p_col, is_bld, font_sz in prop_lines:
            painter.setFont(QFont("Segoe UI", font_sz, QFont.Weight.Bold if is_bld else QFont.Weight.Normal))
            painter.setPen(p_col)
            painter.drawText(QPointF(prop_x, py), p_txt)
            py += 17.5

        # ---------------------------------------------------------------------
        # 6. Animated Bohr Orbital Shells (Bottom Right)
        # ---------------------------------------------------------------------
        bohr_x = spot_x + spot_w + 14.0
        bohr_y = spot_y
        bohr_w = self.WIDTH - bohr_x - 14.0
        bohr_h = spot_h

        bohr_rect = QRectF(bohr_x, bohr_y, bohr_w, bohr_h)
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(bohr_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(bohr_x + 10.0, bohr_y + 18.0), "MÔ HÌNH HẠT NHÂN & LỚP VỎ BOHR")

        # Nucleus Center
        bc_x = bohr_x + bohr_w * 0.5
        bc_y = bohr_y + bohr_h * 0.55

        # Draw Nucleus (Z protons)
        nuc_r = 12.0
        painter.setBrush(QBrush(QColor("#ef4444")))
        painter.setPen(QPen(QColor("#fca5a5"), 1.5))
        painter.drawEllipse(QPointF(bc_x, bc_y), nuc_r, nuc_r)

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(
            QRectF(bc_x - nuc_r, bc_y - nuc_r, 2 * nuc_r, 2 * nuc_r),
            Qt.AlignmentFlag.AlignCenter,
            f"{elem.z}+",
        )

        # Concentric Bohr Shells (K, L, M, N)
        shell_names = ("K", "L", "M", "N")
        base_r = 20.0
        r_step = 16.0

        for s_idx, e_count in enumerate(elem.bohr_shells):
            s_radius = base_r + s_idx * r_step
            # Shell orbit line
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#334155"), 1.0, Qt.PenStyle.DashLine))
            painter.drawEllipse(QPointF(bc_x, bc_y), s_radius, s_radius)

            # Shell label (K, L, M, N)
            s_name = shell_names[s_idx] if s_idx < len(shell_names) else f"S{s_idx+1}"
            painter.setFont(QFont("Segoe UI", 6))
            painter.setPen(QColor("#64748b"))
            painter.drawText(QPointF(bc_x + s_radius - 6.0, bc_y - 2.0), s_name)

            # Orbiting Electrons
            speed_mult = 1.0 / (s_idx + 1)
            angle_offset = self._electron_orbit_angle * speed_mult * (1 if s_idx % 2 == 0 else -1)

            for ei in range(e_count):
                theta = math.radians(angle_offset + (360.0 * ei / max(1, e_count)))
                ex = bc_x + s_radius * math.cos(theta)
                ey = bc_y + s_radius * math.sin(theta)

                # Cyan glowing electron dot
                painter.setBrush(QBrush(QColor("#38bdf8")))
                painter.setPen(QPen(QColor("#ffffff"), 0.5))
                painter.drawEllipse(QPointF(ex, ey), 2.5, 2.5)
