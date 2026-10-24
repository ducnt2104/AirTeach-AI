"""
AirTeach AI — Interactive Physics & Geometrical Optics: Thin Lens Systems & Ray Transfer Matrix.
Simulates thin lens Gaussian imaging 1/f = 1/d_o + 1/d_i, lateral magnification m = -d_i/d_o,
real and virtual image formation, multi-lens compound systems (Microscope, Telescope),
principal ray tracing (Parallel, Central, Focal), and 2x2 Ray Transfer ABCD Matrix invariants.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QTimer
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QBrush,
    QFont,
    QPainterPath,
    QLinearGradient,
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


@dataclass(frozen=True)
class LensSpec:
    name: str
    focal_length_mm: float  # + for convex converging, - for concave diverging
    x_pos_mm: float         # Position along optical bench (mm)
    diameter_mm: float      # Physical aperture diameter (mm)


@dataclass(frozen=True)
class LensSystemPreset:
    code: str
    title_vi: str
    badge_label: str
    is_two_lens: bool
    lens1: LensSpec
    lens2: Optional[LensSpec]
    object_x_mm: float
    object_height_mm: float
    description: str


THIN_LENS_PRESETS: dict[str, LensSystemPreset] = {
    "CONVEX_LENS_REAL_IMAGE": LensSystemPreset(
        code="CONVEX_LENS_REAL_IMAGE",
        title_vi="Thấu kính Hội tụ — Tạo Ảnh Thật, Ngược chiều (Mô hình Máy ảnh, Mắt)",
        badge_label="CONVEX REAL",
        is_two_lens=False,
        lens1=LensSpec("Thấu kính L₁ (Hội tụ)", 100.0, 300.0, 140.0),
        lens2=None,
        object_x_mm=50.0,  # d_o = 250 mm > 2f
        object_height_mm=45.0,
        description="Vật thật đặt ngoài khoảng 2f (d_o = 250 mm > 200 mm). Thấu kính hội tụ tạo ảnh thật, ngược chiều, thu nhỏ (d_i = 166.7 mm, m = -0.67) trên tiêu diện sau, tương tự cơ chế tạo ảnh trên võng mạc mắt hoặc cảm biến máy ảnh.",
    ),
    "MAGNIFYING_GLASS_VIRTUAL_IMAGE": LensSystemPreset(
        code="MAGNIFYING_GLASS_VIRTUAL_IMAGE",
        title_vi="Kính lúp Cầm tay — Tạo Ảnh Ảo, Cùng chiều, Phóng đại (d_o < f)",
        badge_label="MAGNIFIER",
        is_two_lens=False,
        lens1=LensSpec("Kính lúp L₁", 100.0, 300.0, 140.0),
        lens2=None,
        object_x_mm=240.0,  # d_o = 60 mm < f
        object_height_mm=30.0,
        description="Vật đặt trong khoảng tiêu cự (d_o = 60 mm < f = 100 mm). Chùm tia ló phân kỳ, đường kéo dài ngược chiều hội tụ tạo ảnh ảo, cùng chiều và phóng đại lớn (d_i = -150 mm, m = +2.50).",
    ),
    "CONCAVE_DIVERGING_LENS": LensSystemPreset(
        code="CONCAVE_DIVERGING_LENS",
        title_vi="Thấu kính Phân kỳ — Luôn Tạo Ảnh Ảo, Thu nhỏ (Kính Cận thị)",
        badge_label="CONCAVE DIVERGING",
        is_two_lens=False,
        lens1=LensSpec("Thấu kính L₁ (Phân kỳ)", -120.0, 300.0, 140.0),
        lens2=None,
        object_x_mm=120.0,  # d_o = 180 mm
        object_height_mm=45.0,
        description="Thấu kính phân kỳ (f = -120 mm) có rìa dày hơn giữa. Mọi vị trí vật thật trước thấu kính đều cho ảnh ảo, cùng chiều và luôn nhỏ hơn vật (d_i = -72 mm, m = +0.40), dùng để chỉnh tật cận thị.",
    ),
    "COMPOUND_MICROSCOPE_TWO_LENS": LensSystemPreset(
        code="COMPOUND_MICROSCOPE_TWO_LENS",
        title_vi="Kính hiển vi Quang học 2 Thấu kính — Vật kính L₁ + Thị kính L₂",
        badge_label="MICROSCOPE",
        is_two_lens=True,
        lens1=LensSpec("Vật kính L₁ (Objective)", 40.0, 220.0, 100.0),
        lens2=LensSpec("Thị kính L₂ (Eyepiece)", 80.0, 480.0, 120.0),
        object_x_mm=170.0,  # d_o1 = 50 mm > f1
        object_height_mm=20.0,
        description="Hệ 2 thấu kính phóng đại 2 cấp. Vật kính L₁ tạo ảnh thật trung gian ngược chiều phóng đại 4x (d_i1 = 200 mm). Thị kính L₂ đóng vai trò kính lúp tiếp tục phóng đại ảnh trung gian tạo ảnh ảo khổng lồ (Độ bội giác tổng M = -16x).",
    ),
    "ASTRONOMICAL_KEPLERIAN_TELESCOPE": LensSystemPreset(
        code="ASTRONOMICAL_KEPLERIAN_TELESCOPE",
        title_vi="Kính thiên văn Khúc xạ Kepler — Chiều dài Ống d = f₁ + f₂",
        badge_label="TELESCOPE",
        is_two_lens=True,
        lens1=LensSpec("Vật kính L₁", 180.0, 240.0, 140.0),
        lens2=LensSpec("Thị kính L₂", 60.0, 480.0, 100.0),
        object_x_mm=40.0,  # Far object
        object_height_mm=35.0,
        description="Kính thiên văn khúc xạ Keplerian: Khoảng cách giữa 2 thấu kính đúng bằng tổng tiêu cự d = f₁ + f₂ = 240 mm. Tiêu điểm ảnh của vật kính trùng với tiêu điểm vật của thị kính, tạo ảnh ở vô cực cho mắt quan sát không điều tiết (Độ phóng đại góc M = -f₁/f₂ = -3x).",
    ),
}


class InteractiveThinLensSystemItem(QGraphicsObject):
    """
    Interactive Geometrical Optics: Thin Lens Systems & Ray Tracing Simulator.
    Simulates single and multi-lens systems, Gaussian thin lens equation 1/f = 1/d_o + 1/d_i,
    lateral magnification m = -d_i/d_o, principal rays (Parallel, Central, Focal),
    real and virtual image formation, and 2x2 Ray Transfer ABCD Matrix.
    """

    preset_changed = Signal(str)
    optics_computed = Signal(float, float, float)  # di_mm, magnification, focal_length
    stamp_requested = Signal(dict)

    CARD_W = 950.0
    CARD_H = 640.0

    def __init__(self, pos: tuple[float, float] = (100.0, 50.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation Preset & Parameters
        self._active_preset_code: str = "CONVEX_LENS_REAL_IMAGE"
        self._preset: LensSystemPreset = THIN_LENS_PRESETS[self._active_preset_code]

        self._is_two_lens: bool = self._preset.is_two_lens
        self._f1_mm: float = self._preset.lens1.focal_length_mm
        self._x1_mm: float = self._preset.lens1.x_pos_mm
        self._f2_mm: float = self._preset.lens2.focal_length_mm if self._preset.lens2 else 80.0
        self._x2_mm: float = self._preset.lens2.x_pos_mm if self._preset.lens2 else 480.0

        self._object_x_mm: float = self._preset.object_x_mm
        self._object_h_mm: float = self._preset.object_height_mm

        # Animation state (ray pulsation)
        self._ray_phase: float = 0.0
        self._is_running: bool = True

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # 30 fps animation timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Geometrical Optics Mathematical Formulations
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> LensSystemPreset:
        return self._preset

    @property
    def object_distance_d1_mm(self) -> float:
        """Object distance to Lens 1: d_o1 = x_lens1 - x_obj."""
        return max(5.0, self._x1_mm - self._object_x_mm)

    @property
    def image_distance_d1_mm(self) -> float:
        """
        Calculates image distance from Lens 1:
        1/f = 1/d_o + 1/d_i => d_i = (f * d_o) / (d_o - f)
        """
        f = self._f1_mm
        do = self.object_distance_d1_mm
        denom = do - f
        if abs(denom) < 0.1:
            return 1e5  # Infinite distance / collimated
        return (f * do) / denom

    @property
    def magnification_m1(self) -> float:
        """Lateral magnification m1 = -d_i1 / d_o1."""
        do = self.object_distance_d1_mm
        di = self.image_distance_d1_mm
        return -di / do

    @property
    def image_height_h1_mm(self) -> float:
        """Image 1 height h_i1 = m1 * h_o."""
        return self.magnification_m1 * self._object_h_mm

    @property
    def image1_x_pos_mm(self) -> float:
        """Absolute position of Image 1 on optical bench: x_i1 = x1 + d_i1."""
        return self._x1_mm + self.image_distance_d1_mm

    @property
    def is_image1_real(self) -> bool:
        """Image is real if d_i > 0, virtual if d_i < 0."""
        return self.image_distance_d1_mm > 0.0

    # Two-Lens Compound Optics (Lens 2)
    @property
    def object_distance_d2_mm(self) -> float:
        """Object distance for Lens 2: d_o2 = x_lens2 - x_image1."""
        return self._x2_mm - self.image1_x_pos_mm

    @property
    def image_distance_d2_mm(self) -> float:
        """Image distance from Lens 2: d_i2 = (f2 * d_o2) / (d_o2 - f2)."""
        if not self._is_two_lens:
            return 0.0
        f2 = self._f2_mm
        do2 = self.object_distance_d2_mm
        denom = do2 - f2
        if abs(denom) < 0.1:
            return 1e5
        return (f2 * do2) / denom

    @property
    def total_magnification_m(self) -> float:
        """Total system magnification: M = m1 * m2."""
        if not self._is_two_lens:
            return self.magnification_m1
        m1 = self.magnification_m1
        do2 = self.object_distance_d2_mm
        di2 = self.image_distance_d2_mm
        m2 = -di2 / max(1.0, do2)
        return m1 * m2

    @property
    def ray_transfer_matrix(self) -> tuple[float, float, float, float]:
        """
        Calculates 2x2 Ray Transfer ABCD Matrix from object plane to image plane:
        M = [1, d_i; 0, 1] * [1, 0; -1/f, 1] * [1, d_o; 0, 1]
        Yields:
        A = 1 - d_i / f
        B = d_o + d_i - (d_o * d_i) / f  (= 0 on imaging conjugate plane!)
        C = -1 / f
        D = 1 - d_o / f
        """
        f = self._f1_mm
        do = self.object_distance_d1_mm
        di = self.image_distance_d1_mm
        a = 1.0 - (di / f)
        b = do + di - ((do * di) / f)
        c = -1.0 / f
        d = 1.0 - (do / f)
        return a, b, c, d

    @property
    def optical_power_diopters(self) -> float:
        """Optical power P = 1 / f (in Diopters m⁻¹)."""
        f_m = self._f1_mm * 1e-3
        return 1.0 / f_m

    # -------------------------------------------------------------------------
    # Animation & Parameter Controls
    # -------------------------------------------------------------------------
    def _on_sim_tick(self) -> None:
        if self._is_running:
            self._ray_phase = (self._ray_phase + 0.1) % (2.0 * math.pi)
        self.update()

    def load_preset(self, preset_code: str) -> None:
        """Loads a geometrical optics preset."""
        if preset_code not in THIN_LENS_PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = THIN_LENS_PRESETS[preset_code]
        self._is_two_lens = self._preset.is_two_lens
        self._f1_mm = self._preset.lens1.focal_length_mm
        self._x1_mm = self._preset.lens1.x_pos_mm
        if self._preset.lens2:
            self._f2_mm = self._preset.lens2.focal_length_mm
            self._x2_mm = self._preset.lens2.x_pos_mm
        self._object_x_mm = self._preset.object_x_mm
        self._object_h_mm = self._preset.object_height_mm

        self.preset_changed.emit(preset_code)
        self.optics_computed.emit(self.image_distance_d1_mm, self.total_magnification_m, self._f1_mm)
        self.update()

    def step_object_distance(self, delta_mm: float) -> None:
        """Steps object position x along bench."""
        self._object_x_mm = max(20.0, min(self._x1_mm - 10.0, self._object_x_mm + delta_mm))
        self.optics_computed.emit(self.image_distance_d1_mm, self.total_magnification_m, self._f1_mm)
        self.update()

    def step_focal_length(self, delta_mm: float) -> None:
        """Steps Lens 1 focal length."""
        sign = 1.0 if self._f1_mm >= 0 else -1.0
        abs_f = abs(self._f1_mm) + delta_mm
        self._f1_mm = sign * max(30.0, min(250.0, abs_f))
        self.optics_computed.emit(self.image_distance_d1_mm, self.total_magnification_m, self._f1_mm)
        self.update()

    def toggle_lens_type(self) -> None:
        """Toggles Lens 1 between Convex (+f) and Concave (-f)."""
        self._f1_mm = -self._f1_mm
        self.optics_computed.emit(self.image_distance_d1_mm, self.total_magnification_m, self._f1_mm)
        self.update()

    def toggle_animation(self) -> None:
        """Toggles ray pulsation animation."""
        self._is_running = not self._is_running
        self.update()

    # -------------------------------------------------------------------------
    # UI Buttons & Touchless Air Gestures
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Initializes touchless air gesture button bounds."""
        bar_y = 574.0
        self._buttons["btn_preset"] = QRectF(20.0, bar_y, 160.0, 34.0)
        self._buttons["btn_do_down"] = QRectF(188.0, bar_y, 45.0, 34.0)
        self._buttons["btn_do_up"] = QRectF(238.0, bar_y, 45.0, 34.0)
        self._buttons["btn_f_down"] = QRectF(290.0, bar_y, 45.0, 34.0)
        self._buttons["btn_f_up"] = QRectF(340.0, bar_y, 45.0, 34.0)
        self._buttons["btn_lens_type"] = QRectF(392.0, bar_y, 75.0, 34.0)
        self._buttons["btn_p_real"] = QRectF(474.0, bar_y, 75.0, 34.0)
        self._buttons["btn_p_mag"] = QRectF(556.0, bar_y, 75.0, 34.0)
        self._buttons["btn_p_concave"] = QRectF(638.0, bar_y, 75.0, 34.0)
        self._buttons["btn_p_micro"] = QRectF(720.0, bar_y, 75.0, 34.0)
        self._buttons["btn_play"] = QRectF(802.0, bar_y, 55.0, 34.0)
        self._buttons["btn_stamp"] = QRectF(864.0, bar_y, 66.0, 34.0)

    def _handle_button_click(self, key: str) -> None:
        preset_keys = list(THIN_LENS_PRESETS.keys())
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_do_down":
            self.step_object_distance(-15.0)
        elif key == "btn_do_up":
            self.step_object_distance(15.0)
        elif key == "btn_f_down":
            self.step_focal_length(-10.0)
        elif key == "btn_f_up":
            self.step_focal_length(10.0)
        elif key == "btn_lens_type":
            self.toggle_lens_type()
        elif key == "btn_p_real":
            self.load_preset("CONVEX_LENS_REAL_IMAGE")
        elif key == "btn_p_mag":
            self.load_preset("MAGNIFYING_GLASS_VIRTUAL_IMAGE")
        elif key == "btn_p_concave":
            self.load_preset("CONCAVE_DIVERGING_LENS")
        elif key == "btn_p_micro":
            self.load_preset("COMPOUND_MICROSCOPE_TWO_LENS")
        elif key == "btn_play":
            self.toggle_animation()
        elif key == "btn_stamp":
            self.stamp_diagram()
        self.update()

    def handle_air_pinch(self, pos: QPointF) -> bool:
        """Handles touchless air pinch gesture targeting interactive UI buttons."""
        local_pos = self.mapFromScene(pos)
        for key, rect in self._buttons.items():
            if rect.contains(local_pos):
                self._handle_button_click(key)
                return True
        return False

    def handle_air_drag(self, delta: QPointF) -> None:
        """Translates item across whiteboard when dragged by air pinch."""
        self.setPos(self.pos() + delta)

    def stamp_diagram(self) -> None:
        """Emits signal requesting permanent vector stamping of current diagram."""
        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 680.0,
            "height": 380.0,
            "preset_code": self._active_preset_code,
            "f1_mm": self._f1_mm,
            "do_mm": self.object_distance_d1_mm,
            "di_mm": self.image_distance_d1_mm,
            "magnification": self.total_magnification_m,
            "is_real": self.is_image1_real,
            "is_two_lens": self._is_two_lens,
        }
        self.stamp_requested.emit(stamp_data)

    # -------------------------------------------------------------------------
    # QGraphicsItem Rendering
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card_background(painter)
        self._paint_header(painter)
        self._paint_optical_bench_panel(painter)
        self._paint_abcd_matrix_panel(painter)
        self._paint_metrics_panel(painter)
        self._paint_controls_bar(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor(9, 14, 25, 252))
        grad.setColorAt(1.0, QColor(5, 8, 16, 255))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.8))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        title_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(22, 30, f"QUANG HÌNH HỌC: HỆ THẤU KÍNH MỎNG & VẼ TIA SÁNG: {self._preset.badge_label}")

        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(148, 163, 184)))
        type_str = "Hội tụ (Convex)" if self._f1_mm > 0 else "Phân kỳ (Concave)"
        image_type = "ẢNH THẬT (Real, Ngược chiều)" if self.is_image1_real else "ẢNH ẢO (Virtual, Cùng chiều)"
        painter.drawText(
            22,
            48,
            f"Thấu kính L₁: {type_str} f = {self._f1_mm:+.1f} mm ({self.optical_power_diopters:+.1f} D) | d_o = {self.object_distance_d1_mm:.1f} mm | d_i = {self.image_distance_d1_mm:+.1f} mm | Độ phóng đại m = {self.total_magnification_m:+.2f} | {image_type}",
        )

    def _paint_optical_bench_panel(self, painter: QPainter) -> None:
        """Renders principal axis, lenses, focal points, object arrow, rays, and image arrow."""
        bx = 20.0
        by = 60.0
        bw = 910.0
        bh = 260.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(bx, by, bw, bh, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(bx + 12.0, by + 20.0, "BĂNG QUANG HỌC & ĐƯỜNG TRUYỀN CÁC TIA SÁNG ĐẶC BIỆT:")

        # Bench coordinate mapping
        # Optical bench spans from x = 0 mm to x = 600 mm
        axis_y = by + bh / 2.0 + 8.0
        scale_x = (bw - 60.0) / 600.0
        bench_x0 = bx + 30.0

        def bench_to_px(x_mm: float) -> float:
            return bench_x0 + x_mm * scale_x

        def height_to_py(h_mm: float) -> float:
            # Positive height points UP (-y in Qt)
            return axis_y - h_mm * 1.5

        # Principal Optical Axis (horizontal line)
        painter.setPen(QPen(QColor(71, 85, 105), 1.4, Qt.PenStyle.DashLine))
        painter.drawLine(bench_x0 - 15.0, axis_y, bench_x0 + (bw - 30.0), axis_y)

        # Lens 1 representation
        l1_px = bench_to_px(self._x1_mm)
        l1_top = axis_y - 85.0
        l1_bot = axis_y + 85.0

        # Lens vertical line
        painter.setPen(QPen(QColor(56, 189, 248), 2.5))
        painter.drawLine(l1_px, l1_top, l1_px, l1_bot)

        # Lens Arrowheads (Convex: arrows out; Concave: arrows inward)
        is_convex = self._f1_mm > 0
        arrow_size = 8.0
        if is_convex:
            # Top arrow pointing UP
            painter.drawLine(l1_px - arrow_size, l1_top + arrow_size, l1_px, l1_top)
            painter.drawLine(l1_px + arrow_size, l1_top + arrow_size, l1_px, l1_top)
            # Bottom arrow pointing DOWN
            painter.drawLine(l1_px - arrow_size, l1_bot - arrow_size, l1_px, l1_bot)
            painter.drawLine(l1_px + arrow_size, l1_bot - arrow_size, l1_px, l1_bot)
        else:
            # Top arrow pointing INWARD (down)
            painter.drawLine(l1_px - arrow_size, l1_top, l1_px, l1_top + arrow_size)
            painter.drawLine(l1_px + arrow_size, l1_top, l1_px, l1_top + arrow_size)
            # Bottom arrow pointing INWARD (up)
            painter.drawLine(l1_px - arrow_size, l1_bot, l1_px, l1_bot - arrow_size)
            painter.drawLine(l1_px + arrow_size, l1_bot, l1_px, l1_bot - arrow_size)

        # Optical center O1
        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.drawEllipse(QPointF(l1_px, axis_y), 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(l1_px + 4.0, axis_y + 14.0, "O₁")
        painter.drawText(l1_px - 14.0, l1_top - 6.0, f"L₁ ({'+' if is_convex else ''}{self._f1_mm:.0f}mm)")

        # Focal points F1 and F'1
        f1 = self._f1_mm
        f1_left_px = bench_to_px(self._x1_mm - abs(f1))
        f1_right_px = bench_to_px(self._x1_mm + abs(f1))

        painter.setBrush(QBrush(QColor(244, 63, 94)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(f1_left_px, axis_y), 3.5, 3.5)
        painter.drawEllipse(QPointF(f1_right_px, axis_y), 3.5, 3.5)

        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(244, 63, 94)))
        painter.drawText(f1_left_px - 8.0, axis_y + 12.0, "F₁")
        painter.drawText(f1_right_px - 8.0, axis_y + 12.0, "F'₁")

        # Lens 2 if active (Two-lens compound system)
        if self._is_two_lens:
            l2_px = bench_to_px(self._x2_mm)
            l2_top = axis_y - 75.0
            l2_bot = axis_y + 75.0
            painter.setPen(QPen(QColor(168, 85, 247), 2.5))
            painter.drawLine(l2_px, l2_top, l2_px, l2_bot)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(168, 85, 247)))
            painter.drawText(l2_px - 14.0, l2_top - 6.0, f"L₂ (+{self._f2_mm:.0f}mm)")

        # Object Arrow AB (Yellow)
        obj_px = bench_to_px(self._object_x_mm)
        obj_tip_py = height_to_py(self._object_h_mm)

        painter.setPen(QPen(QColor(250, 204, 21), 2.5))
        painter.drawLine(obj_px, axis_y, obj_px, obj_tip_py)
        # Arrowhead
        painter.drawLine(obj_px - 5.0, obj_tip_py + 8.0, obj_px, obj_tip_py)
        painter.drawLine(obj_px + 5.0, obj_tip_py + 8.0, obj_px, obj_tip_py)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(250, 204, 21)))
        painter.drawText(obj_px - 10.0, obj_tip_py - 6.0, f"Vật AB ({self._object_h_mm:.0f}mm)")

        # Image Arrow A'B'
        img_px = bench_to_px(self.image1_x_pos_mm)
        img_tip_py = height_to_py(self.image_height_h1_mm)
        is_real = self.is_image1_real

        img_col = QColor(34, 197, 94) if is_real else QColor(244, 63, 94)
        pen_style = Qt.PenStyle.SolidLine if is_real else Qt.PenStyle.DashLine
        painter.setPen(QPen(img_col, 2.5, pen_style))

        if 0 < img_px < bx + bw:
            painter.drawLine(img_px, axis_y, img_px, img_tip_py)
            # Arrowhead pointing towards tip
            dir_sign = 1.0 if self.image_height_h1_mm > 0 else -1.0
            painter.drawLine(img_px - 5.0, img_tip_py + dir_sign * 8.0, img_px, img_tip_py)
            painter.drawLine(img_px + 5.0, img_tip_py + dir_sign * 8.0, img_px, img_tip_py)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(img_col))
            painter.drawText(img_px - 15.0, img_tip_py - (12.0 if dir_sign > 0 else -14.0), f"Ảnh A'B' ({self.image_height_h1_mm:+.1f}mm)")

        # ---------------------------------------------------------------------
        # Principal Ray Tracing
        # ---------------------------------------------------------------------
        # Ray 1: Parallel to axis, refracts through F'1
        # From (obj_px, obj_tip_py) -> (l1_px, obj_tip_py)
        painter.setPen(QPen(QColor(239, 68, 68), 1.5))
        painter.drawLine(obj_px, obj_tip_py, l1_px, obj_tip_py)

        # Refraction from lens 1
        if is_convex:
            # Passes through right focal point F'1
            # Slope = (axis_y - obj_tip_py) / (f1_right_px - l1_px)
            slope1 = (axis_y - obj_tip_py) / max(1.0, (f1_right_px - l1_px))
            ray1_end_x = min(bx + bw - 15.0, max(l1_px + 50.0, img_px + 50.0))
            ray1_end_y = obj_tip_py + slope1 * (ray1_end_x - l1_px)
            painter.drawLine(l1_px, obj_tip_py, ray1_end_x, ray1_end_y)

            # Virtual extension backward if virtual image
            if not is_real and img_px < l1_px:
                painter.setPen(QPen(QColor(239, 68, 68, 140), 1.2, Qt.PenStyle.DashLine))
                painter.drawLine(l1_px, obj_tip_py, img_px, img_tip_py)
        else:
            # Diverging: emerges as if coming from left focal point F1
            slope1 = (obj_tip_py - axis_y) / max(1.0, (l1_px - f1_left_px))
            ray1_end_x = bx + bw - 15.0
            ray1_end_y = obj_tip_py + slope1 * (ray1_end_x - l1_px)
            painter.drawLine(l1_px, obj_tip_py, ray1_end_x, ray1_end_y)
            # Virtual extension back to left focus F1
            painter.setPen(QPen(QColor(239, 68, 68, 140), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(l1_px, obj_tip_py, f1_left_px, axis_y)

        # Ray 2: Central ray through optical center O1 (no deviation)
        slope2 = (axis_y - obj_tip_py) / max(1.0, (l1_px - obj_px))
        painter.setPen(QPen(QColor(34, 197, 94), 1.5))
        painter.drawLine(obj_px, obj_tip_py, l1_px, axis_y)
        ray2_end_x = min(bx + bw - 15.0, max(l1_px + 50.0, img_px + 50.0))
        ray2_end_y = axis_y + slope2 * (ray2_end_x - l1_px)
        painter.drawLine(l1_px, axis_y, ray2_end_x, ray2_end_y)

        # Virtual extension backward for Ray 2 if virtual image
        if not is_real and img_px < l1_px:
            painter.setPen(QPen(QColor(34, 197, 94, 140), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(l1_px, axis_y, img_px, img_tip_py)

    def _paint_abcd_matrix_panel(self, painter: QPainter) -> None:
        """Renders 2x2 Ray Transfer ABCD Matrix and Paraxial Optics Verification."""
        ax = 20.0
        ay = 330.0
        aw = 440.0
        ah = 225.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(ax, ay, aw, ah, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(ax + 12.0, ay + 20.0, "MA TRẬN CHUYỂN TIA 2×2 (RAY TRANSFER ABCD MATRIX):")

        a, b, c, d = self.ray_transfer_matrix
        det_m = a * d - b * c

        # Render 2x2 Matrix Brackets
        mx = ax + 35.0
        my = ay + 45.0
        mw = 180.0
        mh = 80.0

        painter.setPen(QPen(QColor(56, 189, 248), 2.0))
        # Left bracket [
        painter.drawLine(mx, my, mx - 8.0, my)
        painter.drawLine(mx - 8.0, my, mx - 8.0, my + mh)
        painter.drawLine(mx - 8.0, my + mh, mx, my + mh)
        # Right bracket ]
        rx = mx + mw
        painter.drawLine(rx, my, rx + 8.0, my)
        painter.drawLine(rx + 8.0, my, rx + 8.0, my + mh)
        painter.drawLine(rx + 8.0, my + mh, rx, my + mh)

        # Matrix element values
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(mx + 10.0, my + 30.0, f"A = {a:+.2f}")
        painter.drawText(mx + 95.0, my + 30.0, f"B = {b:+.1f}")
        painter.drawText(mx + 10.0, my + 65.0, f"C = {c:+.4f}")
        painter.drawText(mx + 95.0, my + 65.0, f"D = {d:+.2f}")

        # Invariant Explanations
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(148, 163, 184)))
        info_lines = [
            f"• Điều kiện Tạo ảnh: B = 0 mm (vật & ảnh là cặp điểm liên hợp)",
            f"• Độ phóng đại ngang: A = m = {a:+.2f} (A'B' / AB)",
            f"• Định thức Bảo toàn Etendue: det(M) = AD - BC = {det_m:.3f} ≡ 1.0",
            f"• Công suất khúc xạ: C = -1/f = {c:+.4f} mm⁻¹",
        ]
        row_y = ay + 145.0
        for line in info_lines:
            painter.drawText(ax + 16.0, row_y, line)
            row_y += 18.0

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """Displays rigorous analytical optical formulas and physical metrics."""
        mx = 480.0
        my = 330.0
        mw = 450.0
        mh = 225.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(mx, my, mw, mh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(mx + 12.0, my + 20.0, "CÔNG THỨC QUANG HỌC & THÔNG SỐ TẠO ẢNH:")

        metrics = [
            ("Tiêu cự Thấu kính L₁ (Focal Length f₁):", f"f₁ = {self._f1_mm:+.1f} mm (Độ tụ D = {self.optical_power_diopters:+.1f} dp)"),
            ("Khoảng cách Vật đến Thấu kính (d_o):", f"d_o = {self.object_distance_d1_mm:.1f} mm"),
            ("Khoảng cách Ảnh đến Thấu kính (d_i):", f"d_i = (f·d_o)/(d_o - f) = {self.image_distance_d1_mm:+.1f} mm"),
            ("Độ Phóng đại Ngang (Magnification m):", f"m = -d_i / d_o = {self.total_magnification_m:+.2f}"),
            ("Chiều cao Vật AB & Chiều cao Ảnh A'B':", f"h_vật = {self._object_h_mm:.1f} mm | h_ảnh = {self.image_height_h1_mm:+.1f} mm"),
            ("Tính chất Ảnh (Image Nature):", f"{'ẢNH THẬT (Ngược chiều)' if self.is_image1_real else 'ẢNH ẢO (Cùng chiều, Phóng đại)'}"),
            ("Hệ 2 Thấu kính Ghép (Compound System):", f"{'Kính hiển vi / Thiên văn (Active)' if self._is_two_lens else 'Hệ 1 Thấu kính đơn'}"),
        ]

        row_y = my + 44.0
        line_h = 24.0
        for name, val in metrics:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(mx + 14.0, row_y, name)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(226, 232, 240)))
            painter.drawText(mx + 225.0, row_y, val)
            row_y += line_h

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture buttons."""
        buttons_info = [
            ("btn_preset", f"Preset: {self._preset.badge_label}", QColor(14, 165, 233)),
            ("btn_do_down", "do -", QColor(51, 65, 85)),
            ("btn_do_up", "do +", QColor(51, 65, 85)),
            ("btn_f_down", "f -", QColor(51, 65, 85)),
            ("btn_f_up", "f +", QColor(51, 65, 85)),
            ("btn_lens_type", "Hội tụ" if self._f1_mm > 0 else "Phân kỳ", QColor(51, 65, 85)),
            ("btn_p_real", "Ảnh thật", QColor(30, 41, 59)),
            ("btn_p_mag", "Kính lúp", QColor(30, 41, 59)),
            ("btn_p_concave", "Phân kỳ", QColor(30, 41, 59)),
            ("btn_p_micro", "Hiển vi", QColor(30, 41, 59)),
            ("btn_play", "Dừng" if self._is_running else "Chạy", QColor(245, 158, 11)),
            ("btn_stamp", "Ghim", QColor(16, 185, 129)),
        ]

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        for key, text, color in buttons_info:
            rect = self._buttons[key]
            hovered = self._hovered_btn == key
            btn_col = color.lighter(125) if hovered else color

            painter.setBrush(QBrush(btn_col))
            painter.setPen(QPen(QColor(255, 255, 255, 160) if hovered else QColor(0, 0, 0, 80), 1.0))
            painter.drawRoundedRect(rect, 6.0, 6.0)

            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
