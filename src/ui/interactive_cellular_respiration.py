"""
AirTeach AI — Interactive Cellular Respiration, Krebs Cycle & Mitochondrial ETC Simulator
Authentic bioenergetics, 5-complex Electron Transport Chain, chemiosmotic proton motive force,
rotary ATP synthase, metabolic inhibitors (Cyanide, Rotenone, DNP), and touchless air gesture integration.
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
class RespirationPreset:
    code: str
    title_vi: str
    oxygen_level: float       # Fraction [0.0 .. 1.0]
    complex_i_active: bool
    complex_iii_active: bool
    complex_iv_active: bool
    atp_synthase_active: bool
    uncoupled_dnp: bool       # Proton leak without ATP synthesis
    atp_yield: int            # ATP per glucose molecule
    description: str


PRESETS: dict[str, RespirationPreset] = {
    "STANDARD_AEROBIC_RESPIRATION": RespirationPreset(
        code="STANDARD_AEROBIC_RESPIRATION",
        title_vi="Hô hấp hiếu khí tiêu chuẩn (30-32 ATP)",
        oxygen_level=1.0,
        complex_i_active=True,
        complex_iii_active=True,
        complex_iv_active=True,
        atp_synthase_active=True,
        uncoupled_dnp=False,
        atp_yield=32,
        description="Oxy hoạt động như chất nhận electron cuối cùng; bơm proton tạo gradient tổng hợp 30-32 ATP/glucose.",
    ),
    "CYANIDE_POISONING_COMPLEX_IV": RespirationPreset(
        code="CYANIDE_POISONING_COMPLEX_IV",
        title_vi="Ngộ độc Cyanide / CO (Khóa Phức hệ IV)",
        oxygen_level=1.0,
        complex_i_active=True,
        complex_iii_active=True,
        complex_iv_active=False,  # Blocked by CN-
        atp_synthase_active=False,
        uncoupled_dnp=False,
        atp_yield=2,
        description="Cyanide liên kết không thuận nghịch vào Cytochrome a3 của Phức hệ IV làm tê liệt chuỗi ETC và ngừng sinh ATP.",
    ),
    "DNP_UNCOUPLING_THERMOGENESIS": RespirationPreset(
        code="DNP_UNCOUPLING_THERMOGENESIS",
        title_vi="Mất cặp hóa thẩm DNP / Sinh nhiệt (Thermogenesis)",
        oxygen_level=1.0,
        complex_i_active=True,
        complex_iii_active=True,
        complex_iv_active=True,
        atp_synthase_active=False,
        uncoupled_dnp=True,  # Proton gradient short-circuited to heat
        atp_yield=2,
        description="DNP hoặc Thermogenin (UCP1) rò rỉ H+ qua màng; ETC tiêu thụ oxy cực đại nhưng năng lượng chỉ tỏa nhiệt.",
    ),
    "ROTENONE_COMPLEX_I_INHIBITION": RespirationPreset(
        code="ROTENONE_COMPLEX_I_INHIBITION",
        title_vi="Ức chế Phức hệ I (Thuốc trừ sâu Rotenone)",
        oxygen_level=1.0,
        complex_i_active=False,  # Blocked by rotenone
        complex_iii_active=True,
        complex_iv_active=True,
        atp_synthase_active=True,
        uncoupled_dnp=False,
        atp_yield=16,
        description="Rotenone chặn truyền electron từ NADH; chỉ còn FADH2 vào Phức hệ II, làm giảm 50% hiệu suất sinh ATP.",
    ),
    "HYPOXIA_ANAEROBIC_FERMENTATION": RespirationPreset(
        code="HYPOXIA_ANAEROBIC_FERMENTATION",
        title_vi="Thiếu Oxy / Lên men Lactic (Anaerobic Fermentation)",
        oxygen_level=0.0,  # Zero oxygen
        complex_i_active=False,
        complex_iii_active=False,
        complex_iv_active=False,
        atp_synthase_active=False,
        uncoupled_dnp=False,
        atp_yield=2,
        description="Môi trường thiếu oxy làm ngừng trệ ETC; tế bào chuyển sang lên men Lactic ở tế bào chất chỉ thu được 2 ATP.",
    ),
}


class InteractiveCellularRespirationItem(QGraphicsObject):
    """
    Interactive QGraphicsObject for demonstrating Cellular Respiration,
    Mitochondrial Electron Transport Chain (ETC), and rotary ATP Synthase.
    """

    WIDTH = 640.0
    HEIGHT = 380.0

    stamp_requested = Signal()
    oxygen_changed = Signal(float)

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
        self._active_preset_code: str = "STANDARD_AEROBIC_RESPIRATION"
        self._preset: RespirationPreset = PRESETS[self._active_preset_code]

        self._oxygen_level: float = self._preset.oxygen_level
        self._atp_rotor_angle: float = 0.0

        # Dynamic proton particles (H+)
        self._h_plus_particles: list[dict[str, float]] = []
        self._init_proton_particles()

        # Animation Timer (30 FPS)
        self._is_animating: bool = True
        self._anim_timer = QTimer()
        self._anim_timer.setInterval(33)
        self._anim_timer.timeout.connect(self._on_animation_tick)
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
            ("STANDARD_AEROBIC_RESPIRATION", "Hiếu khí"),
            ("CYANIDE_POISONING_COMPLEX_IV", "Cyanide"),
            ("DNP_UNCOUPLING_THERMOGENESIS", "DNP Sinh nhiệt"),
            ("ROTENONE_COMPLEX_I_INHIBITION", "Rotenone"),
            ("HYPOXIA_ANAEROBIC_FERMENTATION", "Lên men Lactic"),
        ]
        for p_code, p_lbl in preset_names:
            pw = 76.0
            self._btn_rects[f"preset_{p_code}"] = QRectF(cur_x, btn_y, pw, btn_h)
            cur_x += pw + 5.0

        # Controls on top right
        r_play = QRectF(cur_x + 8.0, btn_y, 38.0, btn_h)
        self._btn_rects["toggle_anim"] = r_play

        r_o2_down = QRectF(r_play.right() + 5.0, btn_y, 28.0, btn_h)
        self._btn_rects["o2_down"] = r_o2_down

        r_o2_up = QRectF(r_o2_down.right() + 4.0, btn_y, 28.0, btn_h)
        self._btn_rects["o2_up"] = r_o2_up

        r_stamp = QRectF(r_o2_up.right() + 6.0, btn_y, 48.0, btn_h)
        self._btn_rects["stamp"] = r_stamp

        # Slider bounds for O2 level
        self._btn_rects["o2_slider_bounds"] = QRectF(18.0, 75.0, 320.0, 20.0)

    def _init_proton_particles(self) -> None:
        """Initializes flowing H+ proton particles across complexes and ATP synthase."""
        self._h_plus_particles.clear()
        # 24 circulating protons
        for i in range(24):
            self._h_plus_particles.append({
                "x": 30.0 + (i % 6) * 55.0 + (i // 6) * 8.0,
                "y": 115.0 + (i % 4) * 20.0,
                "mode": "pumped" if i < 16 else "returning",
                "phase": (i * 0.25) % 1.0,
            })

    def cleanup(self) -> None:
        """Stops animation timer to avoid memory leaks."""
        if self._anim_timer.isActive():
            self._anim_timer.stop()

    # -------------------------------------------------------------------------
    # Biochemical & Bioenergetics Analytical Engine
    # -------------------------------------------------------------------------
    @property
    def proton_motive_force_mv(self) -> float:
        """
        Computes proton motive force (PMF) Delta p in millivolts (standard ~180-220 mV).
        Delta p collapses if uncoupled by DNP or inhibited by Cyanide/Hypoxia.
        """
        if self._preset.uncoupled_dnp:
            return 25.0  # Collapsed by proton ionophore leak
        if not self._preset.complex_iv_active or self._oxygen_level < 0.1:
            return 35.0  # Collapsed by electron blockage

        base_pmf = 200.0
        if not self._preset.complex_i_active:
            base_pmf *= 0.65  # Only Complex II contributing

        return base_pmf * (0.3 + 0.7 * self._oxygen_level)

    @property
    def atp_synthesis_rate_per_sec(self) -> float:
        """Live ATP production rate in ATP/second."""
        if self._preset.uncoupled_dnp or not self._preset.atp_synthase_active:
            return 0.5  # Only minimal substrate-level phosphorylation
        if self._oxygen_level < 0.1:
            return 1.0  # Lactic fermentation only (2 ATP)

        pmf = self.proton_motive_force_mv
        return max(0.0, (pmf / 200.0) * 16.0)

    @property
    def oxygen_level(self) -> float:
        return self._oxygen_level

    @property
    def active_preset(self) -> RespirationPreset:
        return self._preset

    @property
    def is_animating(self) -> bool:
        return self._is_animating

    def load_preset(self, preset_code: str) -> None:
        """Loads a predefined cellular respiration preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._oxygen_level = self._preset.oxygen_level
        self.oxygen_changed.emit(self._oxygen_level)
        self.update()

    def set_oxygen_level(self, o2_frac: float) -> None:
        """Sets oxygen availability fraction (clamped between 0.0 and 1.0)."""
        self._oxygen_level = max(0.0, min(1.0, float(o2_frac)))
        self.oxygen_changed.emit(self._oxygen_level)
        self.update()

    def toggle_animation(self) -> bool:
        """Toggles animation timer."""
        self._is_animating = not self._is_animating
        if self._is_animating:
            self._anim_timer.start()
        else:
            self._anim_timer.stop()
        self.update()
        return self._is_animating

    def _on_animation_tick(self) -> None:
        """Updates rotary ATP synthase rotation and proton particle positions."""
        if not self._is_animating:
            return

        # Rotor rotation speed proportional to proton motive force
        pmf_frac = self.proton_motive_force_mv / 200.0
        if self._preset.atp_synthase_active and not self._preset.uncoupled_dnp:
            self._atp_rotor_angle = (self._atp_rotor_angle + 6.0 * pmf_frac) % 360.0

        # Move H+ protons
        for p in self._h_plus_particles:
            if p["mode"] == "pumped":
                # Upward pumping from matrix (y=210) to intermembrane space (y=110)
                if self.proton_motive_force_mv > 50.0:
                    p["y"] -= 2.0 * pmf_frac
                    if p["y"] < 110.0:
                        p["y"] = 210.0
            else:
                # Downward stream through ATP Synthase (x=280..320)
                if self._preset.atp_synthase_active and not self._preset.uncoupled_dnp:
                    p["y"] += 2.5 * pmf_frac
                    if p["y"] > 210.0:
                        p["y"] = 110.0
                elif self._preset.uncoupled_dnp:
                    # Direct DNP leak anywhere
                    p["y"] += 3.0
                    if p["y"] > 210.0:
                        p["y"] = 110.0

        self.update()

    # -------------------------------------------------------------------------
    # Touchless Air Gesture Handling & Stamping
    # -------------------------------------------------------------------------
    def stamp_to_board(self) -> None:
        """Stamps active Cellular Respiration diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import BiologyTools

        center = (self.pos().x() + self.WIDTH / 2.0, self.pos().y() + self.HEIGHT + 40.0)
        items = BiologyTools.create_cellular_respiration_diagram(
            center=center,
            preset_code=self._active_preset_code,
            oxygen_level=self._oxygen_level,
            pmf_mv=self.proton_motive_force_mv,
            atp_rate=self.atp_synthesis_rate_per_sec,
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

        # 2. Oxygen steppers
        r_od = self._btn_rects.get("o2_down")
        if r_od and r_od.contains(local_pos):
            self.set_oxygen_level(self._oxygen_level - 0.1)
            return True

        r_ou = self._btn_rects.get("o2_up")
        if r_ou and r_ou.contains(local_pos):
            self.set_oxygen_level(self._oxygen_level + 0.1)
            return True

        # 3. Play/Pause
        r_play = self._btn_rects.get("toggle_anim")
        if r_play and r_play.contains(local_pos):
            self.toggle_animation()
            return True

        # 4. Stamp
        r_stamp = self._btn_rects.get("stamp")
        if r_stamp and r_stamp.contains(local_pos):
            self.stamp_to_board()
            return True

        # 5. Direct Oxygen Slider click
        r_sl = self._btn_rects.get("o2_slider_bounds")
        if r_sl and r_sl.contains(local_pos):
            frac = (local_pos.x() - r_sl.x()) / r_sl.width()
            self.set_oxygen_level(frac)
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

        # 1. Main Background Card
        bg_rect = self.boundingRect()
        painter.setBrush(QBrush(QColor("#090d16")))
        painter.setPen(QPen(QColor("#1e293b"), 1.5))
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # 2. Header Banner
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(0.0, 0.0, self.WIDTH, 36.0), 10.0, 10.0)
        painter.drawRect(QRectF(0.0, 26.0, self.WIDTH, 10.0))

        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.drawText(
            QRectF(14.0, 0.0, self.WIDTH - 28.0, 36.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            f"⚡ HÔ HẤP TẾ BÀO: CHUỖI TRUYỀN ELECTRON (ETC) & ATP SYNTHASE — {self._preset.title_vi}",
        )

        # 3. Preset Buttons
        preset_names = [
            ("STANDARD_AEROBIC_RESPIRATION", "Hiếu khí"),
            ("CYANIDE_POISONING_COMPLEX_IV", "Cyanide"),
            ("DNP_UNCOUPLING_THERMOGENESIS", "DNP Sinh nhiệt"),
            ("ROTENONE_COMPLEX_I_INHIBITION", "Rotenone"),
            ("HYPOXIA_ANAEROBIC_FERMENTATION", "Lên men Lactic"),
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
        r_play = self._btn_rects["toggle_anim"]
        painter.setBrush(QBrush(QColor("#10b981") if self._is_animating else QColor("#334155")))
        painter.setPen(QPen(QColor("#34d399"), 1.0))
        painter.drawRoundedRect(r_play, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(r_play, Qt.AlignmentFlag.AlignCenter, "⏸" if self._is_animating else "▶")

        r_od = self._btn_rects["o2_down"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_od, 4.0, 4.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(r_od, Qt.AlignmentFlag.AlignCenter, "O₂-")

        r_ou = self._btn_rects["o2_up"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_ou, 4.0, 4.0)
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(r_ou, Qt.AlignmentFlag.AlignCenter, "O₂+")

        r_stamp = self._btn_rects["stamp"]
        painter.setBrush(QBrush(QColor("#7c3aed")))
        painter.setPen(QPen(QColor("#a78bfa"), 1.0))
        painter.drawRoundedRect(r_stamp, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng")

        # ---------------------------------------------------------------------
        # 4. Mitochondrial Membrane & ETC Apparatus (Left Panel)
        # ---------------------------------------------------------------------
        mem_x = 18.0
        mem_y = 75.0
        mem_w = 340.0
        mem_h = 195.0

        mem_rect = QRectF(mem_x, mem_y, mem_w, mem_h)
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(mem_rect, 6.0, 6.0)

        # Region labels: Intermembrane Space (IMS) vs Matrix
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#f87171"))
        painter.drawText(QPointF(mem_x + 8.0, mem_y + 14.0), "KHOANG GIAN MÀNG (Intermembrane Space) — [H+] cao, pH ≈ 7.0")

        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(mem_x + 8.0, mem_y + mem_h - 8.0), "CHẤT NỀN TY THỂ (Matrix) — Chu trình Krebs, [H+] thấp, pH ≈ 8.0")

        # Phospholipid Bilayer (Inner Membrane)
        bilayer_y = mem_y + 70.0
        bilayer_h = 45.0
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRect(QRectF(mem_x, bilayer_y, mem_w, bilayer_h))

        # Complexes I, II, III, IV, and ATP Synthase
        complex_defs = [
            ("I", mem_x + 25.0, 42.0, 60.0, QColor("#3b82f6"), self._preset.complex_i_active),
            ("II", mem_x + 80.0, 32.0, 44.0, QColor("#eab308"), True),
            ("III", mem_x + 125.0, 42.0, 60.0, QColor("#10b981"), self._preset.complex_iii_active),
            ("IV", mem_x + 180.0, 44.0, 62.0, QColor("#ec4899"), self._preset.complex_iv_active),
            ("V", mem_x + 255.0, 52.0, 75.0, QColor("#a855f7"), self._preset.atp_synthase_active),
        ]

        for c_name, cx_pos, cw, ch, col, is_act in complex_defs:
            cy_pos = bilayer_y + bilayer_h * 0.5 - ch * 0.5
            cr = QRectF(cx_pos, cy_pos, cw, ch)

            # Dim if inhibited
            fill_col = col if is_act else QColor("#475569")
            painter.setBrush(QBrush(fill_col))
            painter.setPen(QPen(QColor("#ffffff") if is_act else QColor("#ef4444"), 1.5 if is_act else 2.0))
            painter.drawRoundedRect(cr, 6.0, 6.0)

            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_act else QColor("#fca5a5"))
            c_title = f"Phức hệ {c_name}" if c_name != "V" else "ATP Synthase"
            painter.drawText(cr, Qt.AlignmentFlag.AlignCenter, f"{c_name}\n{'✔' if is_act else '✘'}")

        # Cyanide warning icon over Complex IV if inhibited
        if not self._preset.complex_iv_active:
            cn_badge = QRectF(mem_x + 175.0, bilayer_y - 24.0, 54.0, 18.0)
            painter.setBrush(QBrush(QColor("#7f1d1d")))
            painter.setPen(QPen(QColor("#ef4444"), 1.0))
            painter.drawRoundedRect(cn_badge, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#fecaca"))
            painter.drawText(cn_badge, Qt.AlignmentFlag.AlignCenter, "CN⁻ Khóa!")

        # DNP leak warning if uncoupled
        if self._preset.uncoupled_dnp:
            dnp_badge = QRectF(mem_x + mem_w - 110.0, mem_y + 20.0, 100.0, 18.0)
            painter.setBrush(QBrush(QColor("#854d0e")))
            painter.setPen(QPen(QColor("#eab308"), 1.0))
            painter.drawRoundedRect(dnp_badge, 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#fef08a"))
            painter.drawText(dnp_badge, Qt.AlignmentFlag.AlignCenter, "🔥 DNP Thoát nhiệt!")

        # Render streaming H+ Protons
        for p in self._h_plus_particles:
            px = p["x"]
            py = p["y"]
            painter.setBrush(QBrush(QColor("#f59e0b")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(px, py), 3.0, 3.0)
            painter.setFont(QFont("Segoe UI", 5, QFont.Weight.Bold))
            painter.setPen(QColor("#000000"))
            painter.drawText(QRectF(px - 4.0, py - 4.0, 8.0, 8.0), Qt.AlignmentFlag.AlignCenter, "+")

        # ---------------------------------------------------------------------
        # 5. Diagnostic Bioenergetics & ATP Yield Ledger (Right Panel)
        # ---------------------------------------------------------------------
        card_x = mem_x + mem_w + 14.0
        card_y = mem_y
        card_w = self.WIDTH - card_x - 14.0
        card_h = mem_h

        card_rect = QRectF(card_x, card_y, card_w, card_h)
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(card_rect, 6.0, 6.0)

        pmf = self.proton_motive_force_mv
        atp_rate = self.atp_synthesis_rate_per_sec

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(card_x + 10.0, card_y + 18.0), "THÔNG SỐ NĂNG LƯỢNG SINH HỌC")

        metrics_lines = [
            (f"• Nồng độ Oxy O₂: {self._oxygen_level * 100:.0f}%", QColor("#38bdf8"), True),
            (f"• Lực động proton (PMF): {pmf:.1f} mV", QColor("#10b981") if pmf > 100.0 else QColor("#ef4444"), True),
            (f"• Tốc độ sinh ATP: {atp_rate:.1f} ATP/s", QColor("#fbbf24"), True),
            (f"• Năng suất lý thuyết: {self._preset.atp_yield} ATP/glucose", QColor("#a78bfa"), True),
            (f"• Đường phân (Glycolysis): 2 ATP + 2 NADH", QColor("#cbd5e1"), False),
            (f"• Chu trình Krebs: 2 ATP + 6 NADH + 2 FADH₂", QColor("#cbd5e1"), False),
            (f"• Chuỗi ETC & Hóa thẩm: ~26-28 ATP", QColor("#34d399"), False),
            (f"• Trạng thái: {self._preset.title_vi[:26]}", QColor("#f472b6"), True),
        ]
        my = card_y + 36.0
        for m_txt, m_col, is_bld in metrics_lines:
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_bld else QFont.Weight.Normal))
            painter.setPen(m_col)
            painter.drawText(QPointF(card_x + 10.0, my), m_txt)
            my += 18.0

        # ---------------------------------------------------------------------
        # 6. Theory & Mitchell Chemiosmosis Principles Card (Bottom Panel)
        # ---------------------------------------------------------------------
        bot_y = mem_y + mem_h + 12.0
        bot_h = self.HEIGHT - bot_y - 12.0
        bot_rect = QRectF(14.0, bot_y, self.WIDTH - 28.0, bot_h)

        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(bot_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(22.0, bot_y + 16.0), "THUYẾT HÓA THẨM MITCHELL & NĂNG LƯỢNG HỌC TY THỂ:")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        b_desc = (
            f"• Phương trình tổng quát: C₆H₁₂O₆ + 6 O₂ ➔ 6 CO₂ + 6 H₂O + 30-32 ATP (ΔG° = -2870 kJ/mol).\n"
            f"• Cơ chế hóa thẩm (Chemiosmosis): Phức hệ I, III, IV bơm H+ từ chất nền vào khoang gian màng tạo gradient điện hóa (PMF ≈ 200 mV) quay rotor ATP Synthase sinh năng lượng.\n"
            f"• Đặc điểm mô phỏng: {self._preset.description}"
        )
        painter.drawText(
            QRectF(22.0, bot_y + 22.0, bot_rect.width() - 20.0, 48.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            b_desc,
        )
