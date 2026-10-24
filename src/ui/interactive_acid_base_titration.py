"""
AirTeach AI - Interactive Acid-Base Titration & pH Buffer Curve Simulator
========================================================================
Provides an authentic analytical chemistry simulation of acid-base titrations:
  - Systems:
      1. Weak Acid - Strong Base (CH₃COOH + NaOH, pKa = 4.76)
      2. Strong Acid - Strong Base (HCl + NaOH, equivalence at pH = 7.00)
  - Theoretical analytical pH equations:
      - Initial weak acid dissociation: [H+] = sqrt(Ka * Ca)
      - Buffer Region (Vùng đệm) via Henderson-Hasselbalch:
          pH = pKa + log10([A-] / [HA])
      - Half-equivalence point (Nửa điểm tương đương, V = 0.5 * Veq):
          [A-] = [HA] => pH = pKa (direct measurement of dissociation constant)
      - Equivalence point (Điểm tương đương, V = Veq):
          Salt hydrolysis: A- + H2O <=> HA + OH- => pH = 8.73 (basic salt)
      - Post-equivalence: excess strong base OH-
  - Real-time Color Changing Chemical Indicators:
      - Phenolphthalein: colorless (pH < 8.2) -> vivid magenta/pink (pH >= 8.2)
      - Methyl Orange: red (pH < 3.1) -> orange -> yellow (pH > 4.4)
      - Universal: Red -> Orange -> Yellow -> Green -> Blue -> Purple
  - Laboratory Apparatus visualization:
      Burette with fluid graduations, stopcock valve, animated drops,
      Erlenmeyer flask with spinning magnetic stir bar and digital pH probe.
  - Real-time Sigmoidal Titration Curve & First Derivative (dpH/dV) peak.
  - Touchless air pinch dragging on titrant volume and air tap controls.
  - Vector diagram stamping onto LessonBoard via ChemistryTools.create_titration_curve_diagram.
"""
from __future__ import annotations

import logging
import math
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
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

from src.ui.stem_tools import ChemistryTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


class InteractiveAcidBaseTitrationItem(QGraphicsObject):
    """
    Interactive Acid-Base Titration & pH Henderson-Hasselbalch Buffer Curve Simulator.
    """

    titration_updated = Signal(float, float)  # (volume_titrant_ml, ph_value)
    acid_changed = Signal(str)                # ("WEAK_ACID", "STRONG_ACID")
    indicator_changed = Signal(str)           # ("PHENOLPHTHALEIN", "METHYL_ORANGE", "UNIVERSAL")
    stamp_requested = Signal()

    def __init__(
        self,
        pos: tuple[float, float] = (460.0, 280.0),
        width: float = 760.0,
        height: float = 490.0,
        acid_type: str = "WEAK_ACID",
        indicator: str = "PHENOLPHTHALEIN",
        v_titrant: float = 0.0,
        c_acid: float = 0.10,
        c_base: float = 0.10,
        v_acid_init: float = 25.0,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(95.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._width = float(width)
        self._height = float(height)

        # Chemical parameters
        self._acid_type = acid_type.upper() if acid_type.upper() in ("WEAK_ACID", "STRONG_ACID") else "WEAK_ACID"
        self._indicator = indicator.upper() if indicator.upper() in (
            "PHENOLPHTHALEIN", "METHYL_ORANGE", "UNIVERSAL"
        ) else "PHENOLPHTHALEIN"

        self._c_acid = float(c_acid)       # 0.10 M
        self._c_base = float(c_base)       # 0.10 M
        self._v_acid_init = float(v_acid_init) # 25.0 mL
        self._pka = 4.76                   # Acetic acid pKa

        # Current State
        self._v_titrant = max(0.0, min(50.0, float(v_titrant))) # mL added
        self._stir_angle = 0.0             # Magnetic stir bar rotation
        self._is_dispensing = False        # Auto-titrant dispensing flow

        # Animation Timer (35 FPS)
        self._timer = QTimer()
        self._timer.setInterval(28)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

        # Droplet animation state
        self._drop_anim_y = 0.0
        self._is_dropping = False

        # Interactive Button Rectangles (local coordinates)
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)

        # Row 1 Controls (y = 360)
        self._rect_btn_drop_small = QRectF(30.0, 360.0, 95.0, 26.0)
        self._rect_btn_drop_large = QRectF(130.0, 360.0, 95.0, 26.0)
        self._rect_btn_auto_flow = QRectF(230.0, 360.0, 110.0, 26.0)
        self._rect_btn_acid_toggle = QRectF(350.0, 360.0, 195.0, 26.0)
        self._rect_btn_reset = QRectF(555.0, 360.0, 85.0, 26.0)
        self._rect_btn_stamp = QRectF(645.0, 360.0, 85.0, 26.0)

        # Row 2 Controls (Indicators, y = 398)
        self._rect_btn_ind_phenol = QRectF(30.0, 398.0, 160.0, 26.0)
        self._rect_btn_ind_methyl = QRectF(195.0, 398.0, 150.0, 26.0)
        self._rect_btn_ind_univ = QRectF(350.0, 398.0, 130.0, 26.0)

    # ─── Public Properties & Accessors ───────────────────────────────────────

    @property
    def acid_type(self) -> str:
        return self._acid_type

    @property
    def indicator(self) -> str:
        return self._indicator

    @property
    def v_titrant_ml(self) -> float:
        return self._v_titrant

    @property
    def v_eq_ml(self) -> float:
        """Equivalence point volume: Veq = (Ca * Va) / Cb."""
        return (self._c_acid * self._v_acid_init) / self._c_base

    @property
    def current_ph(self) -> float:
        """Calculates authentic analytical chemical pH at current titrant volume."""
        return self.calculate_ph_at_volume(self._v_titrant)

    @property
    def is_dispensing(self) -> bool:
        return self._is_dispensing

    # ─── Mathematical Analytical Titration Solver ───────────────────────────

    def calculate_ph_at_volume(self, v_b: float) -> float:
        """
        Calculates exact analytical pH using Henderson-Hasselbalch equation,
        strong acid dilution, and salt hydrolysis equations.
        """
        v = max(0.0, min(50.0, v_b))
        v_eq = self.v_eq_ml
        c_a = self._c_acid
        c_b = self._c_base
        v_a = self._v_acid_init

        if self._acid_type == "WEAK_ACID":
            pka = self._pka
            if v <= 0.001:
                # Initial weak acid dissociation: [H+] = sqrt(Ka * Ca)
                return 0.5 * (pka - math.log10(c_a))

            elif v < v_eq - 0.08:
                # Buffer Region: Henderson-Hasselbalch equation
                # moles_salt = Cb * Vb, moles_acid_rem = Ca * Va - Cb * Vb
                moles_salt = c_b * v
                moles_acid = max(1e-7, c_a * v_a - c_b * v)
                ratio = moles_salt / moles_acid
                return pka + math.log10(ratio)

            elif abs(v - v_eq) <= 0.08:
                # Equivalence Point: Acetate salt hydrolysis (CH3COO- + H2O <=> CH3COOH + OH-)
                c_salt = (c_a * v_a) / (v_a + v_eq)
                ka = 10.0**(-pka)
                kb = 1e-14 / ka
                poh = 0.5 * (-math.log10(kb) - math.log10(c_salt))
                return 14.0 - poh  # ≈ 8.73

            else:
                # Post-Equivalence: Driven by excess strong base NaOH
                excess_moles_oh = c_b * (v - v_eq)
                total_vol = v_a + v
                conc_oh = excess_moles_oh / total_vol
                poh = -math.log10(max(1e-7, conc_oh))
                return min(13.6, 14.0 - poh)

        else:
            # STRONG ACID (HCl + NaOH)
            if v < v_eq - 0.05:
                # Pre-Equivalence: Excess H+ from strong acid
                excess_moles_h = c_a * v_a - c_b * v
                total_vol = v_a + v
                conc_h = excess_moles_h / total_vol
                return max(1.0, -math.log10(max(1e-7, conc_h)))

            elif abs(v - v_eq) <= 0.05:
                # Equivalence Point: Pure neutral salt NaCl
                return 7.00

            else:
                # Post-Equivalence: Excess OH- from strong base
                excess_moles_oh = c_b * (v - v_eq)
                total_vol = v_a + v
                conc_oh = excess_moles_oh / total_vol
                poh = -math.log10(max(1e-7, conc_oh))
                return min(13.8, 14.0 - poh)

    def calculate_ph_derivative(self, v_b: float, dv: float = 0.2) -> float:
        """Calculates numerical first derivative dpH / dV for equivalence peak detection."""
        v = max(dv, min(50.0 - dv, v_b))
        ph_plus = self.calculate_ph_at_volume(v + dv)
        ph_minus = self.calculate_ph_at_volume(v - dv)
        return (ph_plus - ph_minus) / (2.0 * dv)

    def get_solution_flask_color(self) -> QColor:
        """
        Determines the visual liquid color inside the Erlenmeyer flask
        based on active indicator and current pH.
        """
        ph = self.current_ph

        if self._indicator == "PHENOLPHTHALEIN":
            if ph < 8.2:
                # Colorless / translucent water
                return QColor(226, 232, 240, 60)
            elif ph <= 10.0:
                # Transition smoothly into magenta/pink
                ratio = (ph - 8.2) / 1.8
                alpha = int(80 + ratio * 140)
                return QColor(244, 63, 94, alpha)
            else:
                # Vibrant fuchsia/pink
                return QColor(225, 29, 72, 220)

        elif self._indicator == "METHYL_ORANGE":
            if ph < 3.1:
                return QColor(239, 68, 68, 200)  # Red
            elif ph <= 4.4:
                return QColor(249, 115, 22, 210) # Orange
            else:
                return QColor(234, 179, 8, 210)  # Yellow

        else:
            # UNIVERSAL INDICATOR
            if ph <= 3.0:
                return QColor(239, 68, 68, 200)  # Red
            elif ph <= 5.0:
                return QColor(249, 115, 22, 200) # Orange
            elif ph <= 6.5:
                return QColor(132, 204, 22, 200) # Yellow-Green
            elif ph <= 7.5:
                return QColor(16, 185, 129, 210) # Neutral Green
            elif ph <= 9.0:
                return QColor(6, 182, 212, 210)  # Cyan
            elif ph <= 11.0:
                return QColor(59, 130, 246, 210) # Blue
            else:
                return QColor(139, 92, 246, 220) # Deep Violet

    # ─── Simulation Controls & Step ──────────────────────────────────────────

    def add_titrant(self, volume_ml: float) -> None:
        """Adds specific titrant volume in mL, animating falling droplet."""
        self._v_titrant = max(0.0, min(50.0, self._v_titrant + volume_ml))
        self._is_dropping = True
        self._drop_anim_y = 0.0
        self.titration_updated.emit(self._v_titrant, self.current_ph)
        self.update()

    def set_titrant_volume(self, volume_ml: float) -> None:
        """Scrubs titrant volume directly."""
        self._v_titrant = max(0.0, min(50.0, float(volume_ml)))
        self.titration_updated.emit(self._v_titrant, self.current_ph)
        self.update()

    def toggle_auto_dispensing(self) -> None:
        self._is_dispensing = not self._is_dispensing
        self.update()

    def set_acid_type(self, acid_type: str) -> None:
        m = acid_type.upper()
        if m in ("WEAK_ACID", "STRONG_ACID") and m != self._acid_type:
            self._acid_type = m
            self.acid_changed.emit(self._acid_type)
            self.titration_updated.emit(self._v_titrant, self.current_ph)
            self.update()

    def toggle_acid_type(self) -> None:
        new_type = "STRONG_ACID" if self._acid_type == "WEAK_ACID" else "WEAK_ACID"
        self.set_acid_type(new_type)

    def set_indicator(self, indicator: str) -> None:
        ind = indicator.upper()
        if ind in ("PHENOLPHTHALEIN", "METHYL_ORANGE", "UNIVERSAL") and ind != self._indicator:
            self._indicator = ind
            self.indicator_changed.emit(self._indicator)
            self.update()

    def reset_titration(self) -> None:
        """Resets titrant volume to 0.0 mL."""
        self._v_titrant = 0.0
        self._is_dispensing = False
        self._is_dropping = False
        self.titration_updated.emit(self._v_titrant, self.current_ph)
        self.update()

    def _on_sim_tick(self) -> None:
        # Spin magnetic stir bar
        self._stir_angle = (self._stir_angle + 18.0) % 360.0

        # Auto-dispensing flow
        if self._is_dispensing:
            if self._v_titrant < 50.0:
                self.add_titrant(0.20)
            else:
                self._is_dispensing = False

        # Droplet gravity animation
        if self._is_dropping:
            self._drop_anim_y += 5.0
            if self._drop_anim_y > 45.0:
                self._is_dropping = False

        self.update()

    # ─── Stamping to LessonBoard ─────────────────────────────────────────────

    def stamp_to_canvas(self, board: Optional[LessonBoard] = None) -> list[QGraphicsItem]:
        """
        Stamps laboratory titration setup and pH curve vector diagram onto LessonBoard.
        """
        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        items = ChemistryTools.create_titration_curve_diagram(
            center=(cx, cy),
            width=420.0,
            height=260.0,
            acid_type=self._acid_type,
            v_titrant=self._v_titrant,
            v_eq=self.v_eq_ml,
            c_acid=self._c_acid,
            c_base=self._c_base,
        )

        target_board = board
        if target_board is None:
            scene = self.scene()
            if scene is not None:
                if hasattr(scene, "add_item_with_undo"):
                    target_board = scene
                else:
                    target_board = getattr(scene, "board", None)

        if target_board and hasattr(target_board, "add_item_with_undo"):
            for item in items:
                target_board.add_item_with_undo(item)
        elif self.scene():
            for item in items:
                self.scene().addItem(item)

        self.stamp_requested.emit()
        return items

    # ─── Touchless Air Gestures ──────────────────────────────────────────────

    def handle_air_pinch(self, scene_pos: QPointF, is_pressed: bool) -> bool:
        """
        Handles touchless air pinch gesture for scrubbing titrant volume
        by dragging the burette liquid level.
        """
        local_pos = self.mapFromScene(scene_pos)

        # Burette interactive zone: x in [45, 115], y in [55, 230]
        if 40.0 <= local_pos.x() <= 120.0 and 55.0 <= local_pos.y() <= 230.0:
            if is_pressed:
                # Map vertical coordinate to 0 - 50 mL
                ratio = (local_pos.y() - 55.0) / 160.0
                target_v = max(0.0, min(50.0, ratio * 50.0))
                self.set_titrant_volume(target_v)
                return True

        return False

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[LessonBoard] = None) -> bool:
        """
        Handles touchless air tap gesture for clicking control buttons.
        """
        local_pos = self.mapFromScene(scene_pos)

        if self._rect_btn_close.contains(local_pos):
            self.close()
            return True
        elif self._rect_btn_drop_small.contains(local_pos):
            self.add_titrant(0.50)
            return True
        elif self._rect_btn_drop_large.contains(local_pos):
            self.add_titrant(2.00)
            return True
        elif self._rect_btn_auto_flow.contains(local_pos):
            self.toggle_auto_dispensing()
            return True
        elif self._rect_btn_acid_toggle.contains(local_pos):
            self.toggle_acid_type()
            return True
        elif self._rect_btn_ind_phenol.contains(local_pos):
            self.set_indicator("PHENOLPHTHALEIN")
            return True
        elif self._rect_btn_ind_methyl.contains(local_pos):
            self.set_indicator("METHYL_ORANGE")
            return True
        elif self._rect_btn_ind_univ.contains(local_pos):
            self.set_indicator("UNIVERSAL")
            return True
        elif self._rect_btn_reset.contains(local_pos):
            self.reset_titration()
            return True
        elif self._rect_btn_stamp.contains(local_pos):
            self.stamp_to_canvas(board)
            return True

        return False

    def close(self) -> None:
        """Stops animation timer and removes item from scene."""
        self._timer.stop()
        scene = self.scene()
        if scene:
            scene.removeItem(self)

    # ─── Mouse Event Handlers ────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        local_pos = event.pos()

        if self._rect_btn_close.contains(local_pos):
            self.close()
            event.accept()
            return
        elif self._rect_btn_drop_small.contains(local_pos):
            self.add_titrant(0.50)
            event.accept()
            return
        elif self._rect_btn_drop_large.contains(local_pos):
            self.add_titrant(2.00)
            event.accept()
            return
        elif self._rect_btn_auto_flow.contains(local_pos):
            self.toggle_auto_dispensing()
            event.accept()
            return
        elif self._rect_btn_acid_toggle.contains(local_pos):
            self.toggle_acid_type()
            event.accept()
            return
        elif self._rect_btn_ind_phenol.contains(local_pos):
            self.set_indicator("PHENOLPHTHALEIN")
            event.accept()
            return
        elif self._rect_btn_ind_methyl.contains(local_pos):
            self.set_indicator("METHYL_ORANGE")
            event.accept()
            return
        elif self._rect_btn_ind_univ.contains(local_pos):
            self.set_indicator("UNIVERSAL")
            event.accept()
            return
        elif self._rect_btn_reset.contains(local_pos):
            self.reset_titration()
            event.accept()
            return
        elif self._rect_btn_stamp.contains(local_pos):
            self.stamp_to_canvas()
            event.accept()
            return

        # Check click on burette column
        if 65.0 <= local_pos.x() <= 95.0 and 55.0 <= local_pos.y() <= 220.0:
            ratio = (local_pos.y() - 55.0) / 160.0
            self.set_titrant_volume(ratio * 50.0)
            event.accept()
            return

        super().mousePressEvent(event)

    # ─── Bounding Rect & Painting ────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, self._height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Slate Frosted Glass Container
        bg_path = QPainterPath()
        bg_path.addRoundedRect(QRectF(0.0, 0.0, self._width, self._height), 14.0, 14.0)
        painter.fillPath(bg_path, QBrush(QColor(15, 23, 42, 235)))  # Slate 900
        painter.strokePath(bg_path, QPen(QColor(51, 65, 85, 200), 1.5))

        # 2. Header Bar
        title_font = QFont("Segoe UI", 11, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(22, 28, "🧪 CHUẨN ĐỘ AXIT - BAZƠ & ĐƯỜNG CONG pH")

        # Acid System Tag
        sys_name = "CH₃COOH + NaOH (Axit yếu - Bazơ mạnh)" if self._acid_type == "WEAK_ACID" else "HCl + NaOH (Axit mạnh - Bazơ mạnh)"
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor("#a855f7") if self._acid_type == "WEAK_ACID" else QColor("#38bdf8"))
        painter.drawText(390, 28, f"● {sys_name}")

        # Close button [✕]
        painter.setPen(QPen(QColor("#64748b"), 1.5))
        painter.setBrush(QBrush(QColor(30, 41, 59, 180)))
        painter.drawRoundedRect(self._rect_btn_close, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(self._rect_btn_close, Qt.AlignmentFlag.AlignCenter, "✕")

        # 3. Left Panel: Laboratory Apparatus (Burette, Flask, Stirrer, pH Probe)
        self._paint_apparatus(painter)

        # 4. Right Panel: Dynamic Sigmoidal Titration Curve & Indicator HUD
        self._paint_titration_curve_panel(painter)

        # 5. Bottom Control Deck & Interactive Buttons
        self._paint_control_deck(painter)

        painter.restore()

    def _paint_apparatus(self, painter: QPainter) -> None:
        """Paints laboratory burette, falling drops, Erlenmeyer flask, and pH probe."""
        app_rect = QRectF(25.0, 48.0, 260.0, 295.0)
        painter.setPen(QPen(QColor(51, 65, 85, 140), 1.0))
        painter.setBrush(QBrush(QColor(30, 41, 59, 140)))
        painter.drawRoundedRect(app_rect, 8.0, 8.0)

        # Laboratory Apparatus Center
        ax = 90.0

        # Stand Column & Ring Clamp
        painter.setPen(QPen(QColor("#475569"), 4.0))
        painter.drawLine(ax - 40.0, 60.0, ax - 40.0, 310.0)
        painter.drawLine(ax - 55.0, 310.0, ax - 15.0, 310.0)  # Heavy base
        painter.setPen(QPen(QColor("#64748b"), 2.0))
        painter.drawLine(ax - 40.0, 110.0, ax - 8.0, 110.0)   # Clamp arm

        # Glass Burette Column
        bur_w = 14.0
        bur_h = 135.0
        bur_y = 60.0
        painter.setPen(QPen(QColor("#cbd5e1"), 1.5))
        painter.setBrush(QBrush(QColor(241, 245, 249, 120)))
        painter.drawRect(QRectF(ax - bur_w / 2.0, bur_y, bur_w, bur_h))

        # Liquid level in burette
        pct_left = max(0.0, min(1.0, 1.0 - self._v_titrant / 50.0))
        fluid_h = bur_h * pct_left
        if fluid_h > 2.0:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(147, 197, 253, 160)))  # NaOH solution
            painter.drawRect(QRectF(ax - bur_w / 2.0 + 1.0, bur_y + bur_h - fluid_h, bur_w - 2.0, fluid_h))

        # Graduations
        painter.setPen(QPen(QColor("#94a3b8"), 1.0))
        for i in range(6):
            gy = bur_y + (i / 5.0) * bur_h
            painter.drawLine(ax + bur_w / 2.0 - 4.0, gy, ax + bur_w / 2.0, gy)

        # Stopcock Valve
        valve_rect = QRectF(ax - 10.0, bur_y + bur_h + 1.0, 20.0, 6.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#0f172a")))
        painter.drawRoundedRect(valve_rect, 2.0, 2.0)

        # Falling Droplet Animation
        if self._is_dropping:
            drop_y = bur_y + bur_h + 12.0 + self._drop_anim_y
            painter.setBrush(QBrush(QColor(147, 197, 253, 220)))
            painter.drawEllipse(QPointF(ax, drop_y), 2.5, 3.5)

        # Erlenmeyer Conical Flask
        flask_top_y = bur_y + bur_h + 24.0
        flask_top_w = 18.0
        flask_bot_w = 78.0
        flask_h = 80.0

        flask_path = QPainterPath()
        flask_path.moveTo(ax - flask_top_w / 2.0, flask_top_y)
        flask_path.lineTo(ax + flask_top_w / 2.0, flask_top_y)
        flask_path.lineTo(ax + flask_bot_w / 2.0, flask_top_y + flask_h)
        flask_path.lineTo(ax - flask_bot_w / 2.0, flask_top_y + flask_h)
        flask_path.closeSubpath()

        # Dynamic Flask Liquid Color
        flask_col = self.get_solution_flask_color()
        painter.setPen(QPen(QColor("#e2e8f0"), 2.0))
        painter.setBrush(QBrush(flask_col))
        painter.drawPath(flask_path)

        # Magnetic Stir Bar at bottom of flask
        stir_cx = ax
        stir_cy = flask_top_y + flask_h - 8.0
        painter.save()
        painter.translate(stir_cx, stir_cy)
        painter.rotate(self._stir_angle)
        painter.setPen(QPen(QColor("#0f172a"), 1.0))
        painter.setBrush(QBrush(QColor("#ffffff")))
        painter.drawRoundedRect(QRectF(-8.0, -2.5, 16.0, 5.0), 2.0, 2.0)
        painter.restore()

        # Digital pH Meter Display on right side of apparatus pane
        meter_rect = QRectF(160.0, 65.0, 115.0, 95.0)
        painter.setPen(QPen(QColor(51, 65, 85, 120), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 200)))
        painter.drawRoundedRect(meter_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(meter_rect.x() + 8.0, meter_rect.y() + 16.0, "pH Meter Kỹ thuật số")

        # Giant digital pH readout
        painter.setFont(QFont("Consolas", 18, QFont.Bold))
        ph_val = self.current_ph
        # Color match indicator
        painter.setPen(QColor("#22c55e") if abs(ph_val - 7.0) <= 0.5 else (QColor("#f43f5e") if ph_val > 8.0 else QColor("#ef4444")))
        painter.drawText(meter_rect.x() + 10.0, meter_rect.y() + 48.0, f"{ph_val:5.2f}")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(meter_rect.x() + 8.0, meter_rect.y() + 68.0, f"V_NaOH: {self._v_titrant:.1f} mL")
        painter.drawText(meter_rect.x() + 8.0, meter_rect.y() + 84.0, f"Chỉ thị: {self._indicator[:7]}..")

    def _paint_titration_curve_panel(self, painter: QPainter) -> None:
        """Paints sigmoidal pH titration curve, equivalence points, and derivative peak."""
        plot_rect = QRectF(295.0, 48.0, 440.0, 295.0)
        painter.setPen(QPen(QColor(51, 65, 85, 140), 1.0))
        painter.setBrush(QBrush(QColor(30, 41, 59, 140)))
        painter.drawRoundedRect(plot_rect, 8.0, 8.0)

        # Plot boundary
        p_x = plot_rect.x() + 45.0
        p_y = plot_rect.y() + 15.0
        p_w = plot_rect.width() - 60.0
        p_h = plot_rect.height() - 55.0

        # Axes background
        painter.setPen(QPen(QColor(51, 65, 85, 100), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 180)))
        painter.drawRect(QRectF(p_x, p_y, p_w, p_h))

        # Y Axis Ticks (pH 0, 7, 14)
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(p_x - 36.0, p_y + 8.0, "pH 14")
        painter.drawText(p_x - 30.0, p_y + p_h / 2.0 + 4.0, "pH 7")
        painter.drawText(p_x - 30.0, p_y + p_h, "pH 0")

        # Horizontal dashed line at pH = 7
        painter.setPen(QPen(QColor(100, 116, 139, 100), 1.0, Qt.DashLine))
        painter.drawLine(p_x, p_y + p_h / 2.0, p_x + p_w, p_y + p_h / 2.0)

        # Vertical dashed line at Equivalence Point V = 25 mL
        v_eq = self.v_eq_ml
        eq_px = p_x + (v_eq / 50.0) * p_w
        painter.setPen(QPen(QColor("#ef4444"), 1.2, Qt.DashLine))
        painter.drawLine(eq_px, p_y, eq_px, p_y + p_h)

        # Continuous Titration Curve
        curve_path = QPainterPath()
        num_pts = 80
        for i in range(num_pts + 1):
            vb = (i / float(num_pts)) * 50.0
            ph = self.calculate_ph_at_volume(vb)
            cx = p_x + (vb / 50.0) * p_w
            cy = p_y + p_h - (ph / 14.0) * p_h
            if i == 0:
                curve_path.moveTo(cx, cy)
            else:
                curve_path.lineTo(cx, cy)

        curve_col = QColor("#a855f7") if self._acid_type == "WEAK_ACID" else QColor("#38bdf8")
        painter.setPen(QPen(curve_col, 2.2))
        painter.drawPath(curve_path)

        # Equivalence Point Marker E(Veq, pHeq)
        ph_eq = 8.73 if self._acid_type == "WEAK_ACID" else 7.00
        ey = p_y + p_h - (ph_eq / 14.0) * p_h
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.setBrush(QBrush(QColor("#ef4444")))
        painter.drawEllipse(QPointF(eq_px, ey), 4.5, 4.5)

        # Half-equivalence marker (Weak acid buffer maximum capacity)
        if self._acid_type == "WEAK_ACID":
            hx = p_x + ((0.5 * v_eq) / 50.0) * p_w
            hy = p_y + p_h - (self._pka / 14.0) * p_h
            painter.setBrush(QBrush(QColor("#10b981")))
            painter.drawEllipse(QPointF(hx, hy), 3.5, 3.5)

            painter.setFont(QFont("Segoe UI", 6, QFont.Bold))
            painter.setPen(QColor("#10b981"))
            painter.drawText(hx - 20.0, hy + 14.0, f"pH=pKa={self._pka:.2f}")

        # Current Indicator Probe Dot M(V, pH)
        cur_ph = self.current_ph
        mx = p_x + (self._v_titrant / 50.0) * p_w
        my = p_y + p_h - (cur_ph / 14.0) * p_h

        # Tracer Halo
        grad = QRadialGradient(mx, my, 8.0)
        grad.setColorAt(0.0, QColor("#fbbf24"))
        grad.setColorAt(1.0, QColor(251, 191, 36, 0))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(QPointF(mx, my), 8.0, 8.0)

        painter.setPen(QPen(QColor("#0f172a"), 1.5))
        painter.setBrush(QBrush(QColor("#fbbf24")))
        painter.drawEllipse(QPointF(mx, my), 4.0, 4.0)

        # Bottom X Axis Ticks
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(p_x, p_y + p_h + 14.0, "0 mL")
        painter.drawText(eq_px - 14.0, p_y + p_h + 14.0, f"{v_eq:.0f} mL")
        painter.drawText(p_x + p_w - 24.0, p_y + p_h + 14.0, "50 mL")

        # Current Status Tag
        region_tag = "Vùng đệm (Buffer)" if self._v_titrant < v_eq - 0.2 else (
            "Điểm tương đương E" if abs(self._v_titrant - v_eq) <= 0.2 else "Sau tương đương (Dư NaOH)"
        )
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(p_x + 10.0, p_y + 20.0, f"● Trạng thái: {region_tag}")

    def _paint_control_deck(self, painter: QPainter) -> None:
        """Paints interactive button deck."""
        btn_font = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(btn_font)

        def draw_button(rect: QRectF, text: str, is_active: bool, bg_hex: str = "#334155") -> None:
            col = QColor("#0284c7") if is_active else QColor(bg_hex)
            painter.setPen(QPen(col.lighter(130), 1.0))
            painter.setBrush(QBrush(col))
            painter.drawRoundedRect(rect, 4.0, 4.0)
            painter.setPen(QColor("#ffffff") if is_active else QColor("#cbd5e1"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        # Row 1 Controls
        draw_button(self._rect_btn_drop_small, "💧 +0.5 mL", False, "#0369a1")
        draw_button(self._rect_btn_drop_large, "💧 +2.0 mL", False, "#0369a1")

        flow_txt = "⏸ Dừng nhỏ" if self._is_dispensing else "▶ Nhỏ tự động"
        draw_button(self._rect_btn_auto_flow, flow_txt, self._is_dispensing, "#0284c7")

        acid_txt = "Axit yếu (CH₃COOH)" if self._acid_type == "WEAK_ACID" else "Axit mạnh (HCl)"
        draw_button(self._rect_btn_acid_toggle, f"🧪 {acid_txt}", self._acid_type == "WEAK_ACID", "#7c3aed")

        draw_button(self._rect_btn_reset, "🔄 Đặt lại", False, "#475569")
        draw_button(self._rect_btn_stamp, "🖊️ Đóng dấu", False, "#059669")

        # Row 2 Controls (Indicators)
        draw_button(self._rect_btn_ind_phenol, "🔴 Phenolphthalein", self._indicator == "PHENOLPHTHALEIN", "#be123c")
        draw_button(self._rect_btn_ind_methyl, "🟠 Metyl Da Cam", self._indicator == "METHYL_ORANGE", "#c2410c")
        draw_button(self._rect_btn_ind_univ, "🌈 Chỉ thị vạn năng", self._indicator == "UNIVERSAL", "#047857")
