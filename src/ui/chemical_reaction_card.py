"""
AirTeach AI - Interactive Chemical Reaction Card Item
======================================================
Interactive QGraphicsObject displaying balanced chemical equations,
conservation of mass proofs, and stoichiometric analysis directly on LessonBoard:
- Modern dark glassmorphic card design with rounded borders and chemistry badges
- Highlighting stoichiometric coefficients in vibrant amber/cyan
- Interactive collapsible/expandable step-by-step element balancing breakdown
- Law of Conservation of Mass verification indicator (Lomonosov - Lavoisier)
- Stoichiometric molar mass and theoretical yield data table
- Interactive touchless and mouse controls: drag, toggle expansion, close card
- Full JSON serialization for .airteach lesson packages and replay synchronization
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
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

from src.services.chemical_equation_balancer import ChemicalReactionResult, ChemicalSpecies

logger = logging.getLogger(__name__)


class ChemicalReactionCardItem(QGraphicsObject):
    """
    Interactive QGraphicsObject presenting a balanced chemical equation and stoichiometry.
    """

    # Signals
    expanded_changed = Signal(bool)
    card_closed = Signal(object)

    # Styling Constants
    CARD_WIDTH = 500.0
    CARD_RADIUS = 12.0
    HEADER_HEIGHT = 44.0
    COLLAPSED_HEIGHT = 110.0

    # Colors
    BG_COLOR = QColor(15, 23, 42, 245)         # Slate 900
    BORDER_COLOR = QColor(245, 158, 11, 220)    # Amber 500
    HEADER_BG = QColor(30, 41, 59, 230)        # Slate 800
    REACTION_BG = QColor(24, 24, 27, 220)      # Zinc 900
    REACTION_BORDER = QColor(245, 158, 11, 190) # Amber 500
    STEP_BG = QColor(30, 41, 59, 160)          # Slate 800 translucent
    STEP_BORDER = QColor(51, 65, 85, 180)      # Slate 700

    def __init__(
        self,
        result: ChemicalReactionResult,
        parent: Optional[QGraphicsItem] = None,
        is_expanded: bool = True,
    ) -> None:
        super().__init__(parent)
        self._result = result
        self._is_expanded = is_expanded
        self._width = self.CARD_WIDTH
        self._total_height = self.COLLAPSED_HEIGHT
        self._step_layouts: list[dict[str, Any]] = []

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(15.0)

        self._recalculate_layout()

    # ── Geometry and Layout ───────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, self._total_height)

    def _recalculate_layout(self) -> None:
        """Calculate dynamic card height and element rectangles."""
        self.prepareGeometryChange()
        w = self._width
        y = self.HEADER_HEIGHT + 10.0

        # Balanced equation box height
        eq_box_h = 58.0
        self._eq_box_rect = QRectF(14.0, y, w - 28.0, eq_box_h)
        y += eq_box_h + 8.0

        # Mass conservation bar height
        bar_h = 28.0
        self._mass_bar_rect = QRectF(14.0, y, w - 28.0, bar_h)
        y += bar_h + 10.0

        if not self._is_expanded or not self._result.steps:
            self._total_height = y + 8.0
            self._step_layouts.clear()
            self.update()
            return

        # Steps section header
        y += 24.0

        self._step_layouts = []
        for step in self._result.steps:
            step_y = y
            desc_lines = max(1, len(step.description) // 52 + 1)
            step_h = 24.0 + (desc_lines * 16.0) + 14.0
            step_rect = QRectF(14.0, step_y, w - 28.0, step_h)
            self._step_layouts.append({
                "step": step,
                "rect": step_rect,
            })
            y += step_h + 8.0

        self._total_height = y + 10.0
        self.update()

    # ── State and Toggles ─────────────────────────────────────────────────────

    @property
    def result(self) -> ChemicalReactionResult:
        return self._result

    @property
    def is_expanded(self) -> bool:
        return self._is_expanded

    def set_expanded(self, expanded: bool) -> None:
        if self._is_expanded != expanded:
            self._is_expanded = expanded
            self._recalculate_layout()
            self.expanded_changed.emit(self._is_expanded)

    def toggle_expanded(self) -> None:
        self.set_expanded(not self._is_expanded)

    # ── Mouse Interaction ─────────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        # 1. Close button (top right: w - 34 to w - 10, y: 8 to 34)
        if (self._width - 34.0) <= pos.x() <= (self._width - 10.0) and 8.0 <= pos.y() <= 34.0:
            event.accept()
            self.card_closed.emit(self)
            if self.scene():
                self.scene().removeItem(self)
            return

        # 2. Toggle expand button (w - 68 to w - 40, y: 8 to 34)
        if (self._width - 68.0) <= pos.x() <= (self._width - 40.0) and 8.0 <= pos.y() <= 34.0:
            event.accept()
            self.toggle_expanded()
            return

        # 3. Collapse/expand prompt click
        if self._result.steps and not self._is_expanded:
            if self._mass_bar_rect.bottom() <= pos.y() <= self._total_height:
                event.accept()
                self.toggle_expanded()
                return

        super().mousePressEvent(event)

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[Any] = None) -> bool:
        """Process touchless air tap gesture from webcam tracking."""
        pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(pos):
            return False

        # 1. Close button
        if (self._width - 34.0) <= pos.x() <= (self._width - 10.0) and 8.0 <= pos.y() <= 34.0:
            self.card_closed.emit(self)
            if self.scene():
                self.scene().removeItem(self)
            return True

        # 2. Toggle expand button
        if (self._width - 68.0) <= pos.x() <= (self._width - 40.0) and 8.0 <= pos.y() <= 34.0:
            self.toggle_expanded()
            return True

        # 3. Collapse/expand prompt click
        if self._result.steps and not self._is_expanded:
            if self._mass_bar_rect.bottom() <= pos.y() <= self._total_height:
                self.toggle_expanded()
                return True

        return True

    # ── Painting ──────────────────────────────────────────────────────────────

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        w = self._width
        h = self._total_height

        # 1. Outer Card Background & Shadow Border
        card_rect = QRectF(0.0, 0.0, w, h)
        card_path = QPainterPath()
        card_path.addRoundedRect(card_rect, self.CARD_RADIUS, self.CARD_RADIUS)

        painter.fillPath(card_path, QBrush(self.BG_COLOR))
        border_pen = QPen(
            QColor(251, 191, 36, 240) if self.isSelected() else self.BORDER_COLOR,
            2.0 if self.isSelected() else 1.5,
        )
        painter.setPen(border_pen)
        painter.drawPath(card_path)

        # 2. Header Bar
        header_rect = QRectF(0.0, 0.0, w, self.HEADER_HEIGHT)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, self.CARD_RADIUS, self.CARD_RADIUS)
        painter.save()
        painter.setClipRect(0.0, 0.0, w, self.HEADER_HEIGHT)
        painter.fillPath(header_path, QBrush(self.HEADER_BG))
        painter.restore()

        # Separator line under header
        painter.setPen(QPen(QColor(51, 65, 85, 220), 1.0))
        painter.drawLine(QPointF(0.0, self.HEADER_HEIGHT), QPointF(w, self.HEADER_HEIGHT))

        # 3. Operation Badge
        self._paint_header_badge(painter)

        # 4. Header Control Buttons (Close [×] and Expand [▲/▼])
        self._paint_header_buttons(painter)

        # 5. Balanced Chemical Equation Box
        self._paint_equation_box(painter)

        # 6. Mass Conservation Indicator Bar
        self._paint_mass_bar(painter)

        # 7. Derivation Steps (if expanded)
        if self._is_expanded and self._result.steps:
            self._paint_steps(painter)
        elif self._result.steps:
            # "Show N derivation steps ▼" prompt
            painter.setPen(QPen(QColor(148, 163, 184, 220)))
            painter.setFont(QFont("Segoe UI", 9))
            steps_txt = f"▼ Xem {len(self._result.steps)} bước cân bằng chi tiết..."
            painter.drawText(
                QRectF(14.0, self._mass_bar_rect.bottom() + 6.0, w - 28.0, 20.0),
                Qt.AlignmentFlag.AlignCenter,
                steps_txt,
            )

        painter.restore()

    def _paint_header_badge(self, painter: QPainter) -> None:
        """Paint the chemistry badge and execution time indicator."""
        badge_rect = QRectF(14.0, 10.0, 175.0, 24.0)
        badge_path = QPainterPath()
        badge_path.addRoundedRect(badge_rect, 6.0, 6.0)
        painter.fillPath(badge_path, QBrush(QColor(245, 158, 11, 40)))
        painter.setPen(QPen(QColor(245, 158, 11), 1.2))
        painter.drawPath(badge_path)

        painter.setPen(QPen(QColor(245, 158, 11)))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, "⚗ CÂN BẰNG PHẢN ỨNG")

        # Execution time
        if self._result.execution_time_ms > 0:
            time_txt = f"{self._result.execution_time_ms:.1f}ms"
            painter.setPen(QPen(QColor(148, 163, 184, 180)))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(QRectF(195.0, 10.0, 80.0, 24.0), Qt.AlignmentFlag.AlignVCenter, time_txt)

    def _paint_header_buttons(self, painter: QPainter) -> None:
        """Paint collapse and close buttons in the header."""
        w = self._width

        # Toggle Expand Button [▲ / ▼]
        toggle_rect = QRectF(w - 68.0, 9.0, 24.0, 24.0)
        toggle_path = QPainterPath()
        toggle_path.addRoundedRect(toggle_rect, 5.0, 5.0)
        painter.fillPath(toggle_path, QBrush(QColor(51, 65, 85, 160)))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(
            toggle_rect,
            Qt.AlignmentFlag.AlignCenter,
            "▲" if self._is_expanded else "▼",
        )

        # Close Button [×]
        close_rect = QRectF(w - 36.0, 9.0, 24.0, 24.0)
        close_path = QPainterPath()
        close_path.addRoundedRect(close_rect, 5.0, 5.0)
        painter.fillPath(close_path, QBrush(QColor(239, 68, 68, 40)))
        painter.setPen(QPen(QColor(239, 68, 68)))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(close_rect, Qt.AlignmentFlag.AlignCenter, "×")

    def _paint_equation_box(self, painter: QPainter) -> None:
        """Paint the prominent balanced equation with colored coefficients."""
        rect = self._eq_box_rect
        eq_path = QPainterPath()
        eq_path.addRoundedRect(rect, 8.0, 8.0)
        painter.fillPath(eq_path, QBrush(self.REACTION_BG))
        painter.setPen(QPen(self.REACTION_BORDER, 1.2))
        painter.drawPath(eq_path)

        # Draw balanced equation text
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f8fafc")))
        eq_str = self._result.balanced_equation_text or self._result.original_equation
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, eq_str)

    def _paint_mass_bar(self, painter: QPainter) -> None:
        """Paint mass conservation indicator strip."""
        rect = self._mass_bar_rect
        bar_path = QPainterPath()
        bar_path.addRoundedRect(rect, 6.0, 6.0)

        if self._result.mass_conservation_verified:
            painter.fillPath(bar_path, QBrush(QColor(6, 78, 59, 180)))  # Emerald 900
            painter.setPen(QPen(QColor(16, 185, 129), 1.0))
            painter.drawPath(bar_path)
            painter.setPen(QPen(QColor("#34d399")))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            txt = f"✓ Bảo toàn khối lượng tuyệt đối: Σ M = {self._result.total_reactant_mass:.2f} g/mol"
        else:
            painter.fillPath(bar_path, QBrush(QColor(127, 29, 29, 180)))
            painter.setPen(QPen(QColor(239, 68, 68), 1.0))
            painter.drawPath(bar_path)
            painter.setPen(QPen(QColor("#fca5a5")))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            txt = "⚠ Chưa bảo toàn khối lượng"

        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)

    def _paint_steps(self, painter: QPainter) -> None:
        """Paint step-by-step element balancing breakdown."""
        w = self._width
        painter.setPen(QPen(QColor(226, 232, 240)))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(
            QRectF(16.0, self._mass_bar_rect.bottom() + 10.0, w - 32.0, 20.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Các bước cân bằng chi tiết ({len(self._step_layouts)} bước):",
        )

        for layout in self._step_layouts:
            step = layout["step"]
            rect = layout["rect"]

            step_path = QPainterPath()
            step_path.addRoundedRect(rect, 6.0, 6.0)
            painter.fillPath(step_path, QBrush(self.STEP_BG))
            painter.setPen(QPen(self.STEP_BORDER, 1.0))
            painter.drawPath(step_path)

            # Step number badge
            badge_rect = QRectF(rect.x() + 8.0, rect.y() + 8.0, 20.0, 20.0)
            b_path = QPainterPath()
            b_path.addRoundedRect(badge_rect, 4.0, 4.0)
            painter.fillPath(b_path, QBrush(QColor(245, 158, 11, 180)))
            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
            painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, str(step.step_number))

            # Step title
            painter.setPen(QPen(QColor(241, 245, 249)))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(
                QRectF(rect.x() + 34.0, rect.y() + 8.0, rect.width() - 44.0, 20.0),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                step.title,
            )

            # Step description
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.setFont(QFont("Segoe UI", 8))
            desc_rect = QRectF(rect.x() + 12.0, rect.y() + 30.0, rect.width() - 24.0, rect.height() - 34.0)
            painter.drawText(
                desc_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
                step.description,
            )

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize card state for .airteach lesson packaging."""
        return {
            "type": "chemical_reaction_card",
            "pos": [self.x(), self.y()],
            "expanded": self._is_expanded,
            "result": self._result.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChemicalReactionCardItem:
        """Reconstruct card item from serialized dictionary."""
        result_data = data.get("result", {})
        result = ChemicalReactionResult.from_dict(result_data)
        item = cls(result=result, is_expanded=bool(data.get("expanded", True)))
        pos = data.get("pos", [0.0, 0.0])
        item.setPos(pos[0], pos[1])
        return item
