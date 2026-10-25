"""
AirTeach AI - Interactive Math Derivation Card Item
====================================================
Interactive QGraphicsObject displaying symbolic solutions and pedagogical
step-by-step mathematical derivations directly on the LessonBoard canvas:
- Modern dark glassmorphic card design with rounded borders and operation badges
- Highlighting final solution in emerald/cyan with exact fractions and radicals
- Collapsible/expandable step-by-step reasoning panel
- Crisp LaTeX rendering of mathematical formulas via matplotlib.mathtext
- Interactive touchless and mouse controls: drag, toggle expansion, close card
- Full JSON serialization for .airteach lesson packages
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from src.services.smart_ink_ocr import LaTeXRenderer
from src.services.symbolic_math import DerivationStep, MathDerivationResult

logger = logging.getLogger(__name__)


def _rgba_to_qpixmap(rgba: np.ndarray) -> QPixmap:
    """Convert RGBA numpy array to QPixmap safely."""
    h, w = rgba.shape[:2]
    # Ensure memory is contiguous
    c_rgba = np.ascontiguousarray(rgba)
    qimg = QImage(c_rgba.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


class MathDerivationCardItem(QGraphicsObject):
    """
    Interactive QGraphicsObject presenting a step-by-step mathematical derivation.
    """

    # Signals
    expanded_changed = Signal(bool)
    card_closed = Signal(object)
    solution_applied = Signal(str)

    # Styling Constants
    CARD_WIDTH = 460.0
    CARD_RADIUS = 12.0
    HEADER_HEIGHT = 44.0
    COLLAPSED_HEIGHT = 90.0

    # Colors
    BG_COLOR = QColor(15, 23, 42, 245)         # Slate 900
    BORDER_COLOR = QColor(59, 130, 246, 220)    # Blue 500
    HEADER_BG = QColor(30, 41, 59, 230)        # Slate 800
    SOLUTION_BG = QColor(6, 78, 59, 210)       # Emerald 900
    SOLUTION_BORDER = QColor(16, 185, 129, 240) # Emerald 500
    STEP_BG = QColor(30, 41, 59, 160)          # Slate 800 translucent
    STEP_BORDER = QColor(51, 65, 85, 180)      # Slate 700

    def __init__(
        self,
        result: MathDerivationResult,
        parent: Optional[QGraphicsItem] = None,
        is_expanded: bool = True,
    ) -> None:
        super().__init__(parent)
        self._result = result
        self._is_expanded = is_expanded
        self._width = self.CARD_WIDTH
        self._total_height = self.COLLAPSED_HEIGHT

        # Cached Pixmaps: key -> QPixmap
        self._pixmap_cache: dict[str, QPixmap] = {}
        self._step_layouts: list[dict[str, Any]] = []

        # Graphics item configuration
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(15.0)  # On top of drawings

        # Prepare rendered math pixmaps and measure layout
        self._prepare_pixmaps()
        self._recalculate_layout()

    # ── Geometry and Layout ───────────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, self._total_height)

    def _prepare_pixmaps(self) -> None:
        """Pre-render LaTeX formula strings to crisp RGBA QPixmaps."""
        # 1. Solution LaTeX
        if self._result.solution_latex:
            try:
                rgba = LaTeXRenderer.render_to_rgba(
                    self._result.solution_latex,
                    fontsize=14,
                    dpi=120,
                    color="#34d399",  # Emerald light
                )
                if rgba is not None:
                    self._pixmap_cache["solution"] = _rgba_to_qpixmap(rgba)
            except Exception as exc:
                logger.debug("Failed to render solution LaTeX: %s", exc)

        # 2. Original Expression LaTeX
        if self._result.original_latex:
            try:
                rgba = LaTeXRenderer.render_to_rgba(
                    self._result.original_latex,
                    fontsize=12,
                    dpi=110,
                    color="#94a3b8",  # Slate 400
                )
                if rgba is not None:
                    self._pixmap_cache["original"] = _rgba_to_qpixmap(rgba)
            except Exception as exc:
                logger.debug("Failed to render original LaTeX: %s", exc)

        # 3. Derivation Steps LaTeX
        for i, step in enumerate(self._result.steps):
            if step.latex:
                cache_key = f"step_{i}"
                try:
                    rgba = LaTeXRenderer.render_to_rgba(
                        step.latex,
                        fontsize=12,
                        dpi=110,
                        color="#f8fafc",  # White
                    )
                    if rgba is not None:
                        self._pixmap_cache[cache_key] = _rgba_to_qpixmap(rgba)
                except Exception as exc:
                    logger.debug("Failed to render step %d LaTeX: %s", i, exc)

    def _recalculate_layout(self) -> None:
        """Calculate card height and relative positions for all components."""
        self.prepareGeometryChange()
        w = self._width
        y = self.HEADER_HEIGHT + 8.0

        # Solution box height
        sol_pix = self._pixmap_cache.get("solution")
        sol_box_h = max(42.0, (sol_pix.height() + 16.0) if sol_pix else 42.0)
        self._sol_box_rect = QRectF(14.0, y, w - 28.0, sol_box_h)
        y += sol_box_h + 10.0

        if not self._is_expanded or not self._result.steps:
            self._total_height = y + 8.0
            self._step_layouts.clear()
            self.update()
            return

        # Toggle bar for steps
        y += 24.0

        # Steps calculation
        self._step_layouts = []
        for i, step in enumerate(self._result.steps):
            step_y = y
            pix = self._pixmap_cache.get(f"step_{i}")
            pix_h = pix.height() if pix else 0.0

            # Title + description height estimation
            desc_lines = max(1, len(step.description) // 48 + 1)
            text_h = 20.0 + (desc_lines * 16.0)
            step_h = text_h + (pix_h + 10.0 if pix_h > 0 else 6.0) + 12.0

            step_rect = QRectF(14.0, step_y, w - 28.0, step_h)
            self._step_layouts.append({
                "step": step,
                "rect": step_rect,
                "pixmap": pix,
                "text_h": text_h,
            })
            y += step_h + 8.0

        self._total_height = y + 10.0
        self.update()

    # ── State and Toggles ─────────────────────────────────────────────────────

    @property
    def result(self) -> MathDerivationResult:
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
        # 1. Close button (top right: w - 34 to w - 10, y: 8 to 32)
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

        # 3. Steps section toggle click
        if self._result.steps and not self._is_expanded:
            if self._sol_box_rect.bottom() <= pos.y() <= self._total_height:
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
        if (self._width - 34.0) <= pos.x() <= (self._width - 10.0) and 8.0 <= pos.y() <= 32.0:
            self.card_closed.emit(self)
            if self.scene():
                self.scene().removeItem(self)
            return True

        # 2. Toggle expand button
        if (self._width - 68.0) <= pos.x() <= (self._width - 40.0) and 8.0 <= pos.y() <= 34.0:
            self.toggle_expanded()
            return True

        # 3. Steps section toggle click
        if self._result.steps and not self._is_expanded:
            if self._sol_box_rect.bottom() <= pos.y() <= self._total_height:
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
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        w = self._width
        h = self._total_height

        # 1. Outer Card Background & Shadow Border
        card_rect = QRectF(0.0, 0.0, w, h)
        card_path = QPainterPath()
        card_path.addRoundedRect(card_rect, self.CARD_RADIUS, self.CARD_RADIUS)

        painter.fillPath(card_path, QBrush(self.BG_COLOR))
        border_pen = QPen(
            QColor(96, 165, 250, 240) if self.isSelected() else self.BORDER_COLOR,
            2.0 if self.isSelected() else 1.5,
        )
        painter.setPen(border_pen)
        painter.drawPath(card_path)

        # 2. Header Bar
        header_rect = QRectF(0.0, 0.0, w, self.HEADER_HEIGHT)
        header_path = QPainterPath()
        header_path.addRoundedRect(header_rect, self.CARD_RADIUS, self.CARD_RADIUS)
        # Intersect with top half to clip round corners only at top
        painter.save()
        painter.setClipRect(0.0, 0.0, w, self.HEADER_HEIGHT)
        painter.fillPath(header_path, QBrush(self.HEADER_BG))
        painter.restore()

        # Separator line under header
        painter.setPen(QPen(QColor(51, 65, 85, 220), 1.0))
        painter.drawLine(QPointF(0.0, self.HEADER_HEIGHT), QPointF(w, self.HEADER_HEIGHT))

        # 3. Operation Badge
        self._paint_operation_badge(painter)

        # 4. Header Control Buttons (Close [×] and Expand [▲/▼])
        self._paint_header_buttons(painter)

        # 5. Solution Box
        self._paint_solution_box(painter)

        # 6. Steps Section (if expanded)
        if self._is_expanded and self._result.steps:
            self._paint_derivation_steps(painter)
        elif self._result.steps:
            # Show "Show N derivation steps ▼" prompt
            painter.setPen(QPen(QColor(148, 163, 184, 220)))
            font = QFont("Segoe UI", 9)
            painter.setFont(font)
            steps_txt = f"▼ Xem {len(self._result.steps)} bước giải chi tiết..."
            painter.drawText(
                QRectF(14.0, self._sol_box_rect.bottom() + 6.0, w - 28.0, 20.0),
                Qt.AlignmentFlag.AlignCenter,
                steps_txt,
            )

        painter.restore()

    def _paint_operation_badge(self, painter: QPainter) -> None:
        """Paint the operation tag badge (SOLVE, DERIVATIVE, etc.)."""
        op = (self._result.operation or "solve").lower()
        badge_configs = {
            "solve": ("GIẢI PHƯƠNG TRÌNH", QColor(16, 185, 129)),
            "derivative": ("ĐẠO HÀM f'(x)", QColor(245, 158, 11)),
            "integral": ("TÍCH PHÂN ∫", QColor(139, 92, 246)),
            "simplify": ("RÚT GỌN", QColor(14, 165, 233)),
            "evaluate": ("TÍNH TOÁN", QColor(59, 130, 246)),
        }
        text, color = badge_configs.get(op, ("KẾT QUẢ TOÁN HỌC", QColor(59, 130, 246)))

        # Badge pill
        badge_rect = QRectF(14.0, 10.0, 150.0, 24.0)
        badge_path = QPainterPath()
        badge_path.addRoundedRect(badge_rect, 6.0, 6.0)
        painter.fillPath(badge_path, QBrush(QColor(color.red(), color.green(), color.blue(), 45)))
        painter.setPen(QPen(color, 1.2))
        painter.drawPath(badge_path)

        painter.setPen(QPen(color))
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, text)

        # Execution time indicator
        if self._result.execution_time_ms > 0:
            time_txt = f"{self._result.execution_time_ms:.1f}ms"
            painter.setPen(QPen(QColor(148, 163, 184, 180)))
            painter.setFont(QFont("Segoe UI", 8))
            painter.drawText(QRectF(170.0, 10.0, 80.0, 24.0), Qt.AlignmentFlag.AlignVCenter, time_txt)

    def _paint_header_buttons(self, painter: QPainter) -> None:
        """Paint the collapse and close buttons in the header."""
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

    def _paint_solution_box(self, painter: QPainter) -> None:
        """Paint the prominent solution card."""
        sol_rect = self._sol_box_rect
        sol_path = QPainterPath()
        sol_path.addRoundedRect(sol_rect, 8.0, 8.0)
        painter.fillPath(sol_path, QBrush(self.SOLUTION_BG))
        painter.setPen(QPen(self.SOLUTION_BORDER, 1.2))
        painter.drawPath(sol_path)

        # Draw rendered solution pixmap or text
        sol_pix = self._pixmap_cache.get("solution")
        if sol_pix and not sol_pix.isNull():
            px = sol_rect.x() + (sol_rect.width() - sol_pix.width()) / 2.0
            py = sol_rect.y() + (sol_rect.height() - sol_pix.height()) / 2.0
            painter.drawPixmap(QPointF(px, py), sol_pix)
        else:
            # Fallback to plain text
            painter.setPen(QPen(QColor("#34d399")))
            font = QFont("Segoe UI", 11, QFont.Weight.Bold)
            painter.setFont(font)
            raw_text = self._result.solution_raw or self._result.solution_latex or "Đã giải xong"
            painter.drawText(sol_rect, Qt.AlignmentFlag.AlignCenter, raw_text)

    def _paint_derivation_steps(self, painter: QPainter) -> None:
        """Paint each step card in the derivation list."""
        w = self._width
        # Section header
        painter.setPen(QPen(QColor(226, 232, 240)))
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            QRectF(16.0, self._sol_box_rect.bottom() + 10.0, w - 32.0, 20.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            f"Các bước giải chi tiết ({len(self._step_layouts)} bước):",
        )

        for layout in self._step_layouts:
            step: DerivationStep = layout["step"]
            rect: QRectF = layout["rect"]
            pix: Optional[QPixmap] = layout["pixmap"]

            # Step card container
            step_path = QPainterPath()
            step_path.addRoundedRect(rect, 6.0, 6.0)
            painter.fillPath(step_path, QBrush(self.STEP_BG))
            painter.setPen(QPen(self.STEP_BORDER, 1.0))
            painter.drawPath(step_path)

            # Step number badge
            badge_rect = QRectF(rect.x() + 8.0, rect.y() + 8.0, 20.0, 20.0)
            b_path = QPainterPath()
            b_path.addRoundedRect(badge_rect, 4.0, 4.0)
            painter.fillPath(b_path, QBrush(QColor(59, 130, 246, 180)))
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
            desc_rect = QRectF(rect.x() + 12.0, rect.y() + 30.0, rect.width() - 24.0, 36.0)
            painter.drawText(
                desc_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap),
                step.description,
            )

            # Step formula (rendered LaTeX pixmap)
            if pix and not pix.isNull():
                # Center formula pixmap horizontally
                px = rect.x() + (rect.width() - pix.width()) / 2.0
                py = rect.bottom() - pix.height() - 8.0
                painter.drawPixmap(QPointF(px, py), pix)

    # ── Serialization ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Serialize card state for .airteach lesson packaging."""
        return {
            "type": "math_derivation_card",
            "pos": [self.x(), self.y()],
            "expanded": self._is_expanded,
            "result": self._result.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MathDerivationCardItem:
        """Reconstruct card item from serialized dictionary."""
        result_data = data.get("result", {})
        result = MathDerivationResult.from_dict(result_data)
        item = cls(result=result, is_expanded=bool(data.get("expanded", True)))
        pos = data.get("pos", [0.0, 0.0])
        item.setPos(pos[0], pos[1])
        return item
