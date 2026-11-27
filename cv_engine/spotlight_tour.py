#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
True Interactive Spotlight Tour Overlay (`cv_engine/spotlight_tour.py`)
for AirTeach AI (DNT - Victer).

Renders a live semi-transparent dark backdrop (`QColor(2, 6, 23, 188)`) using
`QPainterPath` with `Qt.OddEvenFill` (outer window rect minus rounded cutout)
so ONLY the active UI widget / workspace region shines through at 100% clarity,
framed by a glowing accent border + layered outer halo ring, accompanied by a
smart auto-positioned Vietnamese callout card that stays clamped inside parent
bounds at 1366x768, 1920x1080, and any dynamic resize.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

from PySide6.QtCore import QEvent, QPoint, QRect, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


BG_DARK = "#0F172A"
SURFACE_DARK = "#1E293B"
BORDER_SLATE = "#334155"
PRIMARY_INDIGO = "#4F46E5"
SECONDARY_TEAL = "#06B6D4"
NEON_CYAN = "#00F5D4"
BRIGHT_YELLOW = "#FEE440"
SUCCESS_GREEN = "#10B981"
TEXT_LIGHT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

BACKDROP_COLOR = QColor(2, 6, 23, 188)


@dataclass
class SpotlightStep:
    """Definition of a single interactive Spotlight Tour step."""

    step_id: str
    title: str
    body: str
    target_getter: Callable[[], Optional[QWidget]]
    accent_hex: str = "#00F5D4"
    fallback_norm_rect: Tuple[float, float, float, float] = (0.25, 0.25, 0.50, 0.50)
    tip: str = ""


def _resolve_doc_target(win: Any) -> Optional[QWidget]:
    """Locate the Documents button (`doc_menu_btn` or header 'Tài Liệu' button) or header bar."""
    doc_btn = getattr(win, "doc_menu_btn", None)
    if isinstance(doc_btn, QWidget) and doc_btn.isVisible() and doc_btn.width() > 4 and doc_btn.height() > 4:
        return doc_btn

    header = getattr(win, "header_bar", None)
    if isinstance(header, QWidget) and header.isVisible():
        for btn in header.findChildren(QPushButton):
            if btn.isVisible() and btn.width() > 4 and btn.height() > 4:
                if "tài liệu" in btn.text().lower():
                    return btn
        if header.width() > 4 and header.height() > 4:
            return header
    return None


def _resolve_gestures_target(win: Any) -> Optional[QWidget]:
    """Highlight `win.canvas.mini_hud` when visible, else fall back to `win.toggle_cam_btn`."""
    canvas = getattr(win, "canvas", None)
    mini_hud = getattr(canvas, "mini_hud", None) if canvas is not None else None
    if (
        isinstance(mini_hud, QWidget)
        and mini_hud.isVisible()
        and mini_hud.width() > 4
        and mini_hud.height() > 4
    ):
        return mini_hud

    cam_btn = getattr(win, "toggle_cam_btn", None)
    if isinstance(cam_btn, QWidget) and cam_btn.isVisible() and cam_btn.width() > 4 and cam_btn.height() > 4:
        return cam_btn
    return None


def _resolve_ai_target(win: Any) -> Optional[QWidget]:
    """Highlight `win.toggle_ai_btn` or `win.local_mgr_btn`."""
    for attr in ("toggle_ai_btn", "local_mgr_btn", "ai_mode_quick_btn"):
        w = getattr(win, attr, None)
        if isinstance(w, QWidget) and w.isVisible() and w.width() > 4 and w.height() > 4:
            return w
    return None


def build_default_airteach_spotlight_steps(win: Any) -> List[SpotlightStep]:
    """Build the canonical 6-step Vietnamese Spotlight Tour for `AirTeachMainWindow`."""
    return [
        SpotlightStep(
            step_id="radial_menu",
            title="Menu Tròn Chuột Phải 8 Hướng & Vòng Chọn Con",
            body=(
                "Nhấn giữ Chuột Phải tại bất kỳ đâu trên bảng để mở vòng tròn 8 công cụ "
                "(Bút vẽ, Tẩy, Chọn, Kéo bảng, Chữ, Hình khối, Hoàn tác, Xóa/Mới).\n"
                "Kéo chuột ra vòng ngoài ở mục Bút vẽ, Hình khối hoặc Xóa/Mới để chọn nhanh "
                "màu mực, loại hình, Thêm trang mới hoặc Mở AI Chat."
            ),
            target_getter=lambda: None,
            accent_hex="#00F5D4",
            fallback_norm_rect=(0.31, 0.16, 0.38, 0.34),
            tip="Thả chuột trong vùng tâm vòng tròn (< 22px) hoặc bấm Esc để hủy an toàn mà không đổi công cụ.",
        ),
        SpotlightStep(
            step_id="pan_zoom",
            title="Di Chuyển Bảng Vô Tận & Zoom Đúng Tâm Con Trỏ",
            body=(
                "Giữ phím Space + kéo Chuột Trái, kéo Chuột Giữa hoặc chọn công cụ ✥ Kéo Bảng "
                "để trượt bảng vô tận ở 60fps.\n"
                "Cuộn con lăn chuột để phóng to/thu nhỏ đúng tâm con trỏ (25%–400%), hoặc dùng "
                "thanh điều hướng nổi góc phải dưới để Vừa nội dung, Về 100% (Ctrl+0) và Khóa Pan."
            ),
            target_getter=lambda: getattr(getattr(win, "canvas", None), "nav_hud", None),
            accent_hex="#38BDF8",
            fallback_norm_rect=(0.68, 0.84, 0.30, 0.10),
            tip="Bật khóa 🔓 Pan khi đang viết nhanh để chống trượt bảng ngoài ý muốn.",
        ),
        SpotlightStep(
            step_id="object_canvas",
            title="Hệ Thống Object Độc Lập — Chọn, Resize 8 Góc, Xoay & Đổi Màu",
            body=(
                "Mọi Nét vẽ, Hình khối, Văn bản, Ảnh và Sticky Note đều là Object độc lập "
                "từ thanh công cụ nổi phía dưới.\n"
                "Dùng ⬉ Chọn (V) để kéo di chuyển, co giãn theo 8 góc, xoay bằng tay cầm trên đỉnh, "
                "đổi màu trực tiếp, nhân bản (Ctrl+D) hoặc xóa (Delete)."
            ),
            target_getter=lambda: getattr(win, "floating_dock", None),
            accent_hex="#A855F7",
            fallback_norm_rect=(0.12, 0.90, 0.76, 0.08),
            tip="Nhấn đúp vào Hộp chữ hoặc Sticky Note để sửa ngay; Ctrl+Z / Ctrl+Y để hoàn tác mọi thay đổi.",
        ),
        SpotlightStep(
            step_id="gestures",
            title="Cử Chỉ 1 Tay & 2 Tay Không Chạm (Lõi Rust 60Hz)",
            body=(
                "• 1 Tay: ☝️ 1 ngón trỏ = Vẽ nét · ✌️ 2 ngón chữ V = Di chuyển con trỏ · "
                "🤏 Chụm Cái+Trỏ = Cầm kéo Object · 🖐️ Xòe 5 ngón = Tẩy · ↔ Vẫy ngang = Đổi Slide.\n"
                "• 2 Tay: 🔍 2 tay cùng chụm = Zoom mượt · ✋ 2 tay cùng nắm/xòe = Di chuyển bảng hoặc cuộn tài liệu."
            ),
            target_getter=lambda: _resolve_gestures_target(win),
            accent_hex="#FEE440",
            fallback_norm_rect=(0.74, 0.08, 0.24, 0.22),
            tip="Bấm nút 🧪 Thử Cử Chỉ ngay bên dưới để kiểm tra thực tế trên Webcam với phản hồi Xanh / Đỏ tức thì.",
        ),
        SpotlightStep(
            step_id="documents",
            title="Mở Bài Giảng Đa Định Dạng PDF / PPTX / DOCX / Ảnh",
            body=(
                "Mở trực tiếp file PDF, PowerPoint (PPTX), Word (DOCX), Ảnh nền hoặc Thư mục ảnh "
                "từ mục 📂 Tài Liệu trên thanh tiêu đề.\n"
                "Mỗi trang lưu riêng lớp chú thích Object độc lập, hỗ trợ cuộn dọc liên tục và "
                "xuất trọn vẹn bài giảng kèm ghi chú ra PDF hoặc bộ ảnh PNG."
            ),
            target_getter=lambda: _resolve_doc_target(win),
            accent_hex="#10B981",
            fallback_norm_rect=(0.06, 0.01, 0.14, 0.06),
            tip="Bạn cũng có thể kéo thả trực tiếp file PDF/PPTX/DOCX/Ảnh vào giữa bảng để mở ngay.",
        ),
        SpotlightStep(
            step_id="ai_hybrid",
            title="Trợ Lý AI Cloud Gemini Full Model & Local AI GGUF Offline",
            body=(
                "Mở ngăn Trợ Lý AI để hỏi bài theo ngữ cảnh Slide, khoanh vùng hỏi AI (Lasso), "
                "nhận lời giải Markdown/LaTeX đầy đủ và đưa trực tiếp lên bảng.\n"
                "Chuyển 1-click giữa Cloud Gemini ↔ Local AI GGUF Offline hoặc tải model từ "
                "HuggingFace / GitHub ngay trong 🤖 Hub."
            ),
            target_getter=lambda: _resolve_ai_target(win),
            accent_hex="#06B6D4",
            fallback_norm_rect=(0.64, 0.01, 0.24, 0.06),
            tip="Nhấn F1 hoặc nút '?' bất cứ lúc nào để mở lại Bảng tra cứu, Phòng Thử Cử Chỉ và Tour Spotlight.",
        ),
    ]


class SpotlightTourOverlay(QWidget):
    """
    Interactive full-window Spotlight Tour Overlay with OddEvenFill cutout,
    glowing accent halo ring, and auto-clamped floating Vietnamese callout card.
    """

    tour_finished = Signal(bool)
    open_sandbox_requested = Signal()

    CUTOUT_PADDING_PX = 8.0
    CUTOUT_RADIUS_PX = 12.0
    CARD_GAP_PX = 14.0
    EDGE_MARGIN_PX = 16.0

    def __init__(
        self,
        parent: QWidget,
        steps: Optional[Sequence[SpotlightStep]] = None,
    ) -> None:
        super().__init__(parent)
        self._steps: List[SpotlightStep] = (
            list(steps) if steps is not None else build_default_airteach_spotlight_steps(parent)
        )
        self._step_idx: int = 0
        self._spotlight_rect: QRectF = QRectF()
        self._dot_widgets: List[QFrame] = []

        self.setObjectName("SpotlightTourOverlay")
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)

        if parent is not None:
            self.setGeometry(parent.rect())
            parent.installEventFilter(self)

        self._build_callout_card()
        self.hide()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def current_step_index(self) -> int:
        return self._step_idx

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def dont_show_again(self) -> bool:
        return bool(self.dont_show_chk.isChecked())

    # ------------------------------------------------------------------
    # UI Construction (4px spacing grid, Vietnamese-safe multiline layout)
    # ------------------------------------------------------------------
    def _build_callout_card(self) -> None:
        self.card = QFrame(self)
        self.card.setObjectName("SpotlightCalloutCard")
        self.card.installEventFilter(self)
        self.card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)

        root = QVBoxLayout(self.card)
        root.setContentsMargins(16, 16, 16, 14)
        root.setSpacing(10)

        # Header row: Progress pill + step dots + Skip button
        hdr = QHBoxLayout()
        hdr.setSpacing(8)

        self.progress_pill = QLabel("Bước 1 / 6")
        self.progress_pill.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        hdr.addWidget(self.progress_pill)

        self._dots_row = QHBoxLayout()
        self._dots_row.setSpacing(4)
        self._rebuild_step_dots()
        hdr.addLayout(self._dots_row)
        hdr.addStretch(1)

        self.skip_btn = QPushButton("Bỏ qua")
        self.skip_btn.setCursor(Qt.PointingHandCursor)
        self.skip_btn.setFixedHeight(26)
        self.skip_btn.setStyleSheet(
            f"QPushButton {{ background: transparent; color: {TEXT_MUTED}; border: 1px solid {BORDER_SLATE}; "
            f"border-radius: 6px; padding: 0 10px; font-size: 11px; font-weight: 700; }}"
            f"QPushButton:hover {{ background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border-color: {TEXT_MUTED}; }}"
        )
        self.skip_btn.clicked.connect(self.skip_tour)
        hdr.addWidget(self.skip_btn)
        root.addLayout(hdr)

        # Title
        self.title_lbl = QLabel("")
        self.title_lbl.setWordWrap(True)
        self.title_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        root.addWidget(self.title_lbl)

        # Body explanation
        self.body_lbl = QLabel("")
        self.body_lbl.setWordWrap(True)
        self.body_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.body_lbl.setStyleSheet(
            f"color: {TEXT_LIGHT}; font-size: 12.5px; line-height: 1.5; "
            f"background: transparent; border: none;"
        )
        root.addWidget(self.body_lbl)

        # Tip box
        self.tip_frame = QFrame(self.card)
        tip_layout = QHBoxLayout(self.tip_frame)
        tip_layout.setContentsMargins(10, 8, 10, 8)
        tip_layout.setSpacing(6)
        self.tip_lbl = QLabel("")
        self.tip_lbl.setWordWrap(True)
        self.tip_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        tip_layout.addWidget(self.tip_lbl)
        root.addWidget(self.tip_frame)

        # Quick action row for Gestures step ("🧪 Thử Cử Chỉ")
        self.sandbox_row_widget = QWidget(self.card)
        self.sandbox_row_widget.setStyleSheet("background: transparent; border: none;")
        sb_row = QHBoxLayout(self.sandbox_row_widget)
        sb_row.setContentsMargins(0, 2, 0, 0)
        sb_row.setSpacing(8)
        self.sandbox_btn = QPushButton("🧪 Thử Cử Chỉ")
        self.sandbox_btn.setCursor(Qt.PointingHandCursor)
        self.sandbox_btn.setFixedHeight(32)
        self.sandbox_btn.setStyleSheet(
            f"QPushButton {{ background: #0F766E; color: {TEXT_LIGHT}; border: 1.5px solid {NEON_CYAN}; "
            f"border-radius: 8px; padding: 0 14px; font-size: 12px; font-weight: 800; }}"
            f"QPushButton:hover {{ background: #0D9488; }}"
        )
        self.sandbox_btn.clicked.connect(self._on_open_sandbox)
        sb_row.addWidget(self.sandbox_btn)
        sb_row.addStretch(1)
        root.addWidget(self.sandbox_row_widget)

        # Divider
        divider = QFrame(self.card)
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background: {BORDER_SLATE}; border: none;")
        root.addWidget(divider)

        # Footer: Checkbox + Navigation buttons
        foot = QHBoxLayout()
        foot.setSpacing(8)

        self.dont_show_chk = QCheckBox("Không hiện lại lần sau")
        self.dont_show_chk.setChecked(True)
        self.dont_show_chk.setCursor(Qt.PointingHandCursor)
        self.dont_show_chk.setStyleSheet(
            f"QCheckBox {{ color: {TEXT_MUTED}; font-size: 11.5px; font-weight: 600; background: transparent; border: none; spacing: 6px; }}"
            f"QCheckBox::indicator {{ width: 15px; height: 15px; border-radius: 4px; border: 1.5px solid {BORDER_SLATE}; background: {BG_DARK}; }}"
            f"QCheckBox::indicator:checked {{ background: {PRIMARY_INDIGO}; border-color: {NEON_CYAN}; }}"
        )
        foot.addWidget(self.dont_show_chk)
        foot.addStretch(1)

        self.prev_btn = QPushButton("Quay lại")
        self.prev_btn.setCursor(Qt.PointingHandCursor)
        self.prev_btn.setFixedHeight(32)
        self.prev_btn.setStyleSheet(
            f"QPushButton {{ background: {SURFACE_DARK}; color: {TEXT_LIGHT}; border: 1px solid {BORDER_SLATE}; "
            f"border-radius: 7px; padding: 0 12px; font-size: 12px; font-weight: 700; }}"
            f"QPushButton:hover {{ border-color: {TEXT_MUTED}; }}"
            f"QPushButton:disabled {{ color: #475569; border-color: #1E293B; background: #0F172A; }}"
        )
        self.prev_btn.clicked.connect(self.prev_step)
        foot.addWidget(self.prev_btn)

        self.next_btn = QPushButton("Tiếp theo")
        self.next_btn.setCursor(Qt.PointingHandCursor)
        self.next_btn.setFixedHeight(32)
        self.next_btn.clicked.connect(self.next_step)
        foot.addWidget(self.next_btn)

        root.addLayout(foot)

    def _rebuild_step_dots(self) -> None:
        while self._dots_row.count():
            item = self._dots_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._dot_widgets.clear()
        for _ in range(len(self._steps)):
            dot = QFrame(self.card)
            dot.setFixedSize(8, 6)
            self._dots_row.addWidget(dot)
            self._dot_widgets.append(dot)

    # ------------------------------------------------------------------
    # Public Navigation API
    # ------------------------------------------------------------------
    def start(self, step_idx: int = 0) -> None:
        """Start (or restart) the interactive spotlight tour at `step_idx`."""
        if not self._steps:
            self.hide()
            return
        self._step_idx = max(0, min(len(self._steps) - 1, int(step_idx)))
        parent = self.parentWidget()
        if parent is not None:
            self.setGeometry(parent.rect())
        self._apply_step_content()
        self.show()
        self.raise_()
        self._refresh_layout()
        self.setFocus(Qt.ActiveWindowFocusReason)

    def next_step(self) -> None:
        """Advance to the next step, or finish when on the final step."""
        if not self._steps:
            self.finish_tour()
            return
        if self._step_idx < len(self._steps) - 1:
            self._step_idx += 1
            self._apply_step_content()
            self._refresh_layout()
        else:
            self.finish_tour()

    def prev_step(self) -> None:
        """Step backward if not on the first step."""
        if self._step_idx > 0:
            self._step_idx -= 1
            self._apply_step_content()
            self._refresh_layout()

    def skip_tour(self) -> None:
        """Dismiss the tour early and emit `tour_finished(dont_show_again)`."""
        self.hide()
        self.tour_finished.emit(bool(self.dont_show_chk.isChecked()))

    def finish_tour(self) -> None:
        """Complete the tour cleanly and emit `tour_finished(dont_show_again)`."""
        self.hide()
        self.tour_finished.emit(bool(self.dont_show_chk.isChecked()))

    def _on_open_sandbox(self) -> None:
        self.hide()
        self.open_sandbox_requested.emit()

    def active_spotlight_rect(self) -> QRectF:
        """Return the current target cutout rectangle in overlay/parent coordinates."""
        self._spotlight_rect = self._compute_spotlight_rect()
        return QRectF(self._spotlight_rect)

    # ------------------------------------------------------------------
    # Geometry & Auto-Positioning Engine
    # ------------------------------------------------------------------
    def _current_step(self) -> Optional[SpotlightStep]:
        if 0 <= self._step_idx < len(self._steps):
            return self._steps[self._step_idx]
        return None

    def _compute_spotlight_rect(self) -> QRectF:
        w = float(max(1, self.width()))
        h = float(max(1, self.height()))
        step = self._current_step()
        if step is None:
            return QRectF(w * 0.25, h * 0.25, w * 0.50, h * 0.50)

        target: Optional[QWidget] = None
        try:
            target = step.target_getter()
        except Exception:
            target = None

        parent = self.parentWidget()
        rect: Optional[QRectF] = None

        if (
            isinstance(target, QWidget)
            and target.isVisible()
            and target.width() > 4
            and target.height() > 4
        ):
            if parent is not None and target is parent:
                tl = QPoint(0, 0)
            elif parent is not None and parent.isAncestorOf(target):
                tl = target.mapTo(parent, QPoint(0, 0))
            else:
                tl = self.mapFromGlobal(target.mapToGlobal(QPoint(0, 0)))
            pad = self.CUTOUT_PADDING_PX
            rect = QRectF(
                float(tl.x()) - pad,
                float(tl.y()) - pad,
                float(target.width()) + 2.0 * pad,
                float(target.height()) + 2.0 * pad,
            )

        if rect is None:
            nx, ny, nw, nh = step.fallback_norm_rect
            rect = QRectF(nx * w, ny * h, nw * w, nh * h)

        # Clamp strictly inside overlay bounds with subtle 4px inset so border stays visible
        inset = 4.0
        max_w = max(12.0, w - 2.0 * inset)
        max_h = max(12.0, h - 2.0 * inset)
        x1 = max(inset, min(w - inset - 12.0, rect.left()))
        y1 = max(inset, min(h - inset - 12.0, rect.top()))
        x2 = max(x1 + 12.0, min(w - inset, rect.right()))
        y2 = max(y1 + 12.0, min(h - inset, rect.bottom()))
        return QRectF(x1, y1, min(max_w, x2 - x1), min(max_h, y2 - y1))

    def _apply_step_content(self) -> None:
        step = self._current_step()
        if step is None:
            return
        n = len(self._steps)
        accent = step.accent_hex or NEON_CYAN
        accent_col = QColor(accent)
        if not accent_col.isValid():
            accent = NEON_CYAN
            accent_col = QColor(NEON_CYAN)

        self.progress_pill.setText(f"Bước {self._step_idx + 1} / {n}")
        self.progress_pill.setStyleSheet(
            f"background: rgba({accent_col.red()}, {accent_col.green()}, {accent_col.blue()}, 38); "
            f"color: {accent}; border: 1px solid {accent}; border-radius: 10px; "
            f"padding: 3px 10px; font-size: 11px; font-weight: 800;"
        )

        if len(self._dot_widgets) != n:
            self._rebuild_step_dots()
        for idx, dot in enumerate(self._dot_widgets):
            if idx == self._step_idx:
                dot.setFixedSize(18, 6)
                dot.setStyleSheet(f"background: {accent}; border-radius: 3px; border: none;")
            elif idx < self._step_idx:
                dot.setFixedSize(8, 6)
                dot.setStyleSheet(f"background: {SECONDARY_TEAL}; border-radius: 3px; border: none;")
            else:
                dot.setFixedSize(8, 6)
                dot.setStyleSheet(f"background: {BORDER_SLATE}; border-radius: 3px; border: none;")

        self.card.setStyleSheet(
            f"QFrame#SpotlightCalloutCard {{ "
            f"background: rgba(15, 23, 42, 246); "
            f"border: 1.8px solid {accent}; "
            f"border-radius: 14px; "
            f"}}"
        )

        self.title_lbl.setText(step.title)
        self.title_lbl.setStyleSheet(
            f"color: {accent}; font-size: 15px; font-weight: 800; line-height: 1.35; "
            f"background: transparent; border: none;"
        )

        self.body_lbl.setText(step.body)

        tip_txt = (step.tip or "").strip()
        self.tip_frame.setVisible(bool(tip_txt))
        if tip_txt:
            self.tip_lbl.setText(f"💡 Mẹo: {tip_txt}" if not tip_txt.lower().startswith("mẹo") else f"💡 {tip_txt}")
            self.tip_frame.setStyleSheet(
                f"background: rgba(30, 41, 59, 220); border: 1px solid {BORDER_SLATE}; "
                f"border-left: 3px solid {accent}; border-radius: 8px;"
            )
            self.tip_lbl.setStyleSheet(
                f"color: #E2E8F0; font-size: 11.5px; font-weight: 600; line-height: 1.45; "
                f"background: transparent; border: none;"
            )

        is_gestures = step.step_id == "gestures"
        self.sandbox_row_widget.setVisible(is_gestures)
        self.sandbox_btn.setVisible(is_gestures)

        self.prev_btn.setEnabled(self._step_idx > 0)
        is_last = self._step_idx == n - 1
        self.next_btn.setText("Hoàn tất ✓" if is_last else "Tiếp theo")
        if is_last:
            self.next_btn.setStyleSheet(
                f"QPushButton {{ background: {SUCCESS_GREEN}; color: #022C22; border: none; "
                f"border-radius: 7px; padding: 0 16px; font-size: 12px; font-weight: 800; }}"
                f"QPushButton:hover {{ background: #34D399; }}"
            )
        else:
            self.next_btn.setStyleSheet(
                f"QPushButton {{ background: {PRIMARY_INDIGO}; color: {TEXT_LIGHT}; border: none; "
                f"border-radius: 7px; padding: 0 16px; font-size: 12px; font-weight: 800; }}"
                f"QPushButton:hover {{ background: #6366F1; }}"
            )

    def _position_callout_card(self, cutout: QRectF) -> QRect:
        pw = max(1, self.width())
        ph = max(1, self.height())
        margin = int(min(self.EDGE_MARGIN_PX, max(4, pw // 20), max(4, ph // 20)))
        gap = self.CARD_GAP_PX

        max_avail_w = max(240, pw - 2 * margin)
        card_w = min(432, max_avail_w)
        self.card.setFixedWidth(card_w)
        lay = self.card.layout()
        if lay is not None:
            lay.invalidate()
            lay.activate()
        self.card.adjustSize()
        hint_h = self.card.sizeHint().height()
        if lay is not None and lay.hasHeightForWidth():
            hint_h = max(hint_h, lay.heightForWidth(card_w))
        elif self.card.hasHeightForWidth():
            hint_h = max(hint_h, self.card.heightForWidth(card_w))
        max_avail_h = max(140, ph - 2 * margin)
        card_h = min(max_avail_h, max(160, hint_h))

        def _clamp_xy(x_val: float, y_val: float) -> Tuple[int, int]:
            lo_x = float(margin)
            hi_x = max(lo_x, float(pw - margin - card_w))
            lo_y = float(margin)
            hi_y = max(lo_y, float(ph - margin - card_h))
            return int(round(max(lo_x, min(hi_x, x_val)))), int(round(max(lo_y, min(hi_y, y_val))))

        # Preferred order: below -> above -> right -> left
        raw_candidates: List[Tuple[str, float, float]] = [
            ("below", cutout.center().x() - card_w * 0.5, cutout.bottom() + gap),
            ("above", cutout.center().x() - card_w * 0.5, cutout.top() - gap - card_h),
            ("right", cutout.right() + gap, cutout.center().y() - card_h * 0.5),
            ("left", cutout.left() - gap - card_w, cutout.center().y() - card_h * 0.5),
        ]

        chosen_rect: Optional[QRect] = None
        best_fallback: Optional[Tuple[float, QRect]] = None

        for side, rx, ry in raw_candidates:
            cx, cy = _clamp_xy(rx, ry)
            cand = QRect(cx, cy, card_w, card_h)
            fits_side = (
                (side == "below" and ry + card_h <= ph - margin and cand.top() >= cutout.bottom() + 4.0)
                or (side == "above" and ry >= margin and cand.bottom() <= cutout.top() - 4.0)
                or (side == "right" and rx + card_w <= pw - margin and cand.left() >= cutout.right() + 4.0)
                or (side == "left" and rx >= margin and cand.right() <= cutout.left() - 4.0)
            )
            inter = QRectF(cand).intersected(cutout)
            overlap_area = max(0.0, inter.width()) * max(0.0, inter.height())
            if fits_side and overlap_area <= 1.0:
                chosen_rect = cand
                break
            if best_fallback is None or overlap_area < best_fallback[0]:
                best_fallback = (overlap_area, cand)

        if chosen_rect is None:
            assert best_fallback is not None
            chosen_rect = best_fallback[1]

        # Final hard clamp so card NEVER clips off-screen
        final_x = max(0, min(max(0, pw - card_w), chosen_rect.x()))
        final_y = max(0, min(max(0, ph - card_h), chosen_rect.y()))
        final_w = min(card_w, pw)
        final_h = min(card_h, ph)
        final_rect = QRect(final_x, final_y, final_w, final_h)
        self.card.setGeometry(final_rect)
        return final_rect

    def _refresh_layout(self) -> None:
        self._spotlight_rect = self._compute_spotlight_rect()
        self._position_callout_card(self._spotlight_rect)
        self.update()

    def _refresh_if_visible(self) -> None:
        parent = self.parentWidget()
        if self.isVisible() and parent is not None:
            self.setGeometry(parent.rect())
            self._refresh_layout()

    # ------------------------------------------------------------------
    # Qt Events & Painting
    # ------------------------------------------------------------------
    def eventFilter(self, watched: Any, event: QEvent) -> bool:
        parent = self.parentWidget()
        if watched is parent:
            etype = event.type()
            if etype in (QEvent.Resize, QEvent.Show, QEvent.LayoutRequest):
                self.setGeometry(parent.rect())
                if self.isVisible():
                    self.raise_()
                    self._refresh_layout()
                    QTimer.singleShot(0, self._refresh_if_visible)
            elif etype == QEvent.KeyPress and self.isVisible():
                if isinstance(event, QKeyEvent) and event.key() == Qt.Key_Escape:
                    self.skip_tour()
                    return True
        elif watched is self.card and event.type() == QEvent.KeyPress and self.isVisible():
            if isinstance(event, QKeyEvent) and event.key() == Qt.Key_Escape:
                self.skip_tour()
                return True
        return super().eventFilter(watched, event)

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if self._steps:
            self._refresh_layout()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            event.accept()
            self.skip_tour()
            return
        super().keyPressEvent(event)

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        outer = QRectF(self.rect())
        cutout = self._compute_spotlight_rect()
        self._spotlight_rect = cutout

        # 1. Semi-transparent dark backdrop with OddEvenFill rounded cutout
        backdrop_path = QPainterPath()
        backdrop_path.setFillRule(Qt.OddEvenFill)
        backdrop_path.addRect(outer)
        backdrop_path.addRoundedRect(cutout, self.CUTOUT_RADIUS_PX, self.CUTOUT_RADIUS_PX)

        painter.setPen(Qt.NoPen)
        painter.setBrush(BACKDROP_COLOR)
        painter.drawPath(backdrop_path)

        # 2. Glowing Neon Cyan / step accent border + layered outer soft halo ring
        step = self._current_step()
        accent = QColor(step.accent_hex if step is not None else NEON_CYAN)
        if not accent.isValid():
            accent = QColor(NEON_CYAN)

        painter.setBrush(Qt.NoBrush)
        for expand_px, alpha, pen_w in (
            (10.0, 34, 4.0),
            (6.0, 68, 3.0),
            (2.8, 128, 2.2),
        ):
            halo_col = QColor(accent.red(), accent.green(), accent.blue(), alpha)
            painter.setPen(QPen(halo_col, pen_w))
            halo_rect = cutout.adjusted(-expand_px, -expand_px, expand_px, expand_px)
            painter.drawRoundedRect(
                halo_rect,
                self.CUTOUT_RADIUS_PX + expand_px * 0.6,
                self.CUTOUT_RADIUS_PX + expand_px * 0.6,
            )

        # Primary crisp accent border
        painter.setPen(QPen(accent, 2.6))
        painter.drawRoundedRect(cutout, self.CUTOUT_RADIUS_PX, self.CUTOUT_RADIUS_PX)

        # Subtle inner luminous rim for maximum clarity
        inner_rim = cutout.adjusted(1.2, 1.2, -1.2, -1.2)
        if inner_rim.width() > 4.0 and inner_rim.height() > 4.0:
            painter.setPen(QPen(QColor(248, 250, 252, 150), 1.0))
            painter.drawRoundedRect(
                inner_rim,
                max(2.0, self.CUTOUT_RADIUS_PX - 1.2),
                max(2.0, self.CUTOUT_RADIUS_PX - 1.2),
            )
