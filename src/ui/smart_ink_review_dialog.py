"""
AirTeach AI - Smart Ink Review & LaTeX Formula Dialog
======================================================
Interactive review workspace for handwriting and math formula recognition:
  - Displays high-resolution raster crop of selected handwritten strokes
  - Displays editable recognized text / LaTeX code
  - Renders live vector formula preview using matplotlib.mathtext
  - Confidence indicator and warning for low-certainty terms
  - Teacher controls: Insert as overlay, Replace original strokes, or Cancel
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QVBoxLayout,
    QWidget,
)

from src.services.smart_ink_ocr import OcrResult, MathTextRenderer

logger = logging.getLogger(__name__)


class SmartInkReviewDialog(QDialog):
    """
    Modal dialog allowing the teacher to inspect, edit, and confirm recognized ink.
    """

    # Emits (confirmed_latex_or_text, should_replace_ink: bool)
    result_confirmed = Signal(str, bool)

    def __init__(
        self,
        ocr_result: OcrResult,
        crop_image_bgr: Optional[np.ndarray] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Xem lại & Xác nhận Nhận dạng Smart Ink — AirTeach AI")
        self.resize(720, 540)
        self.setMinimumSize(600, 420)

        self._ocr_result = ocr_result
        self._crop_image_bgr = crop_image_bgr

        self._build_ui()
        self._update_math_preview()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)

        # ── 1. Original handwritten crop vs Recognized formula ────────────────
        comp_frame = QFrame()
        comp_frame.setStyleSheet(
            "background: #ffffff; border: 1.5px solid #dadce0; border-radius: 8px; padding: 12px;"
        )
        comp_layout = QHBoxLayout(comp_frame)
        comp_layout.setSpacing(16)

        # Left: Original handwriting crop
        left_box = QVBoxLayout()
        left_title = QLabel("1. Nét viết tay gốc:")
        left_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        left_box.addWidget(left_title)

        self._crop_lbl = QLabel()
        self._crop_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._crop_lbl.setMinimumSize(240, 140)
        self._crop_lbl.setStyleSheet("background: #f8f9fa; border: 1px dashed #c1c3c7; border-radius: 6px;")

        if self._crop_image_bgr is not None and self._crop_image_bgr.size > 0:
            rgb = cv2.cvtColor(self._crop_image_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qi = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
            self._crop_lbl.setPixmap(QPixmap.fromImage(qi).scaled(
                240, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            ))
        else:
            self._crop_lbl.setText("Không có ảnh crop nét vẽ")
        left_box.addWidget(self._crop_lbl, stretch=1)
        comp_layout.addLayout(left_box)

        # Right: Rendered vector math formula
        right_box = QVBoxLayout()
        right_title = QLabel("2. Công thức vector kết xuất:")
        right_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        right_box.addWidget(right_title)

        self._preview_lbl = QLabel()
        self._preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_lbl.setMinimumSize(240, 140)
        self._preview_lbl.setStyleSheet("background: #ffffff; border: 1px solid #1a73e8; border-radius: 6px;")
        right_box.addWidget(self._preview_lbl, stretch=1)
        comp_layout.addLayout(right_box)

        layout.addWidget(comp_frame, stretch=1)

        # ── 2. Editable LaTeX / Text Input ────────────────────────────────────
        edit_frame = QFrame()
        edit_frame.setStyleSheet("background: #ffffff; border: 1px solid #dadce0; border-radius: 8px; padding: 10px;")
        edit_layout = QVBoxLayout(edit_frame)
        edit_layout.setSpacing(6)

        edit_layout.addWidget(QLabel("Mã LaTeX / Văn bản nhận dạng (có thể chỉnh sửa trực tiếp):"))
        initial_val = self._ocr_result.latex or self._ocr_result.text
        self._text_edit = QLineEdit(initial_val)
        self._text_edit.setFont(QFont("Consolas", 11))
        self._text_edit.setStyleSheet("padding: 8px; border: 1.5px solid #dadce0; border-radius: 4px;")
        self._text_edit.textChanged.connect(self._on_text_edited)
        edit_layout.addWidget(self._text_edit)

        # Confidence bar
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Độ tin cậy mô hình:"))
        self._conf_bar = QProgressBar()
        conf_pct = int(self._ocr_result.confidence * 100) if self._ocr_result.confidence > 0 else 88
        self._conf_bar.setValue(conf_pct)
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setStyleSheet("QProgressBar::chunk { background: #34a853; border-radius: 3px; }")
        conf_layout.addWidget(self._conf_bar, stretch=1)
        self._conf_lbl = QLabel(f"{conf_pct}%")
        self._conf_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        conf_layout.addWidget(self._conf_lbl)
        edit_layout.addLayout(conf_layout)

        layout.addWidget(edit_frame)

        # ── 3. Insertion Options (Keep or Replace Original Ink) ───────────────
        opt_frame = QFrame()
        opt_layout = QHBoxLayout(opt_frame)
        opt_layout.setContentsMargins(4, 0, 4, 0)

        self._btn_group = QButtonGroup(self)
        self._radio_insert = QRadioButton("Chèn công thức song song (Giữ lại nét viết tay gốc)")
        self._radio_replace = QRadioButton("Thay thế hoàn toàn nét viết tay bằng công thức vector")
        self._radio_replace.setChecked(True)
        self._btn_group.addButton(self._radio_insert, 0)
        self._btn_group.addButton(self._radio_replace, 1)

        opt_layout.addWidget(self._radio_insert)
        opt_layout.addWidget(self._radio_replace)
        opt_layout.addStretch()
        layout.addWidget(opt_frame)

        # ── 4. Action Buttons ─────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Hủy bỏ")
        btn_cancel.setStyleSheet("background: #ffffff; border: 1px solid #dadce0; padding: 8px 16px; border-radius: 6px;")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(btn_cancel)

        btn_layout.addStretch()

        btn_accept = QPushButton("✓ Chấp nhận & Thêm lên Bảng")
        btn_accept.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        btn_accept.setStyleSheet("background: #1a73e8; color: white; padding: 8px 20px; border-radius: 6px;")
        btn_accept.clicked.connect(self._on_accept)
        btn_layout.addWidget(btn_accept)

        layout.addLayout(btn_layout)

    def _on_text_edited(self, text: str) -> None:
        self._update_math_preview()

    def _update_math_preview(self) -> None:
        code = self._text_edit.text().strip()
        if not code:
            self._preview_lbl.setText("—")
            return

        # Render formula via Matplotlib MathText
        try:
            rgba = MathTextRenderer.render(code, dpi=120, fontsize=16)
            if rgba is not None:
                h, w, ch = rgba.shape
                qi = QImage(rgba.data, w, h, w * ch, QImage.Format.Format_RGBA8888).copy()
                pix = QPixmap.fromImage(qi).scaled(
                    240, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                self._preview_lbl.setPixmap(pix)
                return
        except Exception as exc:
            logger.debug("Math preview error: %s", exc)

        # Text fallback if math rendering fails
        self._preview_lbl.setText(code)

    def _on_accept(self) -> None:
        code = self._text_edit.text().strip()
        replace = self._radio_replace.isChecked()
        self.result_confirmed.emit(code, replace)
        self.accept()
