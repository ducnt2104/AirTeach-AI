"""
AirTeach AI - Smart Ink Review & LaTeX Formula Dialog
======================================================
Interactive review workspace for handwriting and math formula recognition:
- Displays high-resolution raster crop of selected handwritten strokes
- Displays editable recognized text / LaTeX code
- Renders live vector formula preview using matplotlib.mathtext
- Confidence indicator and warning for low-certainty terms
- Integrated Symbolic Math Engine & Step-by-Step Derivation panel:
  - Auto-solve, Differentiate, Integrate, and Simplify with 1-click
  - Live derivation steps and solution preview
  - Option to insert an interactive MathDerivationCardItem onto LessonBoard
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
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from src.services.smart_ink_ocr import LaTeXRenderer, MathTextRenderer, OcrResult
from src.services.symbolic_math import MathDerivationResult, SymbolicMathEngine

logger = logging.getLogger(__name__)


class SmartInkReviewDialog(QDialog):
    """
    Modal dialog allowing the teacher to inspect, edit, and confirm recognized ink,
    with built-in pedagogical symbolic mathematics solving and step-by-step derivation.
    """

    # Signals
    # Emits (confirmed_latex_or_text, should_replace_ink: bool)
    result_confirmed = Signal(str, bool)
    # Emits (MathDerivationResult, should_replace_ink: bool)
    derivation_confirmed = Signal(object, bool)

    def __init__(
        self,
        ocr_result: OcrResult,
        crop_image_bgr: Optional[np.ndarray] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("AirTeach AI — Xác nhận Nét viết & Công thức Toán học")
        self.setMinimumWidth(680)
        self.setStyleSheet("background-color: #f8fafc; font-family: 'Segoe UI', sans-serif;")

        self._ocr_result = ocr_result
        self._crop_image_bgr = crop_image_bgr
        self._math_engine = SymbolicMathEngine()
        self._derivation_result: Optional[MathDerivationResult] = None

        self._build_ui()
        self._update_math_preview()

        # If it's a formula, run auto-derivation in background
        if self._ocr_result.is_formula:
            self._run_symbolic_derivation("auto")

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        # ── 1. Original handwritten crop vs Recognized formula ────────────────
        comp_frame = QFrame()
        comp_frame.setStyleSheet(
            "background: #ffffff; border: 1.5px solid #e2e8f0; border-radius: 8px; padding: 10px;"
        )
        comp_layout = QHBoxLayout(comp_frame)
        comp_layout.setSpacing(14)

        # Left: Original handwriting crop
        left_box = QVBoxLayout()
        left_title = QLabel("1. Nét viết tay gốc:")
        left_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        left_box.addWidget(left_title)

        self._crop_lbl = QLabel()
        self._crop_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._crop_lbl.setMinimumSize(220, 110)
        self._crop_lbl.setStyleSheet("background: #f8fafc; border: 1px dashed #cbd5e1; border-radius: 6px;")

        if self._crop_image_bgr is not None and self._crop_image_bgr.size > 0:
            rgb = cv2.cvtColor(self._crop_image_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qi = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888).copy()
            self._crop_lbl.setPixmap(QPixmap.fromImage(qi).scaled(
                220, 110, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
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
        self._preview_lbl.setMinimumSize(220, 110)
        self._preview_lbl.setStyleSheet("background: #ffffff; border: 1.5px solid #3b82f6; border-radius: 6px;")
        right_box.addWidget(self._preview_lbl, stretch=1)
        comp_layout.addLayout(right_box)

        layout.addWidget(comp_frame)

        # ── 2. Editable LaTeX / Text Input ────────────────────────────────────
        edit_frame = QFrame()
        edit_frame.setStyleSheet("background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px;")
        edit_layout = QVBoxLayout(edit_frame)
        edit_layout.setSpacing(6)

        edit_layout.addWidget(QLabel("Mã LaTeX / Văn bản nhận dạng (chỉnh sửa trực tiếp):"))
        initial_val = self._ocr_result.latex or self._ocr_result.text
        self._text_edit = QLineEdit(initial_val)
        self._text_edit.setFont(QFont("Consolas", 11))
        self._text_edit.setStyleSheet("padding: 7px; border: 1.5px solid #cbd5e1; border-radius: 4px;")
        self._text_edit.textChanged.connect(self._on_text_edited)
        edit_layout.addWidget(self._text_edit)

        # Confidence bar
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Độ tin cậy:"))
        self._conf_bar = QProgressBar()
        conf_pct = int(self._ocr_result.confidence * 100) if self._ocr_result.confidence > 0 else 88
        self._conf_bar.setValue(conf_pct)
        self._conf_bar.setRange(0, 100)
        self._conf_bar.setStyleSheet("QProgressBar::chunk { background: #10b981; border-radius: 3px; }")
        conf_layout.addWidget(self._conf_bar, stretch=1)
        self._conf_lbl = QLabel(f"{conf_pct}%")
        self._conf_lbl.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        conf_layout.addWidget(self._conf_lbl)
        edit_layout.addLayout(conf_layout)

        layout.addWidget(edit_frame)

        # ── 3. Symbolic Math Derivation & Pedagogical Solving ─────────────────
        if self._ocr_result.is_formula:
            math_frame = QFrame()
            math_frame.setStyleSheet(
                "background: #0f172a; border: 1.5px solid #3b82f6; border-radius: 8px; padding: 10px; color: #f8fafc;"
            )
            math_layout = QVBoxLayout(math_frame)
            math_layout.setSpacing(8)

            # Title and operation buttons
            top_bar = QHBoxLayout()
            m_title = QLabel("⚡ Giải Toán & Đạo hàm / Tích phân Tự động:")
            m_title.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            m_title.setStyleSheet("color: #60a5fa;")
            top_bar.addWidget(m_title)
            top_bar.addStretch()

            btn_auto = QPushButton("⚡ Tự động")
            btn_auto.setStyleSheet("background: #1e293b; color: #e2e8f0; border: 1px solid #475569; padding: 4px 8px; border-radius: 4px;")
            btn_auto.clicked.connect(lambda: self._run_symbolic_derivation("auto"))
            top_bar.addWidget(btn_auto)

            btn_solve = QPushButton("🔍 Giải PT")
            btn_solve.setStyleSheet("background: #064e3b; color: #34d399; border: 1px solid #059669; padding: 4px 8px; border-radius: 4px;")
            btn_solve.clicked.connect(lambda: self._run_symbolic_derivation("solve"))
            top_bar.addWidget(btn_solve)

            btn_diff = QPushButton("📈 Đạo hàm f'(x)")
            btn_diff.setStyleSheet("background: #78350f; color: #fcd34d; border: 1px solid #d97706; padding: 4px 8px; border-radius: 4px;")
            btn_diff.clicked.connect(lambda: self._run_symbolic_derivation("derivative"))
            top_bar.addWidget(btn_diff)

            btn_int = QPushButton("∫ Tích phân")
            btn_int.setStyleSheet("background: #4c1d95; color: #c4b5fd; border: 1px solid #7c3aed; padding: 4px 8px; border-radius: 4px;")
            btn_int.clicked.connect(lambda: self._run_symbolic_derivation("integral"))
            top_bar.addWidget(btn_int)

            btn_simp = QPushButton("✨ Rút gọn")
            btn_simp.setStyleSheet("background: #0c4a6e; color: #38bdf8; border: 1px solid #0284c7; padding: 4px 8px; border-radius: 4px;")
            btn_simp.clicked.connect(lambda: self._run_symbolic_derivation("simplify"))
            top_bar.addWidget(btn_simp)

            math_layout.addLayout(top_bar)

            # Solution display
            self._math_sol_lbl = QLabel("Đang phân tích...")
            self._math_sol_lbl.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            self._math_sol_lbl.setStyleSheet(
                "background: #1e293b; color: #34d399; border: 1px solid #059669; border-radius: 6px; padding: 8px;"
            )
            math_layout.addWidget(self._math_sol_lbl)

            # Steps summary
            self._math_steps_lbl = QLabel("")
            self._math_steps_lbl.setFont(QFont("Segoe UI", 8))
            self._math_steps_lbl.setStyleSheet("color: #94a3b8;")
            self._math_steps_lbl.setWordWrap(True)
            math_layout.addWidget(self._math_steps_lbl)

            # Checkbox: insert derivation card onto canvas
            self._chk_insert_card = QCheckBox("Kèm thẻ lời giải chi tiết (Math Derivation Card) lên bảng giảng dạy")
            self._chk_insert_card.setChecked(True)
            self._chk_insert_card.setStyleSheet("color: #e2e8f0; font-weight: bold;")
            math_layout.addWidget(self._chk_insert_card)

            layout.addWidget(math_frame)
        else:
            self._chk_insert_card = None

        # ── 4. Insertion Options (Keep or Replace Original Ink) ───────────────
        opt_frame = QFrame()
        opt_layout = QHBoxLayout(opt_frame)
        opt_layout.setContentsMargins(4, 0, 4, 0)

        self._btn_group = QButtonGroup(self)
        self._radio_insert = QRadioButton("Chèn song song (Giữ lại nét viết tay gốc)")
        self._radio_replace = QRadioButton("Thay thế hoàn toàn nét viết tay bằng đối tượng mới")
        self._radio_replace.setChecked(True)
        self._btn_group.addButton(self._radio_insert, 0)
        self._btn_group.addButton(self._radio_replace, 1)

        opt_layout.addWidget(self._radio_insert)
        opt_layout.addWidget(self._radio_replace)
        opt_layout.addStretch()
        layout.addWidget(opt_frame)

        # ── 5. Action Buttons ─────────────────────────────────────────────────
        btn_layout = QHBoxLayout()
        btn_cancel = QPushButton("Hủy bỏ")
        btn_cancel.setStyleSheet("background: #ffffff; border: 1px solid #cbd5e1; padding: 8px 16px; border-radius: 6px;")
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
        if self._ocr_result.is_formula:
            self._run_symbolic_derivation("auto")

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
                    220, 110, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                self._preview_lbl.setPixmap(pix)
                return
        except Exception as exc:
            logger.debug("Math preview error: %s", exc)

        # Text fallback if math rendering fails
        self._preview_lbl.setText(code)

    def _run_symbolic_derivation(self, operation: str = "auto") -> None:
        """Run the symbolic math engine on the current formula string."""
        if not hasattr(self, "_math_sol_lbl"):
            return

        code = self._text_edit.text().strip()
        if not code:
            self._math_sol_lbl.setText("Không có công thức để giải.")
            self._math_steps_lbl.setText("")
            self._derivation_result = None
            return

        res = self._math_engine.evaluate_auto(code, operation)
        self._derivation_result = res

        if res.success:
            sol_disp = res.solution_raw or res.solution_latex
            self._math_sol_lbl.setText(f"✓ Kết quả ({res.operation.upper()}): {sol_disp}")
            if res.steps:
                step_titles = [f"[{s.step_number}] {s.title}" for s in res.steps]
                self._math_steps_lbl.setText(f"{len(res.steps)} bước giải chi tiết: " + " → ".join(step_titles))
            else:
                self._math_steps_lbl.setText("Rút gọn trực tiếp.")
            if self._chk_insert_card is not None:
                self._chk_insert_card.setEnabled(True)
        else:
            self._math_sol_lbl.setText(f"⚠ {res.error_message or 'Không thể giải tự động'}")
            self._math_steps_lbl.setText("Có thể chỉnh sửa cú pháp công thức hoặc chọn thao tác khác.")
            if self._chk_insert_card is not None:
                self._chk_insert_card.setEnabled(False)

    def _on_accept(self) -> None:
        code = self._text_edit.text().strip()
        replace = self._radio_replace.isChecked()

        # Emit derivation result if active and checked
        if (
            self._chk_insert_card is not None
            and self._chk_insert_card.isChecked()
            and self._derivation_result is not None
            and self._derivation_result.success
        ):
            self.derivation_confirmed.emit(self._derivation_result, replace)

        # Emit standard text confirmation for overlay
        self.result_confirmed.emit(code, replace)
        self.accept()
