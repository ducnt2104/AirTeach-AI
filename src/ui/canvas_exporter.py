"""
Canvas Image Exporter for AirTeach AI.

Provides high-resolution image rendering, 2x supersampling (4K UHD),
transparent canvas export, and batch multi-page lesson exports
to PNG and JPEG.
"""

from __future__ import annotations
import uuid

import math
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from PySide6.QtWidgets import QGraphicsItem, QGraphicsScene
    from src.ui.main_window import MainWindow


class CanvasExporter:
    """Core static utilities for rendering QGraphicsScene to high-res images."""

    @staticmethod
    def render_scene_to_image(
        scene: QGraphicsScene,
        width: int = 1920,
        height: int = 1080,
        scale: float = 1.0,
        transparent_bg: bool = False,
        include_overlays: bool = False,
    ) -> QImage:
        """
        Render a QGraphicsScene to a high-resolution QImage.

        Args:
            scene: The QGraphicsScene (e.g. LessonBoard).
            width: Base target width in pixels (default 1920).
            height: Base target height in pixels (default 1080).
            scale: Resolution supersampling factor (e.g. 1.0 for 1080p, 2.0 for 4K UHD).
            transparent_bg: If True, background is transparent (useful for transparent PNGs).
            include_overlays: If False, hides temporary interactive items (laser reticles,
                              eraser reticles, spotlight, loupe, live drawing previews).
        """
        target_w = max(1, int(round(width * scale)))
        target_h = max(1, int(round(height * scale)))

        # Temporarily hide interactive overlay items if requested
        hidden_items: list[tuple[QGraphicsItem, bool]] = []
        old_bg_brush: Optional[QBrush] = None

        if not include_overlays:
            # Multi-laser reticles
            lasers = getattr(scene, "_laser_items", {})
            for item in lasers.values():
                if item is not None and item.isVisible():
                    hidden_items.append((item, item.isVisible()))
                    item.setVisible(False)

            # Multi-eraser reticles
            erasers = getattr(scene, "_eraser_items", {})
            for item in erasers.values():
                if item is not None and item.isVisible():
                    hidden_items.append((item, item.isVisible()))
                    item.setVisible(False)

            # Live drawing previews
            previews = getattr(scene, "_live_previews", {})
            for prev_info in previews.values():
                p_item = prev_info.get("item")
                if p_item is not None and p_item.isVisible():
                    hidden_items.append((p_item, p_item.isVisible()))
                    p_item.setVisible(False)

            # Legacy live preview
            p_leg = getattr(scene, "_preview_item", None)
            if p_leg is not None and p_leg.isVisible():
                hidden_items.append((p_leg, p_leg.isVisible()))
                p_leg.setVisible(False)

            # Selection bounding rect
            sel_rect = getattr(scene, "_select_rect_item", None)
            if sel_rect is not None and sel_rect.isVisible():
                hidden_items.append((sel_rect, sel_rect.isVisible()))
                sel_rect.setVisible(False)

            # Spotlight & Virtual Loupe
            spot = getattr(scene, "_spotlight_item", None)
            if spot is not None and spot.isVisible():
                hidden_items.append((spot, spot.isVisible()))
                spot.setVisible(False)

            loupe = getattr(scene, "_loupe_item", None)
            if loupe is not None and loupe.isVisible():
                hidden_items.append((loupe, loupe.isVisible()))
                loupe.setVisible(False)

            timer_itm = getattr(scene, "_timer_item", None)
            if timer_itm is not None and timer_itm.isVisible():
                hidden_items.append((timer_itm, timer_itm.isVisible()))
                timer_itm.setVisible(False)

            metronome_itm = getattr(scene, "_metronome_item", None)
            if metronome_itm is not None and metronome_itm.isVisible():
                hidden_items.append((metronome_itm, metronome_itm.isVisible()))
                metronome_itm.setVisible(False)

            compass_itm = getattr(scene, "_compass_item", None)
            if compass_itm is not None and compass_itm.isVisible():
                hidden_items.append((compass_itm, compass_itm.isVisible()))
                compass_itm.setVisible(False)

            # Laser trails
            trails = getattr(scene, "_laser_trails", {})
            for t_item in trails.values():
                if t_item is not None and t_item.isVisible():
                    hidden_items.append((t_item, t_item.isVisible()))
                    t_item.setVisible(False)

        if transparent_bg:
            old_bg_brush = scene.backgroundBrush()
            scene.setBackgroundBrush(QBrush(Qt.GlobalColor.transparent))

        try:
            image = QImage(target_w, target_h, QImage.Format.Format_ARGB32_Premultiplied)
            if transparent_bg:
                image.fill(Qt.GlobalColor.transparent)
            else:
                image.fill(Qt.GlobalColor.white)

            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            target_rect = QRectF(0.0, 0.0, float(target_w), float(target_h))
            source_rect = scene.sceneRect()
            if source_rect.isEmpty():
                source_rect = QRectF(0.0, 0.0, float(width), float(height))

            scene.render(painter, target_rect, source_rect)
            painter.end()

            return image
        finally:
            # Restore scene background brush if modified
            if old_bg_brush is not None:
                scene.setBackgroundBrush(old_bg_brush)

            # Restore visibility of temporarily hidden items
            for item, was_vis in hidden_items:
                try:
                    item.setVisible(was_vis)
                except Exception:
                    pass

    @classmethod
    def export_scene_to_file(
        cls,
        scene: QGraphicsScene,
        file_path: Union[str, Path],
        scale: float = 1.0,
        quality: int = 95,
        transparent_bg: bool = False,
        include_overlays: bool = False,
    ) -> Path:
        """
        Export a scene to a PNG or JPEG file on disk.

        Args:
            scene: The QGraphicsScene to render.
            file_path: Target path (.png, .jpg, .jpeg).
            scale: Resolution multiplier (1.0 = 1080p, 2.0 = 4K).
            quality: Compression quality for JPG (0-100, default 95).
            transparent_bg: Transparent canvas for PNG.
            include_overlays: Whether to keep in-air cursors.

        Returns:
            Resolved Path of the saved image.
        """
        out_p = Path(file_path).resolve()
        out_p.parent.mkdir(parents=True, exist_ok=True)

        is_png = out_p.suffix.lower() == ".png"
        use_transparent = transparent_bg and is_png

        qimage = cls.render_scene_to_image(
            scene=scene,
            width=1920,
            height=1080,
            scale=scale,
            transparent_bg=use_transparent,
            include_overlays=include_overlays,
        )

        ext = out_p.suffix.lower()
        temp_out = out_p.with_name(f".tmp_{uuid.uuid4().hex}_{out_p.name}")
        try:
            if ext in (".jpg", ".jpeg"):
                # Ensure background is opaque for JPEG
                if qimage.hasAlphaChannel():
                    opaque_img = QImage(qimage.size(), QImage.Format.Format_RGB32)
                    opaque_img.fill(Qt.GlobalColor.white)
                    p = QPainter(opaque_img)
                    p.drawImage(0, 0, qimage)
                    p.end()
                    qimage = opaque_img
                saved = qimage.save(str(temp_out), "JPEG", quality)
            else:
                saved = qimage.save(str(temp_out), "PNG")

            if not saved:
                raise IOError(f"Không thể ghi tệp tạm thời: {temp_out}")

            temp_out.replace(out_p)
        except Exception:
            if temp_out.exists():
                try:
                    temp_out.unlink()
                except Exception:
                    pass
            raise

        return out_p

    @classmethod
    def export_all_pages(
        cls,
        main_window: MainWindow,
        output_dir: Union[str, Path],
        prefix: str = "trang",
        ext: str = ".png",
        scale: float = 1.0,
        quality: int = 95,
        transparent_bg: bool = False,
    ) -> list[Path]:
        """
        Batch export all lesson pages to numbered image files in output_dir.

        Returns:
            List of Paths to all exported page images.
        """
        out_d = Path(output_dir).resolve()
        out_d.mkdir(parents=True, exist_ok=True)

        pages = getattr(main_window, "_pages", [])
        if not pages:
            # Single whiteboard canvas export
            p = out_d / f"{prefix}_01{ext}"
            saved = cls.export_scene_to_file(
                scene=main_window._scene,
                file_path=p,
                scale=scale,
                quality=quality,
                transparent_bg=transparent_bg,
            )
            return [saved]

        # Multi-page presentation export
        main_window._stash_current_page_strokes()
        original_idx = getattr(main_window, "_current_page_index", 0)
        exported: list[Path] = []

        try:
            total_pages = len(pages)
            num_digits = max(2, len(str(total_pages)))

            for idx in range(total_pages):
                main_window._current_page_index = idx
                main_window._show_current_page()

                page_num_str = str(idx + 1).zfill(num_digits)
                file_name = f"{prefix}_{page_num_str}{ext}"
                target_path = out_d / file_name

                saved_p = cls.export_scene_to_file(
                    scene=main_window._scene,
                    file_path=target_path,
                    scale=scale,
                    quality=quality,
                    transparent_bg=transparent_bg,
                )
                exported.append(saved_p)
        finally:
            main_window._current_page_index = original_idx
            main_window._show_current_page()

        return exported


class ExportImageDialog(QDialog):
    """
    Interactive export dialog allowing teachers to customize
    resolution, format, scope, and destination path.
    """

    def __init__(self, main_window: MainWindow, parent: Optional[QWidget] = None) -> None:
        q_parent = parent
        if q_parent is None and isinstance(main_window, QWidget):
            q_parent = main_window
        super().__init__(q_parent)
        self.main_window = main_window
        self.setWindowTitle("Xuất ảnh Bảng vẽ / Bài giảng (High-Res Export)")
        self.setMinimumWidth(500)
        self.setStyleSheet("""
            QDialog {
                background: #1e1e2e;
                color: #cdd6f4;
                font-family: 'Segoe UI', system-ui, sans-serif;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 8px;
                margin-top: 14px;
                padding-top: 10px;
                font-weight: bold;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
            }
            QRadioButton, QCheckBox, QLabel {
                color: #cdd6f4;
                font-size: 13px;
            }
            QComboBox, QLineEdit {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 6px 10px;
                color: #ffffff;
                font-size: 13px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QPushButton {
                background: #313244;
                color: #ffffff;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 7px 16px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #45475a;
            }
            QPushButton#btn_export {
                background: #89b4fa;
                color: #11111b;
                font-weight: bold;
                border: none;
            }
            QPushButton#btn_export:hover {
                background: #b4befe;
            }
            QProgressBar {
                background: #313244;
                border-radius: 4px;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #a6e3a1;
                border-radius: 4px;
            }
        """)

        pages = getattr(main_window, "_pages", [])
        self.total_pages = len(pages)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(18, 18, 18, 18)

        # ── Scope Group ──
        scope_box = QGroupBox("Phạm vi xuất ảnh")
        scope_layout = QVBoxLayout(scope_box)
        self.radio_current = QRadioButton(
            f"Trang hiện tại (Trang {getattr(self.main_window, '_current_page_index', 0) + 1})"
            if self.total_pages > 0 else "Bảng vẽ hiện tại"
        )
        self.radio_current.setChecked(True)
        scope_layout.addWidget(self.radio_current)

        self.radio_all = QRadioButton(f"Toàn bộ bài giảng ({self.total_pages} trang)")
        self.radio_all.setEnabled(self.total_pages > 1)
        scope_layout.addWidget(self.radio_all)
        layout.addWidget(scope_box)

        # ── Format & Resolution Group ──
        fmt_box = QGroupBox("Định dạng & Độ phân giải")
        fmt_layout = QVBoxLayout(fmt_box)

        # Format row
        row_fmt = QHBoxLayout()
        row_fmt.addWidget(QLabel("Định dạng tệp:"))
        self.combo_format = QComboBox()
        self.combo_format.addItems([
            "PNG (.png) — Độ nét tuyệt đối, không vỡ nét",
            "JPEG (.jpg) — Nén tối ưu, dung lượng nhẹ",
        ])
        self.combo_format.currentIndexChanged.connect(self._on_format_changed)
        row_fmt.addWidget(self.combo_format, stretch=1)
        fmt_layout.addLayout(row_fmt)

        # Resolution row
        row_res = QHBoxLayout()
        row_res.addWidget(QLabel("Độ phân giải:"))
        self.combo_res = QComboBox()
        self.combo_res.addItem("1080p Full HD (1920 × 1080) — Chuẩn trình chiếu", 1.0)
        self.combo_res.addItem("1440p 2K QHD (2560 × 1440) — Nét cao", 1.3333)
        self.combo_res.addItem("2160p 4K UHD (3840 × 2160) — Siêu nét (Supersampling 2x)", 2.0)
        self.combo_res.setCurrentIndex(0)
        row_res.addWidget(self.combo_res, stretch=1)
        fmt_layout.addLayout(row_res)

        # Options
        self.chk_transparent = QCheckBox("Nền trong suốt (Chỉ áp dụng khi xuất bảng vẽ PNG không có trang nền)")
        self.chk_transparent.setChecked(False)
        fmt_layout.addWidget(self.chk_transparent)

        layout.addWidget(fmt_box)

        # ── Destination Path Group ──
        dest_box = QGroupBox("Nơi lưu trữ")
        dest_layout = QHBoxLayout(dest_box)
        self.txt_dest = QLineEdit()
        default_dir = Path.home() / "Pictures" / "AirTeach"
        default_dir.mkdir(parents=True, exist_ok=True)
        default_name = default_dir / "airteach_canvas.png"
        self.txt_dest.setText(str(default_name))
        dest_layout.addWidget(self.txt_dest, stretch=1)

        self.btn_browse = QPushButton("Chọn nơi lưu…")
        self.btn_browse.clicked.connect(self._on_browse)
        dest_layout.addWidget(self.btn_browse)
        layout.addWidget(dest_box)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_box = QHBoxLayout()
        btn_box.addStretch()

        btn_cancel = QPushButton("Hủy bỏ")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        self.btn_export = QPushButton("🚀 Bắt đầu xuất ảnh")
        self.btn_export.setObjectName("btn_export")
        self.btn_export.clicked.connect(self._on_start_export)
        btn_box.addWidget(self.btn_export)

        layout.addLayout(btn_box)

    @Slot(int)
    def _on_format_changed(self, idx: int) -> None:
        cur_p = Path(self.txt_dest.text())
        ext = ".png" if idx == 0 else ".jpg"
        self.chk_transparent.setEnabled(idx == 0)
        if not self.chk_transparent.isEnabled():
            self.chk_transparent.setChecked(False)
        if cur_p.is_file() or cur_p.suffix:
            new_p = cur_p.with_suffix(ext)
            self.txt_dest.setText(str(new_p))

    @Slot()
    def _on_browse(self) -> None:
        is_all = self.radio_all.isChecked()
        is_png = self.combo_format.currentIndex() == 0
        ext = ".png" if is_png else ".jpg"

        if is_all:
            folder = QFileDialog.getExistingDirectory(
                self, "Chọn thư mục lưu bộ ảnh bài giảng", str(Path.home() / "Pictures")
            )
            if folder:
                self.txt_dest.setText(folder)
        else:
            filter_str = "Ảnh PNG (*.png)" if is_png else "Ảnh JPEG (*.jpg *.jpeg)"
            file_p, _ = QFileDialog.getSaveFileName(
                self, "Lưu ảnh bài giảng", self.txt_dest.text(), filter_str
            )
            if file_p:
                self.txt_dest.setText(file_p)

    @Slot()
    def _on_start_export(self) -> None:
        dest_str = self.txt_dest.text().strip()
        if not dest_str:
            QMessageBox.warning(self, "Chưa chọn nơi lưu", "Vui lòng chọn đường dẫn tệp hoặc thư mục lưu ảnh.")
            return

        is_all = self.radio_all.isChecked()
        scale = float(self.combo_res.currentData())
        is_png = self.combo_format.currentIndex() == 0
        ext = ".png" if is_png else ".jpg"
        transparent = self.chk_transparent.isChecked()

        self.btn_export.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(10)

        try:
            if is_all:
                out_dir = Path(dest_str)
                if out_dir.suffix:
                    out_dir = out_dir.parent
                exported = CanvasExporter.export_all_pages(
                    main_window=self.main_window,
                    output_dir=out_dir,
                    prefix="bai_giang",
                    ext=ext,
                    scale=scale,
                    transparent_bg=transparent,
                )
                self.progress_bar.setValue(100)

                # Play acoustic feedback
                try:
                    from src.utils.audio_feedback import AudioFeedback
                    AudioFeedback.play(AudioFeedback.SoundType.SNAP)
                except Exception:
                    pass

                QMessageBox.information(
                    self,
                    "Xuất ảnh hoàn tất",
                    f"Đã xuất thành công {len(exported)} trang ảnh vào thư mục:\n{out_dir}",
                )
            else:
                out_file = Path(dest_str)
                if out_file.is_dir():
                    out_file = out_file / f"airteach_canvas{ext}"
                if not out_file.suffix:
                    out_file = out_file.with_suffix(ext)

                saved = CanvasExporter.export_scene_to_file(
                    scene=self.main_window._scene,
                    file_path=out_file,
                    scale=scale,
                    transparent_bg=transparent,
                )
                self.progress_bar.setValue(100)

                # Play acoustic feedback
                try:
                    from src.utils.audio_feedback import AudioFeedback
                    AudioFeedback.play(AudioFeedback.SoundType.SNAP)
                except Exception:
                    pass

                self.main_window._status.showMessage(f"Đã xuất ảnh bảng vẽ: {saved.name}", 4000)
                QMessageBox.information(
                    self,
                    "Xuất ảnh hoàn tất",
                    f"Đã xuất thành công ảnh bảng vẽ độ phân giải cao:\n{saved}",
                )

            self.accept()
        except Exception as exc:  # noqa: BLE001
            self.progress_bar.setVisible(False)
            self.btn_export.setEnabled(True)
            QMessageBox.critical(self, "Lỗi xuất ảnh", f"Đã xảy ra lỗi trong quá trình xuất ảnh:\n{exc}")
