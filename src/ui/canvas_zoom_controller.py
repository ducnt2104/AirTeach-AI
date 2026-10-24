"""
AirTeach AI - Whiteboard Canvas Zoom & Pan Controller
=====================================================
Provides interactive canvas magnification (0.25x - 4.0x), smooth centered zooming,
touchless bimanual pinch/pan, spacebar/middle-click drag, and a floating acrylic
HUD widget with quick fit-to-view presets.
"""
from __future__ import annotations

import logging
from typing import Optional

from PySide6.QtCore import QEvent, QObject, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QCursor, QMouseEvent, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

logger = logging.getLogger(__name__)

# Zoom constraints
MIN_ZOOM_FACTOR: float = 0.25
MAX_ZOOM_FACTOR: float = 4.0
DEFAULT_ZOOM_FACTOR: float = 1.0
ZOOM_STEP_IN: float = 1.20
ZOOM_STEP_OUT: float = 0.8333


class CanvasZoomWidget(QWidget):
    """
    Floating acrylic HUD widget displaying current zoom percentage,
    with quick buttons for zoom in, zoom out, fit-to-page reset, and pan mode toggle.
    """

    zoom_in_requested = Signal()
    zoom_out_requested = Signal()
    zoom_reset_requested = Signal()
    pan_toggled = Signal(bool)
    minimap_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("CanvasZoomWidget")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._pan_active: bool = False
        self._minimap_active: bool = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        # Style sheet: Frosted acrylic floating pill
        self.setStyleSheet("""
            QWidget#CanvasZoomWidget {
                background: rgba(26, 27, 38, 0.88);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 16px;
            }
            QPushButton {
                background: rgba(255, 255, 255, 0.08);
                color: #e0e6ed;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 12px;
                font-size: 13px;
                font-weight: bold;
                min-width: 26px;
                min-height: 26px;
                max-width: 26px;
                max-height: 26px;
                padding: 0px;
            }
            QPushButton:hover {
                background: rgba(255, 255, 255, 0.18);
                border-color: rgba(255, 255, 255, 0.3);
                color: #ffffff;
            }
            QPushButton:pressed {
                background: rgba(79, 70, 229, 0.6);
            }
            QPushButton#btn_reset {
                min-width: 52px;
                max-width: 64px;
                font-size: 11px;
                font-weight: 600;
                color: #c7d2fe;
            }
            QPushButton#btn_reset:hover {
                background: rgba(99, 102, 241, 0.35);
                color: #ffffff;
            }
            QPushButton#btn_pan:checked {
                background: rgba(16, 185, 129, 0.45);
                border-color: #10b981;
                color: #ffffff;
            }
            QPushButton#btn_map:checked {
                background: rgba(59, 130, 246, 0.45);
                border-color: #3b82f6;
                color: #ffffff;
            }
        """)

        # 1. Pan button
        self.btn_pan = QPushButton("🖐️", self)
        self.btn_pan.setObjectName("btn_pan")
        self.btn_pan.setCheckable(True)
        self.btn_pan.setToolTip("Chế độ cuộn/kéo bảng (Pan Mode - Phím H hoặc Space)")
        self.btn_pan.toggled.connect(self._on_pan_toggled)
        layout.addWidget(self.btn_pan)

        # 2. Mini-map toggle button
        self.btn_map = QPushButton("🗺️", self)
        self.btn_map.setObjectName("btn_map")
        self.btn_map.setCheckable(True)
        self.btn_map.setToolTip("Bản đồ thu nhỏ toàn cảnh (Mini-Map - Ctrl+Shift+V)")
        self.btn_map.toggled.connect(self._on_map_toggled)
        layout.addWidget(self.btn_map)

        # 3. Zoom out button
        self.btn_zoom_out = QPushButton("−", self)
        self.btn_zoom_out.setToolTip("Thu nhỏ (Ctrl+-)")
        self.btn_zoom_out.clicked.connect(self.zoom_out_requested.emit)
        layout.addWidget(self.btn_zoom_out)

        # 3. Zoom reset / percentage label button
        self.btn_reset = QPushButton("100%", self)
        self.btn_reset.setObjectName("btn_reset")
        self.btn_reset.setToolTip("Vừa khung hình 100% (Ctrl+0)")
        self.btn_reset.clicked.connect(self.zoom_reset_requested.emit)
        layout.addWidget(self.btn_reset)

        # 4. Zoom in button
        self.btn_zoom_in = QPushButton("+", self)
        self.btn_zoom_in.setToolTip("Phóng to (Ctrl++)")
        self.btn_zoom_in.clicked.connect(self.zoom_in_requested.emit)
        layout.addWidget(self.btn_zoom_in)

    def _on_pan_toggled(self, checked: bool) -> None:
        self._pan_active = checked
        self.pan_toggled.emit(checked)

    def _on_map_toggled(self, checked: bool) -> None:
        self._minimap_active = checked
        self.minimap_toggled.emit(checked)

    def set_zoom_factor(self, factor: float) -> None:
        """Update zoom percentage text."""
        pct = int(round(factor * 100))
        self.btn_reset.setText(f"{pct}%")

    def set_pan_active(self, active: bool) -> None:
        """Update pan button toggle state."""
        self._pan_active = bool(active)
        if self.btn_pan.isChecked() != self._pan_active:
            self.btn_pan.setChecked(self._pan_active)

    def is_pan_active(self) -> bool:
        return self._pan_active

    def set_minimap_active(self, active: bool) -> None:
        """Update minimap button toggle state."""
        self._minimap_active = bool(active)
        if self.btn_map.isChecked() != self._minimap_active:
            self.btn_map.setChecked(self._minimap_active)

    def is_minimap_active(self) -> bool:
        return self._minimap_active


class CanvasZoomController(QObject):
    """
    Manages interactive zoom and pan transformations for a QGraphicsView displaying LessonBoard.
    Supports mouse wheel zooming, middle-click dragging, touchless gesture zooming,
    and maintains scene coordinate fidelity.
    """

    zoom_changed = Signal(float)
    pan_mode_changed = Signal(bool)

    def __init__(self, view: QGraphicsView, scene: QGraphicsScene, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._view = view
        self._scene = scene

        self._zoom_factor: float = DEFAULT_ZOOM_FACTOR
        self._pan_mode: bool = False
        self._is_dragging: bool = False
        self._drag_start_pos: QPoint = QPoint()

        # Create companion floating HUD widget
        self._widget = CanvasZoomWidget(parent=self._view)
        self._widget.zoom_in_requested.connect(self.zoom_in)
        self._widget.zoom_out_requested.connect(self.zoom_out)
        self._widget.zoom_reset_requested.connect(self.reset_zoom)
        self._widget.pan_toggled.connect(self.set_pan_mode)

        # Create companion bird's eye view mini-map widget
        from src.ui.canvas_minimap import CanvasMiniMapWidget
        self._minimap = CanvasMiniMapWidget(view=self._view, scene=self._scene, parent=self._view)
        self._minimap.setVisible(False)
        self._minimap_visible: bool = False
        self._widget.minimap_toggled.connect(self.set_minimap_visible)
        self._minimap.visibility_toggled.connect(self._on_minimap_visibility_changed)

        # Connect viewport scrollbars to update mini-map
        h_bar = self._view.horizontalScrollBar()
        v_bar = self._view.verticalScrollBar()
        if h_bar:
            h_bar.valueChanged.connect(self._on_viewport_scroll)
        if v_bar:
            v_bar.valueChanged.connect(self._on_viewport_scroll)

        # Install event filter on view viewport to capture wheel and mouse drag
        self._view.viewport().installEventFilter(self)

        # Configure view scrollbar behavior
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)

        # Reposition widget on view resize
        self._update_widget_geometry()

    def get_widget(self) -> CanvasZoomWidget:
        """Return the companion floating HUD widget."""
        return self._widget

    def zoom_factor(self) -> float:
        """Return current magnification scale (1.0 = 100%)."""
        return self._zoom_factor

    def is_pan_mode(self) -> bool:
        """Return True if panning mode is currently active."""
        return self._pan_mode

    def set_pan_mode(self, enabled: bool) -> None:
        """Toggle pan drag mode."""
        self._pan_mode = bool(enabled)
        self._widget.set_pan_active(self._pan_mode)
        if self._pan_mode:
            self._view.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self._view.viewport().unsetCursor()
        self.pan_mode_changed.emit(self._pan_mode)

    def toggle_pan_mode(self) -> bool:
        """Toggle pan mode; returns new state."""
        self.set_pan_mode(not self._pan_mode)
        return self._pan_mode

    def zoom_in(self, step: float = ZOOM_STEP_IN) -> float:
        """Increase magnification by step factor."""
        new_zoom = min(MAX_ZOOM_FACTOR, self._zoom_factor * step)
        return self.set_zoom_factor(new_zoom)

    def zoom_out(self, step: float = ZOOM_STEP_OUT) -> float:
        """Decrease magnification by step factor."""
        new_zoom = max(MIN_ZOOM_FACTOR, self._zoom_factor * step)
        return self.set_zoom_factor(new_zoom)

    def set_zoom_factor(self, factor: float) -> float:
        """Set absolute zoom factor, clamped to [MIN_ZOOM_FACTOR, MAX_ZOOM_FACTOR]."""
        clamped = max(MIN_ZOOM_FACTOR, min(MAX_ZOOM_FACTOR, float(factor)))
        if abs(clamped - self._zoom_factor) < 0.001:
            return self._zoom_factor

        scale_change = clamped / self._zoom_factor
        self._zoom_factor = clamped
        self._view.scale(scale_change, scale_change)
        self._widget.set_zoom_factor(self._zoom_factor)
        self.zoom_changed.emit(self._zoom_factor)

        if hasattr(self, "_minimap") and self._minimap:
            self._minimap.update_viewport()

        # Trigger acoustic click feedback
        try:
            from src.utils.audio_feedback import AudioFeedbackManager
            AudioFeedbackManager.instance().play_click()
        except Exception:
            pass

        return self._zoom_factor

    def reset_zoom(self) -> None:
        """Reset view transformation back to fit the entire scene (100%)."""
        self._view.resetTransform()
        self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = DEFAULT_ZOOM_FACTOR
        self._widget.set_zoom_factor(self._zoom_factor)
        self.zoom_changed.emit(self._zoom_factor)

        if hasattr(self, "_minimap") and self._minimap:
            self._minimap.update_viewport()

        # Trigger acoustic click feedback
        try:
            from src.utils.audio_feedback import AudioFeedbackManager
            AudioFeedbackManager.instance().play_click()
        except Exception:
            pass

    def pan_by(self, dx: float, dy: float) -> None:
        """Pan view scrollbars by offset delta (in viewport pixels)."""
        h_bar = self._view.horizontalScrollBar()
        v_bar = self._view.verticalScrollBar()
        if h_bar:
            h_bar.setValue(int(h_bar.value() - dx))
        if v_bar:
            v_bar.setValue(int(v_bar.value() - dy))
        if hasattr(self, "_minimap") and self._minimap and self._minimap.isVisible():
            self._minimap.update_viewport()

    def update_position(self) -> None:
        """Update floating HUD position within parent view (bottom-right anchor)."""
        self._update_widget_geometry()
        if hasattr(self, "_minimap") and self._minimap:
            self._minimap.update_position()

    def _update_widget_geometry(self) -> None:
        if not self._widget or not self._view:
            return
        vw = self._view.viewport().width()
        vh = self._view.viewport().height()
        ww = 205
        wh = 38
        margin_x = 24
        margin_y = 20
        self._widget.setGeometry(vw - ww - margin_x, vh - wh - margin_y, ww, wh)
        self._widget.raise_()
        if hasattr(self, "_minimap") and self._minimap:
            self._minimap.update_position()
            self._minimap.raise_()

    def get_minimap(self):
        """Return the companion bird's eye view mini-map widget."""
        return getattr(self, "_minimap", None)

    def set_minimap_visible(self, visible: bool) -> None:
        """Set mini-map visibility and synchronize button state."""
        self._minimap_visible = bool(visible)
        if hasattr(self, "_minimap") and self._minimap:
            self._minimap.setVisible(visible)
            self._widget.set_minimap_active(visible)
            if visible:
                self._minimap.update_position()
                self._minimap.update_viewport()
                self._minimap.raise_()

    def toggle_minimap(self) -> bool:
        """Toggle mini-map visibility; returns new visibility state."""
        new_vis = not self._minimap_visible
        self.set_minimap_visible(new_vis)
        return new_vis

    def is_minimap_visible(self) -> bool:
        """Return True if mini-map is currently displayed."""
        return self._minimap_visible

    def _on_minimap_visibility_changed(self, visible: bool) -> None:
        self._minimap_visible = bool(visible)
        self._widget.set_minimap_active(visible)

    def _on_viewport_scroll(self, _val: int) -> None:
        if hasattr(self, "_minimap") and self._minimap and self._minimap.isVisible():
            self._minimap.update_viewport()

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Intercept viewport mouse wheel and drag events for zoom and pan."""
        try:
            view = getattr(self, "_view", None)
            if not view or watched != view.viewport():
                return super().eventFilter(watched, event)
        except (RuntimeError, AttributeError):
            return False

        try:
            etype = event.type()

            # 1. Mouse wheel zoom with Ctrl key
            if etype == QEvent.Type.Wheel:
                wheel_event: QWheelEvent = event  # type: ignore[assignment]
                if wheel_event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    num_degrees = wheel_event.angleDelta().y() / 8.0
                    num_steps = num_degrees / 15.0
                    if num_steps > 0:
                        self.zoom_in(step=1.0 + 0.12 * num_steps)
                    elif num_steps < 0:
                        self.zoom_out(step=1.0 / (1.0 - 0.12 * num_steps))
                    return True

            # 2. Mouse Press for panning
            elif etype == QEvent.Type.MouseButtonPress:
                mouse_event: QMouseEvent = event  # type: ignore[assignment]
                # Pan with middle button OR left button in pan mode
                if (mouse_event.button() == Qt.MouseButton.MiddleButton or
                        (self._pan_mode and mouse_event.button() == Qt.MouseButton.LeftButton)):
                    self._is_dragging = True
                    self._drag_start_pos = mouse_event.pos()
                    self._view.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True

            # 3. Mouse Move for panning
            elif etype == QEvent.Type.MouseMove:
                if self._is_dragging:
                    mouse_event: QMouseEvent = event  # type: ignore[assignment]
                    delta = mouse_event.pos() - self._drag_start_pos
                    self._drag_start_pos = mouse_event.pos()
                    self.pan_by(delta.x(), delta.y())
                    return True

            # 4. Mouse Release
            elif etype == QEvent.Type.MouseButtonRelease:
                if self._is_dragging:
                    mouse_event: QMouseEvent = event  # type: ignore[assignment]
                    if (mouse_event.button() == Qt.MouseButton.MiddleButton or
                            (self._pan_mode and mouse_event.button() == Qt.MouseButton.LeftButton)):
                        self._is_dragging = False
                        if self._pan_mode:
                            self._view.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
                        else:
                            self._view.viewport().unsetCursor()
                        return True

            # 5. Viewport resize: keep floating HUD anchored
            elif etype == QEvent.Type.Resize:
                self._update_widget_geometry()

        except RuntimeError:
            return False

        return super().eventFilter(watched, event)
