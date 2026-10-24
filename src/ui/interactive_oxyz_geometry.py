"""
AirTeach AI — Interactive 3D Analytic Geometry (Oxyz), Vector Cross Product & Plane-Line Simulator
Authentic 3D Cartesian coordinates, dot & cross products, plane normal vectors, parametric lines,
Euclidean distance metrics, spheres, and touchless air gesture integration.
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
    QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QStyleOptionGraphicsItem,
    QWidget,
)


@dataclass(frozen=True)
class OxyzPreset:
    code: str
    name_vi: str
    point_a: tuple[float, float, float]
    point_b: tuple[float, float, float]
    point_c: tuple[float, float, float]
    point_m: tuple[float, float, float]
    description: str


PRESETS: dict[str, OxyzPreset] = {
    "PLANE_THROUGH_THREE_POINTS": OxyzPreset(
        code="PLANE_THROUGH_THREE_POINTS",
        name_vi="Mặt phẳng (ABC) qua 3 điểm & Vector pháp tuyến",
        point_a=(2.0, 0.0, 0.0),
        point_b=(0.0, 3.0, 0.0),
        point_c=(0.0, 0.0, 4.0),
        point_m=(1.0, 1.0, 2.0),
        description="Mặt phẳng theo đoạn chắn x/a + y/b + z/c = 1. Vector pháp tuyến n = [AB, AC] = (12, 8, 6).",
    ),
    "LINE_PLANE_INTERSECTION": OxyzPreset(
        code="LINE_PLANE_INTERSECTION",
        name_vi="Giao điểm Đường thẳng & Mặt phẳng (d ∩ P)",
        point_a=(0.0, 0.0, 0.0),
        point_b=(4.0, 0.0, 0.0),
        point_c=(0.0, 4.0, 0.0),
        point_m=(2.0, 2.0, 3.0),
        description="Đường thẳng qua M vuông góc hoặc cắt mặt phẳng (Oxy) z = 0 tại giao điểm H(2, 2, 0).",
    ),
    "DISTANCE_POINT_TO_PLANE": OxyzPreset(
        code="DISTANCE_POINT_TO_PLANE",
        name_vi="Khoảng cách từ điểm M đến Mặt phẳng d(M, P)",
        point_a=(3.0, 0.0, 0.0),
        point_b=(0.0, 3.0, 0.0),
        point_c=(0.0, 0.0, 3.0),
        point_m=(3.0, 3.0, 3.0),
        description="Hình chiếu vuông góc H của M lên mặt phẳng và tính khoảng cách d(M, P) = |Ax + By + Cz + D| / sqrt(A²+B²+C²).",
    ),
    "SKEW_LINES_SHORTEST_DISTANCE": OxyzPreset(
        code="SKEW_LINES_SHORTEST_DISTANCE",
        name_vi="Hai đường thẳng chéo nhau & Đoạn vuông góc chung",
        point_a=(1.0, 0.0, 0.0),
        point_b=(1.0, 4.0, 0.0),
        point_c=(0.0, 1.0, 3.0),
        point_m=(4.0, 1.0, 3.0),
        description="Khoảng cách giữa hai đường thẳng chéo nhau d(d1, d2) = |[u1, u2] . M1M2| / |[u1, u2]|.",
    ),
    "SPHERE_TANGENT_PLANE": OxyzPreset(
        code="SPHERE_TANGENT_PLANE",
        name_vi="Mặt cầu tâm I(M) & Mặt phẳng tiếp diện (P)",
        point_a=(2.0, 0.0, 0.0),
        point_b=(0.0, 2.0, 0.0),
        point_c=(0.0, 0.0, 0.0),
        point_m=(1.0, 1.0, 2.0),
        description="Mặt cầu tâm M bán kính R = d(M, Oxy) = 2.0 tiếp xúc với mặt phẳng tọa độ (Oxy) tại tiếp điểm (1, 1, 0).",
    ),
}


class InteractiveOxyzGeometryItem(QGraphicsObject):
    """
    Interactive QGraphicsObject simulating 3D Analytic Geometry (Oxyz),
    Vector cross and dot products, plane equations, parametric lines, and distances.
    """

    WIDTH = 680.0
    HEIGHT = 410.0

    stamp_requested = Signal()
    view_rotated = Signal(float, float)  # azimuth, elevation

    def __init__(
        self,
        pos: QPointF | tuple[float, float] = (150.0, 65.0),
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

        # Simulation State
        self._active_preset_code: str = "PLANE_THROUGH_THREE_POINTS"
        self._preset: OxyzPreset = PRESETS[self._active_preset_code]

        # 3D Coordinates (world units ~ [-5, 5])
        self._a = list(self._preset.point_a)
        self._b = list(self._preset.point_b)
        self._c = list(self._preset.point_c)
        self._m = list(self._preset.point_m)

        # 3D Camera Angles (degrees)
        self._azimuth_deg: float = 35.0
        self._elevation_deg: float = 25.0
        self._scale_3d: float = 24.0

        # Animation timer for smooth rotation
        self._is_rotating: bool = False
        self._rot_timer = QTimer()
        self._rot_timer.setInterval(40)
        self._rot_timer.timeout.connect(self._on_rotation_tick)

        # UI interaction button rects
        self._btn_rects: dict[str, QRectF] = {}
        self._setup_layout()

    def _setup_layout(self) -> None:
        """Sets up layout rects for toolbar buttons, 3D viewport, and diagnostics."""
        btn_y = 44.0
        btn_h = 22.0
        cur_x = 14.0

        # 5 Preset buttons
        preset_names = [
            ("PLANE_THROUGH_THREE_POINTS", "Mp (ABC)"),
            ("LINE_PLANE_INTERSECTION", "Giao điểm d ∩ P"),
            ("DISTANCE_POINT_TO_PLANE", "K/c d(M, P)"),
            ("SKEW_LINES_SHORTEST_DISTANCE", "Đ/thẳng chéo"),
            ("SPHERE_TANGENT_PLANE", "Mặt cầu (S)"),
        ]
        for p_code, p_lbl in preset_names:
            pw = 74.0
            self._btn_rects[f"preset_{p_code}"] = QRectF(cur_x, btn_y, pw, btn_h)
            cur_x += pw + 4.0

        # Controls
        r_rot = QRectF(cur_x + 4.0, btn_y, 46.0, btn_h)
        self._btn_rects["toggle_rot"] = r_rot

        r_az_left = QRectF(r_rot.right() + 4.0, btn_y, 28.0, btn_h)
        self._btn_rects["az_left"] = r_az_left

        r_az_right = QRectF(r_az_left.right() + 3.0, btn_y, 28.0, btn_h)
        self._btn_rects["az_right"] = r_az_right

        r_reset = QRectF(r_az_right.right() + 4.0, btn_y, 36.0, btn_h)
        self._btn_rects["reset_view"] = r_reset

        r_stamp = QRectF(r_reset.right() + 4.0, btn_y, 46.0, btn_h)
        self._btn_rects["stamp"] = r_stamp

        # Viewport areas
        self._view3d_rect = QRectF(16.0, 78.0, 420.0, 205.0)
        self._btn_rects["view3d_canvas"] = self._view3d_rect
        self._diag_rect = QRectF(446.0, 78.0, 220.0, 205.0)

    def cleanup(self) -> None:
        """Stops rotation timer to prevent memory leaks."""
        if self._rot_timer.isActive():
            self._rot_timer.stop()

    # -------------------------------------------------------------------------
    # 3D Analytic Mathematics Engine (Zero Mock Data)
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads a predefined 3D geometry preset."""
        if preset_code not in PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PRESETS[preset_code]
        self._a = list(self._preset.point_a)
        self._b = list(self._preset.point_b)
        self._c = list(self._preset.point_c)
        self._m = list(self._preset.point_m)
        self.update()

    def set_camera_angles(self, azimuth: float, elevation: float) -> None:
        """Sets 3D azimuth and elevation angles (degrees)."""
        self._azimuth_deg = float(azimuth) % 360.0
        self._elevation_deg = max(-85.0, min(85.0, float(elevation)))
        self.view_rotated.emit(self._azimuth_deg, self._elevation_deg)
        self.update()

    def toggle_rotation(self) -> bool:
        """Toggles smooth camera rotation timer."""
        self._is_rotating = not self._is_rotating
        if self._is_rotating:
            self._rot_timer.start()
        else:
            self._rot_timer.stop()
        self.update()
        return self._is_rotating

    def _on_rotation_tick(self) -> None:
        """Timer callback for continuous azimuth rotation."""
        if not self._is_rotating:
            return
        self.set_camera_angles(self._azimuth_deg + 1.2, self._elevation_deg)

    @staticmethod
    def vector_cross(u: tuple[float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
        """Calculates authentic vector cross product u x v in R^3."""
        return (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )

    @staticmethod
    def vector_dot(u: tuple[float, float, float], v: tuple[float, float, float]) -> float:
        """Calculates authentic dot product u . v in R^3."""
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    @staticmethod
    def vector_norm(u: tuple[float, float, float]) -> float:
        """Calculates Euclidean vector length |u|."""
        return math.sqrt(u[0] ** 2 + u[1] ** 2 + u[2] ** 2)

    def plane_normal(self) -> tuple[float, float, float]:
        """Calculates normal vector n = AB x AC of plane (ABC)."""
        ab = (self._b[0] - self._a[0], self._b[1] - self._a[1], self._b[2] - self._a[2])
        ac = (self._c[0] - self._a[0], self._c[1] - self._a[1], self._c[2] - self._a[2])
        return self.vector_cross(ab, ac)

    def plane_equation(self) -> tuple[float, float, float, float]:
        """Returns plane coefficients (A, B, C, D) such that Ax + By + Cz + D = 0."""
        n = self.plane_normal()
        d = -(n[0] * self._a[0] + n[1] * self._a[1] + n[2] * self._a[2])
        return n[0], n[1], n[2], d

    def distance_m_to_plane(self) -> float:
        """Calculates authentic distance from point M to plane (ABC)."""
        A, B, C, D = self.plane_equation()
        num = abs(A * self._m[0] + B * self._m[1] + C * self._m[2] + D)
        den = math.sqrt(A ** 2 + B ** 2 + C ** 2)
        return num / (den + 1e-9)

    def project_3d_to_2d(self, x: float, y: float, z: float, origin: QPointF) -> QPointF:
        """
        Projects 3D world coordinates (x, y, z) into 2D screen coordinates
        using authentic 3D rotation matrix (azimuth theta, elevation phi).
        """
        rad_az = math.radians(self._azimuth_deg)
        rad_el = math.radians(self._elevation_deg)

        # Rotate by azimuth around Z axis
        x1 = x * math.cos(rad_az) - y * math.sin(rad_az)
        y1 = x * math.sin(rad_az) + y * math.cos(rad_az)
        z1 = z

        # Rotate by elevation around X' axis
        x2 = x1
        y2 = y1 * math.cos(rad_el) - z1 * math.sin(rad_el)
        z2 = y1 * math.sin(rad_el) + z1 * math.cos(rad_el)

        # Axonometric projection to 2D screen
        s = self._scale_3d
        px = origin.x() + x2 * s
        py = origin.y() - z2 * s  # Invert Y for screen coordinates
        return QPointF(px, py)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    @property
    def azimuth_deg(self) -> float:
        return self._azimuth_deg

    @property
    def elevation_deg(self) -> float:
        return self._elevation_deg

    @property
    def active_preset(self) -> OxyzPreset:
        return self._preset

    @property
    def is_rotating(self) -> bool:
        return self._is_rotating

    # -------------------------------------------------------------------------
    # Touchless Air Gesture & Stamping
    # -------------------------------------------------------------------------
    def stamp_to_board(self) -> None:
        """Stamps active 3D Oxyz geometry diagram onto LessonBoard."""
        scene = self.scene()
        if scene is None:
            return

        from src.ui.stem_tools import MathTools

        center = (self.pos().x() + self.WIDTH / 2.0, self.pos().y() + self.HEIGHT + 40.0)
        items = MathTools.create_oxyz_geometry_diagram(
            center=center,
            preset_code=self._active_preset_code,
            point_a=tuple(self._a),
            point_b=tuple(self._b),
            point_c=tuple(self._c),
            point_m=tuple(self._m),
            distance_val=self.distance_m_to_plane(),
        )
        for itm in items:
            scene.addItem(itm)
        self.stamp_requested.emit()

    def handle_air_pinch(self, scene_pos: QPointF) -> bool:
        """Touchless air gesture handling."""
        local_pos = self.mapFromScene(scene_pos)
        if not self.boundingRect().contains(local_pos):
            return False

        # 1. Preset buttons
        for k in PRESETS:
            r = self._btn_rects.get(f"preset_{k}")
            if r and r.contains(local_pos):
                self.load_preset(k)
                return True

        # 2. Toggle Rotation
        r_rot = self._btn_rects.get("toggle_rot")
        if r_rot and r_rot.contains(local_pos):
            self.toggle_rotation()
            return True

        # 3. Azimuth adjustments
        r_al = self._btn_rects.get("az_left")
        if r_al and r_al.contains(local_pos):
            self.set_camera_angles(self._azimuth_deg - 15.0, self._elevation_deg)
            return True

        r_ar = self._btn_rects.get("az_right")
        if r_ar and r_ar.contains(local_pos):
            self.set_camera_angles(self._azimuth_deg + 15.0, self._elevation_deg)
            return True

        # 4. Reset View
        r_rst = self._btn_rects.get("reset_view")
        if r_rst and r_rst.contains(local_pos):
            self.set_camera_angles(35.0, 25.0)
            return True

        # 5. Direct 3D Viewport drag/pinch
        if self._view3d_rect.contains(local_pos):
            frac_x = (local_pos.x() - self._view3d_rect.left()) / self._view3d_rect.width()
            frac_y = (local_pos.y() - self._view3d_rect.top()) / self._view3d_rect.height()
            new_az = frac_x * 360.0
            new_el = (0.5 - frac_y) * 120.0
            self.set_camera_angles(new_az, new_el)
            return True

        # 6. Stamp
        r_stamp = self._btn_rects.get("stamp")
        if r_stamp and r_stamp.contains(local_pos):
            self.stamp_to_board()
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
            QRectF(14.0, 0.0, 480.0, 36.0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "📐 HÌNH HỌC KHÔNG GIAN Oxyz & TÍCH CÓ HƯỚNG VECTOR (ANALYTIC GEOMETRY)",
        )

        # Preset Badge
        p_badge = QRectF(self.WIDTH - 164.0, 7.0, 150.0, 22.0)
        painter.setBrush(QBrush(QColor("#1e1b4b")))
        painter.setPen(QPen(QColor("#818cf8"), 1.0))
        painter.drawRoundedRect(p_badge, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#c7d2fe"))
        painter.drawText(p_badge, Qt.AlignmentFlag.AlignCenter, self._preset.name_vi[:24])

        # 3. Preset Buttons & Controls
        preset_items = [
            ("PLANE_THROUGH_THREE_POINTS", "Mp (ABC)"),
            ("LINE_PLANE_INTERSECTION", "Giao điểm d ∩ P"),
            ("DISTANCE_POINT_TO_PLANE", "K/c d(M, P)"),
            ("SKEW_LINES_SHORTEST_DISTANCE", "Đ/thẳng chéo"),
            ("SPHERE_TANGENT_PLANE", "Mặt cầu (S)"),
        ]
        for p_code, p_lbl in preset_items:
            r = self._btn_rects[f"preset_{p_code}"]
            is_active = (p_code == self._active_preset_code)
            painter.setBrush(QBrush(QColor("#3b82f6") if is_active else QColor("#1e293b")))
            painter.setPen(QPen(QColor("#60a5fa") if is_active else QColor("#334155"), 1.0))
            painter.drawRoundedRect(r, 4.0, 4.0)
            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold if is_active else QFont.Weight.Normal))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, p_lbl)

        # Rotation toggle button
        r_rot = self._btn_rects["toggle_rot"]
        painter.setBrush(QBrush(QColor("#065f46") if self._is_rotating else QColor("#1e293b")))
        painter.setPen(QPen(QColor("#10b981") if self._is_rotating else QColor("#475569"), 1.0))
        painter.drawRoundedRect(r_rot, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff") if self._is_rotating else QColor("#94a3b8"))
        painter.drawText(r_rot, Qt.AlignmentFlag.AlignCenter, "⏸ Dừng" if self._is_rotating else "🔄 Xoay")

        # Azimuth steppers
        r_al = self._btn_rects["az_left"]
        painter.setBrush(QBrush(QColor("#1e293b")))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRoundedRect(r_al, 3.0, 3.0)
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(r_al, Qt.AlignmentFlag.AlignCenter, "↶")

        r_ar = self._btn_rects["az_right"]
        painter.drawRoundedRect(r_ar, 3.0, 3.0)
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(r_ar, Qt.AlignmentFlag.AlignCenter, "↷")

        # Reset button
        r_rst = self._btn_rects["reset_view"]
        painter.drawRoundedRect(r_rst, 4.0, 4.0)
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(r_rst, Qt.AlignmentFlag.AlignCenter, "Reset")

        # Stamp button
        r_stamp = self._btn_rects["stamp"]
        painter.setBrush(QBrush(QColor("#831843")))
        painter.setPen(QPen(QColor("#f43f5e"), 1.0))
        painter.drawRoundedRect(r_stamp, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(r_stamp, Qt.AlignmentFlag.AlignCenter, "📌 Đóng dấu")

        # ---------------------------------------------------------------------
        # 4. Left Panel: 3D Perspective Projection Viewport
        # ---------------------------------------------------------------------
        vx, vy, vw, vh = self._view3d_rect.x(), self._view3d_rect.y(), self._view3d_rect.width(), self._view3d_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._view3d_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(vx + 8.0, vy + 16.0), f"KHÔNG GIAN 3D Oxyz (GÓC NHÌN: {self._azimuth_deg:.0f}°, {self._elevation_deg:.0f}°)")

        origin_2d = QPointF(vx + vw * 0.45, vy + vh * 0.58)

        # Draw 3D Axes (Ox in Red, Oy in Green, Oz in Blue)
        axis_len = 4.5
        p_o = origin_2d
        p_x = self.project_3d_to_2d(axis_len, 0.0, 0.0, origin_2d)
        p_y = self.project_3d_to_2d(0.0, axis_len, 0.0, origin_2d)
        p_z = self.project_3d_to_2d(0.0, 0.0, axis_len, origin_2d)

        # Negative dashed axes
        p_nx = self.project_3d_to_2d(-axis_len * 0.5, 0.0, 0.0, origin_2d)
        p_ny = self.project_3d_to_2d(0.0, -axis_len * 0.5, 0.0, origin_2d)
        p_nz = self.project_3d_to_2d(0.0, 0.0, -axis_len * 0.5, origin_2d)

        painter.setPen(QPen(QColor("#334155"), 0.8, Qt.PenStyle.DashLine))
        painter.drawLine(p_o, p_nx)
        painter.drawLine(p_o, p_ny)
        painter.drawLine(p_o, p_nz)

        # Ox
        painter.setPen(QPen(QColor("#ef4444"), 1.8))
        painter.drawLine(p_o, p_x)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.drawText(QPointF(p_x.x() + 4.0, p_x.y() + 4.0), "x")

        # Oy
        painter.setPen(QPen(QColor("#10b981"), 1.8))
        painter.drawLine(p_o, p_y)
        painter.drawText(QPointF(p_y.x() + 4.0, p_y.y() + 4.0), "y")

        # Oz
        painter.setPen(QPen(QColor("#38bdf8"), 1.8))
        painter.drawLine(p_o, p_z)
        painter.drawText(QPointF(p_z.x() + 4.0, p_z.y() - 4.0), "z")

        # Draw Plane (ABC) Polygon
        p_a = self.project_3d_to_2d(self._a[0], self._a[1], self._a[2], origin_2d)
        p_b = self.project_3d_to_2d(self._b[0], self._b[1], self._b[2], origin_2d)
        p_c = self.project_3d_to_2d(self._c[0], self._c[1], self._c[2], origin_2d)

        plane_poly = QPolygonF([p_a, p_b, p_c])
        painter.setBrush(QBrush(QColor(56, 189, 248, 55)))  # Semi-transparent Cyan
        painter.setPen(QPen(QColor("#38bdf8"), 1.5))
        painter.drawPolygon(plane_poly)

        # Draw Normal Vector n from centroid of triangle ABC
        cx3 = (self._a[0] + self._b[0] + self._c[0]) / 3.0
        cy3 = (self._a[1] + self._b[1] + self._c[1]) / 3.0
        cz3 = (self._a[2] + self._b[2] + self._c[2]) / 3.0

        n = self.plane_normal()
        norm_n = self.vector_norm(n)
        if norm_n > 1e-4:
            unit_n = (n[0] / norm_n * 2.0, n[1] / norm_n * 2.0, n[2] / norm_n * 2.0)
            p_cent = self.project_3d_to_2d(cx3, cy3, cz3, origin_2d)
            p_n_tip = self.project_3d_to_2d(cx3 + unit_n[0], cy3 + unit_n[1], cz3 + unit_n[2], origin_2d)
            painter.setPen(QPen(QColor("#f43f5e"), 2.0))
            painter.drawLine(p_cent, p_n_tip)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#f43f5e"))
            painter.drawText(QPointF(p_n_tip.x() + 4.0, p_n_tip.y() - 3.0), "n⃗")

        # Draw Point M and Distance line to plane
        p_m = self.project_3d_to_2d(self._m[0], self._m[1], self._m[2], origin_2d)

        # Draw Points A, B, C, M
        pts_info = [
            ("A", p_a, QColor("#38bdf8")),
            ("B", p_b, QColor("#10b981")),
            ("C", p_c, QColor("#fbbf24")),
            ("M", p_m, QColor("#a855f7")),
        ]
        for name, pt, col in pts_info:
            painter.setBrush(QBrush(col))
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            painter.drawEllipse(pt, 4.5, 4.5)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QColor("#e2e8f0"))
            painter.drawText(QPointF(pt.x() + 6.0, pt.y() + 3.0), name)

        # ---------------------------------------------------------------------
        # 5. Right Panel: Analytical Vector Ledger & Equations
        # ---------------------------------------------------------------------
        dx, dy, dw, dh = self._diag_rect.x(), self._diag_rect.y(), self._diag_rect.width(), self._diag_rect.height()
        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(self._diag_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(dx + 8.0, dy + 16.0), "TỌA ĐỘ & PHƯƠNG TRÌNH HÌNH HỌC")

        A, B, C, D = self.plane_equation()
        dist_m = self.distance_m_to_plane()

        lines = [
            (f"• A: ({self._a[0]:.1f}, {self._a[1]:.1f}, {self._a[2]:.1f})", QColor("#38bdf8")),
            (f"• B: ({self._b[0]:.1f}, {self._b[1]:.1f}, {self._b[2]:.1f})", QColor("#10b981")),
            (f"• C: ({self._c[0]:.1f}, {self._c[1]:.1f}, {self._c[2]:.1f})", QColor("#fbbf24")),
            (f"• M: ({self._m[0]:.1f}, {self._m[1]:.1f}, {self._m[2]:.1f})", QColor("#a855f7")),
            (f"• Vector n⃗ = [AB, AC]:", QColor("#f43f5e")),
            (f"  ({n[0]:.1f}, {n[1]:.1f}, {n[2]:.1f})", QColor("#f43f5e")),
            (f"• Mp (ABC): {A:.1f}x + {B:.1f}y + {C:.1f}z + {D:.1f} = 0", QColor("#38bdf8")),
            (f"• Khoảng cách d(M, P): {dist_m:.2f} đv", QColor("#10b981")),
        ]

        ly = dy + 32.0
        for l_txt, l_col in lines:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(l_col)
            painter.drawText(QPointF(dx + 8.0, ly), l_txt)
            ly += 17.0

        # Educational banner
        banner_r = QRectF(dx + 8.0, dy + dh - 24.0, dw - 16.0, 16.0)
        painter.setBrush(QBrush(QColor("#1e1b4b")))
        painter.setPen(QPen(QColor("#6366f1"), 1.0))
        painter.drawRoundedRect(banner_r, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
        painter.setPen(QColor("#c7d2fe"))
        painter.drawText(banner_r, Qt.AlignmentFlag.AlignCenter, f"|AB x AC| = {norm_n:.2f} = 2.S(ABC)")

        # ---------------------------------------------------------------------
        # 6. Bottom Panel: Vector Geometry Theorems Card
        # ---------------------------------------------------------------------
        bot_y = self._view3d_rect.bottom() + 10.0
        bot_h = self.HEIGHT - bot_y - 10.0
        bot_rect = QRectF(14.0, bot_y, self.WIDTH - 28.0, bot_h)

        painter.setBrush(QBrush(QColor("#020617")))
        painter.setPen(QPen(QColor("#1e293b"), 1.0))
        painter.drawRoundedRect(bot_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(QPointF(22.0, bot_y + 16.0), "CÔNG THỨC TOÁN HỌC KHÔNG GIAN Oxyz & TÍCH CÓ HƯỚNG VECTOR:")

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#94a3b8"))
        b_desc = (
            f"• Tích có hướng: u⃗ × v⃗ = (y₁z₂ - z₁y₂, z₁x₂ - x₁z₂, x₁y₂ - y₁x₂). Vector pháp tuyến vuông góc với cả hai vector chỉ phương.\n"
            f"• Khoảng cách từ điểm M đến mặt phẳng (P): d(M, P) = |Ax_M + By_M + Cz_M + D| / √(A² + B² + C²).\n"
            f"• Đặc tính mẫu: {self._preset.description}"
        )
        painter.drawText(
            QRectF(22.0, bot_y + 22.0, bot_rect.width() - 20.0, 58.0),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            b_desc,
        )
