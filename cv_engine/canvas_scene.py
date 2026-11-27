#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Object-Based Canvas Scene Graph (`cv_engine/canvas_scene.py`)
for AirTeach AI (Phase 2 Upgrade).

Every element placed on the Smartboard — Freehand Stroke, Geometric Shape,
Text Box, Inserted Image, and Sticky Note — is an independent `CanvasObject`
with:
  - `id`, `obj_type` ("stroke" | "shape" | "text" | "image" | "sticky")
  - `x, y, w, h` (bounding box in board coordinates)
  - `rotation` (degrees around bbox center)
  - `z_index`, `layer`
Supporting:
  - Single-click selection & marquee multi-selection (Rust-accelerated hit-test)
  - Free drag-move (single or multi-object group)
  - 8-handle resize (nw, n, ne, e, se, s, sw, w) & top rotate handle
  - Duplicate (Ctrl+D), Delete (Del), Bring-to-Front / Send-to-Back
  - Unified per-slide Undo/Redo (Ctrl+Z / Ctrl+Y)
  - Fit-to-Content bounds calculation
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from cv_engine.rust_bridge import (
    hit_test_objects_fast,
    make_spatial_index,
    marquee_select_objects_fast,
)

Point2D = Tuple[float, float]

_SPATIAL_FIELDS = frozenset({"x", "y", "w", "h", "rotation", "z_index"})

HANDLE_NAMES: Tuple[str, ...] = ("nw", "n", "ne", "e", "se", "s", "sw", "w")

STICKY_PALETTE: Tuple[Tuple[str, str, str], ...] = (
    ("#FEF08A", "#CA8A04", "#1E293B"),  # Classic Yellow
    ("#BBF7D0", "#16A34A", "#0F172A"),  # Mint Green
    ("#BFDBFE", "#2563EB", "#0F172A"),  # Sky Blue
    ("#FBCFE8", "#DB2777", "#0F172A"),  # Rose Pink
)


def new_object_id(prefix: str = "obj") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def compute_points_bbox(
    pts: Sequence[Point2D],
    min_size: float = 16.0,
) -> Tuple[float, float, float, float]:
    if not pts:
        return (0.0, 0.0, min_size, min_size)
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x
    h = max_y - min_y
    if w < min_size:
        pad = 0.5 * (min_size - w)
        min_x -= pad
        w = min_size
    if h < min_size:
        pad = 0.5 * (min_size - h)
        min_y -= pad
        h = min_size
    return (min_x, min_y, w, h)


def rotate_point(px: float, py: float, cx: float, cy: float, angle_deg: float) -> Point2D:
    if abs(angle_deg) < 1e-5:
        return (px, py)
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = px - cx, py - cy
    return (cx + dx * c - dy * s, cy + dx * s + dy * c)


@dataclass
class CanvasObject:
    """Unified independent scene-graph object on AirTeach SmartboardCanvas."""

    id: str
    obj_type: str  # "stroke" | "shape" | "text" | "image" | "sticky"
    x: float
    y: float
    w: float
    h: float
    rotation: float = 0.0
    z_index: int = 0
    layer: int = 0
    locked: bool = False

    # Stroke / Shape attributes
    points: List[Point2D] = field(default_factory=list)
    raw_points: List[Point2D] = field(default_factory=list)
    color: str = "#00F5D4"
    fill_color: str = ""
    thickness: float = 4.0
    is_highlighter: bool = False
    is_beautified: bool = False
    shape_type: str = "FREEHAND"
    shape_label_vi: str = ""
    confidence: float = 0.0

    # Text / Sticky / Math Annotation attributes
    text: str = ""
    raw_text: str = ""
    font_size: int = 15
    bg_color: str = ""
    border_color: str = ""
    reveal_until_ms: float = 0.0

    # Image / Rendered Math Pixmap (kept as opaque Qt object to avoid headless import coupling)
    pixmap: Any = None
    name: str = ""
    source_path: str = ""
    is_ai: bool = False
    _owner_scene: Any = field(default=None, init=False, repr=False, compare=False)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name in _SPATIAL_FIELDS:
            owner = self.__dict__.get("_owner_scene")
            if owner is not None:
                owner._spatial_dirty = True

    @property
    def center(self) -> Point2D:
        return (self.x + 0.5 * self.w, self.y + 0.5 * self.h)

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def clone(self, *, new_id: bool = False, dx: float = 0.0, dy: float = 0.0, z_index: Optional[int] = None) -> "CanvasObject":
        pts_copy = [(px + dx, py + dy) for (px, py) in self.points]
        raw_copy = [(px + dx, py + dy) for (px, py) in self.raw_points]
        return CanvasObject(
            id=new_object_id(self.obj_type[:3]) if new_id else self.id,
            obj_type=self.obj_type,
            x=self.x + dx,
            y=self.y + dy,
            w=self.w,
            h=self.h,
            rotation=self.rotation,
            z_index=self.z_index if z_index is None else int(z_index),
            layer=self.layer,
            locked=self.locked,
            points=pts_copy,
            raw_points=raw_copy,
            color=self.color,
            fill_color=self.fill_color,
            thickness=self.thickness,
            is_highlighter=self.is_highlighter,
            is_beautified=self.is_beautified,
            shape_type=self.shape_type,
            shape_label_vi=self.shape_label_vi,
            confidence=self.confidence,
            text=self.text,
            raw_text=self.raw_text,
            font_size=self.font_size,
            bg_color=self.bg_color,
            border_color=self.border_color,
            reveal_until_ms=0.0,
            pixmap=self.pixmap,
            name=self.name,
            source_path=self.source_path,
            is_ai=self.is_ai,
        )

    def translate(self, dx: float, dy: float) -> None:
        if self.locked:
            return
        self.x += float(dx)
        self.y += float(dy)
        if self.points:
            self.points = [(px + dx, py + dy) for (px, py) in self.points]
        if self.raw_points:
            self.raw_points = [(px + dx, py + dy) for (px, py) in self.raw_points]

    def apply_bbox_transform(
        self,
        orig: "CanvasObject",
        new_x: float,
        new_y: float,
        new_w: float,
        new_h: float,
    ) -> None:
        """Apply a new bounding box relative to `orig` cleanly scaling points & text."""
        if self.locked:
            return
        nw = max(20.0, float(new_w))
        nh = max(20.0, float(new_h))
        sx = nw / max(1.0, orig.w)
        sy = nh / max(1.0, orig.h)
        self.x = float(new_x)
        self.y = float(new_y)
        self.w = nw
        self.h = nh
        if orig.points:
            self.points = [
                (self.x + (px - orig.x) * sx, self.y + (py - orig.y) * sy)
                for (px, py) in orig.points
            ]
        if orig.raw_points:
            self.raw_points = [
                (self.x + (px - orig.x) * sx, self.y + (py - orig.y) * sy)
                for (px, py) in orig.raw_points
            ]
        if self.obj_type in ("text", "sticky") and orig.h > 10.0:
            scale_avg = math.sqrt(max(0.25, sx * sy))
            self.font_size = max(10, min(64, int(round(orig.font_size * scale_avg))))

    def world_aabb(self) -> Tuple[float, float, float, float]:
        """Return axis-aligned world bounding box `(min_x, min_y, max_x, max_y)` including rotation."""
        if abs(self.rotation) < 1e-4:
            return (self.x, self.y, self.x + self.w, self.y + self.h)
        cx, cy = self.center
        corners = [
            rotate_point(self.x, self.y, cx, cy, self.rotation),
            rotate_point(self.x + self.w, self.y, cx, cy, self.rotation),
            rotate_point(self.x + self.w, self.y + self.h, cx, cy, self.rotation),
            rotate_point(self.x, self.y + self.h, cx, cy, self.rotation),
        ]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (min(xs), min(ys), max(xs), max(ys))

    def handle_positions_world(self, rotate_offset: float = 28.0) -> Dict[str, Point2D]:
        """Return world coordinates for 8 resize handles + top `rotate` handle."""
        cx, cy = self.center
        x0, y0, x1, y1 = self.x, self.y, self.x + self.w, self.y + self.h
        xm, ym = cx, cy
        local_handles = {
            "nw": (x0, y0),
            "n": (xm, y0),
            "ne": (x1, y0),
            "e": (x1, ym),
            "se": (x1, y1),
            "s": (xm, y1),
            "sw": (x0, y1),
            "w": (x0, ym),
            "rotate": (xm, y0 - rotate_offset),
        }
        return {k: rotate_point(px, py, cx, cy, self.rotation) for k, (px, py) in local_handles.items()}

    def to_hit_test_tuple(self) -> Tuple[str, str, float, float, float, float, float, int, List[Point2D]]:
        return (
            self.id,
            self.obj_type,
            self.x,
            self.y,
            self.w,
            self.h,
            self.rotation,
            self.z_index,
            self.points,
        )

    def to_marquee_tuple(self) -> Tuple[str, float, float, float, float, float]:
        return (self.id, self.x, self.y, self.w, self.h, self.rotation)


def make_shape_points(shape_kind: str, x: float, y: float, w: float, h: float) -> List[Point2D]:
    """Generate clean vector polyline for interactive Shape creation."""
    st = str(shape_kind or "rect").strip().lower()
    bw = max(12.0, float(w))
    bh = max(12.0, float(h))
    if st == "circle":
        r = 0.5 * min(bw, bh)
        cx = x + 0.5 * bw
        cy = y + 0.5 * bh
        return [
            (cx + r * math.cos(2.0 * math.pi * i / 64.0), cy + r * math.sin(2.0 * math.pi * i / 64.0))
            for i in range(65)
        ]
    if st == "ellipse":
        rx, ry = 0.5 * bw, 0.5 * bh
        cx, cy = x + rx, y + ry
        return [
            (cx + rx * math.cos(2.0 * math.pi * i / 64.0), cy + ry * math.sin(2.0 * math.pi * i / 64.0))
            for i in range(65)
        ]
    if st == "triangle":
        return [(x + 0.5 * bw, y), (x + bw, y + bh), (x, y + bh), (x + 0.5 * bw, y)]
    if st == "diamond":
        return [
            (x + 0.5 * bw, y),
            (x + bw, y + 0.5 * bh),
            (x + 0.5 * bw, y + bh),
            (x, y + 0.5 * bh),
            (x + 0.5 * bw, y),
        ]
    if st == "arrow":
        p0 = (x, y + 0.5 * bh)
        p1 = (x + bw, y + 0.5 * bh)
        wing = max(14.0, min(32.0, 0.22 * bw))
        return [
            p0,
            p1,
            (p1[0] - wing, p1[1] - 0.65 * wing),
            p1,
            (p1[0] - wing, p1[1] + 0.65 * wing),
        ]
    if st == "line":
        return [(x, y + bh), (x + bw, y)]
    if st == "square":
        s = min(bw, bh)
        return [(x, y), (x + s, y), (x + s, y + s), (x, y + s), (x, y)]
    # Default: rect
    return [(x, y), (x + bw, y), (x + bw, y + bh), (x, y + bh), (x, y)]


def make_shape_object(
    shape_kind: str,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    color: str = "#00F5D4",
    thickness: float = 3.5,
    z_index: int = 0,
    layer: int = 0,
) -> CanvasObject:
    pts = make_shape_points(shape_kind, x, y, w, h)
    bx, by, bw, bh = compute_points_bbox(pts, min_size=24.0)
    labels = {
        "rect": "Hình chữ nhật",
        "square": "Hình vuông",
        "circle": "Hình tròn",
        "ellipse": "Hình elip",
        "triangle": "Hình tam giác",
        "diamond": "Hình thoi",
        "arrow": "Mũi tên",
        "line": "Đoạn thẳng",
    }
    st = str(shape_kind or "rect").strip().lower()
    return CanvasObject(
        id=new_object_id("shp"),
        obj_type="shape",
        x=bx,
        y=by,
        w=bw,
        h=bh,
        rotation=0.0,
        z_index=z_index,
        layer=layer,
        points=pts,
        raw_points=list(pts),
        color=color,
        thickness=thickness,
        is_beautified=True,
        shape_type=st.upper(),
        shape_label_vi=labels.get(st, "Hình khối"),
        confidence=1.0,
    )


def make_text_object(
    text: str,
    x: float,
    y: float,
    *,
    w: Optional[float] = None,
    h: Optional[float] = None,
    color: str = "#F8FAFC",
    bg_color: str = "",
    border_color: str = "#06B6D4",
    font_size: int = 16,
    z_index: int = 0,
    layer: int = 0,
    is_ai: bool = False,
) -> CanvasObject:
    clean = str(text or "").strip() or "Nhấn đúp để nhập chữ..."
    lines = clean.splitlines() or [clean]
    max_chars = max(len(ln) for ln in lines)
    est_w = float(w) if w is not None else max(180.0, min(680.0, max_chars * (font_size * 0.62) + 32.0))
    # Estimate wrapped line count
    chars_per_line = max(14, int((est_w - 24.0) / max(6.0, font_size * 0.58)))
    total_lines = sum(max(1, math.ceil(len(ln) / float(chars_per_line))) for ln in lines)
    est_h = float(h) if h is not None else max(48.0, min(560.0, total_lines * (font_size * 1.48) + 22.0))
    return CanvasObject(
        id=new_object_id("txt"),
        obj_type="text",
        x=float(x),
        y=float(y),
        w=est_w,
        h=est_h,
        rotation=0.0,
        z_index=z_index,
        layer=layer,
        text=clean,
        raw_text=clean,
        font_size=int(font_size),
        color=color,
        bg_color=bg_color,
        border_color=border_color,
        is_ai=is_ai,
    )


def make_sticky_object(
    text: str,
    x: float,
    y: float,
    *,
    palette_idx: int = 0,
    z_index: int = 0,
    layer: int = 0,
) -> CanvasObject:
    bg, border, fg = STICKY_PALETTE[int(palette_idx) % len(STICKY_PALETTE)]
    clean = str(text or "").strip() or "Ghi chú nhanh..."
    return CanvasObject(
        id=new_object_id("stk"),
        obj_type="sticky",
        x=float(x),
        y=float(y),
        w=220.0,
        h=150.0,
        rotation=0.0,
        z_index=z_index,
        layer=layer,
        text=clean,
        raw_text=clean,
        font_size=14,
        color=fg,
        bg_color=bg,
        border_color=border,
    )


def _derive_sticky_border_color(bg_hex: str) -> str:
    """Pick matching sticky border color from palette or darken hex RGB cleanly."""
    clean = str(bg_hex or "").strip()
    upper = clean.upper()
    for pal_bg, pal_border, _ in STICKY_PALETTE:
        if pal_bg.upper() == upper:
            return pal_border
    if clean.startswith("#") and len(clean) == 7:
        try:
            r = int(clean[1:3], 16)
            g = int(clean[3:5], 16)
            b = int(clean[5:7], 16)
            return f"#{int(r * 0.72):02X}{int(g * 0.72):02X}{int(b * 0.72):02X}"
        except ValueError:
            pass
    return clean


class SlideSceneGraph:
    """
    Per-slide Object-Based Canvas scene graph with Rust Quadtree spatial indexing,
    exact hit-testing, viewport culling, multi-selection, recoloring, z-order control,
    and unified Undo/Redo history.
    """

    def __init__(self, layer: int = 0, max_history: int = 80) -> None:
        self.layer = int(layer)
        self.max_history = max(10, int(max_history))
        self.objects: List[CanvasObject] = []
        self.selected_ids: List[str] = []
        self._undo_stack: List[Tuple[List[CanvasObject], List[str]]] = []
        self._redo_stack: List[Tuple[List[CanvasObject], List[str]]] = []
        self._next_z: int = 1
        self._spatial = make_spatial_index()
        self._spatial_dirty: bool = True
        self._obj_by_id: Dict[str, CanvasObject] = {}

    def mark_spatial_dirty(self) -> None:
        self._spatial_dirty = True

    def _ensure_spatial_index(self) -> None:
        if (
            not self._spatial_dirty
            and self._spatial.len() == len(self.objects)
            and len(self._obj_by_id) == len(self.objects)
        ):
            return
        obj_map: Dict[str, CanvasObject] = {}
        items: List[Tuple[str, float, float, float, float, int]] = []
        for o in self.objects:
            object.__setattr__(o, "_owner_scene", self)
            obj_map[o.id] = o
            x0, y0, x1, y1 = o.world_aabb()
            items.append((o.id, x0, y0, x1, y1, int(o.z_index)))
        self._obj_by_id = obj_map
        self._spatial.rebuild(items)
        self._spatial_dirty = False

    def next_z_index(self) -> int:
        z = self._next_z
        self._next_z += 1
        return z

    def _snapshot(self) -> Tuple[List[CanvasObject], List[str]]:
        return ([o.clone(new_id=False) for o in self.objects], list(self.selected_ids))

    def push_undo_snapshot(self) -> None:
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self.max_history:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def can_undo(self) -> bool:
        return bool(self._undo_stack)

    def can_redo(self) -> bool:
        return bool(self._redo_stack)

    def _rebind_objects_after_restore(self) -> None:
        obj_map: Dict[str, CanvasObject] = {}
        for o in self.objects:
            object.__setattr__(o, "_owner_scene", self)
            obj_map[o.id] = o
        self._obj_by_id = obj_map
        self._spatial_dirty = True

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        self._redo_stack.append(self._snapshot())
        objs, sel_ids = self._undo_stack.pop()
        self.objects = objs
        self._rebind_objects_after_restore()
        valid = self._obj_by_id
        self.selected_ids = [oid for oid in sel_ids if oid in valid]
        self._recompute_next_z()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        self._undo_stack.append(self._snapshot())
        objs, sel_ids = self._redo_stack.pop()
        self.objects = objs
        self._rebind_objects_after_restore()
        valid = self._obj_by_id
        self.selected_ids = [oid for oid in sel_ids if oid in valid]
        self._recompute_next_z()
        return True

    def _recompute_next_z(self) -> None:
        self._next_z = max((o.z_index for o in self.objects), default=0) + 1

    def get_by_id(self, oid: str) -> Optional[CanvasObject]:
        if len(self._obj_by_id) == len(self.objects):
            return self._obj_by_id.get(oid)
        for o in self.objects:
            if o.id == oid:
                return o
        return None

    def ordered_objects(self) -> List[CanvasObject]:
        return sorted(self.objects, key=lambda o: o.z_index)

    def add_object(self, obj: CanvasObject, *, record_undo: bool = True, select: bool = False) -> CanvasObject:
        if record_undo:
            self.push_undo_snapshot()
        if obj.z_index <= 0:
            obj.z_index = self.next_z_index()
        else:
            self._next_z = max(self._next_z, obj.z_index + 1)
        obj.layer = self.layer
        object.__setattr__(obj, "_owner_scene", self)
        prev_len = len(self.objects)
        self.objects.append(obj)
        self._obj_by_id[obj.id] = obj
        if not self._spatial_dirty and self._spatial.len() == prev_len:
            x0, y0, x1, y1 = obj.world_aabb()
            self._spatial.upsert(obj.id, x0, y0, x1, y1, int(obj.z_index))
        else:
            self._spatial_dirty = True
        if select:
            self.selected_ids = [obj.id]
        return obj

    def selected_objects(self) -> List[CanvasObject]:
        if not self.selected_ids:
            return []
        if len(self._obj_by_id) == len(self.objects):
            seen = set()
            objs: List[CanvasObject] = []
            for oid in self.selected_ids:
                if oid not in seen:
                    obj = self._obj_by_id.get(oid)
                    if obj is not None:
                        seen.add(oid)
                        objs.append(obj)
            objs.sort(key=lambda o: o.z_index)
            return objs
        sel_set = set(self.selected_ids)
        return [o for o in self.ordered_objects() if o.id in sel_set]

    def clear_selection(self) -> None:
        self.selected_ids = []

    def select_only(self, oid: Optional[str]) -> None:
        if oid is None:
            self.selected_ids = []
        elif self.get_by_id(oid) is not None:
            self.selected_ids = [oid]

    def toggle_select(self, oid: str) -> None:
        if self.get_by_id(oid) is None:
            return
        if oid in self.selected_ids:
            self.selected_ids = [x for x in self.selected_ids if x != oid]
        else:
            self.selected_ids.append(oid)

    def delete_selected(self) -> int:
        if not self.selected_ids:
            return 0
        self.push_undo_snapshot()
        sel = set(self.selected_ids)
        before = len(self.objects)
        self.objects = [o for o in self.objects if o.id not in sel]
        for oid in sel:
            self._obj_by_id.pop(oid, None)
            if not self._spatial_dirty:
                self._spatial.remove(oid)
        self.selected_ids = []
        if len(self._obj_by_id) != len(self.objects) or self._spatial.len() != len(self.objects):
            self._spatial_dirty = True
        return before - len(self.objects)

    def duplicate_selected(self, offset_x: float = 24.0, offset_y: float = 24.0) -> List[CanvasObject]:
        sel = self.selected_objects()
        if not sel:
            return []
        self.push_undo_snapshot()
        clones: List[CanvasObject] = []
        new_ids: List[str] = []
        for obj in sel:
            c = obj.clone(new_id=True, dx=offset_x, dy=offset_y, z_index=self.next_z_index())
            object.__setattr__(c, "_owner_scene", self)
            self.objects.append(c)
            self._obj_by_id[c.id] = c
            if not self._spatial_dirty:
                x0, y0, x1, y1 = c.world_aabb()
                self._spatial.upsert(c.id, x0, y0, x1, y1, int(c.z_index))
            clones.append(c)
            new_ids.append(c.id)
        self.selected_ids = new_ids
        if self._spatial.len() != len(self.objects):
            self._spatial_dirty = True
        return clones

    def recolor_selected(self, color_hex: str) -> int:
        """Recolor selected objects (undoable): `color` for strokes/shapes/text, `bg_color`/`border_color` for sticky notes."""
        col = str(color_hex or "").strip()
        if not col:
            return 0
        targets = [
            o
            for o in self.selected_objects()
            if not o.locked and o.obj_type in ("stroke", "shape", "text", "sticky")
        ]
        if not targets:
            return 0
        self.push_undo_snapshot()
        for obj in targets:
            if obj.obj_type == "sticky":
                obj.bg_color = col
                obj.border_color = _derive_sticky_border_color(col)
            else:
                obj.color = col
        return len(targets)

    def bring_selected_to_front(self) -> bool:
        sel = self.selected_objects()
        if not sel:
            return False
        self.push_undo_snapshot()
        for obj in sel:
            obj.z_index = self.next_z_index()
        self._spatial_dirty = True
        return True

    def send_selected_to_back(self) -> bool:
        sel = self.selected_objects()
        if not sel:
            return False
        self.push_undo_snapshot()
        min_z = min((o.z_index for o in self.objects), default=1)
        for idx, obj in enumerate(sel):
            obj.z_index = min_z - len(sel) + idx
        # Normalize z_index back to >= 1
        ordered = self.ordered_objects()
        for i, o in enumerate(ordered, start=1):
            o.z_index = i
        self._recompute_next_z()
        self._spatial_dirty = True
        return True

    def hit_test(self, bx: float, by: float, tolerance: float = 10.0) -> Optional[CanvasObject]:
        if not self.objects:
            return None
        self._ensure_spatial_index()
        query_tol = max(2.0, float(tolerance)) + 4.0
        cand_ids = self._spatial.query_point(float(bx), float(by), query_tol)
        if not cand_ids:
            return None
        cand_objs = [self._obj_by_id[cid] for cid in cand_ids if cid in self._obj_by_id]
        if not cand_objs:
            return None
        # Reverse so highest (z_index, insertion_order) sits at the highest index for tie-break parity
        tuples = [o.to_hit_test_tuple() for o in reversed(cand_objs)]
        hit_id = hit_test_objects_fast(bx, by, tuples, tolerance=tolerance)
        return self._obj_by_id.get(hit_id) if hit_id is not None else None

    def marquee_select(self, rx: float, ry: float, rw: float, rh: float, *, additive: bool = False) -> List[CanvasObject]:
        if not self.objects:
            if not additive:
                self.selected_ids = []
            return []
        self._ensure_spatial_index()
        mx0 = min(float(rx), float(rx) + float(rw))
        my0 = min(float(ry), float(ry) + float(rh))
        mw = abs(float(rw))
        mh = abs(float(rh))
        cand_ids = self._spatial.query_rect(mx0 - 2.0, my0 - 2.0, mw + 4.0, mh + 4.0)
        if cand_ids:
            cand_objs = [self._obj_by_id[cid] for cid in cand_ids if cid in self._obj_by_id]
            tuples = [o.to_marquee_tuple() for o in cand_objs]
            hit_ids = marquee_select_objects_fast(rx, ry, rw, rh, tuples)
        else:
            hit_ids = []
        if additive:
            sel_set = set(self.selected_ids)
            for hid in hit_ids:
                if hid not in sel_set:
                    self.selected_ids.append(hid)
                    sel_set.add(hid)
        else:
            self.selected_ids = list(hit_ids)
        return self.selected_objects()

    def visible_objects_in_rect(self, rx: float, ry: float, rw: float, rh: float) -> List[CanvasObject]:
        """Fast viewport culling via Quadtree spatial index, ordered back-to-front by z_index."""
        if not self.objects:
            return []
        self._ensure_spatial_index()
        cand_ids = self._spatial.query_rect(float(rx), float(ry), float(rw), float(rh))
        if not cand_ids:
            return []
        return [self._obj_by_id[cid] for cid in cand_ids if cid in self._obj_by_id]

    def hit_test_transform_handle(
        self,
        bx: float,
        by: float,
        zoom_scale: float = 1.0,
    ) -> Optional[Tuple[str, CanvasObject]]:
        """Check if `(bx, by)` hits a resize/rotate handle on a single selected object."""
        sel = self.selected_objects()
        if len(sel) != 1:
            return None
        obj = sel[0]
        if obj.locked:
            return None
        z = max(0.25, float(zoom_scale))
        hit_r = 12.0 / z
        rot_offset = 28.0 / z
        handles = obj.handle_positions_world(rotate_offset=rot_offset)
        # Prioritize rotate handle first
        rx, ry = handles["rotate"]
        if math.hypot(bx - rx, by - ry) <= hit_r * 1.25:
            return ("rotate", obj)
        for name in HANDLE_NAMES:
            hx, hy = handles[name]
            if math.hypot(bx - hx, by - hy) <= hit_r:
                return (name, obj)
        return None

    def selection_world_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        sel = self.selected_objects()
        if not sel:
            return None
        aabbs = [o.world_aabb() for o in sel]
        x0 = min(a[0] for a in aabbs)
        y0 = min(a[1] for a in aabbs)
        x1 = max(a[2] for a in aabbs)
        y1 = max(a[3] for a in aabbs)
        return (x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))

    def content_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.objects:
            return None
        aabbs = [o.world_aabb() for o in self.objects]
        x0 = min(a[0] for a in aabbs)
        y0 = min(a[1] for a in aabbs)
        x1 = max(a[2] for a in aabbs)
        y1 = max(a[3] for a in aabbs)
        return (x0, y0, max(1.0, x1 - x0), max(1.0, y1 - y0))

    def clear_all(self, *, record_undo: bool = True) -> None:
        if record_undo and self.objects:
            self.push_undo_snapshot()
        self.objects.clear()
        self.selected_ids.clear()
        self._obj_by_id.clear()
        self._spatial.clear()
        self._spatial_dirty = True
        self._next_z = 1


def compute_resized_bbox(
    orig: CanvasObject,
    handle: str,
    world_pt: Point2D,
    min_size: float = 24.0,
) -> Tuple[float, float, float, float]:
    """Compute new local unrotated `(x, y, w, h)` when dragging one of the 8 resize handles."""
    cx, cy = orig.center
    lx, ly = rotate_point(world_pt[0], world_pt[1], cx, cy, -orig.rotation)
    x0, y0 = orig.x, orig.y
    x1, y1 = orig.x + orig.w, orig.y + orig.h

    if "w" in handle:
        x0 = min(lx, x1 - min_size)
    if "e" in handle:
        x1 = max(lx, x0 + min_size)
    if "n" in handle:
        y0 = min(ly, y1 - min_size)
    if "s" in handle:
        y1 = max(ly, y0 + min_size)

    return (x0, y0, max(min_size, x1 - x0), max(min_size, y1 - y0))
