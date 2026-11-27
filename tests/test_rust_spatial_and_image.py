#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification & Benchmark Suite for:
  1. Rust 2D Quadtree Spatial Index (`RustSpatialIndex`) & Pure-Python Fallback (`PythonSpatialIndex`)
  2. `SlideSceneGraph` Quadtree hit-test, marquee select, viewport culling (< 1 ms/query on 3,000 objects)
  3. `SlideSceneGraph.recolor_selected(...)` + Undo/Redo
  4. Multi-threaded GIL-free Rust RGB24 Resize + 3x3 Unsharp Mask (`preprocess_image_resize_sharpen_fast`)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import airteach_core_rs
from cv_engine.canvas_scene import (
    CanvasObject,
    SlideSceneGraph,
    make_shape_object,
    make_sticky_object,
    make_text_object,
)
from cv_engine.rust_bridge import (
    HAS_RUST_CORE,
    RUST_CORE_VERSION,
    PythonSpatialIndex,
    _py_resize_sharpen_rgb,
    make_spatial_index,
    preprocess_image_resize_sharpen_fast,
)


def _laplacian_variance(rgb: np.ndarray) -> float:
    """Measure edge sharpness via 3x3 discrete Laplacian variance on luma."""
    luma = (
        0.299 * rgb[:, :, 0].astype(np.float64)
        + 0.587 * rgb[:, :, 1].astype(np.float64)
        + 0.114 * rgb[:, :, 2].astype(np.float64)
    )
    lap = (
        -4.0 * luma[1:-1, 1:-1]
        + luma[:-2, 1:-1]
        + luma[2:, 1:-1]
        + luma[1:-1, :-2]
        + luma[1:-1, 2:]
    )
    return float(np.var(lap))


def test_rust_spatial_index_crud_ordering_and_parity() -> None:
    assert HAS_RUST_CORE is True
    assert "2.0.0" in RUST_CORE_VERSION
    assert airteach_core_rs.version() == "airteach_core_rs 2.0.0"
    assert hasattr(airteach_core_rs, "RustSpatialIndex")
    assert hasattr(airteach_core_rs, "preprocess_image_resize_sharpen_rs")

    rs_idx = make_spatial_index(-1000.0, -1000.0, 1000.0, 1000.0, 4, 6)
    py_idx = PythonSpatialIndex(-1000.0, -1000.0, 1000.0, 1000.0, 4, 6)
    assert type(rs_idx).__name__ == "RustSpatialIndex"

    items: List[Tuple[str, float, float, float, float, int]] = [
        ("a", 0.0, 0.0, 100.0, 100.0, 1),
        ("b", 40.0, 40.0, 140.0, 140.0, 5),
        ("c", 50.0, 50.0, 120.0, 120.0, 3),
        ("d", -500.0, -400.0, -300.0, -250.0, 10),
        ("straddle", -20.0, -20.0, 20.0, 20.0, 7),
        ("oob", 2500.0, 2500.0, 2600.0, 2600.0, 9),
    ]
    for idx in (rs_idx, py_idx):
        for oid, x0, y0, x1, y1, z in items:
            idx.upsert(oid, x0, y0, x1, y1, z)
        assert idx.len() == len(items)
        assert len(idx) == len(items)

    # Descending z_index ordering on point query at (60, 60) -> hits b (z=5), c (z=3), a (z=1)
    assert rs_idx.query_point(60.0, 60.0, 0.0) == ["b", "c", "a"]
    assert py_idx.query_point(60.0, 60.0, 0.0) == ["b", "c", "a"]

    # Straddling & out-of-bounds items must still be found cleanly
    assert rs_idx.query_point(0.0, 0.0, 0.0) == ["straddle", "a"]
    assert py_idx.query_point(0.0, 0.0, 0.0) == ["straddle", "a"]
    assert rs_idx.query_point(2550.0, 2550.0, 5.0) == ["oob"]
    assert py_idx.query_point(2550.0, 2550.0, 5.0) == ["oob"]

    # Upsert moves an existing ID to a new location & updates z_index
    for idx in (rs_idx, py_idx):
        idx.upsert("a", 400.0, 400.0, 460.0, 460.0, 20)
        assert idx.len() == len(items)
        assert "a" not in idx.query_point(60.0, 60.0, 0.0)
        assert idx.query_point(430.0, 430.0, 0.0) == ["a"]

    # Query rect supports negative rw/rh & returns ascending z_index
    assert rs_idx.query_rect(150.0, 150.0, -140.0, -140.0) == py_idx.query_rect(150.0, 150.0, -140.0, -140.0)
    assert set(rs_idx.query_rect(150.0, 150.0, -140.0, -140.0)) == {"straddle", "c", "b"}

    # Remove returns True once, False on second call
    for idx in (rs_idx, py_idx):
        assert idx.remove("c") is True
        assert idx.remove("c") is False
        assert idx.len() == len(items) - 1
        assert "c" not in idx.query_point(60.0, 60.0, 0.0)

    # Rebuild expands root bounds for out-of-bounds items & resets accurately
    batch = [(f"obj_{i}", float(i * 30), float(i * 20), float(i * 30 + 25), float(i * 20 + 18), i) for i in range(120)]
    for idx in (rs_idx, py_idx):
        idx.rebuild(batch)
        assert idx.len() == 120
    assert rs_idx.query_rect(280.0, 180.0, 400.0, 300.0) == py_idx.query_rect(280.0, 180.0, 400.0, 300.0)

    # Clear empties index cleanly
    for idx in (rs_idx, py_idx):
        idx.clear()
        assert idx.len() == 0
        assert idx.query_point(0.0, 0.0, 5000.0) == []
        assert idx.query_rect(-5000.0, -5000.0, 10000.0, 10000.0) == []


def test_quadtree_3000_objects_hit_test_marquee_and_viewport_culling_benchmark() -> None:
    sc = SlideSceneGraph(layer=0)
    assert type(sc._spatial).__name__ == "RustSpatialIndex"

    # Build 3,000 objects on a 60 x 50 grid across (0..12000, 0..10000)
    cols, rows = 60, 50
    spacing_x, spacing_y = 200.0, 200.0
    grid_ids: dict[Tuple[int, int], str] = {}

    for r in range(rows):
        for c in range(cols):
            x = c * spacing_x + 20.0
            y = r * spacing_y + 20.0
            idx = r * cols + c
            kind = idx % 4
            if kind == 0:
                obj = make_shape_object("rect", x, y, 80.0, 60.0)
            elif kind == 1:
                obj = make_text_object(f"Cell {c},{r}", x, y, w=85.0, h=55.0)
            elif kind == 2:
                obj = make_sticky_object(f"Note {idx}", x, y)
                obj.w = 90.0
                obj.h = 65.0
            else:
                pts = [(x, y), (x + 40.0, y + 30.0), (x + 80.0, y + 55.0)]
                obj = CanvasObject(
                    id=f"str_{idx}",
                    obj_type="stroke",
                    x=x,
                    y=y,
                    w=80.0,
                    h=55.0,
                    points=pts,
                    raw_points=list(pts),
                    color="#00F5D4",
                )
            sc.add_object(obj, record_undo=False, select=False)
            grid_ids[(c, r)] = obj.id

    assert len(sc.objects) == 3000

    # Add a rotated rectangle on top of cell (10, 10) to validate OBB + z_index
    top_rot = make_shape_object("rect", 10 * spacing_x + 25.0, 10 * spacing_y + 25.0, 80.0, 50.0)
    top_rot.rotation = 30.0
    sc.add_object(top_rot, record_undo=False, select=False)

    # First query triggers single batch Quadtree rebuild
    t_rebuild0 = time.perf_counter()
    hit_top = sc.hit_test(10 * spacing_x + 65.0, 10 * spacing_y + 50.0, tolerance=8.0)
    rebuild_ms = (time.perf_counter() - t_rebuild0) * 1000.0
    assert hit_top is not None and hit_top.id == top_rot.id
    assert sc._spatial.len() == 3001
    assert sc._spatial_dirty is False

    # Verify exact hit-test across multiple cells + empty gap miss
    for (c, r) in [(0, 0), (15, 22), (45, 39), (59, 49)]:
        cx = c * spacing_x + 60.0
        cy = r * spacing_y + 48.0
        hit = sc.hit_test(cx, cy, tolerance=8.0)
        assert hit is not None and hit.id == grid_ids[(c, r)]
    assert sc.hit_test(155.0, 155.0, tolerance=5.0) is None

    # Verify direct property mutation automatically dirties & refreshes Quadtree
    moved_obj = sc.get_by_id(grid_ids[(0, 0)])
    assert moved_obj is not None
    moved_obj.translate(50000.0, 0.0)
    assert sc._spatial_dirty is True
    assert sc.hit_test(60.0, 48.0, tolerance=5.0) is None
    assert sc.hit_test(50060.0, 48.0, tolerance=8.0).id == moved_obj.id
    moved_obj.translate(-50000.0, 0.0)
    assert sc.hit_test(60.0, 48.0, tolerance=8.0).id == moved_obj.id

    # Verify visible_objects_in_rect parity vs brute-force AABB & ascending z_index order
    vx, vy, vw, vh = 2000.0, 2000.0, 620.0, 420.0
    vis = sc.visible_objects_in_rect(vx, vy, vw, vh)
    brute = [
        o
        for o in sc.ordered_objects()
        if (
            o.world_aabb()[2] >= vx
            and o.world_aabb()[0] <= vx + vw
            and o.world_aabb()[3] >= vy
            and o.world_aabb()[1] <= vy + vh
        )
    ]
    assert [o.id for o in vis] == [o.id for o in brute]
    assert 6 <= len(vis) <= 25

    # Verify marquee_select matches visible subset cleanly
    selected = sc.marquee_select(vx, vy, vw, vh)
    assert {o.id for o in selected} == {o.id for o in brute}

    # --- Benchmarks on 3,000+ objects (must be < 1.0 ms/query!) ---
    n_pt = 500
    t0 = time.perf_counter()
    for i in range(n_pt):
        c = (i * 7) % cols
        r = (i * 11) % rows
        _ = sc._spatial.query_point(c * spacing_x + 60.0, r * spacing_y + 48.0, 10.0)
    qt_pt_ms = (time.perf_counter() - t0) * 1000.0 / n_pt

    t0 = time.perf_counter()
    for i in range(n_pt):
        c = (i * 7) % cols
        r = (i * 11) % rows
        res = sc.hit_test(c * spacing_x + 60.0, r * spacing_y + 48.0, tolerance=8.0)
        assert res is not None
    scene_hit_ms = (time.perf_counter() - t0) * 1000.0 / n_pt

    n_rect = 200
    t0 = time.perf_counter()
    for i in range(n_rect):
        c = (i * 5) % (cols - 4)
        r = (i * 3) % (rows - 4)
        res_m = sc.marquee_select(c * spacing_x + 10.0, r * spacing_y + 10.0, 520.0, 320.0)
        assert len(res_m) >= 4
    scene_marquee_ms = (time.perf_counter() - t0) * 1000.0 / n_rect

    t0 = time.perf_counter()
    for i in range(n_rect):
        c = (i * 5) % (cols - 4)
        r = (i * 3) % (rows - 4)
        res_v = sc.visible_objects_in_rect(c * spacing_x, r * spacing_y, 800.0, 600.0)
        assert len(res_v) >= 6
    cull_ms = (time.perf_counter() - t0) * 1000.0 / n_rect

    print(
        f"[BENCH] 3001-obj Quadtree: rebuild={rebuild_ms:.2f}ms | "
        f"raw_query_point={qt_pt_ms:.4f}ms | scene.hit_test={scene_hit_ms:.4f}ms | "
        f"scene.marquee_select={scene_marquee_ms:.4f}ms | visible_cull={cull_ms:.4f}ms",
        flush=True,
    )
    assert qt_pt_ms < 1.0, f"RustSpatialIndex.query_point too slow: {qt_pt_ms:.4f} ms"
    assert scene_hit_ms < 1.0, f"SlideSceneGraph.hit_test too slow: {scene_hit_ms:.4f} ms"
    assert scene_marquee_ms < 1.0, f"SlideSceneGraph.marquee_select too slow: {scene_marquee_ms:.4f} ms"
    assert cull_ms < 1.0, f"SlideSceneGraph.visible_objects_in_rect too slow: {cull_ms:.4f} ms"


def test_scene_recolor_selected_and_undo_redo() -> None:
    sc = SlideSceneGraph(layer=0)
    stroke = CanvasObject(
        id="stk_free_1",
        obj_type="stroke",
        x=10.0,
        y=10.0,
        w=100.0,
        h=40.0,
        points=[(10.0, 20.0), (110.0, 40.0)],
        color="#00F5D4",
    )
    shape = make_shape_object("circle", 150.0, 100.0, 120.0, 120.0, color="#38BDF8")
    text = make_text_object("Công thức đạo hàm", 320.0, 120.0, color="#F8FAFC")
    sticky = make_sticky_object("Ghi chú kiểm tra", 520.0, 120.0, palette_idx=0)
    locked_shape = make_shape_object("rect", 760.0, 120.0, 90.0, 70.0, color="#22C55E")
    locked_shape.locked = True

    for o in (stroke, shape, text, sticky, locked_shape):
        sc.add_object(o, record_undo=False, select=False)

    orig_sticky_bg = sticky.bg_color
    orig_sticky_border = sticky.border_color

    # No selection -> returns 0 and does not push undo snapshot
    assert sc.recolor_selected("#FF3366") == 0
    assert sc.can_undo() is False

    # Select all 5 objects -> 4 unlocked recolored, 1 locked untouched
    sc.selected_ids = [stroke.id, shape.id, text.id, sticky.id, locked_shape.id]
    changed = sc.recolor_selected("#FF3366")
    assert changed == 4
    assert sc.get_by_id(stroke.id).color == "#FF3366"
    assert sc.get_by_id(shape.id).color == "#FF3366"
    assert sc.get_by_id(text.id).color == "#FF3366"
    assert sc.get_by_id(sticky.id).bg_color == "#FF3366"
    assert sc.get_by_id(sticky.id).border_color.startswith("#")
    assert len(sc.get_by_id(sticky.id).border_color) == 7
    assert sc.get_by_id(sticky.id).border_color != orig_sticky_border
    assert sc.get_by_id(locked_shape.id).color == "#22C55E"

    # Undo restores exact original colors
    assert sc.undo() is True
    assert sc.get_by_id(stroke.id).color == "#00F5D4"
    assert sc.get_by_id(shape.id).color == "#38BDF8"
    assert sc.get_by_id(text.id).color == "#F8FAFC"
    assert sc.get_by_id(sticky.id).bg_color == orig_sticky_bg
    assert sc.get_by_id(sticky.id).border_color == orig_sticky_border

    # Redo reapplies "#FF3366"
    assert sc.redo() is True
    assert sc.get_by_id(stroke.id).color == "#FF3366"
    assert sc.get_by_id(shape.id).color == "#FF3366"
    assert sc.get_by_id(text.id).color == "#FF3366"
    assert sc.get_by_id(sticky.id).bg_color == "#FF3366"

    # Palette sticky recolor uses exact curated border color
    sc.select_only(sticky.id)
    assert sc.recolor_selected("#BBF7D0") == 1
    assert sc.get_by_id(sticky.id).bg_color == "#BBF7D0"
    assert sc.get_by_id(sticky.id).border_color == "#16A34A"


def test_preprocess_image_resize_sharpen_fast_quality_multithread_and_fallback() -> None:
    # 1. Synthetic crisp slide image (1920x1080) with high-contrast text-like bars & grid
    src_w, src_h = 1920, 1080
    img = np.full((src_h, src_w, 3), 240, dtype=np.uint8)
    # Dark header band + crisp vertical/horizontal diagram lines + colored blocks
    img[40:140, 80:1840] = (24, 38, 68)
    for y in range(200, 980, 32):
        img[y : y + 4, 100:1820] = (30, 41, 59)
    for x in range(120, 1800, 48):
        img[200:980, x : x + 3] = (15, 118, 110)
    img[320:640, 360:780] = (79, 70, 229)

    # Multi-threaded bilinear resize (1920x1080 -> 1280x720, 921,600 px >= 65,536)
    t0 = time.perf_counter()
    unsharp_off = preprocess_image_resize_sharpen_fast(img, src_w, src_h, 1280, 720, sharpen_amount=0.0)
    t1 = time.perf_counter()
    unsharp_on = preprocess_image_resize_sharpen_fast(img, src_w, src_h, 1280, 720, sharpen_amount=0.60)
    t2 = time.perf_counter()
    print(
        f"[BENCH] Rust MT Image 1920x1080->1280x720: resize_only={(t1 - t0) * 1000.0:.2f}ms | "
        f"resize+sharpen={(t2 - t1) * 1000.0:.2f}ms",
        flush=True,
    )

    assert isinstance(unsharp_off, np.ndarray)
    assert isinstance(unsharp_on, np.ndarray)
    assert unsharp_off.shape == (720, 1280, 3) and unsharp_off.dtype == np.uint8
    assert unsharp_on.shape == (720, 1280, 3) and unsharp_on.dtype == np.uint8

    # Unsharp mask strictly increases edge Laplacian variance on resized slide
    sharp_off = _laplacian_variance(unsharp_off)
    sharp_on = _laplacian_variance(unsharp_on)
    assert sharp_on > sharp_off * 1.15, f"Expected sharpened Laplacian > 1.15x unsharpened ({sharp_on:.1f} vs {sharp_off:.1f})"

    # Raw bytes input returns identical RGB24 bytes buffer
    raw_out = preprocess_image_resize_sharpen_fast(img.tobytes(), src_w, src_h, 1280, 720, sharpen_amount=0.60)
    assert isinstance(raw_out, bytes)
    assert len(raw_out) == 1280 * 720 * 3
    assert raw_out == unsharp_on.tobytes()

    # >2x Box/Area downscale path (1920x1080 -> 640x360, 230,400 px >= 65,536)
    box_out = preprocess_image_resize_sharpen_fast(img, src_w, src_h, 640, 360, sharpen_amount=0.40)
    assert isinstance(box_out, np.ndarray)
    assert box_out.shape == (360, 640, 3) and box_out.dtype == np.uint8
    assert float(np.std(box_out)) > 20.0

    # Constant flat image preserves exact color with zero overshoot/drift
    flat = np.full((120, 160, 3), (42, 128, 210), dtype=np.uint8)
    flat_out = preprocess_image_resize_sharpen_fast(flat, 160, 120, 320, 240, sharpen_amount=0.95)
    assert flat_out.shape == (240, 320, 3)
    assert np.all(flat_out[:, :, 0] == 42)
    assert np.all(flat_out[:, :, 1] == 128)
    assert np.all(flat_out[:, :, 2] == 210)

    # Small single-threaded path (< 65,536 px) + parity with pure-NumPy fallback
    small = np.zeros((96, 128, 3), dtype=np.uint8)
    small[:, :64] = (30, 80, 160)
    small[:, 64:] = (235, 220, 190)
    small[24:72, 32:96] = (190, 45, 75)
    rs_small = preprocess_image_resize_sharpen_fast(small, 128, 96, 96, 72, sharpen_amount=0.45)
    py_small = _py_resize_sharpen_rgb(small, 96, 72, 0.45)
    assert rs_small.shape == py_small.shape == (72, 96, 3)
    mad = float(np.mean(np.abs(rs_small.astype(np.float32) - py_small.astype(np.float32))))
    assert mad < 2.5, f"Rust vs Python fallback diverged too far (MAD={mad:.2f})"

    # Invalid arguments raise ValueError cleanly
    with pytest.raises(ValueError):
        preprocess_image_resize_sharpen_fast(b"\x00" * 10, 10, 10, 5, 5, 0.35)
    with pytest.raises(ValueError):
        preprocess_image_resize_sharpen_fast(small, 128, 96, 0, 72, 0.35)
