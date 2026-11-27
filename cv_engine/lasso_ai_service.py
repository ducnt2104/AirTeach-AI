#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Lasso-to-AI Service (`cv_engine/lasso_ai_service.py`)
for AirTeach AI (DNT - Victer):
- Supports both freehand polygon Lasso and rectangular region selection.
- Computes tight bounding box with configurable 15-20px padding (`DEFAULT_LASSO_PADDING_PX = 18`).
- Enhances cropped region contrast (grayscale/LAB contrast boost, grid reduction) and scales `<= 1024px`
  before transmitting to Local Vision LLM or Cloud Gemini.
- Defines the 4 Floating Action Bar flows:
  1. `SOLVE_AI`: Giải bằng AI
  2. `WRITE_SOLUTION_TO_BOARD`: Viết lời giải lên bảng
  3. `BEAUTIFY_LATEX`: Làm đẹp chữ/LaTeX (replaces handwritten strokes cleanly)
  4. `EXPLAIN_QUICK`: Giải thích nhanh
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from cv_engine.gemini_client import compress_snapshot_png


DEFAULT_LASSO_PADDING_PX = 18


class LassoQuickAction(str, Enum):
    SOLVE_AI = "SOLVE_AI"
    WRITE_SOLUTION_TO_BOARD = "WRITE_SOLUTION_TO_BOARD"
    BEAUTIFY_LATEX = "BEAUTIFY_LATEX"
    EXPLAIN_QUICK = "EXPLAIN_QUICK"


@dataclass
class LassoSelectionRegion:
    polygon: List[Tuple[float, float]]
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float

    @property
    def rect_tuple(self) -> Tuple[float, float, float, float]:
        return (self.bbox_x, self.bbox_y, self.bbox_w, self.bbox_h)


def compute_lasso_bounding_box(
    points: Sequence[Tuple[float, float]],
    canvas_size: Tuple[float, float],
    padding: float = DEFAULT_LASSO_PADDING_PX,
) -> Optional[LassoSelectionRegion]:
    """Compute bounded rectangle `(x, y, w, h)` with `18px` padding around a Lasso polygon."""
    if len(points) < 2:
        return None
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    cw = max(64.0, float(canvas_size[0]))
    ch = max(64.0, float(canvas_size[1]))
    x0 = max(0.0, min(xs) - padding)
    y0 = max(0.0, min(ys) - padding)
    x1 = min(cw, max(xs) + padding)
    y1 = min(ch, max(ys) + padding)
    w = x1 - x0
    h = y1 - y0
    if w < 16.0 or h < 16.0:
        return None
    return LassoSelectionRegion(
        polygon=[(float(p[0]), float(p[1])) for p in points],
        bbox_x=x0,
        bbox_y=y0,
        bbox_w=w,
        bbox_h=h,
    )


def point_in_polygon(px: float, py: float, poly: Sequence[Tuple[float, float]]) -> bool:
    """Standard ray-casting algorithm checking whether `(px, py)` lies inside Lasso polygon."""
    if len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / ((yj - yi) if abs(yj - yi) > 1e-9 else 1e-9) + xi
        )
        if intersect:
            inside = not inside
        j = i
    return inside


def stroke_inside_lasso(
    stroke_points: Sequence[Tuple[float, float]],
    region: LassoSelectionRegion,
) -> bool:
    """Return True if a majority or centroid of stroke points fall inside the lasso bbox/polygon."""
    if not stroke_points:
        return False
    if len(region.polygon) >= 5:
        hits = sum(1 for (x, y) in stroke_points if point_in_polygon(x, y, region.polygon))
        return hits >= max(1, int(math_ceil(len(stroke_points) * 0.4)))
    x0, y0, w, h = region.rect_tuple
    x1, y1 = x0 + w, y0 + h
    hits = sum(1 for (x, y) in stroke_points if x0 <= x <= x1 and y0 <= y <= y1)
    return hits >= max(1, int(math_ceil(len(stroke_points) * 0.4)))


def math_ceil(val: float) -> int:
    iv = int(val)
    return iv if val == iv else iv + 1


def enhance_lasso_crop_png(
    raw_png_bytes: bytes,
    max_dim: int = 1024,
) -> Optional[bytes]:
    """
    Boost contrast & crispness of a cropped handwritten canvas PNG region
    and compress cleanly below `max_dim` before sending to Local/Cloud Vision AI.
    """
    if not raw_png_bytes:
        return None
    arr = np.frombuffer(raw_png_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None or bgr.size == 0:
        return raw_png_bytes

    # Boost contrast on luminance channel so thin/neon strokes stand out cleanly
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge([l_eq, a_ch, b_ch])
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    ok, encoded = cv2.imencode(".png", enhanced, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    if not ok:
        return raw_png_bytes
    return compress_snapshot_png(encoded.tobytes(), max_dim=max_dim)


def build_lasso_prompt(
    action: LassoQuickAction,
    free_regions_norm: Optional[Sequence[Sequence[int]]] = None,
) -> str:
    """Build targeted prompts for each of the 4 Smart Lasso Floating Action Bar buttons."""
    free_hint = ""
    if free_regions_norm:
        free_hint = (
            f" Hãy đặt nội dung vẽ/viết lời giải vào vùng trống bên cạnh theo hệ tọa độ [0..1000]: "
            f"{list(free_regions_norm)}."
        )
    if action == LassoQuickAction.SOLVE_AI:
        return (
            "Đọc chính xác đề bài, biểu thức hoặc sơ đồ viết tay trong ảnh vùng khoanh này "
            "và giải chi tiết từng bước bằng tiếng Việt. "
            "Không dùng ký tự $$$ hay thẻ <arrow>, dùng ký hiệu toán chuẩn."
            + free_hint
        )
    if action == LassoQuickAction.WRITE_SOLUTION_TO_BOARD:
        return (
            "Đọc bài toán trong ảnh vùng khoanh, giải ngắn gọn theo từng bước rõ ràng. "
            "Hãy trả về lời giải ngắn gọn và đính kèm khối JSON "
            "```whiteboard_actions\n"
            '{"speech_or_text": "Đã viết lời giải lên bảng.", "actions": ['
            '{"type": "draw_text", "x": 420, "y": 140, "text": "Bước 1: ...", "color": "#00F5D4", "fontSize": 17}'
            "]}\n"
            "``` (theo hệ tọa độ [0..1000]) để đưa thẳng vào vùng trống bên cạnh."
            + free_hint
        )
    if action == LassoQuickAction.BEAUTIFY_LATEX:
        return (
            "Hãy nhận diện chính xác duy nhất công thức toán học hoặc dòng chữ viết tay trong ảnh này. "
            "Chỉ trả về đúng biểu thức LaTeX chuẩn (ví dụ: 2x + 1 = 3 hoặc \\frac{-b \\pm \\sqrt{\\Delta}}{2a}), "
            "không giải thích dài dòng."
        )
    return (
        "Hãy tóm tắt và giải thích ngắn gọn khái niệm hoặc nội dung trong ảnh vùng khoanh này "
        "theo ngôn ngữ sư phạm dễ hiểu cho học sinh."
        + free_hint
    )


def extract_clean_beautified_formula(ai_reply: str) -> str:
    """Extract crisp formula string from AI beautify response without extra markdown noise."""
    cleaned = (ai_reply or "").strip()
    cleaned = re.sub(r"^```(?:latex|tex|text)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Remove trailing JSON blocks if LLM included any
    cleaned = re.sub(r"```whiteboard_actions.*?```", "", cleaned, flags=re.DOTALL).strip()
    first_line = next((ln.strip() for ln in cleaned.splitlines() if ln.strip()), "")
    return first_line.strip("$ ").strip()
