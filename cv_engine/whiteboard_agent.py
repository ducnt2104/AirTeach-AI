#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agentic Whiteboard Protocol (`cv_engine/whiteboard_agent.py`)
for AirTeach AI (DNT - Victer):
1. Whiteboard Action Protocol (JSON Tool Calling compatible with both Gemini & OpenAI-compatible Local LLMs):
   - `write_on_board({ content, format: "latex" | "text", position: "auto" | {x, y}, fontSize, color })`
   - `solve_on_board({ steps: [...], targetRegion })`
   - `draw_shape({ type: "arrow" | "circle" | "rect" | "line" | "axis", ... })`
   - `clear_board()`
   - `undo_last_ai_action()`
2. Fast Local Intent Parser (< 5ms): detects direct natural Vietnamese/English whiteboard commands
   like `"ghi lên bảng hộ tôi 2x + 1 = 3 đi"`, `"viết công thức định lý Pytago lên bảng"`,
   `"vẽ hình tròn"`, `"vẽ trục tọa độ"`, `"xóa bảng"` immediately without waiting for an LLM roundtrip.
3. Smart Non-Overlapping Region Finder: scans occupied bounding boxes (ink strokes, slides, annotations)
   and places new AI formulas/steps onto open canvas space cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cv_engine.app_paths import get_app_logger
from cv_engine.output_sanitizer import OutputSanitizer

_LOGGER = get_app_logger("whiteboard_agent")


RectTuple = Tuple[float, float, float, float]  # (x, y, w, h)


WHITEBOARD_TOOL_SCHEMAS: Tuple[Dict[str, Any], ...] = (
    {
        "name": "write_on_board",
        "description": "Write formatted mathematical formula (LaTeX) or crisp pedagogical text onto an empty spot on the Smartboard.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Formula or concise text to write."},
                "format": {"type": "string", "enum": ["latex", "text"]},
                "position": {"description": "'auto' or {x, y} object"},
                "fontSize": {"type": "number"},
                "color": {"type": "string"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "solve_on_board",
        "description": "Write neatly aligned step-by-step mathematical solution lines onto the Smartboard.",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {"type": "array", "items": {"type": "string"}},
                "targetRegion": {"description": "Optional [x, y, w, h] near which solution should appear"},
            },
            "required": ["steps"],
        },
    },
    {
        "name": "draw_shape",
        "description": "Draw real vector shapes or coordinate axes onto the Smartboard.",
        "parameters": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ["arrow", "circle", "rect", "line", "axis"]},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "w": {"type": "number"},
                "h": {"type": "number"},
                "radius": {"type": "number"},
                "x2": {"type": "number"},
                "y2": {"type": "number"},
                "color": {"type": "string"},
            },
            "required": ["type"],
        },
    },
    {"name": "clear_board", "description": "Clear all strokes and annotations on the Smartboard.", "parameters": {"type": "object", "properties": {}}},
    {"name": "undo_last_ai_action", "description": "Undo the most recent AI whiteboard action.", "parameters": {"type": "object", "properties": {}}},
)


_KNOWN_FORMULAS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:định\s*lý\s*)?py[-\s]*ta[-\s]*go|pythagor", re.IGNORECASE), r"a^2 + b^2 = c^2"),
    (re.compile(r"công\s*thức\s*nghi[ệe]m|phương\s*trình\s*b[ậa]c\s*(?:2|hai)", re.IGNORECASE), r"ax^2 + bx + c = 0 \Rightarrow x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"),
    (re.compile(r"\bdelta\b|biệt\s*thức", re.IGNORECASE), r"\Delta = b^2 - 4ac"),
    (re.compile(r"vi[-\s]*[eéê]t|vieta", re.IGNORECASE), r"x_1 + x_2 = -\frac{b}{a},\quad x_1 x_2 = \frac{c}{a}"),
    (re.compile(r"định\s*luật\s*ii?\s*newton|lực\s*newton", re.IGNORECASE), r"\vec{F} = m \cdot \vec{a}"),
    (re.compile(r"einstein|tương\s*đối|năng\s*lượng\s*khối\s*lượng", re.IGNORECASE), r"E = mc^2"),
    (re.compile(r"chu\s*vi\s*hình\s*tròn", re.IGNORECASE), r"C = 2\pi R"),
    (re.compile(r"diện\s*tích\s*hình\s*tròn", re.IGNORECASE), r"S = \pi R^2"),
)


@dataclass
class WhiteboardAction:
    tool: str  # "write_on_board" | "solve_on_board" | "draw_shape" | "clear_board" | "undo_last_ai_action"
    args: Dict[str, Any] = field(default_factory=dict)
    source: str = "fast_intent"  # "fast_intent" | "llm_tool"


def _looks_like_math(text: str) -> bool:
    return any(
        tok in text
        for tok in ("=", "+", "-", "*", "/", "^", "_", "\\frac", "\\sqrt", "√", "²", "³", "Δ", "π", "∫", "∑")
    )


def find_free_board_region(
    occupied_rects: Sequence[RectTuple],
    canvas_size: Tuple[float, float],
    needed_size: Tuple[float, float] = (320.0, 64.0),
    preferred_anchor: Optional[Tuple[float, float]] = None,
    margin: float = 28.0,
) -> Tuple[float, float]:
    """
    Scan the Smartboard canvas and select a collision-free `(x, y)` origin
    using minimal overlap score + closeness to `preferred_anchor`.
    """
    cw = max(400.0, float(canvas_size[0]))
    ch = max(300.0, float(canvas_size[1]))
    nw = min(max(120.0, float(needed_size[0])), cw - 2.0 * margin)
    nh = min(max(42.0, float(needed_size[1])), ch - 2.0 * margin)

    def _overlap_area(cand: RectTuple) -> float:
        cx, cy, w, h = cand
        total = 0.0
        for ox, oy, ow, oh in occupied_rects:
            ix0 = max(cx - 10.0, ox)
            iy0 = max(cy - 10.0, oy)
            ix1 = min(cx + w + 10.0, ox + ow)
            iy1 = min(cy + h + 10.0, oy + oh)
            if ix1 > ix0 and iy1 > iy0:
                total += (ix1 - ix0) * (iy1 - iy0)
        return total

    step_x = max(48.0, (cw - nw - 2.0 * margin) / 7.0)
    step_y = max(40.0, (ch - nh - 2.0 * margin) / 6.0)

    best_xy = (margin, max(64.0, margin))
    best_cost = float("inf")

    y_cur = max(56.0, margin)
    while y_cur + nh <= ch - margin + 1.0:
        x_cur = margin
        while x_cur + nw <= cw - margin + 1.0:
            ov = _overlap_area((x_cur, y_cur, nw, nh))
            dist_pen = 0.0
            if preferred_anchor is not None:
                dx = x_cur - preferred_anchor[0]
                dy = y_cur - preferred_anchor[1]
                dist_pen = 0.08 * math.hypot(dx, dy)
            else:
                # Gently prefer upper-middle clean region below slide bar
                dist_pen = 0.03 * y_cur + 0.015 * x_cur
            cost = ov * 100.0 + dist_pen
            if cost < best_cost:
                best_cost = cost
                best_xy = (x_cur, y_cur)
                if ov == 0.0 and preferred_anchor is None:
                    return best_xy
            x_cur += step_x
        y_cur += step_y

    return best_xy


def build_vector_shape_strokes(
    shape_type: str,
    args: Dict[str, Any],
    origin_xy: Tuple[float, float],
) -> List[List[Tuple[float, float]]]:
    """Generate clean vector stroke polylines for `draw_shape` (`arrow`, `circle`, `rect`, `line`, `axis`)."""
    st = (shape_type or "line").strip().lower()
    ox = float(args.get("x", origin_xy[0]))
    oy = float(args.get("y", origin_xy[1]))
    w = float(args.get("w", 160.0))
    h = float(args.get("h", 100.0))
    r = float(args.get("radius", 65.0))

    if st == "circle":
        cx = ox + r
        cy = oy + r
        pts = [
            (cx + r * math.cos(2.0 * math.pi * i / 48.0), cy + r * math.sin(2.0 * math.pi * i / 48.0))
            for i in range(49)
        ]
        return [pts]

    if st == "ellipse":
        rx = max(20.0, w * 0.5)
        ry = max(14.0, h * 0.5)
        cx = ox + rx
        cy = oy + ry
        pts = [
            (cx + rx * math.cos(2.0 * math.pi * i / 48.0), cy + ry * math.sin(2.0 * math.pi * i / 48.0))
            for i in range(49)
        ]
        return [pts]

    if st == "triangle":
        pts_arg = args.get("points")
        if isinstance(pts_arg, list) and len(pts_arg) >= 3:
            tri = [(float(p[0]), float(p[1])) for p in pts_arg[:3] if isinstance(p, (list, tuple)) and len(p) >= 2]
            if len(tri) == 3:
                return [[tri[0], tri[1], tri[2], tri[0]]]
        # Right triangle ABC with right angle at A (bottom-left) by default
        p_a = (ox, oy + h)
        p_b = (ox, oy)
        p_c = (ox + w, oy + h)
        return [[p_a, p_b, p_c, p_a]]

    if st == "rect":
        return [[(ox, oy), (ox + w, oy), (ox + w, oy + h), (ox, oy + h), (ox, oy)]]

    if st == "arrow":
        x2 = float(args.get("x2", ox + max(120.0, w)))
        y2 = float(args.get("y2", oy))
        ang = math.atan2(y2 - oy, x2 - ox)
        head_len = 16.0
        left = (x2 - head_len * math.cos(ang - 0.45), y2 - head_len * math.sin(ang - 0.45))
        right = (x2 - head_len * math.cos(ang + 0.45), y2 - head_len * math.sin(ang + 0.45))
        return [[(ox, oy), (x2, y2)], [left, (x2, y2), right]]

    if st in ("axis", "axes"):
        size = max(140.0, min(w, 260.0))
        cx = ox + size * 0.5
        cy = oy + size * 0.5
        half = size * 0.5
        # X-axis, arrow, Y-axis, arrow
        strokes: List[List[Tuple[float, float]]] = [
            [(cx - half, cy), (cx + half, cy)],
            [(cx + half - 10.0, cy - 6.0), (cx + half, cy), (cx + half - 10.0, cy + 6.0)],
            [(cx, cy + half), (cx, cy - half)],
            [(cx - 6.0, cy - half + 10.0), (cx, cy - half), (cx + 6.0, cy - half + 10.0)],
        ]
        # Subtle unit ticks
        for k in (-2, -1, 1, 2):
            tx = cx + k * (half * 0.35)
            ty = cy - k * (half * 0.35)
            strokes.append([(tx, cy - 4.0), (tx, cy + 4.0)])
            strokes.append([(cx - 4.0, ty), (cx + 4.0, ty)])
        return strokes

    x2 = float(args.get("x2", ox + max(120.0, w)))
    y2 = float(args.get("y2", oy))
    return [[(ox, oy), (x2, y2)]]


def find_free_regions_normalized(
    occupied_rects: Sequence[RectTuple],
    canvas_size: Tuple[float, float],
    max_regions: int = 3,
) -> List[List[int]]:
    """
    Scan the Smartboard and return top `max_regions` collision-free `[x, y, w, h]` boxes
    in normalized `[0..1000]` coordinates so Gemini can place solutions/diagrams cleanly.
    """
    cw = max(400.0, float(canvas_size[0]))
    ch = max(300.0, float(canvas_size[1]))
    candidates: List[Tuple[float, List[int]]] = []
    # Probe 3x2 grid of 300x260 normalized regions
    for gy in (80, 380, 660):
        for gx in (50, 360, 660):
            rw, rh = 290, 250
            px, py = (gx / 1000.0) * cw, (gy / 1000.0) * ch
            pw, ph = (rw / 1000.0) * cw, (rh / 1000.0) * ch
            ov = 0.0
            for ox, oy, ow, oh in occupied_rects:
                ix0 = max(px, ox)
                iy0 = max(py, oy)
                ix1 = min(px + pw, ox + ow)
                iy1 = min(py + ph, oy + oh)
                if ix1 > ix0 and iy1 > iy0:
                    ov += (ix1 - ix0) * (iy1 - iy0)
            candidates.append((ov, [gx, gy, rw, rh]))
    candidates.sort(key=lambda item: item[0])
    return [rect for _, rect in candidates[:max_regions]]


def _clamp_norm(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        if not math.isfinite(v):
            return default
        return max(0.0, min(1000.0, v))
    except Exception:
        return default


def _norm_to_px_x(nx: float, cw: float) -> float:
    return (_clamp_norm(nx) / 1000.0) * cw


def _norm_to_px_y(ny: float, ch: float) -> float:
    return (_clamp_norm(ny) / 1000.0) * ch


class WhiteboardAgent:
    """Fast Intent Parser + Tool Action Extractor for Agentic Whiteboard control."""

    _WRITE_CMD_RE = re.compile(
        r"^(?:hãy\s+|làm\s*ơn\s+|giúp\s*(?:tôi|mình)?\s*|bạn\s+)?"
        r"(?:ghi|viết|thêm|đưa|hiển\s*thị|write|put)\s*"
        r"(?:(?:lên|ra|vào)\s*bảng\s*)?"
        r"(?:hộ\s*(?:tôi|mình|em|anh)?\s*|giùm\s*(?:tôi|mình)?\s*)?"
        r"(?:lên\s*bảng\s*)?"
        r"(?:công\s*thức\s*|phương\s*trình\s*|chữ\s*)?"
        r"[:\-]?\s*(.+?)\s*(?:lên\s*bảng|ra\s*bảng|đi|nhé|nha|giúp\s*tôi|hộ\s*tôi|!|\.)*$",
        re.IGNORECASE,
    )

    @classmethod
    def parse_fast_intent(cls, user_text: str) -> Optional[WhiteboardAction]:
        """
        Instant Local Intent Parser (< 5ms, zero network latency) for direct whiteboard commands.
        Returns `None` when the query requires full LLM reasoning (e.g., open-ended problem solving).
        """
        raw = (user_text or "").strip()
        if not raw:
            return None
        low = raw.lower()

        if re.match(r"^(?:hãy\s+)?(?:xóa|clear)\s*(?:toàn\s*bộ\s*)?(?:bảng|board)(?:\s*(?:đi|nhé|hộ))?[!.]*$", low):
            return WhiteboardAction(tool="clear_board", source="fast_intent")

        if re.match(r"^(?:hoàn\s*tác|undo)\s*(?:thao\s*tác\s*)?(?:ai|vừa\s*viết)?[!.]*$", low):
            return WhiteboardAction(tool="undo_last_ai_action", source="fast_intent")

        # Shape commands (only when NOT combined with multi-part "solve & write" prompt)
        if re.match(r"^(?:hãy\s+)?(?:vẽ|draw)\s+", low) and "giải" not in low:
            if any(k in low for k in ("trục tọa độ", "hệ trục", "oxy", "axis", "axes")):
                return WhiteboardAction(tool="draw_shape", args={"type": "axis", "w": 220.0}, source="fast_intent")
            if any(k in low for k in ("tam giác", "triangle")):
                return WhiteboardAction(tool="draw_shape", args={"type": "triangle", "w": 190.0, "h": 140.0}, source="fast_intent")
            if any(k in low for k in ("elip", "ellipse")):
                return WhiteboardAction(tool="draw_shape", args={"type": "ellipse", "w": 200.0, "h": 110.0}, source="fast_intent")
            if any(k in low for k in ("mũi tên", "arrow")):
                return WhiteboardAction(tool="draw_shape", args={"type": "arrow", "w": 170.0}, source="fast_intent")
            if any(k in low for k in ("hình tròn", "vòng tròn", "circle")):
                return WhiteboardAction(tool="draw_shape", args={"type": "circle", "radius": 65.0}, source="fast_intent")
            if any(k in low for k in ("hình vuông", "hình chữ nhật", "rect")):
                return WhiteboardAction(tool="draw_shape", args={"type": "rect", "w": 180.0, "h": 110.0}, source="fast_intent")
            if any(k in low for k in ("đường thẳng", "line")):
                return WhiteboardAction(tool="draw_shape", args={"type": "line", "w": 180.0}, source="fast_intent")

        # Don't swallow solve/explain requests as literal write commands
        has_solve_word = any(w in low for w in ("giải ", "tìm x", "chứng minh", "tóm tắt", "giải thích"))
        has_board_target = ("lên bảng" in low) or ("ra bảng" in low) or low.startswith(("ghi ", "viết ", "write "))
        if has_board_target and not has_solve_word:
            # Known canonical formula lookup
            for pat, formula in _KNOWN_FORMULAS:
                if pat.search(raw):
                    return WhiteboardAction(
                        tool="write_on_board",
                        args={"content": formula, "format": "latex", "position": "auto"},
                        source="fast_intent",
                    )

            m = cls._WRITE_CMD_RE.match(raw)
            if m:
                payload = m.group(1).strip()
                # Clean trailing filler like "lên bảng" or "đi" if captured
                payload = re.sub(r"\s+(?:lên\s*bảng|ra\s*bảng|đi|nhé|nha|hộ\s*tôi)$", "", payload, flags=re.IGNORECASE).strip()
                if payload:
                    fmt = "latex" if _looks_like_math(payload) else "text"
                    return WhiteboardAction(
                        tool="write_on_board",
                        args={"content": OutputSanitizer.normalize_delimiters_and_tags(payload).strip("$ "), "format": fmt, "position": "auto"},
                        source="fast_intent",
                    )

        return None

    @staticmethod
    def build_tool_prompt_instruction(
        canvas_size: Optional[Tuple[float, float]] = None,
        free_regions_norm: Optional[Sequence[Sequence[int]]] = None,
    ) -> str:
        """
        Agentic Whiteboard JSON Protocol [0..1000] instruction for Gemini.
        Instructs Gemini to return `speech_or_text` + `actions` in normalized `[0..1000]` coordinates.
        """
        ctx_parts: List[str] = []
        if canvas_size is not None:
            ctx_parts.append(f"Kích thước bảng hiện tại: {int(canvas_size[0])}x{int(canvas_size[1])}px.")
        if free_regions_norm:
            ctx_parts.append(f"Các vùng trống ưu tiên [x,y,w,h] hệ [0..1000]: {list(free_regions_norm)}.")
        ctx_str = (" " + " ".join(ctx_parts)) if ctx_parts else ""
        return (
            "Hãy trình bày giải thích SƯ PHẠM CHI TIẾT, đầy đủ từng bước với cấu trúc Markdown rõ ràng (tiêu đề, bảng so sánh nếu phù hợp, công thức LaTeX $...$ / $$...$$).\n"
            "Chỉ KHI giáo viên yêu cầu vẽ hình, viết công thức, giải bài hoặc chú thích trực tiếp lên bảng, "
            "hãy đính kèm thêm ở cuối phản hồi một block JSON trong thẻ ```whiteboard_actions ... ``` theo hệ tọa độ chuẩn hóa [0..1000]:\n"
            "{\n"
            '  "speech_or_text": "Tóm tắt nội dung vừa trình bày trên bảng cho học sinh",\n'
            '  "actions": [\n'
            '    {"type": "draw_shape", "shape": "triangle|line|arrow|rect|circle|ellipse|axes", "x": 150, "y": 200, "w": 220, "h": 180, "color": "#00F5D4", "strokeWidth": 3, "label": "ABC"},\n'
            '    {"type": "draw_text", "x": 150, "y": 420, "text": "a² + b² = c²", "color": "#FEE440", "fontSize": 18},\n'
            '    {"type": "draw_polyline", "points": [[100,500],[200,450],[300,520]], "color": "#00F5D4", "smooth": true},\n'
            '    {"type": "highlight_region", "x": 120, "y": 180, "w": 280, "h": 240, "color": "#FEE440", "label": "Chú ý"},\n'
            '    {"type": "clear_region", "x": 100, "y": 100, "w": 200, "h": 200},\n'
            '    {"type": "clear_all"}\n'
            "  ]\n"
            "}"
            + ctx_str
        )

    @classmethod
    def extract_actions_from_response(cls, response_text: str) -> Tuple[List[WhiteboardAction], str]:
        """
        Parse both:
        1. Normalized [0..1000] Agentic Whiteboard JSON (`{"speech_or_text": ..., "actions": [...]}`),
        2. Legacy `[{"tool": ..., "args": ...}]` arrays inside ```whiteboard_actions or ```json blocks,
        and safely validate every field without crashing on partial/malformed AI output.
        """
        actions: List[WhiteboardAction] = []
        cleaned_text = response_text or ""
        extracted_speech: Optional[str] = None

        _ALLOWED_NEW = {
            "draw_text",
            "draw_shape",
            "draw_polyline",
            "highlight_region",
            "clear_region",
            "clear_all",
        }
        _ALLOWED_LEGACY = {
            "write_on_board",
            "solve_on_board",
            "draw_shape",
            "clear_board",
            "undo_last_ai_action",
        }

        def _append_action_dict(it: Dict[str, Any]) -> None:
            nonlocal extracted_speech
            if not isinstance(it, dict):
                return
            tool_name = str(it.get("tool") or "").strip()
            act_type = str(it.get("type") or "").strip().lower()

            # Allow normalized actions specified via {"tool": "draw_text", "args": {...}}
            if not act_type and tool_name.lower() in _ALLOWED_NEW:
                merged: Dict[str, Any] = dict(it)
                if isinstance(it.get("args"), dict):
                    merged.update(it["args"])
                it = merged
                act_type = tool_name.lower()

            # Case A: Legacy {"tool": "...", "args": {...}}
            if not act_type and tool_name in _ALLOWED_LEGACY:
                actions.append(
                    WhiteboardAction(
                        tool=tool_name,
                        args=dict(it.get("args") or {}),
                        source="llm_tool",
                    )
                )
                return

            # Case B: Normalized [0..1000] {"type": "draw_text" | "draw_shape" | ...}
            if act_type not in _ALLOWED_NEW:
                return
            if act_type == "clear_all":
                actions.append(WhiteboardAction(tool="clear_board", args={}, source="llm_tool"))
                return
            if act_type == "clear_region":
                actions.append(
                    WhiteboardAction(
                        tool="clear_region",
                        args={
                            "x": _clamp_norm(it.get("x"), 0.0),
                            "y": _clamp_norm(it.get("y"), 0.0),
                            "w": max(20.0, _clamp_norm(it.get("w"), 200.0)),
                            "h": max(20.0, _clamp_norm(it.get("h"), 200.0)),
                            "normalized": True,
                        },
                        source="llm_tool",
                    )
                )
                return
            if act_type == "draw_text":
                txt = str(it.get("text") or it.get("content") or "").strip()
                if not txt:
                    return
                actions.append(
                    WhiteboardAction(
                        tool="draw_text",
                        args={
                            "x": _clamp_norm(it.get("x"), 120.0),
                            "y": _clamp_norm(it.get("y"), 160.0),
                            "text": txt,
                            "color": str(it.get("color") or "#FEE440"),
                            "fontSize": float(it.get("fontSize") or it.get("font_size") or 16),
                            "normalized": True,
                        },
                        source="llm_tool",
                    )
                )
                return
            if act_type == "highlight_region":
                actions.append(
                    WhiteboardAction(
                        tool="highlight_region",
                        args={
                            "x": _clamp_norm(it.get("x"), 100.0),
                            "y": _clamp_norm(it.get("y"), 100.0),
                            "w": max(30.0, _clamp_norm(it.get("w"), 220.0)),
                            "h": max(30.0, _clamp_norm(it.get("h"), 140.0)),
                            "color": str(it.get("color") or "#FEE440"),
                            "label": str(it.get("label") or "").strip(),
                            "normalized": True,
                        },
                        source="llm_tool",
                    )
                )
                return
            if act_type == "draw_polyline":
                raw_pts = it.get("points") or []
                clean_pts: List[List[float]] = []
                if isinstance(raw_pts, list):
                    for pt in raw_pts:
                        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                            clean_pts.append([_clamp_norm(pt[0]), _clamp_norm(pt[1])])
                if len(clean_pts) < 2:
                    return
                actions.append(
                    WhiteboardAction(
                        tool="draw_polyline",
                        args={
                            "points": clean_pts,
                            "color": str(it.get("color") or "#00F5D4"),
                            "strokeWidth": float(it.get("strokeWidth") or it.get("stroke_width") or 3.2),
                            "smooth": bool(it.get("smooth", True)),
                            "normalized": True,
                        },
                        source="llm_tool",
                    )
                )
                return
            if act_type == "draw_shape":
                shape_kind = str(it.get("shape") or it.get("shape_type") or it.get("kind") or "rect").strip().lower()
                bbox = it.get("bbox")
                rad_norm = _clamp_norm(it["radius"], 70.0) if "radius" in it else None
                if "center_x" in it and "x" not in it:
                    r_val = rad_norm if rad_norm is not None else 90.0
                    bx = max(0.0, float(it["center_x"]) - r_val)
                    by = max(0.0, float(it.get("center_y", 500.0)) - r_val)
                    bw = float(it.get("w", r_val * 2.0))
                    bh = float(it.get("h", r_val * 2.0))
                else:
                    bx = it.get("x", bbox[0] if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else 160.0)
                    by = it.get("y", bbox[1] if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else 180.0)
                    bw = it.get("w", bbox[2] if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else 200.0)
                    bh = it.get("h", bbox[3] if isinstance(bbox, (list, tuple)) and len(bbox) >= 4 else 150.0)
                args_out: Dict[str, Any] = {
                    "type": shape_kind,
                    "x": _clamp_norm(bx, 160.0),
                    "y": _clamp_norm(by, 180.0),
                    "w": max(20.0, _clamp_norm(bw, 200.0)),
                    "h": max(20.0, _clamp_norm(bh, 150.0)),
                    "color": str(it.get("color") or "#00F5D4"),
                    "strokeWidth": float(it.get("strokeWidth") or it.get("stroke_width") or 3.2),
                    "label": str(it.get("label") or "").strip(),
                    "normalized": True,
                }
                if rad_norm is not None:
                    args_out["radius"] = max(15.0, rad_norm)
                if "x2" in it and "y2" in it:
                    args_out["x2"] = _clamp_norm(it.get("x2"), 360.0)
                    args_out["y2"] = _clamp_norm(it.get("y2"), 180.0)
                if isinstance(it.get("points"), list):
                    args_out["points"] = [
                        [_clamp_norm(p[0]), _clamp_norm(p[1])]
                        for p in it["points"]
                        if isinstance(p, (list, tuple)) and len(p) >= 2
                    ]
                actions.append(WhiteboardAction(tool="draw_shape", args=args_out, source="llm_tool"))

        def _parse_json_blob(blob: str) -> None:
            nonlocal extracted_speech
            try:
                data = json.loads(blob.strip())
            except Exception as exc:
                _LOGGER.debug("Whiteboard JSON parse skipped malformed block: %s", exc)
                return
            if isinstance(data, dict) and isinstance(data.get("actions"), list):
                if isinstance(data.get("speech_or_text"), str) and data["speech_or_text"].strip():
                    extracted_speech = data["speech_or_text"].strip()
                for it in data["actions"]:
                    _append_action_dict(it)
                return
            items = data if isinstance(data, list) else [data]
            for it in items:
                _append_action_dict(it)

        pattern = re.compile(r"```(?:whiteboard_actions|json)\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)
        matches = list(pattern.finditer(cleaned_text))
        for m in reversed(matches):
            blob = m.group(1)
            before_count = len(actions)
            _parse_json_blob(blob)
            if len(actions) > before_count or extracted_speech is not None:
                cleaned_text = cleaned_text[: m.start()] + cleaned_text[m.end() :]

        # Also handle raw standalone JSON object {"speech_or_text": ..., "actions": [...]}
        if not actions and cleaned_text.strip().startswith("{"):
            _parse_json_blob(cleaned_text.strip())
            if actions or extracted_speech is not None:
                cleaned_text = extracted_speech or ""

        if extracted_speech:
            if not cleaned_text.strip():
                cleaned_text = extracted_speech
            elif extracted_speech not in cleaned_text:
                cleaned_text = f"{cleaned_text.strip()}\n{extracted_speech}"

        return actions, OutputSanitizer.sanitize_plain_text(cleaned_text)
