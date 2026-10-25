"""
AirTeach AI - Smart Ink OCR Service
====================================
Converts hand-drawn strokes to recognized text and LaTeX formulas.

Tier 1: RapidOCR (rapidocr_onnxruntime) — Vietnamese/English handwriting.
Tier 2: rapid-latex-ocr / pix2tex       — math formula → LaTeX string.
Tier 3: matplotlib.mathtext             — LaTeX → crisp RGBA PNG image.

All heavy inference runs on a background QRunnable to avoid UI blocking.
Degrades gracefully when ONNX OCR packages are unavailable (Python 3.15+).
"""
from __future__ import annotations

import io
import logging
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Optional heavy OCR imports (not available on Python 3.15 RC) ──────────────
try:
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR
    _RAPIDOCR_AVAILABLE = True
except ImportError:
    _RapidOCR = None  # type: ignore
    _RAPIDOCR_AVAILABLE = False
    logger.warning("rapidocr_onnxruntime not installed; handwriting OCR disabled.")

try:
    from rapid_latex_ocr import LatexOCR as _LatexOCR
    _LATEX_OCR_AVAILABLE = True
except ImportError:
    try:
        from pix2tex.cli import LatexOCR as _LatexOCR  # type: ignore
        _LATEX_OCR_AVAILABLE = True
    except ImportError:
        _LatexOCR = None  # type: ignore
        _LATEX_OCR_AVAILABLE = False
        logger.warning("rapid_latex_ocr/pix2tex not installed; formula OCR disabled.")


# ─── Data Structures ─────────────────────────────────────────────────────────

@dataclass
class StrokePoint:
    """A single timestamped point in a stroke."""
    x: float
    y: float
    t_ms: float
    pressure: float = 1.0


@dataclass
class Stroke:
    """A complete ink stroke (sequence of points with style metadata)."""
    points: list[StrokePoint] = field(default_factory=list)
    color_bgr: tuple[int, int, int] = (0, 0, 0)
    width: float = 3.0
    stroke_id: str = ""

    def to_numpy(self) -> np.ndarray:
        """Return (N, 2) array of (x, y) coordinates."""
        return np.array([[p.x, p.y] for p in self.points], dtype=np.float32)

    def bounding_box(self) -> tuple[float, float, float, float]:
        """Returns (x_min, y_min, x_max, y_max)."""
        if not self.points:
            return 0.0, 0.0, 0.0, 0.0
        pts = self.to_numpy()
        return float(pts[:, 0].min()), float(pts[:, 1].min()), \
               float(pts[:, 0].max()), float(pts[:, 1].max())


@dataclass
class OcrResult:
    """Result from OCR recognition."""
    text: str = ""
    latex: str = ""
    confidence: float = 0.0
    inference_ms: float = 0.0
    is_formula: bool = False
    render_image_rgba: Optional[np.ndarray] = None   # RGBA numpy array if rendered


# ─── Vietnamese OCR & LaTeX Sanitization Utilities ───────────────────────────

def normalize_vietnamese_ocr_text(text: str) -> str:
    """
    Standardize raw OCR recognized text for Vietnamese and multilingual handwriting.
    - Applies Unicode NFC normalization (composing decomposed accents/diacritics).
    - Converts full-width Asian punctuation to standard ASCII equivalents.
    - Eliminates irregular whitespace (e.g., spaces before punctuation marks).
    - Preserves logical paragraph structure and line breaks.
    """
    if not text:
        return ""

    # 1. Unicode NFC canonical composition
    normalized = unicodedata.normalize("NFC", text)

    # 2. Full-width punctuation mapping
    fw_map = {
        "，": ", ",
        "。": ". ",
        "！": "! ",
        "？": "? ",
        "：": ": ",
        "；": "; ",
        "（": " (",
        "）": ") ",
        "【": " [",
        "】": "] ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
    for fw_char, ascii_char in fw_map.items():
        normalized = normalized.replace(fw_char, ascii_char)

    # 3. Clean line by line
    cleaned_lines = []
    for line in normalized.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()
        line = re.sub(r"\s+([,.:;?!])", r"\1", line)
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def sanitize_latex_string(latex_str: str) -> str:
    """
    Sanitize and normalize math formula string for robust Matplotlib MathText rendering.
    - Strips outer dollar delimiters.
    - Converts common raw Unicode math operators to their LaTeX macro equivalents.
    - Cleans duplicate superscripts/subscripts (e.g. x^^2 -> x^2, x__1 -> x_1).
    - Balances unmatched curly braces.
    """
    if not latex_str:
        return ""

    s = latex_str.strip().strip("$").strip()
    if not s:
        return ""

    # Unicode symbol to LaTeX macro mapping
    math_symbols = {
        "×": r"\times ",
        "÷": r"\div ",
        "±": r"\pm ",
        "∓": r"\mp ",
        "≤": r"\le ",
        "≥": r"\ge ",
        "≠": r"\neq ",
        "≈": r"\approx ",
        "≡": r"\equiv ",
        "∞": r"\infty ",
        "√": r"\sqrt ",
        "π": r"\pi ",
        "θ": r"\theta ",
        "α": r"\alpha ",
        "β": r"\beta ",
        "γ": r"\gamma ",
        "δ": r"\delta ",
        "λ": r"\lambda ",
        "μ": r"\mu ",
        "σ": r"\sigma ",
        "ω": r"\omega ",
        "Δ": r"\Delta ",
        "Ω": r"\Omega ",
        "∑": r"\sum ",
        "∏": r"\prod ",
        "∫": r"\int ",
        "∂": r"\partial ",
        "→": r"\to ",
        "⇒": r"\Rightarrow ",
        "∈": r"\in ",
        "∉": r"\notin ",
        "⊂": r"\subset ",
    }
    for uni, macro in math_symbols.items():
        s = s.replace(uni, macro)

    # Clean double exponent or subscript marks
    s = re.sub(r"\^{2,}", "^", s)
    s = re.sub(r"_{2,}", "_", s)

    # Balance curly braces { }
    open_count = s.count("{")
    close_count = s.count("}")
    if open_count > close_count:
        s = s + ("}" * (open_count - close_count))
    elif close_count > open_count:
        s = ("{" * (close_count - open_count)) + s

    return s


# ─── Stroke → Image Renderer ──────────────────────────────────────────────────

class StrokeRenderer:
    """
    Renders a list of Stroke objects onto a white numpy image (BGR) for OCR input.
    """

    PADDING = 24  # pixels

    @classmethod
    def render(
        cls,
        strokes: list[Stroke],
        output_width: int = 0,
        output_height: int = 0,
        dpi_scale: float = 2.0,
    ) -> np.ndarray:
        """
        Render *strokes* to a white BGR image. If output_width/height are 0,
        the canvas is sized to the bounding box of all strokes + padding.

        Args:
            strokes:       List of Stroke objects to render.
            output_width:  Canvas width override (0 = auto).
            output_height: Canvas height override (0 = auto).
            dpi_scale:     Scale factor applied to coordinates (2.0 = 2× resolution).

        Returns:
            White BGR uint8 numpy array.
        """
        if not strokes:
            return np.full((64, 256, 3), 255, dtype=np.uint8)

        # ── Compute bounding box across all strokes ───────────────────────────
        xs, ys = [], []
        for stroke in strokes:
            for pt in stroke.points:
                xs.append(pt.x)
                ys.append(pt.y)

        if not xs:
            return np.full((64, 256, 3), 255, dtype=np.uint8)

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)
        pad = cls.PADDING

        scale = max(0.5, float(dpi_scale)) if dpi_scale > 0 else 2.0
        span_x = max(0.0, x_max - x_min)
        span_y = max(0.0, y_max - y_min)
        w = output_width if output_width > 0 else max(int(span_x * scale) + 2 * pad, 64)
        h = output_height if output_height > 0 else max(int(span_y * scale) + 2 * pad, 64)

        canvas = np.full((h, w, 3), 255, dtype=np.uint8)

        # ── Draw each stroke as a polyline ───────────────────────────────────
        for stroke in strokes:
            if len(stroke.points) < 2:
                if len(stroke.points) == 1:
                    pt = stroke.points[0]
                    cx = int((pt.x - x_min) * dpi_scale) + pad
                    cy = int((pt.y - y_min) * dpi_scale) + pad
                    r = max(1, int(stroke.width * dpi_scale / 2))
                    cv2.circle(canvas, (cx, cy), r, stroke.color_bgr, -1, cv2.LINE_AA)
                continue

            pts_px = np.array(
                [
                    [
                        int((pt.x - x_min) * dpi_scale) + pad,
                        int((pt.y - y_min) * dpi_scale) + pad,
                    ]
                    for pt in stroke.points
                ],
                dtype=np.int32,
            ).reshape(-1, 1, 2)

            thickness = max(1, int(stroke.width * dpi_scale))
            cv2.polylines(
                canvas, [pts_px], isClosed=False,
                color=stroke.color_bgr, thickness=thickness,
                lineType=cv2.LINE_AA,
            )

        return canvas


# ─── Handwriting OCR ─────────────────────────────────────────────────────────

class HandwritingOCR:
    """
    Wraps RapidOCR for offline Vietnamese/English handwriting recognition.
    Lazy-initializes the model on first call.
    """

    def __init__(self) -> None:
        self._engine = None

    def _ensure_engine(self) -> bool:
        if not _RAPIDOCR_AVAILABLE:
            return False
        if self._engine is None:
            self._engine = _RapidOCR()
        return True

    def recognize(
        self,
        image_bgr: np.ndarray,
    ) -> OcrResult:
        """
        Run handwriting OCR on a white-background BGR image.

        Returns:
            OcrResult with text, confidence, and inference_ms.
        """
        t0 = time.perf_counter()

        if not self._ensure_engine():
            return OcrResult(
                text="[OCR engine unavailable on Python 3.15+]",
                confidence=0.0,
                inference_ms=0.0,
            )

        result, _ = self._engine(image_bgr)
        elapsed = (time.perf_counter() - t0) * 1000.0

        if not result:
            return OcrResult(text="", confidence=0.0, inference_ms=elapsed)

        # RapidOCR returns list of [bbox, text, score]
        texts = [item[1] for item in result if item]
        scores = [float(item[2]) for item in result if item]
        combined_text = normalize_vietnamese_ocr_text(" ".join(texts))
        avg_confidence = float(np.mean(scores)) if scores else 0.0

        return OcrResult(
            text=combined_text,
            confidence=avg_confidence,
            inference_ms=elapsed,
        )


# ─── Geometric Math Parser (Offline Zero-Dependency Math Engine) ──────────────

@dataclass
class _MathToken:
    symbol: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    cx: float
    cy: float
    is_superscript: bool = False
    is_subscript: bool = False


class GeometricMathParser:
    """
    Native deterministic geometric parser for handwritten mathematical expressions.
    Classifies elementary mathematical strokes, fuses multi-stroke symbols (+, =, \\times),
    resolves spatial relationships (fractions, exponents, subscripts), and formats
    clean, renderable LaTeX strings.
    """

    @classmethod
    def parse(cls, strokes: list[Stroke]) -> str:
        if not strokes:
            return ""

        primitives: list[_MathToken] = []
        for s in strokes:
            if s is None:
                continue
            if hasattr(s, "to_numpy"):
                try:
                    pts = s.to_numpy()
                except Exception:
                    continue
            elif isinstance(s, np.ndarray):
                pts = s
            elif isinstance(s, (list, tuple)):
                try:
                    pts = np.asarray(s, dtype=np.float64)
                except Exception:
                    continue
            else:
                continue

            if not isinstance(pts, np.ndarray) or pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 2:
                continue
            if not np.all(np.isfinite(pts)):
                continue

            token = cls._classify_single_stroke(pts)
            if token is not None:
                primitives.append(token)

        if not primitives:
            return ""

        # Step 2: Multi-stroke symbol fusion (+, =, \times, \div, \pi, x)
        fused = cls._fuse_multi_stroke(primitives)

        # Step 3: Spatial structure parsing (fractions, superscripts, subscripts)
        latex = cls._build_expression(fused)
        return latex

    @classmethod
    def _classify_single_stroke(cls, pts: np.ndarray) -> Optional[_MathToken]:
        xmin, ymin = float(pts[:, 0].min()), float(pts[:, 1].min())
        xmax, ymax = float(pts[:, 0].max()), float(pts[:, 1].max())
        w = max(xmax - xmin, 1.0)
        h = max(ymax - ymin, 1.0)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        diffs = np.diff(pts, axis=0)
        arc_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
        chord_len = float(np.linalg.norm(pts[-1] - pts[0]))
        is_closed = (arc_len > 1e-4) and (chord_len / arc_len < 0.28)

        # PCA Linearity and angle
        centered = pts - pts.mean(axis=0)
        cov = np.dot(centered.T, centered) / max(len(pts) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        total_var = float(np.sum(eigvals))
        linearity = float(eigvals[-1] / total_var) if total_var > 1e-6 else 1.0

        principal = eigvecs[:, -1]
        angle_deg = math.degrees(math.atan2(float(principal[1]), float(principal[0])))
        angle_deg = (angle_deg + 90) % 180 - 90  # range [-90, 90]

        # 0. Dot / decimal point / multiplication dot
        if max(w, h) <= 8.0 or (arc_len <= 10.0 and max(w, h) <= 12.0):
            return _MathToken(".", xmin, ymin, xmax, ymax, cx, cy)

        # 1. Closed loop: 0, o, O vs \Delta (triangle)
        if is_closed or (w / h > 0.55 and w / h < 1.8 and chord_len / max(arc_len, 1e-4) < 0.35):
            pts_f32 = pts.astype(np.float32)
            contour_area = abs(cv2.contourArea(pts_f32))
            perim = cv2.arcLength(pts_f32, closed=True)
            if perim > 1e-4:
                circularity = (4.0 * np.pi * contour_area) / (perim * perim)
                approx = cv2.approxPolyDP(pts_f32, 0.07 * perim, closed=True)
                if len(approx) == 3 or (circularity < 0.68 and len(approx) <= 4):
                    return _MathToken(r"\Delta", xmin, ymin, xmax, ymax, cx, cy)
            return _MathToken("0", xmin, ymin, xmax, ymax, cx, cy)

        # 2. Chevrons < and > (sharp apex angle < 110 deg)
        if 0.3 <= h / w <= 2.8 and not is_closed and len(pts) >= 3:
            apex_min_idx = int(np.argmin(pts[:, 0]))
            apex_max_idx = int(np.argmax(pts[:, 0]))
            start_x, end_x = pts[0, 0], pts[-1, 0]
            if (start_x - pts[apex_min_idx, 0] > 0.28 * w) and (end_x - pts[apex_min_idx, 0] > 0.28 * w):
                # Calculate apex angle
                v1 = pts[0] - pts[apex_min_idx]
                v2 = pts[-1] - pts[apex_min_idx]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-4 and n2 > 1e-4:
                    cos_ang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                    ang_deg = math.degrees(math.acos(cos_ang))
                    if ang_deg < 112.0:
                        return _MathToken("<", xmin, ymin, xmax, ymax, cx, cy)
            elif (pts[apex_max_idx, 0] - start_x > 0.28 * w) and (pts[apex_max_idx, 0] - end_x > 0.28 * w):
                v1 = pts[0] - pts[apex_max_idx]
                v2 = pts[-1] - pts[apex_max_idx]
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 > 1e-4 and n2 > 1e-4:
                    cos_ang = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                    ang_deg = math.degrees(math.acos(cos_ang))
                    if ang_deg < 112.0:
                        return _MathToken(">", xmin, ymin, xmax, ymax, cx, cy)

        # 3. Parentheses ( and ) (smooth arc, apex angle >= 112 deg)
        if h > 1.35 * w and not is_closed and len(pts) >= 3 and w > 4.0:
            chord_x = 0.5 * (pts[0, 0] + pts[-1, 0])
            mid_idx = len(pts) // 2
            mid_x = pts[mid_idx, 0]
            if (chord_x - mid_x) > 0.20 * w:
                return _MathToken("(", xmin, ymin, xmax, ymax, cx, cy)
            elif (mid_x - chord_x) > 0.20 * w:
                return _MathToken(")", xmin, ymin, xmax, ymax, cx, cy)

        # 4. Straight line candidates
        if linearity > 0.85:
            abs_ang = abs(angle_deg)
            if abs_ang < 25.0:
                # Horizontal line (minus, fraction bar, or = component)
                return _MathToken("HORIZ_BAR", xmin, ymin, xmax, ymax, cx, cy)
            elif abs_ang > 70.0:
                # Vertical line (1, |, or + component)
                return _MathToken("VERT_BAR", xmin, ymin, xmax, ymax, cx, cy)
            elif 25.0 <= abs_ang <= 70.0:
                # Diagonal line (/ or component of \times, x, or \Delta)
                sym = "DIAG_POS" if angle_deg > 0 else "DIAG_NEG"
                return _MathToken(sym, xmin, ymin, xmax, ymax, cx, cy)

        # 5. Square root radical check
        # Starts high, dips low, rises high, then goes right
        if len(pts) >= 6:
            start_y, mid_min_y, end_x = pts[0, 1], pts[:, 1].max(), pts[-1, 0]
            if mid_min_y > start_y and (pts[-1, 1] < mid_min_y) and (end_x > cx + w * 0.2):
                return _MathToken(r"\sqrt", xmin, ymin, xmax, ymax, cx, cy)

        # 6. Integral S-curve
        if h > 2.0 * w and not is_closed:
            # Check inflection in x-direction
            dxs = diffs[:, 0]
            if np.any(dxs > 0) and np.any(dxs < 0):
                return _MathToken(r"\int", xmin, ymin, xmax, ymax, cx, cy)

        # 7. Curve heuristics for digits and common variables
        # Aspect ratio and curvature
        if w / h > 1.3:
            return _MathToken("HORIZ_BAR", xmin, ymin, xmax, ymax, cx, cy)
        elif h / w > 1.8:
            return _MathToken("1", xmin, ymin, xmax, ymax, cx, cy)

        # General single-stroke fallback (digits & variables)
        # Check start/end points to distinguish 2, 3, 5, 7, x, y, a, b, c, r
        start_pt = pts[0]
        end_pt = pts[-1]

        # 7: horizontal top, diagonal left-down
        if start_pt[0] < cx and end_pt[0] < cx and end_pt[1] > cy:
            return _MathToken("7", xmin, ymin, xmax, ymax, cx, cy)

        # 2: top curved, base horizontal
        if abs(end_pt[1] - ymax) < h * 0.2 and (end_pt[0] > xmin + w * 0.5):
            return _MathToken("2", xmin, ymin, xmax, ymax, cx, cy)

        # Fallback to variable 'x' or 'y' or digit
        return _MathToken("x", xmin, ymin, xmax, ymax, cx, cy)

    @classmethod
    def _fuse_multi_stroke(cls, tokens: list[_MathToken]) -> list[_MathToken]:
        fused: list[_MathToken] = []
        used = set()
        n = len(tokens)

        # Pass 0.1: Division sign \div (1 HORIZ_BAR + 2 dots: one above, one below)
        for i in range(n):
            if i in used or tokens[i].symbol != "HORIZ_BAR":
                continue
            th = tokens[i]
            w_bar = max(th.xmax - th.xmin, 1.0)
            dot_above = None
            dot_below = None
            for j in range(n):
                if j == i or j in used or tokens[j].symbol != ".":
                    continue
                tj = tokens[j]
                if abs(tj.cx - th.cx) <= 0.45 * w_bar:
                    if tj.cy < th.ymin and (th.ymin - tj.cy) <= 3.0 * w_bar:
                        dot_above = j
                    elif tj.cy > th.ymax and (tj.cy - th.ymax) <= 3.0 * w_bar:
                        dot_below = j
            if dot_above is not None and dot_below is not None:
                d_top = tokens[dot_above]
                d_bot = tokens[dot_below]
                div_token = _MathToken(
                    r"\div",
                    min(th.xmin, d_top.xmin, d_bot.xmin),
                    min(th.ymin, d_top.ymin, d_bot.ymin),
                    max(th.xmax, d_top.xmax, d_bot.xmax),
                    max(th.ymax, d_top.ymax, d_bot.ymax),
                    th.cx,
                    th.cy,
                )
                fused.append(div_token)
                used.add(i)
                used.add(dot_above)
                used.add(dot_below)

        # Pass 0.2: Plus-minus \pm (1 VERT_BAR + 2 HORIZ_BARs: one crossing, one underneath)
        for i in range(n):
            if i in used or tokens[i].symbol != "VERT_BAR":
                continue
            tv = tokens[i]
            h_v = max(tv.ymax - tv.ymin, 1.0)
            crossing_bars = []
            below_bars = []
            for j in range(n):
                if j == i or j in used or tokens[j].symbol != "HORIZ_BAR":
                    continue
                th = tokens[j]
                if abs(th.cx - tv.cx) <= 0.5 * max(h_v, th.xmax - th.xmin):
                    if abs(th.cy - tv.cy) <= 0.35 * h_v:
                        crossing_bars.append(j)
                    elif th.cy > tv.cy + 0.3 * h_v:
                        below_bars.append(j)
            if crossing_bars and below_bars:
                j_cross = crossing_bars[0]
                j_below = below_bars[0]
                th_c = tokens[j_cross]
                th_b = tokens[j_below]
                pm_token = _MathToken(
                    r"\pm",
                    min(tv.xmin, th_c.xmin, th_b.xmin),
                    min(tv.ymin, th_c.ymin, th_b.ymin),
                    max(tv.xmax, th_c.xmax, th_b.xmax),
                    max(tv.ymax, th_c.ymax, th_b.ymax),
                    tv.cx,
                    (tv.cy + th_b.cy) / 2.0,
                )
                fused.append(pm_token)
                used.add(i)
                used.add(j_cross)
                used.add(j_below)

        # Pass 0.3: Triangle \Delta (1 base HORIZ_BAR + 2 diagonal lines meeting at apex)
        for i in range(n):
            if i in used or tokens[i].symbol != "HORIZ_BAR":
                continue
            th = tokens[i]
            pos_legs = []
            neg_legs = []
            for j in range(n):
                if j == i or j in used:
                    continue
                if tokens[j].symbol == "DIAG_POS" and tokens[j].cy < th.cy:
                    pos_legs.append(j)
                elif tokens[j].symbol == "DIAG_NEG" and tokens[j].cy < th.cy:
                    neg_legs.append(j)
            if pos_legs and neg_legs:
                j_pos = pos_legs[0]
                j_neg = neg_legs[0]
                tp = tokens[j_pos]
                tn = tokens[j_neg]
                if abs(tp.cx - tn.cx) <= 0.6 * (th.xmax - th.xmin):
                    delta_token = _MathToken(
                        r"\Delta",
                        min(th.xmin, tp.xmin, tn.xmin),
                        min(th.ymin, tp.ymin, tn.ymin),
                        max(th.xmax, tp.xmax, tn.xmax),
                        max(th.ymax, tp.ymax, tn.ymax),
                        th.cx,
                        (th.cy + min(tp.ymin, tn.ymin)) / 2.0,
                    )
                    fused.append(delta_token)
                    used.add(i)
                    used.add(j_pos)
                    used.add(j_neg)

        # Pass 1: Look for 3-stroke symbol Pi (pi): 1 HORIZ_BAR + 2 VERT_BAR / '1' underneath
        for i in range(n):
            if i in used or tokens[i].symbol != "HORIZ_BAR":
                continue
            th = tokens[i]
            cand_legs = []
            for j in range(n):
                if j == i or j in used:
                    continue
                tj = tokens[j]
                if tj.symbol in ("VERT_BAR", "1"):
                    if tj.ymin >= th.ymin - 5.0 and th.xmin - 10.0 <= tj.cx <= th.xmax + 10.0:
                        cand_legs.append(j)
            if len(cand_legs) >= 2:
                j1, j2 = cand_legs[0], cand_legs[1]
                t_j1, t_j2 = tokens[j1], tokens[j2]
                if abs(t_j1.cx - t_j2.cx) > 0.18 * (th.xmax - th.xmin):
                    pi_token = _MathToken(
                        r"\pi",
                        min(th.xmin, t_j1.xmin, t_j2.xmin),
                        min(th.ymin, t_j1.ymin, t_j2.ymin),
                        max(th.xmax, t_j1.xmax, t_j2.xmax),
                        max(th.ymax, t_j1.ymax, t_j2.ymax),
                        (th.cx + t_j1.cx + t_j2.cx) / 3.0,
                        (th.cy + t_j1.cy + t_j2.cy) / 3.0,
                    )
                    fused.append(pi_token)
                    used.add(i)
                    used.add(j1)
                    used.add(j2)

        # Pass 2: Look for 2-stroke or 3-stroke symbols (=, \neq, +, x, \leq, \geq)
        for i in range(n):
            if i in used:
                continue
            t1 = tokens[i]

            # Look for companion stroke
            matched = False
            for j in range(i + 1, n):
                if j in used:
                    continue
                t2 = tokens[j]

                # Check proximity
                dx = abs(t1.cx - t2.cx)
                dy = abs(t1.cy - t2.cy)
                max_dim = max(t1.xmax - t1.xmin, t1.ymax - t1.ymin, t2.xmax - t2.xmin, t2.ymax - t2.ymin)

                # Equal sign '=': two horizontal bars stacked vertically
                if t1.symbol == "HORIZ_BAR" and t2.symbol == "HORIZ_BAR":
                    if dx < max_dim * 0.6 and 0.2 * max_dim < dy < 1.4 * max_dim:
                        # Check for 3rd stroke crossing = (\neq)
                        neq_matched = False
                        for k in range(n):
                            if k not in (i, j) and k not in used:
                                tk = tokens[k]
                                if tk.symbol in ("DIAG_NEG", "DIAG_POS", "VERT_BAR", "/"):
                                    if abs(tk.cx - (t1.cx + t2.cx)/2.0) < max_dim * 0.6 and abs(tk.cy - (t1.cy + t2.cy)/2.0) < max_dim * 0.6:
                                        comb = _MathToken(
                                            r"\neq",
                                            min(t1.xmin, t2.xmin, tk.xmin), min(t1.ymin, t2.ymin, tk.ymin),
                                            max(t1.xmax, t2.xmax, tk.xmax), max(t1.ymax, t2.ymax, tk.ymax),
                                            (t1.cx + t2.cx + tk.cx) / 3.0, (t1.cy + t2.cy + tk.cy) / 3.0,
                                        )
                                        fused.append(comb)
                                        used.add(i)
                                        used.add(j)
                                        used.add(k)
                                        neq_matched = True
                                        matched = True
                                        break
                        if neq_matched:
                            break
                        comb = _MathToken(
                            "=",
                            min(t1.xmin, t2.xmin), min(t1.ymin, t2.ymin),
                            max(t1.xmax, t2.xmax), max(t1.ymax, t2.ymax),
                            (t1.cx + t2.cx) / 2.0, (t1.cy + t2.cy) / 2.0,
                        )
                        fused.append(comb)
                        used.add(i)
                        used.add(j)
                        matched = True
                        break

                # Less-than or equal '\leq': '<' with horizontal bar underneath
                elif (
                    (t1.symbol == "<" and t2.symbol == "HORIZ_BAR" and t2.cy > t1.cy)
                    or (t2.symbol == "<" and t1.symbol == "HORIZ_BAR" and t1.cy > t2.cy)
                ):
                    if dx < max_dim * 0.6 and 0.1 * max_dim < dy < 1.4 * max_dim:
                        comb = _MathToken(
                            r"\leq",
                            min(t1.xmin, t2.xmin), min(t1.ymin, t2.ymin),
                            max(t1.xmax, t2.xmax), max(t1.ymax, t2.ymax),
                            (t1.cx + t2.cx) / 2.0, (t1.cy + t2.cy) / 2.0,
                        )
                        fused.append(comb)
                        used.add(i)
                        used.add(j)
                        matched = True
                        break

                # Greater-than or equal '\geq': '>' with horizontal bar underneath
                elif (
                    (t1.symbol == ">" and t2.symbol == "HORIZ_BAR" and t2.cy > t1.cy)
                    or (t2.symbol == ">" and t1.symbol == "HORIZ_BAR" and t1.cy > t2.cy)
                ):
                    if dx < max_dim * 0.6 and 0.1 * max_dim < dy < 1.4 * max_dim:
                        comb = _MathToken(
                            r"\geq",
                            min(t1.xmin, t2.xmin), min(t1.ymin, t2.ymin),
                            max(t1.xmax, t2.xmax), max(t1.ymax, t2.ymax),
                            (t1.cx + t2.cx) / 2.0, (t1.cy + t2.cy) / 2.0,
                        )
                        fused.append(comb)
                        used.add(i)
                        used.add(j)
                        matched = True
                        break

                # Plus sign '+': one horizontal and one vertical intersecting
                elif (
                    (t1.symbol == "HORIZ_BAR" and t2.symbol == "VERT_BAR")
                    or (t1.symbol == "VERT_BAR" and t2.symbol == "HORIZ_BAR")
                ):
                    if dx < max_dim * 0.45 and dy < max_dim * 0.45:
                        comb = _MathToken(
                            "+",
                            min(t1.xmin, t2.xmin), min(t1.ymin, t2.ymin),
                            max(t1.xmax, t2.xmax), max(t1.ymax, t2.ymax),
                            (t1.cx + t2.cx) / 2.0, (t1.cy + t2.cy) / 2.0,
                        )
                        fused.append(comb)
                        used.add(i)
                        used.add(j)
                        matched = True
                        break

                # Times '\times' or 'x': two opposing diagonals intersecting
                elif (
                    (t1.symbol == "DIAG_POS" and t2.symbol == "DIAG_NEG")
                    or (t1.symbol == "DIAG_NEG" and t2.symbol == "DIAG_POS")
                ):
                    if dx < max_dim * 0.45 and dy < max_dim * 0.45:
                        comb = _MathToken(
                            "x",
                            min(t1.xmin, t2.xmin), min(t1.ymin, t2.ymin),
                            max(t1.xmax, t2.xmax), max(t1.ymax, t2.ymax),
                            (t1.cx + t2.cx) / 2.0, (t1.cy + t2.cy) / 2.0,
                        )
                        fused.append(comb)
                        used.add(i)
                        used.add(j)
                        matched = True
                        break

            if not matched:
                # Standalone stroke conversion
                sym = t1.symbol
                if sym == "HORIZ_BAR":
                    sym = "-"
                elif sym == "VERT_BAR":
                    sym = "1"
                elif sym in ("DIAG_POS", "DIAG_NEG"):
                    sym = "/"
                fused.append(
                    _MathToken(sym, t1.xmin, t1.ymin, t1.xmax, t1.ymax, t1.cx, t1.cy)
                )
                used.add(i)

        return fused

    @classmethod
    def _build_expression(cls, tokens: list[_MathToken]) -> str:
        if not tokens:
            return ""

        # Step 1: Detect Fraction Bar
        # A horizontal '-' or line with tokens both above and below it
        for i, t in enumerate(tokens):
            if t.symbol in ("-", "HORIZ_BAR") and (t.xmax - t.xmin) > 25.0:
                above = [
                    k for k in tokens
                    if k is not t and k.cy < t.ymin and (t.xmin - 15 <= k.cx <= t.xmax + 15)
                ]
                below = [
                    k for k in tokens
                    if k is not t and k.cy > t.ymax and (t.xmin - 15 <= k.cx <= t.xmax + 15)
                ]
                if above and below:
                    # Construct fraction
                    num_str = cls._build_sub_expr(above)
                    den_str = cls._build_sub_expr(below)
                    # Remaining tokens outside fraction
                    frac_token = _MathToken(
                        f"\\frac{{{num_str}}}{{{den_str}}}",
                        t.xmin, min(k.ymin for k in above),
                        t.xmax, max(k.ymax for k in below),
                        t.cx, t.cy,
                    )
                    remaining = [k for k in tokens if k is not t and k not in above and k not in below]
                    remaining.append(frac_token)
                    return cls._build_sub_expr(remaining)

        return cls._build_sub_expr(tokens)

    @classmethod
    def _build_sub_expr(cls, tokens: list[_MathToken]) -> str:
        if not tokens:
            return ""

        # Sort tokens horizontally left-to-right
        sorted_tokens = sorted(tokens, key=lambda t: t.xmin)

        # Baseline height reference
        heights = [t.ymax - t.ymin for t in sorted_tokens if not t.symbol.startswith(r"\frac")]
        base_h = float(np.median(heights)) if heights else 30.0

        parts: list[str] = []
        for i, t in enumerate(sorted_tokens):
            sym = t.symbol

            # Check if this token is a superscript (exponent) of preceding token
            if i > 0:
                prev = sorted_tokens[i - 1]
                prev_h = prev.ymax - prev.ymin
                cur_h = t.ymax - t.ymin
                # Superscript condition: significantly higher center and smaller height
                if (t.cy < prev.cy - 0.28 * prev_h) and (cur_h <= 0.88 * base_h):
                    parts.append(f"^{{{sym}}}")
                    continue
                # Subscript condition: significantly lower center and smaller height
                elif (t.cy > prev.cy + 0.28 * prev_h) and (cur_h <= 0.88 * base_h):
                    parts.append(f"_{{{sym}}}")
                    continue

            # Spacing between operator and operands
            if sym in ("+", "-", "=", r"\times", r"\div", "<", ">", r"\leq", r"\geq", r"\neq", r"\pm"):
                parts.append(f" {sym} ")
            elif parts and parts[-1].endswith(" "):
                parts.append(sym)
            else:
                parts.append(sym)

        expr = "".join(parts).strip()
        # Clean up double spaces
        while "  " in expr:
            expr = expr.replace("  ", " ")
        return expr


# ─── Formula OCR ──────────────────────────────────────────────────────────────

class FormulaOCR:
    """
    Wraps rapid-latex-ocr (or pix2tex) for ONNX-based LaTeX formula recognition.
    Equipped with GeometricMathParser as an offline, zero-dependency fallback.
    """

    def __init__(self) -> None:
        self._engine = None
        self._geometric_parser = GeometricMathParser()

    def _ensure_engine(self) -> bool:
        if not _LATEX_OCR_AVAILABLE:
            return False
        if self._engine is None:
            self._engine = _LatexOCR()
        return True

    def recognize(
        self,
        image_bgr: np.ndarray,
        strokes: Optional[list[Stroke]] = None,
    ) -> OcrResult:
        """
        Recognize a handwritten math formula and return LaTeX code.

        Returns:
            OcrResult with latex string, confidence, and inference_ms.
        """
        t0 = time.perf_counter()

        # 1. Try deep learning ONNX engine if available
        latex_str = ""
        if self._ensure_engine():
            from PIL import Image as _PILImage  # noqa: PLC0415
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil_img = _PILImage.fromarray(rgb)
            try:
                out = self._engine(pil_img)
                if isinstance(out, tuple):
                    latex_str = out[0]
                elif isinstance(out, str):
                    latex_str = out
            except Exception as exc:
                logger.warning("Formula OCR inference failed: %s", exc)

        # 2. Offline geometric fallback when engine is missing or returned empty
        if not latex_str and strokes:
            try:
                latex_str = self._geometric_parser.parse(strokes)
            except Exception as exc:
                logger.debug("Geometric math fallback error: %s", exc)

        # 3. Empty result if nothing detected from engine or geometry
        confidence = 0.88 if (latex_str and latex_str.strip()) else 0.0
        elapsed = (time.perf_counter() - t0) * 1000.0
        return OcrResult(
            latex=latex_str,
            confidence=confidence,
            inference_ms=elapsed,
            is_formula=True,
        )


# ─── LaTeX → PNG Renderer ─────────────────────────────────────────────────────

class LaTeXRenderer:
    """
    Renders a LaTeX math string to a crisp RGBA numpy array using
    matplotlib.mathtext (no TeX installation required).
    """

    DEFAULT_DPI = 300
    DEFAULT_FONTSIZE = 20

    @classmethod
    def render(
        cls,
        latex_str: str,
        dpi: int = 120,
        fontsize: int = 16,
        color: str = "black",
    ):
        return cls.render_to_rgba(latex_str, fontsize=fontsize, dpi=dpi, color=color)

    @staticmethod
    def validate_latex_syntax(latex_str: str) -> bool:
        """
        Validate whether latex_str can be parsed by matplotlib.mathtext
        without rendering a full graphical figure.
        """
        if not latex_str or not latex_str.strip():
            return False
        expr = sanitize_latex_string(latex_str)
        if not expr:
            return False
        try:
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib.mathtext import MathTextParser
            parser = MathTextParser("path")
            parser.parse(f"${expr}$")
            return True
        except Exception:
            return False

    @staticmethod
    def render_to_rgba(
        latex_str: str,
        fontsize: int = DEFAULT_FONTSIZE,
        dpi: int = DEFAULT_DPI,
        color: str = "black",
    ) -> Optional[np.ndarray]:
        """
        Render *latex_str* (optionally wrapped in $ ... $) to an RGBA numpy array.

        Args:
            latex_str: Pure LaTeX math string (e.g. ``r'\\frac{a}{b}'``).
            fontsize:  Font size in points.
            dpi:       Output resolution.
            color:     Foreground color name (default 'black').

        Returns:
            RGBA uint8 numpy array, or None on failure.
        """
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.mathtext import MathTextParser  # noqa: F401

        if not latex_str or not latex_str.strip():
            return None

        # Sanitize and wrap in math delimiters
        expr = sanitize_latex_string(latex_str)
        if not expr:
            return None
        expr = f"${expr}$"

        fig = None
        try:
            fig = plt.figure(figsize=(0.01, 0.01))
            fig.patch.set_alpha(0.0)  # transparent background

            text_obj = fig.text(
                0.5, 0.5, expr,
                fontsize=fontsize,
                color=color,
                ha="center", va="center",
                transform=fig.transFigure,
            )

            # First pass: measure the bounding box
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                        transparent=True, pad_inches=0.1)
            buf.seek(0)

            # Decode PNG → RGBA numpy array
            png_bytes = np.frombuffer(buf.read(), dtype=np.uint8)
            rgba = cv2.imdecode(png_bytes, cv2.IMREAD_UNCHANGED)

            if rgba is None:
                logger.warning("cv2.imdecode returned None for rendered LaTeX.")
                return None

            if rgba.ndim == 3 and rgba.shape[2] == 3:
                # Add alpha channel
                alpha = np.full(rgba.shape[:2], 255, dtype=np.uint8)
                rgba = np.dstack([rgba, alpha])

            return rgba

        except Exception as exc:
            logger.warning("LaTeX render failed for '%s': %s", latex_str, exc)
            return None
        finally:
            if fig is not None:
                plt.close(fig)
            else:
                plt.close("all")

    @staticmethod
    def render_to_png_bytes(
        latex_str: str,
        fontsize: int = DEFAULT_FONTSIZE,
        dpi: int = DEFAULT_DPI,
    ) -> Optional[bytes]:
        """Return PNG bytes (RGBA) for the rendered LaTeX expression."""
        rgba = LaTeXRenderer.render_to_rgba(latex_str, fontsize=fontsize, dpi=dpi)
        if rgba is None:
            return None
        success, buf = cv2.imencode(".png", rgba)
        if success:
            return bytes(buf)
        return None


# ─── Smart Ink Recognizer (Façade) ────────────────────────────────────────────

class SmartInkRecognizer:
    """
    Unified entry point: decides whether ink is handwriting or math formula,
    runs the appropriate OCR, and optionally renders LaTeX to vector image.

    Heuristic for formula detection:
      - Stroke group contains mathematical operators (×, ÷, integral-like arcs)
      - OR the bounding box aspect ratio is wide and stroke count is ≥ 2
      - OR caller passes force_formula=True
    """

    def __init__(self) -> None:
        self._renderer = StrokeRenderer()
        self._handwriting_ocr = HandwritingOCR()
        self._formula_ocr = FormulaOCR()
        self._latex_renderer = LaTeXRenderer()

    def recognize_strokes(
        self,
        strokes: list[Stroke],
        force_formula: bool = False,
        render_latex: bool = True,
    ) -> OcrResult:
        """
        Full pipeline: Stroke list → OcrResult.

        Args:
            strokes:       List of ink strokes to recognize.
            force_formula: Skip heuristic and always use formula OCR.
            render_latex:  If formula detected, render LaTeX → RGBA image.

        Returns:
            OcrResult with text/latex/render_image_rgba.
        """
        image = self._renderer.render(strokes)
        is_formula = force_formula or self._detect_formula_heuristic(strokes, image)

        if is_formula:
            result = self._formula_ocr.recognize(image, strokes=strokes)
            if result.latex and render_latex:
                result.render_image_rgba = self._latex_renderer.render_to_rgba(result.latex)
        else:
            result = self._handwriting_ocr.recognize(image)

        result.is_formula = is_formula
        return result

    def recognize_image(
        self,
        image_bgr: np.ndarray,
        force_formula: bool = False,
        render_latex: bool = True,
    ) -> OcrResult:
        """Same pipeline but starts from a pre-rendered image."""
        is_formula = force_formula
        if is_formula:
            result = self._formula_ocr.recognize(image_bgr)
            if result.latex and render_latex:
                result.render_image_rgba = self._latex_renderer.render_to_rgba(result.latex)
        else:
            result = self._handwriting_ocr.recognize(image_bgr)
        result.is_formula = is_formula
        return result

    @staticmethod
    def _detect_formula_heuristic(strokes: list[Stroke], image: np.ndarray) -> bool:
        """
        Detect if the ink is likely a math formula by simple heuristics:
        1. Multiple strokes with small vertical variance (horizontal writing)
        2. High black-pixel density suggests compact notation
        3. Bounding box is wide (aspect ratio > 3:1) with ≥ 3 strokes
        """
        if len(strokes) >= 5:
            # Many small strokes → likely a complex formula
            avg_len = np.mean([len(s.points) for s in strokes])
            if avg_len < 8:
                return True

        # Check bounding box aspect ratio
        h, w = image.shape[:2]
        if h > 0 and w / h > 3.0 and len(strokes) >= 3:
            return True

        return False


# Backward-compatible alias
MathTextRenderer = LaTeXRenderer
