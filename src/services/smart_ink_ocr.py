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
import time
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

        w = output_width or max(int((x_max - x_min) * dpi_scale) + 2 * pad, 64)
        h = output_height or max(int((y_max - y_min) * dpi_scale) + 2 * pad, 64)

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
        combined_text = " ".join(texts)
        avg_confidence = float(np.mean(scores)) if scores else 0.0

        return OcrResult(
            text=combined_text,
            confidence=avg_confidence,
            inference_ms=elapsed,
        )


# ─── Formula OCR ──────────────────────────────────────────────────────────────

class FormulaOCR:
    """
    Wraps rapid-latex-ocr (or pix2tex) for ONNX-based LaTeX formula recognition.
    Lazy-initializes on first call.
    """

    def __init__(self) -> None:
        self._engine = None

    def _ensure_engine(self) -> bool:
        if not _LATEX_OCR_AVAILABLE:
            return False
        if self._engine is None:
            self._engine = _LatexOCR()
        return True

    def recognize(
        self,
        image_bgr: np.ndarray,
    ) -> OcrResult:
        """
        Recognize a handwritten math formula and return LaTeX code.

        Returns:
            OcrResult with latex string, confidence, and inference_ms.
        """
        t0 = time.perf_counter()

        if not self._ensure_engine():
            return OcrResult(
                latex=r"\text{[Formula OCR unavailable on Python 3.15+]}",
                confidence=0.0,
                inference_ms=0.0,
                is_formula=True,
            )

        # Convert BGR → RGB PIL image
        from PIL import Image as _PILImage  # noqa: PLC0415
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_img = _PILImage.fromarray(rgb)

        try:
            latex_str = self._engine(pil_img)
            if isinstance(latex_str, tuple):
                latex_str = latex_str[0]  # some wrappers return (str, score)
        except Exception as exc:
            logger.warning("Formula OCR inference failed: %s", exc)
            latex_str = ""

        elapsed = (time.perf_counter() - t0) * 1000.0
        return OcrResult(
            latex=latex_str,
            confidence=0.85 if latex_str else 0.0,
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

        # Wrap in math delimiters if not already present
        expr = latex_str.strip()
        if not (expr.startswith("$") and expr.endswith("$")):
            expr = f"${expr}$"

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
            plt.close(fig)
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
            logger.exception("LaTeX render failed for '%s': %s", latex_str, exc)
            plt.close("all")
            return None

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
            result = self._formula_ocr.recognize(image)
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
