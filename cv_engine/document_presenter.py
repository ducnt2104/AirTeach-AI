#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Native PDF & PowerPoint (PPT / PPTX) Presentation & Annotation Export Engine
(`cv_engine/document_presenter.py`) for AirTeach AI (DNT - Victer).

Capabilities:
1. High-DPI PDF rendering via PyMuPDF (`pymupdf` / `fitz`) at `2.0x` scale with
   thread-safe LRU page cache and async background adjacent-page pre-rendering.
2. Three-tier PowerPoint (`.pptx` / `.ppt`) -> cached PDF pipeline keyed by SHA-256
   in `%APPDATA%/AirTeachAI/doc_cache` (re-opening takes < 0.2s):
   - Tier 1: Headless LibreOffice (`soffice --headless --convert-to pdf`)
   - Tier 2: Windows MS PowerPoint COM automation (when MS Office is installed)
   - Tier 3: Pure-Python offline renderer (`python-pptx` + `Pillow` + `PyMuPDF`)
     so PPTX/PPT import never fails on any host machine.
3. Non-destructive Annotated PDF Export burning per-slide Pen, Highlighter,
   AI-beautified vector shapes, and Math/LaTeX annotations cleanly onto pages.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import html as html_lib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

fitz: Any = None
python_docx: Any = None
Presentation: Any = None
MSO_SHAPE_TYPE: Any = None
PP_ALIGN: Any = None

HAS_PYTHON_DOCX = importlib.util.find_spec("docx") is not None
HAS_PYTHON_PPTX = importlib.util.find_spec("pptx") is not None


def _ensure_fitz() -> Any:
    """Lazy-load PyMuPDF (`fitz`) on first document use so app startup stays < 1.5s."""
    global fitz
    if fitz is not None:
        return fitz
    try:
        import pymupdf as _fitz  # type: ignore
    except ImportError:  # pragma: no cover
        import fitz as _fitz  # type: ignore
    fitz = _fitz
    return fitz


def _ensure_python_docx() -> bool:
    """Lazy-load `python-docx` on first Word fallback parse."""
    global python_docx, HAS_PYTHON_DOCX
    if python_docx is not None:
        return True
    if not HAS_PYTHON_DOCX:
        return False
    try:
        import docx as _docx  # type: ignore

        python_docx = _docx
        HAS_PYTHON_DOCX = True
        return True
    except Exception:  # pragma: no cover
        python_docx = None
        HAS_PYTHON_DOCX = False
        return False


def _ensure_python_pptx() -> bool:
    """Lazy-load `python-pptx` on first PowerPoint parse/fallback render."""
    global Presentation, MSO_SHAPE_TYPE, PP_ALIGN, HAS_PYTHON_PPTX
    if Presentation is not None:
        return True
    if not HAS_PYTHON_PPTX:
        return False
    try:
        from pptx import Presentation as _Pres  # type: ignore
        from pptx.enum.shapes import MSO_SHAPE_TYPE as _ShapeType  # type: ignore
        from pptx.enum.text import PP_ALIGN as _PpAlign  # type: ignore

        Presentation = _Pres
        MSO_SHAPE_TYPE = _ShapeType
        PP_ALIGN = _PpAlign
        HAS_PYTHON_PPTX = True
        return True
    except Exception:  # pragma: no cover
        Presentation = None
        MSO_SHAPE_TYPE = None
        PP_ALIGN = None
        HAS_PYTHON_PPTX = False
        return False


def warmup_document_engines() -> Dict[str, bool]:
    """Warm document parsing modules safely in background thread."""
    fitz_ok = False
    try:
        _ensure_fitz()
        fitz_ok = True
    except Exception:
        fitz_ok = False
    return {
        "pymupdf": fitz_ok,
        "python_docx": _ensure_python_docx(),
        "python_pptx": _ensure_python_pptx(),
    }


from PIL import Image, ImageDraw, ImageFont

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QTransform

from cv_engine.app_paths import get_app_logger, get_doc_cache_dir, resource_path
from cv_engine.file_attachment_parser import normalize_vietnamese_text

_LOGGER = get_app_logger("document_presenter")

SUPPORTED_DOC_EXTENSIONS = {".pdf", ".pptx", ".ppt", ".ppsx", ".potx", ".docx", ".doc"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".svg"}


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a document using 1 MB streaming chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _find_soffice_executable() -> Optional[str]:
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    if sys.platform == "win32":
        candidates = (
            Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
            Path(r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"),
        )
    elif sys.platform == "darwin":
        candidates = (Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),)
    else:
        candidates = (Path("/usr/bin/soffice"), Path("/usr/local/bin/soffice"))
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return str(cand)
    return None


def _has_windows_powerpoint_com() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"PowerPoint.Application"):
            return True
    except OSError:
        return False


def _has_windows_word_com() -> bool:
    if sys.platform != "win32":
        return False
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Word.Application"):
            return True
    except OSError:
        return False


def _subprocess_creationflags() -> int:
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    return 0


def _convert_via_libreoffice(src_path: Path, entry_dir: Path, pdf_out: Path) -> bool:
    soffice = _find_soffice_executable()
    if not soffice:
        return False
    try:
        work_tmp = entry_dir / "_lo_tmp"
        work_tmp.mkdir(parents=True, exist_ok=True)
        cmd = [
            soffice,
            "--headless",
            "--norestore",
            "--nolockcheck",
            "--convert-to",
            "pdf",
            "--outdir",
            str(work_tmp),
            str(src_path),
        ]
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            creationflags=_subprocess_creationflags(),
            check=False,
        )
        produced = work_tmp / f"{src_path.stem}.pdf"
        if not produced.exists():
            pdfs = list(work_tmp.glob("*.pdf"))
            if pdfs:
                produced = pdfs[0]
        if res.returncode == 0 and produced.exists() and produced.stat().st_size > 200:
            shutil.move(str(produced), str(pdf_out))
            shutil.rmtree(work_tmp, ignore_errors=True)
            return True
        shutil.rmtree(work_tmp, ignore_errors=True)
    except Exception as exc:
        _LOGGER.warning("LibreOffice headless conversion failed (%s): %s", src_path.name, exc)
    return False


def _convert_via_powerpoint_com(src_path: Path, pdf_out: Path) -> bool:
    if not _has_windows_powerpoint_com():
        return False
    # Out-of-process PowerShell COM automation isolates Office RPC/SEH faults
    # completely from the PySide6 Qt GUI thread apartment.
    try:
        src_esc = str(src_path.resolve()).replace("'", "''")
        out_esc = str(pdf_out.resolve()).replace("'", "''")
        ps_script = (
            "$ErrorActionPreference='Stop';"
            "$pp = New-Object -ComObject PowerPoint.Application;"
            "$pres = $null;"
            "try {"
            f"  $pres = $pp.Presentations.Open('{src_esc}', $true, $false, $false);"
            f"  $pres.SaveAs('{out_esc}', 32);"
            "} finally {"
            "  if ($pres) { try { $pres.Close() } catch {} };"
            "  if ($pp) { try { $pp.Quit() } catch {} };"
            "}"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            creationflags=_subprocess_creationflags(),
            check=False,
        )
        if res.returncode == 0 and pdf_out.exists() and pdf_out.stat().st_size > 200:
            return True
    except Exception as exc:
        _LOGGER.warning("PowerPoint COM conversion failed (%s): %s", src_path.name, exc)
    return False


_FONT_CACHE: Dict[Tuple[str, int], Any] = {}


def _load_pil_font(size_px: int, bold: bool = False, italic: bool = False) -> Any:
    sz = max(10, min(140, int(size_px)))
    variant = "bi" if (bold and italic) else ("b" if bold else ("i" if italic else "r"))
    key = (variant, sz)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached

    bundled_fonts_dir = resource_path("assets/fonts")
    bundled_map = {
        "r": (bundled_fonts_dir / "NotoSans-Regular.ttf",),
        "b": (bundled_fonts_dir / "NotoSans-Bold.ttf", bundled_fonts_dir / "NotoSans-Regular.ttf"),
        "i": (bundled_fonts_dir / "NotoSans-Italic.ttf", bundled_fonts_dir / "NotoSans-Regular.ttf"),
        "bi": (bundled_fonts_dir / "NotoSans-BoldItalic.ttf", bundled_fonts_dir / "NotoSans-Bold.ttf"),
    }
    bundled_paths = list(bundled_map.get(variant, bundled_map["r"]))

    if sys.platform == "win32":
        fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        candidates_map = {
            "r": ("segoeui.ttf", "arial.ttf", "tahoma.ttf", "calibri.ttf"),
            "b": ("segoeuib.ttf", "arialbd.ttf", "tahomabd.ttf", "calibrib.ttf"),
            "i": ("segoeuii.ttf", "ariali.ttf", "calibrii.ttf"),
            "bi": ("segoeuiz.ttf", "arialbi.ttf", "calibriz.ttf"),
        }
        names = candidates_map.get(variant, candidates_map["r"]) + candidates_map["r"]
        paths = [fonts_dir / n for n in names] + bundled_paths
    else:
        candidates_map = {
            "r": ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            "b": ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
            "i": ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",),
            "bi": ("/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",),
        }
        paths = bundled_paths + [Path(p) for p in candidates_map.get(variant, candidates_map["r"])]

    font = None
    for fp in paths:
        if fp.exists():
            try:
                font = ImageFont.truetype(str(fp), sz)
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.load_default(size=sz)
        except TypeError:
            font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _safe_rgb_from_color(color_obj: Any) -> Optional[Tuple[int, int, int]]:
    if color_obj is None:
        return None
    try:
        rgb = getattr(color_obj, "rgb", None)
        if rgb is not None:
            return (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    except Exception as exc:
        _LOGGER.debug("RGB color read fallback: %s", exc)
    return None


def _safe_fill_rgb(fill_obj: Any) -> Optional[Tuple[int, int, int]]:
    if fill_obj is None:
        return None
    try:
        if fill_obj.type is None:
            return None
        return _safe_rgb_from_color(fill_obj.fore_color)
    except Exception:
        return None


def _wrap_text_to_width(draw: ImageDraw.ImageDraw, text: str, font: Any, max_w: float) -> List[str]:
    if not text:
        return [""]
    limit = max(40.0, float(max_w))
    lines: List[str] = []
    for raw_line in text.splitlines() or [""]:
        words = raw_line.split(" ")
        cur = ""
        for word in words:
            trial = word if not cur else f"{cur} {word}"
            try:
                tw = float(draw.textlength(trial, font=font))
            except Exception:
                tw = float(len(trial) * 9)
            if tw <= limit or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def _extract_pptx_slide_texts(src_path: Path) -> List[str]:
    if not _ensure_python_pptx() or Presentation is None:
        return []
    try:
        prs = Presentation(io.BytesIO(src_path.read_bytes()))
        texts: List[str] = []

        def _collect_shape(shape: Any, parts: List[str]) -> None:
            if MSO_SHAPE_TYPE is not None and getattr(shape, "shape_type", None) == MSO_SHAPE_TYPE.GROUP:
                for sub in getattr(shape, "shapes", ()):
                    _collect_shape(sub, parts)
                return
            if getattr(shape, "has_text_frame", False):
                txt = normalize_vietnamese_text(shape.text_frame.text).strip()
                if txt:
                    parts.append(txt)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    row_txt = " | ".join(
                        normalize_vietnamese_text(c.text).strip()
                        for c in row.cells
                        if normalize_vietnamese_text(c.text).strip()
                    )
                    if row_txt:
                        parts.append(row_txt)

        for slide in prs.slides:
            parts: List[str] = []
            for shape in slide.shapes:
                _collect_shape(shape, parts)
            texts.append(normalize_vietnamese_text("\n".join(parts)).strip())
        return texts
    except Exception:
        return []


def _render_pptx_pure_python_to_pdf(
    src_path: Path,
    pdf_out: Path,
    max_slides: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    _ensure_fitz()
    if not _ensure_python_pptx() or Presentation is None:
        return False, []
    try:
        prs = Presentation(io.BytesIO(src_path.read_bytes()))
    except Exception as exc:
        _LOGGER.warning("python-pptx failed to open %s: %s", src_path.name, exc)
        return False, []

    sw_emu = max(1, int(prs.slide_width or 12192000))
    sh_emu = max(1, int(prs.slide_height or 6858000))
    canvas_w = 1920
    canvas_h = max(720, min(1920, int(round(canvas_w * (sh_emu / float(sw_emu))))))
    sx = canvas_w / float(sw_emu)
    sy = canvas_h / float(sh_emu)

    def _px_x(emu: Optional[int]) -> int:
        return int(round(float(emu or 0) * sx))

    def _px_y(emu: Optional[int]) -> int:
        return int(round(float(emu or 0) * sy))

    def _detect_slide_bg(slide: Any) -> Tuple[int, int, int]:
        for container in (slide, getattr(slide, "slide_layout", None), getattr(getattr(slide, "slide_layout", None), "slide_master", None)):
            if container is None:
                continue
            try:
                bg = container.background
                rgb = _safe_fill_rgb(bg.fill)
                if rgb is not None:
                    return rgb
            except Exception:
                continue
        return (248, 250, 252)

    def _draw_text_frame(
        draw: ImageDraw.ImageDraw,
        tf: Any,
        box: Tuple[int, int, int, int],
        default_fg: Tuple[int, int, int],
    ) -> str:
        bx0, by0, bx1, by1 = box
        pad_x = max(8, int(round((bx1 - bx0) * 0.025)))
        pad_y = max(6, int(round((by1 - by0) * 0.04)))
        avail_w = max(40, (bx1 - bx0) - 2 * pad_x)
        cur_y = by0 + pad_y
        collected: List[str] = []

        for p_idx, para in enumerate(tf.paragraphs):
            raw_p = normalize_vietnamese_text(para.text or "")
            if raw_p.strip():
                collected.append(raw_p.strip())
            level = int(getattr(para, "level", 0) or 0)
            indent = level * 26
            bullet_prefix = "• " if (level > 0 or (p_idx > 0 and raw_p.strip() and len(tf.paragraphs) > 2)) else ""

            # Determine representative run formatting
            pt_size = 24.0
            is_bold = False
            is_italic = False
            fg_col = default_fg

            runs = list(getattr(para, "runs", ()) or ())
            if runs:
                for r in runs:
                    f = r.font
                    if f.size is not None:
                        pt_size = float(f.size.pt)
                        break
                is_bold = any(bool(r.font.bold) for r in runs)
                is_italic = any(bool(r.font.italic) for r in runs)
                for r in runs:
                    c = _safe_rgb_from_color(r.font.color)
                    if c is not None:
                        fg_col = c
                        break
            elif getattr(para, "font", None) is not None:
                if para.font.size is not None:
                    pt_size = float(para.font.size.pt)
                is_bold = bool(para.font.bold)
                is_italic = bool(para.font.italic)
                c = _safe_rgb_from_color(para.font.color)
                if c is not None:
                    fg_col = c

            # 1920px slide canvas corresponds roughly to 13.33in (960pt) -> 2.0x px per pt
            font_px = max(16, min(110, int(round(pt_size * (canvas_w / 960.0)))))
            pil_font = _load_pil_font(font_px, bold=is_bold, italic=is_italic)
            line_h = int(round(font_px * 1.28))

            display_txt = f"{bullet_prefix}{raw_p}" if raw_p.strip() else ""
            wrapped = _wrap_text_to_width(draw, display_txt, pil_font, avail_w - indent)
            align = getattr(para, "alignment", None)

            for line in wrapped:
                if cur_y + line_h > by1 + 8:
                    break
                tx = bx0 + pad_x + indent
                if PP_ALIGN is not None and align == PP_ALIGN.CENTER:
                    try:
                        lw = float(draw.textlength(line, font=pil_font))
                        tx = max(bx0 + pad_x, int(bx0 + ((bx1 - bx0) - lw) * 0.5))
                    except Exception as exc:
                        _LOGGER.debug("Center alignment width fallback: %s", exc)
                elif PP_ALIGN is not None and align == PP_ALIGN.RIGHT:
                    try:
                        lw = float(draw.textlength(line, font=pil_font))
                        tx = max(bx0 + pad_x, int(bx1 - pad_x - lw))
                    except Exception as exc:
                        _LOGGER.debug("Right alignment width fallback: %s", exc)
                if line:
                    draw.text((tx, cur_y), line, fill=fg_col, font=pil_font)
                cur_y += line_h
            cur_y += max(4, int(round(font_px * 0.22)))
        return "\n".join(collected)

    def _render_shape(
        img: Image.Image,
        draw: ImageDraw.ImageDraw,
        shape: Any,
        default_fg: Tuple[int, int, int],
        slide_texts_acc: List[str],
    ) -> None:
        if MSO_SHAPE_TYPE is not None and getattr(shape, "shape_type", None) == MSO_SHAPE_TYPE.GROUP:
            for sub in getattr(shape, "shapes", ()):
                _render_shape(img, draw, sub, default_fg, slide_texts_acc)
            return

        x0 = max(0, _px_x(getattr(shape, "left", 0)))
        y0 = max(0, _px_y(getattr(shape, "top", 0)))
        w_px = max(1, _px_x(getattr(shape, "width", 0)))
        h_px = max(1, _px_y(getattr(shape, "height", 0)))
        x1 = min(canvas_w, x0 + w_px)
        y1 = min(canvas_h, y0 + h_px)

        # Embedded picture
        is_picture = (
            MSO_SHAPE_TYPE is not None and getattr(shape, "shape_type", None) == MSO_SHAPE_TYPE.PICTURE
        )
        if is_picture:
            try:
                blob = shape.image.blob
                pic = Image.open(io.BytesIO(blob))
                if pic.mode not in ("RGB", "RGBA"):
                    pic = pic.convert("RGBA")
                pic_resized = pic.resize((max(1, x1 - x0), max(1, y1 - y0)), Image.Resampling.LANCZOS)
                if pic_resized.mode == "RGBA":
                    img.paste(pic_resized, (x0, y0), pic_resized)
                else:
                    img.paste(pic_resized, (x0, y0))
                return
            except Exception as exc:
                _LOGGER.debug("PPTX image paste fallback: %s", exc)

        # Shape fill & border
        fill_rgb = None
        try:
            if hasattr(shape, "fill"):
                fill_rgb = _safe_fill_rgb(shape.fill)
        except Exception:
            fill_rgb = None

        line_rgb = None
        line_w = 2
        try:
            if hasattr(shape, "line"):
                line_rgb = _safe_rgb_from_color(shape.line.color)
                if shape.line.width is not None:
                    line_w = max(1, min(10, int(round(shape.line.width.pt * 1.8))))
        except Exception:
            line_rgb = None

        if fill_rgb is not None or line_rgb is not None:
            auto_name = str(getattr(shape, "auto_shape_type", "") or "").upper()
            if "OVAL" in auto_name:
                draw.ellipse([x0, y0, x1, y1], fill=fill_rgb, outline=line_rgb, width=line_w)
            elif "ROUNDED" in auto_name:
                draw.rounded_rectangle([x0, y0, x1, y1], radius=min(24, w_px // 8, h_px // 8), fill=fill_rgb, outline=line_rgb, width=line_w)
            else:
                draw.rectangle([x0, y0, x1, y1], fill=fill_rgb, outline=line_rgb, width=line_w)

        # Table rendering
        if getattr(shape, "has_table", False):
            try:
                tbl = shape.table
                n_rows = max(1, len(tbl.rows))
                n_cols = max(1, len(tbl.columns))
                col_ws = [max(20, _px_x(c.width)) for c in tbl.columns]
                row_hs = [max(24, _px_y(r.height)) for r in tbl.rows]
                tot_cw = max(1, sum(col_ws))
                tot_rh = max(1, sum(row_hs))
                col_ws = [max(20, int(round(cw * w_px / tot_cw))) for cw in col_ws]
                row_hs = [max(24, int(round(rh * h_px / tot_rh))) for rh in row_hs]

                cy = y0
                for r_idx, row in enumerate(tbl.rows):
                    rh = row_hs[r_idx]
                    cx = x0
                    for c_idx, cell in enumerate(row.cells):
                        cw = col_ws[c_idx]
                        c_fill = _safe_fill_rgb(getattr(cell, "fill", None))
                        if c_fill is None:
                            c_fill = (226, 232, 240) if r_idx == 0 else (255, 255, 255)
                        draw.rectangle([cx, cy, cx + cw, cy + rh], fill=c_fill, outline=(148, 163, 184), width=2)
                        cell_luma = 0.299 * c_fill[0] + 0.587 * c_fill[1] + 0.114 * c_fill[2]
                        cell_fg = (15, 23, 42) if cell_luma > 128 else (248, 250, 252)
                        if getattr(cell, "text_frame", None) is not None:
                            txt = _draw_text_frame(draw, cell.text_frame, (cx + 2, cy + 2, cx + cw - 2, cy + rh - 2), cell_fg)
                            if txt:
                                slide_texts_acc.append(txt)
                        cx += cw
                    cy += rh
                return
            except Exception as exc:
                _LOGGER.debug("PPTX table render fallback: %s", exc)

        # Text frame rendering
        if getattr(shape, "has_text_frame", False):
            txt_fg = default_fg
            if fill_rgb is not None:
                luma = 0.299 * fill_rgb[0] + 0.587 * fill_rgb[1] + 0.114 * fill_rgb[2]
                txt_fg = (15, 23, 42) if luma > 135 else (248, 250, 252)
            txt = _draw_text_frame(draw, shape.text_frame, (x0, y0, x1, y1), txt_fg)
            if txt:
                slide_texts_acc.append(txt)

    slide_texts: List[str] = []
    out_pdf = fitz.open()
    try:
        slides_iter = list(prs.slides)
        if max_slides is not None and max_slides > 0:
            slides_iter = slides_iter[: int(max_slides)]
        if not slides_iter:
            blank = Image.new("RGB", (canvas_w, canvas_h), (248, 250, 252))
            buf = io.BytesIO()
            blank.save(buf, format="PNG")
            p = out_pdf.new_page(width=float(canvas_w), height=float(canvas_h))
            p.insert_image(p.rect, stream=buf.getvalue())
            slide_texts.append("")
        else:
            for s_idx, slide in enumerate(slides_iter):
                bg_rgb = _detect_slide_bg(slide)
                bg_luma = 0.299 * bg_rgb[0] + 0.587 * bg_rgb[1] + 0.114 * bg_rgb[2]
                default_fg = (15, 23, 42) if bg_luma > 130 else (248, 250, 252)
                img = Image.new("RGB", (canvas_w, canvas_h), bg_rgb)
                draw = ImageDraw.Draw(img)
                # Subtle top accent bar
                draw.rectangle([0, 0, canvas_w, 8], fill=(79, 70, 229))

                s_acc: List[str] = []
                # Render non-placeholder background shapes from slide layout for higher visual fidelity
                layout = getattr(slide, "slide_layout", None)
                if layout is not None:
                    for lshape in getattr(layout, "shapes", ()):
                        if getattr(lshape, "is_placeholder", False):
                            continue
                        _render_shape(img, draw, lshape, default_fg, [])
                for shape in slide.shapes:
                    _render_shape(img, draw, shape, default_fg, s_acc)

                # Slide number watermark in bottom-right corner
                num_font = _load_pil_font(18, bold=True)
                draw.text((canvas_w - 92, canvas_h - 32), f"{s_idx + 1}/{len(slides_iter)}", fill=(100, 116, 139), font=num_font)

                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=True)
                page = out_pdf.new_page(width=float(canvas_w), height=float(canvas_h))
                page.insert_image(page.rect, stream=buf.getvalue())
                slide_texts.append(normalize_vietnamese_text("\n".join(s_acc)).strip())

        out_pdf.save(str(pdf_out), deflate=True)
    finally:
        out_pdf.close()
    return pdf_out.exists() and pdf_out.stat().st_size > 200, slide_texts


def _render_legacy_ppt_fallback_to_pdf(src_path: Path, pdf_out: Path) -> Tuple[bool, List[str]]:
    """Pure-Python fallback for legacy binary `.ppt` when LibreOffice / MS PowerPoint are unavailable."""
    raw = src_path.read_bytes()
    utf16_chunks = re.findall(rb"(?:[\x20-\x7e\u00a0-\u024f\u1e00-\u1eff]\x00){5,}", raw)
    extracted: List[str] = []
    for ch in utf16_chunks:
        try:
            txt = normalize_vietnamese_text(ch.decode("utf-16le", errors="ignore")).strip()
            if txt and len(txt) >= 5 and txt not in extracted:
                extracted.append(txt)
        except Exception:
            continue
    if not extracted:
        ascii_chunks = re.findall(rb"[\x20-\x7e]{8,}", raw)
        extracted = [normalize_vietnamese_text(c.decode("ascii", errors="ignore")).strip() for c in ascii_chunks[:40]]

    chunks_per_slide = 8
    groups = [extracted[i : i + chunks_per_slide] for i in range(0, max(1, len(extracted)), chunks_per_slide)]
    _ensure_fitz()
    out_pdf = fitz.open()
    slide_texts: List[str] = []
    title_font = _load_pil_font(42, bold=True)
    body_font = _load_pil_font(26, bold=False)
    try:
        for idx, grp in enumerate(groups):
            img = Image.new("RGB", (1920, 1080), (248, 250, 252))
            draw = ImageDraw.Draw(img)
            draw.rectangle([0, 0, 1920, 12], fill=(79, 70, 229))
            draw.text((90, 64), f"{src_path.name} — Slide {idx + 1}", fill=(30, 41, 59), font=title_font)
            y = 160
            for ln in grp:
                for wln in _wrap_text_to_width(draw, f"• {ln}", body_font, 1740):
                    if y < 980:
                        draw.text((90, y), wln, fill=(51, 65, 85), font=body_font)
                        y += 40
                y += 12
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            p = out_pdf.new_page(width=1920.0, height=1080.0)
            p.insert_image(p.rect, stream=buf.getvalue())
            slide_texts.append("\n".join(grp))
        out_pdf.save(str(pdf_out), deflate=True)
    finally:
        out_pdf.close()
    return pdf_out.exists() and pdf_out.stat().st_size > 200, slide_texts


def convert_presentation_to_cached_pdf(
    src_path: Path | str,
    cache_dir: Optional[Path] = None,
) -> Tuple[Path, List[str], str]:
    """
    Convert `.pptx` / `.ppt` to a high-fidelity cached PDF keyed by SHA-256 inside
    `%APPDATA%/AirTeachAI/doc_cache`.
    Returns `(pdf_path, slide_texts, backend_used)`.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Không tìm thấy file bài giảng: {src}")
    _ensure_fitz()

    doc_hash = compute_file_sha256(src)
    root_cache = Path(cache_dir) if cache_dir is not None else get_doc_cache_dir()
    entry_dir = root_cache / doc_hash[:24]
    entry_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = entry_dir / "converted.pdf"
    manifest_path = entry_dir / "manifest.json"

    # Fast SHA-256 Cache Hit (< 0.2s)
    if pdf_out.exists() and pdf_out.stat().st_size > 200 and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("sha256") == doc_hash:
                probe = fitz.open(str(pdf_out))
                cnt = probe.page_count
                probe.close()
                if cnt >= 1:
                    texts = [normalize_vietnamese_text(str(t)) for t in (manifest.get("slide_texts") or [])]
                    return pdf_out, texts, "CACHE_HIT"
        except Exception as exc:
            _LOGGER.debug("PPTX PDF cache probe miss (%s): %s", src.name, exc)

    backend = ""
    slide_texts: List[str] = []

    if _convert_via_libreoffice(src, entry_dir, pdf_out):
        backend = "LIBREOFFICE"
    elif _convert_via_powerpoint_com(src, pdf_out):
        backend = "POWERPOINT_COM"
    elif src.suffix.lower() in (".pptx", ".ppsx", ".potx"):
        ok, slide_texts = _render_pptx_pure_python_to_pdf(src, pdf_out)
        if ok:
            backend = "PYTHON_PPTX"
    elif src.suffix.lower() == ".ppt":
        ok, slide_texts = _render_legacy_ppt_fallback_to_pdf(src, pdf_out)
        if ok:
            backend = "LEGACY_PPT_FALLBACK"

    if not backend or not pdf_out.exists():
        raise RuntimeError(
            f"Không thể chuyển đổi '{src.name}' sang PDF (LibreOffice, PowerPoint COM và Python-PPTX đều thất bại)."
        )

    # Ensure slide_texts matches page count
    doc = fitz.open(str(pdf_out))
    try:
        n_pages = doc.page_count
        if not slide_texts:
            pptx_texts = _extract_pptx_slide_texts(src) if src.suffix.lower() in (".pptx", ".ppsx", ".potx") else []
            for idx in range(n_pages):
                pdf_txt = normalize_vietnamese_text(doc.load_page(idx).get_text("text")).strip()
                pt_txt = normalize_vietnamese_text(pptx_texts[idx]).strip() if idx < len(pptx_texts) else ""
                slide_texts.append(pdf_txt or pt_txt)
        else:
            slide_texts = [normalize_vietnamese_text(t).strip() for t in slide_texts]
    finally:
        doc.close()

    manifest_path.write_text(
        json.dumps(
            {
                "sha256": doc_hash,
                "source_name": src.name,
                "backend": backend,
                "page_count": len(slide_texts),
                "slide_texts": slide_texts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return pdf_out, slide_texts, backend


def _convert_via_word_com(src_path: Path, pdf_out: Path) -> bool:
    if not _has_windows_word_com():
        return False
    # Out-of-process PowerShell COM automation prevents RPC_E_DISCONNECTED SEH faults inside Qt.
    try:
        src_esc = str(src_path.resolve()).replace("'", "''")
        out_esc = str(pdf_out.resolve()).replace("'", "''")
        ps_script = (
            "$ErrorActionPreference='Stop';"
            "$w = New-Object -ComObject Word.Application;"
            "$w.Visible = $false;"
            "$d = $null;"
            "try {"
            f"  $d = $w.Documents.Open('{src_esc}', $false, $true, $false);"
            f"  $d.ExportAsFixedFormat('{out_esc}', 17);"
            "} finally {"
            "  if ($d) { try { $d.Close($false) } catch {} };"
            "  if ($w) { try { $w.Quit($false) } catch {} };"
            "}"
        )
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=45,
            creationflags=_subprocess_creationflags(),
            check=False,
        )
        if res.returncode == 0 and pdf_out.exists() and pdf_out.stat().st_size > 200:
            return True
    except Exception as exc:
        _LOGGER.warning("Word COM conversion failed (%s): %s", src_path.name, exc)
    return False


def _render_story_html_to_pdf_bytes(html_str: str, assets_dir: Optional[Path] = None) -> bytes:
    _ensure_fitz()
    bio = io.BytesIO()
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (42, 42, -42, -42)
    fonts_dir = resource_path("assets/fonts")
    noto_reg = fonts_dir / "NotoSans-Regular.ttf"
    noto_bold = fonts_dir / "NotoSans-Bold.ttf"
    archive = fitz.Archive()
    has_archive = False
    if assets_dir is not None and Path(assets_dir).exists():
        archive.add(str(assets_dir))
        has_archive = True
    if noto_reg.exists():
        archive.add(str(fonts_dir))
        has_archive = True
        font_rules = [
            "@font-face { font-family: 'Noto Sans'; src: url('NotoSans-Regular.ttf'); font-weight: 400; font-style: normal; }"
        ]
        if noto_bold.exists():
            font_rules.append(
                "@font-face { font-family: 'Noto Sans'; src: url('NotoSans-Bold.ttf'); font-weight: 700; font-style: normal; }"
            )
        font_style = (
            "<style>\n"
            + "\n".join(font_rules)
            + "\nbody, p, h1, h2, h3, th, td, span, b, i, u { font-family: 'Noto Sans', 'Segoe UI', Arial, sans-serif; }\n"
            + "</style>"
        )
        if "</head>" in html_str:
            html_str = html_str.replace("</head>", font_style + "</head>", 1)
        elif "<body" in html_str:
            html_str = html_str.replace("<body", f"<head>{font_style}</head><body", 1)
        else:
            html_str = f"<head>{font_style}</head>{html_str}"
    story = fitz.Story(html=html_str, archive=archive) if has_archive else fitz.Story(html=html_str)
    writer = fitz.DocumentWriter(bio)
    try:
        more = 1
        pages = 0
        while more and pages < 400:
            dev = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
            pages += 1
        if pages == 0:
            dev = writer.begin_page(mediabox)
            writer.end_page()
    finally:
        writer.close()
    del story, writer
    return bio.getvalue()


def _render_docx_pure_python_to_pdf(src_path: Path, pdf_out: Path) -> Tuple[bool, List[str]]:
    _ensure_fitz()
    if not _ensure_python_docx() or python_docx is None:
        return False, []
    try:
        doc = python_docx.Document(io.BytesIO(src_path.read_bytes()))
    except Exception as exc:
        _LOGGER.warning("python-docx failed to open %s: %s", src_path.name, exc)
        return False, []

    w_ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    r_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    a_ns = "{http://schemas.openxmlformats.org/drawingml/2006/main}"

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as asset_tmp:
        assets_dir = Path(asset_tmp)
        rid_to_fname: Dict[str, str] = {}
        try:
            for r_id, rel in doc.part.rels.items():
                if "image" in str(getattr(rel, "reltype", "")).lower():
                    blob = rel.target_part.blob
                    ext = Path(str(getattr(rel, "target_ref", "img.png"))).suffix or ".png"
                    if ext.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
                        try:
                            im = Image.open(io.BytesIO(blob)).convert("RGB")
                            out_buf = io.BytesIO()
                            im.save(out_buf, format="PNG")
                            blob = out_buf.getvalue()
                            ext = ".png"
                        except Exception:
                            continue
                    safe_name = f"rel_{len(rid_to_fname)}{ext}"
                    (assets_dir / safe_name).write_bytes(blob)
                    rid_to_fname[r_id] = safe_name
        except Exception as exc:
            _LOGGER.debug("DOCX image relation extraction note: %s", exc)

        para_map = {p._element: p for p in doc.paragraphs}
        table_map = {t._element: t for t in doc.tables}
        html_blocks: List[str] = []

        def _render_paragraph_html(p: Any) -> str:
            style_name = str(getattr(getattr(p, "style", None), "name", "") or "")
            s_low = style_name.lower()
            if s_low == "title":
                tag = "h1"
                extra_css = "font-size: 20pt; color: #1E1B4B; margin: 0 0 10px 0;"
            elif s_low == "subtitle":
                tag = "h2"
                extra_css = "font-size: 13.5pt; color: #475569; margin: 0 0 12px 0;"
            elif s_low.startswith("heading 1"):
                tag = "h1"
                extra_css = "font-size: 16.5pt; color: #1E293B; margin: 14px 0 6px 0;"
            elif s_low.startswith("heading 2"):
                tag = "h2"
                extra_css = "font-size: 14pt; color: #1E40AF; margin: 12px 0 5px 0;"
            elif s_low.startswith("heading"):
                tag = "h3"
                extra_css = "font-size: 12.5pt; color: #0F172A; margin: 10px 0 4px 0;"
            else:
                tag = "p"
                extra_css = "margin: 0 0 7px 0; line-height: 1.45;"

            align_css = ""
            try:
                al = str(getattr(p, "alignment", "") or "").upper()
                if "CENTER" in al:
                    align_css = "text-align: center;"
                elif "RIGHT" in al:
                    align_css = "text-align: right;"
                elif "JUSTIFY" in al:
                    align_css = "text-align: justify;"
            except Exception as exc:
                _LOGGER.debug("DOCX paragraph alignment fallback: %s", exc)

            bullet = "• " if "list" in s_low or p._element.find(f"{w_ns}pPr/{w_ns}numPr") is not None else ""
            if bullet:
                extra_css += " padding-left: 14px;"

            parts: List[str] = [bullet]
            for run in p.runs:
                txt = html_lib.escape(normalize_vietnamese_text(run.text or ""))
                txt = txt.replace("\n", "<br/>")
                r_styles: List[str] = []
                rf = run.font
                if rf.color is not None and rf.color.rgb is not None:
                    r_styles.append(f"color: #{rf.color.rgb}")
                if rf.size is not None:
                    r_styles.append(f"font-size: {max(8.0, min(36.0, float(rf.size.pt))):.1f}pt")
                if run.bold:
                    txt = f"<b>{txt}</b>"
                if run.italic:
                    txt = f"<i>{txt}</i>"
                if run.underline:
                    txt = f"<u>{txt}</u>"
                if r_styles:
                    txt = f"<span style=\"{'; '.join(r_styles)}\">{txt}</span>"
                if txt:
                    parts.append(txt)

                # Inline embedded images inside this run (`a:blip r:embed="..."`)
                for blip in run._element.iter(f"{a_ns}blip"):
                    rid = blip.get(f"{r_ns}embed")
                    if rid and rid in rid_to_fname:
                        parts.append(f"<br/><img src=\"{rid_to_fname[rid]}\" width=\"440\"/><br/>")

            if not parts and p.text:
                parts.append(html_lib.escape(normalize_vietnamese_text(p.text)))
            inner = "".join(parts).strip()
            if not inner:
                return "<p style=\"margin: 0 0 6px 0;\">&nbsp;</p>"
            return f"<{tag} style=\"{extra_css} {align_css}\">{inner}</{tag}>"

        for child in doc.element.body:
            if child in para_map:
                html_blocks.append(_render_paragraph_html(para_map[child]))
            elif child in table_map:
                tbl = table_map[child]
                rows_html: List[str] = []
                for r_i, row in enumerate(tbl.rows):
                    cells_html: List[str] = []
                    cell_tag = "th" if r_i == 0 else "td"
                    bg = "#E2E8F0" if r_i == 0 else "#FFFFFF"
                    for cell in row.cells:
                        c_txt = "<br/>".join(
                            html_lib.escape(normalize_vietnamese_text(cp.text).strip())
                            for cp in cell.paragraphs
                            if normalize_vietnamese_text(cp.text).strip()
                        ) or "&nbsp;"
                        cells_html.append(
                            f"<{cell_tag} style=\"background: {bg}; border: 1px solid #94A3B8; padding: 6px 8px; text-align: left;\">{c_txt}</{cell_tag}>"
                        )
                    rows_html.append("<tr>" + "".join(cells_html) + "</tr>")
                html_blocks.append(
                    "<table style=\"border-collapse: collapse; width: 100%; margin: 10px 0;\">"
                    + "".join(rows_html)
                    + "</table>"
                )

        if not html_blocks:
            html_blocks.append(f"<p>{html_lib.escape(src_path.name)}</p>")

        full_html = (
            "<html><head><meta charset=\"utf-8\"/></head>"
            "<body style=\"font-family: 'Segoe UI', Arial, Tahoma, sans-serif; font-size: 11.5pt; color: #0F172A; background: #FFFFFF;\">"
            + "\n".join(html_blocks)
            + "</body></html>"
        )
        pdf_bytes = _render_story_html_to_pdf_bytes(full_html, assets_dir=assets_dir)
        pdf_out.write_bytes(pdf_bytes)

    page_texts: List[str] = []
    out_doc = fitz.open(str(pdf_out))
    try:
        for i in range(out_doc.page_count):
            page_texts.append(normalize_vietnamese_text(out_doc.load_page(i).get_text("text")).strip())
    finally:
        out_doc.close()
    return pdf_out.exists() and pdf_out.stat().st_size > 200, page_texts


def _render_legacy_doc_fallback_to_pdf(src_path: Path, pdf_out: Path) -> Tuple[bool, List[str]]:
    raw = src_path.read_bytes()
    utf16_chunks = re.findall(rb"(?:[\x20-\x7e\u00a0-\u024f\u1e00-\u1eff]\x00){5,}", raw)
    extracted: List[str] = []
    for ch in utf16_chunks:
        try:
            txt = normalize_vietnamese_text(ch.decode("utf-16le", errors="ignore")).strip()
            if txt and len(txt) >= 5 and txt not in extracted:
                extracted.append(txt)
        except Exception:
            continue
    if not extracted:
        ascii_chunks = re.findall(rb"[\x20-\x7e]{8,}", raw)
        extracted = [normalize_vietnamese_text(c.decode("ascii", errors="ignore")).strip() for c in ascii_chunks[:80]]

    body_html = "".join(f"<p>{html_lib.escape(t)}</p>" for t in extracted) or f"<p>{html_lib.escape(normalize_vietnamese_text(src_path.name))}</p>"
    full_html = (
        "<html><body style=\"font-family: 'Segoe UI', Arial, sans-serif; font-size: 12pt; color: #0F172A;\">"
        f"<h2>{html_lib.escape(normalize_vietnamese_text(src_path.name))}</h2>{body_html}</body></html>"
    )
    pdf_out.write_bytes(_render_story_html_to_pdf_bytes(full_html))
    page_texts: List[str] = []
    _ensure_fitz()
    out_doc = fitz.open(str(pdf_out))
    try:
        for i in range(out_doc.page_count):
            page_texts.append(normalize_vietnamese_text(out_doc.load_page(i).get_text("text")).strip())
    finally:
        out_doc.close()
    return pdf_out.exists() and pdf_out.stat().st_size > 200, page_texts


def convert_word_to_cached_pdf(
    src_path: Path | str,
    cache_dir: Optional[Path] = None,
) -> Tuple[Path, List[str], str]:
    """
    Convert `.docx` / `.doc` to a paginated cached PDF keyed by SHA-256 in `%APPDATA%/AirTeachAI/doc_cache`:
    Tier 1: MS Word COM / LibreOffice headless -> Tier 2: Pure-Python `python-docx` + `PyMuPDF.Story`.
    """
    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(f"Không tìm thấy tài liệu Word: {src}")
    _ensure_fitz()

    doc_hash = compute_file_sha256(src)
    root_cache = Path(cache_dir) if cache_dir is not None else get_doc_cache_dir()
    entry_dir = root_cache / doc_hash[:24]
    entry_dir.mkdir(parents=True, exist_ok=True)
    pdf_out = entry_dir / "converted.pdf"
    manifest_path = entry_dir / "manifest.json"

    if pdf_out.exists() and pdf_out.stat().st_size > 200 and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("sha256") == doc_hash:
                probe = fitz.open(str(pdf_out))
                cnt = probe.page_count
                probe.close()
                if cnt >= 1:
                    texts = [normalize_vietnamese_text(str(t)) for t in (manifest.get("page_texts") or [])]
                    return pdf_out, texts, "CACHE_HIT"
        except Exception as exc:
            _LOGGER.debug("Word PDF cache probe miss (%s): %s", src.name, exc)

    backend = ""
    page_texts: List[str] = []
    if _convert_via_word_com(src, pdf_out):
        backend = "WORD_COM"
    elif _convert_via_libreoffice(src, entry_dir, pdf_out):
        backend = "LIBREOFFICE"
    elif src.suffix.lower() == ".docx":
        ok, page_texts = _render_docx_pure_python_to_pdf(src, pdf_out)
        if ok:
            backend = "PYTHON_DOCX_STORY"
    elif src.suffix.lower() == ".doc":
        ok, page_texts = _render_legacy_doc_fallback_to_pdf(src, pdf_out)
        if ok:
            backend = "LEGACY_DOC_FALLBACK"

    if not backend or not pdf_out.exists():
        raise RuntimeError(
            f"Không thể chuyển đổi tài liệu Word '{src.name}' sang PDF."
        )

    doc = fitz.open(str(pdf_out))
    try:
        if not page_texts or len(page_texts) != doc.page_count:
            page_texts = [
                normalize_vietnamese_text(doc.load_page(idx).get_text("text")).strip()
                for idx in range(doc.page_count)
            ]
        else:
            page_texts = [normalize_vietnamese_text(t).strip() for t in page_texts]
    finally:
        doc.close()

    manifest_path.write_text(
        json.dumps(
            {
                "sha256": doc_hash,
                "source_name": src.name,
                "backend": backend,
                "page_count": len(page_texts),
                "page_texts": page_texts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return pdf_out, page_texts, backend


class LectureDocument:
    """
    Thread-safe high-DPI PDF/PPTX presentation document with LRU page/thumbnail cache
    and asynchronous background adjacent-page pre-rendering.
    """

    def __init__(
        self,
        source_path: Path,
        pdf_path: Path,
        doc_hash: str,
        backend_used: str,
        slide_texts: Optional[Sequence[str]] = None,
        doc_type: str = "PDF",
        max_page_cache: int = 24,
        max_thumb_cache: int = 96,
    ) -> None:
        self.source_path = Path(source_path)
        self.pdf_path = Path(pdf_path)
        self.doc_hash = doc_hash
        self.backend_used = backend_used
        self.doc_type = doc_type
        self._lock = threading.Lock()
        _ensure_fitz()
        try:
            self._doc = fitz.open(str(self.pdf_path))
        except Exception:
            self._doc = fitz.open(stream=self.pdf_path.read_bytes(), filetype="pdf")
        self.page_count = int(self._doc.page_count)
        self.slide_texts: List[str] = [normalize_vietnamese_text(str(t)) for t in (slide_texts or [])]
        self._max_page_cache = max(6, int(max_page_cache))
        self._max_thumb_cache = max(16, int(max_thumb_cache))
        self._page_cache: OrderedDict[Tuple[int, int], QImage] = OrderedDict()
        self._thumb_cache: OrderedDict[Tuple[int, int], QImage] = OrderedDict()
        self._closed = False

    @classmethod
    def open(cls, path: Path | str, cache_dir: Optional[Path] = None) -> "LectureDocument":
        src = Path(path)
        if not src.exists():
            raise FileNotFoundError(f"Không tìm thấy tài liệu: {src}")
        ext = src.suffix.lower()
        if ext == ".pdf":
            doc_hash = compute_file_sha256(src)
            return cls(
                source_path=src,
                pdf_path=src,
                doc_hash=doc_hash,
                backend_used="PYMUPDF",
                slide_texts=[],
                doc_type="PDF",
            )
        if ext in (".pptx", ".ppt", ".ppsx", ".potx"):
            pdf_path, slide_texts, backend = convert_presentation_to_cached_pdf(src, cache_dir=cache_dir)
            doc_hash = compute_file_sha256(src)
            return cls(
                source_path=src,
                pdf_path=pdf_path,
                doc_hash=doc_hash,
                backend_used=backend,
                slide_texts=slide_texts,
                doc_type="PPTX",
            )
        if ext in (".docx", ".doc"):
            pdf_path, page_texts, backend = convert_word_to_cached_pdf(src, cache_dir=cache_dir)
            doc_hash = compute_file_sha256(src)
            return cls(
                source_path=src,
                pdf_path=pdf_path,
                doc_hash=doc_hash,
                backend_used=backend,
                slide_texts=page_texts,
                doc_type="WORD",
            )
        raise ValueError(f"Định dạng không được hỗ trợ: {ext}")

    def _render_qimage_unlocked(self, page_idx: int, matrix: fitz.Matrix) -> QImage:
        page = self._doc.load_page(page_idx)
        pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
        qimg = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888).copy()
        return qimg

    def render_page_qimage(self, page_idx: int, scale: float = 2.25) -> QImage:
        idx = max(0, min(self.page_count - 1, int(page_idx)))
        scale_key = int(round(max(0.5, min(4.0, float(scale))) * 100))
        key = (idx, scale_key)
        with self._lock:
            if self._closed:
                return QImage()
            cached = self._page_cache.get(key)
            if cached is not None:
                self._page_cache.move_to_end(key)
                return cached
            qimg = self._render_qimage_unlocked(idx, fitz.Matrix(scale_key / 100.0, scale_key / 100.0))
            self._page_cache[key] = qimg
            self._page_cache.move_to_end(key)
            while len(self._page_cache) > self._max_page_cache:
                self._page_cache.popitem(last=False)
            return qimg

    def render_thumbnail_qimage(self, page_idx: int, max_w: int = 160) -> QImage:
        idx = max(0, min(self.page_count - 1, int(page_idx)))
        target_w = max(48, int(max_w))
        key = (idx, target_w)
        with self._lock:
            if self._closed:
                return QImage()
            cached = self._thumb_cache.get(key)
            if cached is not None:
                self._thumb_cache.move_to_end(key)
                return cached
            page = self._doc.load_page(idx)
            pw = max(1.0, float(page.rect.width))
            sc = max(0.08, min(0.5, float(target_w) / pw))
            qimg = self._render_qimage_unlocked(idx, fitz.Matrix(sc, sc))
            self._thumb_cache[key] = qimg
            self._thumb_cache.move_to_end(key)
            while len(self._thumb_cache) > self._max_thumb_cache:
                self._thumb_cache.popitem(last=False)
            return qimg

    def prewarm_around(
        self,
        center_idx: int,
        radius: int = 2,
        scale: float = 2.25,
        offsets: Optional[Sequence[int]] = (-1, 0, 1, 2),
    ) -> None:
        if self._closed or self.page_count <= 1:
            return
        if offsets is not None:
            raw_indices = [center_idx + int(off) for off in offsets]
        else:
            raw_indices = list(range(center_idx - radius, center_idx + radius + 1))
        indices = []
        for i in raw_indices:
            if 0 <= i < self.page_count and i not in indices:
                indices.append(i)
        if not indices:
            return

        def _worker() -> None:
            for idx in indices:
                if self._closed:
                    break
                try:
                    self.render_page_qimage(idx, scale=scale)
                except Exception:
                    break

        threading.Thread(target=_worker, name="airteach-doc-prewarm", daemon=True).start()

    def get_page_text(self, page_idx: int) -> str:
        idx = max(0, min(self.page_count - 1, int(page_idx)))
        with self._lock:
            if self._closed:
                return ""
            pdf_txt = normalize_vietnamese_text(self._doc.load_page(idx).get_text("text")).strip()
        extra = normalize_vietnamese_text(self.slide_texts[idx]).strip() if idx < len(self.slide_texts) else ""
        if pdf_txt and extra and extra not in pdf_txt:
            return normalize_vietnamese_text(f"{pdf_txt}\n{extra}").strip()
        return pdf_txt or extra

    def get_page_aspect_ratio(self, page_idx: int) -> float:
        idx = max(0, min(self.page_count - 1, int(page_idx)))
        with self._lock:
            if self._closed:
                return 16.0 / 9.0
            r = self._doc.load_page(idx).rect
            return max(0.25, float(r.width) / max(1.0, float(r.height)))

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                try:
                    self._doc.close()
                except Exception as exc:
                    _LOGGER.debug("LectureDocument close note: %s", exc)
                self._page_cache.clear()
                self._thumb_cache.clear()


def compute_slide_rect_on_canvas(
    canvas_size: Tuple[float, float],
    aspect_ratio: float = 16.0 / 9.0,
    is_tutorial: bool = False,
) -> QRectF:
    """Compute canonical centered slide QRectF inside canvas coordinates (matching SmartboardCanvas layout)."""
    cw = max(320.0, float(canvas_size[0]))
    ch = max(240.0, float(canvas_size[1]))
    if is_tutorial:
        s_w = min(740.0, max(320.0, cw - 40.0))
        s_h = float(int(s_w * 9.0 / 16.0))
        return QRectF(16.0, 46.0, s_w, s_h)

    avail_w = max(240.0, cw - 48.0)
    avail_h = max(180.0, ch - 72.0)
    ar = max(0.3, min(3.5, float(aspect_ratio)))
    if avail_w / avail_h >= ar:
        s_h = avail_h
        s_w = s_h * ar
    else:
        s_w = avail_w
        s_h = s_w / ar
    sx = 0.5 * (cw - s_w)
    sy = 46.0 + 0.5 * (avail_h - s_h)
    return QRectF(sx, sy, s_w, s_h)


def _qimage_to_png_bytes(img: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.WriteOnly)
    img.save(buf, "PNG")
    buf.close()
    return bytes(ba)


def get_office_conversion_notice(doc_type: str, backend_used: str) -> str:
    """Return a clear, user-friendly Vietnamese conversion status note for PPTX/DOCX."""
    dt = str(doc_type or "").upper()
    bk = str(backend_used or "").upper()
    if dt not in ("PPTX", "WORD"):
        return ""
    if bk in ("LIBREOFFICE", "POWERPOINT_COM", "WORD_COM"):
        return f"Đã chuyển đổi chuẩn gốc bằng `{bk}` (giữ trọn bố cục & font chữ)."
    if bk == "CACHE_HIT":
        return "Tải tức thì từ bộ đệm SHA-256 cục bộ (<0.2s)."
    return (
        f"Đang dùng trình render nội bộ (`{bk}`) vì máy chưa cài LibreOffice hoặc Microsoft Office. "
        "Các slide/văn bản vẫn hiển thị & chú thích bình thường; cài thêm LibreOffice nếu cần giữ hiệu ứng/SmartArt phức tạp 100%."
    )


def _render_annotation_card_png(ann: Dict[str, Any]) -> Optional[bytes]:
    rot = float(ann.get("rotation", 0.0) or 0.0)
    ann_pm = ann.get("pixmap")
    if isinstance(ann_pm, QPixmap) and not ann_pm.isNull():
        img = ann_pm.toImage()
    else:
        text = str(ann.get("text") or ann.get("raw") or "").strip()
        if not text:
            return None
        w = max(120, int(round(float(ann.get("w", 240.0)))))
        h = max(44, int(round(float(ann.get("h", 52.0)))))
        img = QImage(w, h, QImage.Format_ARGB32)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        is_sticky = str(ann.get("obj_type") or "") == "sticky"
        if is_sticky:
            bg_c = QColor(str(ann.get("bg_color") or "#FEF08A"))
            bd_c = QColor(str(ann.get("border_color") or "#CA8A04"))
            fg_c = QColor(str(ann.get("color") or "#1E293B"))
            p.setBrush(bg_c)
            p.setPen(QPen(bd_c, 2.0))
            p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 8.0, 8.0)
            p.fillRect(QRectF(2, 2, w - 4, 10), bd_c)
            p.setPen(fg_c)
            p.setFont(QFont("Segoe UI", max(10, int(ann.get("font_size") or 13)), QFont.DemiBold))
            p.drawText(QRectF(10, 16, w - 20, h - 22), Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap, text)
        else:
            bg_raw = str(ann.get("bg_color") or "")
            bg_c = QColor(bg_raw) if bg_raw else QColor(15, 23, 42, 225)
            bd_c = QColor(str(ann.get("border_color") or "#06B6D4"))
            p.setBrush(bg_c)
            p.setPen(QPen(bd_c, 1.5))
            p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 8.0, 8.0)
            p.setPen(QColor(str(ann.get("color") or "#00F5D4")))
            p.setFont(QFont("Segoe UI", max(10, int(ann.get("font_size") or 13)), QFont.Bold))
            p.drawText(QRectF(8, 6, w - 16, h - 12), Qt.AlignTop | Qt.AlignLeft | Qt.TextWordWrap, text)
        p.end()

    if abs(rot) > 0.1:
        img = img.transformed(QTransform().rotate(rot), Qt.SmoothTransformation)
    return _qimage_to_png_bytes(img)


def _rotate_xy_deg(px: float, py: float, cx: float, cy: float, angle_deg: float) -> Tuple[float, float]:
    if abs(angle_deg) < 1e-4:
        return (px, py)
    import math as _m

    rad = _m.radians(angle_deg)
    c, s = _m.cos(rad), _m.sin(rad)
    dx, dy = px - cx, py - cy
    return (cx + dx * c - dy * s, cy + dx * s + dy * c)


def export_annotated_lecture_pdf(
    output_path: Path | str,
    slides: Sequence[Dict[str, Any]],
    slide_annotations: Dict[int, Dict[str, Any]],
    canvas_size: Tuple[float, float],
    lecture_doc: Optional[LectureDocument] = None,
) -> Path:
    """
    Export the active slide deck (PDF/PPTX/DOCX/Images or whiteboard slides) with every slide's
    non-destructive Object annotations (strokes, shapes, text boxes, sticky notes, inserted images)
    burned cleanly into a single PDF.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    _ensure_fitz()

    if lecture_doc is not None and lecture_doc.pdf_path.exists():
        out_doc = fitz.open(str(lecture_doc.pdf_path))
    else:
        out_doc = fitz.open()
        cw = max(960.0, float(canvas_size[0]))
        ch = max(540.0, float(canvas_size[1]))
        if not slides:
            out_doc.new_page(width=cw, height=ch)
        else:
            for sl in slides:
                pm = sl.get("pixmap")
                page = out_doc.new_page(width=cw, height=ch)
                if isinstance(pm, QPixmap) and not pm.isNull():
                    png = _qimage_to_png_bytes(pm.toImage())
                    page.insert_image(page.rect, stream=png)
                else:
                    card = QImage(int(cw), int(ch), QImage.Format_RGB888)
                    card.fill(QColor("#0F172A"))
                    if not bool(sl.get("is_blank", False)):
                        qp = QPainter(card)
                        qp.setPen(QColor("#06B6D4"))
                        qp.setFont(QFont("Segoe UI", 22, QFont.Bold))
                        qp.drawText(QPointF(48, 84), str(sl.get("title") or "AirTeach AI Smartboard"))
                        qp.setPen(QColor("#F8FAFC"))
                        qp.setFont(QFont("Segoe UI", 14))
                        for b_i, bullet in enumerate(sl.get("bullets") or []):
                            qp.drawText(QPointF(54, 148 + b_i * 38), str(bullet))
                        qp.end()
                    page.insert_image(page.rect, stream=_qimage_to_png_bytes(card))

    try:
        n_pages = out_doc.page_count
        for idx in range(n_pages):
            layer = slide_annotations.get(idx)
            if not layer:
                continue
            strokes = layer.get("strokes") or []
            annotations = layer.get("annotations") or []
            images = layer.get("images") or []
            if not strokes and not annotations and not images:
                continue

            page = out_doc.load_page(idx)
            prect = page.rect
            pw, ph = float(prect.width), float(prect.height)
            is_tut = bool(slides[idx].get("is_tutorial")) if idx < len(slides) else False
            s_rect = compute_slide_rect_on_canvas(canvas_size, aspect_ratio=pw / max(1.0, ph), is_tutorial=is_tut)
            sx = pw / max(1.0, s_rect.width())
            sy = ph / max(1.0, s_rect.height())

            for img_item in images:
                img_obj = img_item.get("pixmap")
                if isinstance(img_obj, QPixmap) and not img_obj.isNull():
                    qimg_obj = img_obj.toImage()
                elif isinstance(img_obj, QImage) and not img_obj.isNull():
                    qimg_obj = img_obj
                else:
                    continue
                rot = float(img_item.get("rotation", 0.0) or 0.0)
                if abs(rot) > 0.1:
                    qimg_obj = qimg_obj.transformed(QTransform().rotate(rot), Qt.SmoothTransformation)
                img_bytes = _qimage_to_png_bytes(qimg_obj)
                if not img_bytes:
                    continue
                ix = (float(img_item.get("x", 60.0)) - s_rect.left()) * sx
                iy = (float(img_item.get("y", 80.0)) - s_rect.top()) * sy
                iw = max(24.0, float(img_item.get("w", 240.0)) * sx)
                ih = max(24.0, float(img_item.get("h", 180.0)) * sy)
                target = fitz.Rect(ix, iy, ix + iw, iy + ih)
                page.insert_image(target, stream=img_bytes, overlay=True)

            for st in strokes:
                pts = st.get("points") or []
                if len(pts) < 2:
                    continue
                rot = float(st.get("rotation", 0.0) or 0.0)
                cx_s = float(st.get("x", 0.0)) + 0.5 * float(st.get("w", 0.0))
                cy_s = float(st.get("y", 0.0)) + 0.5 * float(st.get("h", 0.0))
                fitz_pts = []
                for p in pts:
                    px0 = float(p.x()) if hasattr(p, "x") else float(p[0])
                    py0 = float(p.y()) if hasattr(p, "y") else float(p[1])
                    if abs(rot) > 0.1:
                        px0, py0 = _rotate_xy_deg(px0, py0, cx_s, cy_s, rot)
                    fitz_pts.append(
                        fitz.Point((px0 - s_rect.left()) * sx, (py0 - s_rect.top()) * sy)
                    )
                col = st.get("color")
                if isinstance(col, QColor):
                    rgb = (col.redF(), col.greenF(), col.blueF())
                    alpha_f = col.alphaF()
                else:
                    qc = QColor(str(col or "#00F5D4"))
                    rgb = (qc.redF(), qc.greenF(), qc.blueF())
                    alpha_f = qc.alphaF()

                is_hi = bool(st.get("is_highlighter", False))
                opacity = min(0.42, alpha_f) if is_hi else max(0.15, alpha_f)
                thick = max(1.0, float(st.get("thickness", 4.0)) * 0.5 * (sx + sy))

                shape = page.new_shape()
                shape.draw_polyline(fitz_pts)
                shape.finish(
                    color=rgb,
                    width=thick,
                    lineCap=0 if is_hi else 1,
                    lineJoin=1,
                    stroke_opacity=opacity,
                )
                shape.commit(overlay=True)

            for ann in annotations:
                png_bytes = _render_annotation_card_png(ann)
                if not png_bytes:
                    continue
                ax = (float(ann.get("x", 40.0)) - s_rect.left()) * sx
                ay = (float(ann.get("y", 60.0)) - s_rect.top()) * sy
                aw = max(40.0, float(ann.get("w", 200.0)) * sx)
                ah = max(20.0, float(ann.get("h", 44.0)) * sy)
                target = fitz.Rect(ax, ay, ax + aw, ay + ah)
                page.insert_image(target, stream=png_bytes, overlay=True)

        out_doc.save(str(out_file), garbage=3, deflate=True)
    finally:
        out_doc.close()

    return out_file


def export_annotated_slides_to_pngs(
    output_dir: Path | str,
    slides: Sequence[Dict[str, Any]],
    slide_annotations: Dict[int, Dict[str, Any]],
    canvas_size: Tuple[float, float],
    lecture_doc: Optional[LectureDocument] = None,
    scale: float = 2.0,
) -> List[Path]:
    """
    Export every slide of the annotated deck as crisp high-DPI PNG files (`slide_001.png`, ...).
    Burns all per-page Objects (strokes, shapes, text boxes, sticky notes, images) accurately.
    """
    out_folder = Path(output_dir)
    out_folder.mkdir(parents=True, exist_ok=True)
    _ensure_fitz()

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpd:
        tmp_pdf = Path(tmpd) / "deck_annotated.pdf"
        export_annotated_lecture_pdf(
            tmp_pdf,
            slides,
            slide_annotations,
            canvas_size,
            lecture_doc=lecture_doc,
        )
        saved_paths: List[Path] = []
        doc = fitz.open(str(tmp_pdf))
        try:
            mat = fitz.Matrix(max(1.0, min(3.5, float(scale))), max(1.0, min(3.5, float(scale))))
            for idx in range(doc.page_count):
                pix = doc.load_page(idx).get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
                out_png = out_folder / f"slide_{idx + 1:03d}.png"
                pix.save(str(out_png))
                saved_paths.append(out_png)
        finally:
            doc.close()
    return saved_paths


def _natural_sort_key(name: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def list_folder_images(folder_path: Path | str) -> List[Path]:
    """Return supported images inside `folder_path` ordered naturally for gallery slide deck."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    files = [
        p
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    files.sort(key=lambda p: _natural_sort_key(p.name))
    return files


def load_image_file_qimage(src_path: Path | str, target_max_dim: int = 2048) -> QImage:
    """
    Load PNG/JPG/WEBP/BMP or vector SVG at crisp high resolution as QImage.
    Hardened for non-ASCII Vietnamese Windows file paths via `Path.read_bytes()`
    -> `QImage.loadFromData` / Pillow `BytesIO` / PyMuPDF stream fallbacks.
    """
    src = Path(src_path)
    if not src.exists():
        return QImage()
    try:
        raw_bytes = src.read_bytes()
    except OSError as exc:
        _LOGGER.warning("Cannot read image bytes (%s): %s", src.name, exc)
        return QImage()
    if not raw_bytes:
        return QImage()

    ext = src.suffix.lower()
    if ext == ".svg":
        try:
            _ensure_fitz()
            doc = fitz.open(stream=raw_bytes, filetype="svg")
            try:
                page = doc.load_page(0)
                rect = page.rect
                base_dim = max(1.0, float(rect.width), float(rect.height))
                scale = max(1.0, min(8.0, float(target_max_dim) / base_dim))
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=True)
                return QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGBA8888).copy()
            finally:
                doc.close()
        except Exception as exc:
            _LOGGER.warning("SVG PyMuPDF raster fallback triggered for %s: %s", src.name, exc)
            try:
                from PySide6.QtSvg import QSvgRenderer

                renderer = QSvgRenderer(QByteArray(raw_bytes))
                if renderer.isValid():
                    sz = renderer.defaultSize()
                    w = max(64, int(sz.width()) or 1024)
                    h = max(64, int(sz.height()) or 768)
                    scale = min(6.0, float(target_max_dim) / max(w, h))
                    out = QImage(int(round(w * scale)), int(round(h * scale)), QImage.Format_ARGB32)
                    out.fill(Qt.transparent)
                    p = QPainter(out)
                    renderer.render(p)
                    p.end()
                    return out
            except Exception:
                return QImage()

    qimg = QImage(str(src))
    if not qimg.isNull():
        return qimg
    qimg_mem = QImage()
    if qimg_mem.loadFromData(raw_bytes) and not qimg_mem.isNull():
        return qimg_mem
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
        raw = pil_img.tobytes("raw", "RGBA")
        return QImage(raw, pil_img.width, pil_img.height, pil_img.width * 4, QImage.Format_RGBA8888).copy()
    except Exception:
        return QImage()


def rasterize_document_to_png_dir(
    src_path: Path | str,
    out_dir: Path | str,
    target_min_width: int = 1600,
    max_pages: int = 80,
) -> Path:
    """Headless high-DPI document rasterizer (PDF / PPTX / PPT / DOCX / DOC) writing PNG pages + manifest.json."""
    src = Path(src_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    if ext == ".pdf":
        pdf_path = src
        texts: List[str] = []
        backend = "PYMUPDF"
        doc_type = "PDF"
    elif ext in (".pptx", ".ppt", ".ppsx", ".potx"):
        pdf_path, texts, backend = convert_presentation_to_cached_pdf(src)
        doc_type = "PPTX"
    elif ext in (".docx", ".doc"):
        pdf_path, texts, backend = convert_word_to_cached_pdf(src)
        doc_type = "WORD"
    else:
        raise ValueError(f"Định dạng tài liệu không được hỗ trợ: {ext}")

    pages_meta: List[Dict[str, Any]] = []
    _ensure_fitz()
    doc = fitz.open(str(pdf_path))
    try:
        total = min(int(doc.page_count), max(1, int(max_pages)))
        for idx in range(total):
            page = doc.load_page(idx)
            pw = max(1.0, float(page.rect.width))
            scale = max(2.0, min(3.2, float(target_min_width) / pw))
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
            fname = f"page_{idx + 1:04d}.png"
            pix.save(str(out / fname))
            txt = normalize_vietnamese_text(page.get_text("text")).strip()
            if not txt and idx < len(texts):
                txt = normalize_vietnamese_text(texts[idx]).strip()
            pages_meta.append({
                "index": idx,
                "image_file": fname,
                "width": int(pix.width),
                "height": int(pix.height),
                "text": txt,
            })
    finally:
        doc.close()

    manifest = out / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_name": src.name,
                "doc_type": doc_type,
                "backend": backend,
                "page_count": len(pages_meta),
                "pages": pages_meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return manifest


def _cli_main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) >= 3 and args[0] == "--rasterize":
        manifest = rasterize_document_to_png_dir(args[1], args[2])
        print(str(manifest))
        return 0
    print("Usage: python -m cv_engine.document_presenter --rasterize <source_doc> <out_dir>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_cli_main())
