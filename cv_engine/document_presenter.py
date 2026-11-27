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

try:
    import pymupdf as fitz  # type: ignore
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

from PIL import Image, ImageDraw, ImageFont

try:
    import docx as python_docx  # type: ignore

    HAS_PYTHON_DOCX = True
except Exception:  # pragma: no cover
    python_docx = None  # type: ignore[assignment]
    HAS_PYTHON_DOCX = False

try:
    from pptx import Presentation  # type: ignore
    from pptx.enum.shapes import MSO_SHAPE_TYPE  # type: ignore
    from pptx.enum.text import PP_ALIGN  # type: ignore

    HAS_PYTHON_PPTX = True
except Exception:  # pragma: no cover
    Presentation = None  # type: ignore[assignment]
    MSO_SHAPE_TYPE = None  # type: ignore[assignment]
    PP_ALIGN = None  # type: ignore[assignment]
    HAS_PYTHON_PPTX = False

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap

from cv_engine.app_paths import get_app_logger, get_doc_cache_dir

_LOGGER = get_app_logger("document_presenter")

SUPPORTED_DOC_EXTENSIONS = {".pdf", ".pptx", ".ppt", ".ppsx", ".potx", ".docx", ".doc"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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
    # Try win32com directly if available
    try:
        import pythoncom  # type: ignore
        import win32com.client  # type: ignore

        pythoncom.CoInitialize()
        app = None
        pres = None
        try:
            app = win32com.client.DispatchEx("PowerPoint.Application")
            pres = app.Presentations.Open(str(src_path.resolve()), WithWindow=False)
            pres.SaveAs(str(pdf_out.resolve()), 32)  # ppSaveAsPDF = 32
        finally:
            if pres is not None:
                pres.Close()
            if app is not None:
                app.Quit()
            pythoncom.CoUninitialize()
        if pdf_out.exists() and pdf_out.stat().st_size > 200:
            return True
    except Exception:
        pass

    # Fallback: PowerShell COM automation (requires no pywin32 package)
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
            "  if ($pres) { $pres.Close() };"
            "  if ($pp) { $pp.Quit() };"
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

    if sys.platform == "win32":
        fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        candidates_map = {
            "r": ("segoeui.ttf", "arial.ttf", "tahoma.ttf", "calibri.ttf"),
            "b": ("segoeuib.ttf", "arialbd.ttf", "tahomabd.ttf", "calibrib.ttf"),
            "i": ("segoeuii.ttf", "ariali.ttf", "calibrii.ttf"),
            "bi": ("segoeuiz.ttf", "arialbi.ttf", "calibriz.ttf"),
        }
        names = candidates_map.get(variant, candidates_map["r"]) + candidates_map["r"]
        paths = [fonts_dir / n for n in names]
    else:
        candidates_map = {
            "r": ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            "b": ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
            "i": ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",),
            "bi": ("/usr/share/fonts/truetype/dejavu/DejaVuSans-BoldOblique.ttf",),
        }
        paths = [Path(p) for p in candidates_map.get(variant, candidates_map["r"])]

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
    except Exception:
        pass
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
    if not HAS_PYTHON_PPTX or Presentation is None:
        return []
    try:
        prs = Presentation(str(src_path))
        texts: List[str] = []
        for slide in prs.slides:
            parts: List[str] = []
            for shape in slide.shapes:
                if getattr(shape, "has_text_frame", False):
                    txt = shape.text_frame.text.strip()
                    if txt:
                        parts.append(txt)
                if getattr(shape, "has_table", False):
                    for row in shape.table.rows:
                        row_txt = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                        if row_txt:
                            parts.append(row_txt)
            texts.append("\n".join(parts))
        return texts
    except Exception:
        return []


def _render_pptx_pure_python_to_pdf(src_path: Path, pdf_out: Path) -> Tuple[bool, List[str]]:
    if not HAS_PYTHON_PPTX or Presentation is None:
        return False, []
    try:
        prs = Presentation(str(src_path))
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
            raw_p = para.text or ""
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
                    except Exception:
                        pass
                elif PP_ALIGN is not None and align == PP_ALIGN.RIGHT:
                    try:
                        lw = float(draw.textlength(line, font=pil_font))
                        tx = max(bx0 + pad_x, int(bx1 - pad_x - lw))
                    except Exception:
                        pass
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
                slide_texts.append("\n".join(s_acc))

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
            txt = ch.decode("utf-16le", errors="ignore").strip()
            if txt and len(txt) >= 5 and txt not in extracted:
                extracted.append(txt)
        except Exception:
            continue
    if not extracted:
        ascii_chunks = re.findall(rb"[\x20-\x7e]{8,}", raw)
        extracted = [c.decode("ascii", errors="ignore").strip() for c in ascii_chunks[:40]]

    chunks_per_slide = 8
    groups = [extracted[i : i + chunks_per_slide] for i in range(0, max(1, len(extracted)), chunks_per_slide)]
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
                    texts = [str(t) for t in (manifest.get("slide_texts") or [])]
                    return pdf_out, texts, "CACHE_HIT"
        except Exception:
            pass

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
                pdf_txt = doc.load_page(idx).get_text("text").strip()
                pt_txt = pptx_texts[idx] if idx < len(pptx_texts) else ""
                slide_texts.append(pdf_txt or pt_txt)
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
    try:
        import pythoncom  # type: ignore
        import win32com.client  # type: ignore

        pythoncom.CoInitialize()
        word_app = None
        word_doc = None
        try:
            word_app = win32com.client.DispatchEx("Word.Application")
            word_app.Visible = False
            word_doc = word_app.Documents.Open(
                str(src_path.resolve()),
                ConfirmConversions=False,
                ReadOnly=True,
                AddToRecentFiles=False,
            )
            word_doc.ExportAsFixedFormat(
                OutputFileName=str(pdf_out.resolve()),
                ExportFormat=17,  # wdExportFormatPDF
            )
        finally:
            if word_doc is not None:
                word_doc.Close(SaveChanges=False)
            if word_app is not None:
                word_app.Quit(SaveChanges=False)
            pythoncom.CoUninitialize()
        if pdf_out.exists() and pdf_out.stat().st_size > 200:
            return True
    except Exception:
        pass

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
            "  if ($d) { $d.Close($false) };"
            "  if ($w) { $w.Quit($false) };"
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
    bio = io.BytesIO()
    mediabox = fitz.paper_rect("a4")
    where = mediabox + (42, 42, -42, -42)
    archive = fitz.Archive(str(assets_dir)) if assets_dir is not None else None
    story = fitz.Story(html=html_str, archive=archive) if archive is not None else fitz.Story(html=html_str)
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
    if not HAS_PYTHON_DOCX or python_docx is None:
        return False, []
    try:
        doc = python_docx.Document(str(src_path))
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
            except Exception:
                pass

            bullet = "• " if "list" in s_low or p._element.find(f"{w_ns}pPr/{w_ns}numPr") is not None else ""
            if bullet:
                extra_css += " padding-left: 14px;"

            parts: List[str] = [bullet]
            for run in p.runs:
                txt = html_lib.escape(run.text or "")
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
                parts.append(html_lib.escape(p.text))
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
                            html_lib.escape(cp.text.strip()) for cp in cell.paragraphs if cp.text.strip()
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
            page_texts.append(out_doc.load_page(i).get_text("text").strip())
    finally:
        out_doc.close()
    return pdf_out.exists() and pdf_out.stat().st_size > 200, page_texts


def _render_legacy_doc_fallback_to_pdf(src_path: Path, pdf_out: Path) -> Tuple[bool, List[str]]:
    raw = src_path.read_bytes()
    utf16_chunks = re.findall(rb"(?:[\x20-\x7e\u00a0-\u024f\u1e00-\u1eff]\x00){5,}", raw)
    extracted: List[str] = []
    for ch in utf16_chunks:
        try:
            txt = ch.decode("utf-16le", errors="ignore").strip()
            if txt and len(txt) >= 5 and txt not in extracted:
                extracted.append(txt)
        except Exception:
            continue
    if not extracted:
        ascii_chunks = re.findall(rb"[\x20-\x7e]{8,}", raw)
        extracted = [c.decode("ascii", errors="ignore").strip() for c in ascii_chunks[:80]]

    body_html = "".join(f"<p>{html_lib.escape(t)}</p>" for t in extracted) or f"<p>{html_lib.escape(src_path.name)}</p>"
    full_html = (
        "<html><body style=\"font-family: 'Segoe UI', Arial, sans-serif; font-size: 12pt; color: #0F172A;\">"
        f"<h2>{html_lib.escape(src_path.name)}</h2>{body_html}</body></html>"
    )
    pdf_out.write_bytes(_render_story_html_to_pdf_bytes(full_html))
    page_texts: List[str] = []
    out_doc = fitz.open(str(pdf_out))
    try:
        for i in range(out_doc.page_count):
            page_texts.append(out_doc.load_page(i).get_text("text").strip())
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
                    texts = [str(t) for t in (manifest.get("page_texts") or [])]
                    return pdf_out, texts, "CACHE_HIT"
        except Exception:
            pass

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
            page_texts = [doc.load_page(idx).get_text("text").strip() for idx in range(doc.page_count)]
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
        self._doc = fitz.open(str(self.pdf_path))
        self.page_count = int(self._doc.page_count)
        self.slide_texts: List[str] = list(slide_texts or [])
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
            pdf_txt = self._doc.load_page(idx).get_text("text").strip()
        extra = self.slide_texts[idx].strip() if idx < len(self.slide_texts) else ""
        if pdf_txt and extra and extra not in pdf_txt:
            return f"{pdf_txt}\n{extra}"
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
                except Exception:
                    pass
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


def _render_annotation_card_png(ann: Dict[str, Any]) -> Optional[bytes]:
    ann_pm = ann.get("pixmap")
    if isinstance(ann_pm, QPixmap) and not ann_pm.isNull():
        return _qimage_to_png_bytes(ann_pm.toImage())

    text = str(ann.get("text") or ann.get("raw") or "").strip()
    if not text:
        return None
    w = max(120, int(round(float(ann.get("w", 240.0)))))
    h = max(40, int(round(float(ann.get("h", 48.0)))))
    img = QImage(w, h, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(15, 23, 42, 225))
    p.setPen(QPen(QColor("#06B6D4"), 1.5))
    p.drawRoundedRect(QRectF(1, 1, w - 2, h - 2), 8.0, 8.0)
    p.setPen(QColor(str(ann.get("color") or "#00F5D4")))
    p.setFont(QFont("Segoe UI", max(10, int(ann.get("font_size") or 13)), QFont.Bold))
    p.drawText(QRectF(8, 4, w - 16, h - 8), Qt.AlignVCenter | Qt.AlignLeft, text)
    p.end()
    return _qimage_to_png_bytes(img)


def export_annotated_lecture_pdf(
    output_path: Path | str,
    slides: Sequence[Dict[str, Any]],
    slide_annotations: Dict[int, Dict[str, Any]],
    canvas_size: Tuple[float, float],
    lecture_doc: Optional[LectureDocument] = None,
) -> Path:
    """
    Export the active slide deck (PDF/PPTX or whiteboard slides) with every slide's
    non-destructive ink strokes, highlighter marks, and AI annotations burned into a single PDF.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

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
                    # Render whiteboard slide card cleanly via QImage
                    card = QImage(int(cw), int(ch), QImage.Format_RGB888)
                    card.fill(QColor("#0F172A"))
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
            if not strokes and not annotations:
                continue

            page = out_doc.load_page(idx)
            prect = page.rect
            pw, ph = float(prect.width), float(prect.height)
            is_tut = bool(slides[idx].get("is_tutorial")) if idx < len(slides) else False
            s_rect = compute_slide_rect_on_canvas(canvas_size, aspect_ratio=pw / max(1.0, ph), is_tutorial=is_tut)
            sx = pw / max(1.0, s_rect.width())
            sy = ph / max(1.0, s_rect.height())

            for st in strokes:
                pts = st.get("points") or []
                if len(pts) < 2:
                    continue
                fitz_pts = [
                    fitz.Point(
                        (float(p.x()) - s_rect.left()) * sx,
                        (float(p.y()) - s_rect.top()) * sy,
                    )
                    for p in pts
                ]
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
            txt = page.get_text("text").strip()
            if not txt and idx < len(texts):
                txt = texts[idx].strip()
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
