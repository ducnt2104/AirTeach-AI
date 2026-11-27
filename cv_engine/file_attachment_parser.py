#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vietnamese-Safe File Attachment Parser & MIME Detector
(`cv_engine/file_attachment_parser.py`) for AirTeach AI (DNT - Victer).

Capabilities:
1. `normalize_vietnamese_text`: Canonical Unicode NFC normalization preserving
   Vietnamese diacritics, math notation, newlines, and tabs while stripping BOM,
   null bytes, broken UTF-16 surrogates, and control garbage.
2. `detect_mime_type`: Explicit, deterministic MIME mapping independent of
   broken Windows registry entries (never returns None).
3. `parse_attachment_file`: Safe multi-encoding text reader (`utf-8-sig`, `utf-8`,
   `utf-16`, `cp1258`, `latin-1`), PDF/DOCX/PPTX text + first-page Vision PNG
   preview extraction, and image normalization to compressed PNG (`<= 1024px`)
   with exact `image/png` MIME type.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass
import io
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import unicodedata


EXPLICIT_MIME_MAP: Dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".ppt": "application/vnd.ms-powerpoint",
    ".ppsx": "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
    ".potx": "application/vnd.openxmlformats-officedocument.presentationml.template",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".json": "application/json",
    # Extra common safe types (never querying Windows registry)
    ".gif": "image/gif",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".xml": "application/xml",
    ".html": "text/html",
    ".htm": "text/html",
    ".yaml": "text/yaml",
    ".yml": "text/yaml",
    ".py": "text/x-python",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".css": "text/css",
}

TEXT_ATTACHMENT_EXTENSIONS = frozenset({".txt", ".md", ".csv", ".json"})
IMAGE_ATTACHMENT_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".svg"})
DOCUMENT_ATTACHMENT_EXTENSIONS = frozenset({".pdf", ".docx", ".doc", ".pptx", ".ppt", ".ppsx", ".potx"})

MAX_ATTACHMENT_PREVIEW_DIM = 1024


def normalize_vietnamese_text(text: str) -> str:
    """
    Normalize text to Unicode NFC so Vietnamese diacritics stay canonical,
    stripping BOM (`\\ufeff`), null bytes, broken surrogates, and non-whitespace
    control characters while preserving math symbols, newlines, and tabs.
    """
    if not text:
        return ""
    s = str(text).replace("\r\n", "\n").replace("\r", "\n")

    cleaned_chars: List[str] = []
    for ch in s:
        code = ord(ch)
        if code == 0 or ch == "\ufeff":
            continue
        if 0xD800 <= code <= 0xDFFF:
            continue
        if ch in ("\n", "\t"):
            cleaned_chars.append(ch)
            continue
        if ch == "\x0c":
            cleaned_chars.append("\n")
            continue
        if unicodedata.category(ch) == "Cc":
            continue
        cleaned_chars.append(ch)

    return unicodedata.normalize("NFC", "".join(cleaned_chars))


def detect_mime_type(file_path: Path | str) -> str:
    """
    Deterministic, registry-independent MIME type resolver for AirTeach AI attachments.
    Never returns `None` or corrupted Windows registry MIME strings.
    """
    ext = Path(file_path).suffix.lower().strip()
    if ext in EXPLICIT_MIME_MAP:
        return EXPLICIT_MIME_MAP[ext]
    return "application/octet-stream"


def _detect_bomless_utf16_codec(raw_bytes: bytes) -> Optional[str]:
    """
    Detect BOM-less UTF-16LE / UTF-16BE streams by checking periodic NUL bytes so
    valid `cp1258` / `latin-1` byte streams are never mis-decoded as CJK mojibake.
    """
    n = len(raw_bytes)
    if n < 4 or (n % 2) != 0:
        return None
    even_nulls = raw_bytes[0::2].count(0)
    odd_nulls = raw_bytes[1::2].count(0)
    threshold = max(2, n // 8)
    if odd_nulls >= threshold and even_nulls == 0:
        return "utf-16-le"
    if even_nulls >= threshold and odd_nulls == 0:
        return "utf-16-be"
    return None


def decode_text_bytes_safely(raw_bytes: bytes) -> Tuple[str, str]:
    """
    Decode raw text bytes trying `utf-8-sig`, `utf-8`, `utf-16`, `cp1258`, `latin-1`
    in order so Vietnamese text never produces mojibake or `UnicodeDecodeError`.
    Returns `(decoded_text, encoding_used)`.
    """
    if not raw_bytes:
        return "", "utf-8"

    has_utf16_bom = raw_bytes.startswith(codecs.BOM_UTF16_LE) or raw_bytes.startswith(codecs.BOM_UTF16_BE)
    bomless_utf16 = None if has_utf16_bom else _detect_bomless_utf16_codec(raw_bytes)

    for enc in ("utf-8-sig", "utf-8", "utf-16", "cp1258", "latin-1"):
        if enc == "utf-16":
            if has_utf16_bom:
                try:
                    return raw_bytes.decode("utf-16"), "utf-16"
                except UnicodeDecodeError:
                    continue
            if bomless_utf16 is not None:
                try:
                    return raw_bytes.decode(bomless_utf16), bomless_utf16
                except UnicodeDecodeError:
                    continue
            continue
        try:
            return raw_bytes.decode(enc), enc
        except UnicodeDecodeError:
            continue

    return raw_bytes.decode("latin-1", errors="replace"), "latin-1"


@dataclass
class ParsedAttachment:
    path: Path
    filename: str
    mime_type: str
    kind: str  # "image" | "document" | "text"
    extracted_text: str
    image_bytes: Optional[bytes]
    page_count: int
    warning_vi: str = ""


def _compress_png_bytes(png_bytes: Optional[bytes], max_dim: int = MAX_ATTACHMENT_PREVIEW_DIM) -> Optional[bytes]:
    if not png_bytes:
        return None
    try:
        from cv_engine.gemini_client import compress_snapshot_png

        out = compress_snapshot_png(png_bytes, max_dim=max_dim)
        if out and out.startswith(b"\x89PNG"):
            return out
    except Exception:
        pass
    try:
        from PIL import Image

        with Image.open(io.BytesIO(png_bytes)) as im:
            work = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
            if max(work.width, work.height) > max_dim:
                work.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            work.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
    except Exception:
        return png_bytes


def _render_pdf_first_page_png(doc: Any, max_dim: int = MAX_ATTACHMENT_PREVIEW_DIM) -> Optional[bytes]:
    if doc is None or int(getattr(doc, "page_count", 0)) < 1:
        return None
    try:
        from cv_engine.document_presenter import _ensure_fitz

        fitz = _ensure_fitz()
        page0 = doc.load_page(0)
        rect = page0.rect
        longest = max(1.0, float(rect.width), float(rect.height))
        scale = max(0.4, min(2.2, float(max_dim) / longest))
        pix = page0.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB, alpha=False)
        return _compress_png_bytes(pix.tobytes("png"), max_dim=max_dim)
    except Exception:
        return None


def _load_image_attachment_png(src: Path, max_dim: int = MAX_ATTACHMENT_PREVIEW_DIM) -> Optional[bytes]:
    """Load raster/SVG image cleanly (even from Vietnamese Windows paths) -> PNG `<= max_dim`."""
    try:
        from PySide6.QtCore import Qt
        from cv_engine.document_presenter import _qimage_to_png_bytes, load_image_file_qimage

        qimg = load_image_file_qimage(src, target_max_dim=max_dim)
        if qimg is not None and not qimg.isNull():
            if max(qimg.width(), qimg.height()) > max_dim:
                qimg = qimg.scaled(max_dim, max_dim, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            raw_png = _qimage_to_png_bytes(qimg)
            if raw_png:
                return _compress_png_bytes(raw_png, max_dim=max_dim)
    except Exception:
        pass

    # Pure-Pillow fallback (handles headless/non-Qt and non-ASCII paths via bytes)
    try:
        from PIL import Image

        raw_bytes = src.read_bytes()
        with Image.open(io.BytesIO(raw_bytes)) as im:
            work = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
            if max(work.width, work.height) > max_dim:
                work.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            work.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
    except Exception:
        return None


def _extract_pptx_shape_texts(shape: Any, parts: List[str]) -> None:
    sub_shapes = getattr(shape, "shapes", None)
    if sub_shapes is not None:
        try:
            for sub in sub_shapes:
                _extract_pptx_shape_texts(sub, parts)
            return
        except Exception:
            pass

    if getattr(shape, "has_text_frame", False):
        try:
            txt = normalize_vietnamese_text(shape.text_frame.text).strip()
            if txt:
                parts.append(txt)
        except Exception:
            pass

    if getattr(shape, "has_table", False):
        try:
            for row in shape.table.rows:
                cells = [normalize_vietnamese_text(c.text).strip() for c in row.cells]
                row_txt = " | ".join(c for c in cells if c)
                if row_txt:
                    parts.append(row_txt)
        except Exception:
            pass


def _truncate_text(text: str, max_text_chars: int, existing_warning: str = "") -> Tuple[str, str]:
    limit = max(1, int(max_text_chars))
    if len(text) <= limit:
        return text, existing_warning
    warn = f"Nội dung dài ({len(text):,} ký tự), đã rút gọn còn {limit:,} ký tự đầu tiên."
    if existing_warning:
        warn = f"{existing_warning} {warn}"
    return text[:limit], warn


def parse_attachment_file(
    file_path: Path | str,
    max_text_chars: int = 12000,
) -> ParsedAttachment:
    """
    Parse an attached file (text, image, PDF, DOCX/DOC, PPTX/PPT) safely for AI Chatbot context:
    - Text files (`.txt/.md/.csv/.json`): decoded via `utf-8-sig -> utf-8 -> utf-16 -> cp1258 -> latin-1`
      and NFC-normalized.
    - Documents (`.pdf/.docx/.pptx/.doc/.ppt`): extracts clean NFC Vietnamese text (including tables)
      AND renders first-page PNG preview `image_bytes` (`<= 1024px`) for Vision context.
    - Images (`.png/.jpg/.jpeg/.webp/.bmp/.svg`): loads via `load_image_file_qimage` / Pillow and
      compresses to PNG `image_bytes` (`<= 1024px`) with exact `image/png` MIME type.
    """
    src = Path(file_path)
    if not src.exists() or not src.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp đính kèm: {src}")

    filename = normalize_vietnamese_text(src.name)
    ext = src.suffix.lower().strip()
    mime = detect_mime_type(src)

    # 1. Image attachments -> normalized PNG <= 1024px with exact `image/png` MIME
    if ext in IMAGE_ATTACHMENT_EXTENSIONS:
        img_bytes = _load_image_attachment_png(src, max_dim=MAX_ATTACHMENT_PREVIEW_DIM)
        warning = "" if img_bytes else f"Không thể giải mã hình ảnh '{filename}'."
        return ParsedAttachment(
            path=src,
            filename=filename,
            mime_type="image/png",
            kind="image",
            extracted_text="",
            image_bytes=img_bytes,
            page_count=1 if img_bytes else 0,
            warning_vi=warning,
        )

    # 2. Text attachments (.txt, .md, .csv, .json)
    if ext in TEXT_ATTACHMENT_EXTENSIONS:
        raw_bytes = src.read_bytes()
        decoded, _enc = decode_text_bytes_safely(raw_bytes)
        clean_text = normalize_vietnamese_text(decoded)
        clean_text, warning = _truncate_text(clean_text, max_text_chars)
        return ParsedAttachment(
            path=src,
            filename=filename,
            mime_type=mime,
            kind="text",
            extracted_text=clean_text,
            image_bytes=None,
            page_count=1,
            warning_vi=warning,
        )

    # 3. PDF documents
    if ext == ".pdf":
        from cv_engine.document_presenter import _ensure_fitz

        fitz = _ensure_fitz()
        raw_bytes = src.read_bytes()
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
        try:
            page_count = int(doc.page_count)
            page_chunks: List[str] = []
            for idx in range(page_count):
                txt = normalize_vietnamese_text(doc.load_page(idx).get_text("text")).strip()
                if txt:
                    if page_count > 1:
                        page_chunks.append(f"[Trang {idx + 1}]\n{txt}")
                    else:
                        page_chunks.append(txt)
            combined = normalize_vietnamese_text("\n\n".join(page_chunks)).strip()
            preview_png = _render_pdf_first_page_png(doc, max_dim=MAX_ATTACHMENT_PREVIEW_DIM)
        finally:
            doc.close()

        warning = ""
        if not combined:
            warning = "PDF không có lớp văn bản (có thể là bản scan); đã đính kèm ảnh trang đầu cho AI Vision."
        combined, warning = _truncate_text(combined, max_text_chars, warning)
        return ParsedAttachment(
            path=src,
            filename=filename,
            mime_type=mime,
            kind="document",
            extracted_text=combined,
            image_bytes=preview_png,
            page_count=page_count,
            warning_vi=warning,
        )

    # 4. Word (.docx)
    if ext == ".docx":
        from cv_engine.document_presenter import (
            _ensure_fitz,
            _ensure_python_docx,
            _render_docx_pure_python_to_pdf,
        )

        warning = ""
        extracted_blocks: List[str] = []
        if _ensure_python_docx():
            import docx as _docx  # type: ignore

            wdoc = _docx.Document(io.BytesIO(src.read_bytes()))
            para_map = {p._element: p for p in wdoc.paragraphs}
            table_map = {t._element: t for t in wdoc.tables}
            for child in wdoc.element.body:
                if child in para_map:
                    ptxt = normalize_vietnamese_text(para_map[child].text).strip()
                    if ptxt:
                        extracted_blocks.append(ptxt)
                elif child in table_map:
                    for row in table_map[child].rows:
                        cells = [normalize_vietnamese_text(c.text).strip() for c in row.cells]
                        rtxt = " | ".join(c for c in cells if c)
                        if rtxt:
                            extracted_blocks.append(rtxt)
        else:
            warning = "Chưa cài thư viện python-docx để trích xuất văn bản Word."

        preview_png: Optional[bytes] = None
        page_count = 1
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpd:
                tmp_pdf = Path(tmpd) / "preview.pdf"
                ok, _pages = _render_docx_pure_python_to_pdf(src, tmp_pdf)
                if ok and tmp_pdf.exists():
                    fitz = _ensure_fitz()
                    pdoc = fitz.open(stream=tmp_pdf.read_bytes(), filetype="pdf")
                    try:
                        page_count = max(1, int(pdoc.page_count))
                        preview_png = _render_pdf_first_page_png(pdoc, max_dim=MAX_ATTACHMENT_PREVIEW_DIM)
                    finally:
                        pdoc.close()
        except Exception:
            pass

        combined = normalize_vietnamese_text("\n\n".join(extracted_blocks)).strip()
        combined, warning = _truncate_text(combined, max_text_chars, warning)
        return ParsedAttachment(
            path=src,
            filename=filename,
            mime_type=mime,
            kind="document",
            extracted_text=combined,
            image_bytes=preview_png,
            page_count=page_count,
            warning_vi=warning,
        )

    # 5. PowerPoint (.pptx / .ppsx / .potx)
    if ext in (".pptx", ".ppsx", ".potx"):
        from cv_engine.document_presenter import (
            _ensure_fitz,
            _ensure_python_pptx,
            _render_pptx_pure_python_to_pdf,
        )

        warning = ""
        slide_blocks: List[str] = []
        page_count = 0
        if _ensure_python_pptx():
            from pptx import Presentation as _Pres  # type: ignore

            prs = _Pres(io.BytesIO(src.read_bytes()))
            slides_list = list(prs.slides)
            page_count = len(slides_list)
            for idx, slide in enumerate(slides_list):
                parts: List[str] = []
                for shape in slide.shapes:
                    _extract_pptx_shape_texts(shape, parts)
                stxt = normalize_vietnamese_text("\n".join(parts)).strip()
                if stxt:
                    slide_blocks.append(f"[Slide {idx + 1}]\n{stxt}")
        else:
            warning = "Chưa cài thư viện python-pptx để trích xuất nội dung PowerPoint."

        preview_png = None
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpd:
                tmp_pdf = Path(tmpd) / "slide1_preview.pdf"
                ok, _s_texts = _render_pptx_pure_python_to_pdf(src, tmp_pdf, max_slides=1)
                if ok and tmp_pdf.exists():
                    fitz = _ensure_fitz()
                    pdoc = fitz.open(stream=tmp_pdf.read_bytes(), filetype="pdf")
                    try:
                        preview_png = _render_pdf_first_page_png(pdoc, max_dim=MAX_ATTACHMENT_PREVIEW_DIM)
                    finally:
                        pdoc.close()
        except Exception:
            pass

        combined = normalize_vietnamese_text("\n\n".join(slide_blocks)).strip()
        combined, warning = _truncate_text(combined, max_text_chars, warning)
        return ParsedAttachment(
            path=src,
            filename=filename,
            mime_type=mime,
            kind="document",
            extracted_text=combined,
            image_bytes=preview_png,
            page_count=page_count,
            warning_vi=warning,
        )

    # 6. Legacy Word / PowerPoint (.doc / .ppt)
    if ext in (".doc", ".ppt"):
        from cv_engine.document_presenter import (
            _ensure_fitz,
            _render_legacy_doc_fallback_to_pdf,
            _render_legacy_ppt_fallback_to_pdf,
        )

        preview_png = None
        page_count = 0
        extracted_pages: List[str] = []
        warning = ""
        try:
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpd:
                tmp_pdf = Path(tmpd) / "legacy_preview.pdf"
                if ext == ".doc":
                    ok, extracted_pages = _render_legacy_doc_fallback_to_pdf(src, tmp_pdf)
                else:
                    ok, extracted_pages = _render_legacy_ppt_fallback_to_pdf(src, tmp_pdf)
                if ok and tmp_pdf.exists():
                    fitz = _ensure_fitz()
                    pdoc = fitz.open(stream=tmp_pdf.read_bytes(), filetype="pdf")
                    try:
                        page_count = int(pdoc.page_count)
                        preview_png = _render_pdf_first_page_png(pdoc, max_dim=MAX_ATTACHMENT_PREVIEW_DIM)
                    finally:
                        pdoc.close()
        except Exception as exc:
            warning = f"Không thể trích xuất đầy đủ tệp nhị phân cũ '{filename}': {exc}"

        combined = normalize_vietnamese_text("\n\n".join(t for t in extracted_pages if t.strip())).strip()
        combined, warning = _truncate_text(combined, max_text_chars, warning)
        return ParsedAttachment(
            path=src,
            filename=filename,
            mime_type=mime,
            kind="document",
            extracted_text=combined,
            image_bytes=preview_png,
            page_count=page_count,
            warning_vi=warning,
        )

    # 7. Generic fallback: attempt safe UTF-8/text read if non-binary, else graceful warning
    raw_bytes = src.read_bytes()
    if raw_bytes and b"\x00" not in raw_bytes[:4096]:
        decoded, _enc = decode_text_bytes_safely(raw_bytes)
        clean_text = normalize_vietnamese_text(decoded)
        clean_text, warning = _truncate_text(clean_text, max_text_chars)
        return ParsedAttachment(
            path=src,
            filename=filename,
            mime_type=mime,
            kind="text",
            extracted_text=clean_text,
            image_bytes=None,
            page_count=1,
            warning_vi=warning,
        )

    return ParsedAttachment(
        path=src,
        filename=filename,
        mime_type=mime,
        kind="document",
        extracted_text="",
        image_bytes=None,
        page_count=0,
        warning_vi=f"Định dạng tệp '{ext or '(không có đuôi)'}' chưa hỗ trợ trích xuất văn bản trực tiếp.",
    )
