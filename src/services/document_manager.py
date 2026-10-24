"""
AirTeach AI - Document Manager & .airteach Package Handler
============================================================
Handles:
  - PDF import (pypdfium2 rendering + pypdf text extraction)
  - PPTX/DOCX import (MS Office COM → PDF, or LibreOffice headless → PDF)
  - .airteach package format (ZIP-based lesson bundle)
  - Annotated PDF export (background + ink vector overlay)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pypdfium2 as pdfium
from pypdf import PdfReader

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PageInfo:
    """Metadata for a single rendered document page."""
    page_index: int           # 0-based
    image_path: str           # absolute path to rendered PNG
    width_px: int
    height_px: int
    text: str                 # extracted text (may be empty for image-only PDFs)


@dataclass
class BezierPointRecord:
    """Serializable record of one Bézier control point with timestamp."""
    x: float
    y: float
    t_ms: float
    pressure: float = 1.0


@dataclass
class StrokeRecord:
    """Serializable ink stroke for the .airteach format."""
    stroke_id: str
    page_index: int
    color_hex: str            # e.g. "#000000"
    width: float
    points: list[BezierPointRecord] = field(default_factory=list)
    bezier_segments: list[dict] = field(default_factory=list)   # BezierSegment.to_dict()
    timestamps_ms: list[float] = field(default_factory=list)    # per-point timestamp array
    shape_type: str = "FREEFORM"


@dataclass
class RecognizedItem:
    """OCR result stored in the .airteach package."""
    item_id: str
    page_index: int
    text: str
    latex: str
    confidence: float
    bounding_box: list[float]   # [x_min, y_min, x_max, y_max] in canvas coords
    is_formula: bool


@dataclass
class AirTeachManifest:
    """Root metadata for a .airteach lesson bundle."""
    version: str = "1.0"
    title: str = ""
    teacher_id: str = ""
    original_filename: str = ""
    original_sha256: str = ""
    page_count: int = 0
    created_at: str = ""
    modified_at: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# PDF Renderer (pypdfium2)
# ─────────────────────────────────────────────────────────────────────────────

class PdfRenderer:
    """Renders PDF pages to PNG images using pypdfium2."""

    def __init__(self, scale: float = 2.0) -> None:
        self.scale = scale  # 1.0 = 72 DPI; 2.0 = 144 DPI; 4.0 = 288 DPI

    def render_file(
        self,
        pdf_path: Path,
        output_dir: Path,
        extract_text: bool = True,
    ) -> list[PageInfo]:
        """
        Render every page of *pdf_path* to a PNG in *output_dir*.

        Args:
            pdf_path:     Path to the PDF file.
            output_dir:   Directory where PNG images will be written.
            extract_text: Also extract text content via pypdf.

        Returns:
            List of PageInfo objects (one per page).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        pages: list[PageInfo] = []

        # Text extraction
        text_by_page: dict[int, str] = {}
        if extract_text:
            try:
                reader = PdfReader(str(pdf_path))
                for i, pg in enumerate(reader.pages):
                    text_by_page[i] = pg.extract_text() or ""
            except Exception as exc:
                logger.warning("pypdf text extraction failed: %s", exc)

        # Rendering
        doc = pdfium.PdfDocument(str(pdf_path))
        try:
            for i in range(len(doc)):
                page = doc[i]
                bitmap = page.render(scale=self.scale, rotation=0)
                pil_img = bitmap.to_pil()

                img_path = output_dir / f"page_{i + 1:03d}.png"
                pil_img.save(str(img_path), format="PNG", optimize=False)

                w, h = pil_img.size
                pages.append(PageInfo(
                    page_index=i,
                    image_path=str(img_path),
                    width_px=w,
                    height_px=h,
                    text=text_by_page.get(i, ""),
                ))
                page.close()
        finally:
            doc.close()

        logger.info("Rendered %d pages from %s", len(pages), pdf_path)
        return pages


# ─────────────────────────────────────────────────────────────────────────────
# Office Converter (PPTX/DOCX → PDF)
# ─────────────────────────────────────────────────────────────────────────────

class OfficeConverter:
    """
    Converts PPTX and DOCX files to PDF using:
      1. MS Office COM automation (Windows only, if Office is installed).
      2. LibreOffice --headless (cross-platform fallback).
    """

    @staticmethod
    def _try_msoffice_com(input_path: Path, output_pdf: Path) -> bool:
        """Attempt conversion via MS Office COM. Returns True on success."""
        if platform.system() != "Windows":
            return False
        try:
            import comtypes.client  # type: ignore  # noqa: PLC0415
        except ImportError:
            return False

        suffix = input_path.suffix.lower()
        app = None
        doc_or_prs = None
        try:
            if suffix in (".pptx", ".ppt"):
                app = comtypes.client.CreateObject("PowerPoint.Application")
                app.Visible = False
                doc_or_prs = app.Presentations.Open(str(input_path.resolve()), WithWindow=False)
                doc_or_prs.SaveAs(str(output_pdf.resolve()), 32)  # 32 = ppSaveAsPDF
                doc_or_prs.Close()
                doc_or_prs = None
                app.Quit()
                app = None
            elif suffix in (".docx", ".doc"):
                app = comtypes.client.CreateObject("Word.Application")
                app.Visible = False
                doc_or_prs = app.Documents.Open(str(input_path.resolve()))
                doc_or_prs.SaveAs2(str(output_pdf.resolve()), FileFormat=17)  # 17 = wdFormatPDF
                doc_or_prs.Close()
                doc_or_prs = None
                app.Quit()
                app = None
            else:
                return False
            logger.info("MS Office COM conversion: %s → %s", input_path.name, output_pdf.name)
            return output_pdf.exists()
        except Exception as exc:
            logger.warning("MS Office COM conversion failed: %s", exc)
            return False
        finally:
            if doc_or_prs is not None:
                try:
                    doc_or_prs.Close()
                except Exception:
                    pass
            if app is not None:
                try:
                    app.Quit()
                except Exception:
                    pass

    @staticmethod
    def _try_libreoffice(input_path: Path, output_dir: Path) -> Optional[Path]:
        """
        Call LibreOffice headless to convert to PDF.
        Returns path to the generated PDF, or None on failure.
        """
        lo_candidates = [
            "libreoffice", "soffice",
            r"C:\Program Files\LibreOffice\program\soffice.exe",
            r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            "/usr/bin/libreoffice", "/usr/bin/soffice",
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        ]
        lo_exe = None
        for candidate in lo_candidates:
            if shutil.which(candidate) or Path(candidate).exists():
                lo_exe = candidate
                break

        if lo_exe is None:
            logger.warning("LibreOffice not found on PATH.")
            return None

        cmd = [
            lo_exe,
            "--headless",
            "--convert-to", "pdf",
            "--outdir", str(output_dir),
            str(input_path),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if result.returncode != 0:
                logger.warning("LibreOffice stderr: %s", result.stderr)
                return None
            expected_pdf = output_dir / (input_path.stem + ".pdf")
            if expected_pdf.exists():
                logger.info("LibreOffice conversion: %s → %s", input_path.name, expected_pdf.name)
                return expected_pdf
        except subprocess.TimeoutExpired:
            logger.error("LibreOffice conversion timed out for %s", input_path)
        except Exception as exc:
            logger.warning("LibreOffice conversion error: %s", exc)
        return None

    def convert_to_pdf(self, input_path: Path, work_dir: Path) -> Optional[Path]:
        """
        Convert *input_path* (PPTX or DOCX) to a PDF in *work_dir*.

        Returns path to the converted PDF, or None if all methods fail.
        """
        output_pdf = work_dir / (input_path.stem + "_converted.pdf")

        # Method 1: MS Office COM
        if self._try_msoffice_com(input_path, output_pdf):
            return output_pdf

        # Method 2: LibreOffice
        pdf_path = self._try_libreoffice(input_path, work_dir)
        if pdf_path is not None:
            return pdf_path

        logger.error(
            "Cannot convert '%s': neither MS Office COM nor LibreOffice available.",
            input_path.name,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# .airteach Package Manager
# ─────────────────────────────────────────────────────────────────────────────

class AirTeachPackage:
    """
    Reads and writes the .airteach ZIP lesson bundle format.

    Internal ZIP structure:
        manifest.json
        pages/page_001.png, page_002.png, ...
        ink/strokes.json
        recognized/items.json
        links.json
    """

    MANIFEST_PATH = "manifest.json"
    INK_PATH = "ink/strokes.json"
    RECOGNIZED_PATH = "recognized/items.json"
    LINKS_PATH = "links.json"

    # ── Save ──────────────────────────────────────────────────────────────────

    @classmethod
    def save(
        cls,
        output_path: Path,
        manifest: AirTeachManifest,
        page_image_paths: list[Path],
        strokes: list[StrokeRecord],
        recognized: list[RecognizedItem],
        original_file_path: Optional[Path] = None,
    ) -> None:
        """
        Create a .airteach file at *output_path*.

        Args:
            output_path:       Where to write the .airteach file.
            manifest:          Lesson metadata.
            page_image_paths:  Ordered list of page PNG paths.
            strokes:           All ink strokes across all pages.
            recognized:        All OCR results.
            original_file_path: Original source document for SHA-256 link.
        """
        manifest.page_count = len(page_image_paths)
        manifest.modified_at = datetime.now(timezone.utc).isoformat()
        if not manifest.created_at:
            manifest.created_at = manifest.modified_at

        # Compute SHA-256 of original source file
        links_data: dict = {}
        if original_file_path and original_file_path.exists():
            sha256 = _sha256_file(original_file_path)
            links_data = {
                "original_filename": original_file_path.name,
                "original_sha256": sha256,
                "original_path": str(original_file_path),
            }
            manifest.original_filename = original_file_path.name
            manifest.original_sha256 = sha256

        # Strokes JSON: each StrokeRecord → dict
        strokes_data = [_stroke_to_dict(s) for s in strokes]

        # RecognizedItems JSON
        recognized_data = [
            {
                "item_id": r.item_id,
                "page_index": r.page_index,
                "text": r.text,
                "latex": r.latex,
                "confidence": r.confidence,
                "bounding_box": r.bounding_box,
                "is_formula": r.is_formula,
            }
            for r in recognized
        ]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = output_path.with_name(f"{output_path.stem}_{uuid.uuid4().hex[:8]}.tmp")

        try:
            with zipfile.ZipFile(str(temp_path), "w", compression=zipfile.ZIP_DEFLATED) as zf:
                # manifest.json
                zf.writestr(cls.MANIFEST_PATH, json.dumps(asdict(manifest), ensure_ascii=False, indent=2))

                # pages/
                for i, img_path in enumerate(page_image_paths):
                    if img_path.exists():
                        arcname = f"pages/page_{i + 1:03d}.png"
                        zf.write(str(img_path), arcname)

                # ink/strokes.json
                zf.writestr(cls.INK_PATH, json.dumps(strokes_data, ensure_ascii=False, indent=2))

                # recognized/items.json
                zf.writestr(cls.RECOGNIZED_PATH, json.dumps(recognized_data, ensure_ascii=False, indent=2))

                # links.json
                zf.writestr(cls.LINKS_PATH, json.dumps(links_data, ensure_ascii=False, indent=2))

            temp_path.replace(output_path)
            logger.info("Saved .airteach package atomically: %s (%d pages, %d strokes)",
                        output_path, len(page_image_paths), len(strokes))
        except Exception:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            raise

    # ── Load ──────────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        airteach_path: Path,
        extract_dir: Path,
    ) -> tuple[AirTeachManifest, list[Path], list[StrokeRecord], list[RecognizedItem]]:
        """
        Extract and parse a .airteach file.

        Args:
            airteach_path: Path to the .airteach file.
            extract_dir:   Directory to extract page images into.

        Returns:
            (manifest, page_image_paths, strokes, recognized_items)
        """
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(str(airteach_path), "r") as zf:
            # Read manifest
            manifest_json = json.loads(zf.read(cls.MANIFEST_PATH).decode("utf-8"))
            manifest = AirTeachManifest(**manifest_json)

            # Extract page images
            page_paths: list[Path] = []
            for name in sorted(zf.namelist()):
                if name.startswith("pages/") and name.endswith(".png"):
                    target = extract_dir / Path(name).name
                    target.write_bytes(zf.read(name))
                    page_paths.append(target)

            # Read ink strokes
            strokes: list[StrokeRecord] = []
            if cls.INK_PATH in zf.namelist():
                strokes_raw = json.loads(zf.read(cls.INK_PATH).decode("utf-8"))
                strokes = [_dict_to_stroke(d) for d in strokes_raw]

            # Read recognized items
            recognized: list[RecognizedItem] = []
            if cls.RECOGNIZED_PATH in zf.namelist():
                rec_raw = json.loads(zf.read(cls.RECOGNIZED_PATH).decode("utf-8"))
                recognized = [
                    RecognizedItem(
                        item_id=d["item_id"],
                        page_index=d["page_index"],
                        text=d["text"],
                        latex=d["latex"],
                        confidence=d["confidence"],
                        bounding_box=d["bounding_box"],
                        is_formula=d["is_formula"],
                    )
                    for d in rec_raw
                ]

        logger.info("Loaded .airteach: %s (%d pages, %d strokes)",
                    airteach_path, len(page_paths), len(strokes))
        return manifest, page_paths, strokes, recognized


# ─────────────────────────────────────────────────────────────────────────────
# Document Manager (Main Facade)
# ─────────────────────────────────────────────────────────────────────────────

class DocumentManager:
    """
    High-level document import and lesson management.

    Workflow:
      1. import_document(path) → returns list[PageInfo]
      2. (Teacher annotates slides with hand gestures)
      3. save_lesson(output_path, strokes, ...) → writes .airteach file
      4. open_lesson(airteach_path) → loads pages + strokes
    """

    def __init__(self, work_root: Path) -> None:
        self.work_root = work_root
        self.work_root.mkdir(parents=True, exist_ok=True)
        self._renderer = PdfRenderer(scale=2.0)
        self._converter = OfficeConverter()

    def import_document(
        self,
        source_path: Path,
        render_dpi_scale: float = 2.0,
    ) -> list[PageInfo]:
        """
        Import a PDF, PPTX, or DOCX file and render all pages to PNG.

        Args:
            source_path:     Path to the source document.
            render_dpi_scale: Rendering scale (2.0 = 144 DPI).

        Returns:
            List of PageInfo objects, one per page.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError:        If the file format is unsupported.
            RuntimeError:      If conversion or rendering fails.
        """
        if not source_path.exists():
            raise FileNotFoundError(f"Document not found: {source_path}")

        suffix = source_path.suffix.lower()
        work_dir = self.work_root / f"import_{source_path.stem}_{int(datetime.now().timestamp())}"
        work_dir.mkdir(parents=True, exist_ok=True)

        if suffix == ".pdf":
            pdf_path = source_path
        elif suffix in (".pptx", ".ppt", ".docx", ".doc"):
            pdf_path = self._converter.convert_to_pdf(source_path, work_dir)
            if pdf_path is None:
                raise RuntimeError(
                    f"Cannot convert '{source_path.name}' to PDF. "
                    "Install Microsoft Office or LibreOffice."
                )
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

        self._renderer.scale = render_dpi_scale
        pages = self._renderer.render_file(pdf_path, work_dir / "pages")
        return pages

    def save_lesson(
        self,
        output_path: Path,
        manifest: AirTeachManifest,
        page_image_paths: list[Path],
        strokes: list[StrokeRecord],
        recognized: list[RecognizedItem],
        original_file_path: Optional[Path] = None,
    ) -> None:
        """Save the current lesson as a .airteach file."""
        AirTeachPackage.save(
            output_path=output_path,
            manifest=manifest,
            page_image_paths=page_image_paths,
            strokes=strokes,
            recognized=recognized,
            original_file_path=original_file_path,
        )

    def open_lesson(
        self,
        airteach_path: Path,
    ) -> tuple[AirTeachManifest, list[Path], list[StrokeRecord], list[RecognizedItem]]:
        """Open a .airteach file and return its contents."""
        extract_dir = self.work_root / f"open_{airteach_path.stem}"
        return AirTeachPackage.load(airteach_path, extract_dir)

    def export_annotated_pdf(
        self,
        page_image_paths: list[Path],
        strokes: list[StrokeRecord],
        output_pdf_path: Path,
    ) -> None:
        """
        Export pages with ink overlay as an annotated PDF.

        Uses reportlab-free approach: renders each page+ink to PNG,
        then assembles via pypdfium2 or Pillow image concatenation
        saved as a multi-page PDF.
        """
        import io as _io  # noqa: PLC0415
        from PIL import Image as _PILImage  # noqa: PLC0415
        import cv2 as _cv2  # noqa: PLC0415

        stroke_by_page: dict[int, list[StrokeRecord]] = {}
        for s in strokes:
            stroke_by_page.setdefault(s.page_index, []).append(s)

        annotated_pages: list[_PILImage.Image] = []

        for page_idx, img_path in enumerate(page_image_paths):
            if not img_path.exists():
                logger.warning("Page image missing: %s", img_path)
                continue

            bgr = _cv2.imread(str(img_path))
            if bgr is None:
                continue

            # Compute canvas (1920x1080) to image coordinate mapping
            bgr_h, bgr_w = bgr.shape[:2]
            scale_factor = min(1920.0 / bgr_w, 1080.0 / bgr_h)
            if scale_factor < 1e-6:
                scale_factor = 1.0
            scaled_w = bgr_w * scale_factor
            scaled_h = bgr_h * scale_factor
            ox = (1920.0 - scaled_w) / 2.0
            oy = (1080.0 - scaled_h) / 2.0

            # Draw strokes onto the page image
            for stroke in stroke_by_page.get(page_idx, []):
                eval_pts: list[tuple[int, int]] = []
                if stroke.bezier_segments:
                    for b in stroke.bezier_segments:
                        p0 = np.array(b["p0"], dtype=np.float64)
                        p1 = np.array(b["p1"], dtype=np.float64)
                        p2 = np.array(b["p2"], dtype=np.float64)
                        p3 = np.array(b["p3"], dtype=np.float64)
                        for t in np.linspace(0.0, 1.0, 8):
                            u = 1.0 - t
                            pt = u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3
                            ix = int(round((pt[0] - ox) / scale_factor))
                            iy = int(round((pt[1] - oy) / scale_factor))
                            eval_pts.append((ix, iy))
                elif stroke.points:
                    for p in stroke.points:
                        ix = int(round((p.x - ox) / scale_factor))
                        iy = int(round((p.y - oy) / scale_factor))
                        eval_pts.append((ix, iy))

                color_bgr = _hex_to_bgr(stroke.color_hex)
                thickness = max(1, int(round(stroke.width / scale_factor)))
                for j in range(1, len(eval_pts)):
                    _cv2.line(bgr, eval_pts[j - 1], eval_pts[j], color_bgr, thickness, _cv2.LINE_AA)

            rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
            annotated_pages.append(_PILImage.fromarray(rgb))

        if not annotated_pages:
            raise RuntimeError("No pages to export.")

        output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        first = annotated_pages[0]
        rest = annotated_pages[1:]
        first.save(
            str(output_pdf_path),
            format="PDF",
            save_all=True,
            append_images=rest,
            resolution=144,
        )
        logger.info("Exported annotated PDF: %s (%d pages)", output_pdf_path, len(annotated_pages))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _stroke_to_dict(s: StrokeRecord) -> dict:
    return {
        "stroke_id": s.stroke_id,
        "page_index": s.page_index,
        "color_hex": s.color_hex,
        "width": s.width,
        "shape_type": s.shape_type,
        "points": [
            {"x": p.x, "y": p.y, "t_ms": p.t_ms, "pressure": p.pressure}
            for p in s.points
        ],
        "bezier_segments": s.bezier_segments,
        "timestamps_ms": s.timestamps_ms,
    }


def _dict_to_stroke(d: dict) -> StrokeRecord:
    points = [
        BezierPointRecord(
            x=pt["x"], y=pt["y"], t_ms=pt["t_ms"], pressure=pt.get("pressure", 1.0)
        )
        for pt in d.get("points", [])
    ]
    return StrokeRecord(
        stroke_id=d["stroke_id"],
        page_index=d["page_index"],
        color_hex=d["color_hex"],
        width=d["width"],
        shape_type=d.get("shape_type", "FREEFORM"),
        points=points,
        bezier_segments=d.get("bezier_segments", []),
        timestamps_ms=d.get("timestamps_ms", []),
    )


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert #RRGGBB hex to (B, G, R) tuple."""
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (0, 0, 0)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)
