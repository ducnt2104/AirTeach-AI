---
name: airteach-document-import
description: >-
  Specialist skill for multi-format document import (PDF, PPTX, DOCX) and .airteach lesson packaging.
  Covers pypdfium2 high-resolution rendering, MS Office COM / LibreOffice headless conversion, and ZIP archives.
---

# AirTeach AI — Document Import & Packaging Skill

## 1. Dual-Path Conversion Pipeline
- **PDF Documents**: Direct vector rendering using `pypdfium2` at 144–200 DPI into high-resolution PNG page images. Fast, deterministic, zero external office dependencies.
- **Office Documents (PPTX / DOCX)**:
  - Windows COM Automation (`comtypes` / `win32com`): Used when Microsoft Office (PowerPoint / Word) is installed. Highest layout fidelity.
  - Headless LibreOffice: Fallback command:
    `soffice --headless --convert-to pdf:writer_pdf_Export --outdir <target> <source>`
  - The generated intermediate PDF is then rendered page-by-page via `pypdfium2`.

## 2. `.airteach` Native Package Standard
The `.airteach` format is a standard ZIP archive containing:
```
manifest.json      -> Format version, lesson metadata, page count, canvas resolution
pages/             -> Rendered background images (page_000.png, page_001.png, ...)
ink/strokes.json   -> Serialized Bézier strokes, control points, width, color, timestamps_ms
recognized/        -> OCR text and LaTeX formulas with bounding box and anchor positions
links.json         -> SHA-256 hash and metadata of the original source file
```
Always use atomic file writing (`tempfile` + `shutil.move`) to protect against corruption during power loss.
