#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 Verification Suite: Multi-Format Lecture Presentation
(PDF, PPTX, DOCX, PNG/JPG/WEBP/BMP, Per-Page Object Annotation Isolation,
Sharp 4x Zoom Rendering, Thumbnail Strip, Annotated PDF & PNG Export).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pymupdf as fitz
from docx import Document as DocxDocument
from PIL import Image, ImageDraw
from pptx import Presentation
from PySide6.QtWidgets import QApplication

import main as app_main
from cv_engine.document_presenter import (
    LectureDocument,
    export_annotated_lecture_pdf,
    export_annotated_slides_to_pngs,
    get_office_conversion_notice,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_multiformat_pdf_pptx_docx_images_per_page_isolation_and_exports(qapp, tmp_path: Path):
    # 1. Create real 2-page PDF
    pdf_path = tmp_path / "bai_giang_toan.pdf"
    pdf_doc = fitz.open()
    for i in range(2):
        p = pdf_doc.new_page(width=960, height=540)
        p.insert_text((72, 90), f"AirTeach PDF Trang {i + 1}: Phuong trinh bac hai", fontsize=24)
    pdf_doc.save(str(pdf_path))
    pdf_doc.close()

    # 2. Create real 2-slide PPTX
    pptx_path = tmp_path / "bai_giang_ly.pptx"
    prs = Presentation()
    for i in range(2):
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = f"Slide Vat Ly {i + 1}"
        s.placeholders[1].text = f"Dinh luat Newton thu {i + 1}: F = m * a"
    prs.save(str(pptx_path))

    # 3. Create real DOCX
    docx_path = tmp_path / "giao_an_hoa.docx"
    wdoc = DocxDocument()
    wdoc.add_heading("Giáo án Hóa học lớp 10", level=1)
    wdoc.add_paragraph("Phản ứng oxi hóa - khử và cân bằng phương trình hóa học.")
    wdoc.save(str(docx_path))

    # 4. Create real PNG and JPG images
    png_path = tmp_path / "hinh_minh_hoa.png"
    jpg_path = tmp_path / "do_thi.jpg"
    im = Image.new("RGB", (640, 360), (240, 249, 255))
    dr = ImageDraw.Draw(im)
    dr.rectangle([40, 40, 600, 320], outline=(79, 70, 229), width=6)
    im.save(str(png_path), format="PNG")
    im.save(str(jpg_path), format="JPEG")

    # Verify PDF, PPTX, DOCX open cleanly & return crisp images
    canvas = app_main.SmartboardCanvas()
    canvas.resize(1280, 720)
    try:
        # Open PDF
        assert canvas.load_document_file(pdf_path)
        assert canvas.lecture_doc is not None and canvas.lecture_doc.page_count == 2
        # Sharp 4.0x high-DPI render (for 400% zoom)
        hi_qimg = canvas.lecture_doc.render_page_qimage(0, scale=4.0)
        assert not hi_qimg.isNull() and hi_qimg.width() >= 2000

        # Add isolated Object annotations on Page 1 ONLY (shape + text + sticky + image)
        canvas.set_current_slide(0)
        canvas.add_shape_object("circle", 200.0, 180.0, 120.0, 120.0, select=False)
        canvas.add_text_object("Ghi chú trang 1", 400.0, 200.0, select=False)
        canvas.add_sticky_object("Công thức cần nhớ", 640.0, 200.0, select=False)
        assert canvas.insert_image_on_board(png_path)
        assert len(canvas.scene.objects) == 4

        # Flip to Page 2 -> ZERO objects from Page 1!
        assert canvas.set_current_slide(1)
        assert len(canvas.scene.objects) == 0
        assert len(canvas.strokes) == 0 and len(canvas.annotations) == 0 and len(canvas.board_images) == 0

        # Add 1 distinct object on Page 2
        canvas.add_sticky_object("Chỉ ở trang 2", 300.0, 240.0, select=False)
        assert len(canvas.scene.objects) == 1

        # Return to Page 1 -> all 4 objects intact!
        assert canvas.set_current_slide(0)
        assert len(canvas.scene.objects) == 4

        # Export annotated deck to BOTH PDF and PNGs
        out_pdf = tmp_path / "exported_annotated.pdf"
        saved_pdf = export_annotated_lecture_pdf(
            out_pdf,
            canvas.slides,
            canvas.slide_annotations,
            (float(canvas.width()), float(canvas.height())),
            lecture_doc=canvas.lecture_doc,
        )
        assert saved_pdf.exists() and saved_pdf.stat().st_size > 1000

        png_out_dir = tmp_path / "exported_pngs"
        saved_pngs = export_annotated_slides_to_pngs(
            png_out_dir,
            canvas.slides,
            canvas.slide_annotations,
            (float(canvas.width()), float(canvas.height())),
            lecture_doc=canvas.lecture_doc,
            scale=1.5,
        )
        assert len(saved_pngs) == 2
        for p_img in saved_pngs:
            assert p_img.exists() and p_img.stat().st_size > 1000

        # Open PPTX & check transparent conversion status notice
        assert canvas.load_document_file(pptx_path)
        assert canvas.lecture_doc is not None and canvas.lecture_doc.page_count == 2
        pptx_notice = get_office_conversion_notice(canvas.lecture_doc.doc_type, canvas.lecture_doc.backend_used)
        assert len(pptx_notice) > 10

        # Open DOCX & check conversion notice
        assert canvas.load_document_file(docx_path)
        assert canvas.lecture_doc is not None and canvas.lecture_doc.page_count >= 1
        docx_notice = get_office_conversion_notice(canvas.lecture_doc.doc_type, canvas.lecture_doc.backend_used)
        assert len(docx_notice) > 10

        # Open JPG as slide background
        assert canvas.load_document_file(jpg_path)
        assert canvas.slides[-1]["pixmap"] is not None
    finally:
        if canvas.lecture_doc is not None:
            canvas.lecture_doc.close()
        canvas.close()
