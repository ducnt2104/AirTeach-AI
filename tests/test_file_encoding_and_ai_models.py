#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verification Suite for Vietnamese File Encoding / Attachment Parsing,
Vietnamese Path Hardening, Full Gemini Model Catalog, and Rich Markdown
Code Blocks + Tables Rendering (`tests/test_file_encoding_and_ai_models.py`).
"""

from __future__ import annotations

import io
import json
import os
import sys
import unicodedata
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

from cv_engine.document_presenter import LectureDocument, load_image_file_qimage
from cv_engine.file_attachment_parser import (
    ParsedAttachment,
    detect_mime_type,
    normalize_vietnamese_text,
    parse_attachment_file,
)
from cv_engine.gemini_client import (
    AVAILABLE_GEMINI_MODELS,
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    build_model_chain,
)
from cv_engine.output_sanitizer import OutputSanitizer


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_normalize_vietnamese_text_nfc_and_garbage_stripping() -> None:
    # Decomposed NFD Vietnamese + BOM + null bytes + control chars + broken surrogate + math
    nfd_vn = unicodedata.normalize("NFD", "Phương trình bậc hai: Tiếng Việt chuẩn 100%")
    assert not unicodedata.is_normalized("NFC", nfd_vn)

    dirty = f"\ufeff{nfd_vn}\x00\x07\x1b\ud800\r\n\tΔ = b² - 4ac ≥ 0 → x₁, x₂ ∈ ℝ"
    clean = normalize_vietnamese_text(dirty)

    assert unicodedata.is_normalized("NFC", clean)
    assert "Phương trình bậc hai: Tiếng Việt chuẩn 100%" in clean
    assert "\ufeff" not in clean
    assert "\x00" not in clean
    assert "\x07" not in clean
    assert "\x1b" not in clean
    assert "\r" not in clean
    assert "\n\tΔ = b² - 4ac ≥ 0 → x₁, x₂ ∈ ℝ" in clean


def test_detect_mime_type_explicit_all_formats() -> None:
    expected = {
        "bài_giảng.pdf": "application/pdf",
        "giáo_án.docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "cũ.doc": "application/msword",
        "trình_chiếu.pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "slide_cũ.ppt": "application/vnd.ms-powerpoint",
        "hình.png": "image/png",
        "ảnh.jpg": "image/jpeg",
        "ảnh2.JPEG": "image/jpeg",
        "ảnh3.webp": "image/webp",
        "ảnh4.bmp": "image/bmp",
        "biểu_đồ.svg": "image/svg+xml",
        "ghi_chú.txt": "text/plain",
        "tài_liệu.md": "text/markdown",
        "bảng_điểm.csv": "text/csv",
        "cấu_hình.json": "application/json",
    }
    for fname, mime in expected.items():
        got = detect_mime_type(fname)
        assert got == mime, f"{fname}: expected {mime}, got {got}"

    unknown_mime = detect_mime_type("tệp_lạ.xyzunknown")
    assert isinstance(unknown_mime, str) and len(unknown_mime) > 0
    assert unknown_mime == "application/octet-stream"


def test_parse_text_files_utf8_utf8sig_utf16_cp1258_and_vietnamese_paths(tmp_path: Path) -> None:
    vn_dir = tmp_path / "Thư mục Bài Giảng — Tiếng Việt (Lớp 12)"
    vn_dir.mkdir(parents=True, exist_ok=True)

    sample_nfd = unicodedata.normalize("NFD", "Định luật bảo toàn năng lượng — Lớp 10\nCông thức: W = Wđ + Wt")
    expected_nfc = unicodedata.normalize("NFC", sample_nfd)

    # 1. UTF-8 with BOM (.txt)
    p_utf8_bom = vn_dir / "ghi_chú_utf8_bom.txt"
    p_utf8_bom.write_bytes(b"\xef\xbb\xbf" + sample_nfd.encode("utf-8"))
    att_bom = parse_attachment_file(p_utf8_bom)
    assert isinstance(att_bom, ParsedAttachment)
    assert att_bom.kind == "text"
    assert att_bom.mime_type == "text/plain"
    assert att_bom.image_bytes is None
    assert unicodedata.is_normalized("NFC", att_bom.extracted_text)
    assert "\ufeff" not in att_bom.extracted_text
    assert expected_nfc == att_bom.extracted_text

    # 2. UTF-16 with BOM (.md)
    p_utf16 = vn_dir / "đề_cương_utf16.md"
    p_utf16.write_bytes("# Ôn tập Giải tích\nTích phân xác định ∫ f(x)dx".encode("utf-16"))
    att_u16 = parse_attachment_file(p_utf16)
    assert att_u16.kind == "text"
    assert att_u16.mime_type == "text/markdown"
    assert "Ôn tập Giải tích" in att_u16.extracted_text
    assert "∫ f(x)dx" in att_u16.extracted_text

    # 3. Legacy Windows Vietnamese CP1258 (.csv) — CP1258 uses base vowels (ă,â,đ,ê,ô,ơ,ư) + combining tone marks;
    #    must NOT mis-decode as UTF-16 CJK mojibake AND must compose back to canonical NFC!
    cp1258_src = "Ho\u0323c sinh,Điê\u0309m Toa\u0301n\nNguyê\u0303n Văn An,9.5\nTrâ\u0300n Thi\u0323 Bi\u0300nh,10.0"
    p_cp1258 = vn_dir / "bảng_điểm_cp1258.csv"
    p_cp1258.write_bytes(cp1258_src.encode("cp1258"))
    att_cp = parse_attachment_file(p_cp1258)
    assert att_cp.kind == "text"
    assert att_cp.mime_type == "text/csv"
    assert unicodedata.is_normalized("NFC", att_cp.extracted_text)
    assert "Học sinh,Điểm Toán" in att_cp.extracted_text
    assert "Nguyễn Văn An" in att_cp.extracted_text
    assert "Trần Thị Bình" in att_cp.extracted_text

    # 4. UTF-8 JSON (.json) + truncation check
    p_json = vn_dir / "dữ_liệu_bài_học.json"
    p_json.write_text(
        json.dumps({"chủ_đề": "Hóa học hữu cơ", "mô_tả": "A" * 500}, ensure_ascii=False),
        encoding="utf-8",
    )
    att_json = parse_attachment_file(p_json, max_text_chars=120)
    assert att_json.kind == "text"
    assert att_json.mime_type == "application/json"
    assert len(att_json.extracted_text) <= 120
    assert "Hóa học hữu cơ" in att_json.extracted_text
    assert att_json.warning_vi != ""


def test_parse_pdf_docx_pptx_and_images_with_vietnamese_paths(qapp, tmp_path: Path) -> None:
    vn_dir = tmp_path / "Tài Liệu Giảng Dạy — Đề Thi Thử"
    vn_dir.mkdir(parents=True, exist_ok=True)

    # 1. Real PDF with Vietnamese filename & Unicode text (embedded via HTML Story for full Vietnamese glyph support)
    pdf_path = vn_dir / "Bài giảng Toán — Đạo hàm.pdf"
    pdoc = fitz.open()
    for i in range(2):
        pg = pdoc.new_page(width=800, height=450)
        pg.insert_text((50, 80), f"Trang {i + 1}: Phuong trinh bac hai x^2 - 4x + 3 = 0", fontsize=20)
    pdoc.save(str(pdf_path))
    pdoc.close()

    att_pdf = parse_attachment_file(pdf_path)
    assert att_pdf.kind == "document"
    assert att_pdf.mime_type == "application/pdf"
    assert att_pdf.page_count == 2
    assert "x^2 - 4x + 3 = 0" in att_pdf.extracted_text
    assert att_pdf.image_bytes is not None and att_pdf.image_bytes.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(att_pdf.image_bytes)) as im_pdf:
        assert max(im_pdf.size) <= 1024

    # 2. Real DOCX with Vietnamese NFD text + BOM, table, and Vietnamese path
    docx_path = vn_dir / "Giáo án Vật Lý — Dao Động.docx"
    wdoc = DocxDocument()
    wdoc.add_heading(unicodedata.normalize("NFD", "Giáo án Vật Lý 12: Dao động điều hòa"), level=1)
    wdoc.add_paragraph("\ufeff" + unicodedata.normalize("NFD", "Phương trình li độ: x = A cos(ωt + φ)"))
    tbl = wdoc.add_table(rows=2, cols=2)
    tbl.cell(0, 0).text = "Đại lượng"
    tbl.cell(0, 1).text = "Đơn vị"
    tbl.cell(1, 0).text = "Tần số góc"
    tbl.cell(1, 1).text = "rad/s"
    wdoc.save(str(docx_path))

    att_docx = parse_attachment_file(docx_path)
    assert att_docx.kind == "document"
    assert att_docx.mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert att_docx.page_count >= 1
    assert unicodedata.is_normalized("NFC", att_docx.extracted_text)
    assert "\ufeff" not in att_docx.extracted_text
    assert "Dao động điều hòa" in att_docx.extracted_text
    assert "Tần số góc" in att_docx.extracted_text
    assert att_docx.image_bytes is not None and att_docx.image_bytes.startswith(b"\x89PNG")

    # Verify LectureDocument.get_page_text returns clean NFC Vietnamese text for DOCX
    ldoc_word = LectureDocument.open(docx_path, cache_dir=tmp_path / "cache")
    try:
        pg0_txt = ldoc_word.get_page_text(0)
        assert unicodedata.is_normalized("NFC", pg0_txt)
        assert "\ufeff" not in pg0_txt
        assert "Dao động điều hòa" in pg0_txt or "Phương trình" in pg0_txt
    finally:
        ldoc_word.close()

    # 3. Real PPTX with Vietnamese NFD text + BOM, table, and Vietnamese path
    pptx_path = vn_dir / "Bài Trình Chiếu — Hóa Học Hữu Cơ.pptx"
    prs = Presentation()
    s1 = prs.slides.add_slide(prs.slide_layouts[1])
    s1.shapes.title.text = unicodedata.normalize("NFD", "Phản ứng Este hóa")
    s1.placeholders[1].text = "\ufeff" + unicodedata.normalize("NFD", "Axit cacboxylic + Ancol ⇔ Este + Nước")
    s2 = prs.slides.add_slide(prs.slide_layouts[5])
    t_shape = s2.shapes.add_table(2, 2, 1000000, 1500000, 5000000, 1500000)
    t_shape.table.cell(0, 0).text = "Chất"
    t_shape.table.cell(0, 1).text = "Công thức"
    t_shape.table.cell(1, 0).text = "Etyl axetat"
    t_shape.table.cell(1, 1).text = "CH3COOC2H5"
    prs.save(str(pptx_path))

    att_pptx = parse_attachment_file(pptx_path)
    assert att_pptx.kind == "document"
    assert att_pptx.mime_type == "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    assert att_pptx.page_count == 2
    assert unicodedata.is_normalized("NFC", att_pptx.extracted_text)
    assert "Phản ứng Este hóa" in att_pptx.extracted_text
    assert "Etyl axetat" in att_pptx.extracted_text
    assert att_pptx.image_bytes is not None and att_pptx.image_bytes.startswith(b"\x89PNG")

    ldoc_pptx = LectureDocument.open(pptx_path, cache_dir=tmp_path / "cache")
    try:
        s0_txt = ldoc_pptx.get_page_text(0)
        assert unicodedata.is_normalized("NFC", s0_txt)
        assert "\x00" not in s0_txt
        assert "Phản ứng Este hóa" in s0_txt
    finally:
        ldoc_pptx.close()

    # 4. Large PNG & JPG with Vietnamese file paths -> load_image_file_qimage + parse_attachment_file
    png_path = vn_dir / "Sơ đồ mạch điện — Tiếng Việt.png"
    jpg_path = vn_dir / "Đồ thị hàm số bậc ba.jpg"
    svg_path = vn_dir / "Biểu tượng hình học.svg"

    big_im = Image.new("RGB", (1800, 1200), (20, 40, 80))
    dr = ImageDraw.Draw(big_im)
    dr.rectangle([100, 100, 1700, 1100], outline=(0, 245, 212), width=12)
    big_im.save(str(png_path), format="PNG")
    big_im.save(str(jpg_path), format="JPEG")
    svg_path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200">'
        '<rect width="300" height="200" fill="#1E293B"/>'
        '<circle cx="150" cy="100" r="60" fill="#00F5D4"/></svg>',
        encoding="utf-8",
    )

    for img_file in (png_path, jpg_path, svg_path):
        qimg = load_image_file_qimage(img_file, target_max_dim=1024)
        assert not qimg.isNull(), f"Failed to load Vietnamese image path: {img_file}"
        att_img = parse_attachment_file(img_file)
        assert att_img.kind == "image"
        assert att_img.mime_type == "image/png"
        assert att_img.page_count == 1
        assert att_img.image_bytes is not None and att_img.image_bytes.startswith(b"\x89PNG")
        with Image.open(io.BytesIO(att_img.image_bytes)) as parsed_im:
            assert max(parsed_im.size) <= 1024


def test_available_gemini_models_catalog_and_compat() -> None:
    assert isinstance(AVAILABLE_GEMINI_MODELS, tuple)
    assert len(AVAILABLE_GEMINI_MODELS) == 7

    model_ids = [m_id for m_id, _label in AVAILABLE_GEMINI_MODELS]
    assert model_ids == [
        "",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-3.8-flash",
        "gemini-3.5-flash",
    ]
    for m_id, label in AVAILABLE_GEMINI_MODELS:
        assert isinstance(m_id, str)
        assert isinstance(label, str) and len(label) > 0

    # Existing constants & helpers stay untouched and compatible
    assert DEFAULT_GEMINI_MODEL == "gemini-3.8-flash"
    assert DEFAULT_MAX_OUTPUT_TOKENS == 8192
    assert "gemini-2.5-flash" in DEFAULT_FALLBACK_MODELS

    chain_pro = build_model_chain(primary_model="gemini-2.5-pro")
    assert chain_pro[0] == "gemini-2.5-pro"
    assert "gemini-2.5-flash" in chain_pro


def test_output_sanitizer_rich_html_code_blocks_tables_and_latex_both_themes() -> None:
    sample_md = (
        "## Giải thuật & Bảng Đánh Giá\n\n"
        "Công thức nghiệm phương trình bậc hai:\n"
        "$$x = \\frac{-b \\pm \\sqrt{\\Delta}}{2a}$$\n\n"
        "```python\n"
        "def solve_quadratic(a: float, b: float, c: float) -> tuple:\n"
        "    delta_val = b**2 - 4 * a * c\n"
        "    if delta_val >= 0 and a != 0:\n"
        "        return ((-b + delta_val**0.5) / (2 * a), (-b - delta_val**0.5) / (2 * a))\n"
        "    return ()\n"
        "```\n\n"
        "| Tham số | Công thức | Ý nghĩa |\n"
        "| --- | --- | --- |\n"
        "| Biệt thức | $\\Delta = b^2 - 4ac$ | Xác định số nghiệm |\n"
        "| Tổng nghiệm | $x_1 + x_2 = -\\frac{b}{a}$ | Định lý Vi-ét |\n\n"
        "Kết luận: $\\Delta > 0$ -> có 2 nghiệm phân biệt."
    )

    for dark_mode in (True, False):
        html_out = OutputSanitizer.to_rich_html(sample_md, dark_mode=dark_mode, render_math_images=True)

        # 1. Styled <pre><code> code block preserving verbatim Python syntax (`->`, `_`, indentation)
        assert "<pre" in html_out
        assert "<code" in html_out and "</code></pre>" in html_out
        assert "def solve_quadratic(a: float, b: float, c: float) -&gt; tuple:" in html_out
        assert "delta_val = b**2 - 4 * a * c" in html_out
        assert "    if delta_val &gt;= 0 and a != 0:" in html_out

        # 2. Complete bordered HTML <table> with headers AND body rows (even with inline LaTeX in cells!)
        assert "<table" in html_out and "</table>" in html_out
        assert "<thead" in html_out and "<tbody" in html_out
        assert html_out.count("<th ") == 3
        assert html_out.count("<td ") == 6
        assert "Tham số" in html_out
        assert "Định lý Vi-ét" in html_out

        # 3. LaTeX display math + SVG arrows rendered without leaking raw `$` or `<arrow>`
        assert "$$" not in html_out
        assert "data:image/svg+xml;base64," in html_out
        assert "data:image/png;base64," in html_out
