# AirTeach AI — Multi-Format Document Importer

**Component:** `src/services/document_manager.py`  
**Formats Supported:** `.pdf`, `.pptx`, `.docx`  
**Rendering Engine:** `pypdfium2` (PDFium) + `pypdf` + MS Office COM / LibreOffice Headless

---

## 1. Dual Conversion Strategy (Windows Office COM / LibreOffice)

To guarantee that teachers' presentation slides and lecture notes preserve 100% of their fonts, math formatting, and image layouts, AirTeach AI deploys an automated dual-conversion strategy:

```mermaid
flowchart TD
    FILE["Imported Document (.pptx, .docx, .pdf)"] --> EXT{Extension?}
    
    EXT -->|PDF| RENDER["Render with pypdfium2 (DPI=150)"]
    
    EXT -->|PPTX / DOCX| CHECK_COM{Windows MS Office COM Available?}
    
    CHECK_COM -->|Yes| COM["Call PowerPoint.Application or Word.Application COM API"]
    COM --> OUT_PDF["Generate Intermediate PDF"]
    
    CHECK_COM -->|No| CHECK_LIBRE{LibreOffice Headless Available?}
    CHECK_LIBRE -->|Yes| LIBRE["Execute: soffice --headless --convert-to pdf"]
    LIBRE --> OUT_PDF
    
    CHECK_LIBRE -->|No| ERR["Raise Clear Error Prompt with Installation Instructions"]
    
    OUT_PDF --> RENDER
    RENDER --> PAGES["Export pages/page_000.png ... page_NNN.png"]
    PAGES --> MANIFEST["Build PageInfo List & Extract Text via pypdf"]
    MANIFEST --> BOARD["Load onto LessonBoardCanvas"]
```

---

## 2. PDF Rendering Precision

- **Resolution:** 150 DPI (equivalent to $1920 \times 1080$ Full HD canvas for standard 16:9 widescreen presentations).
- **Text Extraction:** Uses `pypdf.PdfReader` to extract embedded text per page, allowing the Gemini AI Assistant and search features to read slide content without OCR.
- **Memory Efficiency:** Pages are rendered to disk (`data/work/<doc_id>/pages/page_XXX.png`) and lazy-loaded into Qt memory only as the teacher navigates between slides.
