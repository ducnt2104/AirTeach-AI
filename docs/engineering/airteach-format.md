# AirTeach AI — The `.airteach` Package Format & Stroke Replay

**Component:** `src/services/document_manager.py` & `src/ui/lesson_board.py`  
**File Extension:** `.airteach`  
**Internal Structure:** Standard Deflate ZIP Archive

---

## 1. Specification of the `.airteach` Bundle

An `.airteach` file is a self-contained, offline package preserving the complete state of a lesson:

```text
lesson_package.airteach  (ZIP container)
│
├── manifest.json            # Package metadata, author, schema version, page list
├── pages/                   # High-resolution rendered background images
│   ├── page_000.png
│   ├── page_001.png
│   └── ...
├── ink/
│   └── strokes.json         # Array of vector ink strokes with millisecond timestamps
├── recognized/
│   └── items.json           # OCR text annotations & rendered LaTeX math item metadata
└── links.json               # SHA-256 hash of original imported document (tamper check)
```

### `manifest.json` Schema
```json
{
  "airteach_version": "1.0.0",
  "title": "Chuyên đề Hình học Không gian - Lớp 11",
  "subject": "Toán học",
  "author": "Thầy Nguyễn Văn A",
  "created_at": "2026-10-22T00:00:00Z",
  "updated_at": "2026-10-22T00:15:00Z",
  "page_count": 5,
  "pages": [
    {
      "index": 0,
      "width_px": 1920,
      "height_px": 1080,
      "image_rel_path": "pages/page_000.png"
    }
  ]
}
```

### `ink/strokes.json` Stroke Schema
```json
[
  {
    "stroke_id": "strk_9a8b7c6d",
    "page_index": 0,
    "color": "#1a73e8",
    "width": 3.5,
    "points": [
      {"x": 150.2, "y": 240.5, "t_ms": 120.0},
      {"x": 155.8, "y": 242.1, "t_ms": 136.6}
    ],
    "bezier_segments": [
      {
        "p0": [150.2, 240.5],
        "p1": [152.0, 241.0],
        "p2": [154.0, 241.8],
        "p3": [155.8, 242.1],
        "width": 3.5
      }
    ]
  }
]
```

---

## 2. Stroke Replay Architecture (`StrokeReplayController`)

Each stroke point records a millisecond timestamp `t_ms` relative to stroke initiation. The Replay engine recreates the exact sequence of teacher writing:
- **Playback Controls:** Play, Pause, Resume, Reset, and Jump to Timestamp.
- **Speed Multipliers:** $0.5\times$, $1.0\times$, $2.0\times$, $5.0\times$.
- **Non-Destructive:** Replaying strokes displays an animated playback layer without mutating the saved underlying vector database.
