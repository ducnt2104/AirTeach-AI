---
name: airteach-pyside6-ui
description: >-
  Specialist skill for PySide6/Qt desktop UI architecture in AirTeach AI.
  Covers QGraphicsView/QGraphicsScene layered rendering, custom QGraphicsItem,
  signal/slot thread-safe patterns, high-DPI scaling, and styling.
---

# AirTeach AI — PySide6 UI Architecture Skill

## 1. Scene Layering Architecture (3-Tier Z-Value)
In AirTeach AI's interactive lesson board, all graphics items must adhere to strict Z-index layering:
- `Z = 0`: Background document page / template layer (`QGraphicsPixmapItem`). Locked against user erase actions.
- `Z = 10`: Persistent vector ink strokes (`QGraphicsPathItem`) and recognized LaTeX math / text overlays (`QGraphicsPixmapItem` / `QGraphicsTextItem`).
- `Z = 20`: Transient tool layer: Laser pointer (`QGraphicsEllipseItem`), selection rectangle, shape snap preview, and gesture HUD.

## 2. Thread Safety Rules
- NEVER update Qt GUI objects (`QWidget`, `QGraphicsItem`, `QGraphicsScene`) directly from a background worker thread (`threading.Thread` or `QRunnable`).
- ALWAYS bridge worker outputs to the UI thread via `QObject` signals connected with `Qt.ConnectionType.QueuedConnection`.
- Background workers must use safe lifecycle hooks: `stop()`, `wait(1000)`, `terminate()`.

## 3. High-DPI Scaling & Classroom Legibility
- Set high-contrast academic themes suitable for projection (contrast ratio >= 4.5:1).
- Use vector assets (`QSvgRenderer`, SVG icons via `QIcon`) rather than raster bitmaps for toolbar buttons.
- Handle `resizeEvent` by calling `fitInView(sceneRect, Qt.AspectRatioMode.KeepAspectRatio)`.
