---
name: airteach-math-formula-recognition
description: >-
  Specialist skill for offline mathematical formula recognition (LaTeX-OCR, GeometricMathParser fallback),
  LaTeX syntax validation, and crisp vector rendering via matplotlib.mathtext.
---

# AirTeach AI — Offline Math Formula Recognition Skill

## 1. Dual-Engine Architecture
- **Primary Engine**: `rapid-latex-ocr` or `pix2tex` converting handwritten math crops to standard LaTeX strings.
- **Deterministic Geometric Fallback**: `GeometricMathParser` classifying spatial arrangements of strokes:
  - Stacked horizontal lines $\to =$
  - Single horizontal line $\to -$
  - Intersecting lines $\to +$
  - Superscript/Subscript detection based on relative bounding box vertical offset $\Delta y$ and height ratio $< 0.75$.
  - Horizontal fraction bar separating numerator and denominator groups.

## 2. LaTeX Validation & MathText Rendering
- Do NOT invoke external `pdflatex` or `xelatex` subprocesses, which introduce huge install footprints and security risks.
- Use `matplotlib.mathtext.math_to_image` or `FigureCanvasAgg` to render mathematical expressions directly to 300 DPI RGBA bitmaps.
- Always sanitize formula strings: reject arbitrary macro expansion, script injection, or shell commands.
