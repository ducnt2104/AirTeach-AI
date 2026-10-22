---
name: airteach-visual-qa
description: >-
  Specialist skill for visual verification, native Windows font rendering inspection,
  high-DPI capture, and screenshot regression testing for AirTeach AI.
---

# AirTeach AI — Visual QA & Inspection Skill

## 1. Native Windows Font Engine vs Offscreen Platform
- **CRITICAL FINDING**: In Qt 6 on Windows, setting `QT_QPA_PLATFORM='offscreen'` initializes a minimal headless font engine with **0 system font families**, causing all text and Vietnamese diacritics to render as empty tofu boxes (`□`).
- **SOLUTION**: Use default Windows DirectWrite platform when grabbing screenshots:
  ```python
  widget.show()
  app.processEvents()
  pix = widget.grab()
  pix.save("screenshot.png")
  ```
  This ensures 100% genuine font rendering identical to what teachers and students see.

## 2. Visual Inspection Checklist
- **No Emoji as Toolbar Icons**: Verify that primary toolbars use sharp SVG vector icons (`QIcon("assets/icons/...")`).
- **Vietnamese Diacritics Integrity**: Check that characters such as `ă, â, đ, ê, ô, ơ, ư` and tone marks (`á, à, ả, ã, ạ`) render with correct vertical metrics without clipping.
- **Contrast & Projection Ready**: Verify dark/light contrast ratios, active button states, and distinct laser pointer visibility against white and dark backgrounds.
