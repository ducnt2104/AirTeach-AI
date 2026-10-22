---
name: airteach-shape-snapping
description: >-
  Specialist skill for geometric shape recognition (Line, Arrow, Circle, Ellipse, Rectangle, Triangle),
  Convex Hull, circularity metric, and interactive preview/confirmation lifecycles.
---

# AirTeach AI — Geometric Shape Snapping Skill

## 1. Geometric Descriptors & Metrics
- **Circularity Index**:
  $$C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$$
  A perfect circle has $C \approx 1.0$. Rough hand-drawn circles achieve $C \ge 0.75$.
- **Endpoint Distance Ratio**:
  $$R_{\text{closure}} = \frac{\|P_{\text{start}} - P_{\text{end}}\|}{\text{PathLength}}$$
  If $R_{\text{closure}} < 0.18$, the curve is closed (Circle, Ellipse, Rectangle, Triangle).
  If $R_{\text{closure}} \ge 0.82$, the curve is open (Line, Arrow).
- **Aspect Ratio & Corner Angles**:
  Compute minimum bounding rectangle (`cv2.minAreaRect`) and polygon approximation (`cv2.approxPolyDP`) to classify 3-vertex polygons (Triangle) and 4-vertex orthogonal polygons (Rectangle).

## 2. Interactive Preview & Rejection
- When a shape is recognized, render a dashed/semi-transparent preview guide.
- User can confirm via CONFIRM gesture (👍) or clicking Accept, or reject by continuing to draw or pressing Undo.
- Original raw strokes are always preserved in history until explicitly replaced.
