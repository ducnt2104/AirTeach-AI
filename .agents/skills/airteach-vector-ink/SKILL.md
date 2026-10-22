---
name: airteach-vector-ink
description: >-
  Specialist skill for precision vector ink pipelines, 1€ Filter adaptive filtering,
  Ramer-Douglas-Peucker simplification, Catmull-Rom cubic Bézier splines, and velocity-based stroke widths.
---

# AirTeach AI — Precision Vector Ink Pipeline Skill

## 1. Stage 1: 1€ Filter (Real-Time Low Latency)
- Mitigates hand jitter at low velocities while eliminating lag during fast writing.
- Speed-adaptive cutoff frequency:
  $$f_c = f_{c,\min} + \beta \cdot |\dot{x}|$$
- Typical classroom calibrated parameters: $f_{c,\min} = 1.0\text{ Hz}$, $\beta = 0.007$, derivative cutoff $f_{c,d} = 1.0\text{ Hz}$.
- State must be reset on stroke start and when tracking is lost.

## 2. Stage 2: Post-Stroke Smoothing (RDP + Cubic Bézier)
- **Ramer-Douglas-Peucker (RDP)**: Simplifies raw polylines using perpendicular distance threshold $\epsilon \in [1.5, 3.0]\text{ px}$. Preserves intentional corners while discarding noisy intermediate points.
- **Catmull-Rom to Cubic Bézier Spline**: Converts $N$ points into $N-1$ Bézier segments with continuous $C^1$ tangents.
- **Velocity-Modulated Stroke Width**: Modulates stroke thickness inversely with speed:
  $$w(v) = w_{\max} - (w_{\max} - w_{\min}) \cdot \min\left(1.0, \frac{v}{v_{\text{ref}}}\right)$$
  Fast strokes taper elegantly; slow strokes are deliberate and bold.
