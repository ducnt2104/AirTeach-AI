# AirTeach AI — 3-Stage Vector Ink Beautifier & Geometric Shape Snapper

**Component:** `src/core/stroke_beautifier.py`  
**Algorithms:** 1€ Filter (Casiez 2012) $\rightarrow$ Ramer–Douglas–Peucker $\rightarrow$ Cubic Bézier Spline $\rightarrow$ Shape Snapping

---

## 1. Mathematical Foundation

Writing on air lacks a physical surface, which naturally produces high-frequency involuntary tremor. Standard fixed moving average filters ($\hat{x}_t = \alpha x_t + (1-\alpha)\hat{x}_{t-1}$) either induce unacceptable stroke lag during quick gestures or fail to remove jitters during slow, deliberate writing.

AirTeach AI resolves this through a three-stage mathematical pipeline:

```
[Raw Finger Coordinates (x, y, t_ms)]
                |
    STAGE 1: 1€ Filter (Real-Time Speed-Adaptive Low-Pass)
                |
    STAGE 2: Post-Stroke Cleanup
             ├─ Ramer-Douglas-Peucker (Point Reduction)
             └─ Cubic Bézier Spline (C1 Continuity & Velocity Width)
                |
    STAGE 3: Geometric Shape Snapper (ConvexHull, Circularity, Residuals)
                |
[Final Crisp Vector Ink on QGraphicsScene]
```

---

## 2. Stage 1: 1€ Filter (Casiez et al., 2012)

The filter dynamically modulates its cutoff frequency $f_c$ based on the instantaneous rate of change of the input signal:

$$f_c = f_{c,\min} + \beta \|\dot{x}_t\|$$

Where:
- $f_{c,\min} = 1.0\,\text{Hz}$: Minimum cutoff frequency at near-zero velocity (removes fine handwriting tremors).
- $\beta = 0.007$: Speed coefficient. When the teacher swipes or draws broad strokes, $f_c$ increases, effectively lowering filtering lag to near zero.
- The smoothing factor is computed as:
  $$\alpha = \frac{1}{1 + \frac{1}{2\pi f_c \Delta t}}$$

---

## 3. Stage 2: Ramer–Douglas–Peucker & Cubic Bézier Spline

### RDP Point Simplification
When the teacher closes their fist (`PAUSE`), the raw coordinate buffer is pruned using the Ramer–Douglas–Peucker algorithm with $\epsilon = 2.5\,\text{px}$.
- **Performance:** Prunes $90.0\%$ of redundant intermediate points while preserving sharp corners and trajectory extremes.

### $C^1$ Continuous Cubic Bézier Splines
The reduced points are converted into connected Bézier curves. For each segment $i$ between points $P_i$ and $P_{i+1}$, control points $C_{1,i}$ and $C_{2,i}$ are fitted such that tangents remain continuous across joints ($C^1$ smoothness).

### Dynamic Stroke Width Modulation
Stroke width varies inversely with writing velocity to mimic real calligraphic ink pens:

$$w(v) = \text{clamp}\left(w_{\max} - \gamma \cdot v, \ w_{\min}, \ w_{\max}\right)$$

Where $w_{\min} = 1.5\,\text{px}$, $w_{\max} = 6.0\,\text{px}$ (configurable in settings).

---

## 4. Stage 3: Geometric Shape Snapper

The snapper evaluates geometric invariants to distinguish intentional geometric shapes from freehand sketches:

1. **Straight Line:**
   - First Principal Component (PCA) explains $\ge 96\%$ of coordinate variance.
   - Snaps to orthogonal or 45-degree angles (0°, 45°, 90°, 135°, 180°).
2. **Arrow:**
   - Open stroke with a sharp angular fork ($< 60^\circ$) at the terminal end.
3. **Circle / Ellipse:**
   - Ratio of start-to-end distance to total perimeter $< 0.15$ (closed loop).
   - Circularity metric:
     $$C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2} \ge 0.82$$
4. **Rectangle:**
   - Closed loop with Convex Hull having 4 dominant vertices and interior angles $\approx 90^\circ \pm 15^\circ$.
5. **Triangle:**
   - Closed loop with Convex Hull having 3 dominant vertices.

All snapped shapes can be undone with a single `Ctrl+Z` (Undo), reverting immediately to the un-snapped Bézier ink stroke.
