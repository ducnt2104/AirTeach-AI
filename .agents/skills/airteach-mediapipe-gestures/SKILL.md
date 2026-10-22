---
name: airteach-mediapipe-gestures
description: >-
  Specialist skill for MediaPipe 1.x Tasks API (HandLandmarker) and legacy solutions,
  21 3D landmarks extraction, hand-scale normalization, 6-gesture FSM, and debounce algorithms.
---

# AirTeach AI — Hand Tracking & Gesture FSM Skill

## 1. Dual-Version MediaPipe Support (Tasks API + Legacy Solutions)
- MediaPipe 1.x deprecated `mp.solutions.hands`. The modern API is `mp.tasks.vision.HandLandmarker` initialized with `models/hand_landmarker.task`.
- Both APIs output 21 landmarks indexed identically (0=Wrist, 4=Thumb Tip, 8=Index Tip, 9=Middle MCP, 12=Middle Tip, etc.).
- Always wrap model loading to support both versions seamlessly:
  ```python
  if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
      # Use HandLandmarker with model_asset_path
  elif hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
      # Use legacy mp.solutions.hands.Hands
  ```

## 2. Scale-Invariant Distance Normalization
Never compare raw pixel distances. Hands move closer or further from the webcam.
Always normalize geometric distances by hand length $L_{\text{hand}} = \|P_0 - P_9\|_2$:
$$\text{dist}_{\text{norm}} = \frac{\|P_a - P_b\|_2}{L_{\text{hand}}}$$

## 3. The Six Core Gestures
1. **POINT**: Index finger extended ($P_8.y < P_6.y$), thumb separated ($\text{dist}(P_4, P_8) / L_{\text{hand}} > 0.42$). Laser cursor active, zero ink.
2. **DRAW**: Index finger extended, pinch condition satisfied ($\text{dist}(P_4, P_8) / L_{\text{hand}} < 0.34$). Inks active.
3. **PAUSE / PEN LIFT**: Closed fist (all fingertips curled towards wrist: $\|P_i - P_0\| < \|P_{i-2} - P_0\|$). Finishes stroke.
4. **SELECT**: Index and middle fingers extended in V/peace shape, ring and pinky curled.
5. **SWIPE (LEFT / RIGHT)**: Open palm moving with normalized horizontal velocity $v_x > 0.35$ within $250\text{ ms}$, with an $800\text{ ms}$ cooldown.
6. **CONFIRM**: Thumbs-up pose held stably for $\ge 300\text{ ms}$.
