import os

with open("src/core/hand_gesture_engine.py", "rb") as f:
    text = f.read().decode("utf-8")

old_init = "        swipe_velocity_thr: float = 0.35,"
new_init = "        swipe_velocity_thr: float = 0.8,"

assert old_init in text, "old_init not found!"
text = text.replace(old_init, new_init, 1)

old_detect = """        dx = x_end - x_start
        velocity = dx / (dt_ms / 1000.0)  # normalized units per second

        # Convert to per-frame equivalent for threshold comparison
        per_frame_velocity = abs(velocity) / 30.0

        if per_frame_velocity > self.swipe_velocity_thr:"""

new_detect = """        dx = x_end - x_start
        velocity = dx / (dt_ms / 1000.0)  # normalized units per second

        # Require significant horizontal displacement (>= 12% screen width) and velocity
        if abs(dx) >= 0.12 and abs(velocity) >= self.swipe_velocity_thr:"""

if old_detect not in text:
    old_detect = old_detect.replace("\n", "\r\n")

assert old_detect in text, "old_detect not found!"
text = text.replace(old_detect, new_detect.replace("\n", "\r\n"), 1)

with open("src/core/hand_gesture_engine.py", "wb") as f:
    f.write(text.encode("utf-8"))

print("Updated swipe velocity detection in hand_gesture_engine.py successfully!")
