with open("src/ui/main_window.py", "r", encoding="utf-8") as f:
    text = f.read()

target = 'self._on_toggle_organic_chemistry()\n            elif val == "timer":'
repl = 'self._on_toggle_organic_chemistry()\n            elif val in ("fluid", "fluid_venturi", "venturi", "bernoulli", "chat_luu", "ap_suat_tinh", "manometer"):\n                self._on_toggle_fluid_venturi()\n            elif val == "timer":'

assert target in text, "Target not found"
text = text.replace(target, repl, 1)

with open("src/ui/main_window.py", "w", encoding="utf-8") as f:
    f.write(text)

import py_compile
py_compile.compile("src/ui/main_window.py", doraise=True)
print("Successfully added fluid in-air dispatch!")
