"""
PyInstaller Packaging Script for AirTeach AI (Python / PySide6 Edition)
Usage:
    python build_exe.py
"""

import subprocess
import sys
from pathlib import Path


def build() -> None:
    root = Path(__file__).resolve().parent
    main_script = root / "main.py"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name",
        "AirTeach_AI_PySide6",
        str(main_script),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(root))


if __name__ == "__main__":
    build()
