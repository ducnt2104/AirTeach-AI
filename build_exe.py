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
    spec_file = root / "AirTeachAI.spec"
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        str(spec_file),
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(root))
    out_exe = root / "dist" / "AirTeachAI" / "AirTeachAI.exe"
    if out_exe.exists():
        print(f"[OK] Built release binary: {out_exe} ({out_exe.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    build()
