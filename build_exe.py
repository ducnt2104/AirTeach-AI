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
    required = [
        root / "assets" / "logo.ico",
        root / "models" / "hand_landmarker.task",
        root / "config" / "vision_config.json",
    ]
    missing = [str(p.relative_to(root)) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required bundle resources: {missing}")

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
    out_dir = root / "dist" / "AirTeachAI"
    out_exe = out_dir / "AirTeachAI.exe"
    if not out_exe.exists():
        raise RuntimeError(f"Build finished without creating {out_exe}")
    print(f"[OK] Built release binary: {out_exe} ({out_exe.stat().st_size // 1024} KB)")
    internal_dir = out_dir / "_internal"
    for rel in ("assets/logo.ico", "models/hand_landmarker.task", "config/vision_config.json"):
        p = internal_dir / rel
        if not p.exists():
            p = out_dir / rel
        print(f"  - Bundled {rel}: {'OK' if p.exists() else 'MISSING'}")


if __name__ == "__main__":
    build()
