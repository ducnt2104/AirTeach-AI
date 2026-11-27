"""
PyInstaller Packaging Script for AirTeach AI (Python / PySide6 Edition)
Usage:
    python build_exe.py
"""

import shutil
import subprocess
import sys
from pathlib import Path


def build() -> None:
    root = Path(__file__).resolve().parent
    fetch_script = root / "scripts" / "fetch_assets.py"
    if fetch_script.exists():
        print("Verifying/downloading required AI assets, Vietnamese fonts & icons...")
        subprocess.check_call([sys.executable, str(fetch_script)], cwd=str(root))

    spec_file = root / "AirTeachAI.spec"
    required = [
        root / "assets" / "icons" / "airteach.ico",
        root / "assets" / "models" / "hand_landmarker.task",
        root / "assets" / "fonts" / "NotoSans-Regular.ttf",
        root / "assets" / "fonts" / "NotoSans-Bold.ttf",
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

    autofix_bat = root / "Chay_AirTeachAI_AutoFix.bat"
    if autofix_bat.exists():
        shutil.copy2(autofix_bat, out_dir / "Chay_AirTeachAI_AutoFix.bat")

    print(f"[OK] Built release binary: {out_exe} ({out_exe.stat().st_size // 1024} KB)")
    internal_dir = out_dir / "_internal"
    for rel in (
        "assets/icons/airteach.ico",
        "assets/models/hand_landmarker.task",
        "assets/fonts/NotoSans-Regular.ttf",
        "assets/fonts/NotoSans-Bold.ttf",
        "models/hand_landmarker.task",
        "config/vision_config.json",
    ):
        p = internal_dir / rel
        if not p.exists():
            p = out_dir / rel
        print(f"  - Bundled {rel}: {'OK' if p.exists() else 'MISSING'}")
    print(f"  - Companion launcher: {'OK' if (out_dir / 'Chay_AirTeachAI_AutoFix.bat').exists() else 'MISSING'}")


if __name__ == "__main__":
    build()
