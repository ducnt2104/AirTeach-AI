"""
AirTeach AI - Windows Build & Packaging Script
==============================================
Validates pre-requisites and invokes PyInstaller to produce a verified standalone distribution.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check_prerequisites() -> bool:
    print("[1/4] Checking prerequisites...")
    models_dir = ROOT / "models"
    required_models = [
        "face_detection_yunet_2023mar.onnx",
        "face_recognition_sface_2021dec.onnx",
        "hand_landmarker.task",
    ]
    for m in required_models:
        p = models_dir / m
        if not p.exists() or p.stat().st_size == 0:
            print(f"ERROR: Missing model asset: {p}")
            return False
        print(f"  Found model: {m} ({p.stat().st_size / (1024*1024):.2f} MB)")

    assets_dir = ROOT / "assets"
    svg_count = len(list(assets_dir.glob("**/*.svg")))
    if svg_count < 33:
        print(f"ERROR: Expected >= 33 SVG assets, found {svg_count}")
        return False
    print(f"  Found {svg_count} SVG assets in {assets_dir}")

    spec_file = ROOT / "airteach.spec"
    if not spec_file.exists():
        print(f"ERROR: Missing spec file: {spec_file}")
        return False
    print(f"  Found spec file: {spec_file}")
    return True


def run_pyinstaller() -> bool:
    print("\n[2/4] Running PyInstaller build...")
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "airteach.spec"]
    res = subprocess.run(cmd, cwd=str(ROOT))
    return res.returncode == 0


def verify_build() -> bool:
    print("\n[3/4] Verifying build output...")
    dist_dir = ROOT / "dist" / "AirTeachAI"
    exe_path = dist_dir / "AirTeachAI.exe"

    if not exe_path.exists():
        print(f"ERROR: Executable not found at {exe_path}")
        return False

    exe_size_mb = exe_path.stat().st_size / (1024 * 1024)
    print(f"  Executable generated: {exe_path.name} ({exe_size_mb:.2f} MB)")

    # Verify models bundled
    bundled_models = dist_dir / "_internal" / "models"
    if not bundled_models.exists():
        bundled_models = dist_dir / "models"

    print(f"  Checking bundled models at: {bundled_models}")
    for m in ["face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx", "hand_landmarker.task"]:
        mp = bundled_models / m
        if not mp.exists():
            print(f"ERROR: Bundled model missing: {mp}")
            return False
        print(f"    ✓ Bundled model: {m}")

    # Verify assets bundled
    bundled_assets = dist_dir / "_internal" / "assets"
    if not bundled_assets.exists():
        bundled_assets = dist_dir / "assets"

    svg_count = len(list(bundled_assets.glob("**/*.svg")))
    print(f"  Bundled SVG assets count: {svg_count}")
    if svg_count < 33:
        print(f"ERROR: Missing bundled SVG assets (found {svg_count})")
        return False
    print("    ✓ All vector assets bundled cleanly.")

    print("\n[4/4] Distribution bundle verified 100% complete!")
    return True


def main():
    if not check_prerequisites():
        sys.exit(1)
    if not run_pyinstaller():
        print("ERROR: PyInstaller execution failed.")
        sys.exit(1)
    if not verify_build():
        sys.exit(1)
    print("\nSUCCESS: AirTeach AI Windows build is ready for deployment.")


if __name__ == "__main__":
    main()
