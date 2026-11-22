# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Spec for AirTeach AI (`AirTeachAI.spec`)
Builds `--onedir` + `--windowed` release bundle for fast startup on Intel Core i3 (4th/5th gen)
with all offline AI models, config, and multi-resolution `assets/logo.ico`.
"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = Path(SPECPATH)

datas = [
    (str(ROOT / "models"), "models"),
    (str(ROOT / "config"), "config"),
    (str(ROOT / "assets"), "assets"),
]
datas += collect_data_files("mediapipe")

hiddenimports = [
    "cv_engine",
    "cv_engine.app_paths",
    "cv_engine.drawing_fsm",
    "cv_engine.face_id",
    "cv_engine.gemini_client",
    "cv_engine.hand_gesture",
    "cv_engine.lasso_ai_service",
    "cv_engine.local_ai_manager",
    "cv_engine.output_sanitizer",
    "cv_engine.performance_pipeline",
    "cv_engine.preprocessing",
    "cv_engine.shape_recognizer",
    "cv_engine.sidecar_worker",
    "cv_engine.subject_tracker",
    "cv_engine.vision_config",
    "cv_engine.whiteboard_agent",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "google.genai",
    "onnxruntime",
    "sounddevice",
    "cv2",
    "numpy",
]
hiddenimports += collect_submodules("mediapipe")

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AirTeachAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ROOT / "assets" / "logo.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="AirTeachAI",
)
