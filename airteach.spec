# -*- mode: python ; coding: utf-8 -*-
"""
AirTeach AI - PyInstaller Specification File
============================================
Packages the full desktop application into a standalone Windows executable.
Includes:
  - Qt6 plugins (SVG, styles, imageformats)
  - Pre-trained ONNX models (YuNet, SFace) & MediaPipe task asset
  - Complete SVG vector iconography and gesture cards
  - Matplotlib MathText rendering engine
"""

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Project directories
project_root = Path.cwd()

datas = [
    (str(project_root / "models"), "models"),
    (str(project_root / "assets"), "assets"),
]
env_file = project_root / ".env"
if not env_file.exists():
    env_file = project_root / ".env.example"
if env_file.exists():
    datas.append((str(env_file), "."))

hiddenimports = [
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "PySide6.QtPrintSupport",
    "cv2",
    "mediapipe",
    "numpy",
    "scipy",
    "scipy.spatial",
    "scipy.interpolate",
    "scipy.special",
    "matplotlib",
    "matplotlib.mathtext",
    "matplotlib.font_manager",
    "matplotlib.backends.backend_agg",
    "cryptography",
    "cryptography.fernet",
    "pypdf",
    "pypdfium2",
    "dotenv",
    "sqlite3",
    "google.genai",
    "keyring",
    "keyring.backends",
    "keyring.backends.Windows",
    "src.core.adaptive_vision_pipeline",
    "src.core.multi_hand_coordinator",
    "src.services.credential_service",
    "src.ui.floating_action_menu",
    "src.ui.camera_diagnostics_dialog",
    "src.ui.ai_workbench",
    "src.ui.measurement_tools",
    "src.ui.spotlight_overlay",
    "src.ui.virtual_loupe",
    "src.ui.presentation_timer",
    "src.ui.desktop_overlay_bar",
    "src.ui.gesture_guide_overlay",
    "src.utils.audio_feedback",
]

# Collect any additional dynamic submodules
hiddenimports += collect_submodules("scipy.spatial")
hiddenimports += collect_submodules("matplotlib.mathtext")
hiddenimports += collect_submodules("src")

a = Analysis(
    ["main.py"],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "unittest",
        "IPython",
        "jupyter",
        "notebook",
        "sphinx",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AirTeachAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "assets" / "icons" / "airteach.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AirTeachAI",
)
