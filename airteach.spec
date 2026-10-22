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
    (str(project_root / ".env"), "."),
]

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
]

# Collect any additional dynamic submodules
hiddenimports += collect_submodules("scipy.spatial")
hiddenimports += collect_submodules("matplotlib.mathtext")

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
