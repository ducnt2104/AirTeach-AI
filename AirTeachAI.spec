# -*- mode: python ; coding: utf-8 -*-
"""
Production PyInstaller Spec for AirTeach AI (`AirTeachAI.spec`)
Builds `--onedir` + `--windowed` release bundle into `dist/AirTeachAI/`:
- Collects all MediaPipe graphs/binaries/assets, PySide6 plugins, PyMuPDF (`fitz`),
  OpenCV (`cv2`), python-docx, python-pptx, onnxruntime, and optional `airteach_core_rs`.
- Bundles `assets/models`, `assets/fonts`, `assets/icons`, `assets/branding`, `models`,
  `config`, `.env.example`, and local VC++ runtime DLLs when present.
- Copies `Chay_AirTeachAI_AutoFix.bat` right next to `dist/AirTeachAI/AirTeachAI.exe`.
"""

import importlib.util
import os
from pathlib import Path
import shutil
import sys

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_PYSIDE6_SKIP_TOKENS = (
    "sqldrivers",
    "qtwebengine",
    "qt6webengine",
    "qtquick",
    "qt6quick",
    "qtqml",
    "qt6qml",
    "qt3d",
    "qt63d",
    "qtdesigner",
    "qt6designer",
    "qtpdf",
    "qt6pdf",
    "qtmultimedia",
    "qt6multimedia",
    "qtcharts",
    "qt6charts",
    "qtgraphs",
    "qt6graphs",
    "qtdatavisualization",
    "qt6datavisualization",
    "qtspatialaudio",
    "qt6spatialaudio",
    "qtsensors",
    "qt6sensors",
    "qtbluetooth",
    "qt6bluetooth",
    "qtlocation",
    "qt6location",
    "qtpositioning",
    "qt6positioning",
    "qtremoteobjects",
    "qt6remoteobjects",
    "qtscxml",
    "qt6scxml",
    "qtstatemachine",
    "qt6statemachine",
    "qtwebchannel",
    "qt6webchannel",
    "qtwebsockets",
    "qt6websockets",
    "qtserial",
    "qt6serial",
    "qtnfc",
    "qt6nfc",
    "qttest",
    "qt6test",
    "qtsql",
    "qt6sql",
    "qthelp",
    "qt6help",
    "qthttpserver",
    "qt6httpserver",
    "qttexttospeech",
    "qt6texttospeech",
)


def _keep_pyside_item(text: str) -> bool:
    low = text.lower()
    return not any(tok in low for tok in _PYSIDE6_SKIP_TOKENS)


datas = []
binaries = []
hiddenimports = []


def _add_collect_all(pkg_name: str, filter_pyside: bool = False) -> None:
    d, b, h = collect_all(pkg_name)
    if filter_pyside:
        d = [item for item in d if _keep_pyside_item(f"{item[0]} {item[1]}")]
        b = [item for item in b if _keep_pyside_item(f"{item[0]} {item[1]}")]
        h = [mod for mod in h if _keep_pyside_item(mod)]
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)


for pkg in ("mediapipe", "fitz", "pymupdf", "cv2", "docx", "pptx", "onnxruntime"):
    if importlib.util.find_spec(pkg) is not None:
        _add_collect_all(pkg)

_add_collect_all("PySide6", filter_pyside=True)
_add_collect_all("shiboken6", filter_pyside=False)

if importlib.util.find_spec("airteach_core_rs") is not None:
    _add_collect_all("airteach_core_rs")

datas += collect_data_files("matplotlib")
if importlib.util.find_spec("_sounddevice_data") is not None:
    datas += collect_data_files("_sounddevice_data")

# Explicit application assets, AI models, Vietnamese NotoSans fonts, icons & config templates
for src_rel, dst_rel in (
    ("assets/models", "assets/models"),
    ("assets/fonts", "assets/fonts"),
    ("assets/icons", "assets/icons"),
    ("assets/branding", "assets/branding"),
    ("models", "models"),
    ("config", "config"),
):
    src_path = ROOT / src_rel
    if src_path.exists():
        datas.append((str(src_path), dst_rel))

for file_rel, dst_rel in (
    ("assets/logo.ico", "assets"),
    (".env.example", "."),
):
    file_path = ROOT / file_rel
    if file_path.exists():
        datas.append((str(file_path), dst_rel))

# Bundle MSVC 2015-2022 Runtime DLLs locally inside _internal for unpatched offline PCs
if sys.platform == "win32":
    sys32 = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    for dll_name in (
        "vcruntime140.dll",
        "vcruntime140_1.dll",
        "msvcp140.dll",
        "msvcp140_1.dll",
        "msvcp140_2.dll",
        "concrt140.dll",
        "vcomp140.dll",
    ):
        dll_path = sys32 / dll_name
        if dll_path.exists():
            binaries.append((str(dll_path), "."))

hiddenimports += collect_submodules("cv_engine")
hiddenimports += [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtSvg",
    "PySide6.QtSvgWidgets",
    "pymupdf",
    "fitz",
    "docx",
    "pptx",
    "google.genai",
    "google.genai.types",
    "win32com",
    "win32com.client",
    "pythoncom",
    "pywintypes",
    "win32api",
    "requests",
    "dotenv",
    "matplotlib",
    "matplotlib.figure",
    "matplotlib.backends.backend_agg",
    "PIL",
    "PIL._tkinter_finder",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "sounddevice",
    "speech_recognition",
    "cv2",
    "numpy",
    "scipy",
    "onnxruntime",
    "concurrent.futures.thread",
    "concurrent.futures.process",
]

if importlib.util.find_spec("airteach_core_rs") is not None:
    hiddenimports.append("airteach_core_rs")

# Deduplicate hiddenimports while preserving order
hiddenimports = list(dict.fromkeys(hiddenimports))

icon_primary = ROOT / "assets" / "icons" / "airteach.ico"
icon_fallback = ROOT / "assets" / "logo.ico"
app_icon = str(icon_primary if icon_primary.exists() else icon_fallback)

a = Analysis(
    [str(ROOT / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "scripts" / "pyi_rth_airteach_crash.py")],
    excludes=[
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtWebEngineQuick",
        "PySide6.QtQuick",
        "PySide6.QtQuickWidgets",
        "PySide6.QtQuick3D",
        "PySide6.QtQml",
        "PySide6.QtDesigner",
        "PySide6.QtPdf",
        "PySide6.QtPdfWidgets",
        "PySide6.QtMultimedia",
        "PySide6.QtMultimediaWidgets",
        "pytest",
    ],
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
    disable_windowed_traceback=True,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=app_icon,
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

# Copy 1-click competition AutoFix launcher alongside AirTeachAI.exe
autofix_src = ROOT / "Chay_AirTeachAI_AutoFix.bat"
dist_bundle_dir = Path(DISTPATH) / "AirTeachAI"
if autofix_src.exists() and dist_bundle_dir.exists():
    shutil.copy2(autofix_src, dist_bundle_dir / "Chay_AirTeachAI_AutoFix.bat")
