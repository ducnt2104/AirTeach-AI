# AirTeach AI — Windows Packaging & Distribution Architecture

**Build Tool:** PyInstaller 6.22.3  
**Target Environment:** Windows 10/11 x64  
**Packaging Spec:** `airteach.spec`

---

## 1. Distribution Strategy

AirTeach AI is packaged as a standalone Windows folder bundle (`dist/AirTeachAI/`) containing all necessary runtime binaries, Qt6 C++ plugins, OpenCV DLLs, ONNX models, and SVG asset bundles.

### Key Included Bundles
1. **Pre-Trained Local Models:**
   - `models/face_detection_yunet_2023mar.onnx` (232 KB)
   - `models/face_recognition_sface_2021dec.onnx` (38.7 MB)
   - `models/hand_landmarker.task` (7.8 MB)
2. **Vector Graphics & Icons:**
   - `assets/icons/*.svg` (15 custom teaching icons)
   - `assets/gestures/*.svg` (6 gesture visual guides)
3. **Qt6 Dynamic Plugins:**
   - `PySide6.QtSvg` & `PySide6.QtSvgWidgets` (crisp vector UI rendering)
   - `PySide6.QtPrintSupport` (PDF exporting)
4. **Computational Libraries:**
   - NumPy, SciPy (ConvexHull, interpolation)
   - Matplotlib (MathText rendering engine)
   - pypdfium2 (PDFium native rendering engine)
   - Cryptography (Fernet symmetric encryption)

---

## 2. Build Verification Command

To compile the standalone Windows distribution:

```powershell
pyinstaller --noconfirm airteach.spec
```

The resulting executable is generated at:
```text
dist/AirTeachAI/AirTeachAI.exe
```

### Installation & Launch Instructions
1. Unzip `AirTeachAI.zip` to any folder (e.g. `C:\Program Files\AirTeach AI` or USB drive).
2. Double-click `AirTeachAI.exe`.
3. On first launch, the application initializes `data/profiles.db` and loads the teacher login portal.
