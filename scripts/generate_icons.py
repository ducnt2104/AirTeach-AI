from __future__ import annotations

import os
from pathlib import Path
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
ICONS_DIR = ROOT / "assets" / "icons"
LOGO_SVG = ICONS_DIR / "logo.svg"


def render_svg_to_image(svg_path: Path, size: int) -> Image.Image:
    """Render an SVG file to a PIL Image at the specified square size with transparency."""
    renderer = QSvgRenderer(str(svg_path))
    qimg = QImage(size, size, QImage.Format_ARGB32)
    qimg.fill(QColor(0, 0, 0, 0))

    painter = QPainter(qimg)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setRenderHint(QPainter.SmoothPixmapTransform)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()

    # Convert QImage buffer to PIL Image
    ptr = qimg.bits()
    pil_img = Image.frombytes("RGBA", (size, size), ptr.tobytes(), "raw", "BGRA")
    return pil_img


def generate_all_icons():
    print(f"Generating cross-platform icons from {LOGO_SVG}...")

    # 1. High-resolution 512x512 PNG for Linux desktop entries
    img_512 = render_svg_to_image(LOGO_SVG, 512)
    png_path = ICONS_DIR / "airteach.png"
    img_512.save(str(png_path), format="PNG")
    print(f"  ✓ Created Linux/Cross-platform PNG: {png_path} (512x512)")

    # 2. Windows ICO multi-size
    ico_path = ICONS_DIR / "airteach.ico"
    img_512.save(
        str(ico_path),
        format="ICO",
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
    )
    print(f"  ✓ Created Windows ICO: {ico_path} (256, 128, 64, 48, 32, 16)")

    # 3. macOS ICNS multi-size
    icns_path = ICONS_DIR / "airteach.icns"
    img_512.save(
        str(icns_path),
        format="ICNS",
        sizes=[(512, 512), (256, 256), (128, 128), (64, 64), (32, 32), (16, 16)],
    )
    print(f"  ✓ Created macOS ICNS: {icns_path}")


if __name__ == "__main__":
    generate_all_icons()
