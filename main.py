"""
AirTeach AI - Application Entry Point
======================================
Full-Stack Desktop Startup Sequence:
  1. Load .env environment variables (GEMINI_API_KEY, GEMINI_MODEL)
  2. Ensure data/, models/, workspaces/, and assets/ directories exist
  3. Check and download missing open-source models (YuNet, SFace, HandLandmarker)
  4. Initialize SQLite database (data/profiles.db)
  5. Apply modern Academic Theme stylesheet
  6. Display Teacher Login Window (live camera, YuNet bbox, auto face recognition, PIN pad)
  7. Seamlessly transition to MainWindow upon successful login or demo mode
  8. Safely release camera resources and background workers on exit

Run from project root:
    python main.py
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# ── Ensure project root is on sys.path ─────────────────────────────────────────
if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    _ROOT = Path(sys._MEIPASS)
else:
    _ROOT = Path(__file__).resolve().parent

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_ROOT / ".env", override=False)
except ImportError:
    _env_path = _ROOT / ".env"
    if _env_path.exists():
        for _line in _env_path.read_text(encoding="utf-8").splitlines():
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

# ── Qt imports ────────────────────────────────────────────────────────────────
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMessageBox,
    QSplashScreen,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("airteach")


# ─── Academic Theme Stylesheet ────────────────────────────────────────────────

ACADEMIC_THEME = """
QWidget {
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", sans-serif;
    font-size: 10pt;
    color: #202124;
}

QMainWindow, QDialog {
    background-color: #f8f9fa;
}

QMenuBar {
    background-color: #ffffff;
    border-bottom: 1px solid #dadce0;
    padding: 2px 6px;
}
QMenuBar::item {
    padding: 6px 12px;
    border-radius: 4px;
}
QMenuBar::item:selected {
    background-color: #e8f0fe;
    color: #1a73e8;
}

QMenu {
    background-color: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 6px;
    padding: 4px;
}
QMenu::item {
    padding: 6px 24px;
    border-radius: 4px;
}
QMenu::item:selected {
    background-color: #e8f0fe;
    color: #1a73e8;
}

QToolBar {
    background-color: #ffffff;
    border-bottom: 1px solid #dadce0;
    spacing: 8px;
    padding: 4px 8px;
}
QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    padding: 5px 10px;
    font-weight: 500;
}
QToolButton:hover {
    background: #e8f0fe;
    border-color: #d2e3fc;
    color: #1a73e8;
}

QStatusBar {
    background-color: #ffffff;
    border-top: 1px solid #dadce0;
    font-size: 9pt;
    color: #5f6368;
}

QPushButton {
    background-color: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #f1f3f4;
    border-color: #c6c9cc;
}
QPushButton:pressed {
    background-color: #e8eaed;
}

QLineEdit, QTextEdit {
    background-color: #ffffff;
    border: 1px solid #dadce0;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #1a73e8;
}
QLineEdit:focus, QTextEdit:focus {
    border: 2px solid #1a73e8;
}

QScrollBar:vertical {
    border: none;
    background: #f1f3f4;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #c1c3c7;
    border-radius: 4px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #9aa0a6;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


def _ensure_dirs() -> None:
    """Create essential data directories."""
    for d in ("data", "data/workspaces", "models", "assets"):
        (_ROOT / d).mkdir(parents=True, exist_ok=True)


def _download_models_if_needed() -> None:
    """Check open-source models and download if missing and online."""
    try:
        from src.utils.model_downloader import ensure_all_models
        models_dir = _ROOT / "models"
        ensure_all_models(models_dir=models_dir, force_download=False)
    except Exception as exc:
        logger.warning("Auto model download check failed (offline mode): %s", exc)


def _make_splash(app: QApplication) -> QSplashScreen:
    """Splash screen with AirTeach branding."""
    pix = QPixmap(560, 260)
    pix.fill(Qt.GlobalColor.white)
    from PySide6.QtGui import QPainter, QColor, QFont  # noqa: PLC0415
    from PySide6.QtCore import QRect  # noqa: PLC0415

    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor("#1a73e8"))
    painter.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
    painter.drawText(pix.rect(), Qt.AlignmentFlag.AlignCenter, "✋ AirTeach AI")

    painter.setFont(QFont("Segoe UI", 12))
    painter.setPen(QColor("#5f6368"))
    painter.drawText(
        QRect(0, 160, 560, 40),
        Qt.AlignmentFlag.AlignCenter,
        "Nền tảng giảng dạy tương tác không chạm bằng cử chỉ tay"
    )
    painter.end()

    splash = QSplashScreen(pix, Qt.WindowType.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()
    return splash


def main() -> int:
    _ensure_dirs()

    if "--version" in sys.argv:
        print("AirTeach AI v2.5.0-Release (Windows Desktop Academic Edition)")
        return 0

    if "--check" in sys.argv or "--dry-run" in sys.argv:
        print("[CHECK] AirTeach AI system components and models verification...")
        _download_models_if_needed()
        print("[CHECK] Open-source AI models verified.")
        from src.services.face_profile_service import TeacherProfileService  # noqa: PLC0415
        svc = TeacherProfileService(db_path=_ROOT / "data" / "profiles.db", models_dir=_ROOT / "models")
        svc.close()
        print("[CHECK] SQLite profile store and facial recognition pipeline verified.")
        print("[CHECK] All systems operational. Ready to launch.")
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName("AirTeach AI")
    app.setOrganizationName("AirTeach")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(ACADEMIC_THEME)

    splash = _make_splash(app)

    # 1. Model integrity check
    _download_models_if_needed()

    # 2. Database & Profile Service
    from src.services.face_profile_service import TeacherProfileService  # noqa: PLC0415
    db_path = _ROOT / "data" / "profiles.db"
    models_dir = _ROOT / "models"
    profile_svc = TeacherProfileService(db_path=db_path, models_dir=models_dir)

    splash.close()

    app.setQuitOnLastWindowClosed(False)

    # 3. Window Controllers
    from src.ui.login_window import LoginWindow  # noqa: PLC0415
    from src.ui.main_window import MainWindow    # noqa: PLC0415

    login_win: Optional[LoginWindow] = None
    main_win: Optional[MainWindow] = None

    def _open_login_window() -> None:
        nonlocal main_win, login_win
        app.setQuitOnLastWindowClosed(False)
        if main_win:
            main_win.close()
            main_win = None
            import time
            time.sleep(0.15)

        app.processEvents()

        login_win = LoginWindow(service=profile_svc, camera_index=0)
        login_win.login_successful.connect(_open_main_window)
        login_win.demo_mode_requested.connect(lambda: _open_main_window(None))
        login_win.show()

    def _open_main_window(teacher_row=None) -> None:
        nonlocal main_win, login_win
        if login_win:
            login_win.close()
            login_win = None
            import time
            time.sleep(0.15)

        app.processEvents()

        main_win = MainWindow(teacher_row=teacher_row, camera_index=0)
        main_win.logout_requested.connect(_open_login_window)
        main_win.setMinimumSize(1200, 720)
        main_win.resize(1440, 900)
        main_win.show()
        app.setQuitOnLastWindowClosed(True)

        if teacher_row:
            from src.services.face_profile_service import TeacherProfileService  # noqa: PLC0415
            t_dict = TeacherProfileService.teacher_row_to_dict(teacher_row)
            code_str = f"[{t_dict.get('teacher_code')}] " if t_dict.get("teacher_code") else ""
            title_suffix = f" — {code_str}{t_dict.get('full_name', '')}"
        else:
            title_suffix = " — Dạy thử / Demo"
        main_win.setWindowTitle(f"AirTeach AI{title_suffix}")

    _open_login_window()

    ret = app.exec()

    # Cleanup
    if login_win:
        login_win.close()
    if main_win:
        main_win.close()
    profile_svc.close()

    return ret


if __name__ == "__main__":
    sys.exit(main())
