#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Resource & User Data Path Resolution (`cv_engine/app_paths.py`)
for AirTeach AI (DNT - Victer):
- `resource_path(rel)`: works seamlessly both in Dev mode and in PyInstaller `--onedir`/`--onefile`
  (`sys._MEIPASS` or executable directory).
- `get_user_data_dir()`: stores writable user config, API keys, SQLite/JSON DBs, face embeddings,
  recordings, and logs inside `%APPDATA%/AirTeachAI` on Windows (never in `Program Files`).
- `setup_app_logging()`: writes structured runtime logs to `%APPDATA%/AirTeachAI/logs/app.log`.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
from typing import Union


APP_NAME = "AirTeachAI"
APP_USER_MODEL_ID = "DNT.AirTeachAI.Studio.2.0"


def get_bundle_root() -> Path:
    """Return read-only bundle root (`sys._MEIPASS` when frozen, or project root in dev)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_path(relative_path: Union[str, Path]) -> Path:
    """Resolve a bundled read-only asset/model/config path for both Dev and Frozen `.exe`."""
    rel = Path(relative_path)
    if rel.is_absolute():
        return rel
    candidate = get_bundle_root() / rel
    if candidate.exists():
        return candidate
    if getattr(sys, "frozen", False):
        exe_sibling = Path(sys.executable).resolve().parent / rel
        if exe_sibling.exists():
            return exe_sibling
    return candidate


def get_user_data_dir() -> Path:
    """
    Return writable per-user data directory (`%APPDATA%/AirTeachAI` on Windows).
    Ensures `logs/`, `face_embeddings/`, and `recordings/` subdirectories exist.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))

    data_dir = base / APP_NAME
    for sub in ("", "logs", "face_embeddings", "recordings"):
        (data_dir / sub if sub else data_dir).mkdir(parents=True, exist_ok=True)
    return data_dir


def get_config_file_path() -> Path:
    """
    Return `%APPDATA%/AirTeachAI/config.json` for API key & user preferences.
    If legacy `data/api_settings.json` exists in dev and `config.json` does not yet exist,
    transparently copy it once to preserve existing developer keys.
    """
    cfg_path = get_user_data_dir() / "config.json"
    if not cfg_path.exists() and not getattr(sys, "frozen", False):
        legacy = get_bundle_root() / "data" / "api_settings.json"
        if legacy.exists():
            try:
                cfg_path.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
            except OSError:
                pass
    return cfg_path


def setup_app_logging() -> Path:
    """Configure rotating file logger at `%APPDATA%/AirTeachAI/logs/app.log`."""
    log_path = get_user_data_dir() / "logs" / "app.log"
    root_logger = logging.getLogger("airteach")
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        try:
            handler = RotatingFileHandler(
                str(log_path),
                maxBytes=2 * 1024 * 1024,
                backupCount=3,
                encoding="utf-8",
            )
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            )
            root_logger.addHandler(handler)
        except OSError:
            pass
    return log_path


def set_windows_app_user_model_id(app_id: str = APP_USER_MODEL_ID) -> bool:
    """Ensure Windows Taskbar displays AirTeach AI's custom `.ico` instead of generic Python icon."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        return True
    except Exception:
        return False
