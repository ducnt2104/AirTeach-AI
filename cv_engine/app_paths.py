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
import threading
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
        exe_dir = Path(sys.executable).resolve().parent
        for base in (exe_dir / "_internal", exe_dir):
            alt = base / rel
            if alt.exists():
                return alt
    return candidate


def get_user_data_dir() -> Path:
    """
    Return writable per-user data directory (`%APPDATA%/AirTeachAI` on Windows).
    Ensures `logs/`, `doc_cache/`, `face_embeddings/`, `recordings/`, and `models/` subdirectories exist.
    """
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))

    data_dir = base / APP_NAME
    for sub in ("", "logs", "doc_cache", "face_embeddings", "recordings", "models"):
        (data_dir / sub if sub else data_dir).mkdir(parents=True, exist_ok=True)
    return data_dir


def get_user_models_dir() -> Path:
    """Return writable `%APPDATA%/AirTeachAI/models` directory for runtime model downloads."""
    models_dir = get_user_data_dir() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_user_vision_config_path() -> Path:
    """Return writable `%APPDATA%/AirTeachAI/vision_config.json` user override path."""
    return get_user_data_dir() / "vision_config.json"


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
            except OSError as exc:
                logging.getLogger("airteach.paths").warning("Failed migrating legacy api_settings.json: %s", exc)
    return cfg_path


def setup_app_logging() -> Path:
    """Configure rotating file logger at `%APPDATA%/AirTeachAI/logs/app.log`."""
    log_path = get_user_data_dir() / "logs" / "app.log"
    root_logger = logging.getLogger("airteach")
    if not root_logger.handlers:
        root_logger.setLevel(logging.INFO)
        root_logger.propagate = False
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
        except OSError as exc:
            sys.stderr.write(f"[AirTeachAI] Cannot open log file {log_path}: {exc}\n")
    return log_path


def get_app_logger(module_name: str = "airteach") -> logging.Logger:
    """Return a child logger under `airteach` writing to `%APPDATA%/AirTeachAI/logs/app.log`."""
    setup_app_logging()
    if module_name == "airteach" or module_name.startswith("airteach."):
        return logging.getLogger(module_name)
    return logging.getLogger(f"airteach.{module_name}")


def get_local_cache_db_path() -> Path:
    """Return `%APPDATA%/AirTeachAI/local_ai_cache.sqlite` (writable even when installed to Program Files)."""
    return get_user_data_dir() / "local_ai_cache.sqlite"


def get_doc_cache_dir() -> Path:
    """Return `%APPDATA%/AirTeachAI/doc_cache` for SHA-256 cached converted PDF/PPTX decks."""
    cache_dir = get_user_data_dir() / "doc_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def install_global_excepthook() -> None:
    """Route unhandled main-thread and worker-thread exceptions to `%APPDATA%/AirTeachAI/logs/app.log`."""
    logger = get_app_logger("runtime")

    def _sys_hook(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_tb))

    def _thread_hook(args: threading.ExceptHookArgs) -> None:  # type: ignore[name-defined]
        if args.exc_type is SystemExit:
            return
        logger.critical(
            "Unhandled thread exception in %s",
            getattr(args.thread, "name", "unknown"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    sys.excepthook = _sys_hook
    if hasattr(threading, "excepthook"):
        threading.excepthook = _thread_hook


def set_windows_app_user_model_id(app_id: str = APP_USER_MODEL_ID) -> bool:
    """Ensure Windows Taskbar displays AirTeach AI's custom `.ico` instead of generic Python icon."""
    if sys.platform != "win32":
        return False
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        return True
    except Exception as exc:
        get_app_logger("app_paths").warning("SetCurrentProcessExplicitAppUserModelID failed: %s", exc)
        return False


def ensure_concurrent_futures_eager() -> None:
    """
    Python 3.15 (PEP 810) defines `lazy from .thread import ThreadPoolExecutor` in
    `concurrent/futures/__init__.py`, which can leak an uncallable `lazy_import` proxy into
    MediaPipe's `serial_dispatcher.SerialDispatcher` (`TypeError: 'lazy_import' object is not callable`).
    Binding the real classes directly onto `concurrent.futures` before `create_from_options()` guarantees
    HandLandmarker / FaceDetector / PoseLandmarker never fail silently on Python 3.15+.
    """
    import concurrent.futures
    from concurrent.futures.process import ProcessPoolExecutor as _RealProcessPoolExecutor
    from concurrent.futures.thread import ThreadPoolExecutor as _RealThreadPoolExecutor

    concurrent.futures.ThreadPoolExecutor = _RealThreadPoolExecutor
    concurrent.futures.ProcessPoolExecutor = _RealProcessPoolExecutor
