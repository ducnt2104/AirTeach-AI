# -*- mode: python ; coding: utf-8 -*-
"""
Early PyInstaller runtime hook (`scripts/pyi_rth_airteach_crash.py`).
Captures uncaught startup exceptions before `main.py` initializes Qt logging so
`Chay_AirTeachAI_AutoFix.bat` always records readable tracebacks into `crash_log.txt`
and exits cleanly instead of hanging on a GUI error dialog.
"""

import os
from pathlib import Path
import sys
import traceback
from datetime import datetime


def _airteach_early_excepthook(exc_type, exc_value, exc_tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        return
    tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    try:
        base = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        log_dir = base / "AirTeachAI" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "app.log"
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with log_file.open("a", encoding="utf-8", errors="replace") as fh:
            fh.write(f"\n[{stamp}] [CRITICAL] Startup unhandled exception:\n{tb_text}\n")
    except Exception:
        pass
    try:
        if sys.stderr is not None:
            sys.stderr.write(tb_text + "\n")
            sys.stderr.flush()
    except Exception:
        pass


sys.excepthook = _airteach_early_excepthook
