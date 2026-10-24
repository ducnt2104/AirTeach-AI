"""
AirTeach AI - Crash Diagnostics & Exception Recorder
=====================================================
Captures uncaught exceptions across the main GUI thread and background worker threads.
Writes structured, sanitized diagnostic crash reports to data/crash_reports/.

Features:
  - Automatic secret/credential redaction (Gemini API keys, bearer tokens, PINs)
  - Environment & system telemetry capture (OS, Python, Qt, thread pool status)
  - Bounded storage policy (keeps last 50 crash dumps to prevent disk bloat)
  - Non-blocking, fail-safe execution (reporting failure never masks root exception)
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Type

logger = logging.getLogger("airteach.crash_diagnostics")

# Secret redaction patterns
_SENSITIVE_PATTERNS = [
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "[REDACTED_GOOGLE_API_KEY]"),
    (re.compile(r"sk-[0-9A-Za-z]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"Bearer\s+[A-Za-z0-9\-_\.=]+", re.IGNORECASE), "Bearer [REDACTED_TOKEN]"),
    (re.compile(r"(['\"]?(?:pin|password|secret|key|token)['\"]?\s*[:=]\s*['\"])[^'\"]+(['\"])", re.IGNORECASE), r"\1[REDACTED]\2"),
]

MAX_CRASH_REPORTS = 50


def sanitize_text(text: str) -> str:
    """Sanitize sensitive credentials, tokens, and keys from diagnostic strings."""
    if not text:
        return ""
    sanitized = text
    for pattern, repl in _SENSITIVE_PATTERNS:
        sanitized = pattern.sub(repl, sanitized)
    return sanitized


class CrashDiagnostics:
    """
    Central crash diagnostics monitor.
    Installs global sys.excepthook and threading.excepthook.
    """

    _instance: Optional["CrashDiagnostics"] = None
    _lock = threading.Lock()

    def __init__(self, reports_dir: Optional[Path] = None) -> None:
        if reports_dir is None:
            from src.core.config import DATA_DIR
            self.reports_dir = DATA_DIR / "crash_reports"
        else:
            self.reports_dir = Path(reports_dir)

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self._installed = False
        self._orig_sys_excepthook: Optional[Callable] = None
        self._orig_thread_excepthook: Optional[Callable] = None

    @classmethod
    def instance(cls, reports_dir: Optional[Path] = None) -> "CrashDiagnostics":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(reports_dir=reports_dir)
            return cls._instance

    def install(self) -> None:
        """Install global exception hooks."""
        with self._lock:
            if self._installed:
                return

            self._orig_sys_excepthook = sys.excepthook
            sys.excepthook = self._handle_sys_exception

            if hasattr(threading, "excepthook"):
                self._orig_thread_excepthook = threading.excepthook
                threading.excepthook = self._handle_thread_exception

            self._installed = True
            logger.info("CrashDiagnostics hooks installed -> %s", self.reports_dir)

    def uninstall(self) -> None:
        """Restore original exception hooks."""
        with self._lock:
            if not self._installed:
                return

            if self._orig_sys_excepthook is not None:
                sys.excepthook = self._orig_sys_excepthook
                self._orig_sys_excepthook = None

            if self._orig_thread_excepthook is not None and hasattr(threading, "excepthook"):
                threading.excepthook = self._orig_thread_excepthook
                self._orig_thread_excepthook = None

            self._installed = False
            logger.info("CrashDiagnostics hooks uninstalled.")

    def record_exception(
        self,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_tb: Any,
        thread_name: Optional[str] = None,
    ) -> Optional[Path]:
        """
        Record a structured diagnostic report for an exception.
        Returns the path to the written JSON report, or None on failure.
        """
        try:
            tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
            raw_tb = "".join(tb_lines)
            clean_tb = sanitize_text(raw_tb)
            clean_msg = sanitize_text(str(exc_value))

            uptime_s = round(time.time() - self.start_time, 2)
            timestamp_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            active_threads = [t.name for t in threading.enumerate()]

            report_data = {
                "timestamp_iso": timestamp_iso,
                "uptime_seconds": uptime_s,
                "exception_type": getattr(exc_type, "__name__", str(exc_type)),
                "exception_message": clean_msg,
                "thread_name": thread_name or threading.current_thread().name,
                "traceback": clean_tb,
                "environment": {
                    "platform": platform.platform(),
                    "python_version": platform.python_version(),
                    "architecture": " ".join(platform.architecture()),
                    "cpu_count": os.cpu_count() or 1,
                    "active_threads": active_threads,
                },
            }

            file_stem = f"crash_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
            report_path = self.reports_dir / f"{file_stem}.json"

            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)

            logger.critical("AirTeach AI crash report saved: %s", report_path)
            self._prune_old_reports()
            return report_path
        except Exception as report_err:
            try:
                sys.stderr.write(f"CrashDiagnostics failed to write report: {report_err}\n")
            except Exception:
                pass
            return None

    def get_recent_crashes(self, limit: int = 10) -> list[dict]:
        """Retrieve recent crash reports sorted from newest to oldest."""
        try:
            json_files = sorted(self.reports_dir.glob("crash_*.json"), key=os.path.getmtime, reverse=True)
            results = []
            for jf in json_files[:limit]:
                try:
                    with jf.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        data["file_path"] = str(jf)
                        results.append(data)
                except Exception:
                    continue
            return results
        except Exception:
            return []

    def _prune_old_reports(self) -> None:
        """Keep at most MAX_CRASH_REPORTS files."""
        try:
            files = sorted(self.reports_dir.glob("crash_*.json"), key=os.path.getmtime)
            if len(files) > MAX_CRASH_REPORTS:
                for f in files[:-MAX_CRASH_REPORTS]:
                    try:
                        f.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    def _handle_sys_exception(
        self,
        exc_type: Type[BaseException],
        exc_value: BaseException,
        exc_tb: Any,
    ) -> None:
        self.record_exception(exc_type, exc_value, exc_tb, thread_name="MainThread")
        if self._orig_sys_excepthook:
            self._orig_sys_excepthook(exc_type, exc_value, exc_tb)

    def _handle_thread_exception(self, args: Any) -> None:
        t_name = getattr(args.thread, "name", "WorkerThread") if args.thread else "WorkerThread"
        self.record_exception(args.exc_type, args.exc_value, args.exc_traceback, thread_name=t_name)
        if self._orig_thread_excepthook:
            self._orig_thread_excepthook(args)


def install_crash_handler(reports_dir: Optional[Path] = None) -> CrashDiagnostics:
    """Helper to initialize and install the crash diagnostics handler."""
    diag = CrashDiagnostics.instance(reports_dir=reports_dir)
    diag.install()
    return diag
