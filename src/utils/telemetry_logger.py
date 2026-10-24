"""
AirTeach AI - Real-time Telemetry Logger
=========================================
Records operational metrics during live sessions and exports to CSV.

Metrics captured:
  - fps, frame_latency_ms
  - raw_vs_smoothed_rmse
  - gesture_state
  - shape_snapped_type
  - ocr_inference_ms
  - timestamp_iso

Designed as a thread-safe, zero-blocking logger using an in-memory
deque + periodic disk flush (every FLUSH_INTERVAL_S seconds).
"""
from __future__ import annotations

import csv
import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

FLUSH_INTERVAL_S: float = 5.0      # flush to disk every 5 seconds
MAX_BUFFER_SIZE: int = 10_000      # keep at most N records in memory

CSV_FIELDS = [
    "timestamp_iso",
    "fps",
    "frame_latency_ms",
    "raw_vs_smoothed_rmse",
    "gesture_state",
    "shape_snapped_type",
    "ocr_inference_ms",
    "extra_json",
]


@dataclass
class TelemetryRecord:
    """A single telemetry observation."""
    timestamp_iso: str = ""
    fps: float = 0.0
    frame_latency_ms: float = 0.0
    raw_vs_smoothed_rmse: float = 0.0
    gesture_state: str = "NONE"
    shape_snapped_type: str = "FREEFORM"
    ocr_inference_ms: float = 0.0
    extra_json: str = ""          # arbitrary JSON string for one-off data


class TelemetryLogger:
    """
    Thread-safe telemetry recorder.

    Usage::

        tel = TelemetryLogger(Path("data/telemetry_report.csv"))
        tel.start()

        # Inside your camera frame loop:
        tel.record(fps=30.0, frame_latency_ms=5.2, gesture_state="DRAW")

        tel.stop()  # flushes remaining records and closes file
    """

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._buffer: deque[TelemetryRecord] = deque(maxlen=MAX_BUFFER_SIZE)
        self._lock = threading.Lock()
        self._flush_thread: Optional[threading.Thread] = None
        self._running = False
        self._file_initialized = False

        # FPS / latency accumulators
        self._last_frame_ts: Optional[float] = None
        self._fps_alpha = 0.1            # EMA smoothing factor for FPS

        self._smoothed_fps: float = 0.0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background flush thread."""
        if self._running:
            return
        self._running = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="TelemetryFlush",
        )
        self._flush_thread.start()
        logger.info("TelemetryLogger started → %s", self.output_path)

    def stop(self) -> None:
        """Stop the flush thread and write remaining records."""
        self._running = False
        if self._flush_thread:
            self._flush_thread.join(timeout=10.0)
        self._flush_to_disk()
        logger.info("TelemetryLogger stopped. Records flushed to %s", self.output_path)

    # ── Recording API ─────────────────────────────────────────────────────────

    def record(
        self,
        fps: Optional[float] = None,
        frame_latency_ms: float = 0.0,
        raw_vs_smoothed_rmse: float = 0.0,
        gesture_state: str = "NONE",
        shape_snapped_type: str = "FREEFORM",
        ocr_inference_ms: float = 0.0,
        extra_json: str = "",
    ) -> None:
        """
        Record one telemetry observation.

        Args:
            fps:                  Frames per second (auto-computed if None).
            frame_latency_ms:     Camera capture + processing latency in ms.
            raw_vs_smoothed_rmse: RMSE between raw and 1€-filtered stroke.
            gesture_state:        Current FSM gesture state name.
            shape_snapped_type:   Shape type returned by GeometricShapeSnapper.
            ocr_inference_ms:     OCR model inference time.
            extra_json:           Optional JSON string for additional metrics.
        """
        now = time.perf_counter()
        ts_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        # Auto-compute FPS from frame timestamps
        if fps is None:
            if self._last_frame_ts is not None:
                dt = now - self._last_frame_ts
                instant_fps = 1.0 / dt if dt > 1e-6 else 0.0
                self._smoothed_fps = (
                    self._fps_alpha * instant_fps
                    + (1 - self._fps_alpha) * self._smoothed_fps
                )
            fps = self._smoothed_fps
        self._last_frame_ts = now

        rec = TelemetryRecord(
            timestamp_iso=ts_iso,
            fps=round(fps, 2),
            frame_latency_ms=round(frame_latency_ms, 3),
            raw_vs_smoothed_rmse=round(raw_vs_smoothed_rmse, 4),
            gesture_state=gesture_state,
            shape_snapped_type=shape_snapped_type,
            ocr_inference_ms=round(ocr_inference_ms, 2),
            extra_json=extra_json,
        )

        with self._lock:
            self._buffer.append(rec)

    def record_frame_tick(self) -> float:
        """
        Convenience method: call once per camera frame to auto-track FPS.
        Returns the current smoothed FPS estimate.
        """
        self.record()
        return self._smoothed_fps

    # ── Statistics ────────────────────────────────────────────────────────────

    def compute_summary(self) -> dict:
        """
        Compute summary statistics from buffered records.

        Returns:
            Dictionary with mean/max/min for numeric fields.
        """
        import json as _json  # noqa: PLC0415
        import statistics  # noqa: PLC0415

        with self._lock:
            records = list(self._buffer)

        if not records:
            return {"count": 0}

        fps_vals = [r.fps for r in records if r.fps > 0]
        latency_vals = [r.frame_latency_ms for r in records if r.frame_latency_ms > 0]
        rmse_vals = [r.raw_vs_smoothed_rmse for r in records if r.raw_vs_smoothed_rmse > 0]
        ocr_vals = [r.ocr_inference_ms for r in records if r.ocr_inference_ms > 0]

        def _stats(vals: list[float]) -> dict:
            if not vals:
                return {"mean": 0, "min": 0, "max": 0, "count": 0}
            return {
                "mean": round(statistics.mean(vals), 3),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
                "count": len(vals),
            }

        # Gesture distribution
        from collections import Counter  # noqa: PLC0415
        gesture_counts = Counter(r.gesture_state for r in records)
        shape_counts = Counter(r.shape_snapped_type for r in records)

        return {
            "count": len(records),
            "fps": _stats(fps_vals),
            "frame_latency_ms": _stats(latency_vals),
            "raw_vs_smoothed_rmse": _stats(rmse_vals),
            "ocr_inference_ms": _stats(ocr_vals),
            "gesture_distribution": dict(gesture_counts),
            "shape_distribution": dict(shape_counts),
        }

    # ── Internal flush ─────────────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        """Background loop: flush to disk every FLUSH_INTERVAL_S seconds."""
        while self._running:
            time.sleep(FLUSH_INTERVAL_S)
            self._flush_to_disk()

    def _flush_to_disk(self) -> None:
        """Write buffered records to CSV. Creates header on first write."""
        with self._lock:
            if not self._buffer:
                return
            records_to_write = list(self._buffer)
            self._buffer.clear()

        write_header = not self._file_initialized and not self.output_path.exists()
        self._file_initialized = True

        try:
            with self.output_path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if write_header:
                    writer.writeheader()
                for rec in records_to_write:
                    writer.writerow(asdict(rec))
        except OSError as exc:
            logger.error("TelemetryLogger flush failed: %s", exc)
            with self._lock:
                for rec in reversed(records_to_write):
                    if len(self._buffer) < MAX_BUFFER_SIZE:
                        self._buffer.appendleft(rec)

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "TelemetryLogger":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()
