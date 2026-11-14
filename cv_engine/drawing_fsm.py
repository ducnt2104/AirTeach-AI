"""
AirTeach AI (DNT - Victer) — 4-Tier Zero-Stray-Stroke Drawing FSM
=================================================================
Giải quyết triệt để lỗi vẽ thêm nét thừa khi nhấc/hủy bút:
1. Schmitt Trigger (Ngưỡng kép chống rung):
   - pinch_ratio = dist(4, 8) / dist(0, 9)
   - Vào DRAWING khi pinch_ratio < 0.28 giữ ổn định >= 3 frame.
   - Ra HOVER chỉ khi pinch_ratio > 0.40 (vùng trễ rộng chống nhấp nháy).
2. Pre-Release Tail Trimming:
   - Ring buffer có timestamp cho từng điểm.
   - Khi chuyển từ DRAWING -> HOVER hoặc CANCELLED, tự động cắt bỏ các điểm
     thu trong 90-120ms cuối (mặc định 105ms / 3-4 frame cuối) trước khi chốt nét.
3. Instant Cancel & Pen-Lock Cooldown (300ms):
   - Xòe 5 ngón (OPEN_PALM >= 150ms) hoặc Nắm tay (FIST >= 150ms) khi đang vẽ
     hoặc vừa vẽ xong (trong 700ms), hoặc phím Esc -> Hủy/Thu hồi nét hiện tại
     và kích hoạt khóa ngòi bút 300ms (LOCKED_COOLDOWN).
4. 4 Chế độ HUD tách biệt rõ ràng: HOVER | DRAWING | LOCKED_COOLDOWN | PAN_ERASE
5. Undo (nút / Ctrl+Z / cử chỉ 3 ngón giữ 220ms) với cooldown chống lặp 400ms.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple


class DrawingHudMode(str, Enum):
    HOVER = "HOVER"
    DRAWING = "DRAWING"
    LOCKED_COOLDOWN = "LOCKED_COOLDOWN"
    PAN_ERASE = "PAN_ERASE"

    @property
    def label_vi(self) -> str:
        return {
            DrawingHudMode.HOVER: "Hover (Di chuyển)",
            DrawingHudMode.DRAWING: "Drawing (Đang vẽ)",
            DrawingHudMode.LOCKED_COOLDOWN: "Locked/Cooldown (Đã hủy • Khóa tạm)",
            DrawingHudMode.PAN_ERASE: "Pan/Erase (Tẩy / Di chuyển)",
        }[self]


PINCH_ENTER_THRESHOLD = 0.28
PINCH_EXIT_THRESHOLD = 0.40
PINCH_ENTER_MIN_FRAMES = 3
TAIL_TRIM_MS = 105.0
TAIL_TRIM_MAX_POINTS = 4
CANCEL_HOLD_MS = 150.0
POST_COMMIT_CANCEL_WINDOW_MS = 700.0
PEN_LOCK_COOLDOWN_MS = 300.0
UNDO_HOLD_MS = 220.0
UNDO_COOLDOWN_MS = 400.0


@dataclass
class TimedPoint:
    x: float
    y: float
    t_ms: float


class StrokeRingBuffer:
    """Bounded timestamped point buffer supporting exact pre-release tail trimming."""

    def __init__(self, maxlen: int = 4096) -> None:
        self._buf: Deque[TimedPoint] = deque(maxlen=maxlen)

    def clear(self) -> None:
        self._buf.clear()

    def append(self, x: float, y: float, t_ms: float) -> None:
        self._buf.append(TimedPoint(float(x), float(y), float(t_ms)))

    def points_xy(self) -> List[Tuple[float, float]]:
        return [(p.x, p.y) for p in self._buf]

    def __len__(self) -> int:
        return len(self._buf)

    def trim_and_extract(
        self,
        tail_trim_ms: float = TAIL_TRIM_MS,
        max_trim_points: int = TAIL_TRIM_MAX_POINTS,
    ) -> List[Tuple[float, float]]:
        """
        Discard trailing points collected during final `90-120ms` (`tail_trim_ms`)
        before pen lift/cancel to eliminate flick hooks.
        """
        if not self._buf:
            return []
        t_end = self._buf[-1].t_ms
        cutoff = t_end - max(90.0, min(120.0, tail_trim_ms))

        trimmed = list(self._buf)
        removed = 0
        # Always trim time-windowed tail up to max_trim_points (or full window if >60fps)
        while len(trimmed) > 0 and trimmed[-1].t_ms > cutoff:
            # Avoid emptying a valid multi-segment high-fps tail partially unless whole buffer is short flicker
            if removed >= max(max_trim_points, 6) and len(trimmed) >= 2:
                break
            trimmed.pop()
            removed += 1

        # If timing had sparse timestamps (>35ms step), ensure at least 1-3 release transition frames are evaluated
        if removed == 0 and len(trimmed) >= 5:
            trimmed.pop()

        self._buf.clear()
        if len(trimmed) < 2:
            return []
        return [(p.x, p.y) for p in trimmed]


@dataclass
class DrawingFsmUpdate:
    hud_mode: DrawingHudMode
    action: str
    cursor_xy: Optional[Tuple[float, float]]
    committed_points: List[Tuple[float, float]] = field(default_factory=list)
    active_points: List[Tuple[float, float]] = field(default_factory=list)
    cooldown_remaining_ms: float = 0.0
    undo_cooldown_remaining_ms: float = 0.0
    cancel_reason: str = ""
    pinch_ratio: Optional[float] = None
    cancel_hold_progress: float = 0.0


class ZeroStrayStrokeDrawingFsm:
    """
    Deterministic 4-tier drawing state machine for AirTeach AI.
    Accepts explicit timestamps (`t_ms`) so real-time camera streams and headless unit tests
    produce bit-exact state transitions.
    """

    def __init__(
        self,
        pinch_enter: float = PINCH_ENTER_THRESHOLD,
        pinch_exit: float = PINCH_EXIT_THRESHOLD,
        pinch_enter_frames: int = PINCH_ENTER_MIN_FRAMES,
        tail_trim_ms: float = TAIL_TRIM_MS,
        cancel_hold_ms: float = CANCEL_HOLD_MS,
        pen_lock_cooldown_ms: float = PEN_LOCK_COOLDOWN_MS,
        undo_cooldown_ms: float = UNDO_COOLDOWN_MS,
    ) -> None:
        self.pinch_enter = float(pinch_enter)
        self.pinch_exit = float(pinch_exit)
        self.pinch_enter_frames = max(1, int(pinch_enter_frames))
        self.tail_trim_ms = max(90.0, min(120.0, float(tail_trim_ms)))
        self.cancel_hold_ms = float(cancel_hold_ms)
        self.pen_lock_cooldown_ms = float(pen_lock_cooldown_ms)
        self.undo_cooldown_ms = float(undo_cooldown_ms)

        self.hud_mode: DrawingHudMode = DrawingHudMode.HOVER
        self.buffer = StrokeRingBuffer()

        self._pinch_hold_count: int = 0
        self._locked_until_ms: float = -1e9
        self._last_commit_ms: float = -1e9
        self._last_undo_ms: float = -1e9

        self._cancel_candidate: str = ""
        self._cancel_since_ms: Optional[float] = None
        self._undo_since_ms: Optional[float] = None
        self._undo_fired_for_hold: bool = False
        self._eraser_suppress_until_ms: float = -1e9

    def reset(self) -> None:
        self.hud_mode = DrawingHudMode.HOVER
        self.buffer.clear()
        self._pinch_hold_count = 0
        self._locked_until_ms = -1e9
        self._last_commit_ms = -1e9
        self._last_undo_ms = -1e9
        self._cancel_candidate = ""
        self._cancel_since_ms = None
        self._undo_since_ms = None
        self._undo_fired_for_hold = False
        self._eraser_suppress_until_ms = -1e9

    def step(
        self,
        t_ms: float,
        hand_present: bool,
        cursor_xy: Optional[Tuple[float, float]],
        pinch_ratio: Optional[float],
        pose_tag: str,
        *,
        left_pan_active: bool = False,
        esc_pressed: bool = False,
        undo_requested: bool = False,
    ) -> DrawingFsmUpdate:
        was_drawing = self.hud_mode == DrawingHudMode.DRAWING
        recently_committed = (t_ms - self._last_commit_ms) <= POST_COMMIT_CANCEL_WINDOW_MS

        # ------------------------------------------------------------------
        # 1. Explicit Keyboard Escape -> Instant Cancel + 300ms Pen Lock
        # ------------------------------------------------------------------
        if esc_pressed:
            self._pinch_hold_count = 0
            self._locked_until_ms = t_ms + self.pen_lock_cooldown_ms
            self.hud_mode = DrawingHudMode.LOCKED_COOLDOWN
            if was_drawing:
                _ = self.buffer.trim_and_extract(self.tail_trim_ms)
                self.buffer.clear()
                return DrawingFsmUpdate(
                    hud_mode=self.hud_mode,
                    action="CANCEL_STROKE",
                    cursor_xy=cursor_xy,
                    cooldown_remaining_ms=self.pen_lock_cooldown_ms,
                    cancel_reason="ESC",
                    pinch_ratio=pinch_ratio,
                )
            if recently_committed:
                self._last_commit_ms = -1e9
                return DrawingFsmUpdate(
                    hud_mode=self.hud_mode,
                    action="ROLLBACK_LAST_STROKE",
                    cursor_xy=cursor_xy,
                    cooldown_remaining_ms=self.pen_lock_cooldown_ms,
                    cancel_reason="ESC",
                    pinch_ratio=pinch_ratio,
                )
            return DrawingFsmUpdate(
                hud_mode=self.hud_mode,
                action="NONE",
                cursor_xy=cursor_xy,
                cooldown_remaining_ms=self.pen_lock_cooldown_ms,
                cancel_reason="ESC",
                pinch_ratio=pinch_ratio,
            )

        # ------------------------------------------------------------------
        # 2. Active Post-Cancel Pen Lock Cooldown (`300ms`)
        # ------------------------------------------------------------------
        if t_ms < self._locked_until_ms:
            self.hud_mode = DrawingHudMode.LOCKED_COOLDOWN
            self._pinch_hold_count = 0
            self.buffer.clear()
            rem = self._locked_until_ms - t_ms
            # Allow Ctrl+Z / button Undo even during pen lock if cooldown allows
            undo_rem = max(0.0, self.undo_cooldown_ms - (t_ms - self._last_undo_ms))
            if undo_requested and undo_rem <= 0.0:
                self._last_undo_ms = t_ms
                return DrawingFsmUpdate(
                    hud_mode=self.hud_mode,
                    action="UNDO",
                    cursor_xy=cursor_xy,
                    cooldown_remaining_ms=rem,
                    undo_cooldown_remaining_ms=self.undo_cooldown_ms,
                    pinch_ratio=pinch_ratio,
                )
            return DrawingFsmUpdate(
                hud_mode=self.hud_mode,
                action="NONE",
                cursor_xy=cursor_xy,
                cooldown_remaining_ms=rem,
                undo_cooldown_remaining_ms=undo_rem,
                pinch_ratio=pinch_ratio,
            )

        # ------------------------------------------------------------------
        # 3. Cancel Gesture Hold Detection (`OPEN_PALM` or `FIST` >= 150ms)
        # ------------------------------------------------------------------
        cancel_progress = 0.0
        can_trigger_cancel = was_drawing or recently_committed
        if hand_present and pose_tag in ("OPEN_PALM", "FIST") and can_trigger_cancel:
            if self._cancel_candidate != pose_tag:
                self._cancel_candidate = pose_tag
                self._cancel_since_ms = t_ms
            held_ms = t_ms - (self._cancel_since_ms if self._cancel_since_ms is not None else t_ms)
            cancel_progress = min(1.0, max(0.0, held_ms / max(self.cancel_hold_ms, 1.0)))
            if held_ms >= self.cancel_hold_ms:
                reason = pose_tag
                self._cancel_candidate = ""
                self._cancel_since_ms = None
                self._pinch_hold_count = 0
                self._locked_until_ms = t_ms + self.pen_lock_cooldown_ms
                # Suppress accidental immediate eraser sweep from canceling OPEN_PALM
                self._eraser_suppress_until_ms = t_ms + 550.0
                self.hud_mode = DrawingHudMode.LOCKED_COOLDOWN
                _ = self.buffer.trim_and_extract(self.tail_trim_ms)
                self.buffer.clear()
                if was_drawing:
                    return DrawingFsmUpdate(
                        hud_mode=self.hud_mode,
                        action="CANCEL_STROKE",
                        cursor_xy=cursor_xy,
                        cooldown_remaining_ms=self.pen_lock_cooldown_ms,
                        cancel_reason=reason,
                        pinch_ratio=pinch_ratio,
                        cancel_hold_progress=1.0,
                    )
                self._last_commit_ms = -1e9
                return DrawingFsmUpdate(
                    hud_mode=self.hud_mode,
                    action="ROLLBACK_LAST_STROKE",
                    cursor_xy=cursor_xy,
                    cooldown_remaining_ms=self.pen_lock_cooldown_ms,
                    cancel_reason=reason,
                    pinch_ratio=pinch_ratio,
                    cancel_hold_progress=1.0,
                )
        else:
            self._cancel_candidate = ""
            self._cancel_since_ms = None

        # ------------------------------------------------------------------
        # 4. Undo Gesture (`THREE_FINGERS` >= 220ms) or Quick Button / Ctrl+Z (`400ms` cooldown)
        # ------------------------------------------------------------------
        gesture_undo_trigger = False
        if hand_present and not was_drawing and pose_tag == "THREE_FINGERS":
            if self._undo_since_ms is None:
                self._undo_since_ms = t_ms
                self._undo_fired_for_hold = False
            if (t_ms - self._undo_since_ms) >= UNDO_HOLD_MS and not self._undo_fired_for_hold:
                self._undo_fired_for_hold = True
                gesture_undo_trigger = True
        else:
            self._undo_since_ms = None
            self._undo_fired_for_hold = False

        if undo_requested or gesture_undo_trigger:
            if (t_ms - self._last_undo_ms) >= self.undo_cooldown_ms:
                self._last_undo_ms = t_ms
                self.hud_mode = DrawingHudMode.HOVER
                return DrawingFsmUpdate(
                    hud_mode=self.hud_mode,
                    action="UNDO",
                    cursor_xy=cursor_xy,
                    undo_cooldown_remaining_ms=self.undo_cooldown_ms,
                    pinch_ratio=pinch_ratio,
                )

        # ------------------------------------------------------------------
        # 5. Schmitt Trigger Drawing & Transition with Pre-Release Tail Trim
        # ------------------------------------------------------------------
        undo_rem = max(0.0, self.undo_cooldown_ms - (t_ms - self._last_undo_ms))

        if was_drawing:
            # Stay in DRAWING until pinch_ratio explicitly opens past PINCH_EXIT_THRESHOLD (> 0.40) or hand vanishes.
            # Defer release briefly while an active cancel hold (OPEN_PALM/FIST) is accumulating toward 150ms.
            should_release = (
                not hand_present
                or cursor_xy is None
                or (
                    pinch_ratio is not None
                    and pinch_ratio > self.pinch_exit
                    and not self._cancel_candidate
                )
            )
            if should_release:
                committed = self.buffer.trim_and_extract(self.tail_trim_ms)
                self._pinch_hold_count = 0
                self.hud_mode = DrawingHudMode.HOVER
                if len(committed) >= 2:
                    self._last_commit_ms = t_ms
                    return DrawingFsmUpdate(
                        hud_mode=self.hud_mode,
                        action="COMMIT_STROKE",
                        cursor_xy=cursor_xy,
                        committed_points=committed,
                        undo_cooldown_remaining_ms=undo_rem,
                        pinch_ratio=pinch_ratio,
                        cancel_hold_progress=cancel_progress,
                    )
                return DrawingFsmUpdate(
                    hud_mode=self.hud_mode,
                    action="CANCEL_STROKE",
                    cursor_xy=cursor_xy,
                    undo_cooldown_remaining_ms=undo_rem,
                    cancel_reason="SHORT_FLICK",
                    pinch_ratio=pinch_ratio,
                    cancel_hold_progress=cancel_progress,
                )

            # Still pinching (`pinch_ratio <= 0.40`): append live point unless FIST hold already starting to cancel
            if self._cancel_candidate == "":
                self.buffer.append(cursor_xy[0], cursor_xy[1], t_ms)
            self.hud_mode = DrawingHudMode.DRAWING
            return DrawingFsmUpdate(
                hud_mode=self.hud_mode,
                action="APPEND_POINT",
                cursor_xy=cursor_xy,
                active_points=self.buffer.points_xy(),
                undo_cooldown_remaining_ms=undo_rem,
                pinch_ratio=pinch_ratio,
                cancel_hold_progress=cancel_progress,
            )

        # Currently NOT drawing (`HOVER` or `PAN_ERASE`):
        if (
            hand_present
            and cursor_xy is not None
            and pinch_ratio is not None
            and pinch_ratio < self.pinch_enter
            and pose_tag not in ("OPEN_PALM", "FIST", "THREE_FINGERS")
        ):
            self._pinch_hold_count += 1
            if self._pinch_hold_count >= self.pinch_enter_frames:
                self._pinch_hold_count = 0
                self.hud_mode = DrawingHudMode.DRAWING
                self.buffer.clear()
                self.buffer.append(cursor_xy[0], cursor_xy[1], t_ms)
                return DrawingFsmUpdate(
                    hud_mode=self.hud_mode,
                    action="BEGIN_STROKE",
                    cursor_xy=cursor_xy,
                    active_points=self.buffer.points_xy(),
                    undo_cooldown_remaining_ms=undo_rem,
                    pinch_ratio=pinch_ratio,
                )
        else:
            self._pinch_hold_count = 0

        # Pan or Idle-Palm Erase Mode
        if hand_present and pose_tag == "OPEN_PALM" and not can_trigger_cancel and t_ms >= self._eraser_suppress_until_ms:
            self.hud_mode = DrawingHudMode.PAN_ERASE
            return DrawingFsmUpdate(
                hud_mode=self.hud_mode,
                action="ERASE",
                cursor_xy=cursor_xy,
                undo_cooldown_remaining_ms=undo_rem,
                pinch_ratio=pinch_ratio,
                cancel_hold_progress=cancel_progress,
            )

        if left_pan_active:
            self.hud_mode = DrawingHudMode.PAN_ERASE
            return DrawingFsmUpdate(
                hud_mode=self.hud_mode,
                action="PAN",
                cursor_xy=cursor_xy,
                undo_cooldown_remaining_ms=undo_rem,
                pinch_ratio=pinch_ratio,
            )

        self.hud_mode = DrawingHudMode.HOVER
        return DrawingFsmUpdate(
            hud_mode=self.hud_mode,
            action="NONE",
            cursor_xy=cursor_xy,
            undo_cooldown_remaining_ms=undo_rem,
            pinch_ratio=pinch_ratio,
            cancel_hold_progress=cancel_progress,
        )
