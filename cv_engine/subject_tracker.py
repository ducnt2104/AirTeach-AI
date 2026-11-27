"""
Tier 2 Vision Engine: Multi-Modal Subject Tracking, Pose Wrist Anchoring & Primary Teacher Lock
(`cv_engine/subject_tracker.py`).

Prevents control hijacking when students or passersby walk behind the teacher and provides
proactive Pose-Wrist ROI search windows for 1.5m-3.0m long-range hand tracking:
1. Combines optional MediaPipe PoseLandmarker (when `pose_landmarker*.task` is present) with
   robust Face-to-Torso proxy estimation so offline environments always operate deterministically.
2. Persists `track_id` via Bounding-Box IoU + Centroid Distance + optional 512-D ArcFace similarity.
3. Enforces a configurable `1.0s–2.0s` (default `1.5s`) Grace Period so a passing background
   person CANNOT steal the Teacher Lock during brief occlusion/turning.
4. Executes Wrist-to-Hand Spatial Gating to discard hands belonging to non-locked people while
   preserving long-range hands anchored to the locked teacher.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from cv_engine.app_paths import get_app_logger

_LOGGER = get_app_logger("subject_tracker")

try:
    import mediapipe as mp  # type: ignore

    HAS_MEDIAPIPE_POSE = hasattr(mp, "tasks") and hasattr(mp.tasks, "python")
except Exception as exc:
    _LOGGER.warning("MediaPipe Pose unavailable: %s", exc)
    mp = None  # type: ignore
    HAS_MEDIAPIPE_POSE = False


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = float(a[0]), float(a[1]), max(1e-5, float(a[2])), max(1e-5, float(a[3]))
    bx, by, bw, bh = float(b[0]), float(b[1]), max(1e-5, float(b[2])), max(1e-5, float(b[3]))
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter / max(1e-6, union))


def _bbox_center(b: Sequence[float]) -> Tuple[float, float]:
    return (float(b[0]) + 0.5 * float(b[2]), float(b[1]) + 0.5 * float(b[3]))


def _point_dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _point_in_expanded_box(
    pt: Tuple[float, float],
    box: Sequence[float],
    pad_x: float = 0.0,
    pad_y: float = 0.0,
) -> bool:
    bx, by, bw, bh = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return (
        (bx - pad_x) <= pt[0] <= (bx + bw + pad_x)
        and (by - pad_y) <= pt[1] <= (by + bh + pad_y)
    )


def estimate_torso_and_wrists_from_face(
    face_bbox: Sequence[float],
    reach_scale: float = 2.35,
) -> Tuple[List[float], List[Tuple[float, float]]]:
    """
    Estimate upper-body torso + reaching envelope & wrist anchor proxies from normalized face `[fx, fy, fw, fh]`.
    Works reliably at both 0.6m close-up and 1.5m–3.0m distance even when Pose model is offline.
    """
    fx, fy, fw, fh = (
        float(face_bbox[0]),
        float(face_bbox[1]),
        max(0.02, float(face_bbox[2])),
        max(0.02, float(face_bbox[3])),
    )
    cx = fx + 0.5 * fw
    torso_w = max(fw * 2.1, fh * 1.65)
    torso_h = fh * 3.8
    tx = max(0.0, cx - 0.5 * torso_w)
    ty = max(0.0, fy - 0.35 * fh)
    tw = min(1.0 - tx, torso_w)
    th = min(1.0 - ty, torso_h)

    shoulder_y = min(0.95, fy + fh * 1.55)
    shoulder_dx = max(fw * 1.35, 0.10)
    left_wrist_proxy = (max(0.04, min(0.96, cx - shoulder_dx)), shoulder_y)
    right_wrist_proxy = (max(0.04, min(0.96, cx + shoulder_dx)), shoulder_y)
    return [round(tx, 4), round(ty, 4), round(tw, 4), round(th, 4)], [
        left_wrist_proxy,
        right_wrist_proxy,
    ]


@dataclass
class SubjectCandidate:
    face_bbox: List[float]
    torso_bbox: List[float]
    wrist_anchors: List[Tuple[float, float]]
    wrist_search_radius: float
    face_sim: float = 0.0
    pose_backed: bool = False


@dataclass
class TrackState:
    track_id: int
    face_bbox: List[float]
    torso_bbox: List[float]
    centroid: Tuple[float, float]
    area: float
    wrist_anchors: List[Tuple[float, float]]
    wrist_search_radius: float
    last_seen_ts: float
    frames_seen: int = 1
    face_sim: float = 0.0
    pose_backed: bool = False


@dataclass
class SubjectTrackingSnapshot:
    locked_teacher_id: Optional[int] = None
    locked_face_bbox: Optional[List[float]] = None
    locked_torso_bbox: Optional[List[float]] = None
    locked_wrist_anchors: List[Tuple[float, float]] = field(default_factory=list)
    wrist_search_radius: float = 0.16
    ignored_face_bboxes: List[List[float]] = field(default_factory=list)
    ignored_torso_bboxes: List[List[float]] = field(default_factory=list)
    teacher_in_grace_period: bool = False
    pose_landmarker_active: bool = False
    teacher_lock_enabled: bool = True
    bypass_active: bool = True
    person_count: int = 0
    mode: str = "UNLOCKED"

    def to_dict(self) -> Dict[str, object]:
        return {
            "locked_teacher_id": self.locked_teacher_id,
            "locked_face_bbox": self.locked_face_bbox,
            "locked_torso_bbox": self.locked_torso_bbox,
            "locked_wrist_anchors": [[round(x, 4), round(y, 4)] for x, y in self.locked_wrist_anchors],
            "ignored_face_bboxes": self.ignored_face_bboxes,
            "ignored_torso_bboxes": self.ignored_torso_bboxes,
            "teacher_in_grace_period": self.teacher_in_grace_period,
            "pose_landmarker_active": self.pose_landmarker_active,
            "teacher_lock_enabled": self.teacher_lock_enabled,
            "bypass_active": self.bypass_active,
            "person_count": self.person_count,
            "mode": self.mode,
        }


class PoseTorsoEstimator:
    """
    Optional MediaPipe `PoseLandmarker` wrapper (`num_poses=3`).
    If no `pose_landmarker*.task` model is on disk, returns `[]` cleanly in ~0ms so
    face-to-torso estimation handles everything offline without errors.
    """

    def __init__(
        self,
        models_dir: Path = Path("models"),
        enable_pose_landmarker: bool = True,
        min_det_conf: float = 0.50,
        min_pres_conf: float = 0.50,
        min_track_conf: float = 0.55,
    ) -> None:
        self.landmarker = None
        if not enable_pose_landmarker or not HAS_MEDIAPIPE_POSE:
            return
        model_candidates = [
            models_dir / "pose_landmarker_full.task",
            models_dir / "pose_landmarker_lite.task",
            models_dir / "pose_landmarker_heavy.task",
        ]
        chosen = next((p for p in model_candidates if p.exists() and p.stat().st_size > 100_000), None)
        if chosen is None:
            return
        try:
            from cv_engine.app_paths import ensure_concurrent_futures_eager

            ensure_concurrent_futures_eager()
            base_opts = mp.tasks.BaseOptions(model_asset_path=str(chosen))
            opts = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_opts,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_poses=3,
                min_pose_detection_confidence=float(min_det_conf),
                min_pose_presence_confidence=float(min_pres_conf),
                min_tracking_confidence=float(min_track_conf),
            )
            self.landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(opts)
        except Exception as exc:
            _LOGGER.warning("PoseLandmarker init failed (%s): %s", chosen, exc)
            self.landmarker = None

    def detect_poses(self, rgb_frame: np.ndarray) -> List[Dict[str, object]]:
        if self.landmarker is None or rgb_frame is None or rgb_frame.size == 0:
            return []
        try:
            rgb_contig = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contig)
            res = self.landmarker.detect(mp_img)
            poses: List[Dict[str, object]] = []
            for plm in (res.pose_landmarks or []):
                if len(plm) < 17:
                    continue
                xs = [max(0.0, min(1.0, float(p.x))) for p in plm[:25]]
                ys = [max(0.0, min(1.0, float(p.y))) for p in plm[:25]]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                shoulder_span = math.hypot(plm[11].x - plm[12].x, plm[11].y - plm[12].y)
                wrists: List[Tuple[float, float]] = []
                for w_idx in (15, 16):
                    vis = float(getattr(plm[w_idx], "visibility", 1.0) or 0.0)
                    if vis >= 0.25:
                        wrists.append(
                            (
                                max(0.0, min(1.0, float(plm[w_idx].x))),
                                max(0.0, min(1.0, float(plm[w_idx].y))),
                            )
                        )
                nose = (max(0.0, min(1.0, float(plm[0].x))), max(0.0, min(1.0, float(plm[0].y))))
                poses.append(
                    {
                        "nose": nose,
                        "wrists": wrists,
                        "torso_bbox": [
                            round(max(0.0, x_min - 0.06), 4),
                            round(max(0.0, y_min - 0.04), 4),
                            round(min(1.0, (x_max - x_min) + 0.12), 4),
                            round(min(1.0, (y_max - y_min) + 0.08), 4),
                        ],
                        "search_radius": max(0.11, min(0.28, shoulder_span * 0.85)),
                    }
                )
            return poses
        except Exception as exc:
            _LOGGER.warning("PoseLandmarker inference failed: %s", exc)
            return []

    def close(self) -> None:
        if self.landmarker is not None and hasattr(self.landmarker, "close"):
            try:
                self.landmarker.close()
            except Exception as exc:
                _LOGGER.warning("PoseLandmarker close error: %s", exc)
            self.landmarker = None


class PrimaryTeacherLockTracker:
    """
    Stateful Multi-Person Track Manager + Primary Teacher Lock + Anti-Interference Gate.

    - Assigns stable IDs to detected people (IoU + Centroid Distance).
    - Locks onto verified ArcFace teacher OR dominant central teacher.
    - Holds lock through up to `teacher_grace_sec` (1.5s default) occlusion without
      handing control to passersby.
    - Provides `gate_hands_to_locked_teacher` and proactive Pose-Wrist ROI search boxes.
    """

    def __init__(
        self,
        teacher_grace_sec: float = 1.5,
        match_iou_min: float = 0.18,
        face_sim_threshold: float = 0.48,
        wrist_gate_radius_scale: float = 2.35,
        enabled: bool = True,
    ) -> None:
        self.teacher_grace_sec = max(0.5, float(teacher_grace_sec))
        self.match_iou_min = float(match_iou_min)
        self.face_sim_threshold = float(face_sim_threshold)
        self.wrist_gate_radius_scale = float(wrist_gate_radius_scale)
        self.enabled = bool(enabled)

        self._next_track_id = 1
        self._tracks: Dict[int, TrackState] = {}
        self._locked_track_id: Optional[int] = None
        self._last_snapshot = SubjectTrackingSnapshot(teacher_lock_enabled=self.enabled)

    @property
    def locked_track_id(self) -> Optional[int]:
        return self._locked_track_id

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self._locked_track_id = None

    def reset(self) -> None:
        self._tracks.clear()
        self._locked_track_id = None
        self._last_snapshot = SubjectTrackingSnapshot(teacher_lock_enabled=self.enabled)

    def force_unlock(self) -> None:
        self._locked_track_id = None

    def update(
        self,
        face_boxes: Sequence[Sequence[float]],
        poses: Optional[Sequence[Dict[str, object]]] = None,
        primary_face_sim: float = 0.0,
        now_sec: Optional[float] = None,
    ) -> SubjectTrackingSnapshot:
        ts = float(time.monotonic() if now_sec is None else now_sec)
        pose_list = list(poses or [])

        # Deduplicate face boxes that overlap heavily (face_id prepends primary_box sometimes)
        dedup_faces: List[List[float]] = []
        for fb in face_boxes:
            if len(fb) < 4:
                continue
            cand = [float(fb[0]), float(fb[1]), float(fb[2]), float(fb[3])]
            if any(_bbox_iou(cand, existing) > 0.72 for existing in dedup_faces):
                continue
            dedup_faces.append(cand)

        candidates: List[SubjectCandidate] = []
        used_pose_indices = set()
        for f_idx, fbox in enumerate(dedup_faces):
            f_center = _bbox_center(fbox)
            # Match closest pose nose if available
            matched_pose = None
            best_p_idx = -1
            best_p_dist = 0.20
            for p_idx, p in enumerate(pose_list):
                if p_idx in used_pose_indices:
                    continue
                d = _point_dist(f_center, tuple(p["nose"]))  # type: ignore[arg-type]
                if d < best_p_dist:
                    best_p_dist = d
                    matched_pose = p
                    best_p_idx = p_idx

            if matched_pose is not None:
                used_pose_indices.add(best_p_idx)
                torso = list(matched_pose["torso_bbox"])  # type: ignore[arg-type]
                wrists = [tuple(w) for w in (matched_pose.get("wrists") or [])]  # type: ignore[union-attr]
                radius = float(matched_pose.get("search_radius") or 0.16)
                pose_backed = True
            else:
                torso, wrists = estimate_torso_and_wrists_from_face(
                    fbox, reach_scale=self.wrist_gate_radius_scale
                )
                radius = max(0.12, min(0.28, max(fbox[2], fbox[3]) * 1.45))
                pose_backed = False

            sim = float(primary_face_sim) if f_idx == 0 else 0.0
            candidates.append(
                SubjectCandidate(
                    face_bbox=fbox,
                    torso_bbox=torso,
                    wrist_anchors=wrists,
                    wrist_search_radius=radius,
                    face_sim=sim,
                    pose_backed=pose_backed,
                )
            )

        # Also add orphan poses if face wasn't detected (e.g. teacher turned away briefly)
        for p_idx, p in enumerate(pose_list):
            if p_idx in used_pose_indices:
                continue
            nx, ny = p["nose"]  # type: ignore[misc]
            proxy_face = [max(0.0, nx - 0.05), max(0.0, ny - 0.06), 0.10, 0.12]
            candidates.append(
                SubjectCandidate(
                    face_bbox=proxy_face,
                    torso_bbox=list(p["torso_bbox"]),  # type: ignore[arg-type]
                    wrist_anchors=[tuple(w) for w in (p.get("wrists") or [])],  # type: ignore[union-attr]
                    wrist_search_radius=float(p.get("search_radius") or 0.16),
                    face_sim=0.0,
                    pose_backed=True,
                )
            )

        # Associate candidates to existing tracks (IoU + Centroid Distance)
        matched_tracks: Dict[int, SubjectCandidate] = {}
        unmatched_candidates: List[SubjectCandidate] = []
        used_track_ids = set()

        for cand in candidates:
            c_center = _bbox_center(cand.face_bbox)
            best_tid: Optional[int] = None
            best_score = -1e9
            for tid, tr in self._tracks.items():
                if tid in used_track_ids:
                    continue
                iou = _bbox_iou(cand.face_bbox, tr.face_bbox)
                dist = _point_dist(c_center, tr.centroid)
                if iou >= self.match_iou_min or dist <= 0.18:
                    score = iou * 2.5 - dist * 3.0
                    if score > best_score:
                        best_score = score
                        best_tid = tid
            if best_tid is not None:
                used_track_ids.add(best_tid)
                matched_tracks[best_tid] = cand
            else:
                unmatched_candidates.append(cand)

        for tid, cand in matched_tracks.items():
            tr = self._tracks[tid]
            tr.face_bbox = cand.face_bbox
            tr.torso_bbox = cand.torso_bbox
            tr.centroid = _bbox_center(cand.face_bbox)
            tr.area = max(1e-5, cand.face_bbox[2] * cand.face_bbox[3])
            tr.wrist_anchors = cand.wrist_anchors
            tr.wrist_search_radius = cand.wrist_search_radius
            tr.last_seen_ts = ts
            tr.frames_seen += 1
            tr.face_sim = max(tr.face_sim * 0.6, cand.face_sim) if cand.face_sim > 0 else tr.face_sim * 0.9
            tr.pose_backed = cand.pose_backed

        for cand in unmatched_candidates:
            tid = self._next_track_id
            self._next_track_id += 1
            self._tracks[tid] = TrackState(
                track_id=tid,
                face_bbox=cand.face_bbox,
                torso_bbox=cand.torso_bbox,
                centroid=_bbox_center(cand.face_bbox),
                area=max(1e-5, cand.face_bbox[2] * cand.face_bbox[3]),
                wrist_anchors=cand.wrist_anchors,
                wrist_search_radius=cand.wrist_search_radius,
                last_seen_ts=ts,
                frames_seen=1,
                face_sim=cand.face_sim,
                pose_backed=cand.pose_backed,
            )

        # Expire stale tracks past grace period
        stale_ids = [
            tid for tid, tr in self._tracks.items() if (ts - tr.last_seen_ts) > self.teacher_grace_sec
        ]
        for tid in stale_ids:
            self._tracks.pop(tid, None)
            if self._locked_track_id == tid:
                self._locked_track_id = None

        # Check explicit Face-ID match hijack protection or initial teacher acquisition
        active_now = [tr for tr in self._tracks.values() if abs(tr.last_seen_ts - ts) < 1e-4]
        if self._locked_track_id is not None and self._locked_track_id not in self._tracks:
            self._locked_track_id = None

        if self._locked_track_id is None and active_now:
            # Prefer verified ArcFace match first; otherwise central + largest face
            verified = [tr for tr in active_now if tr.face_sim >= self.face_sim_threshold]
            pool = verified if verified else active_now
            chosen = max(
                pool,
                key=lambda tr: tr.area * (1.15 - 0.5 * abs(tr.centroid[0] - 0.5))
                + (1.0 if tr.face_sim >= self.face_sim_threshold else 0.0),
            )
            self._locked_track_id = chosen.track_id

        if not self.enabled:
            self._last_snapshot = SubjectTrackingSnapshot(
                pose_landmarker_active=any(tr.pose_backed for tr in active_now),
                teacher_lock_enabled=False,
                bypass_active=True,
                person_count=len(active_now),
                mode="BYPASS (OFF)",
            )
            return self._last_snapshot

        locked_tr = self._tracks.get(self._locked_track_id) if self._locked_track_id is not None else None
        if locked_tr is None:
            self._last_snapshot = SubjectTrackingSnapshot(
                pose_landmarker_active=any(tr.pose_backed for tr in active_now),
                teacher_lock_enabled=True,
                bypass_active=True,
                person_count=len(active_now),
                mode="BYPASS (UNLOCKED)",
            )
            return self._last_snapshot

        in_grace = abs(locked_tr.last_seen_ts - ts) > 1e-4
        ignored_faces = [tr.face_bbox for tr in active_now if tr.track_id != locked_tr.track_id]
        ignored_torsos = [tr.torso_bbox for tr in active_now if tr.track_id != locked_tr.track_id]
        multi_person_active = len(ignored_faces) >= 1
        bypass_active = not multi_person_active
        if in_grace:
            mode = "GRACE_HOLD"
        elif multi_person_active:
            mode = "POSE+FACE_LOCK" if locked_tr.pose_backed else "FACE_TORSO_LOCK"
        else:
            mode = "BYPASS (1-NGƯỜI)"

        self._last_snapshot = SubjectTrackingSnapshot(
            locked_teacher_id=locked_tr.track_id,
            locked_face_bbox=locked_tr.face_bbox,
            locked_torso_bbox=locked_tr.torso_bbox,
            locked_wrist_anchors=list(locked_tr.wrist_anchors),
            wrist_search_radius=locked_tr.wrist_search_radius,
            ignored_face_bboxes=ignored_faces,
            ignored_torso_bboxes=ignored_torsos,
            teacher_in_grace_period=in_grace,
            pose_landmarker_active=locked_tr.pose_backed,
            teacher_lock_enabled=True,
            bypass_active=bypass_active,
            person_count=len(active_now),
            mode=mode,
        )
        return self._last_snapshot

    def get_proactive_wrist_roi_boxes(
        self,
        snap: Optional[SubjectTrackingSnapshot] = None,
        roi_expand_factor: float = 2.2,
    ) -> List[List[float]]:
        """
        Return normalized `[x, y, w, h]` Stage-2 search ROI boxes around locked Teacher's wrists
        when distant full-frame hand detection loses the small hand.
        """
        s = snap or self._last_snapshot
        if not s.locked_wrist_anchors:
            return []
        half = max(0.08, min(0.28, s.wrist_search_radius * (roi_expand_factor * 0.55)))
        rois: List[List[float]] = []
        for wx, wy in s.locked_wrist_anchors:
            x0 = max(0.0, float(wx) - half)
            y0 = max(0.0, float(wy) - half)
            x1 = min(1.0, float(wx) + half)
            y1 = min(1.0, float(wy) + half)
            rois.append([round(x0, 4), round(y0, 4), round(x1 - x0, 4), round(y1 - y0, 4)])
        return rois

    def is_hand_owned_by_locked_teacher(
        self,
        hand_lm: Sequence[Sequence[float]],
        snap: Optional[SubjectTrackingSnapshot] = None,
    ) -> bool:
        """
        Fail-Open Wrist-to-Hand Spatial Gating:
        - If Teacher Lock is disabled, unlocked, or <= 1 person is present -> BYPASS (returns True!).
        - Only filters hands when >= 2 people are confirmed (`len(ignored_face_bboxes) >= 1` or grace hold
          with background person), rejecting hands inside an ignored passerby's torso/face while keeping
          the locked teacher's hands even if raised high or close to camera.
        """
        if not hand_lm or len(hand_lm) < 10:
            return False
        if not self.enabled:
            return True

        s = snap or self._last_snapshot
        if not s.teacher_lock_enabled:
            return True

        # Fail-Open Bypass: never reject hands when no teacher is locked or only <= 1 person in scene
        if s.locked_teacher_id is None or s.locked_torso_bbox is None:
            return True
        if len(s.ignored_face_bboxes) == 0 and len(s.ignored_torso_bboxes) == 0:
            return True

        wrist = (float(hand_lm[0][0]), float(hand_lm[0][1]))
        palm_center = (
            0.5 * (wrist[0] + float(hand_lm[9][0])),
            0.5 * (wrist[1] + float(hand_lm[9][1])),
        )

        # Reject face-interior false positives ONLY against ignored background people
        for fb in s.ignored_face_bboxes:
            fx, fy, fw, fh = fb
            if _point_in_expanded_box(palm_center, fb, pad_x=-fw * 0.12, pad_y=-fh * 0.12):
                return False

        in_locked_torso = _point_in_expanded_box(
            wrist, s.locked_torso_bbox, pad_x=0.14, pad_y=0.16
        ) or _point_in_expanded_box(
            palm_center, s.locked_torso_bbox, pad_x=0.14, pad_y=0.16
        )
        near_locked_wrist = any(
            _point_dist(wrist, anc) <= (s.wrist_search_radius * 1.6)
            for anc in s.locked_wrist_anchors
        )
        locked_center = _bbox_center(s.locked_face_bbox or s.locked_torso_bbox)
        dist_locked = _point_dist(wrist, locked_center)

        # Reject hands clearly belonging to an ignored passerby
        for ig_idx, ig_torso in enumerate(s.ignored_torso_bboxes):
            ig_face = (
                s.ignored_face_bboxes[ig_idx]
                if ig_idx < len(s.ignored_face_bboxes)
                else ig_torso
            )
            ig_center = _bbox_center(ig_face)
            dist_ignored = _point_dist(wrist, ig_center)
            inside_ignored = _point_in_expanded_box(wrist, ig_torso, pad_x=0.05, pad_y=0.05)
            if inside_ignored and dist_ignored < (dist_locked * 0.92):
                return False
            if dist_ignored < (dist_locked * 0.78) and not (near_locked_wrist and dist_locked < dist_ignored * 1.15):
                return False

        # Fail-open for locked teacher: if hand is not owned by any ignored person, allow it
        # unless it is in grace hold without teacher visible and outside a generous reach margin.
        if in_locked_torso or near_locked_wrist:
            return True
        return not s.teacher_in_grace_period or dist_locked <= 0.55
