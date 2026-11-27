#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AirTeach AI — Asset Downloader & Verifier (`scripts/fetch_assets.py`)
Automatically downloads, verifies (size + magic bytes + SHA-256), and caches:
  1. MediaPipe `hand_landmarker.task` -> `assets/models/hand_landmarker.task` & `models/hand_landmarker.task`
  2. MediaPipe `blaze_face_short_range.tflite` -> `models/blaze_face_short_range.tflite`
  3. Vietnamese Unicode fonts (`NotoSans-Regular.ttf`, `NotoSans-Bold.ttf`) -> `assets/fonts/`
  4. Multi-size Windows application icon (`airteach.ico`) -> `assets/icons/airteach.ico` (and `assets/logo.ico`)
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import struct
import sys
import time
from typing import Iterable, Optional, Sequence, Tuple
import urllib.error
import urllib.request


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AssetSpec:
    name: str
    rel_path: str
    urls: Tuple[str, ...]
    min_bytes: int
    expected_sha256: Optional[str]
    magic_prefixes: Tuple[bytes, ...]
    mirror_rel_paths: Tuple[str, ...] = ()


ASSETS: Tuple[AssetSpec, ...] = (
    AssetSpec(
        name="MediaPipe Hand Landmarker",
        rel_path="assets/models/hand_landmarker.task",
        urls=(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        ),
        min_bytes=7_000_000,
        expected_sha256="fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
        magic_prefixes=(b"PK", b"TFL3"),
        mirror_rel_paths=("models/hand_landmarker.task",),
    ),
    AssetSpec(
        name="MediaPipe Face Detector",
        rel_path="models/blaze_face_short_range.tflite",
        urls=(
            "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        ),
        min_bytes=150_000,
        expected_sha256="b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f",
        magic_prefixes=(),
        mirror_rel_paths=(),
    ),
    AssetSpec(
        name="Vietnamese Font NotoSans Regular",
        rel_path="assets/fonts/NotoSans-Regular.ttf",
        urls=(
            "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf",
            "https://cdn.jsdelivr.net/gh/notofonts/notofonts.github.io@main/fonts/NotoSans/hinted/ttf/NotoSans-Regular.ttf",
        ),
        min_bytes=200_000,
        expected_sha256=None,
        magic_prefixes=(b"\x00\x01\x00\x00", b"OTTO"),
    ),
    AssetSpec(
        name="Vietnamese Font NotoSans Bold",
        rel_path="assets/fonts/NotoSans-Bold.ttf",
        urls=(
            "https://github.com/notofonts/notofonts.github.io/raw/main/fonts/NotoSans/hinted/ttf/NotoSans-Bold.ttf",
            "https://cdn.jsdelivr.net/gh/notofonts/notofonts.github.io@main/fonts/NotoSans/hinted/ttf/NotoSans-Bold.ttf",
        ),
        min_bytes=200_000,
        expected_sha256=None,
        magic_prefixes=(b"\x00\x01\x00\x00", b"OTTO"),
    ),
)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_valid_asset(path: Path, spec: AssetSpec) -> Tuple[bool, str]:
    if not path.exists() or not path.is_file():
        return False, "không tồn tại"
    size = path.stat().st_size
    if size < spec.min_bytes:
        return False, f"dung lượng quá nhỏ ({size} < {spec.min_bytes} bytes)"
    if spec.magic_prefixes:
        try:
            with path.open("rb") as fh:
                head = fh.read(32)
            if not any(head.startswith(p) or p in head[:16] for p in spec.magic_prefixes):
                return False, "sai định dạng header tệp"
        except OSError as exc:
            return False, f"lỗi đọc header: {exc}"
    digest = compute_sha256(path)
    if spec.expected_sha256 and digest.lower() != spec.expected_sha256.lower():
        return False, f"sai SHA-256 ({digest[:12]} != {spec.expected_sha256[:12]})"
    return True, digest


def download_file_atomic(url: str, dest: Path, timeout_sec: int = 45) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AirTeachAI-Setup/2.0 (Windows)",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        with tmp_path.open("wb") as out:
            while True:
                buf = resp.read(256 * 1024)
                if not buf:
                    break
                out.write(buf)
    os.replace(tmp_path, dest)


def sync_mirrors(primary: Path, mirrors: Iterable[Path]) -> None:
    for m in mirrors:
        if m.resolve() == primary.resolve():
            continue
        if m.exists() and m.stat().st_size == primary.stat().st_size:
            if compute_sha256(m) == compute_sha256(primary):
                continue
        m.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primary, m)


def fetch_single_asset(spec: AssetSpec, root: Path) -> bool:
    target = root / spec.rel_path
    mirrors = [root / m for m in spec.mirror_rel_paths]

    ok, info = is_valid_asset(target, spec)
    if ok:
        sync_mirrors(target, mirrors)
        print(f"  [ĐÃ CÓ CACHE] {spec.rel_path} ({target.stat().st_size:,} bytes | SHA-256: {info[:12]}...)")
        return True

    for cand in mirrors:
        c_ok, c_info = is_valid_asset(cand, spec)
        if c_ok:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cand, target)
            sync_mirrors(target, mirrors)
            print(f"  [ĐỒNG BỘ CACHE] {spec.rel_path} <- {cand.relative_to(root)} ({target.stat().st_size:,} bytes | SHA-256: {c_info[:12]}...)")
            return True

    last_err = "Không rõ lỗi"
    for attempt in range(1, 4):
        for url in spec.urls:
            print(f"  [TẢI {attempt}/3] {spec.name} <- {url}")
            try:
                download_file_atomic(url, target)
                v_ok, v_info = is_valid_asset(target, spec)
                if v_ok:
                    sync_mirrors(target, mirrors)
                    print(f"  [OK] {spec.rel_path} ({target.stat().st_size:,} bytes | SHA-256: {v_info[:12]}...)")
                    return True
                last_err = v_info
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_err = str(exc)
        time.sleep(min(4, attempt * 1.5))

    print(f"  [LỖI] Không thể tải {spec.rel_path}: {last_err}", file=sys.stderr)
    return False


def is_valid_ico(path: Path) -> Tuple[bool, str]:
    if not path.exists() or not path.is_file():
        return False, "không tồn tại"
    data = path.read_bytes()
    if len(data) < 256:
        return False, f"kích thước quá nhỏ ({len(data)} bytes)"
    reserved, img_type, count = struct.unpack("<HHH", data[:6])
    if reserved != 0 or img_type != 1 or count < 3:
        return False, f"header ICO không hợp lệ (type={img_type}, count={count})"
    sizes = [
        ((data[6 + 16 * i] or 256), (data[6 + 16 * i + 1] or 256))
        for i in range(count)
    ]
    return True, f"{count} kích thước {sizes}"


def generate_fallback_ico(dest: Path) -> bool:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print("  [CẢNH BÁO] Chưa có Pillow để tự tạo icon dự phòng.", file=sys.stderr)
        return False

    size = 256
    img = Image.new("RGBA", (size, size), (15, 23, 42, 0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((8, 8, 248, 248), radius=52, fill=(15, 23, 42, 255), outline=(79, 70, 229, 255), width=10)
    draw.polygon([(128, 44), (66, 206), (96, 206), (128, 114), (160, 206), (190, 206)], fill=(0, 245, 212, 255))
    draw.rectangle((92, 146, 164, 168), fill=(6, 182, 212, 255))
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, format="ICO", sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return dest.exists()


def verify_and_sync_icons(root: Path) -> bool:
    icon_primary = root / "assets" / "icons" / "airteach.ico"
    icon_legacy = root / "assets" / "logo.ico"

    p_ok, p_info = is_valid_ico(icon_primary)
    l_ok, l_info = is_valid_ico(icon_legacy)

    if p_ok:
        if not l_ok:
            icon_legacy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(icon_primary, icon_legacy)
        print(f"  [ĐÃ CÓ CACHE] assets/icons/airteach.ico ({icon_primary.stat().st_size:,} bytes | {p_info})")
        return True

    if l_ok:
        icon_primary.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(icon_legacy, icon_primary)
        print(f"  [ĐỒNG BỘ CACHE] assets/icons/airteach.ico <- assets/logo.ico ({icon_primary.stat().st_size:,} bytes | {l_info})")
        return True

    if generate_fallback_ico(icon_primary):
        shutil.copy2(icon_primary, icon_legacy)
        ok, info = is_valid_ico(icon_primary)
        if ok:
            print(f"  [TẠO MỚI] assets/icons/airteach.ico ({icon_primary.stat().st_size:,} bytes | {info})")
            return True

    print("  [LỖI] Không tìm thấy hoặc không thể tạo assets/icons/airteach.ico!", file=sys.stderr)
    return False


def main(root: Optional[Path] = None) -> int:
    base = root or ROOT_DIR
    print("=== [AirTeach AI] Kiểm tra & Tải Tài nguyên AI, Font Tiếng Việt & Icon ===")
    all_ok = True
    for spec in ASSETS:
        if not fetch_single_asset(spec, base):
            all_ok = False
    if not verify_and_sync_icons(base):
        all_ok = False

    if all_ok:
        print("=== [OK] Toàn bộ tài nguyên AI, Font Tiếng Việt và Icon đã sẵn sàng! ===")
        return 0
    print("=== [THẤT BẠI] Một số tài nguyên chưa sẵn sàng. Kiểm tra kết nối mạng và thử lại. ===", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
