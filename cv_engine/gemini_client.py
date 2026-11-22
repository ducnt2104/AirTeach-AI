#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resilient Gemini API Client & Multi-Layer API Key Manager (`cv_engine/gemini_client.py`)
for AirTeach AI (DNT - Victer).

Implements:
1. Multi-layer API Key priority:
   `(1) User-entered UI Settings key` -> `(2) GEMINI_API_KEY / GOOGLE_API_KEY in .env / OS env`.
   Stores UI key locally in `data/api_settings.json` (git-ignored under `/data/`), NEVER hardcoded.
2. Key masking (`AIza••••••xxxx`), show/hide support, and safe key removal.
3. Live Health Check (`test_connection`) hitting the real Gemini `generateContent` endpoint with
   minimal payload (`maxOutputTokens=5`) and returning structured 3-state status:
   `CONNECTED` (green) | `UNCONFIGURED` (gray) | `ERROR` (red).
4. Resilient error handling & Vietnamese user messages:
   - `400 / 401 / 403`: Invalid key or missing Gemini API access -> fail fast with clear Vietnamese message.
   - `404`: Model not found -> automatic fallback across `model_chain`.
   - `429`: Rate Limit / Quota -> automatic Retry with Exponential Backoff (max 3 retries: 1s, 2s, 4s),
     then fallback to secondary model before surfacing friendly Vietnamese error.
   - `Network / Timeout`: Offline-resilient fallback preserving all local CV & drawing features.
5. Vision snapshot compression (`max_dim <= 1024px`) for fast multimodal latency and token efficiency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from google import genai
    from google.genai import types as genai_types

    HAS_GENAI_SDK = True
except Exception:
    genai = None  # type: ignore[assignment]
    genai_types = None  # type: ignore[assignment]
    HAS_GENAI_SDK = False

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODELS: Tuple[str, ...] = ("gemini-3.5-flash", "gemini-3.5-flash-lite")
EXPONENTIAL_BACKOFF_DELAYS_SEC: Tuple[float, ...] = (1.0, 2.0, 4.0)
MAX_VISION_IMAGE_DIM = 1024
PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here", "YOUR_GEMINI_API_KEY"}
DEFAULT_SETTINGS_PATH = Path("data") / "api_settings.json"


class ApiConnectionStatus(str, Enum):
    UNCONFIGURED = "UNCONFIGURED"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"

    @property
    def label_vi(self) -> str:
        return {
            ApiConnectionStatus.UNCONFIGURED: "Chưa cấu hình",
            ApiConnectionStatus.CONNECTED: "Đã kết nối",
            ApiConnectionStatus.ERROR: "Lỗi Key/Quota",
        }[self]


@dataclass
class ResolvedApiKey:
    key: str
    source: str  # "ui" | "env" | "none"
    masked: str
    ui_key_present: bool
    env_key_present: bool


@dataclass
class GeminiHealthCheckResult:
    status: ApiConnectionStatus
    model_used: str
    message_vi: str
    latency_ms: float = 0.0
    http_status: Optional[int] = None


@dataclass
class GeminiCallResult:
    ok: bool
    text: str
    model_used: str
    error_code: Optional[int] = None
    error_message_vi: str = ""
    from_cache: bool = False
    attempts: int = 0


def mask_api_key(api_key: str) -> str:
    """Format an API key safely as `AIza••••••xxxx` for UI display."""
    cleaned = (api_key or "").strip()
    if not cleaned:
        return ""
    if len(cleaned) <= 8:
        return "•" * len(cleaned)
    return f"{cleaned[:4]}••••••{cleaned[-4:]}"


def _clean_env_value(raw: str) -> str:
    val = raw.strip().strip('"').strip("'")
    if val in PLACEHOLDER_KEYS:
        return ""
    return val


def read_dotenv_values(env_path: Optional[Path] = None) -> Dict[str, str]:
    """Read key=value pairs from `.env` cleanly without mutating OS env unnecessarily."""
    candidates = [env_path] if env_path is not None else [Path(".env"), Path(__file__).resolve().parent.parent / ".env"]
    out: Dict[str, str] = {}
    for p in candidates:
        if p is None or not p.exists():
            continue
        try:
            for raw_line in p.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                key = k.strip()
                val = _clean_env_value(v)
                if key and key not in out:
                    out[key] = val
        except OSError:
            continue
    return out


def load_ui_api_settings(settings_path: Path = DEFAULT_SETTINGS_PATH) -> Dict[str, str]:
    if not settings_path.exists():
        return {}
    try:
        data = json.loads(settings_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items() if v is not None}
    except Exception:
        pass
    return {}


def save_ui_api_settings(
    ui_api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
    settings_path: Path = DEFAULT_SETTINGS_PATH,
) -> Dict[str, str]:
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    data = load_ui_api_settings(settings_path)
    if ui_api_key is not None:
        cleaned = _clean_env_value(ui_api_key)
        if cleaned:
            data["ui_gemini_api_key"] = cleaned
        else:
            data.pop("ui_gemini_api_key", None)
    if preferred_model is not None:
        m = preferred_model.strip()
        if m:
            data["gemini_model_id"] = m
        else:
            data.pop("gemini_model_id", None)
    settings_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def clear_ui_api_key(settings_path: Path = DEFAULT_SETTINGS_PATH) -> None:
    save_ui_api_settings(ui_api_key="", settings_path=settings_path)


def resolve_api_key_layered(
    ui_key_override: Optional[str] = None,
    settings_path: Path = DEFAULT_SETTINGS_PATH,
    env_path: Optional[Path] = None,
) -> ResolvedApiKey:
    """
    Resolve Gemini API key strictly in priority order:
    (1) Explicit UI key (`ui_key_override` or persisted in `data/api_settings.json`)
    (2) `GEMINI_API_KEY` / `GOOGLE_API_KEY` from OS environment or `.env`
    """
    ui_settings = load_ui_api_settings(settings_path)
    ui_key = _clean_env_value(
        ui_key_override if ui_key_override is not None else ui_settings.get("ui_gemini_api_key", "")
    )

    dotenv = read_dotenv_values(env_path)
    env_key = _clean_env_value(
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or dotenv.get("GEMINI_API_KEY")
        or dotenv.get("GOOGLE_API_KEY")
        or ""
    )

    if ui_key:
        return ResolvedApiKey(
            key=ui_key,
            source="ui",
            masked=mask_api_key(ui_key),
            ui_key_present=True,
            env_key_present=bool(env_key),
        )
    if env_key:
        return ResolvedApiKey(
            key=env_key,
            source="env",
            masked=mask_api_key(env_key),
            ui_key_present=False,
            env_key_present=True,
        )
    return ResolvedApiKey(
        key="",
        source="none",
        masked="",
        ui_key_present=False,
        env_key_present=False,
    )


def build_model_chain(
    primary_model: Optional[str] = None,
    fallback_models: Optional[Sequence[str]] = None,
    env_path: Optional[Path] = None,
    settings_path: Path = DEFAULT_SETTINGS_PATH,
) -> List[str]:
    """Build deduplicated ordered list of Gemini models `[primary, fallback1, fallback2, ...]`."""
    dotenv = read_dotenv_values(env_path)
    ui_settings = load_ui_api_settings(settings_path)

    primary = (
        (primary_model or "").strip()
        or ui_settings.get("gemini_model_id", "").strip()
        or os.environ.get("GEMINI_MODEL_ID", "").strip()
        or dotenv.get("GEMINI_MODEL_ID", "").strip()
        or DEFAULT_GEMINI_MODEL
    )

    fb_items: List[str] = []
    if fallback_models is not None:
        fb_items.extend(str(m).strip() for m in fallback_models)
    else:
        env_fb = (
            os.environ.get("GEMINI_FALLBACK_MODELS", "").strip()
            or dotenv.get("GEMINI_FALLBACK_MODELS", "").strip()
        )
        if env_fb:
            fb_items.extend(x.strip() for x in env_fb.split(","))
        fb_items.extend(DEFAULT_FALLBACK_MODELS)
        fb_items.append(DEFAULT_GEMINI_MODEL)

    chain: List[str] = []
    for m in [primary, *fb_items]:
        if m and m not in chain:
            chain.append(m)
    return chain


def compress_snapshot_png(png_bytes: Optional[bytes], max_dim: int = MAX_VISION_IMAGE_DIM) -> Optional[bytes]:
    """
    Ensure PNG snapshot sent to Gemini Vision is bounded to `max_dim` (default 1024px)
    on its longest side while preserving aspect ratio and crisp stroke contrast.
    """
    if not png_bytes:
        return png_bytes
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None or img.size == 0:
        return png_bytes
    h, w = img.shape[:2]
    longest = max(w, h)
    if longest <= max_dim:
        return png_bytes
    scale = float(max_dim) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".png", resized, [int(cv2.IMWRITE_PNG_COMPRESSION), 6])
    if ok:
        return encoded.tobytes()
    return png_bytes


def classify_gemini_error(exc: Exception) -> Tuple[Optional[int], str, str]:
    """
    Classify an exception from `google-genai` / HTTP into `(status_code, category, message_vi)`.
    Categories: `AUTH`, `NOT_FOUND`, `RATE_LIMIT`, `NETWORK`, `SERVER`, `UNKNOWN`.
    """
    status_code: Optional[int] = None
    for attr in ("code", "status_code", "http_status"):
        val = getattr(exc, attr, None)
        if isinstance(val, int) and 100 <= val < 600:
            status_code = val
            break

    raw = str(exc or "")
    low = raw.lower()

    if status_code is None:
        m = re.search(r"\b(400|401|403|404|429|500|502|503|504)\b", raw)
        if m:
            status_code = int(m.group(1))

    if (
        status_code in (401, 403)
        or "api_key_invalid" in low
        or "api key not valid" in low
        or "permission_denied" in low
        or "unauthorized" in low
        or (status_code == 400 and ("api key" in low or "key" in low))
    ):
        return (
            status_code or 401,
            "AUTH",
            "API Key không hợp lệ, đã hết hạn hoặc chưa bật quyền truy cập Gemini API.",
        )

    if status_code == 400:
        return (
            400,
            "BAD_REQUEST",
            "Yêu cầu gửi tới Gemini không hợp lệ (HTTP 400). Vui lòng kiểm tra dữ liệu hoặc API Key.",
        )

    if status_code == 404 or "not_found" in low or "is not found for api version" in low:
        return (
            404,
            "NOT_FOUND",
            "Mô hình Gemini không tồn tại hoặc chưa hỗ trợ phương thức này (HTTP 404).",
        )

    if status_code == 429 or "resource_exhausted" in low or "rate limit" in low or "quota" in low:
        return (
            429,
            "RATE_LIMIT",
            "Vượt giới hạn lượt gọi / hết Quota Gemini API (HTTP 429). Vui lòng thử lại sau ít phút.",
        )

    if (
        isinstance(exc, (TimeoutError, ConnectionError, OSError))
        or "timed out" in low
        or "timeout" in low
        or "connection" in low
        or "name resolution" in low
        or "network" in low
        or "dns" in low
        or "ssl" in low
    ):
        return (
            status_code,
            "NETWORK",
            "Mất kết nối mạng hoặc hết thời gian chờ máy chủ Gemini. Ứng dụng vẫn vẽ và nhận diện cử chỉ bình thường ở chế độ Ngoại tuyến.",
        )

    if status_code in (500, 502, 503, 504) or "unavailable" in low or "internal" in low:
        return (
            status_code or 503,
            "SERVER",
            f"Máy chủ Gemini tạm thời quá tải hoặc gián đoạn (HTTP {status_code or 503}).",
        )

    return (
        status_code,
        "UNKNOWN",
        f"Lỗi kết nối Gemini API: {raw[:220]}",
    )


class ResilientGeminiClient:
    """
    Fault-tolerant Gemini API wrapper with:
    - Multi-layer key resolution (`UI Settings` > `.env`)
    - Live Health Check (`test_connection`)
    - Exponential backoff retry on HTTP 429 (`1s, 2s, 4s`)
    - Automatic fallback model chain on HTTP 404 or exhausted 429
    - Snapshot resizing (`<= 1024px`)
    """

    def __init__(
        self,
        client_factory: Optional[Callable[[str], Any]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._client_factory = client_factory or self._default_client_factory
        self._sleep = sleep_fn or time.sleep

    @staticmethod
    def _default_client_factory(api_key: str) -> Any:
        if not HAS_GENAI_SDK or genai is None:
            raise RuntimeError("Chưa cài đặt thư viện google-genai. Hãy chạy: pip install google-genai")
        return genai.Client(api_key=api_key)

    def _invoke_generate(
        self,
        client: Any,
        model_id: str,
        contents: Any,
        max_output_tokens: Optional[int] = None,
        structured_json: bool = False,
    ) -> str:
        kwargs: Dict[str, Any] = {"model": model_id, "contents": contents}
        cfg_kwargs: Dict[str, Any] = {}
        if max_output_tokens is not None:
            cfg_kwargs["max_output_tokens"] = int(max_output_tokens)
        if structured_json:
            cfg_kwargs["response_mime_type"] = "application/json"

        if cfg_kwargs:
            if genai_types is not None and hasattr(genai_types, "GenerateContentConfig"):
                kwargs["config"] = genai_types.GenerateContentConfig(**cfg_kwargs)
            else:
                kwargs["config"] = cfg_kwargs

        resp = client.models.generate_content(**kwargs)
        text = getattr(resp, "text", None)
        if text is None and isinstance(resp, dict):
            text = resp.get("text")
        return (text or "").strip()

    def test_connection(
        self,
        api_key: Optional[str] = None,
        primary_model: Optional[str] = None,
        fallback_models: Optional[Sequence[str]] = None,
    ) -> GeminiHealthCheckResult:
        """
        Execute a real minimal `generateContent` health check (`max_output_tokens=5`)
        returning `CONNECTED`, `UNCONFIGURED`, or `ERROR`.
        """
        resolved_key = (api_key or "").strip()
        if not resolved_key:
            resolved_key = resolve_api_key_layered().key
        if not resolved_key:
            return GeminiHealthCheckResult(
                status=ApiConnectionStatus.UNCONFIGURED,
                model_used="",
                message_vi="Chưa cấu hình Gemini API Key (UI Settings hoặc .env).",
            )

        t0 = time.perf_counter()
        try:
            client = self._client_factory(resolved_key)
        except Exception as exc:
            code, _, msg_vi = classify_gemini_error(exc)
            return GeminiHealthCheckResult(
                status=ApiConnectionStatus.ERROR,
                model_used="",
                message_vi=msg_vi,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                http_status=code,
            )

        chain = build_model_chain(primary_model=primary_model, fallback_models=fallback_models)
        last_msg = "Không thể kết nối Gemini API."
        last_code: Optional[int] = None

        for model_id in chain:
            try:
                reply = self._invoke_generate(
                    client,
                    model_id=model_id,
                    contents="Reply OK",
                    max_output_tokens=5,
                    structured_json=False,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                note = f"Kết nối thành công tới {model_id}"
                if reply:
                    note = f"{note} (Phản hồi: \"{reply[:24]}\")"
                return GeminiHealthCheckResult(
                    status=ApiConnectionStatus.CONNECTED,
                    model_used=model_id,
                    message_vi=note,
                    latency_ms=latency_ms,
                    http_status=200,
                )
            except Exception as exc:
                code, cat, msg_vi = classify_gemini_error(exc)
                last_code = code
                last_msg = msg_vi
                if cat in ("AUTH", "BAD_REQUEST"):
                    return GeminiHealthCheckResult(
                        status=ApiConnectionStatus.ERROR,
                        model_used=model_id,
                        message_vi=msg_vi,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        http_status=code,
                    )
                if cat == "NOT_FOUND":
                    continue
                if cat in ("RATE_LIMIT", "NETWORK", "SERVER"):
                    return GeminiHealthCheckResult(
                        status=ApiConnectionStatus.ERROR,
                        model_used=model_id,
                        message_vi=msg_vi,
                        latency_ms=(time.perf_counter() - t0) * 1000.0,
                        http_status=code,
                    )

        return GeminiHealthCheckResult(
            status=ApiConnectionStatus.ERROR,
            model_used=chain[0] if chain else "",
            message_vi=last_msg,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            http_status=last_code,
        )

    def generate_multimodal(
        self,
        api_key: str,
        prompt: str,
        png_bytes: Optional[bytes] = None,
        primary_model: Optional[str] = None,
        fallback_models: Optional[Sequence[str]] = None,
        structured_json: bool = False,
    ) -> GeminiCallResult:
        """
        Execute multimodal Gemini request with:
        - Snapshot downscale to <= 1024px
        - Exponential Backoff retries on HTTP 429 (`1s, 2s, 4s`)
        - Model fallback on HTTP 404 or exhausted 429
        - Structured Vietnamese error translation on failure
        """
        clean_key = (api_key or "").strip()
        if not clean_key:
            return GeminiCallResult(
                ok=False,
                text="",
                model_used="",
                error_code=None,
                error_message_vi="Chưa cấu hình GEMINI_API_KEY trong UI Settings hoặc file .env.",
            )

        try:
            client = self._client_factory(clean_key)
        except Exception as exc:
            code, _, msg_vi = classify_gemini_error(exc)
            return GeminiCallResult(
                ok=False,
                text="",
                model_used="",
                error_code=code,
                error_message_vi=msg_vi,
            )

        compressed_png = compress_snapshot_png(png_bytes, max_dim=MAX_VISION_IMAGE_DIM)
        contents: List[Any] = [prompt]
        if compressed_png:
            if genai_types is not None and hasattr(genai_types, "Part"):
                contents.append(
                    genai_types.Part.from_bytes(data=compressed_png, mime_type="image/png")
                )
            else:
                contents.append({"mime_type": "image/png", "data": compressed_png})

        chain = build_model_chain(primary_model=primary_model, fallback_models=fallback_models)
        total_attempts = 0
        last_code: Optional[int] = None
        last_msg = "Không thể kết nối Gemini API."

        for model_id in chain:
            rate_limit_retries = 0
            transient_retries = 0
            while True:
                total_attempts += 1
                try:
                    txt = self._invoke_generate(
                        client,
                        model_id=model_id,
                        contents=contents,
                        structured_json=structured_json,
                    )
                    if not txt:
                        txt = "Không nhận được phản hồi văn bản từ Gemini."
                    return GeminiCallResult(
                        ok=True,
                        text=txt,
                        model_used=model_id,
                        attempts=total_attempts,
                    )
                except Exception as exc:
                    code, cat, msg_vi = classify_gemini_error(exc)
                    last_code = code
                    last_msg = msg_vi

                    if cat in ("AUTH", "BAD_REQUEST"):
                        return GeminiCallResult(
                            ok=False,
                            text="",
                            model_used=model_id,
                            error_code=code,
                            error_message_vi=msg_vi,
                            attempts=total_attempts,
                        )

                    if cat == "NOT_FOUND":
                        # Immediately fall back to next model in chain
                        break

                    if cat == "RATE_LIMIT":
                        if rate_limit_retries < len(EXPONENTIAL_BACKOFF_DELAYS_SEC):
                            delay = EXPONENTIAL_BACKOFF_DELAYS_SEC[rate_limit_retries]
                            rate_limit_retries += 1
                            self._sleep(delay)
                            continue
                        # Exhausted 3 retries (1s, 2s, 4s) on this model -> try fallback model
                        break

                    if cat == "SERVER":
                        if transient_retries < 2:
                            delay = EXPONENTIAL_BACKOFF_DELAYS_SEC[transient_retries]
                            transient_retries += 1
                            self._sleep(delay)
                            continue
                        break

                    # NETWORK or UNKNOWN
                    if cat == "NETWORK" and transient_retries < 1:
                        transient_retries += 1
                        self._sleep(1.0)
                        continue
                    return GeminiCallResult(
                        ok=False,
                        text="",
                        model_used=model_id,
                        error_code=code,
                        error_message_vi=msg_vi,
                        attempts=total_attempts,
                    )

        return GeminiCallResult(
            ok=False,
            text="",
            model_used=chain[-1] if chain else "",
            error_code=last_code,
            error_message_vi=last_msg,
            attempts=total_attempts,
        )
