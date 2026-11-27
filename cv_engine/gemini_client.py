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

from cv_engine.app_paths import get_app_logger, get_config_file_path, resource_path

_LOGGER = get_app_logger("gemini_client")

import importlib.util

HAS_GENAI_SDK = importlib.util.find_spec("google.genai") is not None
genai = None  # type: ignore[assignment]
genai_types = None  # type: ignore[assignment]

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODELS: Tuple[str, ...] = (
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)
EXPONENTIAL_BACKOFF_DELAYS_SEC: Tuple[float, ...] = (1.0, 2.0, 4.0)
MAX_VISION_IMAGE_DIM = 1024
DEFAULT_MAX_OUTPUT_TOKENS = 4096
PLACEHOLDER_KEYS = {"", "your_gemini_api_key_here", "YOUR_GEMINI_API_KEY"}
DEFAULT_SETTINGS_PATH = get_config_file_path()


def _ensure_genai_sdk() -> bool:
    """Lazy-load `google.genai` SDK when needed so UI startup remains < 1.5s."""
    global genai, genai_types, HAS_GENAI_SDK
    if genai is not None and genai_types is not None:
        return True
    if not HAS_GENAI_SDK:
        return False
    try:
        from google import genai as _genai
        from google.genai import types as _genai_types

        genai = _genai
        genai_types = _genai_types
        HAS_GENAI_SDK = True
        return True
    except Exception as exc:
        _LOGGER.warning("google-genai SDK import failed: %s", exc)
        HAS_GENAI_SDK = False
        return False


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

    @property
    def api_key(self) -> str:
        return self.key


@dataclass
class GeminiHealthCheckResult:
    status: ApiConnectionStatus
    model_used: str
    message_vi: str
    latency_ms: float = 0.0
    http_status: Optional[int] = None

    @property
    def ok(self) -> bool:
        return self.status == ApiConnectionStatus.CONNECTED


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
    candidates = [env_path] if env_path is not None else [Path(".env"), resource_path(".env"), Path(__file__).resolve().parent.parent / ".env"]
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
    except Exception as exc:
        _LOGGER.warning("Cannot read UI settings from %s: %s", settings_path, exc)
    return {}


def save_ui_api_settings(
    ui_api_key: Optional[str] = None,
    preferred_model: Optional[str] = None,
    settings_path: Path = DEFAULT_SETTINGS_PATH,
    onboarding_skipped: Optional[bool] = None,
    quick_tip_dismissed: Optional[bool] = None,
    ai_routing_mode: Optional[str] = None,
    active_gguf_path: Optional[str] = None,
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
    if onboarding_skipped is not None:
        data["onboarding_skipped"] = "true" if onboarding_skipped else "false"
    if quick_tip_dismissed is not None:
        data["quick_tip_dismissed"] = "true" if quick_tip_dismissed else "false"
    if ai_routing_mode is not None:
        rm = ai_routing_mode.strip().upper()
        if rm in ("LOCAL", "CLOUD", "HYBRID"):
            data["ai_routing_mode"] = rm
        else:
            data.pop("ai_routing_mode", None)
    if active_gguf_path is not None:
        gp = active_gguf_path.strip()
        if gp:
            data["active_gguf_path"] = gp
        else:
            data.pop("active_gguf_path", None)
    settings_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def clear_ui_api_key(settings_path: Path = DEFAULT_SETTINGS_PATH) -> None:
    save_ui_api_settings(ui_api_key="", settings_path=settings_path)


def split_api_keys(raw: str) -> List[str]:
    """Split comma/semicolon/newline-separated Gemini API keys into a deduplicated ordered list."""
    out: List[str] = []
    for part in re.split(r"[\s,;]+", raw or ""):
        cleaned = _clean_env_value(part)
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def resolve_api_key_pool(
    ui_key_override: Optional[str] = None,
    settings_path: Path = DEFAULT_SETTINGS_PATH,
    env_path: Optional[Path] = None,
) -> List[str]:
    """
    Resolve all configured Gemini API keys in priority order (UI keys first, then `.env`/OS env)
    for automatic rotation on HTTP 429/quota exhaustion or invalid secondary keys.
    """
    ui_settings = load_ui_api_settings(settings_path)
    dotenv = read_dotenv_values(env_path)
    candidates: List[str] = []
    raw_sources = (
        ui_key_override if ui_key_override is not None else ui_settings.get("ui_gemini_api_key", ""),
        os.environ.get("GEMINI_API_KEYS", ""),
        os.environ.get("GEMINI_API_KEY", ""),
        os.environ.get("GOOGLE_API_KEY", ""),
        dotenv.get("GEMINI_API_KEYS", ""),
        dotenv.get("GEMINI_API_KEY", ""),
        dotenv.get("GOOGLE_API_KEY", ""),
    )
    for raw in raw_sources:
        for k in split_api_keys(raw):
            if k not in candidates:
                candidates.append(k)
    return candidates


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
    ui_keys = split_api_keys(
        ui_key_override if ui_key_override is not None else ui_settings.get("ui_gemini_api_key", "")
    )
    ui_key = ui_keys[0] if ui_keys else ""

    dotenv = read_dotenv_values(env_path)
    env_keys = split_api_keys(
        os.environ.get("GEMINI_API_KEYS")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or dotenv.get("GEMINI_API_KEYS")
        or dotenv.get("GEMINI_API_KEY")
        or dotenv.get("GOOGLE_API_KEY")
        or ""
    )
    env_key = env_keys[0] if env_keys else ""

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
        self._key_cooldowns: Dict[str, float] = {}

    def _ordered_healthy_keys(self, api_keys: Sequence[str] | str) -> List[str]:
        raw_list = split_api_keys(api_keys) if isinstance(api_keys, str) else [k.strip() for k in api_keys if str(k).strip()]
        dedup: List[str] = []
        for k in raw_list:
            if k not in dedup:
                dedup.append(k)
        now = time.monotonic()
        healthy = [k for k in dedup if self._key_cooldowns.get(k, 0.0) <= now]
        cooling = [k for k in dedup if self._key_cooldowns.get(k, 0.0) > now]
        return healthy + cooling

    @staticmethod
    def _default_client_factory(api_key: str) -> Any:
        if not _ensure_genai_sdk() or genai is None:
            raise RuntimeError("Chưa cài đặt thư viện google-genai. Hãy chạy: pip install google-genai")
        return genai.Client(api_key=api_key)

    def _invoke_generate(
        self,
        client: Any,
        model_id: str,
        contents: Any,
        max_output_tokens: Optional[int] = DEFAULT_MAX_OUTPUT_TOKENS,
        structured_json: bool = False,
        system_instruction: Optional[str] = None,
    ) -> str:
        kwargs: Dict[str, Any] = {"model": model_id, "contents": contents}
        cfg_kwargs: Dict[str, Any] = {}
        if max_output_tokens is not None:
            cfg_kwargs["max_output_tokens"] = int(max_output_tokens)
        if structured_json:
            cfg_kwargs["response_mime_type"] = "application/json"
        if system_instruction and system_instruction.strip():
            cfg_kwargs["system_instruction"] = system_instruction.strip()

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
        api_key: str | Sequence[str],
        prompt: str,
        png_bytes: Optional[bytes] = None,
        primary_model: Optional[str] = None,
        fallback_models: Optional[Sequence[str]] = None,
        structured_json: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        system_instruction: Optional[str] = None,
        history: Sequence[Dict[str, str]] = (),
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> GeminiCallResult:
        """
        Execute multimodal Gemini request with:
        - Snapshot downscale to <= 1024px
        - Multi-key rotation when multiple keys are configured
        - Exponential Backoff retries on HTTP 429 (`1s, 2s, 4s`)
        - Model fallback on HTTP 404 or exhausted 429
        - Deep pedagogical system_instruction, history context & max_output_tokens >= 4096
        - Optional live token streaming via `on_token`
        - Structured Vietnamese error translation on failure
        """
        keys = self._ordered_healthy_keys(api_key)
        if not keys:
            return GeminiCallResult(
                ok=False,
                text="",
                model_used="",
                error_code=None,
                error_message_vi="Chưa cấu hình GEMINI_API_KEY trong UI Settings hoặc file .env.",
            )

        effective_max_tokens = max(DEFAULT_MAX_OUTPUT_TOKENS, int(max_output_tokens))
        effective_prompt = prompt
        clean_hist = [
            h for h in (history or ())
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and str(h.get("content") or "").strip()
        ]
        if clean_hist:
            hist_lines = []
            for turn in clean_hist[-6:]:
                who = "Giáo viên" if turn["role"] == "user" else "Trợ lý AI"
                hist_lines.append(f"{who}: {str(turn['content']).strip()[:600]}")
            effective_prompt = (
                "Ngữ cảnh hội thoại gần đây:\n"
                + "\n".join(hist_lines)
                + f"\n\nYêu cầu hiện tại của giáo viên:\n{prompt}"
            )

        compressed_png = compress_snapshot_png(png_bytes, max_dim=MAX_VISION_IMAGE_DIM)
        contents: List[Any] = [effective_prompt]
        if compressed_png:
            _ensure_genai_sdk()
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
        has_multi_keys = len(keys) > 1

        for active_key in keys:
            try:
                client = self._client_factory(active_key)
            except Exception as exc:
                code, _, msg_vi = classify_gemini_error(exc)
                last_code, last_msg = code, msg_vi
                if has_multi_keys:
                    continue
                return GeminiCallResult(
                    ok=False,
                    text="",
                    model_used="",
                    error_code=code,
                    error_message_vi=msg_vi,
                )

            rotate_to_next_key = False
            for model_id in chain:
                rate_limit_retries = 0
                transient_retries = 0
                while True:
                    total_attempts += 1
                    try:
                        stream_fn = getattr(getattr(client, "models", None), "generate_content_stream", None)
                        if on_token is not None and not structured_json and callable(stream_fn):
                            _ensure_genai_sdk()
                            cfg_kwargs: Dict[str, Any] = {
                                "max_output_tokens": effective_max_tokens,
                            }
                            if system_instruction and system_instruction.strip():
                                cfg_kwargs["system_instruction"] = system_instruction.strip()
                            if genai_types is not None and hasattr(genai_types, "GenerateContentConfig"):
                                stream_cfg: Any = genai_types.GenerateContentConfig(**cfg_kwargs)
                            else:
                                stream_cfg = cfg_kwargs
                            stream_iter = stream_fn(
                                model=model_id,
                                contents=contents,
                                config=stream_cfg,
                            )
                            chunks: List[str] = []
                            for ev in stream_iter:
                                piece = getattr(ev, "text", None) or ""
                                if piece:
                                    chunks.append(piece)
                                    on_token(piece)
                            txt = "".join(chunks).strip()
                        else:
                            txt = self._invoke_generate(
                                client,
                                model_id=model_id,
                                contents=contents,
                                max_output_tokens=effective_max_tokens,
                                structured_json=structured_json,
                                system_instruction=system_instruction,
                            )
                            if on_token is not None and txt and not structured_json:
                                on_token(txt)
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
                            if has_multi_keys and cat == "AUTH":
                                self._key_cooldowns[active_key] = time.monotonic() + 300.0
                                rotate_to_next_key = True
                                break
                            return GeminiCallResult(
                                ok=False,
                                text="",
                                model_used=model_id,
                                error_code=code,
                                error_message_vi=msg_vi,
                                attempts=total_attempts,
                            )

                        if cat == "NOT_FOUND":
                            break

                        if cat == "RATE_LIMIT":
                            if has_multi_keys:
                                self._key_cooldowns[active_key] = time.monotonic() + 45.0
                                rotate_to_next_key = True
                                break
                            if rate_limit_retries < len(EXPONENTIAL_BACKOFF_DELAYS_SEC):
                                delay = EXPONENTIAL_BACKOFF_DELAYS_SEC[rate_limit_retries]
                                rate_limit_retries += 1
                                self._sleep(delay)
                                continue
                            break

                        if cat == "SERVER":
                            if transient_retries < 2:
                                delay = EXPONENTIAL_BACKOFF_DELAYS_SEC[transient_retries]
                                transient_retries += 1
                                self._sleep(delay)
                                continue
                            break

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
                if rotate_to_next_key:
                    break

        return GeminiCallResult(
            ok=False,
            text="",
            model_used=chain[-1] if chain else "",
            error_code=last_code,
            error_message_vi=last_msg,
            attempts=total_attempts,
        )
