#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local AI Engine Manager & Smart Hybrid Router (`cv_engine/local_ai_manager.py`)
for AirTeach AI (DNT - Victer):
1. Auto-discovers local OpenAI-compatible & native servers:
   - Ollama (`http://localhost:11434`)
   - LM Studio (`http://localhost:1234`)
   - llama.cpp server (`http://localhost:8080`, `http://localhost:8081`)
2. Maximum-Speed Optimizations:
   - Token-by-token SSE & NDJSON streaming with real TTFT and `tokens/s` measurement
   - Non-blocking model warmup + `keep_alive: "15m"`
   - Fixed compact System Prompt prefix maximizing KV-Cache / Prefix Caching hit rate
   - Sliding-window conversation history + summary compression
   - In-memory LRU + persistent local SQLite exact-match response cache (0ms hit)
3. Smart 3-Tier Hybrid Routing:
   - `LOCAL`: 100% offline Local LLM
   - `CLOUD`: Gemini Cloud API (`ResilientGeminiClient`)
   - `HYBRID`: Local LLM preferred -> automatic smooth Cloud fallback if Local unavailable or heavy multimodal needed.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import urllib.error
import urllib.request

from cv_engine.app_paths import get_app_logger, get_local_cache_db_path
from cv_engine.gemini_client import (
    DEFAULT_GEMINI_MODEL,
    ResilientGeminiClient,
    compress_snapshot_png,
    resolve_api_key_layered,
    resolve_api_key_pool,
)
from cv_engine.output_sanitizer import OutputSanitizer, StreamingSanitizer

_LOGGER = get_app_logger("local_ai_manager")


class AiRoutingMode(str, Enum):
    HYBRID = "HYBRID"
    LOCAL = "LOCAL"
    CLOUD = "CLOUD"

    @property
    def label_vi(self) -> str:
        return {
            AiRoutingMode.HYBRID: "Hybrid",
            AiRoutingMode.LOCAL: "Local",
            AiRoutingMode.CLOUD: "Cloud",
        }[self]


@dataclass
class DiscoveredLocalServer:
    provider: str  # "Ollama" | "LM Studio" | "llama.cpp"
    base_url: str
    models: List[str] = field(default_factory=list)
    latency_ms: float = 0.0


@dataclass
class LocalAiRunResult:
    ok: bool
    text: str
    sanitized_text: str
    engine_used: str  # "Local" | "Cloud" | "Cache"
    model_used: str
    tokens_per_sec: float = 0.0
    ttft_ms: float = 0.0
    total_ms: float = 0.0
    from_cache: bool = False
    error_message_vi: str = ""


# Rich pedagogical System Prompt prefix maximizing explanation depth & KV-Cache reuse
IMMUTABLE_SYSTEM_PROMPT_PREFIX = (
    "Bạn là Trợ lý Giảng dạy chuyên sâu của AirTeach AI (chuyên gia sư phạm, khoa học và kỹ thuật). "
    "Nguyên tắc trả lời bắt buộc bằng tiếng Việt:\n"
    "1. Trình bày đầy đủ, sâu sắc, mạch lạc — TUYỆT ĐỐI không trả lời cụt lủn hoặc cắt ngang ý.\n"
    "2. Tổ chức câu trả lời rõ ràng bằng tiêu đề Markdown (##, ###), danh sách có cấu trúc, bảng so sánh Markdown khi cần đối chiếu, và ví dụ minh họa cụ thể.\n"
    "3. Với bài toán / lý / hóa / kỹ thuật: giải chi tiết từng bước (Step-by-step), giải thích bản chất công thức, chỉ rõ lỗi sai học sinh thường gặp và tóm tắt kết luận.\n"
    "4. Sử dụng công thức LaTeX chuẩn ($...$ cho công thức nội dòng, $$...$$ cho công thức khối) và mũi tên chuẩn (→, ⇒, ⇔); không dùng thẻ giả <arrow> hay $$$."
)

RECOMMENDED_LOCAL_MODELS: Tuple[str, ...] = (
    "qwen3:4b",
    "qwen3:8b",
    "gemma4:e4b",
    "gemma3:4b",
    "phi4-mini",
)


class LocalResponseCache:
    """Two-tier (Memory LRU + SQLite) exact/normalized local cache providing 0ms repeated replies."""

    def __init__(self, db_path: Optional[Path] = None, max_mem_items: int = 128) -> None:
        self.db_path = Path(db_path) if db_path is not None else get_local_cache_db_path()
        self.max_mem_items = max(16, int(max_mem_items))
        self._mem: OrderedDict[str, Tuple[str, str]] = OrderedDict()
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.db_path), timeout=2.0) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ai_fast_cache (
                        cache_key TEXT PRIMARY KEY,
                        model_id TEXT NOT NULL,
                        response_text TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
        except Exception as exc:
            _LOGGER.warning("Cannot init local AI cache DB at %s: %s", self.db_path, exc)

    @staticmethod
    def make_key(model_id: str, prompt: str, png_bytes: Optional[bytes] = None) -> str:
        norm_prompt = " ".join((prompt or "").strip().lower().split())
        h = hashlib.sha256()
        h.update(f"{model_id.strip().lower()}|{norm_prompt}|".encode("utf-8"))
        if png_bytes:
            h.update(hashlib.sha256(png_bytes).digest())
        return h.hexdigest()

    def get(self, key: str) -> Optional[Tuple[str, str]]:
        with self._lock:
            if key in self._mem:
                val = self._mem.pop(key)
                self._mem[key] = val
                return val
        try:
            if self.db_path.exists():
                with sqlite3.connect(str(self.db_path), timeout=2.0) as conn:
                    row = conn.execute(
                        "SELECT model_id, response_text FROM ai_fast_cache WHERE cache_key = ?",
                        (key,),
                    ).fetchone()
                    if row:
                        res = (str(row[0]), str(row[1]))
                        with self._lock:
                            self._mem[key] = res
                            while len(self._mem) > self.max_mem_items:
                                self._mem.popitem(last=False)
                        return res
        except Exception as exc:
            _LOGGER.debug("Local cache read error: %s", exc)
        return None

    def put(self, key: str, model_id: str, response_text: str) -> None:
        if not response_text.strip():
            return
        with self._lock:
            self._mem[key] = (model_id, response_text)
            self._mem.move_to_end(key)
            while len(self._mem) > self.max_mem_items:
                self._mem.popitem(last=False)
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.db_path), timeout=2.0) as conn:
                conn.execute(
                    """
                    INSERT INTO ai_fast_cache (cache_key, model_id, response_text, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        model_id = excluded.model_id,
                        response_text = excluded.response_text,
                        updated_at = excluded.updated_at
                    """,
                    (key, model_id, response_text, time.time()),
                )
        except Exception as exc:
            _LOGGER.debug("Local cache write error: %s", exc)


def build_sliding_window_messages(
    system_prompt: str,
    history: Sequence[Dict[str, str]],
    user_prompt: str,
    max_turns: int = 8,
    image_b64: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Construct messages keeping an immutable system prompt prefix (for KV-Cache reuse)
    and bounding history to the last `max_turns` turns + compressed older context.
    """
    msgs: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    clean_hist = [h for h in history if h.get("role") in ("user", "assistant") and h.get("content")]
    if len(clean_hist) > max_turns:
        older = clean_hist[:-max_turns]
        snippets = [
            f"- {m['role']}: {m['content'][:75].replace(chr(10), ' ')}"
            for m in older[-4:]
        ]
        msgs.append(
            {
                "role": "system",
                "content": "Tóm tắt ngữ cảnh trước đó:\n" + "\n".join(snippets),
            }
        )
        recent = clean_hist[-max_turns:]
    else:
        recent = clean_hist

    for m in recent:
        msgs.append({"role": m["role"], "content": m["content"]})

    if image_b64:
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        )
    else:
        msgs.append({"role": "user", "content": user_prompt})
    return msgs


class LocalAIManager:
    """Manages local LLM auto-discovery, warmup/keep-alive, SSE streaming, cache & hybrid routing."""

    DEFAULT_PROBES: Tuple[Tuple[str, str, str], ...] = (
        ("Ollama", "http://127.0.0.1:11434", "/api/tags"),
        ("LM Studio", "http://127.0.0.1:1234", "/v1/models"),
        ("llama.cpp", "http://127.0.0.1:8080", "/v1/models"),
        ("llama.cpp", "http://127.0.0.1:8081", "/v1/models"),
    )

    def __init__(
        self,
        routing_mode: AiRoutingMode = AiRoutingMode.HYBRID,
        preferred_local_model: str = "",
        preferred_base_url: str = "",
        cache: Optional[LocalResponseCache] = None,
        cloud_client: Optional[ResilientGeminiClient] = None,
    ) -> None:
        self.routing_mode = routing_mode
        self.preferred_local_model = preferred_local_model.strip()
        self.preferred_base_url = preferred_base_url.strip().rstrip("/")
        self.cache = cache or LocalResponseCache()
        self.cloud_client = cloud_client or ResilientGeminiClient()
        self.discovered_servers: List[DiscoveredLocalServer] = []
        self.active_provider: str = ""
        self.active_base_url: str = self.preferred_base_url
        self.active_model: str = self.preferred_local_model
        self.active_gguf_path: str = ""
        self._gguf_backend: Any = None
        self.last_tokens_per_sec: float = 0.0
        self._lock = threading.Lock()

    def set_gguf_model(self, gguf_path: str) -> None:
        """Configure or clear a local `.gguf` file path for 100% offline in-process inference."""
        with self._lock:
            cleaned = str(gguf_path or "").strip()
            self.active_gguf_path = cleaned
            if cleaned:
                self.active_provider = "GGUF (In-Process)"
                self.active_model = Path(cleaned).name
            elif self.active_provider == "GGUF (In-Process)":
                self.active_provider = ""
                self.active_model = ""

    def _ensure_gguf_backend(self) -> Any:
        if self._gguf_backend is None:
            from cv_engine.gguf_model_manager import LlamaCppLocalBackend

            self._gguf_backend = LlamaCppLocalBackend()
        return self._gguf_backend

    @staticmethod
    def _http_get_json(url: str, timeout: float = 0.45) -> Tuple[Optional[Dict[str, Any]], float]:
        t0 = time.perf_counter()
        req = urllib.request.Request(url, method="GET", headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        latency = (time.perf_counter() - t0) * 1000.0
        obj = json.loads(raw) if raw.strip() else None
        return (obj if isinstance(obj, dict) else None), latency

    @staticmethod
    def _rank_models(models: Iterable[str]) -> List[str]:
        unique = []
        for m in models:
            s = str(m).strip()
            if s and s not in unique:
                unique.append(s)

        def score(name: str) -> Tuple[int, str]:
            low = name.lower()
            for idx, pref in enumerate(("qwen3", "gemma4", "gemma-4", "phi-4", "phi4", "qwen2.5", "llama")):
                if pref in low:
                    return (idx, low)
            return (99, low)

        return sorted(unique, key=score)

    def discover_local_engines(self, timeout: float = 0.45) -> List[DiscoveredLocalServer]:
        """Auto-scan local ports for running Ollama, LM Studio, or llama.cpp servers."""
        found: List[DiscoveredLocalServer] = []
        probes = list(self.DEFAULT_PROBES)
        if self.preferred_base_url and not any(self.preferred_base_url == p[1] for p in probes):
            probes.insert(0, ("Custom-Local", self.preferred_base_url, "/v1/models"))

        for provider, base_url, path in probes:
            try:
                data, lat = self._http_get_json(f"{base_url}{path}", timeout=timeout)
                if data is None:
                    continue
                models: List[str] = []
                if provider == "Ollama" and isinstance(data.get("models"), list):
                    models = [str(item.get("name", "")) for item in data["models"] if isinstance(item, dict)]
                elif isinstance(data.get("data"), list):
                    models = [str(item.get("id", "")) for item in data["data"] if isinstance(item, dict)]
                ranked = self._rank_models(models)
                if ranked or provider in ("LM Studio", "llama.cpp", "Ollama"):
                    found.append(
                        DiscoveredLocalServer(
                            provider=provider,
                            base_url=base_url,
                            models=ranked,
                            latency_ms=lat,
                        )
                    )
            except Exception:
                continue

        with self._lock:
            self.discovered_servers = found
            if found:
                chosen = found[0]
                if self.preferred_base_url:
                    for srv in found:
                        if srv.base_url == self.preferred_base_url:
                            chosen = srv
                            break
                self.active_provider = chosen.provider
                self.active_base_url = chosen.base_url
                if not self.active_model or (chosen.models and self.active_model not in chosen.models):
                    self.active_model = chosen.models[0] if chosen.models else ""
            else:
                self.active_provider = ""
                if not self.preferred_base_url:
                    self.active_base_url = ""
        return found

    def select_local_model(self, base_url: str, model_id: str, provider: str = "") -> None:
        with self._lock:
            self.active_base_url = base_url.strip().rstrip("/")
            self.active_model = model_id.strip()
            if provider:
                self.active_provider = provider
            self.preferred_base_url = self.active_base_url
            self.preferred_local_model = self.active_model

    def warmup_async(self) -> None:
        """Send non-blocking lightweight keep-alive warmup request so first reply has minimal delay."""
        if not self.active_base_url or not self.active_model:
            return

        def _worker() -> None:
            try:
                if self.active_provider == "Ollama":
                    url = f"{self.active_base_url}/api/chat"
                    body = {
                        "model": self.active_model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False,
                        "keep_alive": "15m",
                        "options": {"num_predict": 1},
                    }
                else:
                    url = f"{self.active_base_url}/v1/chat/completions"
                    body = {
                        "model": self.active_model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "stream": False,
                    }
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=2.5) as resp:
                    _ = resp.read(16)
            except Exception as exc:
                _LOGGER.debug("Local AI warmup skipped (%s): %s", url, exc)

        threading.Thread(target=_worker, daemon=True, name="airteach-local-warmup").start()

    def _stream_local_completion(
        self,
        base_url: str,
        provider: str,
        model_id: str,
        messages: List[Dict[str, Any]],
        has_image: bool,
        on_token: Optional[Callable[[str], None]] = None,
        timeout: float = 45.0,
    ) -> LocalAiRunResult:
        t0 = time.perf_counter()
        ttft_ms = 0.0
        token_count = 0
        raw_accum: List[str] = []
        stream_sanitizer = StreamingSanitizer(preserve_latex=True)

        use_ollama_native = provider == "Ollama" and not has_image
        if use_ollama_native:
            url = f"{base_url}/api/chat"
            payload: Dict[str, Any] = {
                "model": model_id,
                "messages": messages,
                "stream": True,
                "keep_alive": "15m",
                "options": {"temperature": 0.25, "num_ctx": 8192, "num_predict": 8192},
            }
        else:
            url = f"{base_url}/v1/chat/completions"
            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": 0.25,
                "max_tokens": 8192,
                "stream": True,
            }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw_line_bytes in resp:
                    line = raw_line_bytes.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    delta_txt = ""
                    if use_ollama_native:
                        try:
                            obj = json.loads(line)
                            msg = obj.get("message") or {}
                            delta_txt = str(msg.get("content") or "")
                            if obj.get("done") and not delta_txt:
                                break
                        except json.JSONDecodeError:
                            continue
                    else:
                        if line.startswith("data:"):
                            payload_str = line[5:].strip()
                            if payload_str == "[DONE]":
                                break
                            try:
                                obj = json.loads(payload_str)
                                choices = obj.get("choices") or []
                                if choices:
                                    delta_txt = str((choices[0].get("delta") or {}).get("content") or "")
                            except json.JSONDecodeError:
                                continue

                    if delta_txt:
                        if token_count == 0:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        token_count += 1
                        raw_accum.append(delta_txt)
                        safe_chunk = stream_sanitizer.feed(delta_txt)
                        if safe_chunk and on_token is not None:
                            on_token(safe_chunk)
        except Exception as exc:
            return LocalAiRunResult(
                ok=False,
                text="",
                sanitized_text="",
                engine_used="Local",
                model_used=model_id,
                total_ms=(time.perf_counter() - t0) * 1000.0,
                error_message_vi=f"Không thể stream từ {provider or 'Local LLM'} ({url}): {exc}",
            )

        tail = stream_sanitizer.flush()
        if tail and on_token is not None:
            on_token(tail)

        total_ms = (time.perf_counter() - t0) * 1000.0
        gen_sec = max(0.05, (total_ms - ttft_ms) / 1000.0)
        tok_per_sec = float(token_count) / gen_sec if token_count > 0 else 0.0
        self.last_tokens_per_sec = tok_per_sec
        full_raw = "".join(raw_accum)
        full_sanitized = OutputSanitizer.sanitize_plain_text(full_raw)
        return LocalAiRunResult(
            ok=bool(full_raw.strip()),
            text=full_raw,
            sanitized_text=full_sanitized,
            engine_used="Local",
            model_used=model_id,
            tokens_per_sec=tok_per_sec,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            error_message_vi="" if full_raw.strip() else "Local LLM trả về phản hồi rỗng.",
        )

    def generate_hybrid(
        self,
        prompt: str,
        *,
        history: Sequence[Dict[str, str]] = (),
        png_bytes: Optional[bytes] = None,
        cloud_api_key: Optional[str] = None,
        cloud_model: Optional[str] = None,
        system_prompt: str = IMMUTABLE_SYSTEM_PROMPT_PREFIX,
        structured_json: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
        use_cache: bool = True,
    ) -> LocalAiRunResult:
        """
        Execute smart 3-tier routing (Cache -> Local LLM -> Gemini Cloud fallback).
        Automatically compresses PNG `<= 1024px` and sanitizes all responses.
        """
        compressed_png = compress_snapshot_png(png_bytes, max_dim=1024) if png_bytes else None
        active_local_ready = bool(self.active_base_url and self.active_model)
        cache_model = (
            self.active_model
            if (self.routing_mode == AiRoutingMode.LOCAL or (self.routing_mode == AiRoutingMode.HYBRID and active_local_ready))
            else (cloud_model or DEFAULT_GEMINI_MODEL)
        )
        cache_key = self.cache.make_key(cache_model, prompt, compressed_png)
        if use_cache and not structured_json:
            cached = self.cache.get(cache_key)
            if cached:
                c_model, c_text = cached
                sanitized = OutputSanitizer.sanitize_plain_text(c_text)
                if on_token is not None and sanitized:
                    on_token(sanitized)
                return LocalAiRunResult(
                    ok=True,
                    text=c_text,
                    sanitized_text=sanitized,
                    engine_used="Cache",
                    model_used=c_model,
                    tokens_per_sec=999.0,
                    ttft_ms=0.0,
                    total_ms=0.2,
                    from_cache=True,
                )

        b64_img = base64.b64encode(compressed_png).decode("ascii") if compressed_png else None

        # Try Local LLM first when mode is LOCAL or HYBRID (and a local server or GGUF model is active)
        if self.routing_mode in (AiRoutingMode.LOCAL, AiRoutingMode.HYBRID):
            gguf_file = Path(self.active_gguf_path) if self.active_gguf_path else None
            gguf_selected = bool(
                gguf_file
                and gguf_file.exists()
                and (self.active_provider == "GGUF (In-Process)" or not active_local_ready)
            )
            if gguf_selected and not structured_json:
                backend = self._ensure_gguf_backend()
                if backend.is_available():
                    t_gguf0 = time.perf_counter()
                    msgs_txt = [{"role": "system", "content": system_prompt}]
                    for h in history[-6:]:
                        role = str(h.get("role", "user")).strip().lower()
                        content = str(h.get("content", "")).strip()
                        if content:
                            msgs_txt.append({"role": role if role in ("user", "assistant") else "user", "content": content})
                    msgs_txt.append({"role": "user", "content": prompt})
                    stream_san = StreamingSanitizer(preserve_latex=True) if on_token is not None else None

                    def _gguf_tok(piece: str) -> None:
                        if stream_san is not None and on_token is not None:
                            safe_c = stream_san.feed(piece)
                            if safe_c:
                                on_token(safe_c)

                    ok_g, text_g, err_g = backend.stream_chat(
                        gguf_path=gguf_file,
                        messages=msgs_txt,
                        max_tokens=8192,
                        on_token=_gguf_tok if on_token is not None else None,
                    )
                    total_g_ms = (time.perf_counter() - t_gguf0) * 1000.0
                    if ok_g:
                        if stream_san is not None and on_token is not None:
                            tail = stream_san.flush()
                            if tail:
                                on_token(tail)
                        sanitized_g = OutputSanitizer.sanitize_plain_text(text_g)
                        if use_cache:
                            self.cache.put(cache_key, gguf_file.name, text_g)
                        return LocalAiRunResult(
                            ok=True,
                            text=text_g,
                            sanitized_text=sanitized_g,
                            engine_used="Local-GGUF",
                            model_used=gguf_file.name,
                            tokens_per_sec=max(1.0, len(text_g.split()) / max(0.1, total_g_ms / 1000.0)),
                            ttft_ms=total_g_ms,
                            total_ms=total_g_ms,
                        )
                    if self.routing_mode == AiRoutingMode.LOCAL:
                        return LocalAiRunResult(
                            ok=False,
                            text="",
                            sanitized_text="",
                            engine_used="Local-GGUF",
                            model_used=gguf_file.name,
                            error_message_vi=err_g,
                        )

            http_local_ready = bool(self.active_base_url and self.active_model and self.active_provider != "GGUF (In-Process)")
            if not http_local_ready and self.routing_mode == AiRoutingMode.HYBRID:
                self.discover_local_engines(timeout=0.35)
                http_local_ready = bool(self.active_base_url and self.active_model and self.active_provider != "GGUF (In-Process)")

            if http_local_ready and not structured_json:
                msgs = build_sliding_window_messages(
                    system_prompt=system_prompt,
                    history=history,
                    user_prompt=prompt,
                    image_b64=b64_img,
                )
                local_res = self._stream_local_completion(
                    base_url=self.active_base_url,
                    provider=self.active_provider or "OpenAI-Local",
                    model_id=self.active_model,
                    messages=msgs,
                    has_image=bool(b64_img),
                    on_token=on_token,
                )
                if local_res.ok:
                    if use_cache:
                        self.cache.put(cache_key, local_res.model_used, local_res.text)
                    return local_res
                if self.routing_mode == AiRoutingMode.LOCAL:
                    return local_res
            elif self.routing_mode == AiRoutingMode.LOCAL:
                hint = (
                    f" Đã tìm thấy file GGUF '{gguf_file.name}' nhưng chưa cài `llama-cpp-python`."
                    if gguf_file and gguf_file.exists()
                    else " Mở Kho Model Local (Hub) để tải model GGUF hoặc bật Ollama / LM Studio."
                )
                return LocalAiRunResult(
                    ok=False,
                    text="",
                    sanitized_text="",
                    engine_used="Local",
                    model_used="",
                    error_message_vi="Chế độ Local AI đang bật nhưng chưa có engine Local hoạt động." + hint,
                )

        # Tier 3: Gemini Cloud API (supports multi-key pool, deep system prompt, history & live streaming)
        key_pool = resolve_api_key_pool(cloud_api_key if (cloud_api_key or "").strip() else None)
        if not key_pool:
            return LocalAiRunResult(
                ok=False,
                text="",
                sanitized_text="",
                engine_used="Cloud",
                model_used=cloud_model or DEFAULT_GEMINI_MODEL,
                error_message_vi="Chưa có Local LLM đang chạy và chưa cấu hình GEMINI_API_KEY.",
            )

        t0 = time.perf_counter()
        stream_sanitizer = StreamingSanitizer(preserve_latex=True) if on_token is not None else None

        def _cloud_token(chunk: str) -> None:
            if stream_sanitizer is not None and on_token is not None:
                safe_chunk = stream_sanitizer.feed(chunk)
                if safe_chunk:
                    on_token(safe_chunk)

        res = self.cloud_client.generate_multimodal(
            api_key=key_pool,
            prompt=prompt,
            png_bytes=compressed_png,
            primary_model=cloud_model or DEFAULT_GEMINI_MODEL,
            structured_json=structured_json,
            on_token=_cloud_token if (on_token is not None and not structured_json) else None,
            system_instruction=system_prompt,
            history=history,
            max_output_tokens=8192,
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        if res.ok:
            sanitized = OutputSanitizer.sanitize_plain_text(res.text)
            est_tokens = max(1, len(res.text.split()))
            tps = est_tokens / max(0.1, total_ms / 1000.0)
            self.last_tokens_per_sec = tps
            if stream_sanitizer is not None and on_token is not None and not structured_json:
                tail = stream_sanitizer.flush()
                if tail:
                    on_token(tail)
            if use_cache and not structured_json:
                self.cache.put(cache_key, res.model_used, res.text)
            return LocalAiRunResult(
                ok=True,
                text=res.text,
                sanitized_text=sanitized,
                engine_used="Cloud",
                model_used=res.model_used,
                tokens_per_sec=tps,
                ttft_ms=total_ms,
                total_ms=total_ms,
            )

        return LocalAiRunResult(
            ok=False,
            text="",
            sanitized_text="",
            engine_used="Cloud",
            model_used=res.model_used,
            total_ms=total_ms,
            error_message_vi=res.error_message_vi,
        )
