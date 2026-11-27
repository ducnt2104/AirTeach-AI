#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 6 Verification Suite: Rich AI Chatbot (8192 tokens, no truncation, Copy & Push-to-Board
Text Object, 1-click Cloud<->Local toggle) & Local AI Model Hub (curated GGUF catalog,
GitHub/HuggingFace URL normalization, resumable Range download, SHA-256 & GGUF magic check).
"""

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import sys
import threading
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication

import main as app_main
from cv_engine.gemini_client import DEFAULT_FALLBACK_MODELS, DEFAULT_MAX_OUTPUT_TOKENS
from cv_engine.gguf_model_manager import (
    CURATED_GGUF_CATALOG,
    LocalAiManagerDialog,
    ResumableModelDownloader,
    detect_local_ai_runtimes,
    normalize_model_download_url,
)
from cv_engine.local_ai_manager import IMMUTABLE_SYSTEM_PROMPT_PREFIX, AiRoutingMode


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_ai_token_limits_catalog_and_url_normalization():
    # Token budget >= 8192 and stable Gemini fallback chain
    assert DEFAULT_MAX_OUTPUT_TOKENS >= 8192
    assert "gemini-2.5-flash" in DEFAULT_FALLBACK_MODELS
    assert "gemini-2.0-flash" in DEFAULT_FALLBACK_MODELS
    assert "Markdown" in IMMUTABLE_SYSTEM_PROMPT_PREFIX
    assert "LaTeX" in IMMUTABLE_SYSTEM_PROMPT_PREFIX

    # Curated GGUF catalog includes Qwen2.5 1.5B/3B, Llama-3.2 1B/3B, Gemma-2 2B
    ids = {m.model_id for m in CURATED_GGUF_CATALOG}
    assert "qwen2.5-1.5b-instruct-q4_k_m" in ids
    assert "qwen2.5-3b-instruct-q4_k_m" in ids
    assert "llama-3.2-1b-instruct-q4_k_m" in ids
    assert "llama-3.2-3b-instruct-q4_k_m" in ids
    assert "gemma-2-2b-it-q4_k_m" in ids
    for m in CURATED_GGUF_CATALOG:
        assert "RAM" in m.ram_requirement

    # GitHub Releases & HuggingFace blob -> raw/resolve URL normalization
    hf = normalize_model_download_url("https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/blob/main/qwen.gguf")
    assert "/resolve/main/" in hf
    gh = normalize_model_download_url("https://github.com/org/repo/blob/main/model.gguf")
    assert "/raw/main/" in gh

    rt = detect_local_ai_runtimes()
    assert "llama_cpp_python" in rt and "onnxruntime" in rt and "onnxruntime_genai" in rt


def test_resumable_gguf_download_with_sha256_and_progress(tmp_path: Path):
    payload = b"GGUF" + os.urandom(65536)
    expected_sha = hashlib.sha256(payload).hexdigest()

    class _RangeHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            rng = self.headers.get("Range", "")
            if rng.startswith("bytes="):
                start = int(rng.split("=", 1)[1].split("-", 1)[0])
                chunk = payload[start:]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        def log_message(self, *_a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), _RangeHandler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        # Simulate partial download resume from first 16 KB
        part = tmp_path / "mini_test.gguf.part"
        part.write_bytes(payload[:16384])

        events = []
        dl = ResumableModelDownloader()
        final_p, sha_hex = dl.download(
            url=f"http://127.0.0.1:{port}/mini_test.gguf",
            dest_dir=tmp_path,
            filename="mini_test.gguf",
            expected_sha256=expected_sha,
            verify_gguf=True,
            progress_cb=events.append,
        )
        assert final_p.exists()
        assert sha_hex == expected_sha
        assert any(ev.resumed for ev in events)
        assert events[-1].percent == 100.0
    finally:
        srv.shutdown()
        srv.server_close()


def test_push_full_ai_reply_to_canvas_text_object_and_mode_toggle(qapp):
    orig_start = app_main.CameraHandWorker.start
    app_main.CameraHandWorker.start = lambda self: None  # type: ignore[method-assign]
    win = app_main.AirTeachMainWindow()
    try:
        win.stack.setCurrentIndex(1)
        long_reply = "\n".join(f"Bước {i}: Giải thích chi tiết định lý số {i} kèm công thức x_{i} = {i}^2" for i in range(1, 13))
        win.last_ai_raw_text = f"*[Cloud · gemini-2.5-flash · 50.0 tok/s]*\n\n{long_reply}"

        # Push to board creates an independent Text CanvasObject with ALL 12 lines (no 6-line truncation!)
        win._on_push_last_ai_to_board()
        txt_objs = [o for o in win.canvas.scene.objects if o.obj_type == "text"]
        assert len(txt_objs) == 1
        assert "Bước 12:" in txt_objs[0].text
        assert win.canvas.active_tool == "SELECT"

        # Copy button copies full text cleanly
        win._on_copy_last_ai_reply()
        assert "Bước 12:" in QApplication.clipboard().text()

        # Normal Q&A does not force terse board tool JSON; explicit board request does
        assert win._prompt_needs_board_tools("Giải thích chi tiết định luật bảo toàn năng lượng") is False
        assert win._prompt_needs_board_tools("Vẽ đồ thị hàm số y = x^2 lên bảng giúp tôi") is True

        # 1-click toggle cycles Cloud -> Local -> Hybrid cleanly
        win.local_ai_manager.routing_mode = AiRoutingMode.CLOUD
        win._on_quick_toggle_ai_mode()
        assert win.local_ai_manager.routing_mode == AiRoutingMode.LOCAL
        win._on_quick_toggle_ai_mode()
        assert win.local_ai_manager.routing_mode == AiRoutingMode.HYBRID

        # Model Hub dialog opens cleanly with Pause & Cancel buttons
        dlg = LocalAiManagerDialog()
        assert hasattr(dlg, "_pause_dl_btn") and hasattr(dlg, "_cancel_dl_btn")
        dlg.close()
    finally:
        app_main.CameraHandWorker.start = orig_start  # type: ignore[method-assign]
        win.shutdown_resources()
        win.close()
