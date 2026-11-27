"""Local AI GGUF Model Manager & Resumable GitHub/HuggingFace Downloader.

Provides:
1. Curated catalog of official GGUF models hosted on HuggingFace / GitHub Releases.
2. Resumable HTTP Range downloader with live speed (MB/s), percentage progress,
   GGUF header check, and SHA-256 integrity verification stored in
   `%APPDATA%/AirTeachAI/models`.
3. Auto-detection of local inference runtimes (`llama-cpp-python`, Ollama,
   LM Studio, llama-server.exe, ONNX Runtime).
4. Optional in-process `LlamaCppLocalBackend` for 100% offline GGUF inference.
5. `LocalAiManagerDialog` PySide6 UI supporting Hybrid AI mode selection,
   curated 1-click downloads, and custom GitHub/HuggingFace `.gguf` URLs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDesktopServices, QFont
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from cv_engine.app_paths import get_app_logger

_LOGGER = get_app_logger("gguf_model_manager")

_GGUF_MAGIC = b"GGUF"
_USER_AGENT = "AirTeachAI-ModelManager/1.0 (+https://github.com/ducnt2104/AirTeach-AI)"


@dataclass(frozen=True)
class GgufModelSpec:
    """Descriptor for a downloadable open-weights GGUF model."""

    model_id: str
    name: str
    provider: str
    quantization: str
    size_bytes: int
    size_label: str
    ram_requirement: str
    languages: str
    description: str
    download_url: str
    repo_url: str
    filename: str
    sha256: str = ""
    recommended: bool = False


CURATED_GGUF_CATALOG: Tuple[GgufModelSpec, ...] = (
    GgufModelSpec(
        model_id="qwen2.5-1.5b-instruct-q4_k_m",
        name="Qwen 2.5 1.5B Instruct (Q4_K_M)",
        provider="HuggingFace • Alibaba Qwen",
        quantization="Q4_K_M",
        size_bytes=1117320736,
        size_label="1.04 GB",
        ram_requirement="RAM ≥ 4 GB",
        languages="Tiếng Việt tốt • Đa ngôn ngữ • Toán & Khoa học",
        description="Cân bằng tốc độ và chất lượng giải bài tiếng Việt trên CPU phổ thông.",
        download_url="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        repo_url="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
        recommended=True,
    ),
    GgufModelSpec(
        model_id="qwen2.5-3b-instruct-q4_k_m",
        name="Qwen 2.5 3B Instruct (Q4_K_M)",
        provider="HuggingFace • Alibaba Qwen",
        quantization="Q4_K_M",
        size_bytes=2104932768,
        size_label="1.96 GB",
        ram_requirement="RAM ≥ 6 GB",
        languages="Tiếng Việt rất tốt • Lập luận sâu • LaTeX",
        description="Suy luận sư phạm chi tiết, trình bày bảng và công thức toán chính xác cao.",
        download_url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        repo_url="https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF",
        filename="qwen2.5-3b-instruct-q4_k_m.gguf",
        recommended=True,
    ),
    GgufModelSpec(
        model_id="qwen2.5-0.5b-instruct-q4_k_m",
        name="Qwen 2.5 0.5B Instruct (Q4_K_M)",
        provider="HuggingFace • Alibaba Qwen",
        quantization="Q4_K_M",
        size_bytes=491400032,
        size_label="469 MB",
        ram_requirement="RAM ≥ 2 GB",
        languages="Tiếng Việt cơ bản • Siêu nhẹ",
        description="Phù hợp máy cấu hình thấp hoặc Demo nhanh hoàn toàn offline.",
        download_url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf",
        repo_url="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
        recommended=False,
    ),
    GgufModelSpec(
        model_id="gemma-2-2b-it-q4_k_m",
        name="Google Gemma 2 2B IT (Q4_K_M)",
        provider="HuggingFace • bartowski / Google",
        quantization="Q4_K_M",
        size_bytes=1708582752,
        size_label="1.59 GB",
        ram_requirement="RAM ≥ 6 GB",
        languages="Tiếng Anh & Việt • Giáo dục STEM",
        description="Mô hình giáo dục nhẹ từ Google, trình bày khái niệm khoa học mạch lạc.",
        download_url="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf",
        repo_url="https://huggingface.co/bartowski/gemma-2-2b-it-GGUF",
        filename="gemma-2-2b-it-Q4_K_M.gguf",
        recommended=False,
    ),
    GgufModelSpec(
        model_id="llama-3.2-1b-instruct-q4_k_m",
        name="Meta Llama 3.2 1B Instruct (Q4_K_M)",
        provider="HuggingFace • bartowski / Meta",
        quantization="Q4_K_M",
        size_bytes=807694464,
        size_label="770 MB",
        ram_requirement="RAM ≥ 3 GB",
        languages="Nhẹ siêu nhanh • Đa ngôn ngữ",
        description="Khởi chạy cực nhanh trên laptop CPU văn phòng (AVX2), phù hợp giảng dạy offline tức thì.",
        download_url="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        repo_url="https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        recommended=True,
    ),
    GgufModelSpec(
        model_id="llama-3.2-3b-instruct-q4_k_m",
        name="Meta Llama 3.2 3B Instruct (Q4_K_M)",
        provider="HuggingFace • bartowski / Meta",
        quantization="Q4_K_M",
        size_bytes=2019377696,
        size_label="1.88 GB",
        ram_requirement="RAM ≥ 6 GB",
        languages="Đa ngôn ngữ • Hỏi đáp nhanh",
        description="Tốc độ phản hồi ổn định, mạnh về cấu trúc câu trả lời ngắn gọn và ví dụ.",
        download_url="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        repo_url="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        recommended=False,
    ),
    GgufModelSpec(
        model_id="phi-3.5-mini-instruct-q4_k_m",
        name="Microsoft Phi-3.5 Mini Instruct (Q4_K_M)",
        provider="HuggingFace • bartowski / Microsoft",
        quantization="Q4_K_M",
        size_bytes=2393232672,
        size_label="2.23 GB",
        ram_requirement="RAM ≥ 8 GB",
        languages="STEM • Toán • Code & Logic",
        description="Mạnh vượt trội về Toán học, Vật lý, lập trình và suy luận đa bước.",
        download_url="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf",
        repo_url="https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF",
        filename="Phi-3.5-mini-instruct-Q4_K_M.gguf",
        recommended=False,
    ),
)


def get_models_storage_dir() -> Path:
    """Return persistent directory `%APPDATA%/AirTeachAI/models` (with test override)."""
    override = os.environ.get("AIRTEACH_MODELS_DIR", "").strip()
    if override:
        target = Path(override).expanduser()
    elif os.name == "nt":
        appdata = os.environ.get("APPDATA", "").strip()
        base = Path(appdata) if appdata else (Path.home() / "AppData" / "Roaming")
        target = base / "AirTeachAI" / "models"
    else:
        xdg = os.environ.get("XDG_DATA_HOME", "").strip()
        base = Path(xdg) if xdg else (Path.home() / ".local" / "share")
        target = base / "AirTeachAI" / "models"
    target.mkdir(parents=True, exist_ok=True)
    return target


def normalize_model_download_url(raw_url: str) -> str:
    """Normalize HuggingFace / GitHub web URLs into direct raw binary download URLs."""
    url = str(raw_url or "").strip()
    if not url:
        raise ValueError("URL tải model không được để trống.")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(f"URL không hợp lệ (chỉ hỗ trợ http/https): {url}")

    host = parsed.netloc.lower()
    path = parsed.path

    if "huggingface.co" in host and "/blob/" in path:
        path = path.replace("/blob/", "/resolve/", 1)
        parsed = parsed._replace(path=path)
        return urllib.parse.urlunparse(parsed)

    if host == "github.com" and "/blob/" in path:
        path = path.replace("/blob/", "/raw/", 1)
        parsed = parsed._replace(path=path)
        return urllib.parse.urlunparse(parsed)

    return url


def infer_filename_from_url(url: str, fallback: str = "custom_model.gguf") -> str:
    """Extract clean safe filename from a direct model URL."""
    parsed = urllib.parse.urlparse(url)
    candidate = Path(urllib.parse.unquote(parsed.path)).name
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in candidate).strip("._")
    if not safe:
        return fallback
    if not safe.lower().endswith(".gguf") and "." not in safe:
        safe = f"{safe}.gguf"
    return safe


def compute_file_sha256(path: Path, chunk_size: int = 1048576) -> str:
    """Stream SHA-256 digest of a file without loading it all into RAM."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest().lower()


def verify_gguf_magic(path: Path) -> bool:
    """Check whether `path` starts with the standard 4-byte `GGUF` magic header."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == _GGUF_MAGIC
    except Exception:
        return False


@dataclass
class LocalDownloadedModel:
    """Represents a GGUF file present in the local models storage directory."""

    filename: str
    path: Path
    size_bytes: int
    size_label: str
    sha256: str
    valid_gguf: bool
    is_partial: bool = False


def _format_bytes_human(num_bytes: int) -> str:
    size = float(max(0, num_bytes))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}" if unit in ("GB", "TB") else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def list_downloaded_models(models_dir: Optional[Path] = None) -> List[LocalDownloadedModel]:
    """List complete `.gguf` and incomplete `.gguf.part` models inside storage dir."""
    root = models_dir or get_models_storage_dir()
    root.mkdir(parents=True, exist_ok=True)
    items: List[LocalDownloadedModel] = []

    for gguf_path in sorted(root.glob("*.gguf"), key=lambda p: p.name.lower()):
        try:
            size = gguf_path.stat().st_size
        except OSError:
            continue
        sidecar = Path(f"{gguf_path}.sha256")
        sha_val = ""
        if sidecar.exists():
            try:
                sha_val = sidecar.read_text(encoding="utf-8").strip().splitlines()[0].split()[0].lower()
            except Exception:
                sha_val = ""
        items.append(
            LocalDownloadedModel(
                filename=gguf_path.name,
                path=gguf_path,
                size_bytes=size,
                size_label=_format_bytes_human(size),
                sha256=sha_val,
                valid_gguf=verify_gguf_magic(gguf_path),
                is_partial=False,
            )
        )

    for part_path in sorted(root.glob("*.gguf.part"), key=lambda p: p.name.lower()):
        try:
            size = part_path.stat().st_size
        except OSError:
            continue
        items.append(
            LocalDownloadedModel(
                filename=part_path.name[:-5],
                path=part_path,
                size_bytes=size,
                size_label=_format_bytes_human(size),
                sha256="",
                valid_gguf=False,
                is_partial=True,
            )
        )

    return items


@dataclass
class DownloadProgress:
    """Live telemetry snapshot emitted during a resumable download."""

    filename: str
    downloaded_bytes: int
    total_bytes: int
    percent: float
    speed_mb_s: float
    resumed: bool
    status: str


class ResumableModelDownloader:
    """Resumable HTTP Range downloader with live throughput and SHA-256 verification."""

    def download(
        self,
        url: str,
        dest_dir: Optional[Path] = None,
        filename: Optional[str] = None,
        expected_sha256: str = "",
        verify_gguf: bool = True,
        timeout_sec: float = 45.0,
        chunk_size: int = 262144,
        progress_cb: Optional[Callable[[DownloadProgress], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[Path, str]:
        """Download `url` into `dest_dir / filename` with `.part` resume and SHA-256 check."""
        resolved_url = normalize_model_download_url(url)
        target_dir = Path(dest_dir) if dest_dir else get_models_storage_dir()
        target_dir.mkdir(parents=True, exist_ok=True)
        resolved_name = filename or infer_filename_from_url(resolved_url)
        final_path = target_dir / resolved_name
        part_path = target_dir / f"{resolved_name}.part"

        expected = (expected_sha256 or "").strip().lower()
        existing_bytes = part_path.stat().st_size if part_path.exists() else 0

        if final_path.exists() and not part_path.exists():
            actual_existing = compute_file_sha256(final_path)
            if (not expected or actual_existing == expected) and (not verify_gguf or verify_gguf_magic(final_path)):
                if progress_cb:
                    fsize = final_path.stat().st_size
                    progress_cb(
                        DownloadProgress(
                            filename=resolved_name,
                            downloaded_bytes=fsize,
                            total_bytes=fsize,
                            percent=100.0,
                            speed_mb_s=0.0,
                            resumed=False,
                            status="Đã có sẵn và xác thực SHA-256 thành công.",
                        )
                    )
                return final_path, actual_existing

        headers = {"User-Agent": _USER_AGENT}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"

        req = urllib.request.Request(resolved_url, headers=headers, method="GET")
        resumed = False
        total_bytes = 0

        try:
            resp = urllib.request.urlopen(req, timeout=timeout_sec)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and existing_bytes > 0:
                # Local .part file already matches or exceeds remote size -> retry from 0
                existing_bytes = 0
                req = urllib.request.Request(resolved_url, headers={"User-Agent": _USER_AGENT}, method="GET")
                resp = urllib.request.urlopen(req, timeout=timeout_sec)
            else:
                raise

        with resp:
            status_code = getattr(resp, "status", None) or resp.getcode()
            content_range = resp.headers.get("Content-Range", "")
            content_length_raw = resp.headers.get("Content-Length", "")
            content_length = int(content_length_raw) if content_length_raw.isdigit() else 0

            if status_code == 206 and existing_bytes > 0:
                resumed = True
                if "/" in content_range:
                    total_part = content_range.rsplit("/", 1)[-1].strip()
                    if total_part.isdigit():
                        total_bytes = int(total_part)
                if total_bytes <= 0 and content_length > 0:
                    total_bytes = existing_bytes + content_length
                file_mode = "ab"
                downloaded = existing_bytes
            else:
                resumed = False
                total_bytes = content_length
                file_mode = "wb"
                downloaded = 0

            start_t = time.perf_counter()
            session_bytes = 0
            last_emit_t = 0.0

            with open(part_path, file_mode) as out_f:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise InterruptedError("Tải model đã tạm dừng bởi người dùng.")
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out_f.write(chunk)
                    downloaded += len(chunk)
                    session_bytes += len(chunk)

                    now = time.perf_counter()
                    if progress_cb and (now - last_emit_t >= 0.08 or (total_bytes > 0 and downloaded >= total_bytes)):
                        elapsed = max(0.001, now - start_t)
                        speed_mb_s = (session_bytes / (1024.0 * 1024.0)) / elapsed
                        pct = min(100.0, (downloaded * 100.0 / total_bytes)) if total_bytes > 0 else 0.0
                        progress_cb(
                            DownloadProgress(
                                filename=resolved_name,
                                downloaded_bytes=downloaded,
                                total_bytes=total_bytes,
                                percent=pct,
                                speed_mb_s=speed_mb_s,
                                resumed=resumed,
                                status="Đang tải tiếp (Resume)..." if resumed else "Đang tải từ HuggingFace / GitHub...",
                            )
                        )
                        last_emit_t = now

        if total_bytes > 0 and downloaded != total_bytes:
            raise IOError(f"Tải file chưa đủ dung lượng ({downloaded} / {total_bytes} bytes). Hãy nhấn Tải lại để tiếp tục.")

        if verify_gguf and not verify_gguf_magic(part_path):
            part_path.unlink(missing_ok=True)
            raise ValueError(f"File '{resolved_name}' không có header GGUF hợp lệ (có thể URL trả về trang HTML lỗi).")

        if progress_cb:
            progress_cb(
                DownloadProgress(
                    filename=resolved_name,
                    downloaded_bytes=downloaded,
                    total_bytes=total_bytes or downloaded,
                    percent=99.9,
                    speed_mb_s=0.0,
                    resumed=resumed,
                    status="Đang kiểm tra mã băm SHA-256...",
                )
            )

        actual_sha = compute_file_sha256(part_path)
        if expected and actual_sha != expected:
            part_path.unlink(missing_ok=True)
            raise ValueError(
                f"SHA-256 không khớp! Mong đợi {expected[:16]}..., nhận được {actual_sha[:16]}..."
            )

        if final_path.exists():
            final_path.unlink()
        os.replace(part_path, final_path)

        sidecar = Path(f"{final_path}.sha256")
        try:
            sidecar.write_text(f"{actual_sha}  {resolved_name}\n", encoding="utf-8")
        except OSError as exc:
            _LOGGER.debug("Cannot write SHA-256 sidecar %s: %s", sidecar, exc)

        if progress_cb:
            final_size = final_path.stat().st_size
            progress_cb(
                DownloadProgress(
                    filename=resolved_name,
                    downloaded_bytes=final_size,
                    total_bytes=final_size,
                    percent=100.0,
                    speed_mb_s=0.0,
                    resumed=resumed,
                    status=f"Hoàn tất • SHA-256: {actual_sha[:12]}...",
                )
            )

        return final_path, actual_sha


def _probe_http_models(url: str, timeout: float = 0.65) -> Tuple[bool, List[str]]:
    req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if getattr(resp, "status", 200) != 200:
                return False, []
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
        names: List[str] = []
        if isinstance(payload.get("models"), list):
            for item in payload["models"]:
                if isinstance(item, dict) and item.get("name"):
                    names.append(str(item["name"]))
        elif isinstance(payload.get("data"), list):
            for item in payload["data"]:
                if isinstance(item, dict) and item.get("id"):
                    names.append(str(item["id"]))
        return True, names
    except Exception:
        return False, []


def detect_local_ai_runtimes() -> Dict[str, Any]:
    """Detect available local AI execution engines without blocking startup."""
    has_llama_cpp_py = importlib.util.find_spec("llama_cpp") is not None
    has_onnxruntime = importlib.util.find_spec("onnxruntime") is not None
    has_onnx_genai = importlib.util.find_spec("onnxruntime_genai") is not None
    ollama_bin = shutil.which("ollama")
    llama_server_bin = shutil.which("llama-server") or shutil.which("llama-cli")

    ollama_online, ollama_models = _probe_http_models("http://127.0.0.1:11434/api/tags")
    lm_studio_online, lm_models = _probe_http_models("http://127.0.0.1:1234/v1/models")
    llama_server_online, llama_models = _probe_http_models("http://127.0.0.1:8080/v1/models")

    return {
        "llama_cpp_python": has_llama_cpp_py,
        "onnxruntime": has_onnxruntime,
        "onnxruntime_genai": has_onnx_genai,
        "ollama_installed": bool(ollama_bin),
        "ollama_online": ollama_online,
        "ollama_models": ollama_models,
        "lm_studio_online": lm_studio_online,
        "lm_studio_models": lm_models,
        "llama_server_bin": llama_server_bin or "",
        "llama_server_online": llama_server_online,
        "llama_server_models": llama_models,
        "models_dir": str(get_models_storage_dir()),
        "downloaded_gguf": [asdict(m) | {"path": str(m.path)} for m in list_downloaded_models() if not m.is_partial],
    }


class LlamaCppLocalBackend:
    """Thread-safe in-process GGUF runner powered by `llama-cpp-python` when installed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._llm: Any = None
        self._loaded_path: str = ""

    @staticmethod
    def is_available() -> bool:
        return importlib.util.find_spec("llama_cpp") is not None

    def loaded_model_path(self) -> str:
        with self._lock:
            return self._loaded_path

    def unload(self) -> None:
        with self._lock:
            self._llm = None
            self._loaded_path = ""

    def load_model(self, gguf_path: Path, n_ctx: int = 8192) -> Tuple[bool, str]:
        path = Path(gguf_path)
        if not path.exists():
            return False, f"Không tìm thấy file model GGUF: {path}"
        if not verify_gguf_magic(path):
            return False, f"File {path.name} không đúng định dạng GGUF."
        if not self.is_available():
            return (
                False,
                "Thiếu thư viện `llama-cpp-python` để chạy trực tiếp file .gguf trong app. "
                "Bạn có thể cài `pip install llama-cpp-python` hoặc nạp file .gguf này vào LM Studio / Ollama / llama-server.",
            )

        with self._lock:
            resolved = str(path.resolve())
            if self._llm is not None and self._loaded_path == resolved:
                return True, f"Đã sẵn sàng model GGUF: {path.name}"
            try:
                from llama_cpp import Llama  # type: ignore

                cpu_threads = max(2, min(12, (os.cpu_count() or 4) - 1))
                # Try GPU offload first if compiled with CUDA/Vulkan/Metal, gracefully fall back to CPU AVX2
                loaded = None
                for gpu_layers in (-1, 0):
                    for ctx_try in (int(n_ctx), 4096):
                        try:
                            loaded = Llama(
                                model_path=resolved,
                                n_ctx=ctx_try,
                                n_threads=cpu_threads,
                                n_gpu_layers=gpu_layers,
                                verbose=False,
                            )
                            break
                        except Exception:
                            loaded = None
                    if loaded is not None:
                        break
                if loaded is None:
                    raise RuntimeError("Không thể khởi tạo ngữ cảnh GGUF trên GPU hoặc CPU.")
                self._llm = loaded
                self._loaded_path = resolved
                return True, f"Đã nạp model offline: {path.name}"
            except Exception as exc:
                self._llm = None
                self._loaded_path = ""
                return False, f"Lỗi khởi tạo llama-cpp-python: {exc}"

    def stream_chat(
        self,
        gguf_path: Path,
        messages: List[Dict[str, str]],
        max_tokens: int = 4096,
        temperature: float = 0.25,
        on_token: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Tuple[bool, str, str]:
        ok, status_msg = self.load_model(gguf_path)
        if not ok or self._llm is None:
            return False, "", status_msg

        with self._lock:
            try:
                stream_iter = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=max(256, int(max_tokens)),
                    temperature=float(temperature),
                    stream=True,
                )
                chunks: List[str] = []
                for chunk in stream_iter:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or ""
                    if piece:
                        chunks.append(piece)
                        if on_token is not None:
                            on_token(piece)
                full_text = "".join(chunks).strip()
                if full_text:
                    return True, full_text, ""
                return False, "", "Model GGUF trả về phản hồi rỗng."
            except Exception as exc:
                return False, "", f"Lỗi suy luận llama-cpp-python: {exc}"


class ModelDownloadWorker(QThread):
    """Background worker thread running `ResumableModelDownloader` without blocking GUI."""

    progress_changed = Signal(object)
    download_finished = Signal(bool, str, str)  # ok, file_path_or_err, sha256

    def __init__(
        self,
        url: str,
        filename: Optional[str] = None,
        expected_sha256: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.url = url
        self.filename = filename
        self.expected_sha256 = expected_sha256
        self.cancel_event = threading.Event()
        self._downloader = ResumableModelDownloader()

    def request_pause(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            final_path, sha_hex = self._downloader.download(
                url=self.url,
                dest_dir=get_models_storage_dir(),
                filename=self.filename,
                expected_sha256=self.expected_sha256,
                verify_gguf=True,
                progress_cb=lambda prog: self.progress_changed.emit(prog),
                cancel_event=self.cancel_event,
            )
            self.download_finished.emit(True, str(final_path), sha_hex)
        except InterruptedError as exc:
            self.download_finished.emit(False, str(exc), "")
        except Exception as exc:
            self.download_finished.emit(False, f"Tải model thất bại: {exc}", "")


class LocalAiManagerDialog(QDialog):
    """PySide6 Control Center for Local AI GGUF models, downloads & Hybrid routing."""

    settings_applied = Signal(dict)

    def __init__(
        self,
        current_routing_mode: str = "hybrid",
        active_gguf_path: str = "",
        local_provider: str = "Ollama",
        local_model: str = "qwen2.5:3b",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Quản Lý AI Local & Kho Model GGUF (AirTeach AI)")
        self.resize(900, 720)
        self.setMinimumSize(780, 600)

        self._routing_mode = current_routing_mode
        self._active_gguf_path = active_gguf_path
        self._local_provider = local_provider
        self._local_model = local_model
        self._download_worker: Optional[ModelDownloadWorker] = None

        self.setStyleSheet(
            """
            QDialog {
                background-color: #0F172A;
                color: #F8FAFC;
                font-family: "Segoe UI", "Inter", sans-serif;
            }
            QLabel { color: #F8FAFC; }
            QFrame#Card {
                background-color: #1E293B;
                border: 1px solid #334155;
                border-radius: 12px;
            }
            QPushButton {
                background-color: #334155;
                color: #F8FAFC;
                border: 1px solid #475569;
                border-radius: 8px;
                padding: 7px 12px;
                font-weight: 600;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #475569;
                border-color: #38BDF8;
            }
            QPushButton#PrimaryBtn {
                background-color: #4F46E5;
                border-color: #6366F1;
                color: #FFFFFF;
            }
            QPushButton#PrimaryBtn:hover {
                background-color: #4338CA;
            }
            QPushButton#DangerBtn {
                background-color: #7F1D1D;
                border-color: #EF4444;
                color: #FEE2E2;
            }
            QLineEdit {
                background-color: #0B1220;
                color: #F8FAFC;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 7px 10px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #06B6D4;
            }
            QRadioButton {
                color: #E2E8F0;
                font-weight: 600;
                font-size: 12px;
                spacing: 6px;
            }
            QProgressBar {
                background-color: #0B1220;
                border: 1px solid #334155;
                border-radius: 7px;
                height: 16px;
                text-align: center;
                color: #F8FAFC;
                font-size: 11px;
                font-weight: 700;
            }
            QProgressBar::chunk {
                background-color: #06B6D4;
                border-radius: 6px;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            """
        )

        self._build_ui()
        self._refresh_runtime_badges()
        self._refresh_downloaded_list()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        # Header
        hdr = QHBoxLayout()
        title_box = QVBoxLayout()
        title_lbl = QLabel("Quản Lý AI Local Offline & Kho Model GGUF")
        title_lbl.setFont(QFont("Segoe UI", 15, QFont.Bold))
        sub_lbl = QLabel(f"Thư mục lưu trữ: {get_models_storage_dir()}")
        sub_lbl.setStyleSheet("color: #94A3B8; font-size: 11px;")
        title_box.addWidget(title_lbl)
        title_box.addWidget(sub_lbl)
        hdr.addLayout(title_box, 1)

        open_dir_btn = QPushButton("Mở Thư Mục Model")
        open_dir_btn.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(get_models_storage_dir())))
        )
        rescan_btn = QPushButton("Quét Lại Runtime")
        rescan_btn.clicked.connect(self._on_rescan_clicked)
        hdr.addWidget(open_dir_btn)
        hdr.addWidget(rescan_btn)
        root.addLayout(hdr)

        # Hybrid AI Mode Card
        mode_card = QFrame()
        mode_card.setObjectName("Card")
        mode_lay = QVBoxLayout(mode_card)
        mode_lay.setContentsMargins(14, 12, 14, 12)
        mode_lay.setSpacing(8)
        mode_title = QLabel("1. Chế Độ Vận Hành AI (Hybrid Toggle)")
        mode_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        mode_lay.addWidget(mode_title)

        radio_row = QHBoxLayout()
        self._mode_group = QButtonGroup(self)
        self._rb_local = QRadioButton("Local AI (100% Offline)")
        self._rb_cloud = QRadioButton("Gemini Cloud AI")
        self._rb_hybrid = QRadioButton("Auto Hybrid (Cloud + Local Fallback)")
        self._mode_group.addButton(self._rb_local, 0)
        self._mode_group.addButton(self._rb_cloud, 1)
        self._mode_group.addButton(self._rb_hybrid, 2)
        if self._routing_mode == "local_only":
            self._rb_local.setChecked(True)
        elif self._routing_mode == "cloud_only":
            self._rb_cloud.setChecked(True)
        else:
            self._rb_hybrid.setChecked(True)
        self._mode_group.buttonClicked.connect(lambda _: self._emit_applied_settings())
        radio_row.addWidget(self._rb_local)
        radio_row.addWidget(self._rb_cloud)
        radio_row.addWidget(self._rb_hybrid)
        radio_row.addStretch(1)
        mode_lay.addLayout(radio_row)

        self._runtime_status_lbl = QLabel("Đang kiểm tra các engine AI Local...")
        self._runtime_status_lbl.setWordWrap(True)
        self._runtime_status_lbl.setStyleSheet("color: #94A3B8; font-size: 11px;")
        mode_lay.addWidget(self._runtime_status_lbl)
        root.addWidget(mode_card)

        # Download Progress Card (always visible so teacher sees current status clearly)
        dl_card = QFrame()
        dl_card.setObjectName("Card")
        dl_lay = QVBoxLayout(dl_card)
        dl_lay.setContentsMargins(14, 10, 14, 10)
        dl_lay.setSpacing(6)

        dl_top = QHBoxLayout()
        self._dl_status_lbl = QLabel("Sẵn sàng tải model GGUF chính thức từ HuggingFace / GitHub (hỗ trợ tải tiếp khi gián đoạn).")
        self._dl_status_lbl.setStyleSheet("color: #CBD5E1; font-size: 12px;")
        self._pause_dl_btn = QPushButton("Tạm Dừng")
        self._pause_dl_btn.setEnabled(False)
        self._pause_dl_btn.clicked.connect(self._on_pause_download)
        self._cancel_dl_btn = QPushButton("Hủy Tải")
        self._cancel_dl_btn.setObjectName("DangerBtn")
        self._cancel_dl_btn.setEnabled(False)
        self._cancel_dl_btn.setToolTip("Dừng hẳn và xóa tệp .part đang tải dở")
        self._cancel_dl_btn.clicked.connect(self._on_cancel_download)
        dl_top.addWidget(self._dl_status_lbl, 1)
        dl_top.addWidget(self._pause_dl_btn)
        dl_top.addWidget(self._cancel_dl_btn)
        dl_lay.addLayout(dl_top)

        self._dl_progress = QProgressBar()
        self._dl_progress.setRange(0, 100)
        self._dl_progress.setValue(0)
        self._dl_progress.setFormat("0% • Chưa có tác vụ tải")
        dl_lay.addWidget(self._dl_progress)
        root.addWidget(dl_card)

        # Scrollable Area for Downloaded Models + Catalog + Custom URL
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_inner = QWidget()
        scroll_lay = QVBoxLayout(scroll_inner)
        scroll_lay.setContentsMargins(0, 0, 6, 0)
        scroll_lay.setSpacing(12)

        # Section 2: Downloaded Models on Disk
        own_hdr = QLabel("2. Model Đã Tải Về Máy")
        own_hdr.setFont(QFont("Segoe UI", 12, QFont.Bold))
        scroll_lay.addWidget(own_hdr)

        self._downloaded_container = QVBoxLayout()
        self._downloaded_container.setSpacing(6)
        scroll_lay.addLayout(self._downloaded_container)

        # Section 3: Custom URL Downloader
        custom_card = QFrame()
        custom_card.setObjectName("Card")
        custom_lay = QVBoxLayout(custom_card)
        custom_lay.setContentsMargins(14, 12, 14, 12)
        custom_lay.setSpacing(8)
        custom_title = QLabel("3. Tải Model Từ Link GitHub / HuggingFace Bất Kỳ (.gguf)")
        custom_title.setFont(QFont("Segoe UI", 12, QFont.Bold))
        custom_lay.addWidget(custom_title)

        url_row = QHBoxLayout()
        self._custom_url_edit = QLineEdit()
        self._custom_url_edit.setPlaceholderText("Dán link HuggingFace hoặc GitHub Release (.gguf), ví dụ: https://huggingface.co/.../model.gguf")
        self._custom_sha_edit = QLineEdit()
        self._custom_sha_edit.setPlaceholderText("SHA-256 (tùy chọn)")
        self._custom_sha_edit.setFixedWidth(185)
        self._custom_dl_btn = QPushButton("Tải Link")
        self._custom_dl_btn.setObjectName("PrimaryBtn")
        self._custom_dl_btn.clicked.connect(self._on_download_custom_url)
        url_row.addWidget(self._custom_url_edit, 1)
        url_row.addWidget(self._custom_sha_edit)
        url_row.addWidget(self._custom_dl_btn)
        custom_lay.addLayout(url_row)
        scroll_lay.addWidget(custom_card)

        # Section 4: Curated 1-Click Catalog
        cat_hdr = QLabel("4. Kho Model GGUF Khuyên Dùng (1-Click Tải Về)")
        cat_hdr.setFont(QFont("Segoe UI", 12, QFont.Bold))
        scroll_lay.addWidget(cat_hdr)

        for spec in CURATED_GGUF_CATALOG:
            scroll_lay.addWidget(self._build_catalog_card(spec))

        scroll_lay.addStretch(1)
        scroll.setWidget(scroll_inner)
        root.addWidget(scroll, 1)

        # Footer
        foot = QHBoxLayout()
        foot.addStretch(1)
        close_btn = QPushButton("Áp Dụng && Đóng")
        close_btn.setObjectName("PrimaryBtn")
        close_btn.clicked.connect(self._on_apply_and_close)
        foot.addWidget(close_btn)
        root.addLayout(foot)

    def _build_catalog_card(self, spec: GgufModelSpec) -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(12)

        info = QVBoxLayout()
        badge = " • Khuyên dùng" if spec.recommended else ""
        title = QLabel(f"{spec.name}{badge}")
        title.setFont(QFont("Segoe UI", 11, QFont.Bold))
        meta = QLabel(f"{spec.provider}  |  {spec.size_label}  |  {spec.ram_requirement}  |  {spec.languages}")
        meta.setStyleSheet("color: #38BDF8; font-size: 11px;")
        desc = QLabel(spec.description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94A3B8; font-size: 11px;")
        info.addWidget(title)
        info.addWidget(meta)
        info.addWidget(desc)
        lay.addLayout(info, 1)

        dl_btn = QPushButton(f"Tải ({spec.size_label})")
        dl_btn.setObjectName("PrimaryBtn")
        dl_btn.setMinimumWidth(130)
        dl_btn.clicked.connect(lambda _=False, s=spec: self._start_download(s.download_url, s.filename, s.sha256))
        lay.addWidget(dl_btn)
        return card

    def _clear_layout(self, layout: QVBoxLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _refresh_runtime_badges(self) -> None:
        info = detect_local_ai_runtimes()
        parts: List[str] = []
        parts.append(f"llama-cpp-python: {'Sẵn sàng' if info['llama_cpp_python'] else 'Chưa cài'}")
        parts.append(f"ONNX Runtime: {'Sẵn sàng' if info['onnxruntime'] else 'Chưa cài'}")
        parts.append(f"Ollama: {'Đang chạy' if info['ollama_online'] else ('Đã cài (chưa chạy)' if info['ollama_installed'] else 'Chưa cài')}")
        parts.append(f"LM Studio: {'Đang chạy (1234)' if info['lm_studio_online'] else 'Chưa mở'}")
        parts.append(f"llama-server: {'Đang chạy (8080)' if info['llama_server_online'] else ('Có CLI' if info['llama_server_bin'] else 'Chưa mở')}")
        self._runtime_status_lbl.setText("Runtime Local: " + "  |  ".join(parts))

    def _refresh_downloaded_list(self) -> None:
        self._clear_layout(self._downloaded_container)
        models = list_downloaded_models()
        if not models:
            empty_card = QFrame()
            empty_card.setObjectName("Card")
            lay = QHBoxLayout(empty_card)
            lay.setContentsMargins(14, 10, 14, 10)
            lbl = QLabel("Chưa có file .gguf nào trong thư mục models. Chọn một model bên dưới để tải về dùng Offline.")
            lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
            lay.addWidget(lbl)
            self._downloaded_container.addWidget(empty_card)
            return

        for m in models:
            card = QFrame()
            card.setObjectName("Card")
            lay = QHBoxLayout(card)
            lay.setContentsMargins(14, 8, 14, 8)
            lay.setSpacing(10)

            info = QVBoxLayout()
            is_active = str(m.path) == str(self._active_gguf_path) and not m.is_partial
            status_txt = "Đang kích hoạt" if is_active else ("Tải dở (.part)" if m.is_partial else ("GGUF hợp lệ" if m.valid_gguf else "Lỗi header"))
            title = QLabel(f"{m.filename}  [{m.size_label}]  —  {status_txt}")
            title.setFont(QFont("Segoe UI", 11, QFont.Bold))
            if is_active:
                title.setStyleSheet("color: #00F5D4;")
            sub = QLabel(
                f"SHA-256: {m.sha256[:24]}..." if m.sha256 else "Nhấn 'Tải Tiếp' từ kho hoặc link nếu là file .part"
            )
            sub.setStyleSheet("color: #94A3B8; font-size: 11px;")
            info.addWidget(title)
            info.addWidget(sub)
            lay.addLayout(info, 1)

            if not m.is_partial:
                use_btn = QPushButton("Đang Dùng" if is_active else "Kích Hoạt")
                if not is_active:
                    use_btn.setObjectName("PrimaryBtn")
                use_btn.clicked.connect(lambda _=False, p=str(m.path): self._activate_gguf(p))
                lay.addWidget(use_btn)

            del_btn = QPushButton("Xóa")
            del_btn.setObjectName("DangerBtn")
            del_btn.clicked.connect(lambda _=False, p=m.path: self._delete_model_file(p))
            lay.addWidget(del_btn)
            self._downloaded_container.addWidget(card)

    def _on_rescan_clicked(self) -> None:
        self._refresh_runtime_badges()
        self._refresh_downloaded_list()

    def _on_download_custom_url(self) -> None:
        raw_url = self._custom_url_edit.text().strip()
        sha = self._custom_sha_edit.text().strip()
        try:
            norm = normalize_model_download_url(raw_url)
            fname = infer_filename_from_url(norm)
        except Exception as exc:
            QMessageBox.warning(self, "Link Không Hợp Lệ", str(exc))
            return
        self._start_download(norm, fname, sha)

    def _start_download(self, url: str, filename: str, expected_sha256: str = "") -> None:
        if self._download_worker is not None and self._download_worker.isRunning():
            QMessageBox.information(self, "Đang Tải Model", "Vui lòng đợi tác vụ hiện tại hoàn tất hoặc nhấn Tạm Dừng.")
            return
        self._dl_progress.setValue(0)
        self._dl_progress.setFormat(f"Đang kết nối tới {filename}...")
        self._dl_status_lbl.setText(f"Chuẩn bị tải: {filename}")
        self._pause_dl_btn.setEnabled(True)
        self._cancel_dl_btn.setEnabled(True)
        self._active_dl_filename = filename
        self._cancel_delete_part = False

        worker = ModelDownloadWorker(url=url, filename=filename, expected_sha256=expected_sha256, parent=self)
        worker.progress_changed.connect(self._on_download_progress)
        worker.download_finished.connect(self._on_download_finished)
        self._download_worker = worker
        worker.start()

    def _on_pause_download(self) -> None:
        if self._download_worker is not None and self._download_worker.isRunning():
            self._cancel_delete_part = False
            self._download_worker.request_pause()
            self._dl_status_lbl.setText("Đang tạm dừng tải (dữ liệu đã lưu vào file .part để tải tiếp sau)...")

    def _on_cancel_download(self) -> None:
        if self._download_worker is not None and self._download_worker.isRunning():
            self._cancel_delete_part = True
            self._download_worker.request_pause()
            self._dl_status_lbl.setText("Đang hủy tác vụ tải và dọn file tạm...")

    def _on_download_progress(self, prog: DownloadProgress) -> None:
        self._dl_progress.setValue(int(round(prog.percent)))
        total_mb = prog.total_bytes / (1024.0 * 1024.0) if prog.total_bytes > 0 else 0.0
        done_mb = prog.downloaded_bytes / (1024.0 * 1024.0)
        if total_mb > 0:
            self._dl_progress.setFormat(f"{prog.percent:.1f}% • {done_mb:.1f} / {total_mb:.1f} MB • {prog.speed_mb_s:.2f} MB/s")
        else:
            self._dl_progress.setFormat(f"{done_mb:.1f} MB • {prog.speed_mb_s:.2f} MB/s")
        self._dl_status_lbl.setText(f"{prog.filename}: {prog.status}")

    def _on_download_finished(self, ok: bool, path_or_err: str, sha_hex: str) -> None:
        self._pause_dl_btn.setEnabled(False)
        self._cancel_dl_btn.setEnabled(False)
        if not ok and getattr(self, "_cancel_delete_part", False):
            fname = getattr(self, "_active_dl_filename", "")
            if fname:
                part_file = get_models_storage_dir() / f"{fname}.part"
                part_file.unlink(missing_ok=True)
            self._dl_progress.setValue(0)
            self._dl_progress.setFormat("0% • Đã hủy tải")
            self._dl_status_lbl.setText("Đã hủy tải và xóa tệp .part tạm.")
            self._cancel_delete_part = False
            self._refresh_downloaded_list()
            return
        self._refresh_downloaded_list()
        if ok:
            self._dl_progress.setValue(100)
            self._dl_status_lbl.setText(f"Đã tải xong: {Path(path_or_err).name} (SHA-256: {sha_hex[:16]}...)")
            self._activate_gguf(path_or_err)
        else:
            self._dl_status_lbl.setText(path_or_err)

    def _activate_gguf(self, gguf_path: str) -> None:
        self._active_gguf_path = gguf_path
        self._local_provider = "GGUF (In-Process)"
        self._local_model = Path(gguf_path).name
        self._refresh_downloaded_list()
        self._emit_applied_settings()

    def _delete_model_file(self, path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
            sidecar = Path(f"{path}.sha256")
            sidecar.unlink(missing_ok=True)
            if str(path) == self._active_gguf_path:
                self._active_gguf_path = ""
        except OSError as exc:
            QMessageBox.warning(self, "Lỗi Xóa File", str(exc))
        self._refresh_downloaded_list()
        self._emit_applied_settings()

    def current_routing_mode(self) -> str:
        if self._rb_local.isChecked():
            return "local_only"
        if self._rb_cloud.isChecked():
            return "cloud_only"
        return "hybrid"

    def _emit_applied_settings(self) -> None:
        self.settings_applied.emit(
            {
                "routing_mode": self.current_routing_mode(),
                "active_gguf_path": self._active_gguf_path,
                "local_provider": self._local_provider,
                "local_model": self._local_model,
            }
        )

    def _on_apply_and_close(self) -> None:
        self._emit_applied_settings()
        self.accept()
