"""
AirTeach AI - Gemini AI Assistant Service
==========================================
Provides AI-powered teaching assistance via Google Gemini API.

3 core functions (all async via QThread):
  1. chat_with_teacher      – contextual lesson Q&A
  2. explain_canvas_region  – multimodal image explanation
  3. get_usage_guide        – offline gesture/feature help

Offline fallback: local knowledge base when API is unavailable.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Generator, Optional

logger = logging.getLogger(__name__)

# ── Local offline knowledge base ──────────────────────────────────────────────
_OFFLINE_GUIDE: dict[str, str] = {
    "point": (
        "Cử chỉ POINT (Laser Pointer): Duỗi thẳng ngón trỏ, ngón cái giơ rộng ra xa. "
        "Con trỏ laser màu đỏ sẽ xuất hiện trên màn hình chiếu."
    ),
    "draw": (
        "Cử chỉ DRAW (Vẽ/Viết): Duỗi ngón trỏ, chụm ngón cái sát vào đốt ngón trỏ. "
        "Nét vẽ sẽ được ghi lại theo chuyển động ngón trỏ."
    ),
    "pause": (
        "Cử chỉ PAUSE (Nhấc bút): Nắm tay lại (fist). Phần mềm sẽ kết thúc nét vẽ "
        "và tự động làm mịn hình dạng."
    ),
    "select": (
        "Cử chỉ SELECT (Chọn vùng): Giơ đồng thời ngón trỏ và ngón giữa, "
        "ngón áp út và ngón út khép lại. Kéo để khoanh vùng nội dung muốn chọn."
    ),
    "swipe": (
        "Cử chỉ SWIPE (Lật trang): Mở toàn bộ 5 ngón tay, vẫy nhanh sang trái "
        "hoặc phải trong vòng 250ms. Chờ 800ms giữa hai lần lật để tránh lật liên tiếp."
    ),
    "confirm": (
        "Cử chỉ CONFIRM (👍 Xác nhận): Giơ ngón cái thẳng đứng, giữ ít nhất 300ms. "
        "Dùng để xác nhận hành động (lưu, đăng nhập, v.v.)"
    ),
    "calibrate": (
        "Hiệu chỉnh camera: Vào menu Cài đặt > Hiệu chỉnh vùng viết. "
        "Giữ tay tại 4 góc vùng viết theo thứ tự: Trên-Trái → Trên-Phải → Dưới-Phải → Dưới-Trái "
        "và xác nhận bằng cử chỉ 👍."
    ),
    "shape": (
        "Công cụ làm nét tự động (Shape Snapping): Sau khi vẽ xong (cử chỉ PAUSE), "
        "hệ thống phân tích hình dạng và tự động nắn thành: Đường thẳng, Mũi tên, "
        "Hình tròn/Elip, Hình chữ nhật, Tam giác."
    ),
    "ocr": (
        "Nhận diện chữ viết tay: Khoanh vùng nét chữ bằng cử chỉ SELECT, "
        "xác nhận bằng 👍. Hệ thống nhận diện và chuyển thành văn bản/công thức LaTeX."
    ),
    "airteach": (
        "Tệp .airteach: Định dạng gói bài giảng của AirTeach AI gồm ảnh slide, "
        "nét vẽ vector, văn bản nhận diện và liên kết. Dùng menu Tệp > Lưu bài giảng."
    ),
    "default": (
        "AirTeach AI hỗ trợ các cử chỉ: POINT (laser), DRAW (viết), PAUSE (nhấc bút), "
        "SELECT (chọn vùng), SWIPE (lật trang), CONFIRM (👍). "
        "Nhập từ khóa như 'point', 'draw', 'swipe', 'calibrate', 'shape', 'ocr' để biết thêm chi tiết."
    ),
}


def _offline_lookup(query: str) -> str:
    """Find the best matching offline guide entry for *query*."""
    q = query.lower()
    for keyword, answer in _OFFLINE_GUIDE.items():
        if keyword in q:
            return answer
    return _OFFLINE_GUIDE["default"]


# ── Gemini Client Wrapper ─────────────────────────────────────────────────────

class GeminiAssistant:
    """
    Gemini API assistant with streaming support and offline fallback.

    Args:
        api_key:    Google AI Studio API key. If empty, offline mode only.
        model_name: Gemini model to use (default: gemini-2.5-flash).
    """

    SYSTEM_PROMPT = (
        "Bạn là AirTeach AI Assistant — trợ lý giảng dạy thông minh tích hợp trong phần mềm "
        "AirTeach AI. Bạn hỗ trợ giáo viên Việt Nam giảng dạy hiệu quả hơn bằng cách:\n"
        "- Giải thích nội dung chuyên môn theo ngữ cảnh bài học đang mở.\n"
        "- Gợi ý cách dẫn dắt, ví dụ minh hoạ, câu hỏi tương tác.\n"
        "- Hỗ trợ kiểm tra bài giải công thức toán học/vật lý/hoá học.\n"
        "- Giải đáp cách sử dụng phần mềm AirTeach AI.\n"
        "Luôn trả lời bằng tiếng Việt, súc tích và thực tiễn."
    )

    def __init__(self, api_key: str = "", model_name: str = "gemini-2.5-flash") -> None:
        self._api_key = api_key
        self._model_name = model_name
        self._client = None
        self._online = False
        self._init_client()

    def _init_client(self) -> None:
        """Attempt to initialise the Gemini client."""
        if not self._api_key:
            logger.warning("No Gemini API key provided — running in offline mode.")
            return
        try:
            from google import genai  # noqa: PLC0415
            self._client = genai.Client(api_key=self._api_key)
            self._online = True
            logger.info("Gemini client initialised (model=%s).", self._model_name)
        except Exception as exc:
            logger.warning("Gemini client init failed: %s — offline mode.", exc)
            self._client = None
            self._online = False

    def update_api_key(self, new_key: str) -> bool:
        """
        Hot-update the API key from the Settings dialog.

        Returns:
            True if the new key produced a working client.
        """
        self._api_key = new_key
        self._client = None
        self._online = False
        self._init_client()
        return self._online

    # ── Internal streaming helper ──────────────────────────────────────────────

    def _stream_generate(
        self,
        contents: list,
        system_instruction: str = "",
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Stream a Gemini generation request.

        Calls *on_chunk(text)* for each incremental text delta.
        Returns the full concatenated response.
        Falls back to offline mode on any exception.
        """
        from google.genai import types  # noqa: PLC0415

        config = types.GenerateContentConfig(
            system_instruction=system_instruction or self.SYSTEM_PROMPT,
            temperature=0.7,
            max_output_tokens=2048,
        )

        full_text = ""
        try:
            with self._client.models.generate_content_stream(
                model=self._model_name,
                contents=contents,
                config=config,
            ) as stream:
                for chunk in stream:
                    delta = chunk.text or ""
                    full_text += delta
                    if on_chunk and delta:
                        on_chunk(delta)
        except Exception as exc:
            logger.warning("Gemini streaming failed: %s", exc)
            raise

        return full_text

    # ── Public API (3 core functions) ──────────────────────────────────────────

    def chat_with_teacher(
        self,
        history: list[dict],
        user_message: str,
        current_page_text: str = "",
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Contextual lesson chat.

        Args:
            history:           List of {"role": "user"|"model", "text": str} dicts.
            user_message:      Latest message from the teacher.
            current_page_text: Text extracted from the current slide page (context).
            on_chunk:          Streaming callback receiving partial text.

        Returns:
            Full assistant response string.
        """
        if not self._online or self._client is None:
            fallback = (
                "Không thể kết nối Gemini API — đang ở chế độ ngoại tuyến.\n"
                "Vui lòng kiểm tra lại Gemini API Key trong phần Cài đặt.\n\n"
                + _offline_lookup(user_message)
            )
            if on_chunk:
                on_chunk(fallback)
            return fallback

        try:
            from google.genai import types  # noqa: PLC0415

            # Build contents from conversation history
            contents = []
            for turn in history:
                role = turn.get("role", "user")
                text = turn.get("text", "")
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=text)],
                ))

            # Add current page context + new message
            context_prefix = ""
            if current_page_text.strip():
                context_prefix = (
                    f"[Nội dung slide đang mở]:\n{current_page_text[:2000]}\n\n"
                )
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=context_prefix + user_message)],
            ))

            return self._stream_generate(contents, on_chunk=on_chunk)

        except Exception as exc:
            err_msg = (
                f"Không thể xác thực API hoặc đang ngoại tuyến — "
                f"vui lòng kiểm tra lại Gemini API Key trong phần Cài đặt; "
                f"hệ thống chuyển sang Trợ lý Hướng dẫn Offline nội bộ.\n\n"
                f"({type(exc).__name__}: {exc})\n\n"
                + _offline_lookup(user_message)
            )
            logger.warning("chat_with_teacher failed: %s", exc)
            if on_chunk:
                on_chunk(err_msg)
            return err_msg

    def explain_canvas_region(
        self,
        image_bytes: bytes,
        prompt: str = "Hãy giải thích nội dung trong hình này theo ngữ cảnh giảng dạy.",
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Send a cropped canvas region image to Gemini Multimodal for explanation.

        Args:
            image_bytes: PNG/JPEG bytes of the selected canvas region.
            prompt:      Instruction to the model about how to analyse the image.
            on_chunk:    Streaming callback.

        Returns:
            Full assistant response string.
        """
        if not self._online or self._client is None:
            fallback = (
                "Không thể kết nối Gemini API — không thể phân tích hình ảnh.\n"
                "Hãy kiểm tra kết nối mạng và API Key trong Cài đặt."
            )
            if on_chunk:
                on_chunk(fallback)
            return fallback

        try:
            from google.genai import types  # noqa: PLC0415

            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png",
                                data=image_bytes,
                            )
                        ),
                        types.Part(text=prompt),
                    ],
                )
            ]
            return self._stream_generate(contents, on_chunk=on_chunk)

        except Exception as exc:
            err_msg = (
                f"Không thể xác thực API hoặc đang ngoại tuyến — "
                f"hệ thống chuyển sang Trợ lý Hướng dẫn Offline nội bộ.\n"
                f"({type(exc).__name__}: {exc})"
            )
            logger.warning("explain_canvas_region failed: %s", exc)
            if on_chunk:
                on_chunk(err_msg)
            return err_msg

    def get_interactive_usage_guide(
        self,
        query: str,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Answer usage questions about AirTeach AI gestures, camera calibration, etc.

        Always tries online Gemini first; falls back to _OFFLINE_GUIDE immediately.

        Args:
            query:    The user's question about how to use the software.
            on_chunk: Streaming callback.

        Returns:
            Help text string.
        """
        if not self._online or self._client is None:
            answer = _offline_lookup(query)
            if on_chunk:
                on_chunk(answer)
            return answer

        guide_system = (
            self.SYSTEM_PROMPT + "\n\n"
            "Tập trung vào hướng dẫn sử dụng phần mềm AirTeach AI: "
            "cử chỉ tay, hiệu chỉnh camera, làm nét hình học, nhận diện chữ viết và công thức."
        )

        try:
            from google.genai import types  # noqa: PLC0415
            contents = [types.Content(role="user", parts=[types.Part(text=query)])]
            return self._stream_generate(contents, system_instruction=guide_system, on_chunk=on_chunk)

        except Exception as exc:
            answer = _offline_lookup(query)
            logger.warning("get_interactive_usage_guide failed: %s — using offline fallback.", exc)
            if on_chunk:
                on_chunk(answer)
            return answer

    # ── Status ─────────────────────────────────────────────────────────────────

    def is_online(self) -> bool:
        return self._online

    def model_name(self) -> str:
        return self._model_name
