"""
AirTeach AI - Gemini AI Assistant Service
==========================================
Provides AI-powered teaching assistance via Google Gemini API.
Features per-teacher API key and configuration isolation,
contextual lesson chat, multimodal image analysis, lesson planning,
quiz generation, document summarization, and offline fallback.

Offline fallback: local knowledge base when API is unavailable.
Zero secret exposure: all outputs pass through credential sanitization.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Generator, List, Optional

from src.services.credential_service import TeacherAISettings, TeacherCredentialService

logger = logging.getLogger(__name__)

# ── Local offline knowledge base ──────────────────────────────────────────────
_OFFLINE_GUIDE: dict[str, str] = {
    "point": (
        "Cử chỉ POINT (Laser Pointer): Duỗi thẳng ngón trỏ, ngón cái giơ rộng ra xa. "
        "Con trỏ laser màu đỏ sẽ xuất hiện trên màn hình chiếu."
    ),
    "draw": (
        "Cử chỉ DRAW (Vẽ/Viết): Duỗi ngón trỏ để vẽ nét tự nhiên (Chế độ Air Pen), "
        "hoặc chụm ngón trỏ với ngón cái (Chế độ Precision Pinch)."
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
        "Dùng để xác nhận hành động (lưu, đăng nhập, nắn hình, v.v.)"
    ),
    "calibrate": (
        "Hiệu chỉnh camera: Vào menu Cài đặt > Hiệu chỉnh cử chỉ hoặc Hiệu chỉnh camera. "
        "Giữ tay tại các góc hoặc thực hiện cử chỉ theo hướng dẫn của Calibration Wizard."
    ),
    "shape": (
        "Công cụ làm nét tự động (Shape Snapping): Sau khi vẽ xong (cử chỉ PAUSE), "
        "hệ thống phân tích hình dạng và tự động nắn thành: Đường thẳng, Mũi tên, "
        "Hình tròn/Elip, Hình chữ nhật, Tam giác."
    ),
    "ocr": (
        "Nhận diện chữ viết tay: Khoanh vùng nét chữ bằng cử chỉ SELECT, "
        "xác nhận bằng 👍 hoặc nút Review. Hệ thống nhận diện và chuyển thành văn bản/công thức LaTeX."
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
    Gemini API assistant with streaming support, per-teacher isolation,
    and offline fallback.

    Args:
        api_key:            Google AI Studio API key (optional if using credential_service).
        model_name:         Gemini model to use (default: gemini-2.5-flash).
        teacher_id:         Current active teacher profile ID.
        credential_service: TeacherCredentialService instance for per-teacher key management.
    """

    SYSTEM_PROMPT = (
        "Bạn là AirTeach AI Assistant — trợ lý giảng dạy thông minh tích hợp trong phần mềm "
        "AirTeach AI. Bạn hỗ trợ giáo viên Việt Nam giảng dạy hiệu quả hơn bằng cách:\n"
        "- Giải thích nội dung chuyên môn theo ngữ cảnh bài học đang mở.\n"
        "- Soạn giáo án, dàn ý bài giảng, tóm tắt bài học ngắn gọn và dễ hiểu.\n"
        "- Tạo câu hỏi trắc nghiệm, bài tập ôn luyện theo các mức độ nhận thức (Nhận biết, Thông hiểu, Vận dụng, Vận dụng cao).\n"
        "- Gợi ý cách dẫn dắt, ví dụ minh hoạ thực tế, câu hỏi tương tác trên lớp.\n"
        "- Hỗ trợ kiểm tra bài giải công thức toán học/vật lý/hoá học.\n"
        "- Giải đáp cách sử dụng phần mềm AirTeach AI.\n"
        "Luôn trả lời bằng tiếng Việt chuẩn mực, sư phạm, súc tích và thực tiễn."
    )

    def __init__(
        self,
        api_key: str = "",
        model_name: str = "gemini-2.5-flash",
        teacher_id: str = "",
        credential_service: Optional[TeacherCredentialService] = None,
    ) -> None:
        self._teacher_id = teacher_id
        self._cred_service = credential_service or TeacherCredentialService()
        self._model_name = model_name
        self._client = None
        self._online = False

        # Load key from credential service if teacher_id provided, else use passed api_key
        if self._teacher_id:
            stored_key = self._cred_service.get_api_key(self._teacher_id)
            settings = self._cred_service.get_settings(self._teacher_id)
            self._api_key = stored_key or api_key
            self._model_name = settings.model_name or model_name
        else:
            self._api_key = api_key

        self._init_client()

    def set_active_teacher(self, teacher_id: str) -> None:
        """
        Switch active teacher account: reloads that teacher's isolated API key & settings.
        Teacher A and Teacher B never share keys.
        """
        self._teacher_id = teacher_id
        self._client = None
        self._online = False

        if teacher_id:
            self._api_key = self._cred_service.get_api_key(teacher_id)
            settings = self._cred_service.get_settings(teacher_id)
            if settings and settings.model_name:
                self._model_name = settings.model_name
            logger.info("GeminiAssistant switched to teacher: %s (has_key=%s)", teacher_id, bool(self._api_key))
        else:
            self._api_key = ""

        self._init_client()

    def get_teacher_id(self) -> str:
        return self._teacher_id

    def _init_client(self) -> None:
        """Attempt to initialise the Gemini client with the current key."""
        if not self._api_key:
            logger.info("No Gemini API key for current teacher — running in offline pedagogical mode.")
            self._client = None
            self._online = False
            return

        try:
            from google import genai  # noqa: PLC0415
            self._client = genai.Client(api_key=self._api_key)
            self._online = True
            logger.info("Gemini client initialised for model=%s.", self._model_name)
        except Exception as exc:
            redacted_err = TeacherCredentialService.redact_text(str(exc))
            logger.warning("Gemini client init failed: %s — fallback to offline.", redacted_err)
            self._client = None
            self._online = False

    def update_api_key(self, new_key: str) -> bool:
        """
        Hot-update the API key for the current teacher.
        Saves key into secure OS Keyring under teacher namespace.
        """
        clean_key = new_key.strip()
        self._api_key = clean_key
        if self._teacher_id:
            self._cred_service.set_api_key(self._teacher_id, clean_key)

        self._client = None
        self._online = False
        self._init_client()
        return self._online

    def update_model_name(self, model_name: str) -> None:
        """Update active model (e.g. gemini-2.5-flash, gemini-1.5-pro)."""
        self._model_name = model_name
        if self._teacher_id:
            settings = self._cred_service.get_settings(self._teacher_id)
            settings.model_name = model_name
            self._cred_service.save_settings(settings)
        # Re-initialize client if necessary
        self._init_client()

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
            redacted_err = TeacherCredentialService.redact_text(str(exc))
            logger.warning("Gemini streaming failed: %s", redacted_err)
            raise

        return full_text

    # ── Public API: Chat & Teaching Assistance ─────────────────────────────────

    def chat_with_teacher(
        self,
        history: list[dict],
        user_message: str,
        current_page_text: str = "",
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Contextual lesson chat.
        """
        if not self._online or self._client is None:
            fallback = (
                "Không thể kết nối Gemini API — đang ở chế độ ngoại tuyến.\n"
                "Vui lòng kiểm tra lại cấu hình API Key trong mục Cài đặt > AI & API.\n\n"
                + _offline_lookup(user_message)
            )
            if on_chunk:
                on_chunk(fallback)
            return fallback

        try:
            from google.genai import types  # noqa: PLC0415

            contents = []
            for turn in history:
                role = turn.get("role", "user")
                text = turn.get("text", "")
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part(text=text)],
                ))

            context_prefix = ""
            if current_page_text.strip():
                context_prefix = (
                    f"[Nội dung bài giảng/slide đang mở]:\n{current_page_text[:2000]}\n\n"
                )
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=context_prefix + user_message)],
            ))

            return self._stream_generate(contents, on_chunk=on_chunk)

        except Exception as exc:
            redacted_err = TeacherCredentialService.redact_text(str(exc))
            err_msg = (
                f"Không thể xác thực API hoặc mất kết nối mạng. "
                f"Hệ thống tự động chuyển sang Trợ lý Hướng dẫn Offline nội bộ.\n\n"
                f"(Chi tiết: {redacted_err})\n\n"
                + _offline_lookup(user_message)
            )
            logger.warning("chat_with_teacher failed: %s", redacted_err)
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
        Send a cropped canvas region image to Gemini Multimodal for pedagogical explanation.
        """
        if not self._online or self._client is None:
            fallback = (
                "Chế độ ngoại tuyến: Không thể kết nối Gemini API để phân tích hình ảnh.\n"
                "Vui lòng thiết lập khóa API trong Cài đặt > AI & API."
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
            redacted_err = TeacherCredentialService.redact_text(str(exc))
            err_msg = (
                f"Lỗi khi gửi phân tích hình ảnh tới AI.\n"
                f"(Chi tiết: {redacted_err})"
            )
            logger.warning("explain_canvas_region failed: %s", redacted_err)
            if on_chunk:
                on_chunk(err_msg)
            return err_msg

    def get_interactive_usage_guide(
        self,
        query: str,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Answer questions about AirTeach AI features, gestures, calibration.
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
            redacted_err = TeacherCredentialService.redact_text(str(exc))
            logger.warning("get_interactive_usage_guide failed: %s — fallback to offline.", redacted_err)
            if on_chunk:
                on_chunk(answer)
            return answer

    # ── AI Workbench Pedagogical Operations (Part VIII) ────────────────────────

    def generate_lesson_outline(
        self,
        subject: str,
        topic: str,
        grade_level: str = "THPT",
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate a structured lesson outline with objectives, steps, and key questions."""
        prompt = (
            f"Hãy lập dàn ý bài dạy chi tiết cho môn {subject}, bài học/chủ đề: '{topic}', "
            f"đối tượng học sinh: {grade_level}.\n\n"
            f"Dàn ý cần bao gồm:\n"
            f"1. Mục tiêu bài học (Kiến thức, Kỹ năng, Phẩm chất)\n"
            f"2. Khởi động / Dẫn dắt vào bài (3-5 phút)\n"
            f"3. Các hoạt động hình thành kiến thức trọng tâm\n"
            f"4. Hoạt động luyện tập & củng cố\n"
            f"5. Vận dụng thực tế & mở rộng"
        )
        return self._run_prompt_task(prompt, on_chunk=on_chunk)

    def summarize_document_content(
        self,
        text_content: str,
        max_points: int = 5,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Summarize selected lesson document or slide content into concise bullet points."""
        prompt = (
            f"Hãy tóm tắt ngắn gọn nội dung văn bản sau thành tối đa {max_points} ý chính "
            f"thích hợp để giáo viên ghi bảng hoặc trình chiếu:\n\n"
            f"---\n{text_content[:4000]}\n---"
        )
        return self._run_prompt_task(prompt, on_chunk=on_chunk)

    def generate_quiz_questions(
        self,
        content: str,
        num_questions: int = 4,
        difficulty: str = "Đa cấp độ",
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate multiple choice or short-answer review questions with answer key."""
        prompt = (
            f"Dựa vào nội dung kiến thức sau, hãy soạn {num_questions} câu hỏi trắc nghiệm ôn tập "
            f"(độ khó: {difficulty}) kèm theo 4 phương án A, B, C, D, chỉ rõ đáp án đúng và lời giải thích ngắn gọn:\n\n"
            f"---\n{content[:4000]}\n---"
        )
        return self._run_prompt_task(prompt, on_chunk=on_chunk)

    def explain_concept(
        self,
        concept: str,
        target_audience: str = "Học sinh THPT",
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Explain a complex STEM concept with real-world analogies and visual cues."""
        prompt = (
            f"Hãy giải thích khái niệm '{concept}' một cách trực quan, dễ nhớ cho đối tượng {target_audience}.\n"
            f"Bao gồm: định nghĩa ngắn gọn, hình ảnh/ví dụ minh họa thực tế trong cuộc sống, "
            f"và một câu hỏi tương tác để giáo viên hỏi học sinh ngay trên lớp."
        )
        return self._run_prompt_task(prompt, on_chunk=on_chunk)

    def generate_lesson_plan(
        self,
        topic: str,
        duration_minutes: int = 45,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate a complete 45-minute lesson plan according to pedagogical standards."""
        prompt = (
            f"Hãy soạn một kế hoạch bài dạy (giáo án) chuẩn thời lượng {duration_minutes} phút "
            f"cho chủ đề: '{topic}'. Phân bổ thời gian cụ thể cho từng hoạt động dạy học của giáo viên và học sinh."
        )
        return self._run_prompt_task(prompt, on_chunk=on_chunk)

    def _run_prompt_task(self, prompt: str, on_chunk: Optional[Callable[[str], None]] = None) -> str:
        """Internal runner for workbench generation tasks."""
        if not self._online or self._client is None:
            msg = (
                "⚠️ Đang ở chế độ ngoại tuyến: Không thể gọi Gemini API.\n"
                "Vui lòng vào menu Cài đặt > AI & API để cấu hình khóa API riêng cho giáo viên."
            )
            if on_chunk:
                on_chunk(msg)
            return msg

        try:
            from google.genai import types  # noqa: PLC0415
            contents = [types.Content(role="user", parts=[types.Part(text=prompt)])]
            return self._stream_generate(contents, on_chunk=on_chunk)
        except Exception as exc:
            redacted_err = TeacherCredentialService.redact_text(str(exc))
            err_msg = f"Lỗi khi thực hiện yêu cầu AI: {redacted_err}"
            logger.warning("AI Workbench prompt task failed: %s", redacted_err)
            if on_chunk:
                on_chunk(err_msg)
            return err_msg

    # ── Status ─────────────────────────────────────────────────────────────────

    def is_online(self) -> bool:
        return self._online

    def model_name(self) -> str:
        return self._model_name
