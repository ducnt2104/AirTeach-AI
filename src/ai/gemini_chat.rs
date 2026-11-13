//! Real Multimodal AI Pedagogical Assistant, Speech-to-Text & Board Analyzer (`src/ai/gemini_chat.rs`):
//! - Automatically loads `GEMINI_API_KEY` from `.env`, OS environment variable `GEMINI_API_KEY` / `GOOGLE_API_KEY`, or UI Settings
//! - Calls `gemini-2.5-flash` with automatic seamless fallback to `gemini-2.0-flash`
//! - Sends full Smartboard Canvas PNG (`image/png`) + Recorded Lecture Transcript / Audio (`audio/wav`)
//! - Supports 3 Specialized Pedagogical Modes:
//!   1. `SolveAndCheckError`: Chữa lỗi & Giải Toán/Lý/Hóa/Tin từng bước
//!   2. `SummarizeAndQuiz`: Tóm tắt Bài giảng & Xuất Học liệu (Markdown/HTML) + 3-5 câu hỏi trắc nghiệm
//!   3. `PedagogicalInsights`: Gợi ý Sư phạm & Lưu Lịch sử vào SQLite (`airteach_local.db`)

use crate::ai::subtitles_engine::TranscriptSegmentEvent;
use base64::Engine;
use crossbeam_channel::Sender;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::Path;
use std::time::Duration;

pub const PEDAGOGICAL_INSPECTOR_SYSTEM_PROMPT: &str =
    "Bạn là trợ lý giáo viên chuyên nghiệp của hệ thống AirTeach AI. \
     Hãy phân tích hình ảnh bảng giảng (Smartboard Canvas) và nội dung lời giảng của giáo viên: \
     (1) Nhận diện chính xác nội dung chữ viết tay, công thức Toán/Lý/Hóa/Tin sang chuẩn LaTeX và các sơ đồ hình học; \
     (2) Chỉ ra chính xác từng dòng/bước sai nếu có kèm nhận xét sư phạm nhẹ nhàng, dễ hiểu; \
     (3) Cung cấp lời giải từng bước (step-by-step derivation) đúng chuẩn kèm định dạng LaTeX.";

/// 3 Specialized Pedagogical Modes for Module 4 ("Analyze Board & Lecture").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PedagogicalAnalysisMode {
    SolveAndCheckError,
    SummarizeAndQuiz,
    PedagogicalInsights,
}

impl PedagogicalAnalysisMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::SolveAndCheckError => "1. Chữa lỗi & Giải từng bước (LaTeX)",
            Self::SummarizeAndQuiz => "2. Tóm tắt & Tạo 5 Câu Trắc nghiệm",
            Self::PedagogicalInsights => "3. Gợi ý Sư phạm & Ví dụ Thực tế",
        }
    }

    pub fn prompt_instruction(self, transcript_text: &str) -> String {
        let transcript_block = if transcript_text.trim().is_empty() {
            "(Chưa có bản ghi lời giảng từ Micro)".to_string()
        } else {
            format!("Lời giảng vừa thu âm từ Micro:\n\"\"\"\n{}\n\"\"\"", transcript_text.trim())
        };

        match self {
            Self::SolveAndCheckError => format!(
                "[CHẾ ĐỘ 1: CHỮA LỖI & GIẢI TOÁN/LÝ/HÓA/TIN TỪNG BƯỚC]\n\
                 {transcript_block}\n\n\
                 Yêu cầu:\n\
                 1. Đọc chữ viết tay, công thức và hình vẽ trên bảng hiện tại và chuyển sang chuẩn LaTeX.\n\
                 2. Kiểm tra xem phương trình/bài giải trên bảng có bước nào biến đổi sai dấu, sai điều kiện hay sai công thức không.\n\
                 3. Giải chi tiết từng bước (step-by-step) rõ ràng, chuẩn xác."
            ),
            Self::SummarizeAndQuiz => format!(
                "[CHẾ ĐỘ 2: TÓM TẮT BÀI GIẢNG & XUẤT HỌC LIỆU + CÂU HỎI TRẮC NGHIỆM]\n\
                 {transcript_block}\n\n\
                 Yêu cầu:\n\
                 1. Kết hợp hình ảnh trên bảng Smartboard và lời giảng của giáo viên để tóm tắt các ý chính (Key Concepts & Công thức cốt lõi).\n\
                 2. Tự động biên soạn 4 câu hỏi trắc nghiệm khách quan (4 đáp án A, B, C, D) kèm đáp án đúng và giải thích ngắn gọn để kiểm tra mức độ hiểu bài của học sinh."
            ),
            Self::PedagogicalInsights => format!(
                "[CHẾ ĐỘ 3: GỢI Ý SƯ PHẠM & VÍ DỤ MINH HỌA THỰC TẾ]\n\
                 {transcript_block}\n\n\
                 Yêu cầu:\n\
                 1. Đánh giá cách trình bày trực quan trên bảng và mạch giảng bài hiện tại.\n\
                 2. Đề xuất 2 ví dụ thực tế sinh động hoặc thí nghiệm trực quan giúp học sinh dễ hình dung khái niệm trên bảng hơn.\n\
                 3. Gợi ý 2 câu hỏi mở (Socratic questions) để giáo viên đặt vấn đề cho cả lớp thảo luận."
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubjectAssistantMode {
    Math,
    Physics,
    Chemistry,
    General,
}

impl SubjectAssistantMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Math => "Toán học (Math Solver)",
            Self::Physics => "Vật lý (Physics Mechanics)",
            Self::Chemistry => "Hóa học (Chemistry & PubChem)",
            Self::General => "Trợ giảng Đa năng (RAG Co-Pilot)",
        }
    }

    pub const fn short_badge(self) -> &'static str {
        match self {
            Self::Math => "MATH",
            Self::Physics => "PHYS",
            Self::Chemistry => "CHEM",
            Self::General => "GEN",
        }
    }

    pub const fn system_prompt(self) -> &'static str {
        match self {
            Self::Math => {
                "Bạn là trợ lý giáo viên Toán học chuyên nghiệp. Hãy kiểm tra bài làm/công thức viết tay trong ảnh: \
                 (1) Chuyển đổi chữ viết tay sang LaTeX chính xác; \
                 (2) Kiểm tra từng dòng biến đổi, chỉ ra chính xác bước sai kèm góp ý sư phạm nhẹ nhàng; \
                 (3) Trình bày lời giải đúng từng bước kèm định dạng LaTeX."
            }
            Self::Physics => {
                "Bạn là trợ lý giáo viên Vật lý chuyên nghiệp. Hãy kiểm tra bài làm/sơ đồ lực/công thức viết tay trong ảnh: \
                 (1) Nhận diện nội dung, phương trình chuyển động và đại lượng vật lý sang LaTeX; \
                 (2) Chỉ ra chính xác bước sai về chiều vector lực, dấu hoặc đơn vị SI; \
                 (3) Cung cấp lời giải đúng chuẩn từng bước kèm định dạng LaTeX."
            }
            Self::Chemistry => {
                "Bạn là trợ lý giáo viên Hóa học chuyên nghiệp. Hãy kiểm tra phương trình/cấu trúc phân tử viết tay trong ảnh: \
                 (1) Nhận diện công thức hóa học và phương trình phản ứng sang LaTeX; \
                 (2) Chỉ ra chính xác hệ số cân bằng sai, trạng thái oxi hóa hoặc bước tính mol sai; \
                 (3) Cung cấp phương trình cân bằng và lời giải đúng chuẩn từng bước kèm định dạng LaTeX."
            }
            Self::General => PEDAGOGICAL_INSPECTOR_SYSTEM_PROMPT,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String, // "user" or "model"
    pub text: String,
    pub subject: SubjectAssistantMode,
    pub has_image_attachment: bool,
    pub image_dimensions: Option<(u32, u32)>,
    pub rag_slide_badge: Option<String>,
    pub timestamp: String,
}

#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    Started { message_index: usize },
    TokenDelta { message_index: usize, delta: String },
    Completed { message_index: usize },
    Error { message_index: usize, error: String },
}

pub struct GeminiChatClient {
    pub runtime: tokio::runtime::Runtime,
    http_client: reqwest::Client,
}

impl GeminiChatClient {
    pub fn new() -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("Failed to initialize Tokio multi-threaded runtime");

        let http_client = reqwest::Client::builder()
            .timeout(Duration::from_secs(90))
            .build()
            .expect("Failed to build reqwest HTTP client");

        Self {
            runtime,
            http_client,
        }
    }

    /// Resolve `GEMINI_API_KEY` from explicit user input, `.env` file, or OS environment variables.
    pub fn resolve_api_key(configured_key: &str) -> String {
        let trimmed = configured_key.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
        if let Ok(env_key) = std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_API_KEY")) {
            if !env_key.trim().is_empty() {
                return env_key.trim().to_string();
            }
        }
        for env_path in [".env", "../.env"] {
            if let Ok(content) = std::fs::read_to_string(Path::new(env_path)) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.starts_with('#') || line.is_empty() {
                        continue;
                    }
                    if let Some(val) = line
                        .strip_prefix("GEMINI_API_KEY=")
                        .or_else(|| line.strip_prefix("GOOGLE_API_KEY="))
                    {
                        let cleaned = val.trim().trim_matches('"').trim_matches('\'');
                        if !cleaned.is_empty() {
                            return cleaned.to_string();
                        }
                    }
                }
            }
        }
        String::new()
    }

    /// Transcribe real recorded microphone WAV bytes (`audio/wav`) into Vietnamese/English text
    /// using Gemini 2.5 Flash / Gemini 2.0 Flash Multimodal Audio API.
    pub fn transcribe_audio_wav(
        &self,
        api_key: String,
        wav_bytes: Vec<u8>,
        timestamp_ms: u64,
        tx: Sender<Result<TranscriptSegmentEvent, String>>,
    ) {
        let client = self.http_client.clone();
        self.runtime.spawn(async move {
            let resolved_key = Self::resolve_api_key(&api_key);
            if resolved_key.is_empty() {
                let _ = tx.send(Err(
                    "Chưa cấu hình GEMINI_API_KEY để chuyển giọng nói thành văn bản (STT).".to_string(),
                ));
                return;
            }

            let b64_wav = base64::engine::general_purpose::STANDARD.encode(&wav_bytes);
            let payload = json!({
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": "Hãy nghe đoạn thu âm giọng giảng bài này và chép lời (transcribe) chính xác sang tiếng Việt (giữ nguyên các thuật ngữ Toán/Lý/Hóa/Tin hoặc tiếng Anh nếu có). Chỉ trả về nội dung lời nói, nếu chỉ có tiếng ồn hoặc im lặng thì trả về chuỗi rỗng."
                            },
                            {
                                "inline_data": {
                                    "mime_type": "audio/wav",
                                    "data": b64_wav
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 512
                }
            });

            for model in ["gemini-2.5-flash", "gemini-2.0-flash"] {
                let url = format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={resolved_key}"
                );
                if let Ok(resp) = client.post(&url).json(&payload).send().await {
                    if resp.status().is_success() {
                        if let Ok(val) = resp.json::<serde_json::Value>().await {
                            if let Some(text) = val
                                .pointer("/candidates/0/content/parts/0/text")
                                .and_then(|t| t.as_str())
                            {
                                let cleaned = text.trim().to_string();
                                let _ = tx.send(Ok(TranscriptSegmentEvent {
                                    timestamp_ms,
                                    text: cleaned,
                                }));
                                return;
                            }
                        }
                    }
                }
            }
            let _ = tx.send(Err("Không thể kết nối Gemini Audio STT.".to_string()));
        });
    }

    /// Stream real multimodal completion from Google Gemini API (`gemini-2.5-flash` with fallback to `gemini-2.0-flash`)
    /// grounded with `slide_rag_context` from the in-memory `SlideRagIndex`.
    pub fn stream_completion(
        &self,
        api_key: String,
        model_name: String,
        subject: SubjectAssistantMode,
        user_prompt: String,
        slide_rag_context: Option<String>,
        base64_png_selection: Option<String>,
        message_index: usize,
        event_tx: Sender<ChatStreamEvent>,
    ) {
        let client = self.http_client.clone();
        self.runtime.spawn(async move {
            let _ = event_tx.send(ChatStreamEvent::Started { message_index });

            let trimmed_key = Self::resolve_api_key(&api_key);
            if trimmed_key.is_empty() {
                let _ = event_tx.send(ChatStreamEvent::Error {
                    message_index,
                    error: "Chưa cấu hình Google Gemini API Key. Vui lòng nhập API Key vào ô cấu hình ở đầu khung AI Assistant (hoặc đặt biến môi trường GEMINI_API_KEY trong file .env) để phân tích bảng và bài giảng thật.".to_string(),
                });
                return;
            }

            let primary_model = if model_name.trim().is_empty() {
                "gemini-2.5-flash"
            } else {
                model_name.trim()
            };

            let mut user_parts = Vec::new();
            if let Some(rag_ctx) = slide_rag_context {
                if !rag_ctx.trim().is_empty() {
                    user_parts.push(json!({
                        "text": rag_ctx
                    }));
                }
            }
            user_parts.push(json!({
                "text": user_prompt
            }));

            if let Some(b64_img) = base64_png_selection {
                user_parts.push(json!({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64_img
                    }
                }));
            }

            let payload = json!({
                "system_instruction": {
                    "parts": [
                        { "text": PEDAGOGICAL_INSPECTOR_SYSTEM_PROMPT },
                        { "text": subject.system_prompt() }
                    ]
                },
                "contents": [
                    {
                        "role": "user",
                        "parts": user_parts
                    }
                ],
                "generationConfig": {
                    "temperature": 0.25,
                    "maxOutputTokens": 2048
                }
            });

            let candidate_models = if primary_model == "gemini-2.5-flash" {
                vec!["gemini-2.5-flash", "gemini-2.0-flash"]
            } else {
                vec![primary_model, "gemini-2.0-flash"]
            };

            let mut active_resp = None;
            let mut last_err_str = String::new();

            for model in candidate_models {
                let url = format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={trimmed_key}"
                );
                match client.post(&url).json(&payload).send().await {
                    Ok(r) if r.status().is_success() => {
                        active_resp = Some(r);
                        break;
                    }
                    Ok(r) => {
                        let status = r.status();
                        let body = r.text().await.unwrap_or_default();
                        last_err_str = format!("HTTP {status} ({model}): {body}");
                    }
                    Err(e) => {
                        last_err_str = format!("Lỗi mạng ({model}): {e}");
                    }
                }
            }

            let Some(resp) = active_resp else {
                let _ = event_tx.send(ChatStreamEvent::Error {
                    message_index,
                    error: format!("Gemini API trả về lỗi: {last_err_str}"),
                });
                return;
            };

            let mut stream = resp.bytes_stream();
            let mut line_buffer = String::new();

            while let Some(chunk_res) = stream.next().await {
                match chunk_res {
                    Ok(bytes) => {
                        let chunk_str = String::from_utf8_lossy(&bytes);
                        line_buffer.push_str(&chunk_str);

                        while let Some(newline_pos) = line_buffer.find('\n') {
                            let line = line_buffer[..newline_pos].trim().to_string();
                            line_buffer.drain(..=newline_pos);

                            if let Some(json_str) = line.strip_prefix("data:") {
                                let trimmed = json_str.trim();
                                if trimmed.is_empty() || trimmed == "[DONE]" {
                                    continue;
                                }
                                if let Ok(val) = serde_json::from_str::<serde_json::Value>(trimmed)
                                {
                                    if let Some(parts) = val
                                        .pointer("/candidates/0/content/parts")
                                        .and_then(|v| v.as_array())
                                    {
                                        for part in parts {
                                            if let Some(text_delta) =
                                                part.get("text").and_then(|t| t.as_str())
                                            {
                                                let _ = event_tx.send(
                                                    ChatStreamEvent::TokenDelta {
                                                        message_index,
                                                        delta: text_delta.to_string(),
                                                    },
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        let _ = event_tx.send(ChatStreamEvent::Error {
                            message_index,
                            error: format!("Lỗi đọc luồng SSE từ Gemini: {e}"),
                        });
                        return;
                    }
                }
            }

            let _ = event_tx.send(ChatStreamEvent::Completed { message_index });
        });
    }
}
