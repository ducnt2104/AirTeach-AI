//! Real Multimodal AI Pedagogical Assistant, Speech-to-Text & Board Analyzer (`src/ai/gemini_chat.rs`):
//! - Resolves `GEMINI_API_KEY` & `GEMINI_MODEL_ID` from UI Settings, OS env, or `.env` (zero hardcoded secrets)
//! - Uses `gemini-3.5-flash` primary with automatic fallback to `gemini-2.5-flash` -> `gemini-2.0-flash`
//! - Supports Structured JSON Output (`responseMimeType: "application/json"`), retry backoff,
//!   SQLite response caching keys, and offline graceful fallback using `SlideRagIndex` context.

use crate::ai::subtitles_engine::TranscriptSegmentEvent;
use base64::Engine;
use crossbeam_channel::Sender;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::Duration;

pub const DEFAULT_GEMINI_MODEL: &str = "gemini-3.5-flash";

pub const PEDAGOGICAL_INSPECTOR_SYSTEM_PROMPT: &str =
    "Bạn là trợ lý giáo viên chuyên nghiệp của hệ thống AirTeach AI (DNT - Victer). \
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuizQuestionItem {
    #[serde(default)]
    pub question: String,
    #[serde(default)]
    pub options: Vec<String>,
    #[serde(default)]
    pub correct_answer: String,
    #[serde(default)]
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PedagogicalStructuredResponse {
    #[serde(default)]
    pub summary: String,
    #[serde(default)]
    pub key_formulas: Vec<String>,
    #[serde(default)]
    pub detected_errors: Vec<String>,
    #[serde(default)]
    pub step_by_step_solution: Vec<String>,
    #[serde(default)]
    pub quiz_questions: Vec<QuizQuestionItem>,
    #[serde(default)]
    pub interactive_prompts: Vec<String>,
}

impl PedagogicalStructuredResponse {
    pub fn to_formatted_markdown(&self) -> String {
        let mut out = String::new();
        if !self.summary.trim().is_empty() {
            out.push_str("**Tóm tắt trọng tâm:**\n");
            out.push_str(self.summary.trim());
            out.push_str("\n\n");
        }
        if !self.key_formulas.is_empty() {
            out.push_str("**Công thức & Khái niệm cốt lõi:**\n");
            for f in &self.key_formulas {
                if !f.trim().is_empty() {
                    out.push_str(&format!("- {}\n", f.trim()));
                }
            }
            out.push('\n');
        }
        if !self.detected_errors.is_empty() {
            out.push_str("**Kiểm tra & Chữa lỗi:**\n");
            for e in &self.detected_errors {
                if !e.trim().is_empty() {
                    out.push_str(&format!("- {}\n", e.trim()));
                }
            }
            out.push('\n');
        }
        if !self.step_by_step_solution.is_empty() {
            out.push_str("**Lời giải từng bước:**\n");
            for (i, step) in self.step_by_step_solution.iter().enumerate() {
                if !step.trim().is_empty() {
                    out.push_str(&format!("{}. {}\n", i + 1, step.trim()));
                }
            }
            out.push('\n');
        }
        if !self.quiz_questions.is_empty() {
            out.push_str("**Câu hỏi trắc nghiệm gợi ý:**\n");
            for (i, q) in self.quiz_questions.iter().enumerate() {
                if q.question.trim().is_empty() {
                    continue;
                }
                out.push_str(&format!("**Câu {}:** {}\n", i + 1, q.question.trim()));
                for opt in &q.options {
                    out.push_str(&format!("  • {}\n", opt.trim()));
                }
                if !q.correct_answer.trim().is_empty() {
                    out.push_str(&format!("  => Đáp án: {}\n", q.correct_answer.trim()));
                }
                if !q.explanation.trim().is_empty() {
                    out.push_str(&format!("  Giải thích: {}\n", q.explanation.trim()));
                }
                out.push('\n');
            }
        }
        if !self.interactive_prompts.is_empty() {
            out.push_str("**Gợi ý tương tác lớp học:**\n");
            for tip in &self.interactive_prompts {
                if !tip.trim().is_empty() {
                    out.push_str(&format!("- {}\n", tip.trim()));
                }
            }
        }
        out.trim().to_string()
    }
}

#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    Started { message_index: usize },
    TokenDelta { message_index: usize, delta: String },
    CacheStore {
        cache_key: String,
        model_id: String,
        response_text: String,
        structured_json: String,
    },
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

    fn read_dotenv_var(key_names: &[&str]) -> Option<String> {
        for env_path in [".env", "../.env"] {
            if let Ok(content) = std::fs::read_to_string(Path::new(env_path)) {
                for line in content.lines() {
                    let line = line.trim();
                    if line.starts_with('#') || line.is_empty() {
                        continue;
                    }
                    for &k in key_names {
                        if let Some(val) = line.strip_prefix(&format!("{k}=")) {
                            let cleaned = val.trim().trim_matches('"').trim_matches('\'');
                            if !cleaned.is_empty() && cleaned != "your_gemini_api_key_here" {
                                return Some(cleaned.to_string());
                            }
                        }
                    }
                }
            }
        }
        None
    }

    /// Resolve `GEMINI_API_KEY` from explicit user input, OS env, or `.env` file.
    pub fn resolve_api_key(configured_key: &str) -> String {
        let trimmed = configured_key.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
        if let Ok(env_key) =
            std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_API_KEY"))
        {
            let cleaned = env_key.trim();
            if !cleaned.is_empty() && cleaned != "your_gemini_api_key_here" {
                return cleaned.to_string();
            }
        }
        Self::read_dotenv_var(&["GEMINI_API_KEY", "GOOGLE_API_KEY"]).unwrap_or_default()
    }

    /// Resolve active Gemini model ID (`GEMINI_MODEL_ID` env / `.env` > UI config > `gemini-3.5-flash`).
    pub fn resolve_model_id(configured_model: &str) -> String {
        if let Ok(env_model) = std::env::var("GEMINI_MODEL_ID") {
            let clean = env_model.trim();
            if !clean.is_empty() {
                return clean.to_string();
            }
        }
        if let Some(dotenv_model) = Self::read_dotenv_var(&["GEMINI_MODEL_ID"]) {
            return dotenv_model;
        }
        let cfg = configured_model.trim();
        if cfg.is_empty() {
            DEFAULT_GEMINI_MODEL.to_string()
        } else {
            cfg.to_string()
        }
    }

    /// Build ordered fallback chain prioritizing `primary` -> `gemini-3.5-flash` -> `gemini-2.5-flash` -> `gemini-2.0-flash`.
    pub fn build_model_chain(primary: &str) -> Vec<String> {
        let mut chain = Vec::new();
        for candidate in [primary.trim(), DEFAULT_GEMINI_MODEL, "gemini-2.5-flash", "gemini-2.0-flash"] {
            if !candidate.is_empty() && !chain.iter().any(|m: &String| m == candidate) {
                chain.push(candidate.to_string());
            }
        }
        chain
    }

    /// Deterministic SHA-256 request key for SQLite AI response caching.
    pub fn compute_cache_key(
        model_id: &str,
        subject: SubjectAssistantMode,
        prompt: &str,
        rag_context: Option<&str>,
        base64_png: Option<&str>,
        structured_json: bool,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(model_id.as_bytes());
        hasher.update(b":");
        hasher.update(subject.short_badge().as_bytes());
        hasher.update(if structured_json { b":json:" } else { b":text:" });
        hasher.update(prompt.as_bytes());
        if let Some(rag) = rag_context {
            hasher.update(b"|rag:");
            hasher.update(rag.as_bytes());
        }
        if let Some(img) = base64_png {
            hasher.update(b"|img:");
            hasher.update(img.as_bytes());
        }
        let digest = hasher.finalize();
        format!("{:x}", digest)
    }

    /// Parse structured JSON (handling fenced code blocks if returned) into typed `PedagogicalStructuredResponse`.
    pub fn parse_structured_response(raw_text: &str) -> Option<PedagogicalStructuredResponse> {
        let trimmed = raw_text.trim();
        let cleaned = if trimmed.starts_with("```") {
            let no_open = trimmed
                .trim_start_matches("```json")
                .trim_start_matches("```")
                .trim_start_matches(['\r', '\n']);
            no_open.trim_end_matches("```").trim()
        } else {
            trimmed
        };
        serde_json::from_str::<PedagogicalStructuredResponse>(cleaned).ok()
    }

    /// Offline graceful fallback preserving responsiveness & citing local `SlideRagIndex` excerpts.
    pub fn build_offline_fallback_response(
        subject: SubjectAssistantMode,
        user_prompt: &str,
        slide_rag_context: Option<&str>,
        reason: &str,
    ) -> String {
        let mut out = format!(
            "**[Chế độ Ngoại tuyến / Offline Fallback — {}]**\n\
             Không thể kết nối máy chủ Gemini API ({reason}). Toàn bộ tính năng Bảng thông minh, Camera & Nhận diện cử chỉ vẫn hoạt động bình thường.\n\n",
            subject.label()
        );
        if let Some(rag) = slide_rag_context {
            let trimmed = rag.trim();
            if !trimmed.is_empty() {
                let preview: String = trimmed.chars().take(700).collect();
                out.push_str("**Ngữ cảnh trích xuất từ tài liệu/slide cục bộ (SlideRagIndex):**\n");
                out.push_str(&preview);
                if trimmed.chars().count() > 700 {
                    out.push_str("...");
                }
                out.push_str("\n\n");
            }
        }
        let q_preview: String = user_prompt.trim().chars().take(240).collect();
        if !q_preview.is_empty() {
            out.push_str(&format!(
                "**Yêu cầu đã ghi nhận:** {}\n\n\
                 *Mẹo: Khi có mạng trở lại hoặc sau khi lưu GEMINI_API_KEY trong file `.env` / Settings, nhấn phân tích lại để nhận Structured JSON & LaTeX từ Gemini 3.5 Flash.*",
                q_preview
            ));
        }
        out
    }

    /// Transcribe real recorded microphone WAV bytes (`audio/wav`) into Vietnamese/English text
    /// prioritizing `gemini-3.5-flash` with fallback to `gemini-2.5-flash` -> `gemini-2.0-flash`.
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

            let models = Self::build_model_chain(&Self::resolve_model_id(DEFAULT_GEMINI_MODEL));
            for model in models {
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

    /// Stream or fetch completion using Gemini 3.5 Flash (default) + automatic fallback,
    /// Structured JSON support, transient retry backoff, cache persistence hooks, and offline fallback.
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
        self.stream_completion_with_options(
            api_key,
            model_name,
            subject,
            user_prompt,
            slide_rag_context,
            base64_png_selection,
            false,
            String::new(),
            message_index,
            event_tx,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn stream_completion_with_options(
        &self,
        api_key: String,
        model_name: String,
        subject: SubjectAssistantMode,
        user_prompt: String,
        slide_rag_context: Option<String>,
        base64_png_selection: Option<String>,
        use_structured_json: bool,
        cache_key: String,
        message_index: usize,
        event_tx: Sender<ChatStreamEvent>,
    ) {
        let client = self.http_client.clone();
        self.runtime.spawn(async move {
            let _ = event_tx.send(ChatStreamEvent::Started { message_index });

            let trimmed_key = Self::resolve_api_key(&api_key);
            if trimmed_key.is_empty() {
                let offline_msg = Self::build_offline_fallback_response(
                    subject,
                    &user_prompt,
                    slide_rag_context.as_deref(),
                    "Chưa thiết lập GEMINI_API_KEY",
                );
                let _ = event_tx.send(ChatStreamEvent::TokenDelta {
                    message_index,
                    delta: offline_msg,
                });
                let _ = event_tx.send(ChatStreamEvent::Completed { message_index });
                return;
            }

            let primary_model = Self::resolve_model_id(&model_name);
            let candidate_models = Self::build_model_chain(&primary_model);

            let mut user_parts = Vec::new();
            if let Some(ref rag_ctx) = slide_rag_context {
                if !rag_ctx.trim().is_empty() {
                    user_parts.push(json!({ "text": rag_ctx }));
                }
            }
            let effective_prompt = if use_structured_json {
                format!(
                    "{user_prompt}\n\n\
                     TRẢ VỀ STRICT JSON ĐÚNG SCHEMA SAU (không kèm markdown fence thừa):\n\
                     {{\"summary\": string, \"key_formulas\": [string], \"detected_errors\": [string], \"step_by_step_solution\": [string], \"quiz_questions\": [{{\"question\": string, \"options\": [string], \"correct_answer\": string, \"explanation\": string}}], \"interactive_prompts\": [string]}}"
                )
            } else {
                user_prompt.clone()
            };
            user_parts.push(json!({ "text": effective_prompt }));

            if let Some(b64_img) = base64_png_selection {
                user_parts.push(json!({
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": b64_img
                    }
                }));
            }

            let gen_cfg = if use_structured_json {
                json!({
                    "temperature": 0.2,
                    "maxOutputTokens": 2048,
                    "responseMimeType": "application/json"
                })
            } else {
                json!({
                    "temperature": 0.25,
                    "maxOutputTokens": 2048
                })
            };

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
                "generationConfig": gen_cfg
            });

            if use_structured_json {
                let mut last_err = String::new();
                for model in &candidate_models {
                    for attempt in 0..2u8 {
                        if attempt > 0 {
                            tokio::time::sleep(Duration::from_millis(350 * attempt as u64)).await;
                        }
                        let url = format!(
                            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={trimmed_key}"
                        );
                        match client.post(&url).json(&payload).send().await {
                            Ok(r) if r.status().is_success() => {
                                if let Ok(val) = r.json::<serde_json::Value>().await {
                                    if let Some(raw_txt) = val
                                        .pointer("/candidates/0/content/parts/0/text")
                                        .and_then(|t| t.as_str())
                                    {
                                        let formatted = if let Some(parsed) =
                                            Self::parse_structured_response(raw_txt)
                                        {
                                            parsed.to_formatted_markdown()
                                        } else {
                                            raw_txt.trim().to_string()
                                        };
                                        if !formatted.is_empty() {
                                            let _ = event_tx.send(ChatStreamEvent::TokenDelta {
                                                message_index,
                                                delta: formatted.clone(),
                                            });
                                            if !cache_key.is_empty() {
                                                let _ = event_tx.send(ChatStreamEvent::CacheStore {
                                                    cache_key: cache_key.clone(),
                                                    model_id: model.clone(),
                                                    response_text: formatted,
                                                    structured_json: raw_txt.to_string(),
                                                });
                                            }
                                            let _ = event_tx
                                                .send(ChatStreamEvent::Completed { message_index });
                                            return;
                                        }
                                    }
                                }
                            }
                            Ok(r) => {
                                let status = r.status();
                                let body = r.text().await.unwrap_or_default();
                                last_err = format!("HTTP {status} ({model}): {body}");
                                if status.as_u16() == 404 || status.as_u16() == 400 {
                                    break;
                                }
                            }
                            Err(e) => {
                                last_err = format!("Mất kết nối ({model}): {e}");
                            }
                        }
                    }
                }

                let offline_msg = Self::build_offline_fallback_response(
                    subject,
                    &user_prompt,
                    slide_rag_context.as_deref(),
                    &last_err,
                );
                let _ = event_tx.send(ChatStreamEvent::TokenDelta {
                    message_index,
                    delta: offline_msg,
                });
                let _ = event_tx.send(ChatStreamEvent::Completed { message_index });
                return;
            }

            // Streaming SSE flow (Q&A / General conversation) with retry backoff & model fallback
            let mut active_resp = None;
            let mut used_model = primary_model.clone();
            let mut last_err_str = String::new();

            'model_loop: for model in &candidate_models {
                for attempt in 0..2u8 {
                    if attempt > 0 {
                        tokio::time::sleep(Duration::from_millis(350 * attempt as u64)).await;
                    }
                    let url = format!(
                        "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse&key={trimmed_key}"
                    );
                    match client.post(&url).json(&payload).send().await {
                        Ok(r) if r.status().is_success() => {
                            used_model = model.clone();
                            active_resp = Some(r);
                            break 'model_loop;
                        }
                        Ok(r) => {
                            let status = r.status();
                            let body = r.text().await.unwrap_or_default();
                            last_err_str = format!("HTTP {status} ({model}): {body}");
                            if status.as_u16() == 404 || status.as_u16() == 400 {
                                break;
                            }
                        }
                        Err(e) => {
                            last_err_str = format!("Lỗi mạng ({model}): {e}");
                        }
                    }
                }
            }

            let Some(resp) = active_resp else {
                let offline_msg = Self::build_offline_fallback_response(
                    subject,
                    &user_prompt,
                    slide_rag_context.as_deref(),
                    &last_err_str,
                );
                let _ = event_tx.send(ChatStreamEvent::TokenDelta {
                    message_index,
                    delta: offline_msg,
                });
                let _ = event_tx.send(ChatStreamEvent::Completed { message_index });
                return;
            };

            let mut stream = resp.bytes_stream();
            let mut line_buffer = String::new();
            let mut accumulated_text = String::new();

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
                                                accumulated_text.push_str(text_delta);
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

            if !cache_key.is_empty() && !accumulated_text.trim().is_empty() {
                let _ = event_tx.send(ChatStreamEvent::CacheStore {
                    cache_key,
                    model_id: used_model,
                    response_text: accumulated_text,
                    structured_json: String::new(),
                });
            }

            let _ = event_tx.send(ChatStreamEvent::Completed { message_index });
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemini_35_flash_model_chain_and_structured_json_parsing() {
        let chain = GeminiChatClient::build_model_chain("gemini-3.5-flash");
        assert_eq!(chain, vec!["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]);

        let raw_json = r#"```json
        {
            "summary": "Phương trình bậc hai có 2 nghiệm phân biệt.",
            "key_formulas": ["Δ = b² - 4ac"],
            "detected_errors": ["Nhầm dấu -b ở bước 2"],
            "step_by_step_solution": ["Tính Δ = 9 > 0", "Suy ra x₁ = 2, x₂ = -1"],
            "quiz_questions": [
                {
                    "question": "Nghiệm của x² - x - 2 = 0?",
                    "options": ["A. 2 và -1", "B. 1 và -2", "C. 0 và 2", "D. Vô nghiệm"],
                    "correct_answer": "A",
                    "explanation": "Thay x=2 và x=-1 đều thỏa mãn."
                }
            ],
            "interactive_prompts": ["Học sinh thử nhẩm Vi-ét?"]
        }
        ```"#;

        let parsed = GeminiChatClient::parse_structured_response(raw_json)
            .expect("Must parse fenced JSON payload cleanly");
        let md = parsed.to_formatted_markdown();
        assert!(md.contains("Phương trình bậc hai"));
        assert!(md.contains("Δ = b² - 4ac"));
        assert!(md.contains("Câu 1:"));
    }

    #[test]
    fn test_cache_key_and_offline_fallback() {
        let k1 = GeminiChatClient::compute_cache_key(
            "gemini-3.5-flash",
            SubjectAssistantMode::Math,
            "Giải x+2=5",
            Some("Slide 1"),
            None,
            true,
        );
        let k2 = GeminiChatClient::compute_cache_key(
            "gemini-3.5-flash",
            SubjectAssistantMode::Math,
            "Giải x+2=5",
            Some("Slide 1"),
            None,
            true,
        );
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 64);

        let fallback = GeminiChatClient::build_offline_fallback_response(
            SubjectAssistantMode::Physics,
            "Kiểm tra định luật II Newton",
            Some("F = m * a (Slide Vật lý chương 2)"),
            "No internet",
        );
        assert!(fallback.contains("Offline Fallback"));
        assert!(fallback.contains("F = m * a"));
    }
}
