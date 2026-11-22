//! Real Multimodal AI Pedagogical Assistant, Speech-to-Text & Board Analyzer (`src/ai/gemini_chat.rs`):
//! - Resolves `GEMINI_API_KEY` & `GEMINI_MODEL_ID` with multi-layer priority:
//!   `(1) UI Settings` -> `(2) OS env / .env file` (zero hardcoded secrets)
//! - Uses `gemini-3.8-flash` primary with automatic fallback to `gemini-3.5-flash` -> `gemini-3.5-flash-lite`
//!   (or `GEMINI_FALLBACK_MODELS` override)
//! - Supports Live Health Check (`maxOutputTokens: 5`), Vietnamese HTTP 400/401/403/404/429/Network mapping,
//!   3x Exponential Backoff (`1s, 2s, 4s`) on 429, SQLite response caching, and offline fallback.

use crate::ai::subtitles_engine::TranscriptSegmentEvent;
use crate::utils::output_sanitizer::{OutputSanitizer, StreamingSanitizer};
use base64::Engine;
use crossbeam_channel::Sender;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::{Duration, Instant};

pub const DEFAULT_GEMINI_MODEL: &str = "gemini-3.8-flash";
pub const DEFAULT_FALLBACK_MODELS: [&str; 3] = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
];
pub const RETRY_BACKOFF_SECS: [u64; 3] = [1, 2, 4];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApiConnectionStatus {
    Unconfigured,
    Connected,
    Error,
}

impl ApiConnectionStatus {
    pub const fn badge_label_vi(self) -> &'static str {
        match self {
            Self::Unconfigured => "Chưa cấu hình",
            Self::Connected => "Đã kết nối",
            Self::Error => "Lỗi Key/Quota",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiKeySource {
    UiSettings,
    EnvVar,
    DotEnv,
    None,
}

impl ApiKeySource {
    pub const fn label_vi(self) -> &'static str {
        match self {
            Self::UiSettings => "UI Settings",
            Self::EnvVar => "Biến môi trường",
            Self::DotEnv => "File .env",
            Self::None => "Chưa có",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GeminiHealthCheckResult {
    pub ok: bool,
    pub status: ApiConnectionStatus,
    pub model_used: String,
    pub latency_ms: f32,
    pub message_vi: String,
    pub http_status: Option<u16>,
}

pub fn mask_api_key(key: &str) -> String {
    let trimmed = key.trim();
    if trimmed.is_empty() {
        return "(Chưa cấu hình)".to_string();
    }
    let chars: Vec<char> = trimmed.chars().collect();
    if chars.len() <= 8 {
        return "••••••••".to_string();
    }
    let prefix: String = chars[..4].iter().collect();
    let suffix: String = chars[chars.len() - 4..].iter().collect();
    format!("{prefix}••••••{suffix}")
}

/// Translate Gemini HTTP / network errors into friendly Vietnamese messages.
/// Returns `(message_vi, fail_fast_auth, fallback_model)`.
pub fn classify_http_error(status: u16, body: &str, model: &str) -> (String, bool, bool) {
    let body_low = body.to_lowercase();
    match status {
        401 | 403 => (
            "API Key không hợp lệ hoặc chưa bật quyền truy cập Gemini API (401/403).".to_string(),
            true,
            false,
        ),
        400 => {
            if body_low.contains("api_key_invalid")
                || body_low.contains("api key not valid")
                || body_low.contains("invalid api key")
            {
                (
                    "API Key không hợp lệ (400 API_KEY_INVALID). Vui lòng kiểm tra lại Key."
                        .to_string(),
                    true,
                    false,
                )
            } else {
                (
                    format!("Yêu cầu không hợp lệ tới {model} (HTTP 400)."),
                    false,
                    true,
                )
            }
        }
        404 => (
            format!("Model `{model}` không tồn tại hoặc đã ngừng hỗ trợ (404). Tự động chuyển model dự phòng."),
            false,
            true,
        ),
        429 => (
            format!("Vượt giới hạn lượt gọi / Quota Gemini API cho `{model}` (429). Đang chờ thử lại..."),
            false,
            true,
        ),
        500..=599 => (
            format!("Máy chủ Gemini tạm thời quá tải (HTTP {status}) trên `{model}`."),
            false,
            true,
        ),
        _ => (
            format!("Lỗi HTTP {status} từ Gemini API trên `{model}`."),
            false,
            true,
        ),
    }
}

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
    ApiStatusUpdated {
        status: ApiConnectionStatus,
        message: String,
    },
    MetricsUpdated {
        engine: String,
        model: String,
        tokens_per_sec: f32,
    },
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

    /// Resolve `GEMINI_API_KEY` and report source priority: `(1) UI Settings` -> `(2) OS env` -> `(3) .env`.
    pub fn resolve_api_key_with_source(configured_key: &str) -> (String, ApiKeySource) {
        let trimmed = configured_key.trim();
        if !trimmed.is_empty() {
            return (trimmed.to_string(), ApiKeySource::UiSettings);
        }
        if let Ok(env_key) =
            std::env::var("GEMINI_API_KEY").or_else(|_| std::env::var("GOOGLE_API_KEY"))
        {
            let cleaned = env_key.trim();
            if !cleaned.is_empty() && cleaned != "your_gemini_api_key_here" {
                return (cleaned.to_string(), ApiKeySource::EnvVar);
            }
        }
        if let Some(dotenv_key) = Self::read_dotenv_var(&["GEMINI_API_KEY", "GOOGLE_API_KEY"]) {
            return (dotenv_key, ApiKeySource::DotEnv);
        }
        (String::new(), ApiKeySource::None)
    }

    /// Resolve `GEMINI_API_KEY` from explicit user input, OS env, or `.env` file.
    pub fn resolve_api_key(configured_key: &str) -> String {
        Self::resolve_api_key_with_source(configured_key).0
    }

    /// Resolve active Gemini model ID (`GEMINI_MODEL_ID` env / `.env` > UI config > `gemini-3.8-flash`).
    pub fn resolve_model_id(configured_model: &str) -> String {
        let cfg = configured_model.trim();
        if !cfg.is_empty() {
            return cfg.to_string();
        }
        if let Ok(env_model) = std::env::var("GEMINI_MODEL_ID") {
            let clean = env_model.trim();
            if !clean.is_empty() {
                return clean.to_string();
            }
        }
        if let Some(dotenv_model) = Self::read_dotenv_var(&["GEMINI_MODEL_ID"]) {
            return dotenv_model;
        }
        DEFAULT_GEMINI_MODEL.to_string()
    }

    /// Build ordered fallback chain prioritizing `primary`, optional `GEMINI_FALLBACK_MODELS`,
    /// then current `DEFAULT_FALLBACK_MODELS` (`gemini-3.8-flash` -> `gemini-3.5-flash` -> `gemini-3.5-flash-lite`).
    pub fn build_model_chain(primary: &str) -> Vec<String> {
        let mut chain: Vec<String> = Vec::new();
        let push_unique = |chain: &mut Vec<String>, candidate: &str| {
            let c = candidate.trim();
            if !c.is_empty() && !chain.iter().any(|m| m == c) {
                chain.push(c.to_string());
            }
        };

        push_unique(&mut chain, primary);

        if let Some(fb_raw) = std::env::var("GEMINI_FALLBACK_MODELS")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .or_else(|| Self::read_dotenv_var(&["GEMINI_FALLBACK_MODELS"]))
        {
            for item in fb_raw.split(',') {
                push_unique(&mut chain, item);
            }
        }

        for candidate in DEFAULT_FALLBACK_MODELS {
            push_unique(&mut chain, candidate);
        }
        chain
    }

    /// Perform live connection health check (`POST :generateContent` with `maxOutputTokens: 5`)
    /// validating key + model reachability without mutating UI thread.
    pub fn test_connection(
        &self,
        api_key: String,
        primary_model: String,
        tx: Sender<GeminiHealthCheckResult>,
    ) {
        let client = self.http_client.clone();
        self.runtime.spawn(async move {
            let (resolved_key, _src) = Self::resolve_api_key_with_source(&api_key);
            if resolved_key.is_empty() {
                let _ = tx.send(GeminiHealthCheckResult {
                    ok: false,
                    status: ApiConnectionStatus::Unconfigured,
                    model_used: String::new(),
                    latency_ms: 0.0,
                    message_vi: "Chưa cấu hình GEMINI_API_KEY (nhập ở Settings hoặc `.env`)."
                        .to_string(),
                    http_status: None,
                });
                return;
            }

            let primary = Self::resolve_model_id(&primary_model);
            let models = Self::build_model_chain(&primary);
            let payload = json!({
                "contents": [
                    {
                        "role": "user",
                        "parts": [{ "text": "Reply OK" }]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 5
                }
            });

            let t0 = Instant::now();
            let mut last_msg = "Không thể kết nối máy chủ Gemini API.".to_string();
            let mut last_http: Option<u16> = None;

            for model in &models {
                let url = format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                );
                match client
                    .post(&url)
                    .timeout(Duration::from_secs(12))
                    .header("x-goog-api-key", &resolved_key)
                    .json(&payload)
                    .send()
                    .await
                {
                    Ok(resp) if resp.status().is_success() => {
                        let latency_ms = t0.elapsed().as_secs_f32() * 1000.0;
                        let _ = tx.send(GeminiHealthCheckResult {
                            ok: true,
                            status: ApiConnectionStatus::Connected,
                            model_used: model.clone(),
                            latency_ms,
                            message_vi: format!(
                                "Kết nối Gemini thành công ({model}, {latency_ms:.0}ms)."
                            ),
                            http_status: Some(resp.status().as_u16()),
                        });
                        return;
                    }
                    Ok(resp) => {
                        let code = resp.status().as_u16();
                        last_http = Some(code);
                        let body = resp.text().await.unwrap_or_default();
                        let (msg_vi, fail_fast, _fallback) =
                            classify_http_error(code, &body, model);
                        last_msg = msg_vi;
                        if fail_fast {
                            break;
                        }
                    }
                    Err(err) => {
                        last_http = None;
                        last_msg = if err.is_timeout() {
                            "Hết thời gian chờ kết nối Gemini API (Timeout). Kiểm tra lại mạng."
                                .to_string()
                        } else {
                            format!("Mất kết nối mạng tới Gemini API: {err}")
                        };
                    }
                }
            }

            let _ = tx.send(GeminiHealthCheckResult {
                ok: false,
                status: ApiConnectionStatus::Error,
                model_used: String::new(),
                latency_ms: t0.elapsed().as_secs_f32() * 1000.0,
                message_vi: last_msg,
                http_status: last_http,
            });
        });
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

            let models = Self::build_model_chain(&Self::resolve_model_id(""));
            let mut last_err = "Không thể kết nối Gemini Audio STT.".to_string();
            'stt_model: for model in models {
                let url = format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                );
                let mut backoff_idx = 0usize;
                loop {
                    match client
                        .post(&url)
                        .header("x-goog-api-key", &resolved_key)
                        .json(&payload)
                        .send()
                        .await
                    {
                        Ok(resp) if resp.status().is_success() => {
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
                            break;
                        }
                        Ok(resp) => {
                            let code = resp.status().as_u16();
                            let body = resp.text().await.unwrap_or_default();
                            let (msg_vi, fail_fast, _) = classify_http_error(code, &body, &model);
                            last_err = msg_vi;
                            if fail_fast {
                                break 'stt_model;
                            }
                            if code == 429 && backoff_idx < RETRY_BACKOFF_SECS.len() {
                                let wait_s = RETRY_BACKOFF_SECS[backoff_idx];
                                backoff_idx += 1;
                                tokio::time::sleep(Duration::from_secs(wait_s)).await;
                                continue;
                            }
                            break;
                        }
                        Err(e) => {
                            last_err = format!("Mất kết nối Gemini Audio STT ({model}): {e}");
                            break;
                        }
                    }
                }
            }
            let _ = tx.send(Err(last_err));
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
                let _ = event_tx.send(ChatStreamEvent::ApiStatusUpdated {
                    status: ApiConnectionStatus::Unconfigured,
                    message: "Chưa cấu hình GEMINI_API_KEY".to_string(),
                });
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
                'json_model_loop: for model in &candidate_models {
                    let url = format!(
                        "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                    );
                    let mut backoff_idx = 0usize;
                    let mut net_retries = 0u8;
                    loop {
                        match client
                            .post(&url)
                            .header("x-goog-api-key", &trimmed_key)
                            .json(&payload)
                            .send()
                            .await
                        {
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
                                            let _ = event_tx.send(ChatStreamEvent::ApiStatusUpdated {
                                                status: ApiConnectionStatus::Connected,
                                                message: format!("Đã kết nối ({model})"),
                                            });
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
                                last_err = format!("Phản hồi rỗng hoặc sai định dạng từ `{model}`.");
                                break;
                            }
                            Ok(r) => {
                                let code = r.status().as_u16();
                                let body = r.text().await.unwrap_or_default();
                                let (msg_vi, fail_fast, _should_fallback) =
                                    classify_http_error(code, &body, model);
                                last_err = msg_vi;
                                if fail_fast {
                                    break 'json_model_loop;
                                }
                                if code == 429 && backoff_idx < RETRY_BACKOFF_SECS.len() {
                                    let wait_s = RETRY_BACKOFF_SECS[backoff_idx];
                                    backoff_idx += 1;
                                    tokio::time::sleep(Duration::from_secs(wait_s)).await;
                                    continue;
                                }
                                if (500..=599).contains(&code) && net_retries < 1 {
                                    net_retries += 1;
                                    tokio::time::sleep(Duration::from_millis(600)).await;
                                    continue;
                                }
                                break;
                            }
                            Err(e) => {
                                last_err = if e.is_timeout() {
                                    format!("Hết thời gian chờ kết nối mạng ({model}).")
                                } else {
                                    format!("Mất kết nối mạng ({model}): {e}")
                                };
                                if net_retries < 1 {
                                    net_retries += 1;
                                    tokio::time::sleep(Duration::from_millis(500)).await;
                                    continue;
                                }
                                break;
                            }
                        }
                    }
                }

                let _ = event_tx.send(ChatStreamEvent::ApiStatusUpdated {
                    status: ApiConnectionStatus::Error,
                    message: last_err.clone(),
                });
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

            // Streaming SSE flow (Q&A / General conversation) with 1s/2s/4s retry backoff & model fallback
            let mut active_resp = None;
            let mut used_model = primary_model.clone();
            let mut last_err_str = String::new();

            'model_loop: for model in &candidate_models {
                let url = format!(
                    "https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
                );
                let mut backoff_idx = 0usize;
                let mut net_retries = 0u8;
                loop {
                    match client
                        .post(&url)
                        .header("x-goog-api-key", &trimmed_key)
                        .json(&payload)
                        .send()
                        .await
                    {
                        Ok(r) if r.status().is_success() => {
                            used_model = model.clone();
                            active_resp = Some(r);
                            break 'model_loop;
                        }
                        Ok(r) => {
                            let code = r.status().as_u16();
                            let body = r.text().await.unwrap_or_default();
                            let (msg_vi, fail_fast, _should_fallback) =
                                classify_http_error(code, &body, model);
                            last_err_str = msg_vi;
                            if fail_fast {
                                break 'model_loop;
                            }
                            if code == 429 && backoff_idx < RETRY_BACKOFF_SECS.len() {
                                let wait_s = RETRY_BACKOFF_SECS[backoff_idx];
                                backoff_idx += 1;
                                tokio::time::sleep(Duration::from_secs(wait_s)).await;
                                continue;
                            }
                            if (500..=599).contains(&code) && net_retries < 1 {
                                net_retries += 1;
                                tokio::time::sleep(Duration::from_millis(600)).await;
                                continue;
                            }
                            break;
                        }
                        Err(e) => {
                            last_err_str = if e.is_timeout() {
                                format!("Hết thời gian chờ kết nối mạng ({model}).")
                            } else {
                                format!("Mất kết nối mạng ({model}): {e}")
                            };
                            if net_retries < 1 {
                                net_retries += 1;
                                tokio::time::sleep(Duration::from_millis(500)).await;
                                continue;
                            }
                            break;
                        }
                    }
                }
            }

            let Some(resp) = active_resp else {
                let _ = event_tx.send(ChatStreamEvent::ApiStatusUpdated {
                    status: ApiConnectionStatus::Error,
                    message: last_err_str.clone(),
                });
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
            let _ = event_tx.send(ChatStreamEvent::ApiStatusUpdated {
                status: ApiConnectionStatus::Connected,
                message: format!("Đã kết nối ({used_model})"),
            });

            let mut stream = resp.bytes_stream();
            let mut line_buffer = String::new();
            let mut accumulated_text = String::new();
            let mut stream_sanitizer = StreamingSanitizer::new();
            let sse_start = Instant::now();
            let mut chunk_count = 0usize;

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
                                                chunk_count += 1;
                                                let safe_delta = stream_sanitizer.feed(text_delta);
                                                if !safe_delta.is_empty() {
                                                    accumulated_text.push_str(&safe_delta);
                                                    let _ = event_tx.send(
                                                        ChatStreamEvent::TokenDelta {
                                                            message_index,
                                                            delta: safe_delta,
                                                        },
                                                    );
                                                }
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

            let tail = stream_sanitizer.flush();
            if !tail.is_empty() {
                accumulated_text.push_str(&tail);
                let _ = event_tx.send(ChatStreamEvent::TokenDelta {
                    message_index,
                    delta: tail,
                });
            }

            let elapsed_s = sse_start.elapsed().as_secs_f32().max(0.05);
            let words = accumulated_text.split_whitespace().count().max(chunk_count);
            let tps = words as f32 / elapsed_s;
            let _ = event_tx.send(ChatStreamEvent::MetricsUpdated {
                engine: "Cloud".to_string(),
                model: used_model.clone(),
                tokens_per_sec: tps,
            });

            if !cache_key.is_empty() && !accumulated_text.trim().is_empty() {
                let _ = event_tx.send(ChatStreamEvent::CacheStore {
                    cache_key,
                    model_id: used_model,
                    response_text: OutputSanitizer::sanitize_plain_text(&accumulated_text),
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
    fn test_gemini_38_flash_model_chain_and_structured_json_parsing() {
        let chain = GeminiChatClient::build_model_chain("gemini-3.8-flash");
        assert_eq!(
            chain,
            vec![
                "gemini-3.8-flash",
                "gemini-3.5-flash",
                "gemini-3.5-flash-lite"
            ]
        );
        assert_eq!(RETRY_BACKOFF_SECS, [1, 2, 4]);

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

    #[test]
    fn test_api_key_masking_and_http_error_classification() {
        assert_eq!(mask_api_key("AIzaSyD1234567890abcd"), "AIza••••••abcd");
        assert_eq!(mask_api_key("   "), "(Chưa cấu hình)");

        let (m401, ff401, _) = classify_http_error(401, "", "gemini-3.8-flash");
        assert!(ff401 && m401.contains("401/403"));

        let (m400, ff400, _) = classify_http_error(400, "API_KEY_INVALID", "gemini-3.8-flash");
        assert!(ff400 && m400.contains("400"));

        let (m404, ff404, fb404) = classify_http_error(404, "", "gemini-old");
        assert!(!ff404 && fb404 && m404.contains("404"));

        let (m429, ff429, fb429) = classify_http_error(429, "RESOURCE_EXHAUSTED", "gemini-3.8-flash");
        assert!(!ff429 && fb429 && m429.contains("429"));

        let (key, src) = GeminiChatClient::resolve_api_key_with_source("AIzaUIKey123456");
        assert_eq!(key, "AIzaUIKey123456");
        assert_eq!(src, ApiKeySource::UiSettings);
    }
}
