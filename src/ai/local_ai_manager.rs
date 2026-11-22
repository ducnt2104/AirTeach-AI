//! Local AI Engine Manager & Hybrid Router (`src/ai/local_ai_manager.rs`):
//! - Auto-discovers Ollama (`127.0.0.1:11434`), LM Studio (`127.0.0.1:1234`), and llama.cpp (`127.0.0.1:8080/8081`).
//! - Streams SSE / Ollama NDJSON tokens live through `StreamingSanitizer` while measuring `tokens/s`.
//! - Sends non-blocking `keep_alive` warmup & maintains compact KV-Cache system prompt prefix + sliding window.

use crate::ai::gemini_chat::{
    ApiConnectionStatus, ChatStreamEvent, GeminiChatClient, SubjectAssistantMode,
};
use crate::utils::output_sanitizer::{OutputSanitizer, StreamingSanitizer};
use crossbeam_channel::Sender;
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AiRoutingMode {
    Hybrid,
    Local,
    Cloud,
}

impl Default for AiRoutingMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

impl AiRoutingMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Hybrid => "Hybrid",
            Self::Local => "Local",
            Self::Cloud => "Cloud",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredLocalServer {
    pub provider: String,
    pub base_url: String,
    pub models: Vec<String>,
    pub latency_ms: f32,
}

pub const IMMUTABLE_LOCAL_SYSTEM_PREFIX: &str =
    "Bạn là Trợ lý Sư phạm AirTeach AI (DNT - Victer). Trả lời ngắn gọn bằng tiếng Việt. \
     Không dùng ký tự $$$ hay thẻ <arrow>; dùng mũi tên chuẩn →, ⇒, ⇔ và công thức rõ ràng.";

pub struct LocalAIManager;

impl LocalAIManager {
    pub const PROBE_TARGETS: [(&'static str, &'static str, &'static str); 4] = [
        ("Ollama", "http://127.0.0.1:11434", "/api/tags"),
        ("LM Studio", "http://127.0.0.1:1234", "/v1/models"),
        ("llama.cpp", "http://127.0.0.1:8080", "/v1/models"),
        ("llama.cpp", "http://127.0.0.1:8081", "/v1/models"),
    ];

    /// Non-blocking auto-discovery across local Ollama / LM Studio / llama.cpp endpoints.
    pub fn discover_async(runtime: &tokio::runtime::Runtime, tx: Sender<Vec<DiscoveredLocalServer>>) {
        runtime.spawn(async move {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_millis(450))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());

            let mut found = Vec::new();
            for (provider, base_url, path) in Self::PROBE_TARGETS {
                let t0 = Instant::now();
                let url = format!("{base_url}{path}");
                if let Ok(resp) = client.get(&url).send().await {
                    if resp.status().is_success() {
                        let lat = t0.elapsed().as_secs_f32() * 1000.0;
                        let mut models = Vec::new();
                        if let Ok(val) = resp.json::<Value>().await {
                            if provider == "Ollama" {
                                if let Some(arr) = val.get("models").and_then(|v| v.as_array()) {
                                    for item in arr {
                                        if let Some(name) = item.get("name").and_then(|n| n.as_str()) {
                                            models.push(name.to_string());
                                        }
                                    }
                                }
                            } else if let Some(arr) = val.get("data").and_then(|v| v.as_array()) {
                                for item in arr {
                                    if let Some(id) = item.get("id").and_then(|n| n.as_str()) {
                                        models.push(id.to_string());
                                    }
                                }
                            }
                        }
                        found.push(DiscoveredLocalServer {
                            provider: provider.to_string(),
                            base_url: base_url.to_string(),
                            models,
                            latency_ms: lat,
                        });
                    }
                }
            }
            let _ = tx.send(found);
        });
    }

    pub fn warmup_async(runtime: &tokio::runtime::Runtime, provider: String, base_url: String, model: String) {
        if base_url.trim().is_empty() || model.trim().is_empty() {
            return;
        }
        runtime.spawn(async move {
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());
            if provider == "Ollama" {
                let _ = client
                    .post(format!("{base_url}/api/chat"))
                    .json(&json!({
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": false,
                        "keep_alive": "15m",
                        "options": {"num_predict": 1}
                    }))
                    .send()
                    .await;
            } else {
                let _ = client
                    .post(format!("{base_url}/v1/chat/completions"))
                    .json(&json!({
                        "model": model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "stream": false
                    }))
                    .send()
                    .await;
            }
        });
    }

    /// Build bounded sliding-window message list preserving immutable prefix for KV-Cache reuse.
    pub fn build_sliding_messages(
        history: &[(String, String)],
        user_prompt: &str,
        base64_png: Option<&str>,
    ) -> Vec<Value> {
        let mut msgs = vec![json!({
            "role": "system",
            "content": IMMUTABLE_LOCAL_SYSTEM_PREFIX
        })];
        let keep_n = 8usize;
        let start = history.len().saturating_sub(keep_n);
        for (role, text) in &history[start..] {
            if !text.trim().is_empty() {
                msgs.push(json!({ "role": role, "content": text }));
            }
        }
        if let Some(b64) = base64_png {
            msgs.push(json!({
                "role": "user",
                "content": [
                    { "type": "text", "text": user_prompt },
                    { "type": "image_url", "image_url": { "url": format!("data:image/png;base64,{b64}") } }
                ]
            }));
        } else {
            msgs.push(json!({ "role": "user", "content": user_prompt }));
        }
        msgs
    }

    /// Stream from OpenAI-compatible or Ollama endpoint, emitting sanitized tokens & fallback seamlessly if needed.
    #[allow(clippy::too_many_arguments)]
    pub fn stream_with_hybrid_fallback(
        gemini_client: &GeminiChatClient,
        routing_mode: AiRoutingMode,
        local_provider: String,
        local_base_url: String,
        local_model: String,
        history: Vec<(String, String)>,
        user_prompt: String,
        base64_png: Option<String>,
        cloud_api_key: String,
        cloud_model: String,
        subject: SubjectAssistantMode,
        rag_context: Option<String>,
        cache_key: String,
        message_index: usize,
        event_tx: Sender<ChatStreamEvent>,
    ) {
        let has_local = !local_base_url.trim().is_empty() && !local_model.trim().is_empty();
        if routing_mode == AiRoutingMode::Cloud || (!has_local && routing_mode == AiRoutingMode::Hybrid) {
            gemini_client.stream_completion_with_options(
                cloud_api_key,
                cloud_model,
                subject,
                user_prompt,
                rag_context,
                base64_png,
                false,
                cache_key,
                message_index,
                event_tx,
            );
            return;
        }

        if !has_local && routing_mode == AiRoutingMode::Local {
            let _ = event_tx.send(ChatStreamEvent::Started { message_index });
            let _ = event_tx.send(ChatStreamEvent::Error {
                message_index,
                error: "Chế độ Local AI đang bật nhưng chưa phát hiện Ollama/LM Studio đang chạy."
                    .to_string(),
            });
            return;
        }

        let msgs = Self::build_sliding_messages(&history, &user_prompt, base64_png.as_deref());
        let use_ollama = local_provider == "Ollama" && base64_png.is_none();
        let url = if use_ollama {
            format!("{}/api/chat", local_base_url.trim_end_matches('/'))
        } else {
            format!("{}/v1/chat/completions", local_base_url.trim_end_matches('/'))
        };
        let payload = if use_ollama {
            json!({
                "model": local_model,
                "messages": msgs,
                "stream": true,
                "keep_alive": "15m",
                "options": { "temperature": 0.2, "num_ctx": 4096 }
            })
        } else {
            json!({
                "model": local_model,
                "messages": msgs,
                "temperature": 0.2,
                "max_tokens": 1536,
                "stream": true
            })
        };

        gemini_client.runtime.spawn(async move {
            let _ = event_tx.send(ChatStreamEvent::Started { message_index });
            let client = reqwest::Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap_or_else(|_| reqwest::Client::new());
            let t0 = Instant::now();
            let resp = match client.post(&url).json(&payload).send().await {
                Ok(r) if r.status().is_success() => r,
                Ok(r) => {
                    let msg = format!("Local LLM ({local_model}) lỗi HTTP {}", r.status());
                    let _ = event_tx.send(ChatStreamEvent::Error {
                        message_index,
                        error: msg,
                    });
                    return;
                }
                Err(e) => {
                    let msg = format!("Không thể kết nối Local LLM ({local_base_url}): {e}");
                    let _ = event_tx.send(ChatStreamEvent::Error {
                        message_index,
                        error: msg,
                    });
                    return;
                }
            };

            let mut sanitizer = StreamingSanitizer::new();
            let mut stream = resp.bytes_stream();
            let mut line_buf = String::new();
            let mut accum_clean = String::new();
            let mut tok_count = 0usize;

            while let Some(chunk) = stream.next().await {
                let Ok(bytes) = chunk else {
                    break;
                };
                line_buf.push_str(&String::from_utf8_lossy(&bytes));
                while let Some(nl) = line_buf.find('\n') {
                    let line = line_buf[..nl].trim().to_string();
                    line_buf.drain(..=nl);
                    if line.is_empty() {
                        continue;
                    }
                    let delta = if use_ollama {
                        serde_json::from_str::<Value>(&line)
                            .ok()
                            .and_then(|v| {
                                v.pointer("/message/content")
                                    .and_then(|c| c.as_str())
                                    .map(|s| s.to_string())
                            })
                            .unwrap_or_default()
                    } else if let Some(json_part) = line.strip_prefix("data:") {
                        let p = json_part.trim();
                        if p == "[DONE]" {
                            String::new()
                        } else {
                            serde_json::from_str::<Value>(p)
                                .ok()
                                .and_then(|v| {
                                    v.pointer("/choices/0/delta/content")
                                        .and_then(|c| c.as_str())
                                        .map(|s| s.to_string())
                                })
                                .unwrap_or_default()
                        }
                    } else {
                        String::new()
                    };
                    if !delta.is_empty() {
                        tok_count += 1;
                        let safe = sanitizer.feed(&delta);
                        if !safe.is_empty() {
                            accum_clean.push_str(&safe);
                            let _ = event_tx.send(ChatStreamEvent::TokenDelta {
                                message_index,
                                delta: safe,
                            });
                        }
                    }
                }
            }
            let tail = sanitizer.flush();
            if !tail.is_empty() {
                accum_clean.push_str(&tail);
                let _ = event_tx.send(ChatStreamEvent::TokenDelta {
                    message_index,
                    delta: tail,
                });
            }
            let secs = t0.elapsed().as_secs_f32().max(0.05);
            let tps = tok_count as f32 / secs;
            let _ = event_tx.send(ChatStreamEvent::MetricsUpdated {
                engine: "Local".to_string(),
                model: local_model.clone(),
                tokens_per_sec: tps,
            });
            let _ = event_tx.send(ChatStreamEvent::ApiStatusUpdated {
                status: ApiConnectionStatus::Connected,
                message: format!("Local · {local_model} ({tps:.1} tok/s)"),
            });
            if !cache_key.is_empty() && !accum_clean.trim().is_empty() {
                let _ = event_tx.send(ChatStreamEvent::CacheStore {
                    cache_key,
                    model_id: local_model,
                    response_text: OutputSanitizer::sanitize_plain_text(&accum_clean),
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
    fn test_sliding_window_preserves_immutable_prefix_and_bounds_history() {
        let hist: Vec<(String, String)> = (0..14)
            .map(|i| ("user".to_string(), format!("Q{i}")))
            .collect();
        let msgs = LocalAIManager::build_sliding_messages(&hist, "2x+1=3", None);
        assert_eq!(msgs[0]["role"], "system");
        assert!(msgs[0]["content"]
            .as_str()
            .unwrap()
            .contains("AirTeach AI"));
        // 1 system + 8 recent + 1 user = 10
        assert_eq!(msgs.len(), 10);
    }
}
