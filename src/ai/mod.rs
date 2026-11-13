pub mod face_auth;
pub mod gemini_chat;
pub mod latex_engine;
pub mod ocr_engine;
pub mod rag_engine;
pub mod stem_knowledge;
pub mod subtitles_engine;

use crate::utils::model_downloader::ModelDownloader;
use std::path::Path;
use std::sync::OnceLock;

static ORT_ENV: OnceLock<Option<ort::environment::Environment>> = OnceLock::new();

/// Initialize or retrieve the process-global ONNX Runtime `Environment` with CPU + DirectML execution providers.
pub fn global_ort_environment() -> Option<&'static ort::environment::Environment> {
    ORT_ENV
        .get_or_init(|| {
            let dll_candidates = [
                "onnxruntime.dll",
                "models/onnxruntime.dll",
                "assets/models/onnxruntime.dll",
            ];
            for dll in dll_candidates {
                let p = Path::new(dll);
                if p.exists() {
                    if let Ok(builder) = ort::init_from(p) {
                        let builder = builder.with_execution_providers([
                            ort::ep::CPU::default().build(),
                            ort::ep::DirectML::default().build(),
                        ]);
                        if let Ok(env) = builder.build() {
                            return Some(env);
                        }
                    }
                }
            }
            None
        })
        .as_ref()
}

/// Load an `ort::session::Session` from `models/<filename>` (downloaded by `ModelDownloader`) or adjacent paths.
pub fn try_create_ort_session(
    candidates: &[&str],
    embedded_bytes: &[u8],
) -> Option<ort::session::Session> {
    let env = global_ort_environment()?;

    for path_str in candidates {
        let p = Path::new(path_str);
        if ModelDownloader::is_valid_onnx_file(p) {
            if let Ok(bytes) = std::fs::read(p) {
                if let Ok(mut builder) = ort::session::Session::builder(env) {
                    if let Ok(session) = builder.commit_from_memory(&bytes) {
                        return Some(session);
                    }
                }
            }
        }
    }

    if !embedded_bytes.starts_with(b"ONNX_MODEL_PLACEHOLDER") && embedded_bytes.len() >= 128 {
        if let Ok(mut builder) = ort::session::Session::builder(env) {
            if let Ok(session) = builder.commit_from_memory(embedded_bytes) {
                return Some(session);
            }
        }
    }

    None
}
