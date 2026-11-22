//! Encrypted Local-First SQLite Storage (`src/utils/storage.rs`):
//! - Bundled static SQLite (`rusqlite` in WAL mode) for `users`, `sessions`, `app_config`, and `lecture_history`
//! - Clean Slate Policy: zero pre-seeded accounts, zero default PINs, zero hardcoded personal/school names
//! - `aes-gcm` (AES-256-GCM with 96-bit random CSPRNG nonces) protects Gemini API keys and `.airteach` exports
//! - Biometric embeddings are stored strictly as normalized numeric vectors (`face_embedding`) — never raw photos

use crate::ai::face_auth::{
    normalize_l2_in_place, FaceAuthEngine, UserAccount, DEFAULT_FACE_MATCH_THRESHOLD,
    DEFAULT_SFACE_MATCH_THRESHOLD, EMBEDDING_DIM, SFACE_EMBEDDING_DIM,
};
use crate::ai::local_ai_manager::AiRoutingMode;
use crate::core::camera::{CaroGridSpacing, FACE_QUALITY_GATE_MIN_SHARPNESS};
use crate::core::stroke_engine::{Point2D, SnappedShape};
use crate::ui::brand::ThemeMode;
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Key, Nonce,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::{Duration as ChronoDuration, Utc};
use rand::RngCore;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// High-contrast classroom accessibility display modes for projectors and smartboards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ClassroomContrastMode {
    DarkOled,
    ChalkboardGreen,
    WhiteboardClean,
}

impl Default for ClassroomContrastMode {
    fn default() -> Self {
        Self::DarkOled
    }
}

impl ClassroomContrastMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::DarkOled => "Dark OLED (#0f172a)",
            Self::ChalkboardGreen => "Bảng Phấn Xanh (#0f2e1f)",
            Self::WhiteboardClean => "Bảng Trắng Máy Chiếu (#f8fafc)",
        }
    }

    pub const fn canvas_bg_rgb(self) -> [u8; 3] {
        match self {
            Self::DarkOled => [0x0f, 0x17, 0x2a],
            Self::ChalkboardGreen => [0x0f, 0x2e, 0x1f],
            Self::WhiteboardClean => [0xf8, 0xfa, 0xfc],
        }
    }
}

/// Dynamic classroom layout mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoardLayoutMode {
    SlidePresentation,
    FullWhiteboard,
    SplitScreenAi,
}

impl Default for BoardLayoutMode {
    fn default() -> Self {
        Self::SplitScreenAi
    }
}

impl BoardLayoutMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::SlidePresentation => "Trình chiếu Slide",
            Self::FullWhiteboard => "Bảng Trắng Toàn màn hình",
            Self::SplitScreenAi => "Chia đôi Bảng + AI Co-Pilot",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorStroke {
    pub timestamp_ms: u64,
    pub points: Vec<Point2D>,
    pub color: [u8; 4],
    pub brush_size: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedStroke {
    #[serde(default)]
    pub timestamp_ms: u64,
    pub raw_points: Vec<Point2D>,
    pub display_points: Vec<Point2D>,
    pub color_rgba: [u8; 4],
    pub thickness: f32,
    pub snapped_shape: SnappedShape,
}

impl PersistedStroke {
    pub fn to_vector_stroke(&self) -> VectorStroke {
        VectorStroke {
            timestamp_ms: self.timestamp_ms,
            points: self.display_points.clone(),
            color: self.color_rgba,
            brush_size: self.thickness,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoardTextAnnotation {
    pub position: Point2D,
    pub text: String,
    pub is_latex: bool,
    pub confidence: f32,
    #[serde(default = "default_annotation_color")]
    pub color_rgba: [u8; 4],
}

const fn default_annotation_color() -> [u8; 4] {
    [6, 182, 212, 255]
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AirTeachBundle {
    #[serde(default)]
    pub version: String,
    pub title: String,
    pub subject: String,
    pub created_at: String,
    pub slide_index: usize,
    pub strokes: Vec<PersistedStroke>,
    #[serde(default)]
    pub vector_strokes: Vec<VectorStroke>,
    pub annotations: Vec<BoardTextAnnotation>,
    #[serde(default)]
    pub audio_wav_path: Option<String>,
    #[serde(default)]
    pub subtitles_srt_path: Option<String>,
}

/// Global hardware, UI theme & tutorial preferences (NO personal names, NO default PINs or fake biometrics).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub theme_mode: ThemeMode,
    #[serde(default = "default_lecture_title")]
    pub lecture_title: String,
    pub encrypted_api_key_b64: String,
    pub gemini_model: String,
    pub camera_index: u32,
    #[serde(default = "default_face_threshold")]
    pub face_match_threshold: f32,
    #[serde(default = "default_sface_threshold")]
    pub sface_match_threshold: f32,
    #[serde(default = "default_hand_confidence_min")]
    pub hand_confidence_min: f32,
    #[serde(default = "default_quality_gate_sharpness")]
    pub quality_gate_min_sharpness: f32,
    #[serde(default = "default_gesture_hold_frames")]
    pub gesture_hold_frames: u8,
    pub canvas_grid_spacing: CaroGridSpacing,
    pub preview_grid_spacing: CaroGridSpacing,
    #[serde(default = "default_true")]
    pub canvas_grid_enabled: bool,
    #[serde(default = "default_grid_px")]
    pub canvas_grid_px: u32,
    #[serde(default = "default_true")]
    pub hud_grid_enabled: bool,
    #[serde(default = "default_grid_px")]
    pub hud_grid_px: u32,
    #[serde(default)]
    pub contrast_mode: ClassroomContrastMode,
    #[serde(default)]
    pub layout_mode: BoardLayoutMode,
    #[serde(default = "default_true")]
    pub floating_glass_toolbar: bool,
    #[serde(default = "default_true")]
    pub live_subtitles_enabled: bool,
    pub one_euro_min_cutoff: f32,
    pub one_euro_beta: f32,
    pub one_euro_d_cutoff: f32,
    pub rdp_epsilon: f32,
    pub auto_shape_snap: bool,
    #[serde(default)]
    pub hold_to_snap_350ms: bool,
    pub enable_clahe: bool,
    pub clahe_clip_limit: f32,
    pub unsharp_amount: f32,
    #[serde(default)]
    pub tutorial_dismissed: bool,
    #[serde(default)]
    pub tutorial_deleted: bool,
    #[serde(default)]
    pub ai_routing_mode: AiRoutingMode,
    #[serde(default)]
    pub local_endpoint: String,
    #[serde(default)]
    pub local_model: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            theme_mode: ThemeMode::Dark,
            lecture_title: default_lecture_title(),
            encrypted_api_key_b64: String::new(),
            gemini_model: "gemini-3.8-flash".to_string(),
            camera_index: 0,
            face_match_threshold: DEFAULT_FACE_MATCH_THRESHOLD,
            sface_match_threshold: DEFAULT_SFACE_MATCH_THRESHOLD,
            hand_confidence_min: 0.50,
            quality_gate_min_sharpness: FACE_QUALITY_GATE_MIN_SHARPNESS,
            gesture_hold_frames: 6,
            canvas_grid_spacing: CaroGridSpacing::Grid20px,
            preview_grid_spacing: CaroGridSpacing::Grid20px,
            canvas_grid_enabled: true,
            canvas_grid_px: 24,
            hud_grid_enabled: true,
            hud_grid_px: 24,
            contrast_mode: ClassroomContrastMode::DarkOled,
            layout_mode: BoardLayoutMode::SplitScreenAi,
            floating_glass_toolbar: true,
            live_subtitles_enabled: true,
            one_euro_min_cutoff: 1.35,
            one_euro_beta: 0.012,
            one_euro_d_cutoff: 1.0,
            rdp_epsilon: 1.8,
            auto_shape_snap: true,
            hold_to_snap_350ms: false,
            enable_clahe: true,
            clahe_clip_limit: 2.2,
            unsharp_amount: 0.75,
            tutorial_dismissed: false,
            tutorial_deleted: false,
            ai_routing_mode: AiRoutingMode::Hybrid,
            local_endpoint: String::new(),
            local_model: String::new(),
        }
    }
}

fn default_lecture_title() -> String {
    "Bảng trắng mới".to_string()
}

const fn default_face_threshold() -> f32 {
    DEFAULT_FACE_MATCH_THRESHOLD
}

const fn default_sface_threshold() -> f32 {
    DEFAULT_SFACE_MATCH_THRESHOLD
}

const fn default_hand_confidence_min() -> f32 {
    0.50
}

const fn default_quality_gate_sharpness() -> f32 {
    FACE_QUALITY_GATE_MIN_SHARPNESS
}

const fn default_gesture_hold_frames() -> u8 {
    6
}

const fn default_true() -> bool {
    true
}

const fn default_grid_px() -> u32 {
    24
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthSession {
    pub id: String,
    pub user_id: String,
    pub created_at: String,
    pub expires_at: String,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct LectureHistoryEntry {
    pub id: i64,
    pub owner_user_id: String,
    pub is_tutorial: bool,
    pub title: String,
    pub subject: String,
    pub created_at: String,
    pub stroke_count: usize,
    pub audio_wav_path: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredFacePayload {
    avg: Vec<f32>,
    #[serde(default)]
    templates: Vec<Vec<f32>>,
}

fn decode_face_payload(emb_json: &str) -> (Vec<f32>, Vec<Vec<f32>>) {
    let trimmed = emb_json.trim();
    if trimmed.is_empty() {
        return (Vec::new(), Vec::new());
    }
    if let Ok(mut payload) = serde_json::from_str::<StoredFacePayload>(trimmed) {
        if payload.avg.len() == EMBEDDING_DIM || payload.avg.len() == SFACE_EMBEDDING_DIM {
            normalize_l2_in_place(&mut payload.avg);
        }
        for tmpl in &mut payload.templates {
            if !tmpl.is_empty() {
                normalize_l2_in_place(tmpl);
            }
        }
        if payload.templates.is_empty() && !payload.avg.is_empty() {
            payload.templates.push(payload.avg.clone());
        }
        return (payload.avg, payload.templates);
    }
    let mut legacy = serde_json::from_str::<Vec<f32>>(trimmed).unwrap_or_default();
    if legacy.len() == EMBEDDING_DIM || legacy.len() == SFACE_EMBEDDING_DIM {
        normalize_l2_in_place(&mut legacy);
    }
    let templates = if legacy.is_empty() {
        Vec::new()
    } else {
        vec![legacy.clone()]
    };
    (legacy, templates)
}

fn encode_face_payload(templates: &[Vec<f32>], fallback: &[f32]) -> (Vec<f32>, Vec<Vec<f32>>, String) {
    let mut clean_templates: Vec<Vec<f32>> = templates
        .iter()
        .filter(|v| !v.is_empty())
        .map(|v| {
            let mut copy = v.clone();
            normalize_l2_in_place(&mut copy);
            copy
        })
        .take(5)
        .collect();

    if clean_templates.is_empty() && !fallback.is_empty() {
        let mut copy = fallback.to_vec();
        normalize_l2_in_place(&mut copy);
        clean_templates.push(copy);
    }

    let avg = FaceAuthEngine::average_embeddings_dynamic(&clean_templates).unwrap_or_default();
    let json = if clean_templates.is_empty() {
        String::new()
    } else {
        serde_json::to_string(&StoredFacePayload {
            avg: avg.clone(),
            templates: clean_templates.clone(),
        })
        .unwrap_or_default()
    };
    (avg, clean_templates, json)
}

pub struct LocalStorage {
    conn: Connection,
    master_key: [u8; 32],
}

impl LocalStorage {
    pub fn open_default() -> Self {
        let db_path = Self::resolve_db_path();
        Self::open_at_path(&db_path).unwrap_or_else(|_| Self::open_in_memory())
    }

    pub fn open_at_path(path: &Path) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let conn = Connection::open(path).map_err(|e| format!("Failed to open SQLite DB: {e}"))?;
        let storage = Self {
            conn,
            master_key: Self::derive_machine_aes256_key(),
        };
        storage.init_schema()?;
        Ok(storage)
    }

    pub fn open_in_memory() -> Self {
        let conn = Connection::open_in_memory().expect("Failed to open in-memory SQLite");
        let storage = Self {
            conn,
            master_key: Self::derive_machine_aes256_key(),
        };
        let _ = storage.init_schema();
        storage
    }

    fn init_schema(&self) -> Result<(), String> {
        self.conn
            .execute_batch(
                "PRAGMA journal_mode = WAL;
                 PRAGMA synchronous = NORMAL;
                 PRAGMA foreign_keys = ON;
                 CREATE TABLE IF NOT EXISTS users (
                     id TEXT PRIMARY KEY,
                     username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                     display_name TEXT NOT NULL,
                     password_hash TEXT NOT NULL,
                     face_embedding TEXT NOT NULL DEFAULT '',
                     created_at TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS sessions (
                     id TEXT PRIMARY KEY,
                     user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                     created_at TEXT NOT NULL,
                     expires_at TEXT NOT NULL,
                     active INTEGER NOT NULL DEFAULT 1
                 );
                 CREATE TABLE IF NOT EXISTS app_config (
                     id INTEGER PRIMARY KEY CHECK (id = 1),
                     json_payload TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS lecture_history (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     owner_user_id TEXT NOT NULL DEFAULT '',
                     is_tutorial INTEGER NOT NULL DEFAULT 0,
                     title TEXT NOT NULL,
                     subject TEXT NOT NULL,
                     created_at TEXT NOT NULL,
                     stroke_count INTEGER NOT NULL,
                     encrypted_bundle_b64 TEXT NOT NULL
                 );
                 CREATE TABLE IF NOT EXISTS ai_response_cache (
                     cache_key TEXT PRIMARY KEY,
                     model_id TEXT NOT NULL,
                     response_text TEXT NOT NULL,
                     structured_json TEXT NOT NULL DEFAULT '',
                     created_at TEXT NOT NULL
                 );",
            )
            .map_err(|e| format!("SQLite schema error: {e}"))?;

        // Lightweight additive migration if upgrading an older lecture_history table
        let _ = self.conn.execute(
            "ALTER TABLE lecture_history ADD COLUMN owner_user_id TEXT NOT NULL DEFAULT ''",
            [],
        );
        let _ = self.conn.execute(
            "ALTER TABLE lecture_history ADD COLUMN is_tutorial INTEGER NOT NULL DEFAULT 0",
            [],
        );
        Ok(())
    }

    pub fn new_uuid_v4() -> String {
        let mut bytes = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut bytes);
        bytes[6] = (bytes[6] & 0x0f) | 0x40;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        format!(
            "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5],
            bytes[6], bytes[7],
            bytes[8], bytes[9],
            bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        )
    }

    pub fn resolve_recordings_dir() -> PathBuf {
        if let Ok(home) = std::env::var("USERPROFILE") {
            let dir = PathBuf::from(home).join(".airteach").join("recordings");
            if std::fs::create_dir_all(&dir).is_ok() {
                return dir;
            }
        }
        let local = PathBuf::from("recordings");
        let _ = std::fs::create_dir_all(&local);
        local
    }

    fn resolve_db_path() -> PathBuf {
        let local_candidate = PathBuf::from("airteach_data.sqlite3");
        if let Ok(home) = std::env::var("USERPROFILE") {
            let dir = PathBuf::from(home).join(".airteach");
            if std::fs::create_dir_all(&dir).is_ok() {
                return dir.join("airteach_data.sqlite3");
            }
        }
        local_candidate
    }

    fn derive_machine_aes256_key() -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"AIRTEACH_AI_DNT_VICTER_AES256_GCM_SALT_V2:");
        if let Ok(user) = std::env::var("USERNAME") {
            hasher.update(user.as_bytes());
        }
        if let Ok(comp) = std::env::var("COMPUTERNAME") {
            hasher.update(comp.as_bytes());
        }
        let digest = hasher.finalize();
        let mut key = [0u8; 32];
        key.copy_from_slice(&digest[..32]);
        key
    }

    pub fn encrypt_bytes_b64(&self, plaintext: &[u8]) -> Result<String, String> {
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&self.master_key));
        let mut nonce_bytes = [0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext)
            .map_err(|e| format!("AES-256-GCM encryption error: {e}"))?;

        let mut combined = Vec::with_capacity(12 + ciphertext.len());
        combined.extend_from_slice(&nonce_bytes);
        combined.extend_from_slice(&ciphertext);
        Ok(BASE64.encode(combined))
    }

    pub fn decrypt_bytes_b64(&self, encoded_b64: &str) -> Result<Vec<u8>, String> {
        if encoded_b64.trim().is_empty() {
            return Ok(Vec::new());
        }
        let raw = BASE64
            .decode(encoded_b64.trim())
            .map_err(|e| format!("Base64 decode error: {e}"))?;
        if raw.len() < 12 {
            return Err("Encrypted payload shorter than 12-byte GCM nonce".to_string());
        }
        let (nonce_bytes, ciphertext) = raw.split_at(12);
        let cipher = Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(&self.master_key));
        let nonce = Nonce::from_slice(nonce_bytes);
        cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| format!("AES-256-GCM decryption error: {e}"))
    }

    // =========================================================================
    // USER ACCOUNTS & SESSIONS (SQLite Local-First Auth)
    // =========================================================================

    pub fn username_exists(&self, username: &str) -> Result<bool, String> {
        let count: i64 = self
            .conn
            .query_row(
                "SELECT COUNT(1) FROM users WHERE username = ?1 COLLATE NOCASE",
                params![username.trim()],
                |row| row.get(0),
            )
            .map_err(|e| format!("SQLite query error: {e}"))?;
        Ok(count > 0)
    }

    pub fn create_user(
        &self,
        username: &str,
        display_name: &str,
        password_hash: &str,
        face_embedding: &[f32],
    ) -> Result<UserAccount, String> {
        let templates = if face_embedding.is_empty() {
            Vec::new()
        } else {
            vec![face_embedding.to_vec()]
        };
        self.create_user_with_templates(username, display_name, password_hash, &templates)
    }

    pub fn create_user_with_templates(
        &self,
        username: &str,
        display_name: &str,
        password_hash: &str,
        face_templates: &[Vec<f32>],
    ) -> Result<UserAccount, String> {
        let clean_username = username.trim().to_string();
        let clean_display = display_name.trim().to_string();
        if self.username_exists(&clean_username)? {
            return Err(format!("Tên đăng nhập '{clean_username}' đã tồn tại."));
        }

        let (norm_avg, norm_templates, emb_json) = encode_face_payload(face_templates, &[]);

        let user = UserAccount {
            id: Self::new_uuid_v4(),
            username: clean_username,
            display_name: clean_display,
            password_hash: password_hash.to_string(),
            face_embedding: norm_avg,
            face_templates: norm_templates,
            created_at: Utc::now().to_rfc3339(),
        };

        self.conn
            .execute(
                "INSERT INTO users (id, username, display_name, password_hash, face_embedding, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    user.id,
                    user.username,
                    user.display_name,
                    user.password_hash,
                    emb_json,
                    user.created_at
                ],
            )
            .map_err(|e| format!("Không thể tạo tài khoản SQLite: {e}"))?;

        Ok(user)
    }

    fn row_to_user(row: &rusqlite::Row<'_>) -> rusqlite::Result<UserAccount> {
        let id: String = row.get(0)?;
        let username: String = row.get(1)?;
        let display_name: String = row.get(2)?;
        let password_hash: String = row.get(3)?;
        let emb_json: String = row.get(4)?;
        let created_at: String = row.get(5)?;
        let (face_embedding, face_templates) = decode_face_payload(&emb_json);
        Ok(UserAccount {
            id,
            username,
            display_name,
            password_hash,
            face_embedding,
            face_templates,
            created_at,
        })
    }

    pub fn list_users(&self) -> Result<Vec<UserAccount>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, username, display_name, password_hash, face_embedding, created_at
                 FROM users
                 ORDER BY created_at ASC",
            )
            .map_err(|e| format!("SQLite prepare error: {e}"))?;
        let rows = stmt
            .query_map([], Self::row_to_user)
            .map_err(|e| format!("SQLite query_map error: {e}"))?;
        let mut out = Vec::new();
        for r in rows {
            out.push(r.map_err(|e| format!("SQLite row error: {e}"))?);
        }
        Ok(out)
    }

    pub fn find_user_by_username(&self, username: &str) -> Result<Option<UserAccount>, String> {
        self.conn
            .query_row(
                "SELECT id, username, display_name, password_hash, face_embedding, created_at
                 FROM users WHERE username = ?1 COLLATE NOCASE",
                params![username.trim()],
                Self::row_to_user,
            )
            .optional()
            .map_err(|e| format!("SQLite lookup error: {e}"))
    }

    pub fn update_user_password(&self, user_id: &str, new_hash: &str) -> Result<(), String> {
        let rows = self
            .conn
            .execute(
                "UPDATE users SET password_hash = ?1 WHERE id = ?2",
                params![new_hash, user_id],
            )
            .map_err(|e| format!("Không thể cập nhật mật khẩu: {e}"))?;
        if rows == 0 {
            return Err("Không tìm thấy tài khoản để cập nhật mật khẩu.".to_string());
        }
        Ok(())
    }

    pub fn update_user_face_embedding(
        &self,
        user_id: &str,
        face_embedding: &[f32],
    ) -> Result<(), String> {
        let templates = if face_embedding.is_empty() {
            Vec::new()
        } else {
            vec![face_embedding.to_vec()]
        };
        self.update_user_face_templates(user_id, &templates)
    }

    pub fn update_user_face_templates(
        &self,
        user_id: &str,
        face_templates: &[Vec<f32>],
    ) -> Result<(), String> {
        let (_, _, json) = encode_face_payload(face_templates, &[]);
        self.conn
            .execute(
                "UPDATE users SET face_embedding = ?1 WHERE id = ?2",
                params![json, user_id],
            )
            .map_err(|e| format!("Không thể cập nhật vector khuôn mặt: {e}"))?;
        Ok(())
    }

    pub fn create_session(&self, user_id: &str) -> Result<AuthSession, String> {
        let _ = self
            .conn
            .execute("UPDATE sessions SET active = 0 WHERE active = 1", []);

        let now = Utc::now();
        let expires = now + ChronoDuration::hours(24);
        let session = AuthSession {
            id: Self::new_uuid_v4(),
            user_id: user_id.to_string(),
            created_at: now.to_rfc3339(),
            expires_at: expires.to_rfc3339(),
            active: true,
        };

        self.conn
            .execute(
                "INSERT INTO sessions (id, user_id, created_at, expires_at, active)
                 VALUES (?1, ?2, ?3, ?4, 1)",
                params![
                    session.id,
                    session.user_id,
                    session.created_at,
                    session.expires_at
                ],
            )
            .map_err(|e| format!("Không thể tạo phiên đăng nhập SQLite: {e}"))?;

        Ok(session)
    }

    pub fn get_active_session_user(&self) -> Result<Option<(AuthSession, UserAccount)>, String> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT s.id, s.user_id, s.created_at, s.expires_at, s.active,
                        u.id, u.username, u.display_name, u.password_hash, u.face_embedding, u.created_at
                 FROM sessions s
                 INNER JOIN users u ON u.id = s.user_id
                 WHERE s.active = 1
                 ORDER BY s.created_at DESC
                 LIMIT 1",
            )
            .map_err(|e| format!("SQLite session query error: {e}"))?;

        let result = stmt
            .query_row([], |row| {
                let session = AuthSession {
                    id: row.get(0)?,
                    user_id: row.get(1)?,
                    created_at: row.get(2)?,
                    expires_at: row.get(3)?,
                    active: row.get::<_, i64>(4)? != 0,
                };
                let emb_json: String = row.get(9)?;
                let (face_embedding, face_templates) = decode_face_payload(&emb_json);
                let user = UserAccount {
                    id: row.get(5)?,
                    username: row.get(6)?,
                    display_name: row.get(7)?,
                    password_hash: row.get(8)?,
                    face_embedding,
                    face_templates,
                    created_at: row.get(10)?,
                };
                Ok((session, user))
            })
            .optional()
            .map_err(|e| format!("SQLite active session read error: {e}"))?;

        Ok(result)
    }

    pub fn clear_active_sessions(&self) -> Result<(), String> {
        self.conn
            .execute("UPDATE sessions SET active = 0 WHERE active = 1", [])
            .map_err(|e| format!("Không thể kết thúc phiên làm việc: {e}"))?;
        Ok(())
    }

    // =========================================================================
    // CLEAN SLATE APP CONFIG
    // =========================================================================

    pub fn load_or_init_config(&self) -> AppConfig {
        if let Ok(json_str) =
            self.conn
                .query_row("SELECT json_payload FROM app_config WHERE id = 1", [], |row| {
                    row.get::<_, String>(0)
                })
        {
            if let Ok(mut cfg) = serde_json::from_str::<AppConfig>(&json_str) {
                if cfg.gemini_model.trim().is_empty() || cfg.gemini_model == "gemini-2.0-flash" {
                    cfg.gemini_model = "gemini-3.8-flash".to_string();
                    self.save_config(&cfg);
                }
                return cfg;
            }
        }

        let cfg = AppConfig::default();
        self.save_config(&cfg);
        cfg
    }

    /// Retrieve cached AI pedagogical response (`model_id, response_text, structured_json`).
    pub fn get_cached_ai_response(&self, cache_key: &str) -> Option<(String, String, String)> {
        self.conn
            .query_row(
                "SELECT model_id, response_text, structured_json
                 FROM ai_response_cache WHERE cache_key = ?1",
                params![cache_key],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()
            .ok()
            .flatten()
    }

    pub fn put_cached_ai_response(
        &self,
        cache_key: &str,
        model_id: &str,
        response_text: &str,
        structured_json: &str,
    ) {
        let now = Utc::now().to_rfc3339();
        let _ = self.conn.execute(
            "INSERT INTO ai_response_cache (cache_key, model_id, response_text, structured_json, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(cache_key) DO UPDATE SET
                model_id = excluded.model_id,
                response_text = excluded.response_text,
                structured_json = excluded.structured_json,
                created_at = excluded.created_at",
            params![cache_key, model_id, response_text, structured_json, now],
        );
    }

    pub fn save_config(&self, cfg: &AppConfig) {
        if let Ok(json_str) = serde_json::to_string(cfg) {
            let _ = self.conn.execute(
                "INSERT INTO app_config (id, json_payload) VALUES (1, ?1)
                 ON CONFLICT(id) DO UPDATE SET json_payload = excluded.json_payload",
                params![json_str],
            );
        }
    }

    pub fn get_decrypted_api_key(&self, cfg: &AppConfig) -> String {
        self.decrypt_bytes_b64(&cfg.encrypted_api_key_b64)
            .ok()
            .and_then(|b| String::from_utf8(b).ok())
            .unwrap_or_default()
    }

    pub fn set_encrypted_api_key(&self, cfg: &mut AppConfig, plain_api_key: &str) {
        if let Ok(enc) = self.encrypt_bytes_b64(plain_api_key.trim().as_bytes()) {
            cfg.encrypted_api_key_b64 = enc;
            self.save_config(cfg);
        }
    }

    pub fn clear_encrypted_api_key(&self, cfg: &mut AppConfig) {
        cfg.encrypted_api_key_b64.clear();
        self.save_config(cfg);
    }

    // =========================================================================
    // LECTURE HISTORY & TUTORIAL RECORDS
    // =========================================================================

    pub fn save_lecture_bundle_to_db(
        &self,
        owner_user_id: &str,
        is_tutorial: bool,
        bundle: &AirTeachBundle,
    ) -> Result<i64, String> {
        let json_bytes = serde_json::to_vec(bundle).map_err(|e| e.to_string())?;
        let encrypted_b64 = self.encrypt_bytes_b64(&json_bytes)?;
        self.conn
            .execute(
                "INSERT INTO lecture_history (owner_user_id, is_tutorial, title, subject, created_at, stroke_count, encrypted_bundle_b64)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    owner_user_id,
                    if is_tutorial { 1i64 } else { 0i64 },
                    bundle.title,
                    bundle.subject,
                    bundle.created_at,
                    bundle.strokes.len() as i64,
                    encrypted_b64
                ],
            )
            .map_err(|e| e.to_string())?;
        Ok(self.conn.last_insert_rowid())
    }

    pub fn list_lecture_history(&self, owner_user_id: &str) -> Vec<LectureHistoryEntry> {
        let mut out = Vec::new();
        let Ok(mut stmt) = self.conn.prepare(
            "SELECT id, owner_user_id, is_tutorial, title, subject, created_at, stroke_count, encrypted_bundle_b64
             FROM lecture_history
             WHERE owner_user_id = ?1 OR owner_user_id = ''
             ORDER BY id DESC
             LIMIT 25",
        ) else {
            return out;
        };

        if let Ok(rows) = stmt.query_map(params![owner_user_id], |row| {
            let id: i64 = row.get(0)?;
            let owner_uid: String = row.get(1)?;
            let is_tut: i64 = row.get(2)?;
            let title: String = row.get(3)?;
            let subject: String = row.get(4)?;
            let created_at: String = row.get(5)?;
            let stroke_count = row.get::<_, i64>(6)? as usize;
            let enc_b64: String = row.get(7)?;
            let audio_wav_path = self
                .decrypt_bytes_b64(&enc_b64)
                .ok()
                .and_then(|plain| serde_json::from_slice::<AirTeachBundle>(&plain).ok())
                .and_then(|b| b.audio_wav_path);

            Ok(LectureHistoryEntry {
                id,
                owner_user_id: owner_uid,
                is_tutorial: is_tut != 0,
                title,
                subject,
                created_at,
                stroke_count,
                audio_wav_path,
            })
        }) {
            for r in rows.flatten() {
                out.push(r);
            }
        }
        out
    }

    pub fn delete_tutorial_records(&self) -> Result<usize, String> {
        self.conn
            .execute("DELETE FROM lecture_history WHERE is_tutorial = 1", [])
            .map_err(|e| format!("Không thể xóa bài hướng dẫn: {e}"))
    }

    pub fn load_lecture_bundle_from_db(&self, id: i64) -> Result<AirTeachBundle, String> {
        let encrypted_b64: String = self
            .conn
            .query_row(
                "SELECT encrypted_bundle_b64 FROM lecture_history WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;

        let plain = self.decrypt_bytes_b64(&encrypted_b64)?;
        serde_json::from_slice(&plain).map_err(|e| e.to_string())
    }

    pub fn export_airteach_file(&self, path: &Path, bundle: &AirTeachBundle) -> Result<(), String> {
        let json_bytes = serde_json::to_vec_pretty(bundle).map_err(|e| e.to_string())?;
        let encrypted_b64 = self.encrypt_bytes_b64(&json_bytes)?;
        std::fs::write(path, encrypted_b64).map_err(|e| e.to_string())
    }

    pub fn import_airteach_file(&self, path: &Path) -> Result<AirTeachBundle, String> {
        let encrypted_b64 = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let plain = self.decrypt_bytes_b64(&encrypted_b64)?;
        serde_json::from_slice(&plain).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::face_auth::FaceAuthEngine;

    #[test]
    fn clean_slate_db_starts_with_zero_users_zero_sessions_and_no_default_pin() {
        let db = LocalStorage::open_in_memory();
        let users = db.list_users().expect("query users");
        assert!(users.is_empty(), "Fresh DB must have zero pre-seeded users");
        assert!(db.get_active_session_user().expect("query session").is_none());

        let cfg = db.load_or_init_config();
        assert_eq!(cfg.theme_mode, ThemeMode::Dark);
        assert!(!cfg.tutorial_dismissed);
        assert!(!cfg.tutorial_deleted);
    }

    #[test]
    fn user_crud_unique_username_argon2_and_session_lifecycle() {
        let db = LocalStorage::open_in_memory();
        let pw_hash = FaceAuthEngine::hash_password_argon2("TeacherPass99").unwrap();
        let mut emb = vec![0.0_f32; EMBEDDING_DIM];
        emb[0] = 0.7;
        emb[1] = 0.3;

        let created = db
            .create_user("gv_test", "Giáo viên Thử nghiệm", &pw_hash, &emb)
            .expect("create user");
        assert_eq!(created.username, "gv_test");
        assert!(FaceAuthEngine::verify_password_argon2("TeacherPass99", &created.password_hash));

        // Duplicate username (case-insensitive) must be rejected
        assert!(db.create_user("GV_TEST", "Duplicate", &pw_hash, &emb).is_err());

        // Session lifecycle: create -> active -> logout clears session
        let _sess = db.create_session(&created.id).expect("create session");
        let active = db.get_active_session_user().expect("read session");
        assert!(active.is_some());
        assert_eq!(active.unwrap().1.id, created.id);

        db.clear_active_sessions().expect("clear session on logout");
        assert!(db.get_active_session_user().expect("read after logout").is_none());
    }

    #[test]
    fn multi_template_face_and_ai_cache_persistence() {
        let db = LocalStorage::open_in_memory();
        let pw_hash = FaceAuthEngine::hash_password_argon2("SecureTeacher99").unwrap();
        let mut t1 = vec![0.0_f32; SFACE_EMBEDDING_DIM];
        t1[0] = 1.0;
        let mut t2 = vec![0.0_f32; SFACE_EMBEDDING_DIM];
        t2[0] = 0.8;
        t2[1] = 0.6;
        let user = db
            .create_user_with_templates("sface_user", "SFace Teacher", &pw_hash, &[t1, t2])
            .expect("create multi-template user");
        assert_eq!(user.face_templates.len(), 2);
        assert_eq!(user.face_embedding.len(), SFACE_EMBEDDING_DIM);

        let loaded = db.find_user_by_username("sface_user").unwrap().unwrap();
        assert_eq!(loaded.face_templates.len(), 2);

        db.put_cached_ai_response("k123", "gemini-3.5-flash", "Lời giải chi tiết", "{\"summary\":\"OK\"}");
        let cached = db.get_cached_ai_response("k123").expect("cache hit");
        assert_eq!(cached.0, "gemini-3.5-flash");
        assert_eq!(cached.1, "Lời giải chi tiết");
    }
}
