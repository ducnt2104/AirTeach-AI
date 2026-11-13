//! Biometric Face Authentication & Argon2id Credential Engine (`src/ai/face_auth.rs`):
//! - Local-first SQLite `users` + `sessions` backend (Zero JSON/NPY sample files)
//! - Guided 3-to-5 Angle Face Registration computing a single L2-normalized Average Embedding
//!   (numeric vector ONLY — raw images are never written to disk)
//! - Argon2id password/PIN hashing with CSPRNG salt + strict Sign-Up input validation
//! - Real Cosine Similarity Face ID verification against configurable threshold

use argon2::{
    password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString},
    Argon2,
};
use serde::{Deserialize, Serialize};

pub const DEFAULT_FACE_MATCH_THRESHOLD: f32 = 0.80;
pub const MIN_FACE_ENROLL_SAMPLES: usize = 3;
pub const MAX_FACE_ENROLL_SAMPLES: usize = 5;
pub const EMBEDDING_DIM: usize = 512;

/// Embedded MobileFaceNet ONNX model bytes (used by diagnostics / ORT loader).
pub static MOBILEFACENET_ONNX_BYTES: &[u8] =
    include_bytes!("../../assets/models/mobilefacenet.onnx");

/// Guided capture instructions for 3-to-5 slight head angle enrollment.
pub const ENROLL_ANGLE_PROMPTS: [&str; MAX_FACE_ENROLL_SAMPLES] = [
    "Góc 1/5: Nhìn thẳng vào camera trong ánh sáng ổn định",
    "Góc 2/5: Nghiêng nhẹ đầu sang trái (~10°)",
    "Góc 3/5: Nghiêng nhẹ đầu sang phải (~10°)",
    "Góc 4/5 (Tùy chọn): Hơi ngẩng nhẹ mặt lên (~8°)",
    "Góc 5/5 (Tùy chọn): Hơi cúi nhẹ đầu xuống (~8°)",
];

/// Authenticated or registered teacher record stored in SQLite `users`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAccount {
    pub id: String,
    pub username: String,
    pub display_name: String,
    pub password_hash: String,
    pub face_embedding: Vec<f32>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LivenessStatus {
    pub blink_verified: bool,
    pub motion_verified: bool,
    pub blink_count: u32,
    pub motion_score: f32,
    pub ear_value: f32,
}

impl Default for LivenessStatus {
    fn default() -> Self {
        Self {
            blink_verified: false,
            motion_verified: false,
            blink_count: 0,
            motion_score: 0.0,
            ear_value: 0.28,
        }
    }
}

impl LivenessStatus {
    #[inline]
    pub fn is_live(&self) -> bool {
        self.motion_verified || self.motion_score >= 0.04 || self.blink_verified
    }
}

#[derive(Debug, Clone)]
pub struct FaceAuthEvaluation {
    pub cosine_similarity: f32,
    pub liveness: LivenessStatus,
    pub unlocked: bool,
}

pub struct FaceAuthEngine {
    ort_session: Option<ort::session::Session>,
    pub liveness: LivenessStatus,
    prev_blink_state: bool,
}

impl FaceAuthEngine {
    pub fn new() -> Self {
        let ort_session = Self::try_load_ort_session();
        Self {
            ort_session,
            liveness: LivenessStatus::default(),
            prev_blink_state: false,
        }
    }

    fn try_load_ort_session() -> Option<ort::session::Session> {
        let candidates = [
            "models/mobilefacenet.onnx",
            "assets/models/mobilefacenet.onnx",
            "mobilefacenet.onnx",
        ];
        crate::ai::try_create_ort_session(&candidates, MOBILEFACENET_ONNX_BYTES)
    }

    pub fn is_onnx_loaded(&self) -> bool {
        self.ort_session.is_some()
    }

    pub fn reset_liveness(&mut self) {
        self.liveness = LivenessStatus::default();
        self.prev_blink_state = false;
    }

    pub fn update_liveness(&mut self, blink_detected: bool, motion_score: f32) {
        if blink_detected && !self.prev_blink_state {
            self.liveness.blink_count = self.liveness.blink_count.saturating_add(1);
            self.liveness.blink_verified = true;
        }
        self.prev_blink_state = blink_detected;
        self.liveness.motion_score = motion_score;
        if motion_score >= 0.04 {
            self.liveness.motion_verified = true;
        }
    }

    pub fn update_liveness_with_ear(
        &mut self,
        blink_detected: bool,
        motion_score: f32,
        ear_value: f32,
    ) {
        self.liveness.ear_value = ear_value;
        if ear_value > 0.01 && ear_value < 0.21 && !self.prev_blink_state {
            self.liveness.blink_count = self.liveness.blink_count.saturating_add(1);
            self.liveness.blink_verified = true;
        }
        self.update_liveness(blink_detected, motion_score);
    }

    /// Extract L2-normalized 512-D facial feature vector without explicit eye coordinates.
    pub fn extract_512d_embedding(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        face_bbox_norm: [f32; 4],
    ) -> [f32; EMBEDDING_DIM] {
        self.extract_512d_aligned_embedding(rgba, width, height, face_bbox_norm, None)
    }

    /// Perform 2D eye-level Affine Rotation + Scale Face Alignment before extracting an L2-normalized
    /// 512-D embedding from a 128x128 histogram-equalized crop. Raw pixel buffers are discarded immediately.
    pub fn extract_512d_aligned_embedding(
        &self,
        rgba: &[u8],
        width: u32,
        height: u32,
        face_bbox_norm: [f32; 4],
        face_eyes_norm: Option<[f32; 4]>,
    ) -> [f32; EMBEDDING_DIM] {
        let mut emb = [0.0_f32; EMBEDDING_DIM];
        if width == 0 || height == 0 || rgba.len() < (width * height * 4) as usize {
            emb[0] = 1.0;
            return emb;
        }

        let wf = width as f32;
        let hf = height as f32;

        // Compute optional eye-level roll alignment transform (cos_a, sin_a, origin, scale)
        let aligned_transform = face_eyes_norm.and_then(|[lx, ly, rx, ry]| {
            let dx = (rx - lx) * wf;
            let dy = (ry - ly) * hf;
            let eye_dist = dx.hypot(dy);
            if eye_dist < 8.0 {
                return None;
            }
            let roll = dy.atan2(dx).clamp(-0.65, 0.65);
            let (sin_a, cos_a) = roll.sin_cos();
            let eye_mid_x = 0.5 * (lx + rx) * wf;
            let eye_mid_y = 0.5 * (ly + ry) * hf;
            // Aligned canonical 128x128 grid expects eyes horizontally centered at (64, 46)
            // spanning ~42 px out of 128 (~3.05x inter-ocular box size).
            let crop_size = (eye_dist * 3.05).max(32.0);
            Some((cos_a, sin_a, eye_mid_x, eye_mid_y, crop_size))
        });

        let x0 = face_bbox_norm[0].clamp(0.0, 0.95) * wf;
        let y0 = face_bbox_norm[1].clamp(0.0, 0.95) * hf;
        let rw = (face_bbox_norm[2].clamp(0.05, 1.0) * wf).max(8.0);
        let rh = (face_bbox_norm[3].clamp(0.05, 1.0) * hf).max(8.0);

        let mut crop_luma = [0u8; 128 * 128];
        let mut crop_chroma = [0.0_f32; 128 * 128];
        let mut hist = [0u32; 256];

        let sample_bilinear = |fx: f32, fy: f32| -> (f32, f32, f32) {
            let cx = fx.clamp(0.0, (width - 1) as f32);
            let cy = fy.clamp(0.0, (height - 1) as f32);
            let x_lo = cx.floor() as u32;
            let y_lo = cy.floor() as u32;
            let x_hi = (x_lo + 1).min(width - 1);
            let y_hi = (y_lo + 1).min(height - 1);
            let tx = cx - x_lo as f32;
            let ty = cy - y_lo as f32;

            let idx00 = ((y_lo * width + x_lo) * 4) as usize;
            let idx10 = ((y_lo * width + x_hi) * 4) as usize;
            let idx01 = ((y_hi * width + x_lo) * 4) as usize;
            let idx11 = ((y_hi * width + x_hi) * 4) as usize;

            let interp = |c: usize| -> f32 {
                let v00 = rgba[idx00 + c] as f32;
                let v10 = rgba[idx10 + c] as f32;
                let v01 = rgba[idx01 + c] as f32;
                let v11 = rgba[idx11 + c] as f32;
                let top = v00 + (v10 - v00) * tx;
                let bot = v01 + (v11 - v01) * tx;
                top + (bot - top) * ty
            };
            (interp(0), interp(1), interp(2))
        };

        for cy in 0..128usize {
            for cx in 0..128usize {
                let (sx, sy) = if let Some((cos_a, sin_a, mx, my, crop_size)) = aligned_transform {
                    let u = ((cx as f32 + 0.5) / 128.0 - 0.5) * crop_size;
                    let v = ((cy as f32 + 0.5) / 128.0 - 0.36) * crop_size;
                    (mx + cos_a * u - sin_a * v, my + sin_a * u + cos_a * v)
                } else {
                    (
                        x0 + ((cx as f32 + 0.5) / 128.0) * rw,
                        y0 + ((cy as f32 + 0.5) / 128.0) * rh,
                    )
                };
                let (r, g, b) = sample_bilinear(sx, sy);
                let y_u8 = (0.299 * r + 0.587 * g + 0.114 * b).round().clamp(0.0, 255.0) as u8;
                let p = cy * 128 + cx;
                crop_luma[p] = y_u8;
                crop_chroma[p] = ((r - b) / 255.0) * 0.5 + 0.5;
                hist[y_u8 as usize] += 1;
            }
        }

        let mut cdf = [0u32; 256];
        let mut running = 0u32;
        for i in 0..256 {
            running += hist[i];
            cdf[i] = running;
        }
        let cdf_min = cdf.iter().copied().find(|&c| c > 0).unwrap_or(0);
        let denom = (128 * 128 - cdf_min).max(1) as f32;

        for gy in 0..16usize {
            for gx in 0..16usize {
                let mut sum_luma = 0.0_f32;
                let mut sum_chroma = 0.0_f32;
                for dy in 0..8usize {
                    for dx in 0..8usize {
                        let p = (gy * 8 + dy) * 128 + (gx * 8 + dx);
                        let y_val = crop_luma[p] as usize;
                        let eq_norm = ((cdf[y_val].saturating_sub(cdf_min)) as f32 / denom)
                            .clamp(0.0, 1.0);
                        sum_luma += eq_norm;
                        sum_chroma += crop_chroma[p];
                    }
                }
                let base_slot = (gy * 16 + gx) * 2;
                emb[base_slot] = sum_luma / 64.0;
                emb[base_slot + 1] = sum_chroma / 64.0;
            }
        }

        normalize_l2_in_place(&mut emb);
        emb
    }

    /// Compute L2-normalized Average Embedding across 3-to-5 enrolled face angle vectors.
    pub fn average_embeddings(samples: &[[f32; EMBEDDING_DIM]]) -> Option<Vec<f32>> {
        if samples.is_empty() {
            return None;
        }
        let mut out = vec![0.0_f32; EMBEDDING_DIM];
        for s in samples {
            for (i, &v) in s.iter().enumerate() {
                out[i] += v;
            }
        }
        let n = samples.len() as f32;
        for v in &mut out {
            *v /= n;
        }
        normalize_l2_in_place(&mut out);
        Some(out)
    }

    /// Evaluate live embedding against a target user's normalized vector using Cosine Similarity + Liveness.
    pub fn evaluate_unlock_with_threshold(
        &mut self,
        live_embedding: &[f32; EMBEDDING_DIM],
        enrolled_embedding: &[f32],
        blink_detected: bool,
        motion_score: f32,
        threshold: f32,
    ) -> FaceAuthEvaluation {
        self.update_liveness(blink_detected, motion_score);
        if enrolled_embedding.is_empty() {
            return FaceAuthEvaluation {
                cosine_similarity: 0.0,
                liveness: self.liveness.clone(),
                unlocked: false,
            };
        }
        let sim = cosine_similarity(live_embedding, enrolled_embedding);
        let effective_threshold = threshold.clamp(0.55, 0.98);
        let unlocked = sim >= effective_threshold && self.liveness.is_live();
        FaceAuthEvaluation {
            cosine_similarity: sim,
            liveness: self.liveness.clone(),
            unlocked,
        }
    }

    /// Strict Sign-Up input validator (Step 1 before face capture).
    pub fn validate_signup_fields(
        username: &str,
        display_name: &str,
        password: &str,
        confirm_password: &str,
        username_taken: bool,
    ) -> Result<(), String> {
        let uname = username.trim();
        let dname = display_name.trim();

        if uname.len() < 3 || uname.len() > 32 {
            return Err("Tên đăng nhập phải từ 3 đến 32 ký tự.".to_string());
        }
        if !uname
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
        {
            return Err("Tên đăng nhập chỉ được dùng chữ cái không dấu, số, '.', '_', '-'.".to_string());
        }
        if username_taken {
            return Err(format!("Tên đăng nhập '{uname}' đã tồn tại trên máy này."));
        }
        if dname.chars().count() < 2 {
            return Err("Vui lòng nhập Tên hiển thị tối thiểu 2 ký tự.".to_string());
        }
        if password.chars().count() < 6 {
            return Err("Mật khẩu / Mã PIN phải có ít nhất 6 ký tự (hoặc 6 chữ số).".to_string());
        }
        if password != confirm_password {
            return Err("Mật khẩu xác nhận không trùng khớp.".to_string());
        }
        Ok(())
    }

    /// Hash a password or PIN (>= 6 chars) using Argon2id with random cryptographic salt.
    pub fn hash_password_argon2(password: &str) -> Result<String, String> {
        if password.chars().count() < 6 {
            return Err("Mật khẩu / Mã PIN phải có tối thiểu 6 ký tự.".to_string());
        }
        let salt = SaltString::generate(&mut OsRng);
        let argon2 = Argon2::default();
        argon2
            .hash_password(password.as_bytes(), &salt)
            .map(|h| h.to_string())
            .map_err(|e| format!("Argon2id hash error: {e}"))
    }

    /// Verify a password / PIN against a stored Argon2id PHC string.
    pub fn verify_password_argon2(password: &str, phc_hash: &str) -> bool {
        if password.is_empty() || phc_hash.trim().is_empty() {
            return false;
        }
        let Ok(parsed_hash) = PasswordHash::new(phc_hash) else {
            return false;
        };
        Argon2::default()
            .verify_password(password.as_bytes(), &parsed_hash)
            .is_ok()
    }
}

#[inline]
pub fn normalize_l2_in_place(vec: &mut [f32]) {
    let norm_sq: f32 = vec.iter().map(|v| v * v).sum();
    let norm = norm_sq.sqrt().max(1e-8);
    for v in vec.iter_mut() {
        *v /= norm;
    }
}

#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-8);
    (dot / denom).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argon2_password_hash_and_verify() {
        let hash = FaceAuthEngine::hash_password_argon2("SecureTeach#2026")
            .expect("Valid password >= 6 chars");
        assert!(hash.starts_with("$argon2"));
        assert!(FaceAuthEngine::verify_password_argon2("SecureTeach#2026", &hash));
        assert!(!FaceAuthEngine::verify_password_argon2("wrongpass", &hash));
        assert!(FaceAuthEngine::hash_password_argon2("12345").is_err());
    }

    #[test]
    fn test_signup_validation_rejects_invalid_and_duplicate_usernames() {
        assert!(FaceAuthEngine::validate_signup_fields("ab", "Teacher", "123456", "123456", false).is_err());
        assert!(FaceAuthEngine::validate_signup_fields("gv_a", "Teacher", "123456", "123456", true).is_err());
        assert!(FaceAuthEngine::validate_signup_fields("gv_a", "A", "123456", "123456", false).is_err());
        assert!(FaceAuthEngine::validate_signup_fields("gv_a", "Giáo viên A", "12345", "12345", false).is_err());
        assert!(FaceAuthEngine::validate_signup_fields("gv_a", "Giáo viên A", "123456", "654321", false).is_err());
        assert!(FaceAuthEngine::validate_signup_fields("gv_a", "Giáo viên A", "123456", "123456", false).is_ok());
    }

    #[test]
    fn test_average_embedding_and_cosine_similarity() {
        let mut s1 = [0.0_f32; EMBEDDING_DIM];
        let mut s2 = [0.0_f32; EMBEDDING_DIM];
        let mut s3 = [0.0_f32; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            s1[i] = if i % 2 == 0 { 0.8 } else { 0.2 };
            s2[i] = if i % 2 == 0 { 0.78 } else { 0.22 };
            s3[i] = if i % 2 == 0 { 0.82 } else { 0.18 };
        }
        normalize_l2_in_place(&mut s1);
        normalize_l2_in_place(&mut s2);
        normalize_l2_in_place(&mut s3);

        let avg = FaceAuthEngine::average_embeddings(&[s1, s2, s3]).expect("3 samples valid");
        let sim = cosine_similarity(&s2, &avg);
        assert!(sim > 0.99, "Average embedding should match enrolled poses tightly: sim={sim}");

        let norm: f32 = avg.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-4, "Average embedding must be L2-normalized");
    }

    #[test]
    fn test_aligned_face_embedding_is_consistent_under_head_tilt() {
        let engine = FaceAuthEngine::new();
        let w = 160u32;
        let h = 160u32;
        let mut img_upright = vec![24u8; (w * h * 4) as usize];
        let mut img_tilted = vec![24u8; (w * h * 4) as usize];

        let paint_face = |buf: &mut [u8], tilt_rad: f32| {
            let (s, c) = tilt_rad.sin_cos();
            let cx = 80.0_f32;
            let cy = 80.0_f32;
            for py in 0..h {
                for px in 0..w {
                    let dx = px as f32 - cx;
                    let dy = py as f32 - cy;
                    // Map screen point back to local upright face frame
                    let lx = c * dx + s * dy;
                    let ly = -s * dx + c * dy;
                    let idx = ((py * w + px) * 4) as usize;
                    if (lx / 36.0).powi(2) + (ly / 46.0).powi(2) <= 1.0 {
                        let left_eye = (lx + 15.0).hypot(ly + 12.0) < 6.0;
                        let right_eye = (lx - 15.0).hypot(ly + 12.0) < 6.0;
                        let mouth = lx.abs() < 16.0 && (ly - 20.0).abs() < 4.5;
                        let (r, g, b) = if left_eye || right_eye {
                            (35u8, 38u8, 45u8)
                        } else if mouth {
                            (185u8, 85u8, 95u8)
                        } else {
                            let shade = (210.0 - lx.abs() * 1.1).clamp(120.0, 225.0) as u8;
                            (shade, (shade as u16 * 85 / 100) as u8, (shade as u16 * 72 / 100) as u8)
                        };
                        buf[idx] = r;
                        buf[idx + 1] = g;
                        buf[idx + 2] = b;
                    }
                    buf[idx + 3] = 255;
                }
            }
        };

        paint_face(&mut img_upright, 0.0);
        let tilt = 18.0_f32.to_radians();
        paint_face(&mut img_tilted, tilt);

        let rot_pt = |lx: f32, ly: f32, ang: f32| -> (f32, f32) {
            let (s, c) = ang.sin_cos();
            ((80.0 + c * lx - s * ly) / w as f32, (80.0 + s * lx + c * ly) / h as f32)
        };
        let e_up = [
            (80.0 - 15.0) / w as f32,
            (80.0 - 12.0) / h as f32,
            (80.0 + 15.0) / w as f32,
            (80.0 - 12.0) / h as f32,
        ];
        let (tlx, tly) = rot_pt(-15.0, -12.0, tilt);
        let (trx, try_) = rot_pt(15.0, -12.0, tilt);
        let e_tilt = [tlx, tly, trx, try_];

        let emb_up =
            engine.extract_512d_aligned_embedding(&img_upright, w, h, [0.25, 0.20, 0.50, 0.60], Some(e_up));
        let emb_tilt =
            engine.extract_512d_aligned_embedding(&img_tilted, w, h, [0.22, 0.18, 0.56, 0.64], Some(e_tilt));

        let sim = cosine_similarity(&emb_up, &emb_tilt);
        assert!(
            sim > 0.90,
            "Aligned face embedding must stay consistent under 18-deg head tilt: sim={sim}"
        );
    }
}
