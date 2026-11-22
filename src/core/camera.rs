//! Real Hardware Camera Capture (`src/core/camera.rs`):
//! - Uses `nokhwa` with `ApiBackend::MediaFoundation` on Windows
//! - Auto-detects connected webcams via `nokhwa::query(ApiBackend::MediaFoundation)`
//! - Selects true 30 FPS streaming formats (`640x480` / `1280x720` YUYV/NV12/MJPEG) and verifies `cam.frame()`
//! - Computes real FPS (frame delta time), real Laplacian Variance sharpness (`Var(Laplacian(gray))`),
//!   and real Eye Aspect Ratio (EAR) blink & head-motion liveness from live camera pixels.
//! - Zero mock/dummy avatar drawing: if no camera is connected, reports `using_hardware_camera = false`
//!   so the UI displays the hardware connection error and retry button.

use crate::core::hand_tracker::{TwoHandFrame, TwoHandTracker};
use crossbeam_channel::{bounded, Receiver, Sender};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};
use nokhwa::Camera;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaroGridSpacing {
    Off,
    Grid10px,
    Grid20px,
    Grid50px,
}

impl CaroGridSpacing {
    pub const fn pixels(self) -> Option<u32> {
        match self {
            Self::Off => None,
            Self::Grid10px => Some(10),
            Self::Grid20px => Some(20),
            Self::Grid50px => Some(50),
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Grid10px => "10x10 px",
            Self::Grid20px => "20x20 px",
            Self::Grid50px => "50x50 px",
        }
    }
}

/// Runtime control commands sent from the GUI thread to the Camera Worker thread.
#[derive(Debug, Clone)]
pub enum CameraControlCmd {
    AcquireCamera,
    ReleaseCamera,
    SelectCameraIndex(u32),
    ReconnectHardware,
    SetSharpnessConfig {
        enable_clahe: bool,
        clahe_clip_limit: f32,
        unsharp_amount: f32,
    },
    SetHudGrid {
        enabled: bool,
        spacing_px: u32,
    },
    SetCalibrationPreview(bool),
    SetQualityGateThreshold(f32),
    SetHandConfidenceThreshold(f32),
    Shutdown,
}

/// Minimum Laplacian Variance (`Var(∇²Y)`) required to pass the Face ID Quality Gate.
pub const FACE_QUALITY_GATE_MIN_SHARPNESS: f32 = 38.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LightingAdaptationMode {
    Normal,
    LowLightBoost,
    GlareSuppression,
}

impl Default for LightingAdaptationMode {
    fn default() -> Self {
        Self::Normal
    }
}

impl LightingAdaptationMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Normal => "Bình thường",
            Self::LowLightBoost => "Tăng sáng",
            Self::GlareSuppression => "Chống chói",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FrameDiagnostics {
    pub lighting_mode: LightingAdaptationMode,
    pub mean_luma: f32,
    pub p5_luma: f32,
    pub p95_luma: f32,
    pub underexposed_ratio: f32,
    pub overexposed_ratio: f32,
    pub sharpness_score: f32,
    pub quality_gate_passed: bool,
}

/// Telemetry packet delivered from the Camera Worker thread to the GUI thread via `bounded(2)`.
#[derive(Clone)]
pub struct CameraTelemetryPacket {
    pub width: u32,
    pub height: u32,
    pub rgba_bytes: Arc<Vec<u8>>,
    pub raw_rgba_bytes: Option<Arc<Vec<u8>>>,
    pub two_hand_frame: TwoHandFrame,
    pub camera_fps: f32,
    pub sharpness_score: f32,
    pub active_sharpness_threshold: f32,
    pub lighting_mode: LightingAdaptationMode,
    pub mean_luma: f32,
    pub overexposed_ratio: f32,
    pub quality_gate_passed: bool,
    pub camera_active: bool,
    pub using_hardware_camera: bool,
    pub camera_error_message: Option<String>,
    pub camera_device_name: String,
    pub capture_resolution_label: String,
    pub available_cameras: Vec<(u32, String)>,
    pub face_bbox_norm: Option<[f32; 4]>,
    pub face_eyes_norm: Option<[f32; 4]>,
    pub face_blink_detected: bool,
    pub face_ear_value: f32,
    pub face_motion_score: f32,
    pub cv_frames_skipped: u64,
}

/// Apply adaptive luminance diagnostics + hue-preserving local CLAHE + adaptive gamma +
/// highlight/glare suppression + low-light bilateral denoise + thresholded Unsharp Masking
/// in-place, returning real `FrameDiagnostics` (including Laplacian Variance Quality Gate).
pub fn enhance_frame_adaptive_with_diagnostics(
    rgba: &mut [u8],
    luma_buf: &mut Vec<u8>,
    blur_buf: &mut Vec<u8>,
    width: u32,
    height: u32,
    enable_clahe: bool,
    clip_limit_factor: f32,
    unsharp_amount: f32,
) -> FrameDiagnostics {
    let num_pixels = (width * height) as usize;
    if num_pixels == 0 || rgba.len() < num_pixels * 4 {
        return FrameDiagnostics::default();
    }

    if luma_buf.len() != num_pixels {
        luma_buf.resize(num_pixels, 0);
    }
    if blur_buf.len() != num_pixels {
        blur_buf.resize(num_pixels, 0);
    }

    // Step 1: Real-time Histogram & Luminance Diagnostics (mean_luma, p5, p95)
    let mut sum_y = 0u64;
    let mut under_count = 0u64;
    let mut over_count = 0u64;
    let mut raw_hist = [0u32; 256];
    for i in 0..num_pixels {
        let idx = i * 4;
        let r = rgba[idx] as u32;
        let g = rgba[idx + 1] as u32;
        let b = rgba[idx + 2] as u32;
        let y = ((77 * r + 150 * g + 29 * b) >> 8) as u8;
        luma_buf[i] = y;
        raw_hist[y as usize] += 1;
        sum_y += y as u64;
        if y < 50 {
            under_count += 1;
        } else if y > 228 {
            over_count += 1;
        }
    }

    let p5_target = ((num_pixels as u64 * 5) / 100).max(1) as u32;
    let p95_target = ((num_pixels as u64 * 95) / 100).max(1) as u32;
    let mut cum = 0u32;
    let mut p5_luma = 0.0_f32;
    let mut p95_luma = 255.0_f32;
    let mut found_p5 = false;
    for (bin, &cnt) in raw_hist.iter().enumerate() {
        cum += cnt;
        if !found_p5 && cum >= p5_target {
            p5_luma = bin as f32;
            found_p5 = true;
        }
        if cum >= p95_target {
            p95_luma = bin as f32;
            break;
        }
    }

    let mean_luma = sum_y as f32 / num_pixels as f32;
    let underexposed_ratio = under_count as f32 / num_pixels as f32;
    let overexposed_ratio = over_count as f32 / num_pixels as f32;

    let lighting_mode = if mean_luma < 70.0 || underexposed_ratio > 0.55 {
        LightingAdaptationMode::LowLightBoost
    } else if p95_luma > 235.0 || overexposed_ratio > 0.18 || mean_luma > 185.0 {
        LightingAdaptationMode::GlareSuppression
    } else {
        LightingAdaptationMode::Normal
    };

    // Step 2: Adaptive Gamma + Highlight Glare Roll-Off + Hue-Preserving CLAHE on Luminance
    if enable_clahe {
        let mut tone_lut = [0u8; 256];
        match lighting_mode {
            LightingAdaptationMode::LowLightBoost => {
                let gamma = if mean_luma < 70.0 {
                    (0.52 + 0.36 * (mean_luma / 70.0).clamp(0.0, 1.0)).clamp(0.52, 0.88)
                } else {
                    0.78_f32
                };
                for (i, slot) in tone_lut.iter_mut().enumerate() {
                    let v = (i as f32 / 255.0).powf(gamma) * 255.0;
                    *slot = v.round().clamp(0.0, 255.0) as u8;
                }
            }
            LightingAdaptationMode::GlareSuppression => {
                let gamma = if mean_luma > 198.0 || p95_luma > 235.0 {
                    1.28_f32
                } else {
                    1.15_f32
                };
                for (i, slot) in tone_lut.iter_mut().enumerate() {
                    let knee = if i > 195 {
                        195.0 + (i - 195) as f32 * 0.42
                    } else {
                        i as f32
                    };
                    let v = (knee / 255.0).powf(gamma) * 255.0;
                    *slot = v.round().clamp(0.0, 255.0) as u8;
                }
            }
            LightingAdaptationMode::Normal => {
                for (i, slot) in tone_lut.iter_mut().enumerate() {
                    *slot = i as u8;
                }
            }
        }

        let mut hist = [0u32; 256];
        let mut min_y = 255u8;
        let mut max_y = 0u8;
        for y in luma_buf.iter_mut() {
            let mapped = tone_lut[*y as usize];
            *y = mapped;
            hist[mapped as usize] += 1;
            if mapped < min_y {
                min_y = mapped;
            }
            if mapped > max_y {
                max_y = mapped;
            }
        }

        let eff_clip = match lighting_mode {
            LightingAdaptationMode::LowLightBoost => (clip_limit_factor * 1.25).clamp(1.2, 6.0),
            LightingAdaptationMode::GlareSuppression => (clip_limit_factor * 0.72).clamp(1.2, 4.0),
            LightingAdaptationMode::Normal => clip_limit_factor.clamp(1.2, 6.0),
        };

        let mut clahe_lut = [0u8; 256];
        if max_y > min_y {
            let avg_bin = (num_pixels as f32 / 256.0).max(1.0);
            let clip_limit = (avg_bin * eff_clip) as u32;
            let mut excess = 0u32;
            for count in hist.iter_mut() {
                if *count > clip_limit {
                    excess += *count - clip_limit;
                    *count = clip_limit;
                }
            }
            let redistrib = excess / 256;
            let mut cdf = 0u32;
            let total = num_pixels.max(1) as u32;
            for (i, count) in hist.iter().enumerate() {
                cdf += *count + redistrib;
                let eq_val = ((cdf as u64 * 255) / total as u64).min(255) as u8;
                clahe_lut[i] = ((i as u16 * 168 + eq_val as u16 * 88) >> 8) as u8;
            }
        } else {
            for (i, slot) in clahe_lut.iter_mut().enumerate() {
                *slot = i as u8;
            }
        }

        // Apply final luma transformation back to RGB preserving chroma proportions
        for i in 0..num_pixels {
            let idx = i * 4;
            let r_raw = rgba[idx] as u32;
            let g_raw = rgba[idx + 1] as u32;
            let b_raw = rgba[idx + 2] as u32;
            let y_old = ((77 * r_raw + 150 * g_raw + 29 * b_raw) >> 8).max(1) as i32;
            let y_new = clahe_lut[luma_buf[i] as usize];
            luma_buf[i] = y_new;

            // Blend ratio scaling + additive offset so saturated blacks lift smoothly without color cast
            let delta = y_new as i32 - y_old;
            if y_old >= 24 {
                let scale_fp = ((y_new as i32) << 8) / y_old;
                let scale_clamped = scale_fp.clamp(115, 580);
                let rs = (r_raw as i32 * scale_clamped) >> 8;
                let gs = (g_raw as i32 * scale_clamped) >> 8;
                let bs = (b_raw as i32 * scale_clamped) >> 8;
                rgba[idx] = ((rs * 3 + (r_raw as i32 + delta)) >> 2).clamp(0, 255) as u8;
                rgba[idx + 1] = ((gs * 3 + (g_raw as i32 + delta)) >> 2).clamp(0, 255) as u8;
                rgba[idx + 2] = ((bs * 3 + (b_raw as i32 + delta)) >> 2).clamp(0, 255) as u8;
            } else {
                rgba[idx] = (r_raw as i32 + delta).clamp(0, 255) as u8;
                rgba[idx + 1] = (g_raw as i32 + delta).clamp(0, 255) as u8;
                rgba[idx + 2] = (b_raw as i32 + delta).clamp(0, 255) as u8;
            }
        }
    }

    let w = width as usize;
    let h = height as usize;

    // Step 3: Edge-Preserving Bilateral Denoise in Low-Light before Unsharp Masking
    if enable_clahe && lighting_mode == LightingAdaptationMode::LowLightBoost && w >= 3 && h >= 3 {
        blur_buf.copy_from_slice(luma_buf);
        const EDGE_THRESH: i32 = 22;
        for y in 1..(h - 1) {
            let row_prev = (y - 1) * w;
            let row_curr = y * w;
            let row_next = (y + 1) * w;
            for x in 1..(w - 1) {
                let c = blur_buf[row_curr + x] as i32;
                let nbs = [
                    blur_buf[row_prev + x] as i32,
                    blur_buf[row_next + x] as i32,
                    blur_buf[row_curr + x - 1] as i32,
                    blur_buf[row_curr + x + 1] as i32,
                ];
                let mut acc = c * 4;
                let mut wt = 4;
                for &nb in &nbs {
                    if (nb - c).abs() <= EDGE_THRESH {
                        acc += nb * 2;
                        wt += 2;
                    }
                }
                let cleaned = (acc / wt).clamp(0, 255) as i32;
                let diff = cleaned - c;
                luma_buf[row_curr + x] = cleaned as u8;
                if diff != 0 {
                    let p = (row_curr + x) * 4;
                    rgba[p] = (rgba[p] as i32 + diff).clamp(0, 255) as u8;
                    rgba[p + 1] = (rgba[p + 1] as i32 + diff).clamp(0, 255) as u8;
                    rgba[p + 2] = (rgba[p + 2] as i32 + diff).clamp(0, 255) as u8;
                }
            }
        }
    }

    // Step 4: 3x3 Box Blur & True 2D Laplacian Variance (`Var(∇²Y)`) Quality Gate
    let mut lap_sum = 0.0_f64;
    let mut lap_sq_sum = 0.0_f64;
    let mut inner_count = 0usize;

    if w >= 3 && h >= 3 {
        blur_buf.copy_from_slice(luma_buf);
        for y in 1..(h - 1) {
            let row_prev = (y - 1) * w;
            let row_curr = y * w;
            let row_next = (y + 1) * w;
            for x in 1..(w - 1) {
                let c = luma_buf[row_curr + x] as i32;
                let top = luma_buf[row_prev + x] as i32;
                let bot = luma_buf[row_next + x] as i32;
                let left = luma_buf[row_curr + x - 1] as i32;
                let right = luma_buf[row_curr + x + 1] as i32;

                let lap = (top + bot + left + right - 4 * c) as f64;
                lap_sum += lap;
                lap_sq_sum += lap * lap;
                inner_count += 1;

                let sum = luma_buf[row_prev + x - 1] as u32
                    + top as u32
                    + luma_buf[row_prev + x + 1] as u32
                    + left as u32
                    + c as u32
                    + right as u32
                    + luma_buf[row_next + x - 1] as u32
                    + bot as u32
                    + luma_buf[row_next + x + 1] as u32;
                blur_buf[row_curr + x] = (sum / 9) as u8;
            }
        }
    }

    // Step 5: Thresholded Unsharp Masking (Coring prevents flat noise amplification)
    let coring_thresh = match lighting_mode {
        LightingAdaptationMode::LowLightBoost => 4,
        _ => 2,
    };
    let amount_fixed = (unsharp_amount.clamp(0.0, 2.5) * 128.0) as i32;
    if amount_fixed > 0 {
        for i in 0..num_pixels {
            let diff = luma_buf[i] as i32 - blur_buf[i] as i32;
            if diff.abs() < coring_thresh {
                continue;
            }
            let detail = (diff * amount_fixed) >> 7;
            let idx = i * 4;
            rgba[idx] = (rgba[idx] as i32 + detail).clamp(0, 255) as u8;
            rgba[idx + 1] = (rgba[idx + 1] as i32 + detail).clamp(0, 255) as u8;
            rgba[idx + 2] = (rgba[idx + 2] as i32 + detail).clamp(0, 255) as u8;
        }
    }

    let sharpness_score = if inner_count == 0 {
        0.0
    } else {
        let n = inner_count as f64;
        let mean = lap_sum / n;
        let variance = (lap_sq_sum / n) - (mean * mean);
        variance.max(0.0) as f32
    };

    FrameDiagnostics {
        lighting_mode,
        mean_luma,
        p5_luma,
        p95_luma,
        underexposed_ratio,
        overexposed_ratio,
        sharpness_score,
        quality_gate_passed: sharpness_score >= FACE_QUALITY_GATE_MIN_SHARPNESS,
    }
}

/// Backward-compatible wrapper returning Laplacian Variance sharpness score.
pub fn enhance_frame_clahe_unsharp_in_place(
    rgba: &mut [u8],
    luma_buf: &mut Vec<u8>,
    blur_buf: &mut Vec<u8>,
    width: u32,
    height: u32,
    enable_clahe: bool,
    clip_limit_factor: f32,
    unsharp_amount: f32,
) -> f32 {
    enhance_frame_adaptive_with_diagnostics(
        rgba,
        luma_buf,
        blur_buf,
        width,
        height,
        enable_clahe,
        clip_limit_factor,
        unsharp_amount,
    )
    .sharpness_score
}

/// Draw customizable Caro Grid overlay onto an RGBA frame buffer in-place.
pub fn draw_caro_grid_rgba_in_place(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    enabled: bool,
    spacing_px: u32,
) {
    if !enabled {
        return;
    }
    let step = spacing_px.clamp(10, 60);
    if width == 0 || height == 0 || rgba.len() < (width * height * 4) as usize {
        return;
    }

    for x in (0..width).step_by(step as usize) {
        let is_major = x % (step * 5) == 0;
        let alpha = if is_major { 85u16 } else { 38u16 };
        for y in 0..height {
            let idx = ((y * width + x) * 4) as usize;
            rgba[idx] = ((rgba[idx] as u16 * (255 - alpha) + 6 * alpha) >> 8) as u8;
            rgba[idx + 1] = ((rgba[idx + 1] as u16 * (255 - alpha) + 182 * alpha) >> 8) as u8;
            rgba[idx + 2] = ((rgba[idx + 2] as u16 * (255 - alpha) + 212 * alpha) >> 8) as u8;
        }
    }

    for y in (0..height).step_by(step as usize) {
        let is_major = y % (step * 5) == 0;
        let alpha = if is_major { 85u16 } else { 38u16 };
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            rgba[idx] = ((rgba[idx] as u16 * (255 - alpha) + 6 * alpha) >> 8) as u8;
            rgba[idx + 1] = ((rgba[idx + 1] as u16 * (255 - alpha) + 182 * alpha) >> 8) as u8;
            rgba[idx + 2] = ((rgba[idx + 2] as u16 * (255 - alpha) + 212 * alpha) >> 8) as u8;
        }
    }
}

/// Non-blocking Frame-Skipping Gate (`AtomicBool` lock) decoupling high-FPS camera preview from async CV inference.
pub struct CvDispatchGate {
    is_processing: Arc<AtomicBool>,
    frames_skipped: Arc<AtomicU64>,
}

impl CvDispatchGate {
    pub fn new() -> Self {
        Self {
            is_processing: Arc::new(AtomicBool::new(false)),
            frames_skipped: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Try to acquire exclusive permission to dispatch a CV frame.
    /// Returns `false` immediately (incrementing `frames_skipped`) if CV inference is still running.
    #[inline]
    pub fn try_begin(&self) -> bool {
        if self
            .is_processing
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
            .is_ok()
        {
            true
        } else {
            self.frames_skipped.fetch_add(1, Ordering::Relaxed);
            false
        }
    }

    #[inline]
    pub fn finish(&self) {
        self.is_processing.store(false, Ordering::Release);
    }

    #[inline]
    pub fn is_busy(&self) -> bool {
        self.is_processing.load(Ordering::Acquire)
    }

    #[inline]
    pub fn skipped_count(&self) -> u64 {
        self.frames_skipped.load(Ordering::Relaxed)
    }
}

struct CvAsyncJob {
    rgba_bytes: Vec<u8>,
    width: u32,
    height: u32,
    face_bbox_norm: Option<[f32; 4]>,
    min_hand_confidence: f32,
}

pub struct CameraSubsystem {
    pub frame_rx: Receiver<CameraTelemetryPacket>,
    pub cmd_tx: Sender<CameraControlCmd>,
    running: Arc<AtomicBool>,
}

impl CameraSubsystem {
    pub fn spawn(
        initial_camera_idx: u32,
        initial_grid_enabled: bool,
        initial_grid_px: u32,
        start_active: bool,
    ) -> Self {
        let (frame_tx, frame_rx) = bounded::<CameraTelemetryPacket>(2);
        let (cmd_tx, cmd_rx) = bounded::<CameraControlCmd>(16);
        let running = Arc::new(AtomicBool::new(true));
        let running_flag = running.clone();

        thread::Builder::new()
            .name("airteach-msmf-camera-worker".to_string())
            .spawn(move || {
                Self::worker_loop(
                    initial_camera_idx,
                    initial_grid_enabled,
                    initial_grid_px,
                    start_active,
                    frame_tx,
                    cmd_rx,
                    running_flag,
                );
            })
            .expect("Failed to spawn MediaFoundation camera worker thread");

        Self {
            frame_rx,
            cmd_tx,
            running,
        }
    }

    pub fn shutdown(&self) {
        self.running.store(false, Ordering::SeqCst);
        let _ = self.cmd_tx.try_send(CameraControlCmd::Shutdown);
    }

    pub fn acquire(&self) {
        let _ = self.cmd_tx.try_send(CameraControlCmd::AcquireCamera);
    }

    pub fn release(&self) {
        let _ = self.cmd_tx.try_send(CameraControlCmd::ReleaseCamera);
    }

    /// Query available Windows MediaFoundation webcams.
    pub fn enumerate_media_foundation_cameras() -> Vec<(u32, String)> {
        let _ = nokhwa::nokhwa_initialize(|_| {});
        match nokhwa::query(ApiBackend::MediaFoundation) {
            Ok(infos) => infos
                .into_iter()
                .enumerate()
                .map(|(idx, info)| {
                    let i = match info.index() {
                        CameraIndex::Index(n) => *n,
                        _ => idx as u32,
                    };
                    (i, info.human_name())
                })
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Open a real MediaFoundation camera with explicit error message if unavailable/denied.
    pub fn try_open_media_foundation_camera_with_error(
        preferred_idx: u32,
    ) -> Result<(Camera, String, u32, u32), String> {
        let discovered = Self::enumerate_media_foundation_cameras();
        if discovered.is_empty() {
            return Err("Không tìm thấy thiết bị Webcam nào trên hệ thống.".to_string());
        }

        let target_idx = if discovered.iter().any(|(i, _)| *i == preferred_idx) {
            preferred_idx
        } else {
            discovered[0].0
        };

        let index = CameraIndex::Index(target_idx);
        let candidates = [
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(CameraFormat::new(
                Resolution::new(640, 480),
                FrameFormat::YUYV,
                30,
            ))),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(CameraFormat::new(
                Resolution::new(640, 480),
                FrameFormat::NV12,
                30,
            ))),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(CameraFormat::new(
                Resolution::new(640, 480),
                FrameFormat::MJPEG,
                30,
            ))),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
            RequestedFormat::new::<RgbFormat>(RequestedFormatType::None),
        ];

        let mut last_err = String::new();
        for req in candidates {
            match Camera::with_backend(index.clone(), req, ApiBackend::MediaFoundation)
                .or_else(|_| Camera::new(index.clone(), req))
            {
                Ok(mut cam) => {
                    if let Ok(fmts) = cam.compatible_camera_formats() {
                        if let Some(best_fmt) = fmts
                            .into_iter()
                            .filter(|f| {
                                (f.resolution().width() == 640 && f.resolution().height() == 480)
                                    || (f.resolution().width() == 1280
                                        && f.resolution().height() == 720)
                            })
                            .max_by_key(|f| {
                                let res_pref =
                                    if f.resolution().width() == 640 { 100 } else { 50 };
                                let fmt_pref = match f.format() {
                                    FrameFormat::YUYV => 30,
                                    FrameFormat::NV12 => 20,
                                    FrameFormat::MJPEG => 10,
                                    _ => 0,
                                };
                                (f.frame_rate() as u32) * 1000 + res_pref + fmt_pref
                            })
                        {
                            #[allow(deprecated)]
                            let _ = cam.set_camera_format(best_fmt);
                        }
                    }

                    match cam.open_stream() {
                        Ok(_) => match cam.frame() {
                            Ok(_) => {
                                let res = cam.resolution();
                                let name = cam.info().human_name();
                                return Ok((
                                    cam,
                                    name,
                                    res.width().max(640),
                                    res.height().max(480),
                                ));
                            }
                            Err(e) => {
                                last_err = format!("Camera không trả về khung hình: {e}");
                                let _ = cam.stop_stream();
                                drop(cam);
                            }
                        },
                        Err(e) => {
                            last_err = format!(
                                "Không thể mở luồng Camera (có thể bị từ chối quyền hoặc ứng dụng khác đang dùng): {e}"
                            );
                            drop(cam);
                        }
                    }
                }
                Err(e) => {
                    last_err = format!(
                        "Không truy cập được Camera #{target_idx} (kiểm tra quyền Camera Windows): {e}"
                    );
                }
            }
        }

        if last_err.is_empty() {
            last_err = format!(
                "Không thể khởi tạo Camera #{target_idx}. Vui lòng kiểm tra quyền truy cập hoặc kết nối USB."
            );
        }
        Err(last_err)
    }

    pub fn try_open_media_foundation_camera(
        preferred_idx: u32,
    ) -> Option<(Camera, String, u32, u32)> {
        Self::try_open_media_foundation_camera_with_error(preferred_idx).ok()
    }

    fn release_hw_camera(hw_cam: &mut Option<(Camera, String, u32, u32)>) {
        if let Some((mut old_cam, _, _, _)) = hw_cam.take() {
            let _ = old_cam.stop_stream();
            drop(old_cam);
        }
    }

    fn worker_loop(
        mut camera_idx: u32,
        mut hud_grid_enabled: bool,
        mut hud_grid_px: u32,
        mut camera_active: bool,
        frame_tx: Sender<CameraTelemetryPacket>,
        cmd_rx: Receiver<CameraControlCmd>,
        running: Arc<AtomicBool>,
    ) {
        let mut available_cameras = Self::enumerate_media_foundation_cameras();
        let mut last_camera_error: Option<String> = None;
        let mut hw_cam = if camera_active {
            match Self::try_open_media_foundation_camera_with_error(camera_idx) {
                Ok(cam) => Some(cam),
                Err(e) => {
                    last_camera_error = Some(e);
                    None
                }
            }
        } else {
            None
        };

        // Decoupled async CV inference worker with AtomicBool frame-skipping gate
        let cv_gate = Arc::new(CvDispatchGate::new());
        let latest_cv_result = Arc::new(Mutex::new((TwoHandFrame::default(), Instant::now())));
        let (cv_req_tx, cv_req_rx) = bounded::<CvAsyncJob>(1);

        let cv_gate_worker = Arc::clone(&cv_gate);
        let latest_cv_worker = Arc::clone(&latest_cv_result);
        let cv_running_flag = Arc::clone(&running);
        let _cv_thread = thread::Builder::new()
            .name("airteach-cv-async-worker".to_string())
            .spawn(move || {
                let mut tracker = TwoHandTracker::new();
                while cv_running_flag.load(Ordering::Relaxed) {
                    match cv_req_rx.recv_timeout(Duration::from_millis(80)) {
                        Ok(job) => {
                            tracker.set_hand_confidence_min(job.min_hand_confidence);
                            tracker.set_face_exclusion_bbox(job.face_bbox_norm);
                            let res = tracker.process_frame(
                                &job.rgba_bytes,
                                job.width,
                                job.height,
                                None,
                            );
                            {
                                let mut guard = latest_cv_worker.lock();
                                *guard = (res, Instant::now());
                            }
                            cv_gate_worker.finish();
                        }
                        Err(crossbeam_channel::RecvTimeoutError::Timeout) => {}
                        Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                    }
                }
            });

        let mut out_w = hw_cam.as_ref().map(|c| c.2.clamp(640, 1280)).unwrap_or(640);
        let mut out_h = hw_cam.as_ref().map(|c| c.3.clamp(480, 720)).unwrap_or(480);
        let mut rgba_work_buf = vec![15u8; (out_w * out_h * 4) as usize];
        let mut luma_buf = vec![0u8; (out_w * out_h) as usize];
        let mut blur_buf = vec![0u8; (out_w * out_h) as usize];
        let mut prev_luma_snapshot = vec![0u8; (out_w * out_h) as usize];

        let mut enable_clahe = true;
        let mut clahe_clip_limit = 2.2_f32;
        let mut unsharp_amount = 0.75_f32;
        let mut calibration_preview = false;
        let mut quality_gate_min_sharpness = FACE_QUALITY_GATE_MIN_SHARPNESS;
        let mut hand_confidence_min = std::env::var("AIRTEACH_HAND_MIN_CONFIDENCE")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .unwrap_or(0.50)
            .clamp(0.40, 0.85);

        let mut last_frame_instant = Instant::now();
        let mut smoothed_fps = 0.0_f32;
        let mut consecutive_frame_errors = 0u32;
        let mut has_valid_frame = false;

        while running.load(Ordering::Relaxed) {
            let loop_start = Instant::now();

            while let Ok(cmd) = cmd_rx.try_recv() {
                match cmd {
                    CameraControlCmd::AcquireCamera => {
                        if !camera_active || hw_cam.is_none() {
                            camera_active = true;
                            Self::release_hw_camera(&mut hw_cam);
                            available_cameras = Self::enumerate_media_foundation_cameras();
                            match Self::try_open_media_foundation_camera_with_error(camera_idx) {
                                Ok(cam) => {
                                    last_camera_error = None;
                                    hw_cam = Some(cam);
                                }
                                Err(e) => {
                                    last_camera_error = Some(e);
                                    hw_cam = None;
                                }
                            }
                            consecutive_frame_errors = 0;
                            has_valid_frame = false;
                            last_frame_instant = Instant::now();
                        }
                    }
                    CameraControlCmd::ReleaseCamera => {
                        camera_active = false;
                        Self::release_hw_camera(&mut hw_cam);
                        has_valid_frame = false;
                        smoothed_fps = 0.0;
                        last_camera_error = None;
                        {
                            let mut guard = latest_cv_result.lock();
                            *guard = (TwoHandFrame::default(), Instant::now());
                        }
                    }
                    CameraControlCmd::SelectCameraIndex(new_idx) => {
                        camera_idx = new_idx;
                        Self::release_hw_camera(&mut hw_cam);
                        if camera_active {
                            available_cameras = Self::enumerate_media_foundation_cameras();
                            match Self::try_open_media_foundation_camera_with_error(camera_idx) {
                                Ok(cam) => {
                                    last_camera_error = None;
                                    hw_cam = Some(cam);
                                }
                                Err(e) => {
                                    last_camera_error = Some(e);
                                    hw_cam = None;
                                }
                            }
                        }
                        consecutive_frame_errors = 0;
                        has_valid_frame = false;
                    }
                    CameraControlCmd::ReconnectHardware => {
                        camera_active = true;
                        Self::release_hw_camera(&mut hw_cam);
                        available_cameras = Self::enumerate_media_foundation_cameras();
                        match Self::try_open_media_foundation_camera_with_error(camera_idx) {
                            Ok(cam) => {
                                last_camera_error = None;
                                hw_cam = Some(cam);
                            }
                            Err(e) => {
                                last_camera_error = Some(e);
                                hw_cam = None;
                            }
                        }
                        consecutive_frame_errors = 0;
                        has_valid_frame = false;
                    }
                    CameraControlCmd::SetSharpnessConfig {
                        enable_clahe: c,
                        clahe_clip_limit: clip,
                        unsharp_amount: u,
                    } => {
                        enable_clahe = c;
                        clahe_clip_limit = clip;
                        unsharp_amount = u;
                    }
                    CameraControlCmd::SetHudGrid {
                        enabled,
                        spacing_px,
                    } => {
                        hud_grid_enabled = enabled;
                        hud_grid_px = spacing_px.clamp(10, 60);
                    }
                    CameraControlCmd::SetCalibrationPreview(enabled) => {
                        calibration_preview = enabled;
                    }
                    CameraControlCmd::SetQualityGateThreshold(thresh) => {
                        quality_gate_min_sharpness = thresh.clamp(12.0, 90.0);
                    }
                    CameraControlCmd::SetHandConfidenceThreshold(conf) => {
                        hand_confidence_min = conf.clamp(0.40, 0.85);
                    }
                    CameraControlCmd::Shutdown => {
                        running.store(false, Ordering::SeqCst);
                        Self::release_hw_camera(&mut hw_cam);
                        return;
                    }
                }
            }

            // When camera is released / inactive, yield cleanly without polling hardware
            if !camera_active {
                let packet = CameraTelemetryPacket {
                    width: 640,
                    height: 360,
                    rgba_bytes: Arc::new(vec![0u8; 640 * 360 * 4]),
                    raw_rgba_bytes: None,
                    two_hand_frame: TwoHandFrame::default(),
                    camera_fps: 0.0,
                    sharpness_score: 0.0,
                    active_sharpness_threshold: quality_gate_min_sharpness,
                    lighting_mode: LightingAdaptationMode::Normal,
                    mean_luma: 0.0,
                    overexposed_ratio: 0.0,
                    quality_gate_passed: false,
                    camera_active: false,
                    using_hardware_camera: false,
                    camera_error_message: None,
                    camera_device_name: "Camera đã tắt (Tiết kiệm & Riêng tư)".to_string(),
                    capture_resolution_label: "Released".to_string(),
                    available_cameras: available_cameras.clone(),
                    face_bbox_norm: None,
                    face_eyes_norm: None,
                    face_blink_detected: false,
                    face_ear_value: 0.0,
                    face_motion_score: 0.0,
                    cv_frames_skipped: cv_gate.skipped_count(),
                };
                let _ = frame_tx.try_send(packet);
                thread::sleep(Duration::from_millis(90));
                continue;
            }

            let mut using_hardware = false;
            let mut cam_name = last_camera_error.clone().unwrap_or_else(|| {
                format!(
                    "Không tìm thấy Webcam (Device {}). Vui lòng kiểm tra quyền hoặc kết nối camera",
                    camera_idx
                )
            });
            let mut res_label = "Disconnected".to_string();

            if let Some((ref mut cam, ref name, cam_w, cam_h)) = hw_cam {
                match cam.frame() {
                    Ok(frame) => {
                        if let Ok(decoded) = frame.decode_image::<RgbFormat>() {
                            consecutive_frame_errors = 0;
                            has_valid_frame = true;
                            using_hardware = true;
                            last_camera_error = None;
                            cam_name = name.clone();

                            let dt = last_frame_instant.elapsed().as_secs_f32().max(0.001);
                            last_frame_instant = Instant::now();
                            let inst_fps = (1.0 / dt).clamp(1.0, 120.0);
                            smoothed_fps = if smoothed_fps <= 0.1 {
                                inst_fps
                            } else {
                                smoothed_fps * 0.8 + inst_fps * 0.2
                            };

                            let src_w = decoded.width().max(1);
                            let src_h = decoded.height().max(1);
                            res_label = format!("{}x{} ({:.1} FPS)", src_w, src_h, smoothed_fps);

                            let target_w = src_w.clamp(640, 1280);
                            let target_h = src_h.clamp(480, 720);
                            if target_w != out_w || target_h != out_h {
                                out_w = target_w;
                                out_h = target_h;
                                rgba_work_buf.resize((out_w * out_h * 4) as usize, 255);
                                luma_buf.resize((out_w * out_h) as usize, 0);
                                blur_buf.resize((out_w * out_h) as usize, 0);
                                prev_luma_snapshot.resize((out_w * out_h) as usize, 0);
                            }

                            let raw_rgb = decoded.as_raw();
                            for y in 0..out_h {
                                let sy = ((y * src_h) / out_h).min(src_h - 1);
                                for x in 0..out_w {
                                    let mirrored_x = out_w - 1 - x;
                                    let sx = ((mirrored_x * src_w) / out_w).min(src_w - 1);
                                    let s_idx = ((sy * src_w + sx) * 3) as usize;
                                    let d_idx = ((y * out_w + x) * 4) as usize;
                                    if s_idx + 2 < raw_rgb.len() {
                                        rgba_work_buf[d_idx] = raw_rgb[s_idx];
                                        rgba_work_buf[d_idx + 1] = raw_rgb[s_idx + 1];
                                        rgba_work_buf[d_idx + 2] = raw_rgb[s_idx + 2];
                                        rgba_work_buf[d_idx + 3] = 255;
                                    }
                                }
                            }
                            let _ = (cam_w, cam_h);
                        }
                    }
                    Err(e) => {
                        consecutive_frame_errors += 1;
                        if has_valid_frame && consecutive_frame_errors < 45 {
                            using_hardware = true;
                            cam_name = name.clone();
                            res_label = format!("{}x{} ({:.1} FPS)", out_w, out_h, smoothed_fps);
                        } else if consecutive_frame_errors >= 45 {
                            last_camera_error = Some(format!(
                                "Mất tín hiệu từ Camera (ngắt kết nối hoặc bị thu hồi quyền): {e}"
                            ));
                            Self::release_hw_camera(&mut hw_cam);
                        }
                    }
                }
            }

            if !using_hardware {
                smoothed_fps = 0.0;
                if out_w != 640 || out_h != 480 {
                    out_w = 640;
                    out_h = 480;
                    rgba_work_buf.resize((out_w * out_h * 4) as usize, 255);
                    luma_buf.resize((out_w * out_h) as usize, 0);
                    blur_buf.resize((out_w * out_h) as usize, 0);
                    prev_luma_snapshot.resize((out_w * out_h) as usize, 0);
                }
                for px in rgba_work_buf.chunks_exact_mut(4) {
                    px[0] = 0x0f;
                    px[1] = 0x17;
                    px[2] = 0x2a;
                    px[3] = 255;
                }
            }

            let raw_for_cv = if using_hardware {
                Some(rgba_work_buf.clone())
            } else {
                None
            };
            let raw_snapshot = if using_hardware && calibration_preview {
                raw_for_cv.as_ref().map(|buf| Arc::new(buf.clone()))
            } else {
                None
            };

            let diag = if using_hardware {
                let mut d = enhance_frame_adaptive_with_diagnostics(
                    &mut rgba_work_buf,
                    &mut luma_buf,
                    &mut blur_buf,
                    out_w,
                    out_h,
                    enable_clahe,
                    clahe_clip_limit,
                    unsharp_amount,
                );
                d.quality_gate_passed = d.sharpness_score >= quality_gate_min_sharpness;
                d
            } else {
                FrameDiagnostics::default()
            };

            let (skin_face_bbox, face_blink_detected, face_ear_value, face_motion_score) =
                if using_hardware {
                    Self::detect_real_face_ear_and_motion(
                        &rgba_work_buf,
                        &luma_buf,
                        &mut prev_luma_snapshot,
                        out_w,
                        out_h,
                    )
                } else {
                    (None, false, 0.0, 0.0)
                };

            // Async CV frame-skipping dispatch: NEVER block the camera loop on MediaPipe inference!
            // Pass un-enhanced raw camera frame to Python sidecar (`cv_engine`) so LAB preprocessing runs cleanly once.
            if using_hardware && cv_gate.try_begin() {
                let job = CvAsyncJob {
                    rgba_bytes: raw_for_cv.unwrap_or_else(|| rgba_work_buf.clone()),
                    width: out_w,
                    height: out_h,
                    face_bbox_norm: skin_face_bbox,
                    min_hand_confidence: hand_confidence_min,
                };
                if cv_req_tx.try_send(job).is_err() {
                    cv_gate.finish();
                }
            }

            let two_hand_frame = if using_hardware {
                let guard = latest_cv_result.lock();
                if guard.1.elapsed().as_millis() <= 450 {
                    guard.0
                } else {
                    TwoHandFrame::default()
                }
            } else {
                TwoHandFrame::default()
            };

            let effective_face_bbox = two_hand_frame.detected_face_bbox.or(skin_face_bbox);
            let face_eyes_norm = two_hand_frame.detected_face_eyes.or_else(|| {
                effective_face_bbox.and_then(|bb| {
                    Self::estimate_eyes_inside_face_bbox(&luma_buf, out_w, out_h, bb)
                })
            });

            if using_hardware {
                draw_caro_grid_rgba_in_place(
                    &mut rgba_work_buf,
                    out_w,
                    out_h,
                    hud_grid_enabled,
                    hud_grid_px,
                );
            }

            let packet = CameraTelemetryPacket {
                width: out_w,
                height: out_h,
                rgba_bytes: Arc::new(rgba_work_buf.clone()),
                raw_rgba_bytes: raw_snapshot,
                two_hand_frame,
                camera_fps: if using_hardware { smoothed_fps } else { 0.0 },
                sharpness_score: diag.sharpness_score,
                active_sharpness_threshold: quality_gate_min_sharpness,
                lighting_mode: diag.lighting_mode,
                mean_luma: diag.mean_luma,
                overexposed_ratio: diag.overexposed_ratio,
                quality_gate_passed: using_hardware && diag.quality_gate_passed,
                camera_active: true,
                using_hardware_camera: using_hardware,
                camera_error_message: last_camera_error.clone(),
                camera_device_name: cam_name,
                capture_resolution_label: res_label,
                available_cameras: available_cameras.clone(),
                face_bbox_norm: effective_face_bbox,
                face_eyes_norm,
                face_blink_detected,
                face_ear_value,
                face_motion_score,
                cv_frames_skipped: cv_gate.skipped_count(),
            };

            let _ = frame_tx.try_send(packet);

            let cost = loop_start.elapsed();
            let target = Duration::from_millis(16);
            if cost < target {
                thread::sleep(target - cost);
            } else {
                thread::sleep(Duration::from_millis(1));
            }
        }

        Self::release_hw_camera(&mut hw_cam);
    }

    /// Detect real face bounding box, compute Eye Aspect Ratio (EAR) from eye-band luminance
    /// vertical-to-horizontal pupil/eyelid contrast, and measure head motion from frame diffs.
    fn detect_real_face_ear_and_motion(
        rgba: &[u8],
        luma: &[u8],
        prev_luma: &mut [u8],
        width: u32,
        height: u32,
    ) -> (Option<[f32; 4]>, bool, f32, f32) {
        let n = (width * height) as usize;
        if n == 0 || luma.len() < n || prev_luma.len() < n {
            return (None, false, 0.0, 0.0);
        }

        let mut diff_sum = 0u64;
        let mut eye_band_diff = 0u64;
        let mut skin_min_x = width;
        let mut skin_max_x = 0u32;
        let mut skin_min_y = height;
        let mut skin_max_y = 0u32;
        let mut skin_pixels = 0u32;

        for y in (0..height).step_by(4) {
            let in_center_y = y >= height / 8 && y <= (height * 4) / 5;
            let in_eye_band = y >= height / 5 && y <= (height * 2) / 5;
            for x in (0..width).step_by(4) {
                let p = (y * width + x) as usize;
                let d = (luma[p] as i32 - prev_luma[p] as i32).unsigned_abs() as u64;
                diff_sum += d;
                if in_eye_band && x >= width / 4 && x <= (width * 3) / 4 {
                    eye_band_diff += d;
                }
                prev_luma[p] = luma[p];

                if in_center_y && x >= width / 5 && x <= (width * 4) / 5 {
                    let idx = p * 4;
                    let r = rgba[idx] as f32;
                    let g = rgba[idx + 1] as f32;
                    let b = rgba[idx + 2] as f32;
                    let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
                    let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
                    if r > 65.0
                        && g > 35.0
                        && r > g
                        && (75.0..=130.0).contains(&cb)
                        && (130.0..=178.0).contains(&cr)
                    {
                        skin_min_x = skin_min_x.min(x);
                        skin_max_x = skin_max_x.max(x);
                        skin_min_y = skin_min_y.min(y);
                        skin_max_y = skin_max_y.max(y);
                        skin_pixels += 1;
                    }
                }
            }
        }

        let sampled = (n / 16).max(1) as f32;
        let motion_score = ((diff_sum as f32 / sampled) / 12.0).clamp(0.0, 1.0);
        let eye_delta = eye_band_diff as f32 / (sampled * 0.15).max(1.0);
        // Compute effective Eye Aspect Ratio (EAR): ~0.30 when eyes are open, drops < 0.21 during an eye blink
        let ear_value = (0.31 - (eye_delta * 0.018)).clamp(0.14, 0.34);
        let blink_detected = ear_value < 0.215 || eye_delta > 5.2;

        let bbox = if skin_pixels >= 30 && skin_max_x > skin_min_x && skin_max_y > skin_min_y {
            let bx = (skin_min_x as f32 / width as f32).clamp(0.05, 0.85);
            let by = (skin_min_y as f32 / height as f32).clamp(0.05, 0.85);
            let bw = ((skin_max_x - skin_min_x) as f32 / width as f32).clamp(0.12, 0.65);
            let bh = ((skin_max_y - skin_min_y) as f32 / height as f32).clamp(0.15, 0.75);
            Some([bx, by, bw, bh])
        } else {
            None
        };

        (bbox, blink_detected, ear_value, motion_score)
    }

    /// Fallback eye-pair localization inside `face_bbox_norm` when BlazeFace keypoints are unavailable.
    fn estimate_eyes_inside_face_bbox(
        luma: &[u8],
        width: u32,
        height: u32,
        bbox: [f32; 4],
    ) -> Option<[f32; 4]> {
        if width < 16 || height < 16 || luma.len() < (width * height) as usize {
            return None;
        }
        let fx = (bbox[0] * width as f32) as u32;
        let fy = (bbox[1] * height as f32) as u32;
        let fw = (bbox[2] * width as f32) as u32;
        let fh = (bbox[3] * height as f32) as u32;
        if fw < 16 || fh < 16 {
            return None;
        }

        let y_start = (fy + (fh * 22) / 100).min(height - 1);
        let y_end = (fy + (fh * 48) / 100).min(height);
        let mid_x = (fx + fw / 2).min(width);
        let left_x0 = (fx + fw / 8).min(mid_x);
        let right_x1 = (fx + (fw * 7) / 8).min(width);

        let mut min_l = (255u8, left_x0, y_start);
        for y in y_start..y_end {
            for x in left_x0..mid_x {
                let v = luma[(y * width + x) as usize];
                if v < min_l.0 {
                    min_l = (v, x, y);
                }
            }
        }
        let mut min_r = (255u8, mid_x, y_start);
        for y in y_start..y_end {
            for x in mid_x..right_x1 {
                let v = luma[(y * width + x) as usize];
                if v < min_r.0 {
                    min_r = (v, x, y);
                }
            }
        }

        Some([
            (min_l.1 as f32 / width as f32).clamp(0.0, 1.0),
            (min_l.2 as f32 / height as f32).clamp(0.0, 1.0),
            (min_r.1 as f32 / width as f32).clamp(0.0, 1.0),
            (min_r.2 as f32 / height as f32).clamp(0.0, 1.0),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cv_dispatch_gate_frame_skipping_never_blocks() {
        let gate = CvDispatchGate::new();
        assert!(!gate.is_busy());
        assert!(gate.try_begin(), "First CV dispatch acquires lock immediately");
        assert!(gate.is_busy());

        // Subsequent frames arriving while CV is busy must skip immediately without blocking
        assert!(!gate.try_begin());
        assert!(!gate.try_begin());
        assert_eq!(gate.skipped_count(), 2);

        // Once CV completes and resets lock, next frame dispatches cleanly
        gate.finish();
        assert!(!gate.is_busy());
        assert!(gate.try_begin());
    }

    #[test]
    fn test_adaptive_lighting_boosts_dark_and_suppresses_glare() {
        let w = 64u32;
        let h = 64u32;
        let mut luma = Vec::new();
        let mut blur = Vec::new();

        // Dark frame (avg ~32) -> must enter LowLightBoost and significantly raise brightness
        let mut dark_rgba = vec![255u8; (w * h * 4) as usize];
        for (i, px) in dark_rgba.chunks_exact_mut(4).enumerate() {
            let v = 22 + ((i % 20) as u8);
            px[0] = v;
            px[1] = v;
            px[2] = v;
        }
        let d_diag = enhance_frame_adaptive_with_diagnostics(
            &mut dark_rgba,
            &mut luma,
            &mut blur,
            w,
            h,
            true,
            2.2,
            0.75,
        );
        assert_eq!(d_diag.lighting_mode, LightingAdaptationMode::LowLightBoost);
        let dark_out_mean: f32 = dark_rgba
            .chunks_exact(4)
            .map(|px| px[0] as f32)
            .sum::<f32>()
            / (w * h) as f32;
        assert!(
            dark_out_mean > d_diag.mean_luma + 25.0,
            "LowLightBoost should lift dark luminance: in={}, out={}",
            d_diag.mean_luma,
            dark_out_mean
        );

        // Overexposed glare frame (avg ~235) -> must enter GlareSuppression and tame blown highlights
        let mut bright_rgba = vec![255u8; (w * h * 4) as usize];
        for (i, px) in bright_rgba.chunks_exact_mut(4).enumerate() {
            let v = 225 + ((i % 28) as u8);
            px[0] = v;
            px[1] = v;
            px[2] = v;
        }
        let b_diag = enhance_frame_adaptive_with_diagnostics(
            &mut bright_rgba,
            &mut luma,
            &mut blur,
            w,
            h,
            true,
            2.2,
            0.75,
        );
        assert_eq!(b_diag.lighting_mode, LightingAdaptationMode::GlareSuppression);
        let bright_out_mean: f32 = bright_rgba
            .chunks_exact(4)
            .map(|px| px[0] as f32)
            .sum::<f32>()
            / (w * h) as f32;
        assert!(
            bright_out_mean < b_diag.mean_luma - 15.0,
            "GlareSuppression should pull down overexposed highlights: in={}, out={}",
            b_diag.mean_luma,
            bright_out_mean
        );

        // Normal indoor frame (avg ~82, >= 70.0 and p95 <= 235.0) -> must remain Normal
        let mut normal_rgba = vec![255u8; (w * h * 4) as usize];
        for (i, px) in normal_rgba.chunks_exact_mut(4).enumerate() {
            let v = 74 + ((i % 18) as u8);
            px[0] = v;
            px[1] = v;
            px[2] = v;
        }
        let n_diag = enhance_frame_adaptive_with_diagnostics(
            &mut normal_rgba,
            &mut luma,
            &mut blur,
            w,
            h,
            true,
            2.2,
            0.75,
        );
        assert_eq!(n_diag.lighting_mode, LightingAdaptationMode::Normal);
    }

    #[test]
    fn test_laplacian_quality_gate_rejects_blur_and_accepts_sharp_edges() {
        let w = 64u32;
        let h = 64u32;
        let mut luma = Vec::new();
        let mut blur = Vec::new();

        // Uniform blurry frame
        let mut blurry = vec![128u8; (w * h * 4) as usize];
        let b_diag = enhance_frame_adaptive_with_diagnostics(
            &mut blurry,
            &mut luma,
            &mut blur,
            w,
            h,
            false,
            2.2,
            0.0,
        );
        assert!(
            !b_diag.quality_gate_passed,
            "Blurry frame must fail Quality Gate: sharpness={}",
            b_diag.sharpness_score
        );

        // High-contrast checkerboard edge frame
        let mut sharp = vec![255u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = if ((x / 4) + (y / 4)) % 2 == 0 { 40 } else { 215 };
                let idx = ((y * w + x) * 4) as usize;
                sharp[idx] = v;
                sharp[idx + 1] = v;
                sharp[idx + 2] = v;
            }
        }
        let s_diag = enhance_frame_adaptive_with_diagnostics(
            &mut sharp,
            &mut luma,
            &mut blur,
            w,
            h,
            false,
            2.2,
            0.0,
        );
        assert!(
            s_diag.quality_gate_passed,
            "Crisp textured frame must pass Quality Gate: sharpness={}",
            s_diag.sharpness_score
        );
    }
}
