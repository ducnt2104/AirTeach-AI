//! Performance Pipeline & Hardware Telemetry (`src/core/performance_pipeline.rs`):
//! - Tracks active AI Routing Mode, Model ID, tokens/s, CV FPS, Render FPS, and HW backend.
//! - Enforces Dynamic Resolution Scaling (`<= 640x480` aspect-preserved) for CV Hand/Face tracking
//!   while vector Canvas Ink & UI render at crisp native display DPI.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTelemetry {
    pub ai_mode: String,
    pub active_model: String,
    pub tokens_per_sec: f32,
    pub cv_fps: f32,
    pub render_fps: f32,
    pub hw_backend: String,
    pub last_latency_ms: f32,
}

impl Default for PerformanceTelemetry {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            ai_mode: "Hybrid".to_string(),
            active_model: "gemini-3.8-flash".to_string(),
            tokens_per_sec: 0.0,
            cv_fps: 0.0,
            render_fps: 60.0,
            hw_backend: format!("DirectML + CPU-SIMD ({threads}T)"),
            last_latency_ms: 0.0,
        }
    }
}

impl PerformanceTelemetry {
    pub fn status_pill_text(&self) -> String {
        let tok = if self.tokens_per_sec > 0.1 {
            format!("{:.1} tok/s", self.tokens_per_sec)
        } else {
            "Sẵn sàng".to_string()
        };
        let cv = if self.cv_fps > 0.5 {
            format!("CV {:.0} FPS", self.cv_fps)
        } else {
            "CV Standby".to_string()
        };
        format!("AI: {} ({}) • {} • {}", self.ai_mode, self.active_model, tok, cv)
    }
}

/// Compute optimal CV inference resolution (`<= max_w x max_h`, aspect-preserved)
/// so Hand Tracking runs at 60 FPS while UI vectors remain full DPI.
pub fn compute_cv_tracking_size(src_w: u32, src_h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    if src_w == 0 || src_h == 0 {
        return (640, 480);
    }
    if src_w <= max_w && src_h <= max_h {
        return (src_w, src_h);
    }
    let scale = (max_w as f32 / src_w as f32).min(max_h as f32 / src_h as f32);
    (
        ((src_w as f32 * scale).round() as u32).max(160),
        ((src_h as f32 * scale).round() as u32).max(120),
    )
}

pub struct PerformancePipeline {
    pub telemetry: PerformanceTelemetry,
    render_times: VecDeque<Instant>,
}

impl Default for PerformancePipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformancePipeline {
    pub fn new() -> Self {
        Self {
            telemetry: PerformanceTelemetry::default(),
            render_times: VecDeque::with_capacity(128),
        }
    }

    pub fn record_render_tick(&mut self) -> f32 {
        let now = Instant::now();
        self.render_times.push_back(now);
        while let Some(&front) = self.render_times.front() {
            if now.duration_since(front).as_secs_f32() > 1.0 {
                self.render_times.pop_front();
            } else {
                break;
            }
        }
        if self.render_times.len() >= 2 {
            let span = now
                .duration_since(*self.render_times.front().unwrap())
                .as_secs_f32()
                .max(1e-3);
            self.telemetry.render_fps = ((self.render_times.len() - 1) as f32 / span).min(240.0);
        }
        self.telemetry.render_fps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_cv_scaling_bounds_1080p_to_640x360() {
        let (tw, th) = compute_cv_tracking_size(1920, 1080, 640, 480);
        assert_eq!((tw, th), (640, 360));
        let (sw, sh) = compute_cv_tracking_size(640, 480, 640, 480);
        assert_eq!((sw, sh), (640, 480));
    }
}
