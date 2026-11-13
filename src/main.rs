#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]
#![allow(unused_imports, dead_code, non_snake_case)]

//! AirTeach AI — Portable Single-EXE Rust Desktop Application
//! Entry point & `eframe` (`wgpu` DirectX 12 / Vulkan) hardware-accelerated renderer setup.

pub mod ai;
pub mod app;
pub mod core;
pub mod ui;
pub mod utils;

use app::AirTeachApp;
use std::sync::Arc;
use std::time::Instant;

fn build_embedded_app_icon() -> egui::IconData {
    crate::ui::brand::rasterize_brand_logo_icon(64)
}

fn main() -> eframe::Result<()> {
    let launch_instant = Instant::now();

    // Support headless self-test & latency benchmark verification (`airteach-ai.exe --self-test`)
    if std::env::args().any(|a| a == "--self-test") {
        run_headless_self_test(launch_instant);
        return Ok(());
    }

    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();

    let viewport = egui::ViewportBuilder::default()
        .with_title("AirTeach AI — DNT - Victer | Interactive Touchless Smartboard & AI Assistant")
        .with_inner_size([1360.0, 820.0])
        .with_min_inner_size([1024.0, 680.0])
        .with_icon(Arc::new(build_embedded_app_icon()));

    let native_options = eframe::NativeOptions {
        viewport,
        renderer: eframe::Renderer::Wgpu,
        vsync: true,
        ..Default::default()
    };

    eframe::run_native(
        "AirTeach AI — DNT - Victer",
        native_options,
        Box::new(move |cc| Ok(Box::new(AirTeachApp::new(cc, launch_instant)))),
    )
}

fn run_headless_self_test(launch_instant: Instant) {
    use crate::ai::face_auth::{FaceAuthEngine, EMBEDDING_DIM};
    use crate::core::gestures::{
        GestureDebounceFsm, RightGesture, DEFAULT_GESTURE_HOLD_FRAMES,
    };
    use crate::core::hand_tracker::{HandLandmarks, Handedness};
    use crate::core::stroke_engine::{Point2D, StrokeEngine};
    use crate::utils::storage::LocalStorage;

    let storage = LocalStorage::open_in_memory();
    let cfg = storage.load_or_init_config();
    assert!(
        (cfg.face_match_threshold - crate::ai::face_auth::DEFAULT_FACE_MATCH_THRESHOLD).abs()
            < 0.001
    );

    let sample_pw = "AirTeach#Selftest9";
    let pw_hash = FaceAuthEngine::hash_password_argon2(sample_pw).expect("Argon2id hash");
    assert!(FaceAuthEngine::verify_password_argon2(sample_pw, &pw_hash));

    let mut stroke_engine = StrokeEngine::new(1.35, 0.012, 1.0, 1.8);
    let mut out_pts = Vec::with_capacity(512);
    let stroke_t0 = Instant::now();
    let raw: Vec<Point2D> = (0..60)
        .map(|i| {
            stroke_engine.filter_point(
                Point2D::new(i as f32 * 4.0, (i as f32 * 0.1).sin() * 20.0),
                i as f64 * 0.008,
            )
        })
        .collect();
    let _shape = stroke_engine.finalize_stroke(&raw, &mut out_pts);
    let stroke_latency_ms = stroke_t0.elapsed().as_secs_f64() * 1000.0;

    let mut fsm = GestureDebounceFsm::production();
    let rh = HandLandmarks::from_pose_preset(
        Handedness::Right,
        0.5,
        0.5,
        1.0,
        true,
        true,
        false,
        false,
        false,
    );
    let mut t_sec = 0.0_f64;
    let mut last_gs = fsm.update_at(None, Some(&rh), t_sec);
    for _ in 0..DEFAULT_GESTURE_HOLD_FRAMES {
        t_sec += 0.035;
        last_gs = fsm.update_at(None, Some(&rh), t_sec);
    }
    assert_eq!(last_gs.right_gesture, RightGesture::DrawStroke);

    let mut emb = [0.0_f32; EMBEDDING_DIM];
    for (i, v) in emb.iter_mut().enumerate() {
        *v = (i as f32 + 1.0) * 0.01;
    }
    crate::ai::face_auth::normalize_l2_in_place(&mut emb);

    let cold_start_ms = launch_instant.elapsed().as_secs_f64() * 1000.0;
    let cams = crate::core::camera::CameraSubsystem::enumerate_media_foundation_cameras();
    println!("DETECTED_CAMERAS={:?}", cams);
    println!(
        "AIRTEACH_SELF_TEST_OK cold_start_ms={:.2} stroke_latency_ms={:.3} out_vertices={} cameras_detected={}",
        cold_start_ms,
        stroke_latency_ms,
        out_pts.len(),
        cams.len()
    );
}
