//! Master Orchestration Application State Machine (`src/app.rs`):
//! States: `[Welcome, Auth, Dashboard, LectureMode, Settings]`
//!
//! Synthesizes all 4 Workstreams:
//! - Workstream 1: Brand Identity (`AirTeach AI — DNT - Victer`), Dark/Light Design System tokens,
//!   High-Contrast Classroom Accessibility Modes, Floating Glassmorphic Toolbar, Dynamic Layout Modes.
//! - Workstream 2: Async MediaFoundation Camera Lifecycle + `AtomicBool` frame-skipping + One-Euro
//!   21-landmark smoothing + Distance-Invariant Gesture FSM, 44.1kHz PCM `cpal`/`hound` Audio Recording.
//! - Workstream 3: Multimodal Homework Inspector ("Phân Tích Vùng Chọn"), In-Memory Slide Context RAG,
//!   and Interactive STEM Knowledge Engine (PubChem / PDB 2D/3D structure generator).
//! - Workstream 4: Local-First SQLite `users` + `sessions`, Argon2id Auth, Guided 3-5 Angle Face ID,
//!   AES-256-GCM encryption, and `crossbeam_channel::bounded(2)` backpressure.

use crate::ai::face_auth::{
    FaceAuthEngine, UserAccount, DEFAULT_FACE_MATCH_THRESHOLD, EMBEDDING_DIM,
    ENROLL_ANGLE_PROMPTS, MAX_FACE_ENROLL_SAMPLES, MIN_FACE_ENROLL_SAMPLES, SFACE_EMBEDDING_DIM,
};
use crate::ai::gemini_chat::{
    mask_api_key, ApiConnectionStatus, ChatMessage, ChatStreamEvent, GeminiChatClient,
    GeminiHealthCheckResult, PedagogicalAnalysisMode, SubjectAssistantMode,
    PEDAGOGICAL_INSPECTOR_SYSTEM_PROMPT,
};
use crate::ai::lasso_ai_service::{
    extract_clean_beautified_formula, LassoQuickAction, LassoSelectionRegion,
    DEFAULT_LASSO_PADDING_PX,
};
use crate::ai::latex_engine::{LatexEngine, LatexRecognitionResult};
use crate::ai::local_ai_manager::{AiRoutingMode, DiscoveredLocalServer, LocalAIManager};
use crate::ai::ocr_engine::{OcrEngine, OcrRecognitionResult};
use crate::ai::rag_engine::SlideRagIndex;
use crate::ai::stem_knowledge::StemKnowledgeEngine;
use crate::ai::subtitles_engine::{LiveSubtitlesEngine, TranscriptSegmentEvent};
use crate::ai::whiteboard_agent::{
    build_vector_shape_strokes, find_free_board_region, find_free_regions_normalized, BoardRect,
    WhiteboardAction, WhiteboardAgent,
};
use crate::core::audio_recorder::{AudioRecorder, RecordingState};
use crate::core::camera::{CameraControlCmd, CameraSubsystem, LightingAdaptationMode};
use crate::core::gestures::{
    classify_right_hand_raw, compute_pinch_ratio, DrawingController4Tier, DrawingFsmAction,
    DrawingHudMode, DualHandGestureState, GestureDebounceFsm, LeftGesture, RightGesture,
    RightTransitionEvent, UNDO_COOLDOWN_MS,
};
use crate::core::hand_tracker::TwoHandFrame;
use crate::core::performance_pipeline::PerformancePipeline;
use crate::core::stroke_engine::{
    interpolate_bridge_points_2d, Point2D, SnappedShape, StrokeEngine, DEFAULT_INTERP_STEP_PX,
};
use crate::ui::brand::{render_about_modal, ui_brand_logo, ThemeMode};
use crate::ui::toolbar_view::{
    ALERT_ROSE, BG_DARK, BORDER_SLATE, PRIMARY_INDIGO, SECONDARY_TEAL, SURFACE_DARK, TEXT_LIGHT,
    TEXT_MUTED,
};
use crate::ui::{
    auth_view, calibration_view, canvas_view, chat_view, settings_view, toolbar_view, welcome_view,
};
use crate::utils::document_loader::{DocumentLoader, SlidePage};
use crate::utils::model_downloader::ModelDownloader;
use crate::utils::output_sanitizer::OutputSanitizer;
use crate::utils::storage::{
    AirTeachBundle, AppConfig, BoardTextAnnotation, LectureHistoryEntry, LocalStorage,
    PersistedStroke, VectorStroke,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use egui::{
    Color32, ColorImage, FontId, Pos2, Rect, RichText, Rounding, Stroke, TextureHandle, Vec2,
};
use std::collections::VecDeque;
use image::codecs::png::PngEncoder;
use image::{ColorType, ImageBuffer, ImageEncoder, Rgba};
use std::path::PathBuf;
use std::thread;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppScreen {
    Welcome,
    Auth,
    Dashboard,
    LectureMode,
    Settings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActiveBoardTool {
    Pen,
    LaserPointer,
    Lasso,
    SelectRegion,
    Eraser,
}

enum VisionTaskReq {
    RunOcr {
        strokes: Vec<Vec<Point2D>>,
        min_pt: Point2D,
        max_pt: Point2D,
        subject: String,
    },
    RunLatex {
        strokes: Vec<Vec<Point2D>>,
        min_pt: Point2D,
        max_pt: Point2D,
        subject: String,
        force_low_conf: bool,
    },
}

enum VisionTaskResp {
    Ocr(OcrRecognitionResult),
    Latex(LatexRecognitionResult),
}

pub struct AirTeachApp {
    pub screen: AppScreen,
    pub cold_start_ms: f32,
    pub app_epoch: Instant,

    // Storage, Config & Model Downloader
    pub storage: LocalStorage,
    pub config: AppConfig,
    pub lecture_history: Vec<LectureHistoryEntry>,
    pub model_downloader: ModelDownloader,

    // Real Hardware Camera & Pre-allocated ColorImage Buffer (Explicit Acquire/Release Lifecycle)
    pub camera: CameraSubsystem,
    pub active_camera_index: u32,
    pub camera_color_image: ColorImage,
    pub camera_texture: Option<TextureHandle>,
    pub camera_raw_color_image: ColorImage,
    pub camera_raw_texture: Option<TextureHandle>,
    pub calibration_modal_open: bool,
    pub camera_fps: f32,
    pub camera_sharpness: f32,
    pub camera_lighting_mode: LightingAdaptationMode,
    pub camera_mean_luma: f32,
    pub camera_overexposed_ratio: f32,
    pub camera_quality_gate_passed: bool,
    pub camera_acquired: bool,
    pub camera_using_hardware: bool,
    pub camera_error_message: Option<String>,
    pub camera_device_name: String,
    pub camera_resolution_label: String,
    pub last_two_hand_frame: TwoHandFrame,
    pub last_face_eyes_norm: Option<[f32; 4]>,
    pub last_face_5pt_norm: Option<[f32; 10]>,
    pub hud_right_trail: VecDeque<[f32; 2]>,
    pub about_modal_open: bool,

    // Real Microphone Audio Recorder + Stroke-Synced Timeline & Live Subtitles
    pub audio_recorder: AudioRecorder,
    pub recording_epoch: Instant,
    pub replay_active: bool,
    pub replay_started_at: Instant,
    pub replay_cursor_ms: u64,
    pub replay_duration_ms: u64,
    pub subtitles_engine: LiveSubtitlesEngine,
    stt_event_tx: Sender<Result<TranscriptSegmentEvent, String>>,
    stt_event_rx: Receiver<Result<TranscriptSegmentEvent, String>>,

    // Local-First Dual Auth & Active SQLite Session
    pub face_auth: FaceAuthEngine,
    pub auth_tab_is_signup: bool,
    pub signup_step: u8,
    pub signin_face_mode: bool,
    pub registered_users: Vec<UserAccount>,
    pub selected_user_idx: usize,
    pub signup_full_name: String,
    pub signup_username: String,
    pub signup_pin: String,
    pub signup_pin_confirm: String,
    pub signup_face_samples: Vec<Vec<f32>>,
    pub signup_status_msg: String,
    pub logged_in_user_id: Option<String>,
    pub logged_in_teacher_name: String,
    pub logged_in_username: String,
    pub slide_card_visible: bool,
    pub enrolled_face_embedding: Vec<f32>,
    pub enrolled_face_templates: Vec<Vec<f32>>,
    pub last_live_face_embedding: [f32; EMBEDDING_DIM],
    pub last_live_face_vec: Vec<f32>,
    pub last_face_bbox_norm: Option<[f32; 4]>,
    pub last_face_similarity: f32,
    pub auto_face_unlock_enabled: bool,
    pub pin_input_buffer: String,
    pub auth_error_msg: String,

    // Dual-Hand Gesture FSM, 4-Tier Drawing FSM & Navigation State
    pub gesture_fsm: GestureDebounceFsm,
    pub drawing_fsm: DrawingController4Tier,
    pub hud_mode: DrawingHudMode,
    pub pen_lock_remaining_ms: f32,
    pub last_undo_ms: f64,
    pub gesture_state: DualHandGestureState,
    pub prev_left_hand_norm: Option<Point2D>,
    pub prev_left_pinch_span: Option<f32>,
    pub left_anchor_origin: Vec2,

    // Lecture Board, Slide Deck, RAG Index & Zero-Alloc Stroke Buffers
    pub slides: Vec<SlidePage>,
    pub rag_index: SlideRagIndex,
    pub current_slide_idx: usize,
    pub current_slide_texture: Option<TextureHandle>,
    pub slide_texture_dirty: bool,
    pub canvas_pan: Vec2,
    pub canvas_zoom: f32,
    pub last_canvas_rect: Option<Rect>,
    pub floating_toolbar_collapsed: bool,
    pub left_camera_dock_open: bool,

    pub active_tool: ActiveBoardTool,
    pub current_ink_color: [u8; 4],
    pub current_stroke_thickness: f32,
    pub stroke_engine: StrokeEngine,
    pub strokes: Vec<PersistedStroke>,
    pub redo_strokes: Vec<PersistedStroke>,
    pub redo_replace_flags: Vec<bool>,
    pub annotations: Vec<BoardTextAnnotation>,
    pub active_stroke_raw: Vec<Point2D>,
    pub finalize_scratch_points: Vec<Point2D>,
    pub render_scratch_pos2: Vec<Pos2>,
    pub stroke_latency_ms: f32,
    pub last_snapped_shape_banner: String,
    pub last_snapped_shape_index: Option<usize>,
    pub last_snapped_toast_anchor: Option<Point2D>,
    pub last_snapped_toast_until_ms: f64,
    prev_board_hand_present: bool,

    pub laser_pointer_board: Option<Point2D>,
    pub selection_anchor_board: Option<Point2D>,
    pub selection_rect_board: Option<(Point2D, Point2D)>,
    pub lasso_points: Vec<Point2D>,
    pub active_lasso_region: Option<LassoSelectionRegion>,
    pub pending_lasso_action: Option<LassoQuickAction>,
    pub pending_low_confidence_confirmation: Option<BoardTextAnnotation>,
    pub ai_action_undo_stack: Vec<(Vec<PersistedStroke>, Vec<BoardTextAnnotation>)>,
    pub pending_ai_strokes: Vec<(usize, Vec<Point2D>)>,

    // Bounded(2) Background OCR & LaTeX Worker Channels (Workstream 4.2 Zero-Leak Guarantee)
    vision_req_tx: Sender<VisionTaskReq>,
    vision_resp_rx: Receiver<VisionTaskResp>,

    // Gemini & Local Hybrid AI Streaming + Performance Telemetry
    pub perf_pipeline: PerformancePipeline,
    pub gemini_client: GeminiChatClient,
    pub discovered_local_servers: Vec<DiscoveredLocalServer>,
    pub active_local_provider: String,
    local_discovery_tx: Sender<Vec<DiscoveredLocalServer>>,
    local_discovery_rx: Receiver<Vec<DiscoveredLocalServer>>,
    pub api_connection_status: ApiConnectionStatus,
    pub api_status_detail_vi: String,
    pub api_test_in_flight: bool,
    api_health_tx: Sender<GeminiHealthCheckResult>,
    api_health_rx: Receiver<GeminiHealthCheckResult>,
    pub subject_mode: SubjectAssistantMode,
    pub chat_drawer_open: bool,
    pub chat_messages: Vec<ChatMessage>,
    pub chat_input_draft: String,
    pub chat_streaming_active: bool,
    chat_event_tx: Sender<ChatStreamEvent>,
    chat_event_rx: Receiver<ChatStreamEvent>,

    // Settings UI Scratch Buffers
    pub api_key_edit_buffer: String,
    pub show_api_key_plain: bool,
    pub new_pin_edit_buffer: String,
    pub settings_status_banner: String,
    pub vision_debug_hud_enabled: bool,
}

impl AirTeachApp {
    pub fn new(cc: &eframe::CreationContext<'_>, launch_instant: Instant) -> Self {
        Self::configure_modern_theme(&cc.egui_ctx);

        let storage = LocalStorage::open_default();
        let face_auth = FaceAuthEngine::new();

        let config = storage.load_or_init_config();
        config.theme_mode.palette().apply_to_egui(&cc.egui_ctx);
        storage.save_config(&config);

        let registered_users = storage.list_users().unwrap_or_default();
        let auth_tab_is_signup = registered_users.is_empty();

        // Check if an explicit session exists, otherwise Clean Slate starts at Welcome -> Sign-Up/Login.
        let mut logged_in_user_id = None;
        let mut logged_in_teacher_name = String::new();
        let mut logged_in_username = String::new();
        let mut enrolled_face_embedding = registered_users
            .first()
            .map(|u| u.face_embedding.clone())
            .unwrap_or_default();
        let mut enrolled_face_templates = registered_users
            .first()
            .map(|u| u.face_templates.clone())
            .unwrap_or_default();
        let mut initial_screen = AppScreen::Welcome;

        if !registered_users.is_empty() {
            if let Ok(Some((_sess, active_user))) = storage.get_active_session_user() {
                logged_in_user_id = Some(active_user.id.clone());
                logged_in_teacher_name = active_user.display_name.clone();
                logged_in_username = active_user.username.clone();
                enrolled_face_embedding = active_user.face_embedding.clone();
                enrolled_face_templates = active_user.face_templates.clone();
                initial_screen = AppScreen::Dashboard;
            }
        }

        let stored_key = storage.get_decrypted_api_key(&config);
        let api_key_plain = GeminiChatClient::resolve_api_key(&stored_key);
        let lecture_history = logged_in_user_id
            .as_deref()
            .map(|uid| storage.list_lecture_history(uid))
            .unwrap_or_default();

        let gemini_client = GeminiChatClient::new();
        let model_downloader = ModelDownloader::new();
        model_downloader.ensure_models_async(&gemini_client.runtime);

        // Spawn MediaFoundation Camera Worker in Standby (`start_active = false`):
        // Hardware stream activates ONLY when explicitly requested (Face ID / Capture / LectureMode).
        let active_camera_index = config.camera_index;
        let camera = CameraSubsystem::spawn(
            active_camera_index,
            config.hud_grid_enabled,
            config.hud_grid_px,
            false,
        );
        let _ = camera.cmd_tx.try_send(CameraControlCmd::SetSharpnessConfig {
            enable_clahe: config.enable_clahe,
            clahe_clip_limit: config.clahe_clip_limit,
            unsharp_amount: config.unsharp_amount,
        });
        let _ = camera
            .cmd_tx
            .try_send(CameraControlCmd::SetQualityGateThreshold(
                config.quality_gate_min_sharpness,
            ));
        let _ = camera
            .cmd_tx
            .try_send(CameraControlCmd::SetHandConfidenceThreshold(
                config.hand_confidence_min,
            ));

        // Initialize Real Microphone Audio Recorder (cpal + hound)
        let audio_recorder = AudioRecorder::new();

        // Spawn Background OCR & RapidLaTeXOCR Worker Thread with bounded(2) channels (Workstream 4.2)
        let (vision_req_tx, vision_req_rx) = bounded::<VisionTaskReq>(2);
        let (vision_resp_tx, vision_resp_rx) = bounded::<VisionTaskResp>(2);
        thread::Builder::new()
            .name("airteach-ocr-latex-worker".to_string())
            .spawn(move || {
                let ocr_engine = OcrEngine::new();
                let latex_engine = LatexEngine::new();
                while let Ok(req) = vision_req_rx.recv() {
                    match req {
                        VisionTaskReq::RunOcr {
                            strokes,
                            min_pt,
                            max_pt,
                            subject,
                        } => {
                            let res =
                                ocr_engine.recognize_strokes(&strokes, min_pt, max_pt, &subject);
                            let _ = vision_resp_tx.try_send(VisionTaskResp::Ocr(res));
                        }
                        VisionTaskReq::RunLatex {
                            strokes,
                            min_pt,
                            max_pt,
                            subject,
                            force_low_conf,
                        } => {
                            let res = latex_engine.recognize_latex_from_strokes(
                                &strokes,
                                min_pt,
                                max_pt,
                                &subject,
                                force_low_conf,
                            );
                            let _ = vision_resp_tx.try_send(VisionTaskResp::Latex(res));
                        }
                    }
                }
            })
            .expect("Failed to spawn OCR/LaTeX background thread");

        let (chat_event_tx, chat_event_rx) = unbounded::<ChatStreamEvent>();
        let (api_health_tx, api_health_rx) = unbounded::<GeminiHealthCheckResult>();
        let (local_discovery_tx, local_discovery_rx) = unbounded::<Vec<DiscoveredLocalServer>>();
        let (stt_event_tx, stt_event_rx) = unbounded::<Result<TranscriptSegmentEvent, String>>();

        LocalAIManager::discover_async(&gemini_client.runtime, local_discovery_tx.clone());

        let mut stroke_engine = StrokeEngine::new(
            config.one_euro_min_cutoff,
            config.one_euro_beta,
            config.one_euro_d_cutoff,
            config.rdp_epsilon,
        );
        stroke_engine.auto_shape_snap = config.auto_shape_snap;

        let mut slides = Vec::new();
        if !config.tutorial_deleted && !config.tutorial_dismissed {
            slides.extend(DocumentLoader::create_tutorial_deck());
        } else {
            slides.push(DocumentLoader::create_blank_board_slide());
        }
        let mut rag_index = SlideRagIndex::new();
        rag_index.rebuild_from_slides(&slides);

        let mut subtitles_engine = LiveSubtitlesEngine::new();
        subtitles_engine.enabled = config.live_subtitles_enabled;

        let welcome_msg = ChatMessage {
            role: "model".to_string(),
            text: "Xin chào! Tôi là Trợ lý Sư phạm Đa phương thức (Google Gemini + Slide RAG).\n• Chọn Chế độ Phân tích Bảng & Bài giảng bên trên để gửi ảnh bảng độ tương phản cao + bản ghi lời giảng lên Gemini.\n• Nhấn '[Phân Tích Vùng Chọn]' ở thanh công cụ đáy để chấm bài viết tay sang LaTeX.".to_string(),
            subject: SubjectAssistantMode::Math,
            has_image_attachment: false,
            image_dimensions: None,
            rag_slide_badge: Some("RAG: Slide #1".to_string()),
            timestamp: chrono::Local::now().format("%H:%M:%S").to_string(),
        };

        let cold_start_ms = launch_instant.elapsed().as_secs_f32() * 1000.0;
        let zero_live_emb = [0.0_f32; EMBEDDING_DIM];
        let mut gesture_fsm = GestureDebounceFsm::production();
        gesture_fsm.set_hold_frames(config.gesture_hold_frames);

        Self {
            screen: initial_screen,
            cold_start_ms,
            app_epoch: launch_instant,
            storage,
            config,
            lecture_history,
            model_downloader,
            camera,
            active_camera_index,
            camera_color_image: ColorImage::new([640, 360], BG_DARK),
            camera_texture: None,
            camera_raw_color_image: ColorImage::new([640, 360], BG_DARK),
            camera_raw_texture: None,
            calibration_modal_open: false,
            camera_fps: 0.0,
            camera_sharpness: 0.0,
            camera_lighting_mode: LightingAdaptationMode::Normal,
            camera_mean_luma: 0.0,
            camera_overexposed_ratio: 0.0,
            camera_quality_gate_passed: false,
            camera_acquired: false,
            camera_using_hardware: false,
            camera_error_message: None,
            camera_device_name: "Camera đang ở chế độ chờ (Standby)".to_string(),
            camera_resolution_label: "Standby".to_string(),
            last_two_hand_frame: TwoHandFrame::default(),
            last_face_eyes_norm: None,
            last_face_5pt_norm: None,
            hud_right_trail: VecDeque::with_capacity(18),
            about_modal_open: false,
            audio_recorder,
            recording_epoch: launch_instant,
            replay_active: false,
            replay_started_at: launch_instant,
            replay_cursor_ms: 0,
            replay_duration_ms: 0,
            subtitles_engine,
            stt_event_tx,
            stt_event_rx,
            face_auth,
            auth_tab_is_signup,
            signup_step: 1,
            signin_face_mode: true,
            registered_users,
            selected_user_idx: 0,
            signup_full_name: String::new(),
            signup_username: String::new(),
            signup_pin: String::new(),
            signup_pin_confirm: String::new(),
            signup_face_samples: Vec::new(),
            signup_status_msg: String::new(),
            logged_in_user_id,
            logged_in_teacher_name,
            logged_in_username,
            slide_card_visible: true,
            enrolled_face_embedding,
            enrolled_face_templates,
            last_live_face_embedding: zero_live_emb,
            last_live_face_vec: Vec::new(),
            last_face_bbox_norm: None,
            last_face_similarity: 0.0,
            auto_face_unlock_enabled: true,
            pin_input_buffer: String::new(),
            auth_error_msg: String::new(),
            gesture_fsm,
            drawing_fsm: DrawingController4Tier::new(),
            hud_mode: DrawingHudMode::Hover,
            pen_lock_remaining_ms: 0.0,
            last_undo_ms: -1e9,
            gesture_state: DualHandGestureState {
                left_gesture: LeftGesture::Idle,
                right_gesture: RightGesture::Idle,
                left_cursor_norm: None,
                right_cursor_norm: None,
                left_pinch_span: 0.0,
                right_event: RightTransitionEvent::None,
                no_hand_idle: true,
                in_grace_period: false,
                right_hold_progress: 1.0,
            },
            prev_left_hand_norm: None,
            prev_left_pinch_span: None,
            left_anchor_origin: Vec2::ZERO,
            slides,
            rag_index,
            current_slide_idx: 0,
            current_slide_texture: None,
            slide_texture_dirty: true,
            canvas_pan: Vec2::ZERO,
            canvas_zoom: 1.0,
            last_canvas_rect: None,
            floating_toolbar_collapsed: false,
            left_camera_dock_open: true,
            active_tool: ActiveBoardTool::Pen,
            current_ink_color: [0x00, 0xf5, 0xd4, 255],
            current_stroke_thickness: 4.0,
            stroke_engine,
            strokes: Vec::with_capacity(256),
            redo_strokes: Vec::with_capacity(128),
            redo_replace_flags: Vec::with_capacity(128),
            annotations: Vec::with_capacity(64),
            active_stroke_raw: Vec::with_capacity(1024),
            finalize_scratch_points: Vec::with_capacity(1024),
            render_scratch_pos2: Vec::with_capacity(2048),
            stroke_latency_ms: 0.2,
            last_snapped_shape_banner: String::new(),
            last_snapped_shape_index: None,
            last_snapped_toast_anchor: None,
            last_snapped_toast_until_ms: -1e9,
            prev_board_hand_present: false,
            laser_pointer_board: None,
            selection_anchor_board: None,
            selection_rect_board: None,
            lasso_points: Vec::with_capacity(128),
            active_lasso_region: None,
            pending_lasso_action: None,
            pending_low_confidence_confirmation: None,
            ai_action_undo_stack: Vec::with_capacity(16),
            pending_ai_strokes: Vec::new(),
            vision_req_tx,
            vision_resp_rx,
            perf_pipeline: PerformancePipeline::new(),
            gemini_client,
            discovered_local_servers: Vec::new(),
            active_local_provider: String::new(),
            local_discovery_tx,
            local_discovery_rx,
            api_connection_status: ApiConnectionStatus::Unconfigured,
            api_status_detail_vi: String::new(),
            api_test_in_flight: false,
            api_health_tx,
            api_health_rx,
            subject_mode: SubjectAssistantMode::Math,
            chat_drawer_open: true,
            chat_messages: vec![welcome_msg],
            chat_input_draft: String::new(),
            chat_streaming_active: false,
            chat_event_tx,
            chat_event_rx,
            api_key_edit_buffer: api_key_plain,
            show_api_key_plain: false,
            new_pin_edit_buffer: String::new(),
            settings_status_banner: String::new(),
            vision_debug_hud_enabled: false,
        }
    }

    /// Configure the educational design token set AND load native UTF-8 Vietnamese fonts (`Segoe UI`, `Arial`, `Tahoma`)
    /// to guarantee Zero Tofu (`□`) across all Vietnamese diacritics and mathematical symbols.
    fn configure_modern_theme(ctx: &egui::Context) {
        let mut fonts = egui::FontDefinitions::default();
        let font_candidates = [
            ("segoe_ui", r"C:\Windows\Fonts\segoeui.ttf"),
            ("arial_vn", r"C:\Windows\Fonts\arial.ttf"),
            ("tahoma_vn", r"C:\Windows\Fonts\tahoma.ttf"),
            ("segoe_sym", r"C:\Windows\Fonts\seguisym.ttf"),
        ];
        for (name, path) in font_candidates {
            if let Ok(bytes) = std::fs::read(path) {
                fonts
                    .font_data
                    .insert(name.to_string(), egui::FontData::from_owned(bytes).into());
                if let Some(prop) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                    if name == "segoe_sym" {
                        prop.push(name.to_string());
                    } else {
                        prop.insert(0, name.to_string());
                    }
                }
                if let Some(mono) = fonts.families.get_mut(&egui::FontFamily::Monospace) {
                    if name == "segoe_sym" {
                        mono.push(name.to_string());
                    } else {
                        mono.insert(0, name.to_string());
                    }
                }
            }
        }
        ctx.set_fonts(fonts);

        let mut style = (*ctx.style()).clone();
        style.spacing.item_spacing = Vec2::new(8.0, 8.0);
        style.spacing.button_padding = Vec2::new(11.0, 6.5);
        style.spacing.window_margin = egui::Margin::same(14.0);

        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = BG_DARK;
        visuals.window_fill = SURFACE_DARK;
        visuals.extreme_bg_color = BG_DARK;
        visuals.faint_bg_color = SURFACE_DARK;
        visuals.override_text_color = Some(TEXT_LIGHT);

        let r8 = Rounding::same(8.0);
        visuals.window_rounding = r8;
        visuals.menu_rounding = r8;
        visuals.window_stroke = Stroke::new(1.0_f32, BORDER_SLATE);

        visuals.widgets.noninteractive.bg_fill = SURFACE_DARK;
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER_SLATE);
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT_LIGHT);
        visuals.widgets.noninteractive.rounding = r8;

        visuals.widgets.inactive.bg_fill = SURFACE_DARK;
        visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER_SLATE);
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT_LIGHT);
        visuals.widgets.inactive.rounding = r8;

        visuals.widgets.hovered.bg_fill = Color32::from_rgb(0x28, 0x35, 0x4b);
        visuals.widgets.hovered.bg_stroke = Stroke::new(1.2_f32, PRIMARY_INDIGO);
        visuals.widgets.hovered.fg_stroke = Stroke::new(1.2_f32, TEXT_LIGHT);
        visuals.widgets.hovered.rounding = r8;

        visuals.widgets.active.bg_fill = PRIMARY_INDIGO;
        visuals.widgets.active.bg_stroke = Stroke::new(1.2_f32, ALERT_ROSE);
        visuals.widgets.active.fg_stroke = Stroke::new(1.5_f32, TEXT_LIGHT);
        visuals.widgets.active.rounding = r8;

        visuals.selection.bg_fill = PRIMARY_INDIGO;
        visuals.selection.stroke = Stroke::new(1.0_f32, TEXT_LIGHT);

        style.visuals = visuals;
        ctx.set_style(style);
    }

    pub fn current_recording_timestamp_ms(&self) -> u64 {
        if self.audio_recorder.state() != RecordingState::Stopped {
            (self.audio_recorder.recorded_duration_secs() * 1000.0) as u64
        } else {
            self.recording_epoch.elapsed().as_millis() as u64
        }
    }

    /// Start synchronized audio + stroke-timeline recording ("Ghi Bài Giảng" — `Workstream 2.2`).
    pub fn start_synchronized_lecture_recording(&mut self) {
        self.recording_epoch = Instant::now();
        self.replay_active = false;
        self.subtitles_engine.clear();
        self.audio_recorder.start();
        let slide_title = self
            .slides
            .get(self.current_slide_idx)
            .map(|s| s.title.clone())
            .unwrap_or_else(|| self.config.lecture_title.clone());
        self.subtitles_engine.push_cue(
            0,
            3500,
            format!("🔴 Bắt đầu ghi bài giảng đồng bộ (Audio 44.1kHz + Vector Stroke Timeline): {slide_title}"),
        );
    }

    /// Toggle stroke-synced lecture replay (`Workstream 2.2`).
    pub fn toggle_lecture_replay(&mut self) {
        if self.replay_active {
            self.replay_active = false;
            return;
        }
        let max_ts = self
            .strokes
            .iter()
            .map(|s| s.timestamp_ms)
            .max()
            .unwrap_or(0);
        self.replay_duration_ms = max_ts.max(3000);
        self.replay_started_at = Instant::now();
        self.replay_cursor_ms = 0;
        self.replay_active = true;
    }

    fn poll_lecture_replay_and_subtitles(&mut self) {
        if self.replay_active {
            let elapsed = self.replay_started_at.elapsed().as_millis() as u64;
            self.replay_cursor_ms = elapsed;
            if elapsed > self.replay_duration_ms + 800 {
                self.replay_active = false;
            }
        }

        while let Ok(stt_res) = self.stt_event_rx.try_recv() {
            self.subtitles_engine.stt_in_flight = false;
            match stt_res {
                Ok(seg) => {
                    if !seg.text.trim().is_empty() {
                        self.subtitles_engine.push_cue(
                            seg.timestamp_ms,
                            4500,
                            format!("🎙 {}", seg.text.trim()),
                        );
                    }
                }
                Err(err) => {
                    self.subtitles_engine.current_banner_text = format!("🎙 STT: {err}");
                }
            }
        }

        let is_rec = self.audio_recorder.state() == RecordingState::Recording;
        if is_rec {
            let elapsed_ms = self.current_recording_timestamp_ms();
            let rms = self.audio_recorder.rms_level();
            if self
                .subtitles_engine
                .should_trigger_live_stt(is_rec, elapsed_ms, rms)
            {
                self.trigger_live_stt_now();
            }
        }
    }

    /// Capture the most recent 6 seconds of real microphone audio as WAV and dispatch to Gemini STT.
    pub fn trigger_live_stt_now(&mut self) {
        if let Ok(wav_bytes) = self.audio_recorder.encode_recent_wav_bytes(6.0) {
            let elapsed_ms = self.current_recording_timestamp_ms();
            let key = if !self.api_key_edit_buffer.trim().is_empty() {
                self.api_key_edit_buffer.clone()
            } else {
                self.storage.get_decrypted_api_key(&self.config)
            };
            self.subtitles_engine.stt_in_flight = true;
            self.gemini_client.transcribe_audio_wav(
                key,
                wav_bytes,
                elapsed_ms,
                self.stt_event_tx.clone(),
            );
        } else {
            self.subtitles_engine.stt_in_flight = false;
        }
    }

    pub fn sync_stroke_engine_config(&mut self) {
        self.stroke_engine.filter.set_params(
            self.config.one_euro_min_cutoff,
            self.config.one_euro_beta,
            self.config.one_euro_d_cutoff,
        );
        self.stroke_engine.rdp_epsilon = self.config.rdp_epsilon;
        self.stroke_engine.auto_shape_snap = self.config.auto_shape_snap;
    }

    pub fn invalidate_slide_texture(&mut self) {
        self.slide_texture_dirty = true;
    }

    pub fn ensure_slide_texture_loaded(&mut self, ctx: &egui::Context) {
        if !self.slide_texture_dirty && self.current_slide_texture.is_some() {
            return;
        }
        if let Some(slide) = self.slides.get(self.current_slide_idx) {
            let color_img = ColorImage::from_rgba_unmultiplied(
                [slide.width as usize, slide.height as usize],
                &slide.rgba_bytes,
            );
            self.current_slide_texture = Some(ctx.load_texture(
                format!("slide_tex_{}", self.current_slide_idx),
                color_img,
                egui::TextureOptions::LINEAR,
            ));
            self.slide_texture_dirty = false;
        }
    }

    /// Update `self.camera_color_image` and `self.camera_texture` in-place without buffer reallocation (`Workstream 2.1`).
    fn poll_camera_and_gestures(&mut self, ctx: &egui::Context) {
        let mut latest_pkt = None;
        while let Ok(pkt) = self.camera.frame_rx.try_recv() {
            latest_pkt = Some(pkt);
        }

        let Some(pkt) = latest_pkt else {
            return;
        };

        self.camera_fps = pkt.camera_fps;
        self.perf_pipeline.telemetry.cv_fps = pkt.camera_fps;
        self.camera_sharpness = pkt.sharpness_score;
        self.camera_lighting_mode = pkt.lighting_mode;
        self.camera_mean_luma = pkt.mean_luma;
        self.camera_overexposed_ratio = pkt.overexposed_ratio;
        self.camera_quality_gate_passed = pkt.quality_gate_passed;
        self.camera_using_hardware = pkt.using_hardware_camera && pkt.camera_active;
        self.camera_error_message = pkt.camera_error_message;
        self.camera_device_name = pkt.camera_device_name;
        self.camera_resolution_label = pkt.capture_resolution_label;
        self.last_two_hand_frame = pkt.two_hand_frame;
        self.last_face_bbox_norm = pkt.face_bbox_norm;
        self.last_face_eyes_norm = pkt.face_eyes_norm;
        self.last_face_5pt_norm = pkt.two_hand_frame.detected_face_5pt;

        if let Some(ref rh) = self.last_two_hand_frame.right_hand {
            let tip = rh.landmarks[8];
            self.hud_right_trail.push_back([tip.x, tip.y]);
            while self.hud_right_trail.len() > 18 {
                self.hud_right_trail.pop_front();
            }
        } else {
            self.hud_right_trail.clear();
        }

        if !pkt.camera_active {
            self.camera_texture = None;
            self.camera_raw_texture = None;
            return;
        }

        let w = pkt.width as usize;
        let h = pkt.height as usize;
        let num_px = w * h;

        if self.camera_color_image.size != [w, h]
            || self.camera_color_image.pixels.len() != num_px
        {
            self.camera_color_image = ColorImage::new([w, h], BG_DARK);
        }

        if pkt.rgba_bytes.len() >= num_px * 4 {
            for (dst, src) in self
                .camera_color_image
                .pixels
                .iter_mut()
                .zip(pkt.rgba_bytes.chunks_exact(4))
            {
                *dst = Color32::from_rgba_unmultiplied(src[0], src[1], src[2], src[3]);
            }
        }

        match self.camera_texture {
            Some(ref mut tex) => {
                tex.set(self.camera_color_image.clone(), egui::TextureOptions::LINEAR);
            }
            None => {
                self.camera_texture = Some(ctx.load_texture(
                    "airteach_camera_preview",
                    self.camera_color_image.clone(),
                    egui::TextureOptions::LINEAR,
                ));
            }
        }

        if self.calibration_modal_open {
            if let Some(ref raw_bytes) = pkt.raw_rgba_bytes {
                if self.camera_raw_color_image.size != [w, h]
                    || self.camera_raw_color_image.pixels.len() != num_px
                {
                    self.camera_raw_color_image = ColorImage::new([w, h], BG_DARK);
                }
                if raw_bytes.len() >= num_px * 4 {
                    for (dst, src) in self
                        .camera_raw_color_image
                        .pixels
                        .iter_mut()
                        .zip(raw_bytes.chunks_exact(4))
                    {
                        *dst = Color32::from_rgba_unmultiplied(src[0], src[1], src[2], src[3]);
                    }
                }
                match self.camera_raw_texture {
                    Some(ref mut tex) => {
                        tex.set(
                            self.camera_raw_color_image.clone(),
                            egui::TextureOptions::LINEAR,
                        );
                    }
                    None => {
                        self.camera_raw_texture = Some(ctx.load_texture(
                            "airteach_camera_raw_preview",
                            self.camera_raw_color_image.clone(),
                            egui::TextureOptions::LINEAR,
                        ));
                    }
                }
            }
        }

        if (self.screen == AppScreen::Auth || self.calibration_modal_open)
            && self.camera_using_hardware
        {
            self.face_auth.update_liveness_with_ear(
                pkt.face_blink_detected,
                pkt.face_motion_score,
                pkt.face_ear_value,
            );
            if let Some(bbox) = pkt.face_bbox_norm {
                if pkt.quality_gate_passed {
                    // Prefer 512-D 5-Point Affine-Aligned ArcFace embedding (`cv_engine`) or 128-D SFace embedding
                    let enrolled_is_128d = self
                        .enrolled_face_templates
                        .first()
                        .map(|t| t.len() == SFACE_EMBEDDING_DIM)
                        .unwrap_or(false);
                    if enrolled_is_128d {
                        if let Some(sface) = pkt.two_hand_frame.detected_face_emb {
                            self.last_live_face_vec = sface.0.to_vec();
                        } else if let Some(arc512) = pkt.two_hand_frame.detected_face_emb_512 {
                            self.last_live_face_embedding = arc512.0;
                            self.last_live_face_vec = arc512.0.to_vec();
                        }
                    } else if let Some(arc512) = pkt.two_hand_frame.detected_face_emb_512 {
                        self.last_live_face_embedding = arc512.0;
                        self.last_live_face_vec = arc512.0.to_vec();
                    } else if let Some(sface) = pkt.two_hand_frame.detected_face_emb {
                        self.last_live_face_vec = sface.0.to_vec();
                    } else if !pkt.two_hand_frame.using_onnx_session {
                        self.last_live_face_embedding =
                            self.face_auth.extract_512d_aligned_embedding(
                                &pkt.rgba_bytes,
                                pkt.width,
                                pkt.height,
                                bbox,
                                pkt.face_eyes_norm,
                            );
                        self.last_live_face_vec = self.last_live_face_embedding.to_vec();
                    }

                    if self.screen == AppScreen::Auth
                        && !self.auth_tab_is_signup
                        && self.signin_face_mode
                        && !self.registered_users.is_empty()
                        && !self.last_live_face_vec.is_empty()
                    {
                        let active_thresh = if pkt.two_hand_frame.using_onnx_session
                            || pkt.two_hand_frame.detected_face_emb_512.is_some()
                            || pkt.two_hand_frame.detected_face_emb.is_some()
                            || self.last_live_face_vec.len() == SFACE_EMBEDDING_DIM
                            || enrolled_is_128d
                        {
                            self.config.sface_match_threshold
                        } else {
                            self.config.face_match_threshold
                        };
                        let eval = self.face_auth.evaluate_unlock_multi_template(
                            &self.last_live_face_vec,
                            &self.enrolled_face_templates,
                            &self.enrolled_face_embedding,
                            pkt.face_blink_detected,
                            pkt.face_motion_score,
                            active_thresh,
                        );
                        self.last_face_similarity = eval.cosine_similarity;
                        if self.auto_face_unlock_enabled && eval.unlocked {
                            if let Some(user) =
                                self.registered_users.get(self.selected_user_idx).cloned()
                            {
                                self.complete_login_session(user);
                                return;
                            }
                        }
                    }
                }
            }
        }

        if self.screen == AppScreen::LectureMode {
            let esc_pressed = ctx.input(|i| i.key_pressed(egui::Key::Escape));
            let ctrl_z_pressed =
                ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z));
            let ctrl_shift_d_pressed = ctx.input(|i| {
                i.modifiers.command && i.modifiers.shift && i.key_pressed(egui::Key::D)
            });
            if ctrl_shift_d_pressed {
                self.vision_debug_hud_enabled = !self.vision_debug_hud_enabled;
            }
            if !ctx.wants_keyboard_input() && ctx.input(|i| i.key_pressed(egui::Key::L)) {
                self.active_tool = ActiveBoardTool::Lasso;
            }
            let mouse_button_down = ctx.input(|i| i.pointer.any_down());

            if self.camera_using_hardware {
                let state = self.gesture_fsm.update_at(
                    self.last_two_hand_frame.left_hand.as_ref(),
                    self.last_two_hand_frame.right_hand.as_ref(),
                    self.app_epoch.elapsed().as_secs_f64(),
                );
                self.gesture_state = state;
                if !mouse_button_down {
                    self.apply_two_hand_gestures_to_board(state, esc_pressed, ctrl_z_pressed);
                } else if esc_pressed {
                    self.cancel_active_stroke("ESC");
                }
            } else {
                if esc_pressed {
                    self.cancel_active_stroke("ESC");
                }
                if ctrl_z_pressed {
                    self.perform_undo_with_cooldown();
                }
            }
        }
    }

    pub fn cancel_active_stroke(&mut self, reason: &str) {
        if !self.active_stroke_raw.is_empty() {
            self.active_stroke_raw.clear();
            self.last_snapped_shape_banner = format!("Đã hủy nét ({reason}) • Khóa ngòi 300ms");
        }
        self.selection_anchor_board = None;
        self.selection_rect_board = None;
    }

    pub fn perform_undo_with_cooldown(&mut self) -> bool {
        let now_ms = self.app_epoch.elapsed().as_secs_f64() * 1000.0;
        if (now_ms - self.last_undo_ms) < UNDO_COOLDOWN_MS {
            return false;
        }
        self.last_undo_ms = now_ms;
        if let Some(top) = self.strokes.last_mut() {
            if top.snapped_shape.is_geometric() && top.raw_points.len() >= 2 {
                // Step 1 Ctrl+Z: revert geometric snap back to raw hand-drawn stroke
                self.redo_strokes.push(top.clone());
                self.redo_replace_flags.push(true);

                let mut smooth = Vec::with_capacity(top.raw_points.len() * 4);
                let saved_snap = self.stroke_engine.auto_shape_snap;
                self.stroke_engine.auto_shape_snap = false;
                let _ = self.stroke_engine.finalize_stroke(&top.raw_points, &mut smooth);
                self.stroke_engine.auto_shape_snap = saved_snap;
                top.display_points = if smooth.len() >= 2 {
                    smooth
                } else {
                    top.raw_points.clone()
                };
                top.snapped_shape = SnappedShape::Freehand;
                self.last_snapped_shape_index = None;
                self.last_snapped_toast_anchor = None;
                self.last_snapped_shape_banner =
                    "Đã hoàn tác nắn hình về nét gốc (Ctrl+Z lần nữa để xóa nét)".to_string();
                return true;
            }
        }
        if let Some(last) = self.strokes.pop() {
            self.redo_strokes.push(last);
            self.redo_replace_flags.push(false);
            self.last_snapped_shape_index = None;
            self.last_snapped_toast_anchor = None;
            self.last_snapped_shape_banner = "Đã hoàn tác nét vẽ gần nhất".to_string();
            return true;
        }
        false
    }

    pub fn perform_redo(&mut self) -> bool {
        let Some(restored) = self.redo_strokes.pop() else {
            return false;
        };
        let replace_top = self.redo_replace_flags.pop().unwrap_or(false);
        if replace_top && !self.strokes.is_empty() {
            let last_idx = self.strokes.len() - 1;
            self.strokes[last_idx] = restored;
            self.last_snapped_shape_banner = "Đã làm lại hình đã nắn".to_string();
        } else {
            self.strokes.push(restored);
            self.last_snapped_shape_banner = "Đã làm lại nét vẽ".to_string();
        }
        true
    }

    pub fn revert_last_snapped_shape_to_raw(&mut self) {
        if let Some(idx) = self.last_snapped_shape_index {
            if let Some(stroke) = self.strokes.get_mut(idx) {
                let mut smooth = Vec::with_capacity(stroke.raw_points.len() * 4);
                let saved_snap = self.stroke_engine.auto_shape_snap;
                self.stroke_engine.auto_shape_snap = false;
                let _ = self.stroke_engine.finalize_stroke(&stroke.raw_points, &mut smooth);
                self.stroke_engine.auto_shape_snap = saved_snap;
                if smooth.len() >= 2 {
                    stroke.display_points = smooth;
                } else {
                    stroke.display_points = stroke.raw_points.clone();
                }
                stroke.snapped_shape = SnappedShape::Freehand;
                self.last_snapped_shape_banner = "Đã hoàn tác về nét vẽ tay gốc".to_string();
            }
        }
        self.last_snapped_shape_index = None;
        self.last_snapped_toast_anchor = None;
    }

    fn apply_two_hand_gestures_to_board(
        &mut self,
        state: DualHandGestureState,
        esc_pressed: bool,
        ctrl_z_pressed: bool,
    ) {
        let Some(canvas_rect) = self.last_canvas_rect else {
            return;
        };

        // Update Left-hand pan/zoom navigation cleanly
        let left_pan_active = matches!(state.left_gesture, LeftGesture::PinchDragZoom);
        match state.left_gesture {
            LeftGesture::PalmOpenAnchor => {
                if let Some(cur) = state.left_cursor_norm {
                    self.left_anchor_origin = Vec2::new(cur.x, cur.y);
                    self.prev_left_hand_norm = Some(cur);
                    self.floating_toolbar_collapsed = false;
                }
            }
            LeftGesture::PinchDragZoom => {
                if let Some(cur) = state.left_cursor_norm {
                    if let Some(prev) = self.prev_left_hand_norm {
                        let dx = (cur.x - prev.x) * canvas_rect.width();
                        let dy = (cur.y - prev.y) * canvas_rect.height();
                        if dx.abs() < 120.0 && dy.abs() < 120.0 {
                            self.canvas_pan.x += dx;
                            self.canvas_pan.y += dy;
                        }
                    }
                    self.prev_left_hand_norm = Some(cur);

                    if let Some(prev_span) = self.prev_left_pinch_span {
                        let delta_span = state.left_pinch_span - prev_span;
                        if delta_span.abs() > 0.004 && delta_span.abs() < 0.12 {
                            self.canvas_zoom =
                                (self.canvas_zoom * (1.0 + delta_span * 1.5)).clamp(0.45, 2.6);
                        }
                    }
                    self.prev_left_pinch_span = Some(state.left_pinch_span);
                }
            }
            LeftGesture::Idle => {
                self.prev_left_hand_norm = None;
                self.prev_left_pinch_span = None;
            }
        }

        // Compute raw Right-hand landmarks, 3D pose, pinch_ratio, and One-Euro filtered board coordinates
        let right_hand_opt = self.last_two_hand_frame.right_hand.as_ref();
        let hand_present = !state.no_hand_idle && right_hand_opt.is_some();
        let pinch_ratio = right_hand_opt.map(|rh| compute_pinch_ratio(&rh.landmarks));
        let raw_pose = right_hand_opt
            .map(|rh| classify_right_hand_raw(rh).0)
            .unwrap_or(RightGesture::Idle);

        let raw_board_pt_opt = state.right_cursor_norm.map(|r_norm| {
            let margin = 0.12_f32;
            let mapped_x = ((r_norm.x - margin) / (1.0 - 2.0 * margin)).clamp(0.0, 1.0);
            let mapped_y = ((r_norm.y - margin) / (1.0 - 2.0 * margin)).clamp(0.0, 1.0);
            let screen_x = canvas_rect.min.x + mapped_x * canvas_rect.width();
            let screen_y = canvas_rect.min.y + mapped_y * canvas_rect.height();
            Point2D::new(
                (screen_x - canvas_rect.min.x - self.canvas_pan.x) / self.canvas_zoom,
                (screen_y - canvas_rect.min.y - self.canvas_pan.y) / self.canvas_zoom,
            )
        });
        let t_sec = self.app_epoch.elapsed().as_secs_f64();
        let board_pt_opt = if let Some(raw_bp) = raw_board_pt_opt {
            if !self.prev_board_hand_present
                && self.drawing_fsm.hud_mode != DrawingHudMode::Drawing
            {
                self.stroke_engine.reset_filter();
            }
            let smoothed = self.stroke_engine.filter_point(raw_bp, t_sec);
            self.laser_pointer_board = Some(smoothed);
            self.prev_board_hand_present = hand_present;
            Some(smoothed)
        } else {
            self.prev_board_hand_present = false;
            None
        };

        let t_ms = t_sec * 1000.0;
        let step_res = self.drawing_fsm.step(
            t_ms,
            hand_present,
            board_pt_opt,
            pinch_ratio,
            raw_pose,
            left_pan_active,
            esc_pressed,
            ctrl_z_pressed,
            self.config.hold_to_snap_350ms,
        );
        self.hud_mode = step_res.hud_mode;
        self.pen_lock_remaining_ms = step_res.cooldown_remaining_ms;

        match step_res.action {
            DrawingFsmAction::BeginStroke | DrawingFsmAction::AppendPoint => {
                self.active_stroke_raw = self.drawing_fsm.active_points();
                self.stroke_latency_ms = self.last_two_hand_frame.inference_ms;
                if step_res.gap_bridged {
                    self.last_snapped_shape_banner =
                        "Đã nội suy nối liền nét tự động (Gap Bridging)".to_string();
                }
            }
            DrawingFsmAction::HoldGrace => {
                self.active_stroke_raw = self.drawing_fsm.active_points();
                self.last_snapped_shape_banner =
                    "Đang giữ trạng thái vẽ (Chờ nhận lại tay <150ms)...".to_string();
            }
            DrawingFsmAction::CommitStrokeTrimmed(trimmed_pts) => {
                self.active_stroke_raw = trimmed_pts;
                let allow_snap = !self.config.hold_to_snap_350ms || self.drawing_fsm.hold_to_snap_ready;
                self.finish_pointer_gesture_internal(allow_snap);
            }
            DrawingFsmAction::CancelStrokeDiscard => {
                self.active_stroke_raw.clear();
                self.last_snapped_shape_banner = if step_res.cancel_reason.is_empty() {
                    "Đã hủy nét • Khóa ngòi 300ms".to_string()
                } else {
                    format!("Đã hủy nét ({}) • Khóa ngòi 300ms", step_res.cancel_reason)
                };
            }
            DrawingFsmAction::RollbackLastStroke => {
                self.active_stroke_raw.clear();
                let popped = self.strokes.pop();
                if popped.is_some() {
                    self.last_snapped_shape_index = None;
                    self.last_snapped_toast_anchor = None;
                    self.last_snapped_shape_banner = format!(
                        "Đã thu hồi nét ({}) • Khóa ngòi 300ms",
                        step_res.cancel_reason
                    );
                }
            }
            DrawingFsmAction::UndoWithCooldown => {
                self.perform_undo_with_cooldown();
            }
            DrawingFsmAction::PalmErase => {
                if let Some(bp) = board_pt_opt {
                    let erase_r = 45.0 / self.canvas_zoom.max(0.5);
                    self.strokes.retain(|s| {
                        !s.display_points.iter().any(|p| p.dist(bp) < erase_r)
                    });
                    self.annotations.retain(|a| a.position.dist(bp) >= erase_r);
                }
            }
            DrawingFsmAction::PanActive | DrawingFsmAction::None => {}
        }
    }

    pub fn begin_board_stroke(&mut self, raw_pt: Point2D) {
        let t0 = Instant::now();
        self.stroke_engine.reset_filter();
        let t_sec = self.app_epoch.elapsed().as_secs_f64();
        let smoothed = self.stroke_engine.filter_point(raw_pt, t_sec);
        self.active_stroke_raw.clear();
        self.active_stroke_raw.push(smoothed);
        self.stroke_latency_ms =
            self.last_two_hand_frame.inference_ms + t0.elapsed().as_secs_f32() * 1000.0;
    }

    pub fn append_board_stroke(&mut self, raw_pt: Point2D) {
        let t0 = Instant::now();
        let t_sec = self.app_epoch.elapsed().as_secs_f64();
        let smoothed = self.stroke_engine.filter_point(raw_pt, t_sec);
        if let Some(&last) = self.active_stroke_raw.last() {
            let d = last.dist(smoothed);
            if d < 0.78 {
                return;
            }
            if d > DEFAULT_INTERP_STEP_PX {
                let p_before = if self.active_stroke_raw.len() >= 2 {
                    Some(self.active_stroke_raw[self.active_stroke_raw.len() - 2])
                } else {
                    None
                };
                let pts = interpolate_bridge_points_2d(
                    p_before,
                    last,
                    smoothed,
                    DEFAULT_INTERP_STEP_PX,
                );
                self.active_stroke_raw.extend(pts);
            } else {
                self.active_stroke_raw.push(smoothed);
            }
        } else {
            self.active_stroke_raw.push(smoothed);
        }
        self.stroke_latency_ms =
            self.last_two_hand_frame.inference_ms + t0.elapsed().as_secs_f32() * 1000.0;
    }

    pub fn finish_pointer_gesture(&mut self) {
        self.finish_pointer_gesture_internal(true);
    }

    pub fn finish_pointer_gesture_internal(&mut self, allow_snap_if_enabled: bool) {
        if self.active_stroke_raw.len() < 2 {
            self.active_stroke_raw.clear();
            return;
        }

        let t0 = Instant::now();
        let timestamp_ms = self.current_recording_timestamp_ms();
        let prev_auto = self.stroke_engine.auto_shape_snap;
        if !allow_snap_if_enabled {
            self.stroke_engine.auto_shape_snap = false;
        }
        let snapped_shape = self
            .stroke_engine
            .finalize_stroke(&self.active_stroke_raw, &mut self.finalize_scratch_points);
        self.stroke_engine.auto_shape_snap = prev_auto;

        self.last_snapped_shape_banner = if snapped_shape.is_geometric() {
            format!("Đã chuẩn hóa: {}", snapped_shape.label_vi())
        } else {
            "Đã làm mượt One-Euro + Đường cong Bézier".to_string()
        };

        self.redo_strokes.clear();
        self.redo_replace_flags.clear();
        let new_idx = self.strokes.len();
        let anchor_pt = self
            .finalize_scratch_points
            .first()
            .copied()
            .or_else(|| self.active_stroke_raw.first().copied());
        self.strokes.push(PersistedStroke {
            timestamp_ms,
            raw_points: self.active_stroke_raw.clone(),
            display_points: self.finalize_scratch_points.clone(),
            color_rgba: self.current_ink_color,
            thickness: self.current_stroke_thickness,
            snapped_shape: snapped_shape.clone(),
        });
        if snapped_shape.is_geometric() {
            self.last_snapped_shape_index = Some(new_idx);
            self.last_snapped_toast_anchor = anchor_pt;
            self.last_snapped_toast_until_ms =
                self.app_epoch.elapsed().as_secs_f64() * 1000.0 + 7000.0;
        } else {
            self.last_snapped_shape_index = None;
            self.last_snapped_toast_anchor = None;
        }
        self.active_stroke_raw.clear();
        self.stroke_latency_ms =
            self.last_two_hand_frame.inference_ms + t0.elapsed().as_secs_f32() * 1000.0;
    }

    /// Query PubChem / PDB and generate a 2D/3D structural formula directly onto the board (`Workstream 3.3`).
    pub fn spawn_pubchem_structure_on_board(&mut self, molecule_name: &str) {
        let ts = self.current_recording_timestamp_ms();
        let center = Point2D::new(560.0, 285.0);
        let generated = StemKnowledgeEngine::resolve_and_generate_structure(
            molecule_name,
            center,
            ts,
            Some(&self.gemini_client.runtime),
        );
        self.strokes.extend(generated.strokes);
        self.annotations.extend(generated.annotations);
        self.subtitles_engine.push_cue(
            ts,
            4000,
            format!(
                "🧪 [PubChem CID {}]: Đã dựng cấu trúc phân tử {} ({}) — IUPAC: {}",
                generated.info.cid,
                generated.info.common_name,
                generated.info.molecular_formula,
                generated.info.iupac_name
            ),
        );
    }

    /// Toggle Light / Dark Theme across all views and persist in local SQLite config.
    pub fn toggle_theme_mode(&mut self, ctx: &egui::Context) {
        self.config.theme_mode = self.config.theme_mode.toggle();
        self.config.theme_mode.palette().apply_to_egui(ctx);
        self.storage.save_config(&self.config);
    }

    /// Navigate from Welcome screen to Auth Gate with explicit camera lifecycle management.
    pub fn enter_auth_screen(&mut self, start_with_signup: bool) {
        self.auth_error_msg.clear();
        self.signup_status_msg.clear();
        self.pin_input_buffer.clear();
        self.reload_registered_users();

        if start_with_signup || self.registered_users.is_empty() {
            self.auth_tab_is_signup = true;
            self.signup_step = 1;
            self.signup_face_samples.clear();
            self.release_camera_hardware();
        } else {
            self.auth_tab_is_signup = false;
            let idx = self
                .selected_user_idx
                .min(self.registered_users.len().saturating_sub(1));
            self.select_registered_user_for_signin(idx);
            self.signin_face_mode =
                !self.enrolled_face_templates.is_empty() || !self.enrolled_face_embedding.is_empty();
            if self.signin_face_mode {
                self.ensure_camera_acquired();
            } else {
                self.release_camera_hardware();
            }
        }
        self.screen = AppScreen::Auth;
    }

    pub fn reload_registered_users(&mut self) {
        self.registered_users = self.storage.list_users().unwrap_or_default();
        if self.selected_user_idx >= self.registered_users.len() {
            self.selected_user_idx = 0;
        }
        if let Some(u) = self.registered_users.get(self.selected_user_idx) {
            self.enrolled_face_embedding = u.face_embedding.clone();
            self.enrolled_face_templates = u.face_templates.clone();
        } else {
            self.enrolled_face_embedding.clear();
            self.enrolled_face_templates.clear();
        }
    }

    pub fn set_calibration_modal_open(&mut self, open: bool) {
        self.calibration_modal_open = open;
        let _ = self
            .camera
            .cmd_tx
            .try_send(CameraControlCmd::SetCalibrationPreview(open));
        if open {
            self.ensure_camera_acquired();
        }
    }

    /// Explicitly acquire hardware camera stream when UI requires webcam input.
    pub fn ensure_camera_acquired(&mut self) {
        if !self.camera_acquired {
            self.camera_error_message = None;
            self.camera.acquire();
            self.camera_acquired = true;
        }
    }

    /// Explicitly stop & release hardware camera stream on logout, toggle-off, or non-camera views.
    pub fn release_camera_hardware(&mut self) {
        if self.calibration_modal_open {
            return;
        }
        if self.camera_acquired || self.camera_using_hardware {
            self.camera.release();
            self.camera_acquired = false;
            self.camera_using_hardware = false;
            self.camera_fps = 0.0;
            self.camera_sharpness = 0.0;
            self.camera_lighting_mode = LightingAdaptationMode::Normal;
            self.camera_mean_luma = 0.0;
            self.camera_overexposed_ratio = 0.0;
            self.camera_quality_gate_passed = false;
            self.camera_resolution_label = "Standby (Camera Off)".to_string();
            self.camera_texture = None;
            self.camera_raw_texture = None;
            self.last_two_hand_frame = TwoHandFrame::default();
            self.last_face_bbox_norm = None;
            self.last_face_eyes_norm = None;
            self.last_face_5pt_norm = None;
            self.last_live_face_vec.clear();
            self.hud_right_trail.clear();
        }
    }

    /// Hide tutorial slides (`is_tutorial = true`) or permanently delete them from Clean Slate config & DB.
    pub fn dismiss_tutorial_slides(&mut self, permanent_delete: bool) {
        self.config.tutorial_dismissed = true;
        if permanent_delete {
            self.config.tutorial_deleted = true;
            let _ = self.storage.delete_tutorial_records();
        }
        self.storage.save_config(&self.config);
        self.slides.retain(|s| !s.is_tutorial);
        if self.slides.is_empty() {
            self.slides.push(DocumentLoader::create_blank_board_slide());
        }
        self.current_slide_idx = self
            .current_slide_idx
            .min(self.slides.len().saturating_sub(1));
        self.rag_index.rebuild_from_slides(&self.slides);
        self.invalidate_slide_texture();
    }

    /// Step 1 validation before entering Step 2 (face angle registration).
    pub fn validate_signup_step1(&mut self) -> Result<(), String> {
        self.auth_error_msg.clear();
        let username = self.signup_username.trim();
        let username_taken = self.storage.username_exists(username).unwrap_or(false);
        if let Err(err) = FaceAuthEngine::validate_signup_fields(
            username,
            &self.signup_full_name,
            &self.signup_pin,
            &self.signup_pin_confirm,
            username_taken,
        ) {
            self.auth_error_msg = err.clone();
            return Err(err);
        }
        Ok(())
    }

    /// Sign-Up Step 2: capture 1 live face angle vector per click (3 to 5 guided angles).
    pub fn capture_signup_face_sample(&mut self) {
        self.auth_error_msg.clear();
        if !self.camera_using_hardware {
            self.auth_error_msg =
                "Chưa có tín hiệu Webcam! Vui lòng cấp quyền hoặc bấm 'Thử kết nối lại Camera'."
                    .to_string();
            return;
        }
        if self.last_face_bbox_norm.is_none() {
            self.auth_error_msg =
                "Chưa nhận diện thấy khuôn mặt trong khung! Vui lòng nhìn thẳng vào camera."
                    .to_string();
            return;
        }
        if !self.camera_quality_gate_passed {
            self.auth_error_msg =
                "Khung hình chưa đạt độ nét (Laplacian dưới ngưỡng Quality Gate). Hãy giữ camera ổn định 0.5 giây rồi chụp lại."
                    .to_string();
            return;
        }
        if self.signup_face_samples.len() >= MAX_FACE_ENROLL_SAMPLES {
            self.signup_status_msg = format!(
                "Đã thu đủ {}/{} góc khuôn mặt! Bạn có thể bấm [Hoàn tất Đăng ký].",
                MAX_FACE_ENROLL_SAMPLES, MAX_FACE_ENROLL_SAMPLES
            );
            return;
        }
        if self.last_live_face_vec.is_empty()
            || !self.last_live_face_vec.iter().any(|&v| v.abs() > 1e-6)
        {
            self.auth_error_msg =
                "Đang chờ khung hình Face embedding ổn định, vui lòng thử bấm chụp lại sau 0.5 giây."
                    .to_string();
            return;
        }

        self.signup_face_samples.push(self.last_live_face_vec.clone());
        let count = self.signup_face_samples.len();
        let next_prompt = ENROLL_ANGLE_PROMPTS
            .get(count)
            .copied()
            .unwrap_or("Đã thu thập đủ 5 góc! Sẵn sàng lưu vector trung bình.");
        if count >= MIN_FACE_ENROLL_SAMPLES {
            self.signup_status_msg = format!(
                "Đã chụp {count}/{MAX_FACE_ENROLL_SAMPLES} góc khuôn mặt (Đủ điều kiện hoàn tất). {next_prompt}"
            );
        } else {
            self.signup_status_msg = format!(
                "Đã chụp {count}/{MAX_FACE_ENROLL_SAMPLES} góc (tối thiểu {MIN_FACE_ENROLL_SAMPLES}). {next_prompt}"
            );
        }
    }

    /// Finalize Sign-Up into SQLite `users` (Argon2id password hash + L2 averaged face vector ONLY).
    pub fn finalize_signup_account(&mut self, skip_face: bool) {
        if self.validate_signup_step1().is_err() {
            return;
        }
        if !skip_face && self.signup_face_samples.len() < MIN_FACE_ENROLL_SAMPLES {
            self.auth_error_msg = format!(
                "Cần chụp tối thiểu {MIN_FACE_ENROLL_SAMPLES} góc khuôn mặt hoặc chọn 'Bỏ qua Face ID (Dùng mật khẩu)'."
            );
            return;
        }

        let pw_hash = match FaceAuthEngine::hash_password_argon2(self.signup_pin.trim()) {
            Ok(h) => h,
            Err(e) => {
                self.auth_error_msg = e;
                return;
            }
        };

        let face_templates = if skip_face {
            Vec::new()
        } else {
            self.signup_face_samples.clone()
        };
        let has_face = !face_templates.is_empty();

        let clean_uname = self.signup_username.trim().to_string();
        let clean_dname = self.signup_full_name.trim().to_string();

        match self.storage.create_user_with_templates(
            &clean_uname,
            &clean_dname,
            &pw_hash,
            &face_templates,
        ) {
            Ok(created) => {
                self.reload_registered_users();
                let new_idx = self
                    .registered_users
                    .iter()
                    .position(|u| u.id == created.id)
                    .unwrap_or(0);
                self.select_registered_user_for_signin(new_idx);
                self.signup_pin.clear();
                self.signup_pin_confirm.clear();
                self.pin_input_buffer.clear();
                self.signup_face_samples.clear();
                self.signup_step = 1;
                self.auth_tab_is_signup = false;
                self.signin_face_mode = has_face;
                if self.signin_face_mode {
                    self.ensure_camera_acquired();
                } else {
                    self.release_camera_hardware();
                }
                self.signup_status_msg = format!(
                    "Đăng ký thành công tài khoản '@{}'! Vui lòng đăng nhập để bắt đầu.",
                    created.username
                );
            }
            Err(err) => {
                self.auth_error_msg = err;
            }
        }
    }

    pub fn select_registered_user_for_signin(&mut self, idx: usize) {
        if let Some(u) = self.registered_users.get(idx) {
            self.selected_user_idx = idx;
            self.enrolled_face_embedding = u.face_embedding.clone();
            self.enrolled_face_templates = u.face_templates.clone();
            self.face_auth.reset_liveness();
            self.last_face_similarity = 0.0;
            self.auth_error_msg.clear();
        }
    }

    pub fn complete_login_session(&mut self, user: UserAccount) {
        let _ = self.storage.create_session(&user.id);
        self.logged_in_user_id = Some(user.id.clone());
        self.logged_in_teacher_name = user.display_name.clone();
        self.logged_in_username = user.username.clone();
        self.enrolled_face_embedding = user.face_embedding.clone();
        self.enrolled_face_templates = user.face_templates.clone();
        self.lecture_history = self.storage.list_lecture_history(&user.id);
        self.pin_input_buffer.clear();
        self.auth_error_msg.clear();
        self.signup_status_msg.clear();
        self.release_camera_hardware();
        self.screen = AppScreen::Dashboard;
    }

    pub fn logout_to_auth_gate(&mut self) {
        let _ = self.storage.clear_active_sessions();
        self.logged_in_user_id = None;
        self.logged_in_teacher_name.clear();
        self.logged_in_username.clear();
        self.pin_input_buffer.clear();
        self.auth_error_msg.clear();
        self.signup_status_msg.clear();
        self.lecture_history.clear();
        self.face_auth.reset_liveness();
        self.reload_registered_users();
        self.auth_tab_is_signup = self.registered_users.is_empty();
        self.signup_step = 1;
        self.signin_face_mode = false;
        self.set_calibration_modal_open(false);
        self.release_camera_hardware();
        self.screen = AppScreen::Auth;
    }

    pub fn try_unlock_with_password(&mut self) {
        let password = self.pin_input_buffer.trim();
        let Some(user) = self.registered_users.get(self.selected_user_idx).cloned() else {
            self.auth_error_msg =
                "Chưa có tài khoản nào trên máy. Vui lòng tạo tài khoản mới.".to_string();
            return;
        };

        if FaceAuthEngine::verify_password_argon2(password, &user.password_hash) {
            self.complete_login_session(user);
        } else {
            self.auth_error_msg = format!(
                "Mật khẩu / PIN không chính xác cho tài khoản '@{}'.",
                user.username
            );
        }
    }

    fn collect_strokes_in_selection(&self) -> (Vec<Vec<Point2D>>, Point2D, Point2D) {
        let (min_pt, max_pt) = self
            .selection_rect_board
            .unwrap_or((Point2D::new(60.0, 40.0), Point2D::new(920.0, 520.0)));

        let mut matching = Vec::new();
        for s in &self.strokes {
            let overlaps = s.display_points.iter().any(|p| {
                p.x >= min_pt.x - 15.0
                    && p.x <= max_pt.x + 15.0
                    && p.y >= min_pt.y - 15.0
                    && p.y <= max_pt.y + 15.0
            });
            if overlaps {
                matching.push(s.display_points.clone());
            }
        }
        if matching.is_empty() && !self.strokes.is_empty() {
            for s in &self.strokes {
                matching.push(s.display_points.clone());
            }
        }
        (matching, min_pt, max_pt)
    }

    pub fn trigger_ocr_on_selection(&mut self) {
        let (strokes, min_pt, max_pt) = self.collect_strokes_in_selection();
        let _ = self.vision_req_tx.try_send(VisionTaskReq::RunOcr {
            strokes,
            min_pt,
            max_pt,
            subject: self.subject_mode.short_badge().to_string(),
        });
    }

    pub fn trigger_latex_on_selection(&mut self, force_low_conf: bool) {
        let (strokes, min_pt, max_pt) = self.collect_strokes_in_selection();
        let _ = self.vision_req_tx.try_send(VisionTaskReq::RunLatex {
            strokes,
            min_pt,
            max_pt,
            subject: self.subject_mode.short_badge().to_string(),
            force_low_conf,
        });
    }

    fn poll_vision_worker_results(&mut self) {
        let mut auto_pubchem_trigger: Option<String> = None;

        while let Ok(resp) = self.vision_resp_rx.try_recv() {
            match resp {
                VisionTaskResp::Ocr(res) => {
                    let lower = res.recognized_text.to_lowercase();
                    for mol in ["aspirin", "benzene", "caffeine", "glucose"] {
                        if lower.contains(mol) {
                            auto_pubchem_trigger = Some(mol.to_string());
                        }
                    }

                    let ann = BoardTextAnnotation {
                        text: res.recognized_text.clone(),
                        is_latex: false,
                        position: Point2D::new(res.region_min.x, res.region_max.y + 12.0),
                        color_rgba: [6, 182, 212, 255],
                        confidence: res.confidence,
                    };
                    if res.requires_user_confirmation() {
                        self.pending_low_confidence_confirmation = Some(ann);
                    } else {
                        self.annotations.push(ann);
                    }
                }
                VisionTaskResp::Latex(res) => {
                    let ann = BoardTextAnnotation {
                        text: res.latex_formula.clone(),
                        is_latex: true,
                        position: Point2D::new(res.region_min.x, res.region_max.y + 12.0),
                        color_rgba: [79, 70, 229, 255],
                        confidence: res.confidence,
                    };
                    if res.requires_user_confirmation() {
                        self.pending_low_confidence_confirmation = Some(ann);
                    } else {
                        self.annotations.push(ann);
                    }
                }
            }
        }

        if let Some(mol) = auto_pubchem_trigger {
            self.spawn_pubchem_structure_on_board(&mol);
        }
    }

    /// Crop the exact pixel bounding box of the teacher's selected area on canvas
    /// (compositing slide pixels + handwritten vector ink strokes) and encode to PNG
    /// using `image::codecs::png::PngEncoder`.
    pub fn crop_selection_to_base64_png(&self) -> Option<(String, u32, u32)> {
        let (min_pt, max_pt) = match self.selection_rect_board {
            Some((a, b)) if (b.x - a.x).abs() > 12.0 && (b.y - a.y).abs() > 12.0 => (a, b),
            _ => {
                if !self.strokes.is_empty() {
                    let mut min_x = f32::MAX;
                    let mut min_y = f32::MAX;
                    let mut max_x = f32::MIN;
                    let mut max_y = f32::MIN;
                    for s in &self.strokes {
                        for p in &s.display_points {
                            min_x = min_x.min(p.x);
                            min_y = min_y.min(p.y);
                            max_x = max_x.max(p.x);
                            max_y = max_y.max(p.y);
                        }
                    }
                    (
                        Point2D::new((min_x - 28.0).max(0.0), (min_y - 28.0).max(0.0)),
                        Point2D::new(max_x + 28.0, max_y + 28.0),
                    )
                } else {
                    (Point2D::new(60.0, 30.0), Point2D::new(860.0, 480.0))
                }
            }
        };

        let w = ((max_pt.x - min_pt.x).abs() as u32).clamp(180, 1280);
        let h = ((max_pt.y - min_pt.y).abs() as u32).clamp(120, 720);
        let bg_rgb = self.config.contrast_mode.canvas_bg_rgb();
        let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> =
            ImageBuffer::from_pixel(w, h, Rgba([bg_rgb[0], bg_rgb[1], bg_rgb[2], 255]));

        if let Some(slide) = self.slides.get(self.current_slide_idx) {
            let slide_origin_x = 60.0_f32;
            let slide_origin_y = 30.0_f32;
            let slide_disp_w = 920.0_f32;
            let slide_disp_h = 518.0_f32;

            for py in 0..h {
                let board_y = min_pt.y + py as f32;
                let rel_y = (board_y - slide_origin_y) / slide_disp_h;
                if (0.0..=1.0).contains(&rel_y) {
                    let sy =
                        ((rel_y * slide.height as f32) as u32).min(slide.height.saturating_sub(1));
                    for px in 0..w {
                        let board_x = min_pt.x + px as f32;
                        let rel_x = (board_x - slide_origin_x) / slide_disp_w;
                        if (0.0..=1.0).contains(&rel_x) {
                            let sx = ((rel_x * slide.width as f32) as u32)
                                .min(slide.width.saturating_sub(1));
                            let idx = ((sy * slide.width + sx) * 4) as usize;
                            if idx + 3 < slide.rgba_bytes.len() {
                                img.put_pixel(
                                    px,
                                    py,
                                    Rgba([
                                        slide.rgba_bytes[idx],
                                        slide.rgba_bytes[idx + 1],
                                        slide.rgba_bytes[idx + 2],
                                        255,
                                    ]),
                                );
                            }
                        }
                    }
                }
            }
        }

        // MODULE 4.2: High-Contrast Ink Extraction for Gemini AI Vision
        // Render strokes at 1.5x thickness with full contrast (no Caro grid clutter)
        for stroke in &self.strokes {
            let col = Rgba([
                stroke.color_rgba[0],
                stroke.color_rgba[1],
                stroke.color_rgba[2],
                255,
            ]);
            let boosted_half_thick = ((stroke.thickness * 1.5) * 0.5).ceil() as i32;
            for pair in stroke.display_points.windows(2) {
                let x0 = (pair[0].x - min_pt.x) as i32;
                let y0 = (pair[0].y - min_pt.y) as i32;
                let x1 = (pair[1].x - min_pt.x) as i32;
                let y1 = (pair[1].y - min_pt.y) as i32;
                draw_thick_line_rgba(&mut img, x0, y0, x1, y1, boosted_half_thick.max(2), col);
            }
        }

        let (final_img, out_w, out_h) = if w.max(h) > 1024 {
            let scale = 1024.0_f32 / (w.max(h) as f32);
            let nw = ((w as f32 * scale).round() as u32).max(1);
            let nh = ((h as f32 * scale).round() as u32).max(1);
            let resized = image::imageops::resize(
                &img,
                nw,
                nh,
                image::imageops::FilterType::Triangle,
            );
            (resized, nw, nh)
        } else {
            (img, w, h)
        };

        let mut png_bytes = Vec::with_capacity((out_w * out_h) as usize);
        let encoder = PngEncoder::new(&mut png_bytes);
        if encoder
            .write_image(final_img.as_raw(), out_w, out_h, ColorType::Rgba8.into())
            .is_ok()
        {
            Some((BASE64.encode(png_bytes), out_w, out_h))
        } else {
            None
        }
    }

    /// Execute "Phân Tích Vùng" (`MODULE 4.2`):
    /// Crop canvas selection -> Encode PNG -> Open Right Gemini AI Panel -> Stream to `gemini-2.5-flash`
    /// -> Clear selection rectangle so zero fake boxes clutter the Smartboard.
    pub fn inspect_selection_with_gemini(&mut self) {
        self.chat_drawer_open = true;
        let prompt = if self.chat_input_draft.trim().is_empty() {
            "Đang phân tích nét vẽ vùng chọn: Hãy đọc chữ viết tay/công thức trong ảnh vùng chọn này và giải thích từng bước chi tiết.".to_string()
        } else {
            let p = self.chat_input_draft.trim().to_string();
            self.chat_input_draft.clear();
            p
        };
        self.send_chat_message(prompt, true);
        self.selection_rect_board = None;
        self.selection_anchor_board = None;
    }

    /// Module 4 ("Analyze Board & Lecture"):
    /// 1. Export the entire Smartboard Canvas to PNG
    /// 2. Attach the latest transcribed lecture text from the microphone
    /// 3. Send both Board PNG + Lecture Transcript to Gemini 3.5 Flash (fallback 2.5 / 2.0 Flash)
    ///    with Structured JSON Output + SQLite cache, and persist history into `airteach_local.db`.
    pub fn analyze_board_and_lecture(&mut self, mode: PedagogicalAnalysisMode) {
        self.chat_drawer_open = true;
        // Temporarily clear selection_rect_board so crop_selection_to_base64_png exports the full board
        let prev_sel = self.selection_rect_board.take();
        let prompt = mode.prompt_instruction(&self.subtitles_engine.full_transcript_text());
        self.send_chat_message_with_options(prompt, true, true);
        self.selection_rect_board = prev_sel;
        // Also persist the lecture snapshot to SQLite history
        self.save_current_lecture_session();
    }

    pub fn send_chat_message(&mut self, prompt: String, include_selection_image: bool) {
        self.send_chat_message_with_options(prompt, include_selection_image, false);
    }

    pub fn send_chat_message_with_options(
        &mut self,
        prompt: String,
        include_selection_image: bool,
        use_structured_json: bool,
    ) {
        // Check if the user asked to draw a PubChem molecule in chat
        let lower = prompt.to_lowercase();
        for mol in ["aspirin", "benzene", "caffeine", "glucose"] {
            if lower.contains(mol)
                && (lower.contains("vẽ")
                    || lower.contains("cấu trúc")
                    || lower.contains("pubchem")
                    || lower.contains("công thức"))
            {
                self.spawn_pubchem_structure_on_board(mol);
                break;
            }
        }

        let now_str = chrono::Local::now().format("%H:%M:%S").to_string();
        let crop_data = if include_selection_image {
            self.crop_selection_to_base64_png()
        } else {
            None
        };

        let img_dims = crop_data.as_ref().map(|(_, w, h)| (*w, *h));
        let b64_img = crop_data.map(|(b64, _, _)| b64);

        // Retrieve grounded Slide Context RAG
        let rag_hits = self
            .rag_index
            .retrieve(&prompt, self.current_slide_idx, 2);
        let rag_badge = rag_hits.first().map(|h| {
            format!(
                "🧠 RAG Slide #{} ({:.0}%)",
                h.slide_index + 1,
                h.similarity_score * 100.0
            )
        });
        let slide_rag_context = Some(
            self.rag_index
                .build_grounded_prompt_context(&prompt, self.current_slide_idx),
        );

        let resolved_model = GeminiChatClient::resolve_model_id(&self.config.gemini_model);
        let cache_key = GeminiChatClient::compute_cache_key(
            &resolved_model,
            self.subject_mode,
            &prompt,
            slide_rag_context.as_deref(),
            b64_img.as_deref(),
            use_structured_json,
        );

        self.chat_messages.push(ChatMessage {
            role: "user".to_string(),
            text: prompt.clone(),
            subject: self.subject_mode,
            has_image_attachment: include_selection_image,
            image_dimensions: img_dims,
            rag_slide_badge: rag_badge.clone(),
            timestamp: now_str.clone(),
        });

        if let Some((cached_model, cached_text, _structured)) =
            self.storage.get_cached_ai_response(&cache_key)
        {
            if !cached_text.trim().is_empty() {
                self.chat_messages.push(ChatMessage {
                    role: "model".to_string(),
                    text: format!("⚡ [Cache · {cached_model}]\n{cached_text}"),
                    subject: self.subject_mode,
                    has_image_attachment: false,
                    image_dimensions: None,
                    rag_slide_badge: rag_badge,
                    timestamp: now_str,
                });
                self.chat_streaming_active = false;
                return;
            }
        }

        let model_msg_idx = self.chat_messages.len();
        self.chat_messages.push(ChatMessage {
            role: "model".to_string(),
            text: String::new(),
            subject: self.subject_mode,
            has_image_attachment: false,
            image_dimensions: None,
            rag_slide_badge: rag_badge,
            timestamp: now_str,
        });

        let api_key = self.storage.get_decrypted_api_key(&self.config);
        self.chat_streaming_active = true;
        let final_prompt = if !use_structured_json {
            let (cw, ch) = self
                .last_canvas_rect
                .map(|r| (r.width().max(640.0), r.height().max(420.0)))
                .unwrap_or((1020.0, 620.0));
            let free_regs = find_free_regions_normalized(&self.occupied_board_rects(), cw, ch, 3);
            format!(
                "{prompt}\n\n{}",
                WhiteboardAgent::tool_protocol_prompt_with_context(cw, ch, &free_regs)
            )
        } else {
            prompt
        };
        let hist: Vec<(String, String)> = self
            .chat_messages
            .iter()
            .take(model_msg_idx)
            .filter(|m| m.role == "user" || m.role == "model")
            .map(|m| {
                let role = if m.role == "model" {
                    "assistant".to_string()
                } else {
                    "user".to_string()
                };
                (role, m.text.clone())
            })
            .collect();

        if use_structured_json {
            self.gemini_client.stream_completion_with_options(
                api_key,
                resolved_model,
                self.subject_mode,
                final_prompt,
                slide_rag_context,
                b64_img,
                true,
                cache_key,
                model_msg_idx,
                self.chat_event_tx.clone(),
            );
        } else {
            LocalAIManager::stream_with_hybrid_fallback(
                &self.gemini_client,
                self.config.ai_routing_mode,
                self.active_local_provider.clone(),
                self.config.local_endpoint.clone(),
                self.config.local_model.clone(),
                hist,
                final_prompt,
                b64_img,
                api_key,
                resolved_model,
                self.subject_mode,
                slide_rag_context,
                cache_key,
                model_msg_idx,
                self.chat_event_tx.clone(),
            );
        }
    }

    pub fn discover_local_engines(&self) {
        LocalAIManager::discover_async(&self.gemini_client.runtime, self.local_discovery_tx.clone());
    }

    pub fn select_local_model(&mut self, provider: String, base_url: String, model: String) {
        self.active_local_provider = provider.clone();
        self.config.local_endpoint = base_url.clone();
        self.config.local_model = model.clone();
        self.storage.save_config(&self.config);
        self.perf_pipeline.telemetry.ai_mode = self.config.ai_routing_mode.label().to_string();
        if !model.is_empty() {
            self.perf_pipeline.telemetry.active_model = model.clone();
        }
        LocalAIManager::warmup_async(&self.gemini_client.runtime, provider, base_url, model);
    }

    fn poll_local_discovery_events(&mut self) {
        while let Ok(servers) = self.local_discovery_rx.try_recv() {
            self.discovered_local_servers = servers;
            if let Some(first) = self.discovered_local_servers.first() {
                if self.config.local_endpoint.trim().is_empty()
                    || self.config.local_model.trim().is_empty()
                {
                    let chosen_model = first.models.first().cloned().unwrap_or_default();
                    self.active_local_provider = first.provider.clone();
                    self.config.local_endpoint = first.base_url.clone();
                    if !chosen_model.is_empty() {
                        self.config.local_model = chosen_model.clone();
                        self.perf_pipeline.telemetry.active_model = chosen_model.clone();
                    }
                    LocalAIManager::warmup_async(
                        &self.gemini_client.runtime,
                        first.provider.clone(),
                        first.base_url.clone(),
                        self.config.local_model.clone(),
                    );
                }
            }
        }
    }

    pub fn submit_user_chat_or_command(&mut self, user_text: String) {
        let trimmed = user_text.trim().to_string();
        if trimmed.is_empty() {
            return;
        }
        self.chat_drawer_open = true;
        if let Some(fast_act) = WhiteboardAgent::parse_fast_intent(&trimmed) {
            let summary = self.execute_whiteboard_actions(&[fast_act], None);
            let now_str = chrono::Local::now().format("%H:%M:%S").to_string();
            self.perf_pipeline.telemetry.ai_mode = "Instant".to_string();
            self.perf_pipeline.telemetry.active_model = "Local-Intent".to_string();
            self.chat_messages.push(ChatMessage {
                role: "user".to_string(),
                text: trimmed,
                subject: self.subject_mode,
                has_image_attachment: false,
                image_dimensions: None,
                rag_slide_badge: None,
                timestamp: now_str.clone(),
            });
            self.chat_messages.push(ChatMessage {
                role: "model".to_string(),
                text: format!("⚡ [Lệnh tức thì <5ms]\n{summary}"),
                subject: self.subject_mode,
                has_image_attachment: false,
                image_dimensions: None,
                rag_slide_badge: None,
                timestamp: now_str,
            });
            return;
        }
        self.send_chat_message(trimmed, false);
    }

    pub fn clear_lasso_selection(&mut self) {
        self.lasso_points.clear();
        self.active_lasso_region = None;
        self.selection_anchor_board = None;
        self.selection_rect_board = None;
    }

    pub fn execute_lasso_quick_action(&mut self, action: LassoQuickAction) {
        let Some(region) = self.active_lasso_region.clone() else {
            return;
        };
        self.chat_drawer_open = true;
        self.pending_lasso_action = Some(action);
        self.selection_rect_board = Some((region.min_pt, region.max_pt));
        self.send_chat_message(action.prompt().to_string(), true);
        if action != LassoQuickAction::BeautifyLatex {
            self.active_lasso_region = None;
            self.lasso_points.clear();
        }
        self.selection_rect_board = None;
        self.selection_anchor_board = None;
    }

    fn occupied_board_rects(&self) -> Vec<BoardRect> {
        let mut rects = Vec::new();
        if self.slide_card_visible && !self.slides.is_empty() {
            rects.push(BoardRect {
                x: 16.0,
                y: 46.0,
                w: 740.0,
                h: 416.0,
            });
        }
        for s in &self.strokes {
            if s.display_points.len() >= 2 {
                let mut x0 = f32::INFINITY;
                let mut y0 = f32::INFINITY;
                let mut x1 = f32::NEG_INFINITY;
                let mut y1 = f32::NEG_INFINITY;
                for p in &s.display_points {
                    x0 = x0.min(p.x);
                    y0 = y0.min(p.y);
                    x1 = x1.max(p.x);
                    y1 = y1.max(p.y);
                }
                rects.push(BoardRect {
                    x: x0,
                    y: y0,
                    w: (x1 - x0).max(20.0),
                    h: (y1 - y0).max(20.0),
                });
            }
        }
        for ann in &self.annotations {
            let clean = OutputSanitizer::sanitize_plain_text(&ann.text);
            let w = ((clean.chars().count() as f32) * 9.5 + 32.0).clamp(140.0, 480.0);
            let h = ((clean.lines().count().max(1) as f32) * 24.0 + 18.0).max(40.0);
            rects.push(BoardRect {
                x: ann.position.x,
                y: ann.position.y,
                w,
                h,
            });
        }
        rects
    }

    pub fn execute_whiteboard_actions(
        &mut self,
        actions: &[WhiteboardAction],
        preferred_anchor: Option<Point2D>,
    ) -> String {
        if actions.is_empty() {
            return String::new();
        }
        let (cw, ch) = self
            .last_canvas_rect
            .map(|r| (r.width().max(640.0), r.height().max(420.0)))
            .unwrap_or((1020.0, 620.0));
        let mut reports = Vec::new();

        for act in actions {
            match act {
                WhiteboardAction::ClearBoard => {
                    self.ai_action_undo_stack
                        .push((self.strokes.clone(), self.annotations.clone()));
                    self.strokes.clear();
                    self.annotations.clear();
                    reports.push("Đã xóa bảng.".to_string());
                }
                WhiteboardAction::UndoLastAiAction => {
                    if self.undo_last_ai_action() {
                        reports.push("Đã hoàn tác thao tác AI gần nhất.".to_string());
                    } else {
                        reports.push("Không có thao tác AI để hoàn tác.".to_string());
                    }
                }
                WhiteboardAction::WriteOnBoard {
                    content,
                    is_latex,
                    position,
                    ..
                } => {
                    self.ai_action_undo_stack
                        .push((self.strokes.clone(), self.annotations.clone()));
                    let clean = OutputSanitizer::sanitize_plain_text(content);
                    let est_w = (clean.chars().count() as f32 * 10.0 + 36.0).clamp(160.0, 480.0);
                    let pos = position.unwrap_or_else(|| {
                        find_free_board_region(
                            &self.occupied_board_rects(),
                            cw,
                            ch,
                            est_w,
                            52.0,
                            preferred_anchor,
                        )
                    });
                    self.annotations.push(BoardTextAnnotation {
                        position: pos,
                        text: clean.clone(),
                        is_latex: *is_latex,
                        confidence: 1.0,
                        color_rgba: self.current_ink_color,
                    });
                    reports.push(format!(
                        "Đã ghi lên bảng tại ({:.0}, {:.0}): {}",
                        pos.x, pos.y, clean
                    ));
                }
                WhiteboardAction::SolveOnBoard { steps, .. } => {
                    self.ai_action_undo_stack
                        .push((self.strokes.clone(), self.annotations.clone()));
                    let block_h = (steps.len() as f32 * 38.0).clamp(64.0, ch - 80.0);
                    let base = find_free_board_region(
                        &self.occupied_board_rects(),
                        cw,
                        ch,
                        360.0,
                        block_h,
                        preferred_anchor,
                    );
                    for (idx, st) in steps.iter().enumerate() {
                        let clean = OutputSanitizer::sanitize_plain_text(st);
                        self.annotations.push(BoardTextAnnotation {
                            position: Point2D::new(base.x, base.y + idx as f32 * 40.0),
                            text: clean,
                            is_latex: true,
                            confidence: 1.0,
                            color_rgba: [0x06, 0xb6, 0xd4, 255],
                        });
                    }
                    reports.push(format!("Đã ghi {} bước lời giải lên bảng.", steps.len()));
                }
                WhiteboardAction::DrawText {
                    x,
                    y,
                    text,
                    color_rgba,
                    normalized,
                    ..
                } => {
                    self.ai_action_undo_stack
                        .push((self.strokes.clone(), self.annotations.clone()));
                    let clean = OutputSanitizer::sanitize_plain_text(text);
                    let px = if *normalized { (*x / 1000.0) * cw } else { *x };
                    let py = if *normalized { (*y / 1000.0) * ch } else { *y };
                    self.annotations.push(BoardTextAnnotation {
                        position: Point2D::new(px, py),
                        text: clean.clone(),
                        is_latex: true,
                        confidence: 1.0,
                        color_rgba: color_rgba.unwrap_or([0xFE, 0xE4, 0x40, 255]),
                    });
                    reports.push(format!("Đã viết lên bảng tại ({:.0}, {:.0}): {}", px, py, clean));
                }
                WhiteboardAction::DrawPolyline {
                    points,
                    color_rgba,
                    stroke_width,
                    smooth,
                    normalized,
                } => {
                    if points.len() >= 2 {
                        self.ai_action_undo_stack
                            .push((self.strokes.clone(), self.annotations.clone()));
                        let mut px_pts: Vec<Point2D> = points
                            .iter()
                            .map(|p| {
                                if *normalized {
                                    Point2D::new((p.x / 1000.0) * cw, (p.y / 1000.0) * ch)
                                } else {
                                    *p
                                }
                            })
                            .collect();
                        if *smooth && px_pts.len() >= 3 {
                            let mut smoothed = Vec::with_capacity(px_pts.len() * 4);
                            crate::core::stroke_engine::generate_cubic_bezier_spline_into(&px_pts, 4, &mut smoothed);
                            if !smoothed.is_empty() {
                                px_pts = smoothed;
                            }
                        }
                        let ts = self.current_recording_timestamp_ms();
                        let col = color_rgba.unwrap_or([0x00, 0xF5, 0xD4, 255]);
                        let init_len = px_pts.len().min(2);
                        let stroke_idx = self.strokes.len();
                        self.strokes.push(PersistedStroke {
                            timestamp_ms: ts,
                            raw_points: px_pts.clone(),
                            display_points: px_pts[..init_len].to_vec(),
                            color_rgba: col,
                            thickness: *stroke_width,
                            snapped_shape: SnappedShape::Freehand,
                        });
                        if px_pts.len() > 2 {
                            self.pending_ai_strokes.push((stroke_idx, px_pts));
                        }
                        reports.push(format!("Đã vẽ đường cong/đồ thị ({} điểm).", points.len()));
                    }
                }
                WhiteboardAction::HighlightRegion {
                    x,
                    y,
                    w,
                    h,
                    color_rgba,
                    label,
                    normalized,
                } => {
                    self.ai_action_undo_stack
                        .push((self.strokes.clone(), self.annotations.clone()));
                    let rx = if *normalized { (*x / 1000.0) * cw } else { *x };
                    let ry = if *normalized { (*y / 1000.0) * ch } else { *y };
                    let rw = if *normalized { (*w / 1000.0) * cw } else { *w };
                    let rh = if *normalized { (*h / 1000.0) * ch } else { *h };
                    let col = color_rgba.unwrap_or([0xFE, 0xE4, 0x40, 220]);
                    let poly = vec![
                        Point2D::new(rx, ry),
                        Point2D::new(rx + rw, ry),
                        Point2D::new(rx + rw, ry + rh),
                        Point2D::new(rx, ry + rh),
                        Point2D::new(rx, ry),
                    ];
                    let ts = self.current_recording_timestamp_ms();
                    self.strokes.push(PersistedStroke {
                        timestamp_ms: ts,
                        raw_points: poly.clone(),
                        display_points: poly,
                        color_rgba: col,
                        thickness: 2.5,
                        snapped_shape: SnappedShape::Freehand,
                    });
                    if let Some(lbl) = label {
                        if !lbl.is_empty() {
                            self.annotations.push(BoardTextAnnotation {
                                position: Point2D::new(rx + 6.0, (ry - 24.0).max(20.0)),
                                text: OutputSanitizer::sanitize_plain_text(lbl),
                                is_latex: false,
                                confidence: 1.0,
                                color_rgba: col,
                            });
                        }
                    }
                    reports.push(format!("Đã khoanh sáng vùng ({:.0}, {:.0}, {:.0}x{:.0}).", rx, ry, rw, rh));
                }
                WhiteboardAction::ClearRegion {
                    x,
                    y,
                    w,
                    h,
                    normalized,
                } => {
                    self.ai_action_undo_stack
                        .push((self.strokes.clone(), self.annotations.clone()));
                    let rx = if *normalized { (*x / 1000.0) * cw } else { *x };
                    let ry = if *normalized { (*y / 1000.0) * ch } else { *y };
                    let rw = if *normalized { (*w / 1000.0) * cw } else { *w };
                    let rh = if *normalized { (*h / 1000.0) * ch } else { *h };
                    self.strokes.retain(|st| {
                        !st.display_points
                            .iter()
                            .any(|p| p.x >= rx && p.x <= rx + rw && p.y >= ry && p.y <= ry + rh)
                    });
                    self.annotations.retain(|ann| {
                        !(ann.position.x >= rx
                            && ann.position.x <= rx + rw
                            && ann.position.y >= ry
                            && ann.position.y <= ry + rh)
                    });
                    self.pending_ai_strokes.clear();
                    reports.push("Đã xóa vùng chỉ định trên bảng.".to_string());
                }
                WhiteboardAction::DrawShapeNorm {
                    shape_type,
                    x,
                    y,
                    w,
                    h,
                    radius,
                    x2,
                    y2,
                    points,
                    color_rgba,
                    stroke_width,
                    label,
                    normalized,
                } => {
                    self.ai_action_undo_stack
                        .push((self.strokes.clone(), self.annotations.clone()));
                    let px = if *normalized { (*x / 1000.0) * cw } else { *x };
                    let py = if *normalized { (*y / 1000.0) * ch } else { *y };
                    let pw = if *normalized { ((*w / 1000.0) * cw).max(36.0) } else { *w };
                    let ph = if *normalized { ((*h / 1000.0) * ch).max(36.0) } else { *h };
                    let pr = if *normalized { ((*radius / 1000.0) * cw.min(ch)).max(20.0) } else { *radius };
                    let pos = Point2D::new(px, py);
                    let col = color_rgba.unwrap_or([0x00, 0xF5, 0xD4, 255]);
                    let ts = self.current_recording_timestamp_ms();

                    let polylines = if let Some(pts) = points {
                        if pts.len() >= 3 {
                            let mut ring: Vec<Point2D> = pts
                                .iter()
                                .map(|p| {
                                    if *normalized {
                                        Point2D::new((p.x / 1000.0) * cw, (p.y / 1000.0) * ch)
                                    } else {
                                        *p
                                    }
                                })
                                .collect();
                            ring.push(ring[0]);
                            vec![ring]
                        } else {
                            build_vector_shape_strokes(shape_type, pos, pw, ph, pr)
                        }
                    } else if shape_type == "line" || shape_type == "arrow" {
                        if let (Some(nx2), Some(ny2)) = (x2, y2) {
                            let ex = if *normalized { (*nx2 / 1000.0) * cw } else { *nx2 };
                            let ey = if *normalized { (*ny2 / 1000.0) * ch } else { *ny2 };
                            let dx = ex - pos.x;
                            let dy = ey - pos.y;
                            if shape_type == "arrow" {
                                let ang = dy.atan2(dx);
                                let head = 15.0_f32;
                                vec![
                                    vec![pos, Point2D::new(ex, ey)],
                                    vec![
                                        Point2D::new(ex - head * (ang - 0.45).cos(), ey - head * (ang - 0.45).sin()),
                                        Point2D::new(ex, ey),
                                        Point2D::new(ex - head * (ang + 0.45).cos(), ey - head * (ang + 0.45).sin()),
                                    ],
                                ]
                            } else {
                                vec![vec![pos, Point2D::new(ex, ey)]]
                            }
                        } else {
                            build_vector_shape_strokes(shape_type, pos, pw, ph, pr)
                        }
                    } else {
                        build_vector_shape_strokes(shape_type, pos, pw, ph, pr)
                    };

                    for poly in polylines {
                        let init_len = if poly.len() > 3 { 2 } else { poly.len() };
                        let stroke_idx = self.strokes.len();
                        self.strokes.push(PersistedStroke {
                            timestamp_ms: ts,
                            raw_points: poly.clone(),
                            display_points: poly[..init_len].to_vec(),
                            color_rgba: col,
                            thickness: *stroke_width,
                            snapped_shape: SnappedShape::Freehand,
                        });
                        if poly.len() > 3 {
                            self.pending_ai_strokes.push((stroke_idx, poly));
                        }
                    }
                    if let Some(lbl) = label {
                        if !lbl.is_empty() {
                            self.annotations.push(BoardTextAnnotation {
                                position: Point2D::new(pos.x + 6.0, (pos.y - 24.0).max(20.0)),
                                text: OutputSanitizer::sanitize_plain_text(lbl),
                                is_latex: false,
                                confidence: 1.0,
                                color_rgba: col,
                            });
                        }
                    }
                    reports.push(format!("Đã vẽ vector `{}` tại ({:.0}, {:.0}).", shape_type, pos.x, pos.y));
                }
                WhiteboardAction::DrawShape {
                    shape_type,
                    origin,
                    w,
                    h,
                    radius,
                    color_rgba,
                } => {
                    self.ai_action_undo_stack
                        .push((self.strokes.clone(), self.annotations.clone()));
                    let pos = origin.unwrap_or_else(|| {
                        find_free_board_region(
                            &self.occupied_board_rects(),
                            cw,
                            ch,
                            *w,
                            *h,
                            preferred_anchor,
                        )
                    });
                    let ts = self.current_recording_timestamp_ms();
                    let col = color_rgba.unwrap_or(self.current_ink_color);
                    let polylines = build_vector_shape_strokes(shape_type, pos, *w, *h, *radius);
                    for poly in polylines {
                        let init_len = if poly.len() > 3 { 2 } else { poly.len() };
                        let stroke_idx = self.strokes.len();
                        self.strokes.push(PersistedStroke {
                            timestamp_ms: ts,
                            raw_points: poly.clone(),
                            display_points: poly[..init_len].to_vec(),
                            color_rgba: col,
                            thickness: 3.5,
                            snapped_shape: SnappedShape::Freehand,
                        });
                        if poly.len() > 3 {
                            self.pending_ai_strokes.push((stroke_idx, poly));
                        }
                    }
                    reports.push(format!(
                        "Đã vẽ vector `{}` tại ({:.0}, {:.0}).",
                        shape_type, pos.x, pos.y
                    ));
                }
            }
        }
        reports.join("\n")
    }

    pub fn tick_animated_ai_strokes(&mut self, ctx: &egui::Context) {
        if self.pending_ai_strokes.is_empty() {
            return;
        }
        let mut still_pending = Vec::new();
        for (idx, full_pts) in self.pending_ai_strokes.drain(..) {
            if let Some(stroke) = self.strokes.get_mut(idx) {
                let cur = stroke.display_points.len();
                let total = full_pts.len();
                let step = (total / 14).max(2);
                let next_len = (cur + step).min(total);
                stroke.display_points = full_pts[..next_len].to_vec();
                if next_len < total {
                    still_pending.push((idx, full_pts));
                }
            }
        }
        self.pending_ai_strokes = still_pending;
        if !self.pending_ai_strokes.is_empty() {
            ctx.request_repaint();
        }
    }

    pub fn undo_last_ai_action(&mut self) -> bool {
        if let Some((prev_s, prev_a)) = self.ai_action_undo_stack.pop() {
            self.strokes = prev_s;
            self.annotations = prev_a;
            self.last_snapped_shape_banner = "Đã hoàn tác thao tác AI gần nhất".to_string();
            true
        } else {
            false
        }
    }

    fn poll_chat_stream_events(&mut self) {
        while let Ok(ev) = self.chat_event_rx.try_recv() {
            match ev {
                ChatStreamEvent::Started { .. } => {
                    self.chat_streaming_active = true;
                }
                ChatStreamEvent::TokenDelta {
                    message_index,
                    delta,
                } => {
                    if let Some(msg) = self.chat_messages.get_mut(message_index) {
                        msg.text.push_str(&delta);
                    }
                }
                ChatStreamEvent::ApiStatusUpdated { status, message } => {
                    self.api_connection_status = status;
                    self.api_status_detail_vi = message;
                }
                ChatStreamEvent::MetricsUpdated {
                    engine,
                    model,
                    tokens_per_sec,
                } => {
                    self.perf_pipeline.telemetry.ai_mode = engine;
                    self.perf_pipeline.telemetry.active_model = model;
                    self.perf_pipeline.telemetry.tokens_per_sec = tokens_per_sec;
                }
                ChatStreamEvent::Completed { message_index } => {
                    self.chat_streaming_active = false;
                    let raw_reply = self
                        .chat_messages
                        .get(message_index)
                        .map(|m| m.text.clone())
                        .unwrap_or_default();
                    let pending_lasso = self.pending_lasso_action.take();
                    if pending_lasso == Some(LassoQuickAction::BeautifyLatex) {
                        if let Some(region) = self.active_lasso_region.take() {
                            let formula = extract_clean_beautified_formula(&raw_reply);
                            let clean_formula = OutputSanitizer::sanitize_plain_text(&formula);
                            if !clean_formula.is_empty() {
                                self.ai_action_undo_stack
                                    .push((self.strokes.clone(), self.annotations.clone()));
                                self.strokes
                                    .retain(|s| !region.contains_stroke(&s.display_points));
                                self.annotations.push(BoardTextAnnotation {
                                    position: Point2D::new(
                                        region.min_pt.x + 8.0,
                                        region.min_pt.y + 8.0,
                                    ),
                                    text: clean_formula.clone(),
                                    is_latex: true,
                                    confidence: 1.0,
                                    color_rgba: self.current_ink_color,
                                });
                            }
                            self.lasso_points.clear();
                        }
                    }

                    let (actions, clean_txt) = WhiteboardAgent::extract_actions_from_response(&raw_reply);
                    let mut final_msg = clean_txt.clone();
                    if !actions.is_empty() {
                        let summary = self.execute_whiteboard_actions(&actions, None);
                        if !summary.is_empty() {
                            if final_msg.is_empty() {
                                final_msg = summary;
                            } else {
                                final_msg.push_str(&format!("\n\n---\n{summary}"));
                            }
                        }
                    } else if pending_lasso == Some(LassoQuickAction::WriteSolutionToBoard)
                        && !clean_txt.trim().is_empty()
                    {
                        let lines: Vec<String> = clean_txt
                            .lines()
                            .map(|l| l.trim().to_string())
                            .filter(|l| !l.is_empty())
                            .take(6)
                            .collect();
                        if !lines.is_empty() {
                            self.execute_whiteboard_actions(
                                &[WhiteboardAction::SolveOnBoard {
                                    steps: lines,
                                    target_region: None,
                                }],
                                None,
                            );
                        }
                    }
                    if let Some(msg) = self.chat_messages.get_mut(message_index) {
                        msg.text = final_msg;
                    }
                }
                ChatStreamEvent::CacheStore {
                    cache_key,
                    model_id,
                    response_text,
                    structured_json,
                } => {
                    if !cache_key.trim().is_empty() && !response_text.trim().is_empty() {
                        self.storage.put_cached_ai_response(
                            &cache_key,
                            &model_id,
                            &response_text,
                            &structured_json,
                        );
                    }
                }
                ChatStreamEvent::Error {
                    message_index,
                    error,
                } => {
                    self.chat_streaming_active = false;
                    self.api_connection_status = ApiConnectionStatus::Error;
                    self.api_status_detail_vi = error.clone();
                    if let Some(msg) = self.chat_messages.get_mut(message_index) {
                        if msg.text.is_empty() {
                            msg.text = format!("⚠ {error}");
                        } else {
                            msg.text.push_str(&format!("\n\n⚠ {error}"));
                        }
                    }
                }
            }
        }
    }

    pub fn trigger_gemini_health_check(&mut self) {
        if self.api_test_in_flight {
            return;
        }
        let draft = self.api_key_edit_buffer.trim();
        let stored = self.storage.get_decrypted_api_key(&self.config);
        let effective = if !draft.is_empty() {
            draft.to_string()
        } else {
            stored
        };
        let (resolved, _src) = GeminiChatClient::resolve_api_key_with_source(&effective);
        if resolved.is_empty() {
            self.api_connection_status = ApiConnectionStatus::Unconfigured;
            self.api_status_detail_vi =
                "Chưa có API Key để kiểm tra. Hãy nhập Key hoặc đặt trong `.env`.".to_string();
            self.settings_status_banner = self.api_status_detail_vi.clone();
            return;
        }
        self.api_test_in_flight = true;
        self.api_status_detail_vi = "Đang gọi Gemini generateContent kiểm tra thật...".to_string();
        self.settings_status_banner = self.api_status_detail_vi.clone();
        self.gemini_client.test_connection(
            effective,
            self.config.gemini_model.clone(),
            self.api_health_tx.clone(),
        );
    }

    pub fn save_ui_gemini_api_key(&mut self) {
        let trimmed = self.api_key_edit_buffer.trim().to_string();
        if trimmed.is_empty() {
            self.api_status_detail_vi = "API Key trống. Dùng [Xóa Key] để xóa cài đặt.".to_string();
            self.settings_status_banner = self.api_status_detail_vi.clone();
            return;
        }
        self.storage.set_encrypted_api_key(&mut self.config, &trimmed);
        self.api_connection_status = ApiConnectionStatus::Unconfigured;
        self.api_status_detail_vi = format!(
            "Đã mã hóa AES-256-GCM và lưu Key ({}). Nhấn [Test API] để xác thực.",
            mask_api_key(&trimmed)
        );
        self.settings_status_banner = format!("[OK] {}", self.api_status_detail_vi);
    }

    pub fn clear_ui_gemini_api_key(&mut self) {
        self.storage.clear_encrypted_api_key(&mut self.config);
        self.api_key_edit_buffer.clear();
        self.api_connection_status = ApiConnectionStatus::Unconfigured;
        let (fallback_key, _src) = GeminiChatClient::resolve_api_key_with_source("");
        self.api_status_detail_vi = if fallback_key.is_empty() {
            "Đã xóa API Key khỏi cấu hình UI cục bộ.".to_string()
        } else {
            "Đã xóa Key UI Settings. Hệ thống dùng Key dự phòng từ `.env`/ENV.".to_string()
        };
        self.settings_status_banner = self.api_status_detail_vi.clone();
    }

    fn poll_gemini_health_events(&mut self) {
        while let Ok(res) = self.api_health_rx.try_recv() {
            self.api_test_in_flight = false;
            self.api_connection_status = res.status;
            self.api_status_detail_vi = res.message_vi.clone();
            self.settings_status_banner = if res.ok {
                format!("[OK] {}", res.message_vi)
            } else {
                format!("[LỖI] {}", res.message_vi)
            };
        }
    }

    /// Push an AI response summary or formula directly onto the Smartboard Canvas as clean Unicode math lines.
    pub fn push_ai_text_to_board(&mut self, ai_text: &str) {
        let mut y_offset = 110.0_f32 + (self.annotations.len() as f32 * 40.0) % 320.0;
        for line in ai_text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .take(5)
        {
            let clean = crate::utils::document_loader::DocumentLoader::format_latex_to_unicode_math(
                &line.replace("**", "").replace("##", "").replace("###", ""),
            );
            let truncated: String = clean.chars().take(92).collect();
            self.annotations.push(BoardTextAnnotation {
                text: truncated,
                is_latex: true,
                position: Point2D::new(90.0, y_offset),
                color_rgba: [6, 182, 212, 255],
                confidence: 0.98,
            });
            y_offset += 38.0;
        }
        self.last_snapped_shape_banner = "Đã đẩy lời giải AI lên Bảng giảng".to_string();
    }

    pub fn insert_subject_diagram_template(&mut self) {
        let ts = self.current_recording_timestamp_ms();
        match self.subject_mode {
            SubjectAssistantMode::Math => {
                self.strokes.push(PersistedStroke {
                    timestamp_ms: ts,
                    raw_points: vec![Point2D::new(360.0, 320.0), Point2D::new(760.0, 320.0)],
                    display_points: vec![Point2D::new(360.0, 320.0), Point2D::new(760.0, 320.0)],
                    color_rgba: [148, 163, 184, 255],
                    thickness: 2.5,
                    snapped_shape: SnappedShape::Line {
                        start: Point2D::new(360.0, 320.0),
                        end: Point2D::new(760.0, 320.0),
                    },
                });
                self.strokes.push(PersistedStroke {
                    timestamp_ms: ts,
                    raw_points: vec![Point2D::new(560.0, 140.0), Point2D::new(560.0, 460.0)],
                    display_points: vec![Point2D::new(560.0, 140.0), Point2D::new(560.0, 460.0)],
                    color_rgba: [148, 163, 184, 255],
                    thickness: 2.5,
                    snapped_shape: SnappedShape::Line {
                        start: Point2D::new(560.0, 140.0),
                        end: Point2D::new(560.0, 460.0),
                    },
                });
                let mut parabola = Vec::new();
                for i in -20..=20 {
                    let x = i as f32 * 7.0;
                    let y = -0.018 * x * x;
                    parabola.push(Point2D::new(560.0 + x, 320.0 + y));
                }
                self.strokes.push(PersistedStroke {
                    timestamp_ms: ts,
                    raw_points: parabola.clone(),
                    display_points: parabola,
                    color_rgba: [6, 182, 212, 255],
                    thickness: 3.2,
                    snapped_shape: SnappedShape::Freehand,
                });
            }
            SubjectAssistantMode::Physics => {
                let tri = [
                    Point2D::new(380.0, 420.0),
                    Point2D::new(740.0, 420.0),
                    Point2D::new(380.0, 230.0),
                ];
                self.strokes.push(PersistedStroke {
                    timestamp_ms: ts,
                    raw_points: tri.to_vec(),
                    display_points: vec![tri[0], tri[1], tri[2], tri[0]],
                    color_rgba: [79, 70, 229, 255],
                    thickness: 3.0,
                    snapped_shape: SnappedShape::Triangle { vertices: tri },
                });
            }
            SubjectAssistantMode::Chemistry => {
                self.spawn_pubchem_structure_on_board("Benzene");
            }
            SubjectAssistantMode::General => {
                let min = Point2D::new(400.0, 220.0);
                let max = Point2D::new(720.0, 340.0);
                self.strokes.push(PersistedStroke {
                    timestamp_ms: ts,
                    raw_points: vec![min, max],
                    display_points: vec![
                        Point2D::new(min.x, min.y),
                        Point2D::new(max.x, min.y),
                        Point2D::new(max.x, max.y),
                        Point2D::new(min.x, max.y),
                        Point2D::new(min.x, min.y),
                    ],
                    color_rgba: [79, 70, 229, 255],
                    thickness: 3.0,
                    snapped_shape: SnappedShape::Rectangle { min, max },
                });
            }
        }
    }

    pub fn open_slide_file_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter(
                "Tài liệu & Bài giảng (Word, PDF, PowerPoint, Ảnh)",
                &["docx", "doc", "pdf", "pptx", "ppt", "ppsx", "potx", "png", "jpg", "jpeg", "webp"],
            )
            .pick_file()
        {
            match DocumentLoader::load_slides_from_path(&path) {
                Ok(loaded) => {
                    self.slides = loaded;
                    self.rag_index.rebuild_from_slides(&self.slides);
                    self.current_slide_idx = 0;
                    self.invalidate_slide_texture();
                }
                Err(err) => {
                    self.last_snapped_shape_banner = format!("[LỖI MỞ TÀI LIỆU] {err}");
                }
            }
        }
    }

    pub fn export_audio_wav_now(&mut self) -> Option<PathBuf> {
        let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let local_rec_dir = PathBuf::from("recordings");
        let _ = std::fs::create_dir_all(&local_rec_dir);
        let local_wav = local_rec_dir.join(format!("lecture_{ts}.wav"));
        let _ = self.audio_recorder.save_to_wav(&local_wav);

        let wav_path =
            LocalStorage::resolve_recordings_dir().join(format!("lecture_{ts}.wav"));
        match self.audio_recorder.save_to_wav(&wav_path) {
            Ok(saved) => {
                self.last_snapped_shape_banner =
                    format!("[REC] Đã lưu file âm thanh: {}", local_wav.display());
                Some(saved)
            }
            Err(err) => {
                self.last_snapped_shape_banner = format!("[LỖI] Lỗi lưu WAV: {err}");
                None
            }
        }
    }

    /// Save the `.airteach` presentation bundle alongside `.wav` audio and `.srt` subtitles (`Workstream 2.2` & `2.3`).
    pub fn save_current_lecture_session(&mut self) {
        let ts = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let rec_dir = LocalStorage::resolve_recordings_dir();

        let wav_saved = if self.audio_recorder.recorded_duration_secs() > 0.1 {
            self.export_audio_wav_now()
                .map(|p| p.to_string_lossy().to_string())
        } else {
            None
        };

        let srt_path = rec_dir.join(format!("airteach_lecture_{ts}.srt"));
        let srt_saved = self
            .subtitles_engine
            .export_srt(&srt_path)
            .ok()
            .map(|p| p.to_string_lossy().to_string());

        let vector_strokes: Vec<VectorStroke> = self
            .strokes
            .iter()
            .map(PersistedStroke::to_vector_stroke)
            .collect();

        let bundle = AirTeachBundle {
            version: "2.1.0".to_string(),
            title: self.config.lecture_title.clone(),
            subject: self.subject_mode.short_badge().to_string(),
            created_at: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
            slide_index: self.current_slide_idx,
            strokes: self.strokes.clone(),
            vector_strokes,
            annotations: self.annotations.clone(),
            audio_wav_path: wav_saved.clone(),
            subtitles_srt_path: srt_saved,
        };

        let bundle_file_path = rec_dir.join(format!("airteach_lecture_{ts}.airteach"));
        let _ = self.storage.export_airteach_file(&bundle_file_path, &bundle);

        let uid = self.logged_in_user_id.as_deref().unwrap_or("");
        let _ = self.storage.save_lecture_bundle_to_db(uid, false, &bundle);
        self.lecture_history = self.storage.list_lecture_history(uid);
        self.last_snapped_shape_banner = match wav_saved {
            Some(w) => format!("[LƯU] Đã lưu bài giảng (.airteach + .srt) kèm ghi âm ({w})"),
            None => format!(
                "[LƯU] Đã lưu bài giảng (.airteach + .srt) vào {}",
                bundle_file_path.display()
            ),
        };
    }

    fn render_dashboard(&mut self, ctx: &egui::Context) {
        if !self.calibration_modal_open {
            self.release_camera_hardware();
        }
        let palette = self.config.theme_mode.palette();
        let teacher_label = if self.logged_in_teacher_name.trim().is_empty() {
            "Người dùng cục bộ".to_string()
        } else {
            format!(
                "{} (@{})",
                self.logged_in_teacher_name, self.logged_in_username
            )
        };

        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(palette.bg_canvas)
                    .inner_margin(egui::Margin::same(24.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui_brand_logo(ui, 36.0);
                    ui.add_space(6.0);
                    ui.vertical(|ui| {
                        ui.label(
                            RichText::new("AirTeach AI — Studio Bảng Tương Tác (by DNT - Victer)")
                                .font(FontId::proportional(21.0))
                                .strong()
                                .color(palette.text_primary),
                        );
                        ui.label(
                            RichText::new(format!(
                                "Tài khoản: {} | Khởi động: {:.1} ms | RAG: {} đoạn | Mic: {}",
                                teacher_label,
                                self.cold_start_ms,
                                self.rag_index.chunk_count(),
                                self.audio_recorder.device_name()
                            ))
                            .font(FontId::monospace(11.5))
                            .color(palette.secondary),
                        );
                    });

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Đăng xuất").color(palette.danger).strong(),
                                )
                                .fill(palette.surface)
                                .stroke(Stroke::new(1.0_f32, palette.danger))
                                .rounding(Rounding::same(8.0))
                                .min_size(Vec2::new(105.0, 36.0)),
                            )
                            .clicked()
                        {
                            self.logout_to_auth_gate();
                        }
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Giới thiệu").color(palette.text_secondary),
                                )
                                .fill(palette.surface)
                                .stroke(Stroke::new(1.0_f32, palette.border))
                                .rounding(Rounding::same(8.0))
                                .min_size(Vec2::new(92.0, 36.0)),
                            )
                            .clicked()
                        {
                            self.about_modal_open = true;
                        }
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Cài đặt").color(palette.text_primary),
                                )
                                .fill(palette.surface)
                                .stroke(Stroke::new(1.0_f32, palette.border))
                                .rounding(Rounding::same(8.0))
                                .min_size(Vec2::new(95.0, 36.0)),
                            )
                            .clicked()
                        {
                            self.screen = AppScreen::Settings;
                        }
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Mở Bảng Tương Tác ->")
                                        .font(FontId::proportional(14.5))
                                        .strong()
                                        .color(Color32::WHITE),
                                )
                                .fill(palette.primary)
                                .rounding(Rounding::same(8.0))
                                .min_size(Vec2::new(195.0, 36.0)),
                            )
                            .clicked()
                        {
                            self.screen = AppScreen::LectureMode;
                            if self.left_camera_dock_open {
                                self.ensure_camera_acquired();
                            }
                        }
                    });
                });

                ui.add_space(18.0);

                ui.label(
                    RichText::new("Không gian Làm việc Mới (Clean Slate — Nhập Slide hoặc Chọn Chuyên đề)")
                        .font(FontId::proportional(16.0))
                        .strong()
                        .color(palette.text_primary),
                );
                ui.add_space(8.0);

                ui.horizontal_wrapped(|ui| {
                    for (mode, desc, accent) in [
                        (
                            SubjectAssistantMode::General,
                            "Bảng trắng mới không dữ liệu mẫu, tùy chọn lưới Caro 10–60px & mở Slide PDF",
                            palette.primary,
                        ),
                        (
                            SubjectAssistantMode::Math,
                            "Toán học, làm mượt nét One-Euro + Bézier & nhận diện công thức LaTeX",
                            palette.secondary,
                        ),
                        (
                            SubjectAssistantMode::Physics,
                            "Vật lý, sơ đồ phân tích lực & trợ lý sư phạm đa phương thức Gemini",
                            palette.success,
                        ),
                        (
                            SubjectAssistantMode::Chemistry,
                            "Hóa học, dựng công thức phân tử 2D/3D từ PubChem & cân bằng phản ứng",
                            palette.warning,
                        ),
                    ] {
                        egui::Frame::none()
                            .fill(palette.surface)
                            .rounding(Rounding::same(10.0))
                            .stroke(Stroke::new(1.2_f32, accent))
                            .inner_margin(egui::Margin::same(15.0))
                            .show(ui, |ui| {
                                ui.set_width(275.0);
                                ui.label(
                                    RichText::new(mode.label())
                                        .font(FontId::proportional(15.0))
                                        .strong()
                                        .color(accent),
                                );
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new(desc)
                                        .font(FontId::proportional(12.2))
                                        .color(palette.text_muted),
                                );
                                ui.add_space(10.0);
                                if ui
                                    .add(
                                        egui::Button::new(
                                            RichText::new("Bắt đầu với Bảng Trắng ->")
                                                .color(palette.text_primary)
                                                .strong(),
                                        )
                                        .fill(palette.surface_elevated)
                                        .stroke(Stroke::new(1.0_f32, palette.border))
                                        .rounding(Rounding::same(8.0)),
                                    )
                                    .clicked()
                                {
                                    self.subject_mode = mode;
                                    // Prefer Clean Blank Board if tutorial was dismissed, else slide 0
                                    if let Some(blank_idx) =
                                        self.slides.iter().position(|s| !s.is_tutorial)
                                    {
                                        self.current_slide_idx = blank_idx;
                                    } else {
                                        self.current_slide_idx = 0;
                                    }
                                    self.invalidate_slide_texture();
                                    self.screen = AppScreen::LectureMode;
                                    if self.left_camera_dock_open {
                                        self.ensure_camera_acquired();
                                    }
                                }
                            });
                        ui.add_space(10.0);
                    }
                });

                // Optional Quick Tutorial Card (only if not dismissed/deleted)
                if !self.config.tutorial_deleted && !self.config.tutorial_dismissed {
                    ui.add_space(14.0);
                    egui::Frame::none()
                        .fill(palette.surface)
                        .rounding(Rounding::same(10.0))
                        .stroke(Stroke::new(1.0_f32, palette.secondary))
                        .inner_margin(egui::Margin::same(14.0))
                        .show(ui, |ui| {
                            ui.horizontal(|ui| {
                                ui.vertical(|ui| {
                                    ui.label(
                                        RichText::new(
                                            "Hướng dẫn Khởi đầu (Tutorial · Tác giả: DNT - Victer Guide)",
                                        )
                                        .font(FontId::proportional(14.0))
                                        .strong()
                                        .color(palette.secondary),
                                    );
                                    ui.label(
                                        RichText::new(
                                            "2 trang hướng dẫn nhanh cử chỉ không chạm & phím tắt AirTeach AI. Có thể ẩn hoặc xóa vĩnh viễn chỉ bằng 1 bấm.",
                                        )
                                        .font(FontId::proportional(12.0))
                                        .color(palette.text_muted),
                                    );
                                });
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui
                                            .add(
                                                egui::Button::new(
                                                    RichText::new("Xóa vĩnh viễn")
                                                        .color(palette.danger),
                                                )
                                                .fill(palette.surface_elevated)
                                                .rounding(Rounding::same(8.0)),
                                            )
                                            .clicked()
                                        {
                                            self.dismiss_tutorial_slides(true);
                                        }
                                        if ui
                                            .add(
                                                egui::Button::new(
                                                    RichText::new("Ẩn hướng dẫn")
                                                        .color(palette.text_secondary),
                                                )
                                                .fill(palette.surface_elevated)
                                                .rounding(Rounding::same(8.0)),
                                            )
                                            .clicked()
                                        {
                                            self.dismiss_tutorial_slides(false);
                                        }
                                        if ui
                                            .add(
                                                egui::Button::new(
                                                    RichText::new("Xem Hướng dẫn (2 trang)")
                                                        .color(Color32::WHITE)
                                                        .strong(),
                                                )
                                                .fill(palette.secondary)
                                                .rounding(Rounding::same(8.0)),
                                            )
                                            .clicked()
                                        {
                                            self.current_slide_idx = 0;
                                            self.invalidate_slide_texture();
                                            self.screen = AppScreen::LectureMode;
                                            if self.left_camera_dock_open {
                                                self.ensure_camera_acquired();
                                            }
                                        }
                                    },
                                );
                            });
                        });
                }

                ui.add_space(18.0);

                // Saved Lecture History (Filtered per logged-in user)
                egui::Frame::none()
                    .fill(palette.surface)
                    .rounding(Rounding::same(10.0))
                    .stroke(Stroke::new(1.0_f32, palette.border))
                    .inner_margin(egui::Margin::same(16.0))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(
                                    "Lịch sử Bài giảng cá nhân (.airteach + .wav + .srt Mã hóa AES-256-GCM)",
                                )
                                .font(FontId::proportional(15.5))
                                .strong()
                                .color(palette.text_primary),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                RichText::new("[MỞ FILE] Mở file .airteach")
                                                    .color(palette.secondary),
                                            )
                                            .fill(palette.surface_elevated)
                                            .stroke(Stroke::new(1.0_f32, palette.border))
                                            .rounding(Rounding::same(8.0)),
                                        )
                                        .clicked()
                                    {
                                        if let Some(p) = rfd::FileDialog::new()
                                            .add_filter("AirTeach Bundle", &["airteach"])
                                            .pick_file()
                                        {
                                            if let Ok(bundle) =
                                                self.storage.import_airteach_file(&p)
                                            {
                                                self.config.lecture_title = bundle.title;
                                                self.strokes = bundle.strokes;
                                                self.annotations = bundle.annotations;
                                                self.current_slide_idx = bundle
                                                    .slide_index
                                                    .min(self.slides.len().saturating_sub(1));
                                                self.invalidate_slide_texture();
                                                self.screen = AppScreen::LectureMode;
                                                if self.left_camera_dock_open {
                                                    self.ensure_camera_acquired();
                                                }
                                            }
                                        }
                                    }
                                },
                            );
                        });
                        ui.add_space(8.0);

                        if self.lecture_history.is_empty() {
                            ui.label(
                                RichText::new(
                                    "Tài khoản của bạn chưa có bài giảng nào được lưu (Clean Slate). Mở bảng viết để bắt đầu hoặc nhập file .airteach.",
                                )
                                .color(palette.text_muted),
                            );
                        } else {
                            let mut load_id = None;
                            for item in &self.lecture_history {
                                ui.horizontal(|ui| {
                                    let wav_badge = if item.audio_wav_path.is_some() {
                                        "Có kèm WAV + SRT"
                                    } else {
                                        "Bảng viết + Timeline"
                                    };
                                    ui.label(
                                        RichText::new(format!(
                                            "[{}] {} ({} nét vẽ | {}) — {}",
                                            item.subject,
                                            item.title,
                                            item.stroke_count,
                                            wav_badge,
                                            item.created_at
                                        ))
                                        .font(FontId::monospace(12.0))
                                        .color(palette.text_primary),
                                    );
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                RichText::new("Khôi phục & Phát lại")
                                                    .color(palette.secondary),
                                            )
                                            .fill(palette.surface_elevated)
                                            .rounding(Rounding::same(8.0)),
                                        )
                                        .clicked()
                                    {
                                        load_id = Some(item.id);
                                    }
                                });
                            }
                            if let Some(id) = load_id {
                                if let Ok(bundle) = self.storage.load_lecture_bundle_from_db(id) {
                                    self.config.lecture_title = bundle.title;
                                    self.strokes = bundle.strokes;
                                    self.annotations = bundle.annotations;
                                    self.current_slide_idx = bundle
                                        .slide_index
                                        .min(self.slides.len().saturating_sub(1));
                                    self.invalidate_slide_texture();
                                    self.screen = AppScreen::LectureMode;
                                    if self.left_camera_dock_open {
                                        self.ensure_camera_acquired();
                                    }
                                }
                            }
                        }
                    });
            });

        render_about_modal(ctx, &mut self.about_modal_open, palette);
    }
}

impl eframe::App for AirTeachApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.perf_pipeline.record_render_tick();
        self.tick_animated_ai_strokes(ctx);
        self.poll_camera_and_gestures(ctx);
        self.poll_vision_worker_results();
        self.poll_chat_stream_events();
        self.poll_gemini_health_events();
        self.poll_local_discovery_events();
        self.poll_lecture_replay_and_subtitles();

        if self.calibration_modal_open {
            self.ensure_camera_acquired();
        }

        match self.screen {
            AppScreen::Welcome => {
                if !self.calibration_modal_open {
                    self.release_camera_hardware();
                }
                welcome_view::render_welcome_view(self, ctx);
            }
            AppScreen::Auth => {
                auth_view::render_auth_view(self, ctx);
            }
            AppScreen::Dashboard => {
                self.render_dashboard(ctx);
            }
            AppScreen::LectureMode => {
                if self.left_camera_dock_open || self.calibration_modal_open {
                    self.ensure_camera_acquired();
                } else {
                    self.release_camera_hardware();
                }
                // Strict Non-Overlapping Studio Dock Order:
                // 1. Top Header Bar + Sub-Toolbar
                toolbar_view::render_top_toolbar(self, ctx);
                // 2. Bottom Drawing Dock spanning full window width
                toolbar_view::render_bottom_drawing_dock(self, ctx);
                // 3. Left Collapsible Camera PIP HUD + Gesture Guide + Mic VU Meter
                canvas_view::render_left_camera_dock(self, ctx);
                // 4. Right Collapsible AI Pedagogical Assistant Drawer
                chat_view::render_chat_drawer(self, ctx);
                // 5. Center Smartboard Canvas
                canvas_view::render_lecture_canvas(self, ctx);
            }
            AppScreen::Settings => {
                if !self.calibration_modal_open {
                    self.release_camera_hardware();
                }
                settings_view::render_settings_view(self, ctx);
            }
        }

        calibration_view::render_calibration_modal(self, ctx);

        ctx.request_repaint();
    }

    fn on_exit(&mut self) {
        self.camera.shutdown();
        self.audio_recorder.shutdown();
    }
}

fn draw_thick_line_rgba(
    img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>,
    mut x0: i32,
    mut y0: i32,
    x1: i32,
    y1: i32,
    radius: i32,
    color: Rgba<u8>,
) {
    let w = img.width() as i32;
    let h = img.height() as i32;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        for oy in -radius..=radius {
            for ox in -radius..=radius {
                let px = x0 + ox;
                let py = y0 + oy;
                if px >= 0 && px < w && py >= 0 && py < h {
                    img.put_pixel(px as u32, py as u32, color);
                }
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}
