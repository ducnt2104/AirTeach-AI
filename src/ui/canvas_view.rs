//! Minimalist Left Camera Dock (`270px`) + Clean Center Smartboard Canvas (`src/ui/canvas_view.rs` — MODULE 2.2, 2.3, 3.3 & 4):
//! - Minimalist Left Dock (`Fixed Width: 270px`):
//!   1. 4:3 Camera Preview (`250x188px`) + Bright 21-point Hand Skeleton (Neon Cyan `#00F5D4` bones + Yellow `#FEE440` joints)
//!      + 1 concise status line below (`"Trạng thái: Đang Vẽ / Di chuyển / Tẩy / Chờ tay"`).
//!   2. Camera Selector (`Cam 0 / Cam 1`) + Backlight Toggle (`Ngược sáng (Auto-HDR)`).
//!   3. 4 Compact Gesture Badges: `[1 Ngón: Vẽ]`, `[2 Ngón: Di chuyển]`, `[Chụm tay: Click]`, `[5 Ngón: Tẩy]`.
//!   4. Compact Mic Volume Bar (`VU Meter`) at the bottom.
//! - Clean Center Smartboard Canvas (Zero Clutter, 80% Screen Area):
//!   * Slide card with top-left toggle button `[Ẩn nội dung Slide để lấy bảng trống 100%]`.
//!   * Zero floating status banners, zero fake OCR labels, zero bottom subtitle clutter.
//!   * Dragging a region in `[Phân Tích Vùng]` mode crops that exact region to PNG and opens the Right Gemini AI Panel.

use crate::app::{ActiveBoardTool, AirTeachApp};
use crate::core::camera::{CameraControlCmd, LightingAdaptationMode};
use crate::core::gestures::RightGesture;
use crate::core::hand_tracker::HandLandmarks;
use crate::core::stroke_engine::Point2D;
use crate::ui::brand::ThemeMode;
use crate::ui::toolbar_view::{
    ALERT_ROSE, BORDER_SLATE, PRIMARY_INDIGO, SECONDARY_TEAL, TEXT_LIGHT,
};
use crate::utils::document_loader::DocumentLoader;
use crate::utils::storage::ClassroomContrastMode;
use egui::{Align2, Color32, FontId, Pos2, Rect, RichText, Rounding, Sense, Stroke, Vec2};

const HAND_BONE_PAIRS: [(usize, usize); 20] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
];

/// MODULE 2.2: Minimalist Left Dock (`Fixed Width: 270px`) + 60 FPS Adaptive CV HUD.
pub fn render_left_camera_dock(app: &mut AirTeachApp, ctx: &egui::Context) {
    if !app.left_camera_dock_open {
        return;
    }

    let palette = app.config.theme_mode.palette();

    egui::SidePanel::left("airteach_left_camera_dock")
        .resizable(false)
        .exact_width(270.0)
        .frame(
            egui::Frame::none()
                .fill(palette.surface)
                .stroke(Stroke::new(1.0_f32, palette.border))
                .inner_margin(egui::Margin::same(10.0)),
        )
        .show(ctx, |ui| {
            // =================================================================
            // 1. 4:3 CAMERA PREVIEW (250 x 188 PX) + REAL-TIME 60 FPS CV HUD
            // =================================================================
            let cam_size = Vec2::new(250.0, 188.0);
            let (cam_rect, _resp) = ui.allocate_exact_size(cam_size, Sense::hover());
            let painter = ui.painter_at(cam_rect);

            painter.rect_filled(cam_rect, Rounding::same(8.0), palette.bg_canvas);

            let mut active_view_rect = cam_rect;
            if let Some(ref tex) = app.camera_texture {
                let tex_size = tex.size_vec2();
                let tex_aspect = if tex_size.y > 0.0 {
                    tex_size.x / tex_size.y
                } else {
                    4.0 / 3.0
                };
                let box_aspect = cam_rect.width() / cam_rect.height().max(1.0);
                let draw_rect = if (tex_aspect - box_aspect).abs() < 0.05 {
                    cam_rect
                } else if tex_aspect > box_aspect {
                    let h = cam_rect.width() / tex_aspect;
                    Rect::from_center_size(cam_rect.center(), Vec2::new(cam_rect.width(), h))
                } else {
                    let w = cam_rect.height() * tex_aspect;
                    Rect::from_center_size(cam_rect.center(), Vec2::new(w, cam_rect.height()))
                };
                active_view_rect = draw_rect;

                painter.image(
                    tex.id(),
                    draw_rect,
                    Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                    Color32::WHITE,
                );
            }

            if app.camera_using_hardware {
                // 1a. Aligned Face Bounding Box + 2D Eye-Level Horizon Overlay
                if let Some([bx, by, bw, bh]) = app.last_face_bbox_norm {
                    let face_rect = Rect::from_min_size(
                        Pos2::new(
                            active_view_rect.min.x + bx * active_view_rect.width(),
                            active_view_rect.min.y + by * active_view_rect.height(),
                        ),
                        Vec2::new(
                            bw * active_view_rect.width(),
                            bh * active_view_rect.height(),
                        ),
                    );
                    let gate_col = if app.camera_quality_gate_passed {
                        palette.success
                    } else {
                        palette.warning
                    };
                    painter.rect_stroke(
                        face_rect,
                        Rounding::same(6.0),
                        Stroke::new(
                            1.2_f32,
                            Color32::from_rgba_unmultiplied(
                                gate_col.r(),
                                gate_col.g(),
                                gate_col.b(),
                                180,
                            ),
                        ),
                    );
                }

                if let Some(pts5) = app.last_face_5pt_norm {
                    let pt_col = Color32::from_rgba_unmultiplied(0x00, 0xf5, 0xd4, 210);
                    for i in 0..5 {
                        let px = active_view_rect.min.x + pts5[i * 2] * active_view_rect.width();
                        let py =
                            active_view_rect.min.y + pts5[i * 2 + 1] * active_view_rect.height();
                        painter.circle_filled(Pos2::new(px, py), 2.0, pt_col);
                    }
                }

                if let Some([lx, ly, rx, ry]) = app.last_face_eyes_norm {
                    let lp = Pos2::new(
                        active_view_rect.min.x + lx * active_view_rect.width(),
                        active_view_rect.min.y + ly * active_view_rect.height(),
                    );
                    let rp = Pos2::new(
                        active_view_rect.min.x + rx * active_view_rect.width(),
                        active_view_rect.min.y + ry * active_view_rect.height(),
                    );
                    let eye_col = Color32::from_rgba_unmultiplied(0x00, 0xf5, 0xd4, 210);
                    painter.line_segment([lp, rp], Stroke::new(1.0_f32, eye_col));
                    for ep in [lp, rp] {
                        painter.line_segment(
                            [Pos2::new(ep.x - 3.5, ep.y), Pos2::new(ep.x + 3.5, ep.y)],
                            Stroke::new(1.2_f32, eye_col),
                        );
                        painter.line_segment(
                            [Pos2::new(ep.x, ep.y - 3.5), Pos2::new(ep.x, ep.y + 3.5)],
                            Stroke::new(1.2_f32, eye_col),
                        );
                    }
                }

                // 1b. Smooth Neon Cyan Fingertip Trail (18-frame cursor path)
                let trail_len = app.hud_right_trail.len();
                if trail_len >= 2 {
                    for i in 0..(trail_len - 1) {
                        let a = app.hud_right_trail[i];
                        let b = app.hud_right_trail[i + 1];
                        let pa = Pos2::new(
                            active_view_rect.min.x + a[0] * active_view_rect.width(),
                            active_view_rect.min.y + a[1] * active_view_rect.height(),
                        );
                        let pb = Pos2::new(
                            active_view_rect.min.x + b[0] * active_view_rect.width(),
                            active_view_rect.min.y + b[1] * active_view_rect.height(),
                        );
                        let age_t = (i + 1) as f32 / trail_len as f32;
                        let alpha = (age_t * 210.0) as u8;
                        painter.line_segment(
                            [pa, pb],
                            Stroke::new(
                                1.2 + age_t * 2.2,
                                Color32::from_rgba_unmultiplied(0x00, 0xf5, 0xd4, alpha),
                            ),
                        );
                    }
                }

                // 1c. 21-Point Hand Skeletons (mapped onto aspect-correct video rect)
                if let Some(ref lh) = app.last_two_hand_frame.left_hand {
                    draw_hand_skeleton_on_rect(&painter, active_view_rect, lh);
                }
                if let Some(ref rh) = app.last_two_hand_frame.right_hand {
                    draw_hand_skeleton_on_rect(&painter, active_view_rect, rh);
                }

                // 1d. Top HUD Telemetry Pills (Lighting Mode + Quality Gate | FPS + Confidence)
                let (mode_dot, mode_border) = match app.camera_lighting_mode {
                    LightingAdaptationMode::LowLightBoost => (
                        Color32::from_rgb(0xf5, 0x9e, 0x0b),
                        Color32::from_rgba_unmultiplied(0xf5, 0x9e, 0x0b, 170),
                    ),
                    LightingAdaptationMode::GlareSuppression => (
                        Color32::from_rgb(0x38, 0xbd, 0xf8),
                        Color32::from_rgba_unmultiplied(0x38, 0xbd, 0xf8, 170),
                    ),
                    LightingAdaptationMode::Normal => (
                        Color32::from_rgb(0x22, 0xc5, 0x5e),
                        Color32::from_rgba_unmultiplied(0x22, 0xc5, 0x5e, 160),
                    ),
                };
                let gate_tag = if app.camera_quality_gate_passed {
                    "Nét"
                } else {
                    "Mờ"
                };
                let left_pill = format!("{} · {}", app.camera_lighting_mode.label(), gate_tag);
                draw_hud_pill(
                    &painter,
                    Pos2::new(cam_rect.min.x + 6.0, cam_rect.min.y + 6.0),
                    false,
                    Color32::from_rgba_unmultiplied(10, 15, 28, 198),
                    mode_border,
                    if app.camera_quality_gate_passed {
                        mode_dot
                    } else {
                        Color32::from_rgb(0xf5, 0x9e, 0x0b)
                    },
                    &left_pill,
                );

                let hand_conf_pct = app
                    .last_two_hand_frame
                    .right_hand
                    .as_ref()
                    .or(app.last_two_hand_frame.left_hand.as_ref())
                    .map(|h| (h.confidence * 100.0).round() as i32)
                    .unwrap_or(0);
                let right_pill = format!("{:.0}fps · {}%", app.camera_fps, hand_conf_pct);
                draw_hud_pill(
                    &painter,
                    Pos2::new(cam_rect.max.x - 6.0, cam_rect.min.y + 6.0),
                    true,
                    Color32::from_rgba_unmultiplied(10, 15, 28, 198),
                    Color32::from_rgba_unmultiplied(0x00, 0xf5, 0xd4, 150),
                    Color32::from_rgb(0xe2, 0xe8, 0xf0),
                    &right_pill,
                );

                // 1e. Gesture Hold / 150ms Grace Progress Bar at bottom edge of preview
                let show_hold_bar =
                    !app.gesture_state.no_hand_idle || app.gesture_state.in_grace_period;
                if show_hold_bar {
                    let bar_h = 3.5;
                    let bar_bg = Rect::from_min_max(
                        Pos2::new(cam_rect.min.x + 2.0, cam_rect.max.y - bar_h - 1.0),
                        Pos2::new(cam_rect.max.x - 2.0, cam_rect.max.y - 1.0),
                    );
                    painter.rect_filled(
                        bar_bg,
                        Rounding::same(2.0),
                        Color32::from_rgba_unmultiplied(15, 23, 42, 180),
                    );
                    let prog = app.gesture_state.right_hold_progress.clamp(0.05, 1.0);
                    let fill_rect = Rect::from_min_size(
                        bar_bg.min,
                        Vec2::new(bar_bg.width() * prog, bar_bg.height()),
                    );
                    let prog_col = if app.gesture_state.in_grace_period {
                        Color32::from_rgb(0xfe, 0xe4, 0x40)
                    } else {
                        Color32::from_rgb(0x00, 0xf5, 0xd4)
                    };
                    painter.rect_filled(fill_rect, Rounding::same(2.0), prog_col);
                }
            } else {
                draw_hud_pill(
                    &painter,
                    Pos2::new(cam_rect.min.x + 6.0, cam_rect.min.y + 6.0),
                    false,
                    Color32::from_rgba_unmultiplied(10, 15, 28, 198),
                    palette.border,
                    palette.text_muted,
                    "Standby",
                );
                let err_txt = app
                    .camera_error_message
                    .as_deref()
                    .unwrap_or("Chưa kết nối Webcam\n(Chọn Cam 0 / Cam 1 hoặc bật Camera)");
                painter.text(
                    cam_rect.center(),
                    Align2::CENTER_CENTER,
                    err_txt,
                    FontId::proportional(11.0),
                    palette.danger,
                );
            }

            painter.rect_stroke(
                cam_rect,
                Rounding::same(8.0),
                Stroke::new(1.2_f32, palette.border),
            );

            ui.add_space(5.0);

            // 1 single concise status row below Camera Preview
            let right_g = app.gesture_state.right_gesture;
            let status_str = if app.gesture_state.no_hand_idle {
                "Trạng thái: Chờ tay"
            } else {
                right_g.short_status()
            };
            let status_col = if app.gesture_state.no_hand_idle {
                palette.text_muted
            } else if app.gesture_state.in_grace_period {
                palette.warning
            } else {
                Color32::from_rgb(0x00, 0xc8, 0xb0)
            };
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(status_str)
                        .font(FontId::proportional(12.5))
                        .color(status_col)
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if app.camera_using_hardware {
                        let meta = if app.gesture_state.in_grace_period {
                            "Grace 150ms".to_string()
                        } else {
                            format!("L:{:.0} S:{:.0}", app.camera_mean_luma, app.camera_sharpness)
                        };
                        ui.label(
                            RichText::new(meta)
                                .font(FontId::monospace(10.0))
                                .color(palette.text_muted),
                        );
                    }
                });
            });

            ui.add_space(6.0);

            // =================================================================
            // 2. CAMERA SELECTOR (Cam 0 / Cam 1) + BACKLIGHT TOGGLE (Auto-HDR)
            // =================================================================
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 6.0;
                for cam_idx in [0u32, 1u32] {
                    let selected = app.active_camera_index == cam_idx;
                    let fill = if selected {
                        palette.primary
                    } else {
                        palette.bg_panel
                    };
                    let btn_txt_col = if selected {
                        Color32::WHITE
                    } else {
                        palette.text_primary
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(format!("Cam {cam_idx}"))
                                    .font(FontId::proportional(12.0))
                                    .color(btn_txt_col)
                                    .strong(),
                            )
                            .fill(fill)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(6.0))
                            .min_size(Vec2::new(54.0, 26.0)),
                        )
                        .clicked()
                    {
                        app.active_camera_index = cam_idx;
                        app.config.camera_index = cam_idx;
                        let _ = app
                            .camera
                            .cmd_tx
                            .try_send(CameraControlCmd::SelectCameraIndex(cam_idx));
                        app.storage.save_config(&app.config);
                    }
                }

                if ui
                    .checkbox(&mut app.config.enable_clahe, "Auto-HDR")
                    .on_hover_text(
                        "Tự động Adaptive Gamma + CLAHE + Chống chói & Làm nét USM (Threshold Coring)",
                    )
                    .changed()
                {
                    let _ = app
                        .camera
                        .cmd_tx
                        .try_send(CameraControlCmd::SetSharpnessConfig {
                            enable_clahe: app.config.enable_clahe,
                            clahe_clip_limit: 2.5,
                            unsharp_amount: app.config.unsharp_amount,
                        });
                    app.storage.save_config(&app.config);
                }
            });

            ui.add_space(8.0);
            ui.separator();
            ui.add_space(6.0);

            // =================================================================
            // 3. 4 VECTOR-ICON GESTURE CARDS (2x2 GRID)
            // =================================================================
            let hand_active = !app.gesture_state.no_hand_idle;
            let badges = [
                (
                    RightGesture::DrawStroke,
                    hand_active && matches!(right_g, RightGesture::DrawStroke),
                    "1 Ngón: Vẽ",
                    Color32::from_rgb(0x00, 0xf5, 0xd4),
                ),
                (
                    RightGesture::HoverTwoFingers,
                    hand_active && matches!(right_g, RightGesture::HoverTwoFingers),
                    "2 Ngón: Di",
                    Color32::from_rgb(0xfe, 0xe4, 0x40),
                ),
                (
                    RightGesture::PinchToolSelect,
                    hand_active && matches!(right_g, RightGesture::PinchToolSelect),
                    "Chụm: Click",
                    palette.secondary,
                ),
                (
                    RightGesture::PalmEraser,
                    hand_active && matches!(right_g, RightGesture::PalmEraser),
                    "5 Ngón: Tẩy",
                    palette.danger,
                ),
            ];

            for pair in badges.chunks(2) {
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    for &(gesture_kind, is_on, text, active_col) in pair {
                        let fill = if is_on {
                            active_col
                        } else {
                            palette.bg_panel
                        };
                        let txt_col = if is_on {
                            Color32::from_rgb(15, 23, 42)
                        } else {
                            palette.text_primary
                        };
                        let icon_col = if is_on {
                            Color32::from_rgb(15, 23, 42)
                        } else {
                            active_col
                        };
                        let border = if is_on { active_col } else { palette.border };
                        egui::Frame::none()
                            .fill(fill)
                            .stroke(Stroke::new(1.0_f32, border))
                            .rounding(Rounding::same(6.0))
                            .inner_margin(egui::Margin::symmetric(5.0, 5.0))
                            .show(ui, |ui| {
                                ui.set_width(110.0);
                                ui.horizontal(|ui| {
                                    ui.spacing_mut().item_spacing.x = 5.0;
                                    let (icon_rect, _) = ui.allocate_exact_size(
                                        Vec2::new(18.0, 18.0),
                                        Sense::hover(),
                                    );
                                    draw_gesture_hand_icon(
                                        &ui.painter_at(icon_rect),
                                        icon_rect,
                                        gesture_kind,
                                        icon_col,
                                    );
                                    ui.label(
                                        RichText::new(text)
                                            .font(FontId::proportional(11.0))
                                            .color(txt_col)
                                            .strong(),
                                    );
                                });
                            });
                    }
                });
                ui.add_space(5.0);
            }

            ui.add_space(6.0);
            ui.separator();
            ui.add_space(6.0);

            if ui
                .add(
                    egui::Button::new(
                        RichText::new("⚙ Hiệu chỉnh nhanh (Raw vs HDR)")
                            .font(FontId::proportional(11.5))
                            .strong()
                            .color(palette.text_primary),
                    )
                    .fill(palette.surface_elevated)
                    .stroke(Stroke::new(1.0_f32, palette.secondary))
                    .rounding(Rounding::same(6.0))
                    .min_size(Vec2::new(250.0, 26.0)),
                )
                .clicked()
            {
                app.set_calibration_modal_open(true);
            }

            ui.add_space(6.0);

            // =================================================================
            // 4. COMPACT MIC VOLUME BAR (VU METER) AT BOTTOM
            // =================================================================
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Mic VU Meter")
                        .font(FontId::proportional(12.0))
                        .color(palette.text_primary)
                        .strong(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let db = app.audio_recorder.current_vu_db();
                    ui.label(
                        RichText::new(format!("{db:.1} dB"))
                            .font(FontId::proportional(11.5))
                            .color(palette.secondary)
                            .strong(),
                    );
                });
            });

            ui.add_space(3.0);
            let vu_norm = app.audio_recorder.current_vu_normalized();
            let (vu_rect, _) = ui.allocate_exact_size(Vec2::new(250.0, 12.0), Sense::hover());
            let vu_painter = ui.painter_at(vu_rect);
            vu_painter.rect_filled(vu_rect, Rounding::same(6.0), palette.bg_canvas);
            let fill_w = (vu_rect.width() * vu_norm).clamp(2.0, vu_rect.width());
            let vu_col = if vu_norm > 0.80 {
                palette.danger
            } else if vu_norm > 0.45 {
                palette.warning
            } else {
                palette.success
            };
            vu_painter.rect_filled(
                Rect::from_min_size(vu_rect.min, Vec2::new(fill_w, vu_rect.height())),
                Rounding::same(6.0),
                vu_col,
            );
            vu_painter.rect_stroke(
                vu_rect,
                Rounding::same(6.0),
                Stroke::new(1.0_f32, palette.border),
            );
        });
}

/// MODULE 2.3 & MODULE 4: Clean Center Smartboard Canvas (Zero Clutter, 80% Screen Area).
pub fn render_lecture_canvas(app: &mut AirTeachApp, ctx: &egui::Context) {
    let palette = app.config.theme_mode.palette();
    let bg_rgb = app.config.contrast_mode.canvas_bg_rgb();
    let canvas_bg_color = Color32::from_rgb(bg_rgb[0], bg_rgb[1], bg_rgb[2]);
    let is_light_board = app.config.contrast_mode == ClassroomContrastMode::WhiteboardClean;

    egui::CentralPanel::default()
        .frame(egui::Frame::none().fill(canvas_bg_color))
        .show(ctx, |ui| {
            let canvas_rect = ui.available_rect_before_wrap();
            let response = ui.allocate_rect(canvas_rect, Sense::click_and_drag());
            app.last_canvas_rect = Some(canvas_rect);

            handle_pointer_interaction(app, &response, canvas_rect);

            let painter = ui.painter_at(canvas_rect);

            // =========================================================================
            // LAYER 1: SINGLE CARO GRID OVERLAY
            // =========================================================================
            if app.config.canvas_grid_enabled {
                let base_step = app.config.canvas_grid_px.clamp(10, 60) as f32;
                let step = (base_step * app.canvas_zoom).clamp(8.0, 220.0);
                let major_step = step * 5.0;

                let (minor_stroke, major_stroke) = if is_light_board {
                    (
                        Stroke::new(
                            1.0_f32,
                            Color32::from_rgba_unmultiplied(0x64, 0x74, 0x8b, 42),
                        ),
                        Stroke::new(
                            1.2_f32,
                            Color32::from_rgba_unmultiplied(0x4f, 0x46, 0xe5, 78),
                        ),
                    )
                } else {
                    (
                        Stroke::new(
                            1.0_f32,
                            Color32::from_rgba_unmultiplied(0x4f, 0x46, 0xe5, 32),
                        ),
                        Stroke::new(
                            1.0_f32,
                            Color32::from_rgba_unmultiplied(0x06, 0xb6, 0xd4, 62),
                        ),
                    )
                };

                let offset_x = app.canvas_pan.x.rem_euclid(step);
                let mut x = canvas_rect.min.x + offset_x;
                let mut col_idx = 0usize;
                while x <= canvas_rect.max.x {
                    let is_major = ((x - canvas_rect.min.x - app.canvas_pan.x).abs() % major_step)
                        < 1.5
                        || col_idx % 5 == 0;
                    painter.line_segment(
                        [
                            Pos2::new(x, canvas_rect.min.y),
                            Pos2::new(x, canvas_rect.max.y),
                        ],
                        if is_major { major_stroke } else { minor_stroke },
                    );
                    x += step;
                    col_idx += 1;
                }

                let offset_y = app.canvas_pan.y.rem_euclid(step);
                let mut y = canvas_rect.min.y + offset_y;
                let mut row_idx = 0usize;
                while y <= canvas_rect.max.y {
                    let is_major = ((y - canvas_rect.min.y - app.canvas_pan.y).abs() % major_step)
                        < 1.5
                        || row_idx % 5 == 0;
                    painter.line_segment(
                        [
                            Pos2::new(canvas_rect.min.x, y),
                            Pos2::new(canvas_rect.max.x, y),
                        ],
                        if is_major { major_stroke } else { minor_stroke },
                    );
                    y += step;
                    row_idx += 1;
                }
            }

            // =========================================================================
            // LAYER 2: CLEAN SLIDE CONTENT CARD (OPTIONAL, TOP-LEFT)
            // =========================================================================
            if app.slide_card_visible {
                app.ensure_slide_texture_loaded(ctx);
                let palette = app.config.theme_mode.palette();
                let use_light_card = is_light_board || app.config.theme_mode == ThemeMode::Light;
                let slide_rect =
                    compute_slide_screen_rect(canvas_rect, app.canvas_pan, app.canvas_zoom);

                let is_external_media = app
                    .slides
                    .get(app.current_slide_idx)
                    .map(|s| s.subject_tag == "PDF" || s.subject_tag == "IMG")
                    .unwrap_or(false);

                let card_bg = if use_light_card {
                    Color32::from_rgba_unmultiplied(
                        palette.surface.r(),
                        palette.surface.g(),
                        palette.surface.b(),
                        238,
                    )
                } else {
                    Color32::from_rgba_unmultiplied(15, 23, 42, 205)
                };
                let card_border = if use_light_card {
                    palette.border
                } else {
                    palette.border
                };
                let card_title_col = if use_light_card {
                    palette.primary
                } else {
                    palette.secondary
                };
                let card_sub_col = palette.text_muted;
                let card_body_col = palette.text_primary;

                if is_external_media {
                    if let Some(ref slide_tex) = app.current_slide_texture {
                        painter.image(
                            slide_tex.id(),
                            slide_rect,
                            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                            Color32::WHITE,
                        );
                    }
                } else {
                    painter.rect_filled(slide_rect, Rounding::same(12.0), card_bg);
                }

                painter.rect_stroke(
                    slide_rect,
                    Rounding::same(12.0),
                    Stroke::new(1.2_f32, card_border),
                );

                if let Some(slide) = app.slides.get(app.current_slide_idx) {
                    let scale = (slide_rect.width() / 920.0).clamp(0.45, 2.5);
                    let title_pos = Pos2::new(
                        slide_rect.min.x + 24.0 * scale,
                        slide_rect.min.y + 16.0 * scale,
                    );
                    painter.text(
                        title_pos,
                        Align2::LEFT_TOP,
                        &DocumentLoader::format_latex_to_unicode_math(&slide.title),
                        FontId::proportional((20.0 * scale).clamp(11.0, 40.0)),
                        card_title_col,
                    );

                    let sub_pos = Pos2::new(
                        slide_rect.min.x + 24.0 * scale,
                        slide_rect.min.y + 44.0 * scale,
                    );
                    painter.text(
                        sub_pos,
                        Align2::LEFT_TOP,
                        &DocumentLoader::format_latex_to_unicode_math(&slide.subtitle),
                        FontId::proportional((13.0 * scale).clamp(9.0, 26.0)),
                        card_sub_col,
                    );

                    for (i, bullet) in slide.bullet_points.iter().enumerate() {
                        let by = slide_rect.min.y + (88.0 + i as f32 * 34.0) * scale;
                        painter.text(
                            Pos2::new(slide_rect.min.x + 28.0 * scale, by),
                            Align2::LEFT_TOP,
                            &DocumentLoader::format_latex_to_unicode_math(bullet),
                            FontId::proportional((14.0 * scale).clamp(9.5, 28.0)),
                            card_body_col,
                        );
                    }

                    if !slide.key_formula.is_empty() {
                        let clean_formula =
                            DocumentLoader::format_latex_to_unicode_math(&slide.key_formula);
                        let formula_rect = Rect::from_min_size(
                            Pos2::new(
                                slide_rect.min.x + 24.0 * scale,
                                slide_rect.min.y + 210.0 * scale,
                            ),
                            Vec2::new(
                                (slide_rect.width() - 48.0 * scale).max(120.0),
                                50.0 * scale,
                            ),
                        );
                        let formula_bg = if use_light_card {
                            palette.surface_elevated
                        } else {
                            Color32::from_rgba_unmultiplied(0x0f, 0x17, 0x2a, 235)
                        };
                        painter.rect_filled(formula_rect, Rounding::same(8.0), formula_bg);
                        painter.rect_stroke(
                            formula_rect,
                            Rounding::same(8.0),
                            Stroke::new(1.2_f32, palette.primary),
                        );
                        painter.text(
                            formula_rect.center(),
                            Align2::CENTER_CENTER,
                            &clean_formula,
                            FontId::proportional((14.5 * scale).clamp(10.0, 26.0)),
                            palette.secondary,
                        );
                    }
                }
            }

            // =========================================================================
            // LAYER 3: VECTOR INK STROKES
            // =========================================================================
            let pan = app.canvas_pan;
            let zoom = app.canvas_zoom;
            let origin = canvas_rect.min;

            for stroke in &app.strokes {
                if app.replay_active && stroke.timestamp_ms > app.replay_cursor_ms {
                    continue;
                }
                app.render_scratch_pos2.clear();
                for pt in &stroke.display_points {
                    app.render_scratch_pos2.push(Pos2::new(
                        origin.x + pan.x + pt.x * zoom,
                        origin.y + pan.y + pt.y * zoom,
                    ));
                }

                let c = Color32::from_rgba_unmultiplied(
                    stroke.color_rgba[0],
                    stroke.color_rgba[1],
                    stroke.color_rgba[2],
                    stroke.color_rgba[3],
                );
                let width = (stroke.thickness * zoom).clamp(1.0, 24.0);

                if app.render_scratch_pos2.len() == 1 {
                    painter.circle_filled(app.render_scratch_pos2[0], width * 0.5, c);
                } else if app.render_scratch_pos2.len() >= 2 {
                    for pair in app.render_scratch_pos2.windows(2) {
                        painter.line_segment([pair[0], pair[1]], Stroke::new(width, c));
                    }
                }
            }

            if !app.active_stroke_raw.is_empty() {
                app.render_scratch_pos2.clear();
                for pt in &app.active_stroke_raw {
                    app.render_scratch_pos2.push(Pos2::new(
                        origin.x + pan.x + pt.x * zoom,
                        origin.y + pan.y + pt.y * zoom,
                    ));
                }
                let c = Color32::from_rgba_unmultiplied(
                    app.current_ink_color[0],
                    app.current_ink_color[1],
                    app.current_ink_color[2],
                    app.current_ink_color[3],
                );
                let width = (app.current_stroke_thickness * zoom).clamp(1.0, 24.0);
                for pair in app.render_scratch_pos2.windows(2) {
                    painter.line_segment([pair[0], pair[1]], Stroke::new(width, c));
                }
            }

            // =========================================================================
            // LAYER 4: SELECTION BOUNDING BOX ("PHÂN TÍCH VÙNG")
            // =========================================================================
            if let Some((min_b, max_b)) = app.selection_rect_board {
                let s_min = Pos2::new(
                    origin.x + pan.x + min_b.x * zoom,
                    origin.y + pan.y + min_b.y * zoom,
                );
                let s_max = Pos2::new(
                    origin.x + pan.x + max_b.x * zoom,
                    origin.y + pan.y + max_b.y * zoom,
                );
                let sel_rect = Rect::from_two_pos(s_min, s_max);
                painter.rect_filled(
                    sel_rect,
                    Rounding::same(8.0),
                    Color32::from_rgba_unmultiplied(0x4f, 0x46, 0xe5, 36),
                );
                painter.rect_stroke(
                    sel_rect,
                    Rounding::same(8.0),
                    Stroke::new(2.2_f32, palette.danger),
                );
            }

            // =========================================================================
            // LAYER 5: LIVE TARGET CURSOR OVERLAY
            // =========================================================================
            if let Some(cursor_board_pt) = app.laser_pointer_board {
                let cp = Pos2::new(
                    origin.x + pan.x + cursor_board_pt.x * zoom,
                    origin.y + pan.y + cursor_board_pt.y * zoom,
                );
                let right_g = app.gesture_state.right_gesture;
                let right_mouse_down = ctx.input(|i| i.pointer.secondary_down());
                let hand_erasing =
                    !app.gesture_state.no_hand_idle && matches!(right_g, RightGesture::PalmEraser);
                let is_erasing = app.active_tool == ActiveBoardTool::Eraser
                    || right_mouse_down
                    || hand_erasing;
                let is_drawing = !app.active_stroke_raw.is_empty()
                    || (!app.gesture_state.no_hand_idle
                        && matches!(right_g, RightGesture::DrawStroke));
                let is_laser = app.active_tool == ActiveBoardTool::LaserPointer
                    || (!app.gesture_state.no_hand_idle
                        && matches!(right_g, RightGesture::LaserPointer));

                if is_erasing {
                    let erase_r = (45.0 * zoom).clamp(24.0, 90.0);
                    painter.circle_filled(
                        cp,
                        erase_r,
                        Color32::from_rgba_unmultiplied(0xf4, 0x3f, 0x5e, 55),
                    );
                    painter.circle_stroke(cp, erase_r, Stroke::new(2.0_f32, palette.danger));
                    let cross_col = if is_light_board {
                        Color32::from_rgb(0x0f, 0x17, 0x2a)
                    } else {
                        TEXT_LIGHT
                    };
                    painter.line_segment(
                        [Pos2::new(cp.x - 8.0, cp.y), Pos2::new(cp.x + 8.0, cp.y)],
                        Stroke::new(1.5_f32, cross_col),
                    );
                    painter.line_segment(
                        [Pos2::new(cp.x, cp.y - 8.0), Pos2::new(cp.x, cp.y + 8.0)],
                        Stroke::new(1.5_f32, cross_col),
                    );
                } else if is_drawing {
                    let cyan = Color32::from_rgb(0x00, 0xf5, 0xd4);
                    painter.circle_filled(
                        cp,
                        12.0,
                        Color32::from_rgba_unmultiplied(0x00, 0xf5, 0xd4, 70),
                    );
                    painter.circle_stroke(cp, 8.0, Stroke::new(2.0_f32, cyan));
                    painter.circle_filled(cp, 4.0, cyan);
                } else if is_laser {
                    painter.circle_filled(
                        cp,
                        18.0,
                        Color32::from_rgba_unmultiplied(0xf4, 0x3f, 0x5e, 60),
                    );
                    painter.circle_filled(
                        cp,
                        10.0,
                        Color32::from_rgba_unmultiplied(0xf4, 0x3f, 0x5e, 150),
                    );
                    painter.circle_filled(cp, 4.5, Color32::WHITE);
                } else if !app.gesture_state.no_hand_idle || response.hovered() {
                    let hover_yellow = Color32::from_rgb(0xfe, 0xe4, 0x40);
                    painter.circle_stroke(cp, 10.0, Stroke::new(2.0_f32, hover_yellow));
                    painter.circle_stroke(cp, 3.5, Stroke::new(1.2_f32, Color32::WHITE));
                }
            }

            // =========================================================================
            // LAYER 6: TOP-LEFT SLIDE VISIBILITY + 1-CLICK TUTORIAL HIDE/DELETE BAR
            // =========================================================================
            let active_is_tutorial = app
                .slides
                .get(app.current_slide_idx)
                .map(|s| s.is_tutorial)
                .unwrap_or(false);
            let bar_w: f32 = if active_is_tutorial { 620.0 } else { 300.0 };
            let toggle_rect = Rect::from_min_size(
                Pos2::new(canvas_rect.min.x + 14.0, canvas_rect.min.y + 10.0),
                Vec2::new(bar_w.min(canvas_rect.width() - 28.0), 30.0),
            );
            ui.allocate_new_ui(egui::UiBuilder::new().max_rect(toggle_rect), |ui| {
                let palette = app.config.theme_mode.palette();
                ui.horizontal(|ui| {
                    let btn_label = if app.slide_card_visible {
                        "[Ẩn nội dung Slide để lấy bảng trống 100%]"
                    } else {
                        "[Hiện nội dung Slide bài giảng]"
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(btn_label)
                                    .font(FontId::proportional(12.0))
                                    .color(palette.secondary)
                                    .strong(),
                            )
                            .fill(palette.surface)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(6.0)),
                        )
                        .clicked()
                    {
                        app.slide_card_visible = !app.slide_card_visible;
                    }

                    if active_is_tutorial {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("[Ẩn hướng dẫn]")
                                        .font(FontId::proportional(11.5))
                                        .color(palette.text_muted),
                                )
                                .fill(palette.surface)
                                .stroke(Stroke::new(1.0_f32, palette.border))
                                .rounding(Rounding::same(6.0)),
                            )
                            .clicked()
                        {
                            app.dismiss_tutorial_slides(false);
                        }

                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("[Xóa vĩnh viễn HD]")
                                        .font(FontId::proportional(11.5))
                                        .color(palette.danger),
                                )
                                .fill(palette.surface)
                                .stroke(Stroke::new(1.0_f32, palette.danger))
                                .rounding(Rounding::same(6.0)),
                            )
                            .clicked()
                        {
                            app.dismiss_tutorial_slides(true);
                        }
                    }
                });
            });
        });
}

fn compute_slide_screen_rect(canvas_rect: Rect, pan: Vec2, zoom: f32) -> Rect {
    let z = zoom.clamp(0.4, 2.8);
    let top_margin = 46.0_f32;
    let left_margin = 16.0_f32;
    let max_avail_w = (canvas_rect.width() - left_margin * 2.0).max(320.0);
    let max_avail_h = (canvas_rect.height() - top_margin - 24.0).max(200.0);

    let mut base_w = max_avail_w.min(760.0);
    let mut base_h = base_w * (9.0 / 16.0);
    if base_h > max_avail_h {
        base_h = max_avail_h;
        base_w = base_h * (16.0 / 9.0);
    }

    let w = base_w * z;
    let h = base_h * z;
    let min_x = canvas_rect.min.x + left_margin + pan.x;
    let min_y = canvas_rect.min.y + top_margin + pan.y;
    Rect::from_min_size(Pos2::new(min_x, min_y), Vec2::new(w, h))
}

fn handle_pointer_interaction(
    app: &mut AirTeachApp,
    response: &egui::Response,
    canvas_rect: Rect,
) {
    let origin = canvas_rect.min;
    let pan = app.canvas_pan;
    let zoom = app.canvas_zoom;

    if response.dragged_by(egui::PointerButton::Middle)
        || (response.dragged() && response.ctx.input(|i| i.modifiers.shift))
    {
        app.canvas_pan += response.drag_delta();
        return;
    }

    let scroll_y = response.ctx.input(|i| i.smooth_scroll_delta.y);
    if response.hovered() && scroll_y.abs() > 0.1 {
        let factor = (1.0 + scroll_y * 0.0015).clamp(0.85, 1.18);
        app.canvas_zoom = (app.canvas_zoom * factor).clamp(0.4, 2.8);
    }

    if let Some(hover_pos) = response.hover_pos() {
        let board_pt = Point2D::new(
            (hover_pos.x - origin.x - pan.x) / zoom,
            (hover_pos.y - origin.y - pan.y) / zoom,
        );
        app.laser_pointer_board = Some(board_pt);
    }

    let Some(pointer_pos) = response.interact_pointer_pos() else {
        if response.drag_stopped() {
            if app.active_tool == ActiveBoardTool::SelectRegion
                && app.selection_rect_board.is_some()
            {
                app.inspect_selection_with_gemini();
            } else {
                app.finish_pointer_gesture();
            }
        }
        return;
    };

    let board_pt = Point2D::new(
        (pointer_pos.x - origin.x - pan.x) / zoom,
        (pointer_pos.y - origin.y - pan.y) / zoom,
    );
    app.laser_pointer_board = Some(board_pt);

    // Right-Click / Right-Drag acts as an instant Spot Eraser (r = 45px)
    if response.dragged_by(egui::PointerButton::Secondary)
        || response.clicked_by(egui::PointerButton::Secondary)
        || response.ctx.input(|i| i.pointer.secondary_down())
    {
        if !app.active_stroke_raw.is_empty() {
            app.finish_pointer_gesture();
        }
        let erase_r = 45.0 / zoom.max(0.5);
        app.strokes.retain(|s| {
            !s.display_points
                .iter()
                .any(|p| p.dist(board_pt) < erase_r)
        });
        return;
    }

    match app.active_tool {
        ActiveBoardTool::Pen => {
            if response.drag_started_by(egui::PointerButton::Primary) {
                app.redo_strokes.clear();
                app.begin_board_stroke(board_pt);
            } else if response.dragged_by(egui::PointerButton::Primary) {
                app.append_board_stroke(board_pt);
            } else if response.drag_stopped_by(egui::PointerButton::Primary) {
                app.finish_pointer_gesture();
            }
        }
        ActiveBoardTool::LaserPointer => {}
        ActiveBoardTool::SelectRegion => {
            if response.drag_started_by(egui::PointerButton::Primary) {
                app.selection_anchor_board = Some(board_pt);
                app.selection_rect_board = Some((board_pt, board_pt));
            } else if response.dragged_by(egui::PointerButton::Primary) {
                if let Some(anchor) = app.selection_anchor_board {
                    let min_pt = Point2D::new(anchor.x.min(board_pt.x), anchor.y.min(board_pt.y));
                    let max_pt = Point2D::new(anchor.x.max(board_pt.x), anchor.y.max(board_pt.y));
                    app.selection_rect_board = Some((min_pt, max_pt));
                }
            } else if response.drag_stopped_by(egui::PointerButton::Primary) {
                app.inspect_selection_with_gemini();
            }
        }
        ActiveBoardTool::Eraser => {
            if response.dragged_by(egui::PointerButton::Primary)
                || response.clicked_by(egui::PointerButton::Primary)
            {
                let erase_r = 45.0 / zoom.max(0.5);
                app.strokes.retain(|s| {
                    !s.display_points
                        .iter()
                        .any(|p| p.dist(board_pt) < erase_r)
                });
            }
        }
    }
}

/// MODULE 3.3: Draw 21 hand connections in Neon Cyan (#00F5D4) & joints in Bright Yellow (#FEE440).
fn draw_hand_skeleton_on_rect(
    painter: &egui::Painter,
    rect: Rect,
    hand: &HandLandmarks,
) {
    let neon_cyan = Color32::from_rgb(0x00, 0xf5, 0xd4);
    let bright_yellow = Color32::from_rgb(0xfe, 0xe4, 0x40);

    for &(i, j) in &HAND_BONE_PAIRS {
        let a = hand.landmarks[i];
        let b = hand.landmarks[j];
        let pa = Pos2::new(
            rect.min.x + a.x * rect.width(),
            rect.min.y + a.y * rect.height(),
        );
        let pb = Pos2::new(
            rect.min.x + b.x * rect.width(),
            rect.min.y + b.y * rect.height(),
        );
        painter.line_segment([pa, pb], Stroke::new(2.2_f32, neon_cyan));
    }

    for (idx, lm) in hand.landmarks.iter().enumerate() {
        let pt = Pos2::new(
            rect.min.x + lm.x * rect.width(),
            rect.min.y + lm.y * rect.height(),
        );
        let r = if idx == 4 || idx == 8 || idx == 12 || idx == 16 || idx == 20 {
            4.0
        } else {
            2.6
        };
        painter.circle_filled(pt, r, bright_yellow);
    }
}

/// Compact HUD pill overlaid cleanly on camera preview corners.
fn draw_hud_pill(
    painter: &egui::Painter,
    anchor: Pos2,
    align_right: bool,
    bg: Color32,
    border: Color32,
    dot_and_text_col: Color32,
    label: &str,
) {
    let font = FontId::monospace(9.5);
    let galley = painter.layout_no_wrap(label.to_string(), font, Color32::WHITE);
    let pad_x = 6.0;
    let dot_space = 8.0;
    let w = galley.size().x + pad_x * 2.0 + dot_space;
    let h = 17.0;
    let min_x = if align_right { anchor.x - w } else { anchor.x };
    let rect = Rect::from_min_size(Pos2::new(min_x, anchor.y), Vec2::new(w, h));
    painter.rect_filled(rect, Rounding::same(9.0), bg);
    painter.rect_stroke(rect, Rounding::same(9.0), Stroke::new(1.0_f32, border));
    painter.circle_filled(
        Pos2::new(rect.min.x + pad_x + 2.5, rect.center().y),
        2.6,
        dot_and_text_col,
    );
    painter.galley(
        Pos2::new(
            rect.min.x + pad_x + dot_space,
            rect.center().y - galley.size().y * 0.5,
        ),
        galley,
        Color32::from_rgb(0xf1, 0xf5, 0xf9),
    );
}

/// Programmatic geometric hand-pose vector glyph inside gesture status cards.
fn draw_gesture_hand_icon(
    painter: &egui::Painter,
    rect: Rect,
    kind: RightGesture,
    color: Color32,
) {
    let cx = rect.center().x;
    let cy = rect.center().y;
    let u = rect.width().min(rect.height()) * 0.045;
    let wrist = Pos2::new(cx, cy + 5.2 * u);
    let palm_center = Pos2::new(cx, cy + 2.0 * u);

    // Compact rounded palm core
    painter.circle_filled(
        palm_center,
        3.6 * u,
        Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 55),
    );
    painter.circle_stroke(palm_center, 3.6 * u, Stroke::new(1.4_f32, color));

    let bone = Stroke::new(1.8_f32, color);
    let curl = Stroke::new(1.3_f32, Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 120));

    match kind {
        RightGesture::DrawStroke => {
            // Index straight tall up with active pen tip halo
            let base = Pos2::new(cx - 1.2 * u, cy - 1.0 * u);
            let tip = Pos2::new(cx - 1.2 * u, cy - 8.5 * u);
            painter.line_segment([wrist, base], curl);
            painter.line_segment([base, tip], bone);
            painter.circle_filled(tip, 2.2 * u, color);
            // Curled thumb & remaining 3 fingers
            painter.line_segment([Pos2::new(cx + 0.8 * u, cy - 0.8 * u), Pos2::new(cx + 2.6 * u, cy + 1.2 * u)], curl);
        }
        RightGesture::HoverTwoFingers => {
            // Index + Middle in crisp V-shape
            let idx_base = Pos2::new(cx - 1.6 * u, cy - 1.2 * u);
            let idx_tip = Pos2::new(cx - 3.6 * u, cy - 8.2 * u);
            let mid_base = Pos2::new(cx + 0.8 * u, cy - 1.4 * u);
            let mid_tip = Pos2::new(cx + 2.2 * u, cy - 8.5 * u);
            painter.line_segment([idx_base, idx_tip], bone);
            painter.line_segment([mid_base, mid_tip], bone);
            painter.circle_filled(idx_tip, 1.7 * u, color);
            painter.circle_filled(mid_tip, 1.7 * u, color);
        }
        RightGesture::PinchToolSelect => {
            // Thumb & index meeting at glowing pinch ring
            let pinch_pt = Pos2::new(cx - 3.6 * u, cy - 4.8 * u);
            painter.line_segment([Pos2::new(cx - 3.2 * u, cy + 2.0 * u), pinch_pt], bone);
            painter.line_segment([Pos2::new(cx - 0.6 * u, cy - 1.4 * u), pinch_pt], bone);
            painter.circle_stroke(pinch_pt, 2.3 * u, Stroke::new(1.5_f32, color));
            // Relaxed upper three fingers slightly arched
            painter.line_segment([Pos2::new(cx + 1.2 * u, cy - 1.0 * u), Pos2::new(cx + 3.2 * u, cy - 5.2 * u)], curl);
            painter.line_segment([Pos2::new(cx + 2.6 * u, cy), Pos2::new(cx + 5.0 * u, cy - 3.2 * u)], curl);
        }
        _ => {
            // Five splayed fingers (PalmEraser & default open hand)
            let rays = [
                (Pos2::new(cx - 3.0 * u, cy + 2.4 * u), Pos2::new(cx - 7.8 * u, cy - 2.0 * u)),
                (Pos2::new(cx - 2.2 * u, cy - 1.0 * u), Pos2::new(cx - 4.6 * u, cy - 7.6 * u)),
                (Pos2::new(cx, cy - 1.5 * u), Pos2::new(cx, cy - 8.8 * u)),
                (Pos2::new(cx + 2.0 * u, cy - 1.0 * u), Pos2::new(cx + 4.4 * u, cy - 7.2 * u)),
                (Pos2::new(cx + 3.2 * u, cy + 0.6 * u), Pos2::new(cx + 7.2 * u, cy - 4.2 * u)),
            ];
            for (b, t) in rays {
                painter.line_segment([b, t], bone);
                painter.circle_filled(t, 1.4 * u, color);
            }
        }
    }
}
