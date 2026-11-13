//! Live Camera & CV Calibration Modal (`Hiệu chỉnh nhanh`):
//! - Side-by-side comparison of **Before (Raw Webcam)** vs **After (Adaptive HDR + Tracking HUD)**
//! - Real-time tuning sliders for CLAHE, Coring Unsharp Mask, Quality Gate Sharpness,
//!   MediaPipe Hand Confidence, Gesture FSM Dwell (`5..=8`), and SFace 128-D Threshold (`0.50..=0.58`).

use crate::ai::face_auth::{
    SFACE_EMBEDDING_DIM, DEFAULT_FACE_MATCH_THRESHOLD, DEFAULT_SFACE_MATCH_THRESHOLD,
};
use crate::app::AirTeachApp;
use crate::core::camera::{CameraControlCmd, FACE_QUALITY_GATE_MIN_SHARPNESS};
use crate::core::hand_tracker::{HandLandmarks, HAND_CONNECTIONS};
use crate::ui::brand::ThemePalette;
use egui::{Color32, Pos2, Rect, RichText, Rounding, Stroke, Vec2};

pub fn render_calibration_modal(app: &mut AirTeachApp, ctx: &egui::Context) {
    if !app.calibration_modal_open {
        return;
    }

    let palette = app.config.theme_mode.palette();
    let mut is_open = app.calibration_modal_open;
    let mut close_requested = false;

    egui::Window::new("Hiệu chỉnh Camera & Nhận diện AI — DNT - Victer")
        .open(&mut is_open)
        .collapsible(false)
        .resizable(true)
        .default_width(880.0)
        .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
        .frame(
            egui::Frame::window(&ctx.style())
                .fill(palette.surface)
                .stroke(Stroke::new(1.5_f32, palette.border))
                .rounding(Rounding::same(12.0))
                .inner_margin(egui::Margin::same(16.0)),
        )
        .show(ctx, |ui| {
            // Top live diagnostics & quick reconnect bar
            egui::Frame::none()
                .fill(palette.bg_panel)
                .stroke(Stroke::new(1.0_f32, palette.border))
                .rounding(Rounding::same(8.0))
                .inner_margin(egui::Margin::symmetric(12.0, 8.0))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            RichText::new(format!(
                                "📷 {} ({})",
                                app.camera_device_name, app.camera_resolution_label
                            ))
                            .strong()
                            .color(palette.text_primary),
                        );
                        ui.separator();
                        ui.label(
                            RichText::new(format!("{:.1} FPS", app.camera_fps))
                                .small()
                                .color(palette.secondary),
                        );
                        ui.separator();
                        ui.label(
                            RichText::new(format!(
                                "Chế độ sáng: {} | Luma: {:.0}",
                                app.camera_lighting_mode.label(),
                                app.camera_mean_luma
                            ))
                            .small()
                            .color(palette.text_secondary),
                        );
                        ui.separator();
                        let gate_col = if app.camera_quality_gate_passed {
                            palette.success
                        } else {
                            palette.warning
                        };
                        ui.label(
                            RichText::new(format!(
                                "Độ nét: {:.1}/{:.0} ({})",
                                app.camera_sharpness,
                                app.config.quality_gate_min_sharpness,
                                if app.camera_quality_gate_passed {
                                    "ĐẠT"
                                } else {
                                    "MỜ"
                                }
                            ))
                            .small()
                            .strong()
                            .color(gate_col),
                        );

                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("🔄 Kết nối lại Camera")
                                            .small()
                                            .color(palette.text_primary),
                                    )
                                    .fill(palette.surface_elevated)
                                    .stroke(Stroke::new(1.0_f32, palette.border))
                                    .rounding(Rounding::same(6.0)),
                                )
                                .clicked()
                            {
                                let _ = app.camera.cmd_tx.try_send(CameraControlCmd::ReconnectHardware);
                            }
                        });
                    });
                });

            ui.add_space(10.0);

            // Side-by-side Before (Raw) vs After (Preprocessed + Vector HUD)
            ui.columns(2, |cols| {
                // LEFT: BEFORE (Raw Sensor Frame)
                egui::Frame::none()
                    .fill(palette.bg_panel)
                    .stroke(Stroke::new(1.0_f32, palette.border))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(egui::Margin::same(8.0))
                    .show(&mut cols[0], |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("TRƯỚC TIỀN XỬ LÝ (Raw Webcam)")
                                    .strong()
                                    .color(palette.text_secondary),
                            );
                        });
                        ui.add_space(4.0);
                        let (rect, _) =
                            ui.allocate_exact_size(Vec2::new(390.0, 220.0), egui::Sense::hover());
                        let painter = ui.painter_at(rect);
                        painter.rect_filled(rect, Rounding::same(6.0), palette.bg_canvas);

                        if let Some(ref raw_tex) = app.camera_raw_texture {
                            let draw_rect = fit_aspect_rect(rect, raw_tex.size_vec2());
                            painter.image(
                                raw_tex.id(),
                                draw_rect,
                                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                Color32::WHITE,
                            );
                        } else {
                            painter.text(
                                rect.center(),
                                egui::Align2::CENTER_CENTER,
                                "Đang chờ luồng Raw từ cảm biến...",
                                egui::FontId::proportional(13.0),
                                palette.text_muted,
                            );
                        }
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new("Ảnh cảm biến thô chưa qua Gamma / LAB CLAHE / Denoise.")
                                .small()
                                .color(palette.text_muted),
                        );
                    });

                // RIGHT: AFTER (Preprocessed + HUD Overlays)
                egui::Frame::none()
                    .fill(palette.bg_panel)
                    .stroke(Stroke::new(1.2_f32, palette.secondary))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(egui::Margin::same(8.0))
                    .show(&mut cols[1], |ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("SAU TIỀN XỬ LÝ (Adaptive HDR + CV HUD)")
                                    .strong()
                                    .color(palette.secondary),
                            );
                        });
                        ui.add_space(4.0);
                        let (rect, _) =
                            ui.allocate_exact_size(Vec2::new(390.0, 220.0), egui::Sense::hover());
                        let painter = ui.painter_at(rect);
                        painter.rect_filled(rect, Rounding::same(6.0), palette.bg_canvas);

                        if let Some(ref tex) = app.camera_texture {
                            let draw_rect = fit_aspect_rect(rect, tex.size_vec2());
                            painter.image(
                                tex.id(),
                                draw_rect,
                                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                                Color32::WHITE,
                            );
                            paint_calibration_overlays(app, &painter, draw_rect, palette);
                        } else {
                            painter.text(
                                rect.center(),
                                egui::Align2::CENTER_CENTER,
                                "Camera chưa bật hoặc đang khởi tạo...",
                                egui::FontId::proportional(13.0),
                                palette.text_muted,
                            );
                        }

                        ui.add_space(4.0);
                        let emb_badge = if app.last_live_face_vec.len() == SFACE_EMBEDDING_DIM {
                            "Vector 128-D SFace"
                        } else if !app.last_live_face_vec.is_empty() {
                            "Vector 512-D Dự phòng"
                        } else {
                            "Chưa có khuôn mặt"
                        };
                        let hand_cnt = [
                            app.last_two_hand_frame.left_hand.is_some(),
                            app.last_two_hand_frame.right_hand.is_some(),
                        ]
                        .into_iter()
                        .filter(|&x| x)
                        .count();
                        ui.label(
                            RichText::new(format!(
                                "Face: {emb_badge} | Tay phát hiện: {hand_cnt} | Inference: {:.1} ms",
                                app.last_two_hand_frame.inference_ms
                            ))
                            .small()
                            .color(palette.text_secondary),
                        );
                    });
            });

            ui.add_space(12.0);

            // Live parameter sliders
            let mut cv_changed = false;
            let mut sharpness_gate_changed = false;
            let mut hand_conf_changed = false;
            let mut fsm_changed = false;
            let mut config_changed = false;

            ui.columns(2, |cols| {
                // Column 1: Image Enhancement & Quality Gate
                egui::Frame::none()
                    .fill(palette.bg_panel)
                    .stroke(Stroke::new(1.0_f32, palette.border))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(egui::Margin::same(10.0))
                    .show(&mut cols[0], |ui| {
                        ui.label(
                            RichText::new("1. Tiền xử lý ánh sáng & Độ nét")
                                .strong()
                                .color(palette.primary),
                        );
                        ui.add_space(4.0);

                        if ui
                            .checkbox(
                                &mut app.config.enable_clahe,
                                "Bật Adaptive Gamma + LAB CLAHE + Denoise + Chống chói",
                            )
                            .changed()
                        {
                            cv_changed = true;
                            config_changed = true;
                        }

                        ui.horizontal(|ui| {
                            ui.label(RichText::new("CLAHE Clip Limit:").small().color(palette.text_secondary));
                            if ui
                                .add(
                                    egui::Slider::new(&mut app.config.clahe_clip_limit, 1.2..=5.0)
                                        .step_by(0.1),
                                )
                                .changed()
                            {
                                cv_changed = true;
                                config_changed = true;
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Coring Unsharp Mask:").small().color(palette.text_secondary));
                            if ui
                                .add(
                                    egui::Slider::new(&mut app.config.unsharp_amount, 0.0..=2.2)
                                        .step_by(0.05),
                                )
                                .changed()
                            {
                                cv_changed = true;
                                config_changed = true;
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Ngưỡng Quality Gate:").small().color(palette.text_secondary));
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut app.config.quality_gate_min_sharpness,
                                        15.0..=120.0,
                                    )
                                    .step_by(1.0),
                                )
                                .changed()
                            {
                                sharpness_gate_changed = true;
                                config_changed = true;
                            }
                        });
                    });

                // Column 2: Hand Tracking, FSM Dwell & SFace 128-D
                egui::Frame::none()
                    .fill(palette.bg_panel)
                    .stroke(Stroke::new(1.0_f32, palette.border))
                    .rounding(Rounding::same(8.0))
                    .inner_margin(egui::Margin::same(10.0))
                    .show(&mut cols[1], |ui| {
                        ui.label(
                            RichText::new("2. Ngưỡng Nhận diện Tay & Face SFace 128-D")
                                .strong()
                                .color(palette.secondary),
                        );
                        ui.add_space(4.0);

                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Tin cậy bàn tay (MediaPipe):").small().color(palette.text_secondary));
                            if ui
                                .add(
                                    egui::Slider::new(&mut app.config.hand_confidence_min, 0.35..=0.85)
                                        .step_by(0.02),
                                )
                                .changed()
                            {
                                hand_conf_changed = true;
                                config_changed = true;
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Số khung giữ cử chỉ (FSM):").small().color(palette.text_secondary));
                            if ui
                                .add(egui::Slider::new(&mut app.config.gesture_hold_frames, 5..=8))
                                .changed()
                            {
                                fsm_changed = true;
                                config_changed = true;
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Ngưỡng ArcFace / SFace:").small().color(palette.text_secondary));
                            if ui
                                .add(
                                    egui::Slider::new(&mut app.config.sface_match_threshold, 0.42..=0.68)
                                        .step_by(0.01),
                                )
                                .changed()
                            {
                                config_changed = true;
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Ngưỡng dự phòng 512-D:").small().color(palette.text_secondary));
                            if ui
                                .add(
                                    egui::Slider::new(&mut app.config.face_match_threshold, 0.65..=0.92)
                                        .step_by(0.01),
                                )
                                .changed()
                            {
                                config_changed = true;
                            }
                        });
                    });
            });

            if cv_changed {
                let _ = app.camera.cmd_tx.try_send(CameraControlCmd::SetSharpnessConfig {
                    enable_clahe: app.config.enable_clahe,
                    clahe_clip_limit: app.config.clahe_clip_limit,
                    unsharp_amount: app.config.unsharp_amount,
                });
            }
            if sharpness_gate_changed {
                let _ = app
                    .camera
                    .cmd_tx
                    .try_send(CameraControlCmd::SetQualityGateThreshold(
                        app.config.quality_gate_min_sharpness,
                    ));
            }
            if hand_conf_changed {
                let _ = app
                    .camera
                    .cmd_tx
                    .try_send(CameraControlCmd::SetHandConfidenceThreshold(
                        app.config.hand_confidence_min,
                    ));
            }
            if fsm_changed {
                app.gesture_fsm
                    .set_hold_frames(app.config.gesture_hold_frames);
            }
            if config_changed {
                app.storage.save_config(&app.config);
            }

            ui.add_space(10.0);

            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("Khôi phục mặc định tối ưu").color(palette.text_primary),
                        )
                        .fill(palette.surface_elevated)
                        .stroke(Stroke::new(1.0_f32, palette.border))
                        .rounding(Rounding::same(8.0)),
                    )
                    .clicked()
                {
                    app.config.enable_clahe = true;
                    app.config.clahe_clip_limit = 2.6;
                    app.config.unsharp_amount = 0.85;
                    app.config.quality_gate_min_sharpness = FACE_QUALITY_GATE_MIN_SHARPNESS;
                    app.config.hand_confidence_min = 0.45;
                    app.config.gesture_hold_frames = 6;
                    app.config.sface_match_threshold = DEFAULT_SFACE_MATCH_THRESHOLD;
                    app.config.face_match_threshold = DEFAULT_FACE_MATCH_THRESHOLD;

                    let _ = app.camera.cmd_tx.try_send(CameraControlCmd::SetSharpnessConfig {
                        enable_clahe: app.config.enable_clahe,
                        clahe_clip_limit: app.config.clahe_clip_limit,
                        unsharp_amount: app.config.unsharp_amount,
                    });
                    let _ = app
                        .camera
                        .cmd_tx
                        .try_send(CameraControlCmd::SetQualityGateThreshold(
                            app.config.quality_gate_min_sharpness,
                        ));
                    let _ = app
                        .camera
                        .cmd_tx
                        .try_send(CameraControlCmd::SetHandConfidenceThreshold(
                            app.config.hand_confidence_min,
                        ));
                    app.gesture_fsm
                        .set_hold_frames(app.config.gesture_hold_frames);
                    app.storage.save_config(&app.config);
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Đóng & Lưu hiệu chỉnh")
                                    .strong()
                                    .color(Color32::WHITE),
                            )
                            .fill(palette.primary)
                            .rounding(Rounding::same(8.0))
                            .min_size(Vec2::new(170.0, 34.0)),
                        )
                        .clicked()
                    {
                        close_requested = true;
                    }
                });
            });
        });

    if !is_open || close_requested {
        app.storage.save_config(&app.config);
        app.set_calibration_modal_open(false);
    }
}

fn fit_aspect_rect(container: Rect, tex_size: Vec2) -> Rect {
    let aspect = (tex_size.x / tex_size.y.max(1.0)).clamp(0.5, 2.5);
    let max_w = container.width();
    let max_h = container.height();
    let (draw_w, draw_h) = if max_w / max_h.max(1.0) > aspect {
        (max_h * aspect, max_h)
    } else {
        (max_w, max_w / aspect)
    };
    Rect::from_center_size(container.center(), Vec2::new(draw_w, draw_h))
}

fn paint_calibration_overlays(
    app: &AirTeachApp,
    painter: &egui::Painter,
    draw_rect: Rect,
    palette: ThemePalette,
) {
    if let Some([bx, by, bw, bh]) = app.last_face_bbox_norm {
        let face_rect = Rect::from_min_size(
            Pos2::new(
                draw_rect.left() + bx * draw_rect.width(),
                draw_rect.top() + by * draw_rect.height(),
            ),
            Vec2::new(bw * draw_rect.width(), bh * draw_rect.height()),
        );
        let col = if app.camera_quality_gate_passed {
            palette.success
        } else {
            palette.warning
        };
        painter.rect_stroke(face_rect, Rounding::same(4.0), Stroke::new(1.8_f32, col));
    }

    if let Some(pts5) = app.last_face_5pt_norm {
        for i in 0..5 {
            let px = draw_rect.left() + pts5[i * 2] * draw_rect.width();
            let py = draw_rect.top() + pts5[i * 2 + 1] * draw_rect.height();
            painter.circle_filled(Pos2::new(px, py), 2.8, palette.secondary);
        }
    }

    let draw_hand = |hand: &HandLandmarks, bone_col: Color32| {
        for &(a, b) in HAND_CONNECTIONS.iter() {
            let pa = Pos2::new(
                draw_rect.left() + hand.landmarks[a].x * draw_rect.width(),
                draw_rect.top() + hand.landmarks[a].y * draw_rect.height(),
            );
            let pb = Pos2::new(
                draw_rect.left() + hand.landmarks[b].x * draw_rect.width(),
                draw_rect.top() + hand.landmarks[b].y * draw_rect.height(),
            );
            painter.line_segment([pa, pb], Stroke::new(1.5_f32, bone_col));
        }
    };

    if let Some(ref lh) = app.last_two_hand_frame.left_hand {
        draw_hand(lh, palette.secondary);
    }
    if let Some(ref rh) = app.last_two_hand_frame.right_hand {
        draw_hand(rh, palette.primary);
    }
}
