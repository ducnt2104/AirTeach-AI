//! Modernized Settings View (`src/ui/settings_view.rs`):
//! - Google AI Studio Gemini API Key (AES-256-GCM encrypted in SQLite)
//! - MediaFoundation Camera Device selection & CLAHE / Unsharp Mask sharpness sliders
//! - Caro Grid Slider (10 - 60 px) for Canvas and Camera PIP HUD
//! - ONNX Model Downloader status & verification
//! - 1€ Filter, RDP & Shape Snapping parameters

use crate::ai::face_auth::FaceAuthEngine;
use crate::app::{AirTeachApp, AppScreen};
use crate::core::camera::CameraControlCmd;
use crate::ui::brand::{render_about_modal, ThemeMode, ThemePalette};
use crate::utils::model_downloader::{ModelDownloadState, REQUIRED_ONNX_MODELS};
use egui::{Color32, FontId, RichText, Rounding, ScrollArea, Stroke, Vec2};

pub fn render_settings_view(app: &mut AirTeachApp, ctx: &egui::Context) {
    let palette = app.config.theme_mode.palette();

    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(palette.bg_canvas)
                .inner_margin(egui::Margin::same(24.0)),
        )
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("[CÀI ĐẶT] HỆ THỐNG, CAMERA, GHI ÂM & TRỢ LÝ GEMINI AI")
                        .font(FontId::proportional(22.0))
                        .strong()
                        .color(palette.text_primary),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("[Quay lại Bảng giảng]")
                                    .color(Color32::WHITE)
                                    .strong(),
                            )
                            .fill(palette.primary)
                            .rounding(Rounding::same(8.0))
                            .min_size(Vec2::new(180.0, 34.0)),
                        )
                        .clicked()
                    {
                        app.screen = AppScreen::LectureMode;
                        if app.left_camera_dock_open {
                            app.ensure_camera_acquired();
                        }
                    }
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("[Giới thiệu]").color(palette.secondary),
                            )
                            .fill(palette.surface)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(8.0))
                            .min_size(Vec2::new(110.0, 34.0)),
                        )
                        .clicked()
                    {
                        app.about_modal_open = true;
                    }
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("[Trang chủ]").color(palette.text_primary),
                            )
                            .fill(palette.surface)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(8.0))
                            .min_size(Vec2::new(120.0, 34.0)),
                        )
                        .clicked()
                    {
                        app.screen = AppScreen::Dashboard;
                    }
                });
            });

            ui.add_space(16.0);

            ScrollArea::vertical().show(ui, |ui| {
                // 1. Google AI Studio / Gemini API Configuration
                section_card(
                    ui,
                    palette,
                    "[GEMINI API] Google Gemini Multimodal API (Mã hóa AES-256-GCM)",
                    |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Gemini API Key:").color(palette.text_muted));
                            ui.add(
                                egui::TextEdit::singleline(&mut app.api_key_edit_buffer)
                                    .password(!app.show_api_key_plain)
                                    .hint_text("Nhập Gemini API Key (Lưu mã hóa cục bộ AES-256-GCM)")
                                    .desired_width(380.0),
                            );
                            ui.checkbox(&mut app.show_api_key_plain, "Hiện");
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("[Mã hóa & Lưu Key]")
                                            .color(Color32::WHITE)
                                            .strong(),
                                    )
                                    .fill(palette.primary)
                                    .rounding(Rounding::same(8.0)),
                                )
                                .clicked()
                            {
                                app.storage
                                    .set_encrypted_api_key(&mut app.config, &app.api_key_edit_buffer);
                                app.settings_status_banner =
                                    "[OK] Đã mã hóa AES-256-GCM và lưu Gemini API Key vào SQLite."
                                        .to_string();
                            }
                        });

                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Mô hình Gemini:").color(palette.text_muted));
                            for model_id in
                                ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
                            {
                                if ui
                                    .selectable_value(
                                        &mut app.config.gemini_model,
                                        model_id.to_string(),
                                        model_id,
                                    )
                                    .clicked()
                                {
                                    app.storage.save_config(&app.config);
                                }
                            }
                        });
                    },
                );

                ui.add_space(12.0);

                // 2. Real MediaFoundation Camera & CLAHE / Unsharp Masking
                section_card(
                    ui,
                    palette,
                    "[CAMERA] Camera MediaFoundation (4:3 / 16:9 @ 30 FPS) & Tăng cường Độ nét",
                    |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Thiết bị Camera:").color(palette.text_muted));
                            for cam_idx in 0..=2u32 {
                                if ui
                                    .selectable_value(
                                        &mut app.config.camera_index,
                                        cam_idx,
                                        format!("Camera #{cam_idx}"),
                                    )
                                    .clicked()
                                {
                                    let _ = app
                                        .camera
                                        .cmd_tx
                                        .try_send(CameraControlCmd::SelectCameraIndex(cam_idx));
                                    app.storage.save_config(&app.config);
                                }
                            }
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("[Quét lại phần cứng]")
                                            .color(palette.secondary),
                                    )
                                    .fill(palette.bg_panel)
                                    .stroke(Stroke::new(1.0_f32, palette.border))
                                    .rounding(Rounding::same(8.0)),
                                )
                                .clicked()
                            {
                                let _ = app
                                    .camera
                                    .cmd_tx
                                    .try_send(CameraControlCmd::ReconnectHardware);
                            }

                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("[Mở Hiệu chỉnh nhanh (Raw vs HDR)]")
                                            .color(palette.text_primary)
                                            .strong(),
                                    )
                                    .fill(palette.surface_elevated)
                                    .stroke(Stroke::new(1.0_f32, palette.secondary))
                                    .rounding(Rounding::same(8.0)),
                                )
                                .clicked()
                            {
                                app.set_calibration_modal_open(true);
                            }

                            ui.label(
                                RichText::new(format!(
                                    "{} ({})",
                                    app.camera_device_name, app.camera_resolution_label
                                ))
                                .font(FontId::proportional(12.0))
                                .color(palette.secondary),
                            );
                        });

                        ui.add_space(8.0);
                        let mut sharpness_changed = false;
                        if ui
                            .checkbox(
                                &mut app.config.enable_clahe,
                                "Bật cân bằng tương phản thích nghi CLAHE (Contrast Limited Adaptive Histogram Equalization)",
                            )
                            .changed()
                        {
                            sharpness_changed = true;
                        }
                        ui.horizontal(|ui| {
                            ui.label("CLAHE Clip Limit:");
                            if ui
                                .add(
                                    egui::Slider::new(&mut app.config.clahe_clip_limit, 1.2..=5.0)
                                        .step_by(0.1),
                                )
                                .changed()
                            {
                                sharpness_changed = true;
                            }
                            ui.separator();
                            ui.label("Unsharp Mask Gain:");
                            if ui
                                .add(
                                    egui::Slider::new(&mut app.config.unsharp_amount, 0.0..=2.2)
                                        .step_by(0.05),
                                )
                                .changed()
                            {
                                sharpness_changed = true;
                            }
                        });

                        if sharpness_changed {
                            let _ = app.camera.cmd_tx.try_send(
                                CameraControlCmd::SetSharpnessConfig {
                                    enable_clahe: app.config.enable_clahe,
                                    clahe_clip_limit: app.config.clahe_clip_limit,
                                    unsharp_amount: app.config.unsharp_amount,
                                },
                            );
                            app.storage.save_config(&app.config);
                        }

                        ui.add_space(8.0);
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut app.config.canvas_grid_enabled, "Lưới Caro Bảng giảng");
                            ui.add(
                                egui::Slider::new(&mut app.config.canvas_grid_px, 10..=60)
                                    .text("px (Bảng)"),
                            );
                            ui.separator();
                            if ui
                                .checkbox(&mut app.config.hud_grid_enabled, "Lưới Caro Camera HUD")
                                .changed()
                                || ui
                                    .add(
                                        egui::Slider::new(&mut app.config.hud_grid_px, 10..=60)
                                            .text("px (HUD)"),
                                    )
                                    .changed()
                            {
                                let _ = app.camera.cmd_tx.try_send(CameraControlCmd::SetHudGrid {
                                    enabled: app.config.hud_grid_enabled,
                                    spacing_px: app.config.hud_grid_px,
                                });
                                app.storage.save_config(&app.config);
                            }
                        });
                    },
                );

                ui.add_space(12.0);

                // 3. ONNX Model Downloader & Verification Status
                section_card(
                    ui,
                    palette,
                    "[ONNX MODELS] Quản lý & Tải tự động Mô hình ONNX (MediaPipe, MobileFaceNet, RapidOCR, pix2tex)",
                    |ui| {
                        let dl_state = app.model_downloader.current_state();
                        ui.horizontal(|ui| {
                            let status_txt = match &dl_state {
                                ModelDownloadState::Checking => {
                                    "[Đang kiểm tra] Đang kiểm tra mô hình ONNX...".to_string()
                                }
                                ModelDownloadState::Downloading {
                                    current_model,
                                    completed_count,
                                    total_count,
                                    bytes_downloaded,
                                } => format!(
                                    "[Đang tải] {} ({}/{}) — {:.1} KB",
                                    current_model,
                                    completed_count + 1,
                                    total_count,
                                    *bytes_downloaded as f64 / 1024.0
                                ),
                                ModelDownloadState::Ready {
                                    verified_count,
                                    total_count,
                                } => format!(
                                    "[OK] Đã tải và xác thực {verified_count}/{total_count} mô hình ONNX"
                                ),
                                ModelDownloadState::PartialOffline {
                                    verified_count,
                                    total_count,
                                    note,
                                } => format!(
                                    "[Sẵn sàng] ({verified_count}/{total_count} bản tải trực tuyến; dự phòng cục bộ: {note})"
                                ),
                            };
                            ui.label(
                                RichText::new(status_txt)
                                    .color(palette.secondary)
                                    .strong(),
                            );
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("[Kiểm tra & Tải lại ONNX]")
                                            .color(Color32::WHITE),
                                    )
                                    .fill(palette.primary)
                                    .rounding(Rounding::same(8.0)),
                                )
                                .clicked()
                            {
                                app.model_downloader
                                    .ensure_models_async(&app.gemini_client.runtime);
                            }
                        });

                        ui.add_space(6.0);
                        for spec in &REQUIRED_ONNX_MODELS {
                            let p = crate::utils::model_downloader::ModelDownloader::model_path(
                                spec.filename,
                            );
                            let exists = p.exists();
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(if exists { "[OK]" } else { "[...]" }).color(
                                        if exists {
                                            Color32::from_rgb(34, 197, 94)
                                        } else {
                                            palette.warning
                                        },
                                    ),
                                );
                                ui.label(
                                    RichText::new(spec.display_name)
                                        .color(palette.text_primary)
                                        .strong(),
                                );
                                ui.label(
                                    RichText::new(format!("({})", p.display()))
                                        .font(FontId::proportional(11.5))
                                        .color(palette.text_muted),
                                );
                            });
                        }
                    },
                );

                ui.add_space(12.0);

                // 4. 1-Euro Filter, RDP & Shape Snapping
                section_card(
                    ui,
                    palette,
                    "[BỘ LỌC NÉT VẼ] Làm mượt Vector 1-Euro Filter, Ramer-Douglas-Peucker (RDP) & Nắn Hình",
                    |ui| {
                        let mut filter_changed = false;
                        ui.horizontal(|ui| {
                            ui.label("1-Euro Min Cutoff (Hz):");
                            if ui
                                .add(
                                    egui::Slider::new(
                                        &mut app.config.one_euro_min_cutoff,
                                        0.3..=4.0,
                                    )
                                    .step_by(0.05),
                                )
                                .changed()
                            {
                                filter_changed = true;
                            }
                            ui.separator();
                            ui.label("1-Euro Speed Beta:");
                            if ui
                                .add(
                                    egui::Slider::new(&mut app.config.one_euro_beta, 0.001..=0.08)
                                        .step_by(0.001),
                                )
                                .changed()
                            {
                                filter_changed = true;
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label("1-Euro Derivative Cutoff (Hz):");
                            if ui
                                .add(
                                    egui::Slider::new(&mut app.config.one_euro_d_cutoff, 0.5..=3.0)
                                        .step_by(0.1),
                                )
                                .changed()
                            {
                                filter_changed = true;
                            }
                            ui.separator();
                            ui.label("RDP Epsilon (px):");
                            if ui
                                .add(
                                    egui::Slider::new(&mut app.config.rdp_epsilon, 0.5..=6.0)
                                        .step_by(0.1),
                                )
                                .changed()
                            {
                                filter_changed = true;
                            }
                        });

                        if ui
                            .checkbox(
                                &mut app.config.auto_shape_snap,
                                "Tự động nắn nét vẽ tay thành hình học chuẩn (Đoạn thẳng, Hình tròn, Hình chữ nhật, Tam giác)",
                            )
                            .changed()
                        {
                            filter_changed = true;
                        }

                        if filter_changed {
                            app.sync_stroke_engine_config();
                            app.storage.save_config(&app.config);
                        }
                    },
                );

                ui.add_space(12.0);

                // 5. Giao diện, Nhận diện Face ID & Tài khoản Cục bộ (Argon2id)
                section_card(ui, palette, "[GIAO DIỆN & BẢO MẬT] Chủ đề Sáng/Tối, Ngưỡng Face ID & Mật khẩu Argon2id", |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label("Chủ đề Giao diện:");
                        if ui
                            .selectable_value(&mut app.config.theme_mode, ThemeMode::Dark, ThemeMode::Dark.label())
                            .clicked()
                            || ui
                                .selectable_value(&mut app.config.theme_mode, ThemeMode::Light, ThemeMode::Light.label())
                                .clicked()
                        {
                            app.config.theme_mode.palette().apply_to_egui(ctx);
                            app.storage.save_config(&app.config);
                        }

                        ui.separator();
                        ui.label("SFace 128-D (Multi-Template):");
                        if ui
                            .add(
                                egui::Slider::new(&mut app.config.sface_match_threshold, 0.50..=0.58)
                                    .step_by(0.01),
                            )
                            .changed()
                        {
                            app.storage.save_config(&app.config);
                        }

                        ui.separator();
                        ui.label("Dự phòng 512-D:");
                        if ui
                            .add(
                                egui::Slider::new(&mut app.config.face_match_threshold, 0.65..=0.92)
                                    .step_by(0.01),
                            )
                            .changed()
                        {
                            app.storage.save_config(&app.config);
                        }

                        ui.separator();
                        ui.label("Tin cậy bàn tay (MediaPipe):");
                        if ui
                            .add(
                                egui::Slider::new(&mut app.config.hand_confidence_min, 0.35..=0.85)
                                    .step_by(0.02),
                            )
                            .changed()
                        {
                            let _ = app
                                .camera
                                .cmd_tx
                                .try_send(CameraControlCmd::SetHandConfidenceThreshold(
                                    app.config.hand_confidence_min,
                                ));
                            app.storage.save_config(&app.config);
                        }
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        let teacher_lbl = if app.logged_in_teacher_name.is_empty() {
                            "Chưa đăng nhập".to_string()
                        } else {
                            format!("{} (@{})", app.logged_in_teacher_name, app.logged_in_username)
                        };
                        ui.label(RichText::new(format!("Tài khoản hiện tại: {teacher_lbl}")).color(palette.secondary).strong());

                        ui.separator();
                        ui.label("Đổi mật khẩu mới (>= 6 ký tự):");
                        ui.add(
                            egui::TextEdit::singleline(&mut app.new_pin_edit_buffer)
                                .password(true)
                                .desired_width(140.0),
                        );
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Cập nhật Mật khẩu").color(Color32::WHITE),
                                )
                                .fill(palette.primary)
                                .rounding(Rounding::same(8.0)),
                            )
                            .clicked()
                        {
                            let new_pw = app.new_pin_edit_buffer.clone();
                            match FaceAuthEngine::hash_password_argon2(&new_pw) {
                                Ok(hash) => {
                                    if let Some(ref uid) = app.logged_in_user_id.clone() {
                                        match app.storage.update_user_password(uid, &hash) {
                                            Ok(_) => {
                                                app.reload_registered_users();
                                                app.new_pin_edit_buffer.clear();
                                                app.settings_status_banner =
                                                    "[OK] Đã băm Argon2id và cập nhật mật khẩu vào SQLite."
                                                        .to_string();
                                            }
                                            Err(e) => {
                                                app.settings_status_banner = format!("[LỖI] {e}");
                                            }
                                        }
                                    } else {
                                        app.settings_status_banner =
                                            "[CẢNH BÁO] Hãy đăng nhập tài khoản trước khi đổi mật khẩu."
                                                .to_string();
                                    }
                                }
                                Err(err) => {
                                    app.settings_status_banner = format!("[LỖI] {err}");
                                }
                            }
                        }
                    });
                });

                if !app.settings_status_banner.is_empty() {
                    ui.add_space(10.0);
                    ui.label(
                        RichText::new(&app.settings_status_banner)
                            .color(palette.secondary)
                            .strong(),
                    );
                }
            });
        });

    let palette = app.config.theme_mode.palette();
    render_about_modal(ctx, &mut app.about_modal_open, palette);
}

fn section_card(
    ui: &mut egui::Ui,
    palette: ThemePalette,
    title: &str,
    add_contents: impl FnOnce(&mut egui::Ui),
) {
    egui::Frame::none()
        .fill(palette.surface)
        .rounding(Rounding::same(8.0))
        .stroke(Stroke::new(1.0_f32, palette.border))
        .inner_margin(egui::Margin::same(16.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                RichText::new(title)
                    .font(FontId::proportional(15.5))
                    .strong()
                    .color(palette.text_primary),
            );
            ui.add_space(8.0);
            add_contents(ui);
        });
}
