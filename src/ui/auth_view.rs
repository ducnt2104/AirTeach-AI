//! Modern 2-Step Sign-Up & Dual Sign-In View (`src/ui/auth_view.rs`):
//! - Step 1 Sign-Up: Unique username, display name, Argon2id password (>= 6 chars) validation
//! - Step 2 Sign-Up: Guided 3-to-5 slight head angle captures -> L2-normalized Average Embedding stored in SQLite `users`
//! - Sign-In: Default live Face ID (Cosine threshold + live bounding box feedback) with 1-click switch to Argon2id Password login
//! - Explicit Camera lifecycle & hardware/permission error UI + DNT - Victer corner branding

use crate::ai::face_auth::{
    ENROLL_ANGLE_PROMPTS, MAX_FACE_ENROLL_SAMPLES, MIN_FACE_ENROLL_SAMPLES,
};
use crate::app::{AirTeachApp, AppScreen};
use crate::core::camera::{CameraControlCmd, LightingAdaptationMode};
use crate::ui::brand::{
    render_about_modal, render_brand_corner_footer, ui_brand_logo, APP_BRAND_NAME, RADIUS_LG,
    RADIUS_MD, RADIUS_SM, SPACE_LG, SPACE_MD, SPACE_SM,
};
use egui::{Align2, Color32, FontId, Pos2, Rect, RichText, Rounding, Sense, Stroke, Vec2};

pub fn render_auth_view(app: &mut AirTeachApp, ctx: &egui::Context) {
    let palette = app.config.theme_mode.palette();

    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(palette.bg_canvas)
                .inner_margin(egui::Margin::same(SPACE_MD)),
        )
        .show(ctx, |ui| {
            // Top Brand Bar
            ui.horizontal(|ui| {
                ui_brand_logo(ui, 34.0);
                ui.add_space(SPACE_SM);
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(format!("{APP_BRAND_NAME} — Xác thực Cục bộ"))
                            .font(FontId::proportional(17.5))
                            .strong()
                            .color(palette.text_primary),
                    );
                    ui.label(
                        RichText::new("Bảo mật Local-First · SQLite + Argon2id + Face Embedding")
                            .font(FontId::proportional(11.5))
                            .color(palette.text_muted),
                    );
                });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Màn hình Chào")
                                    .font(FontId::proportional(12.0))
                                    .color(palette.text_primary),
                            )
                            .fill(palette.surface)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(RADIUS_SM))
                            .min_size(Vec2::new(112.0, 30.0)),
                        )
                        .clicked()
                    {
                        app.release_camera_hardware();
                        app.screen = AppScreen::Welcome;
                    }

                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(app.config.theme_mode.label())
                                    .font(FontId::proportional(12.0))
                                    .color(palette.secondary),
                            )
                            .fill(palette.surface)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(RADIUS_SM))
                            .min_size(Vec2::new(135.0, 30.0)),
                        )
                        .clicked()
                    {
                        app.toggle_theme_mode(ctx);
                    }
                });
            });

            ui.add_space(SPACE_SM);

            // Tab Switcher: [Đăng ký Tài khoản mới] | [Đăng nhập]
            ui.vertical_centered(|ui| {
                ui.horizontal(|ui| {
                    let total_w = 560.0_f32.min(ui.available_width() - 24.0);
                    let pad = ((ui.available_width() - total_w) * 0.5).max(0.0);
                    ui.add_space(pad);

                    let tab1 = app.auth_tab_is_signup;
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("ĐĂNG KÝ TÀI KHOẢN MỚI")
                                    .font(FontId::proportional(13.5))
                                    .color(if tab1 {
                                        Color32::WHITE
                                    } else {
                                        palette.text_primary
                                    })
                                    .strong(),
                            )
                            .fill(if tab1 {
                                palette.primary
                            } else {
                                palette.surface
                            })
                            .stroke(Stroke::new(
                                1.2_f32,
                                if tab1 {
                                    palette.secondary
                                } else {
                                    palette.border
                                },
                            ))
                            .rounding(Rounding::same(RADIUS_SM))
                            .min_size(Vec2::new(265.0, 36.0)),
                        )
                        .clicked()
                    {
                        app.auth_tab_is_signup = true;
                        app.auth_error_msg.clear();
                        if app.signup_step == 2 {
                            app.ensure_camera_acquired();
                        } else {
                            app.release_camera_hardware();
                        }
                    }

                    ui.add_space(SPACE_SM);

                    let tab2 = !app.auth_tab_is_signup;
                    let login_label =
                        format!("ĐĂNG NHẬP ({} tài khoản)", app.registered_users.len());
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(login_label)
                                    .font(FontId::proportional(13.5))
                                    .color(if tab2 {
                                        Color32::WHITE
                                    } else {
                                        palette.text_primary
                                    })
                                    .strong(),
                            )
                            .fill(if tab2 {
                                palette.primary
                            } else {
                                palette.surface
                            })
                            .stroke(Stroke::new(
                                1.2_f32,
                                if tab2 {
                                    palette.secondary
                                } else {
                                    palette.border
                                },
                            ))
                            .rounding(Rounding::same(RADIUS_SM))
                            .min_size(Vec2::new(265.0, 36.0)),
                        )
                        .clicked()
                    {
                        app.auth_tab_is_signup = false;
                        app.auth_error_msg.clear();
                        if app.signin_face_mode {
                            app.ensure_camera_acquired();
                        } else {
                            app.release_camera_hardware();
                        }
                    }
                });

                ui.add_space(SPACE_MD);

                if app.auth_tab_is_signup {
                    render_signup_flow(app, ui, palette);
                } else {
                    render_signin_flow(app, ui, palette);
                }
            });

            ui.add_space(SPACE_SM);
            render_brand_corner_footer(ui, palette, &mut app.about_modal_open);
        });

    render_about_modal(ctx, &mut app.about_modal_open, palette);
}

fn render_signup_flow(
    app: &mut AirTeachApp,
    ui: &mut egui::Ui,
    palette: crate::ui::brand::ThemePalette,
) {
    let card_w = 880.0_f32.min(ui.available_width() - 24.0);
    egui::Frame::none()
        .fill(palette.surface)
        .stroke(Stroke::new(1.2_f32, palette.border))
        .rounding(Rounding::same(RADIUS_LG))
        .inner_margin(egui::Margin::same(SPACE_LG))
        .show(ui, |ui| {
            ui.set_width(card_w);

            // Step Progress Pills
            ui.horizontal(|ui| {
                let s1_col = if app.signup_step == 1 {
                    palette.secondary
                } else {
                    palette.success
                };
                let s2_col = if app.signup_step == 2 {
                    palette.secondary
                } else {
                    palette.text_muted
                };
                ui.label(
                    RichText::new("Bước 1: Tài khoản & Mật khẩu (Argon2id)")
                        .font(FontId::proportional(13.5))
                        .strong()
                        .color(s1_col),
                );
                ui.label(RichText::new("  →  ").color(palette.text_muted));
                ui.label(
                    RichText::new("Bước 2: Quét 3–5 Góc Khuôn Mặt (Tạo Vector Trung bình)")
                        .font(FontId::proportional(13.5))
                        .strong()
                        .color(s2_col),
                );
            });

            ui.add_space(SPACE_MD);
            ui.separator();
            ui.add_space(SPACE_MD);

            if app.signup_step == 1 {
                // =====================================================================
                // STEP 1: STRICT CREDENTIAL VALIDATION
                // =====================================================================
                ui.label(
                    RichText::new("Tạo Hồ sơ Giảng viên Cục bộ")
                        .font(FontId::proportional(17.0))
                        .strong()
                        .color(palette.text_primary),
                );
                ui.label(
                    RichText::new(
                        "Tất cả thông tin được lưu bảo mật trong SQLite cục bộ. Mật khẩu được băm bằng Argon2id.",
                    )
                    .font(FontId::proportional(12.5))
                    .color(palette.text_muted),
                );
                ui.add_space(SPACE_MD);

                egui::Grid::new("signup_step1_grid")
                    .num_columns(2)
                    .spacing([16.0, 12.0])
                    .show(ui, |ui| {
                        ui.label(RichText::new("Tên đăng nhập (username):").color(palette.text_secondary));
                        ui.add(
                            egui::TextEdit::singleline(&mut app.signup_username)
                                .hint_text("vd: giangvien_toan (3-32 ký tự không dấu)")
                                .desired_width(340.0),
                        );
                        ui.end_row();

                        ui.label(RichText::new("Tên hiển thị (display_name):").color(palette.text_secondary));
                        ui.add(
                            egui::TextEdit::singleline(&mut app.signup_full_name)
                                .hint_text("Tên hiển thị trên bảng giảng")
                                .desired_width(340.0),
                        );
                        ui.end_row();

                        ui.label(RichText::new("Mật khẩu / Mã PIN (>= 6 ký tự):").color(palette.text_secondary));
                        ui.add(
                            egui::TextEdit::singleline(&mut app.signup_pin)
                                .password(true)
                                .hint_text("Tối thiểu 6 ký tự hoặc 6 chữ số")
                                .desired_width(340.0),
                        );
                        ui.end_row();

                        ui.label(RichText::new("Xác nhận Mật khẩu:").color(palette.text_secondary));
                        ui.add(
                            egui::TextEdit::singleline(&mut app.signup_pin_confirm)
                                .password(true)
                                .hint_text("Nhập lại mật khẩu")
                                .desired_width(340.0),
                        );
                        ui.end_row();
                    });

                ui.add_space(SPACE_MD);

                if !app.auth_error_msg.is_empty() {
                    ui.label(
                        RichText::new(&app.auth_error_msg)
                            .font(FontId::proportional(13.0))
                            .strong()
                            .color(palette.danger),
                    );
                    ui.add_space(SPACE_SM);
                }

                ui.horizontal(|ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Tiếp tục: Quét Khuôn Mặt (Bước 2) ->")
                                    .font(FontId::proportional(14.0))
                                    .strong()
                                    .color(Color32::WHITE),
                            )
                            .fill(palette.primary)
                            .rounding(Rounding::same(RADIUS_MD))
                            .min_size(Vec2::new(280.0, 40.0)),
                        )
                        .clicked()
                    {
                        if app.validate_signup_step1().is_ok() {
                            app.signup_step = 2;
                            app.signup_face_samples.clear();
                            app.signup_status_msg.clear();
                            app.ensure_camera_acquired();
                        }
                    }

                    ui.add_space(SPACE_SM);

                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Chỉ đăng ký bằng Mật khẩu (Bỏ qua Camera)")
                                    .font(FontId::proportional(12.5))
                                    .color(palette.text_secondary),
                            )
                            .fill(palette.surface_elevated)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(RADIUS_MD))
                            .min_size(Vec2::new(290.0, 40.0)),
                        )
                        .clicked()
                    {
                        if app.validate_signup_step1().is_ok() {
                            app.finalize_signup_account(true);
                        }
                    }
                });
            } else {
                // =====================================================================
                // STEP 2: 3-TO-5 MULTI-ANGLE FACE CAPTURE -> AVERAGE VECTOR
                // =====================================================================
                app.ensure_camera_acquired();
                ui.columns(2, |cols| {
                    // Left: Camera preview + Face bounding circle
                    render_camera_auth_box(
                        app,
                        &mut cols[0],
                        palette,
                        360.0,
                        "Đặt khuôn mặt vào vòng tròn hướng dẫn",
                    );

                    // Right: Multi-angle enrollment progress
                    let right = &mut cols[1];
                    let captured = app.signup_face_samples.len();
                    let next_prompt =
                        ENROLL_ANGLE_PROMPTS[captured.min(MAX_FACE_ENROLL_SAMPLES - 1)];

                    right.label(
                        RichText::new(format!(
                            "Thu thập Đa Góc Khuôn Mặt ({captured}/{MAX_FACE_ENROLL_SAMPLES} góc)"
                        ))
                        .font(FontId::proportional(16.0))
                        .strong()
                        .color(palette.text_primary),
                    );
                    right.add_space(4.0);
                    right.label(
                        RichText::new(
                            "Chỉ lưu vector đặc trưng chuẩn hóa trong SQLite — tuyệt đối không lưu ảnh thô.",
                        )
                        .font(FontId::proportional(12.0))
                        .color(palette.text_muted),
                    );

                    right.add_space(SPACE_SM);
                    let progress = (captured as f32 / MAX_FACE_ENROLL_SAMPLES as f32).clamp(0.0, 1.0);
                    right.add(
                        egui::ProgressBar::new(progress)
                            .text(format!("Đã lấy {captured}/{MAX_FACE_ENROLL_SAMPLES} mẫu (tối thiểu {MIN_FACE_ENROLL_SAMPLES})")),
                    );

                    right.add_space(SPACE_SM);
                    egui::Frame::none()
                        .fill(palette.bg_panel)
                        .stroke(Stroke::new(1.0_f32, palette.secondary))
                        .rounding(Rounding::same(RADIUS_SM))
                        .inner_margin(egui::Margin::same(10.0))
                        .show(right, |ui| {
                            ui.set_width(ui.available_width());
                            let prompt_text = if captured >= MAX_FACE_ENROLL_SAMPLES {
                                "Đã đủ 5 góc tối ưu! Nhấn [Hoàn Tất Đăng Ký] bên dưới."
                            } else {
                                next_prompt
                            };
                            ui.label(
                                RichText::new(prompt_text)
                                    .font(FontId::proportional(13.0))
                                    .strong()
                                    .color(palette.secondary),
                            );
                        });

                    right.add_space(SPACE_MD);

                    let can_capture = app.camera_using_hardware
                        && app.last_face_bbox_norm.is_some()
                        && captured < MAX_FACE_ENROLL_SAMPLES;
                    if right
                        .add_enabled(
                            can_capture,
                            egui::Button::new(
                                RichText::new(format!(
                                    "Chụp Góc {}/{}",
                                    (captured + 1).min(MAX_FACE_ENROLL_SAMPLES),
                                    MAX_FACE_ENROLL_SAMPLES
                                ))
                                .font(FontId::proportional(14.0))
                                .strong()
                                .color(Color32::WHITE),
                            )
                            .fill(palette.secondary)
                            .rounding(Rounding::same(RADIUS_MD))
                            .min_size(Vec2::new(280.0, 38.0)),
                        )
                        .clicked()
                    {
                        app.capture_signup_face_sample();
                    }

                    right.add_space(SPACE_SM);

                    let ready_finish = captured >= MIN_FACE_ENROLL_SAMPLES;
                    if right
                        .add_enabled(
                            ready_finish,
                            egui::Button::new(
                                RichText::new("Hoàn Tất Đăng Ký Tài Khoản")
                                    .font(FontId::proportional(14.5))
                                    .strong()
                                    .color(Color32::WHITE),
                            )
                            .fill(palette.success)
                            .rounding(Rounding::same(RADIUS_MD))
                            .min_size(Vec2::new(280.0, 42.0)),
                        )
                        .clicked()
                    {
                        app.finalize_signup_account(false);
                    }

                    right.add_space(SPACE_SM);

                    right.horizontal(|ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("<- Quay lại Bước 1")
                                        .color(palette.text_primary),
                                )
                                .fill(palette.surface_elevated)
                                .rounding(Rounding::same(RADIUS_SM)),
                            )
                            .clicked()
                        {
                            app.signup_step = 1;
                            app.release_camera_hardware();
                        }

                        if !app.signup_face_samples.is_empty()
                            && ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("Xóa chụp lại")
                                            .color(palette.warning),
                                    )
                                    .fill(palette.surface_elevated)
                                    .rounding(Rounding::same(RADIUS_SM)),
                                )
                                .clicked()
                        {
                            app.signup_face_samples.clear();
                        }

                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Lưu không dùng Face ID")
                                        .color(palette.text_muted),
                                )
                                .fill(palette.bg_panel)
                                .rounding(Rounding::same(RADIUS_SM)),
                            )
                            .clicked()
                        {
                            app.finalize_signup_account(true);
                        }
                    });

                    if !app.signup_status_msg.is_empty() {
                        right.add_space(SPACE_SM);
                        right.label(
                            RichText::new(&app.signup_status_msg)
                                .font(FontId::proportional(12.5))
                                .strong()
                                .color(palette.secondary),
                        );
                    }
                    if !app.auth_error_msg.is_empty() {
                        right.add_space(4.0);
                        right.label(
                            RichText::new(&app.auth_error_msg)
                                .font(FontId::proportional(12.5))
                                .strong()
                                .color(palette.danger),
                        );
                    }
                });
            }
        });
}

fn render_signin_flow(
    app: &mut AirTeachApp,
    ui: &mut egui::Ui,
    palette: crate::ui::brand::ThemePalette,
) {
    let card_w = 880.0_f32.min(ui.available_width() - 24.0);
    egui::Frame::none()
        .fill(palette.surface)
        .stroke(Stroke::new(1.2_f32, palette.border))
        .rounding(Rounding::same(RADIUS_LG))
        .inner_margin(egui::Margin::same(SPACE_LG))
        .show(ui, |ui| {
            ui.set_width(card_w);

            if app.registered_users.is_empty() {
                ui.vertical_centered(|ui| {
                    ui.add_space(SPACE_MD);
                    ui.label(
                        RichText::new("Chưa có tài khoản giảng viên nào trên thiết bị này.")
                            .font(FontId::proportional(16.0))
                            .strong()
                            .color(palette.warning),
                    );
                    ui.add_space(SPACE_SM);
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Chuyển sang Đăng ký Tài khoản Đầu tiên ->")
                                    .font(FontId::proportional(14.0))
                                    .strong()
                                    .color(Color32::WHITE),
                            )
                            .fill(palette.primary)
                            .rounding(Rounding::same(RADIUS_MD))
                            .min_size(Vec2::new(320.0, 40.0)),
                        )
                        .clicked()
                    {
                        app.auth_tab_is_signup = true;
                        app.signup_step = 1;
                    }
                    ui.add_space(SPACE_MD);
                });
                return;
            }

            if app.selected_user_idx >= app.registered_users.len() {
                app.selected_user_idx = 0;
            }

            // Dual Auth Method Switcher (Face ID Default <-> Instant Password Toggle)
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Chọn phương thức:")
                        .font(FontId::proportional(13.5))
                        .color(palette.text_secondary),
                );
                if ui
                    .selectable_label(app.signin_face_mode, "Face ID (Mặc định)")
                    .clicked()
                {
                    app.signin_face_mode = true;
                    app.ensure_camera_acquired();
                }
                if ui
                    .selectable_label(!app.signin_face_mode, "Mật khẩu / PIN (Khi ngược sáng / không có Camera)")
                    .clicked()
                {
                    app.signin_face_mode = false;
                    app.release_camera_hardware();
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Account selector combo
                    let current_label = app
                        .registered_users
                        .get(app.selected_user_idx)
                        .map(|u| format!("{} (@{})", u.display_name, u.username))
                        .unwrap_or_default();
                    egui::ComboBox::from_id_salt("signin_user_select")
                        .selected_text(current_label)
                        .width(230.0)
                        .show_ui(ui, |ui| {
                            let mut changed_idx = None;
                            for (idx, u) in app.registered_users.iter().enumerate() {
                                if ui
                                    .selectable_label(
                                        app.selected_user_idx == idx,
                                        format!("{} (@{})", u.display_name, u.username),
                                    )
                                    .clicked()
                                {
                                    changed_idx = Some(idx);
                                }
                            }
                            if let Some(idx) = changed_idx {
                                app.select_registered_user_for_signin(idx);
                            }
                        });
                });
            });

            if !app.signup_status_msg.is_empty() {
                ui.add_space(6.0);
                ui.label(
                    RichText::new(&app.signup_status_msg)
                        .font(FontId::proportional(12.5))
                        .strong()
                        .color(palette.success),
                );
            }

            ui.add_space(SPACE_MD);
            ui.separator();
            ui.add_space(SPACE_MD);

            if app.signin_face_mode {
                app.ensure_camera_acquired();
                ui.columns(2, |cols| {
                    render_camera_auth_box(
                        app,
                        &mut cols[0],
                        palette,
                        360.0,
                        "Nhìn thẳng vào camera để nhận diện Face ID",
                    );

                    let right = &mut cols[1];
                    let has_enrolled_face = !app.enrolled_face_embedding.is_empty();
                    let thr = app.config.face_match_threshold;

                    right.label(
                        RichText::new("Xác thực Khuôn Mặt Cục bộ (Face ID)")
                            .font(FontId::proportional(16.0))
                            .strong()
                            .color(palette.text_primary),
                    );
                    right.add_space(SPACE_SM);

                    if !has_enrolled_face {
                        right.label(
                            RichText::new(
                                "Tài khoản này chưa đăng ký Face ID. Vui lòng chuyển sang đăng nhập bằng Mật khẩu.",
                            )
                            .color(palette.warning),
                        );
                    } else {
                        let sim = app.last_face_similarity;
                        let matched = sim >= thr;
                        right.label(
                            RichText::new(format!(
                                "Độ tương đồng Cosine: {:.3} / Ngưỡng {:.2}",
                                sim, thr
                            ))
                            .font(FontId::monospace(13.0))
                            .strong()
                            .color(if matched {
                                palette.success
                            } else {
                                palette.secondary
                            }),
                        );
                        right.add(egui::ProgressBar::new(sim.clamp(0.0, 1.0)));

                        right.add_space(6.0);
                        let live_txt = if app.face_auth.liveness.is_live() {
                            format!(
                                "Liveness: Đã xác thực chuyển động / chớp mắt (EAR {:.2})",
                                app.face_auth.liveness.ear_value
                            )
                        } else {
                            "Liveness: Hãy chớp mắt hoặc khẽ nghiêng đầu nhẹ".to_string()
                        };
                        right.label(
                            RichText::new(live_txt)
                                .font(FontId::proportional(12.0))
                                .color(if app.face_auth.liveness.is_live() {
                                    palette.success
                                } else {
                                    palette.text_muted
                                }),
                        );

                        right.add_space(SPACE_SM);
                        right.checkbox(
                            &mut app.auto_face_unlock_enabled,
                            "Tự động mở khóa ngay khi khớp Face ID",
                        );
                    }

                    right.add_space(SPACE_MD);

                    if !app.camera_using_hardware {
                        if let Some(ref err_msg) = app.camera_error_message {
                            right.label(
                                RichText::new(err_msg)
                                    .font(FontId::proportional(12.5))
                                    .color(palette.danger),
                            );
                            right.add_space(6.0);
                        }
                    }

                    right.horizontal(|ui| {
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Dùng Mật khẩu ngay")
                                        .font(FontId::proportional(13.0))
                                        .strong()
                                        .color(Color32::WHITE),
                                )
                                .fill(palette.primary)
                                .rounding(Rounding::same(RADIUS_MD))
                                .min_size(Vec2::new(165.0, 36.0)),
                            )
                            .clicked()
                        {
                            app.signin_face_mode = false;
                            app.release_camera_hardware();
                        }

                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Kết nối lại Camera")
                                        .color(palette.secondary),
                                )
                                .fill(palette.surface_elevated)
                                .stroke(Stroke::new(1.0_f32, palette.border))
                                .rounding(Rounding::same(RADIUS_MD))
                                .min_size(Vec2::new(155.0, 36.0)),
                            )
                            .clicked()
                        {
                            let _ = app.camera.cmd_tx.try_send(CameraControlCmd::ReconnectHardware);
                        }
                    });
                });
            } else {
                // Password Login Form (Camera released cleanly!)
                ui.vertical_centered(|ui| {
                    ui.set_max_width(420.0);
                    ui.label(
                        RichText::new("Đăng nhập bằng Mật khẩu / PIN (Argon2id)")
                            .font(FontId::proportional(16.0))
                            .strong()
                            .color(palette.text_primary),
                    );
                    ui.add_space(SPACE_SM);

                    if let Some(u) = app.registered_users.get(app.selected_user_idx) {
                        ui.label(
                            RichText::new(format!("Tài khoản: {} (@{})", u.display_name, u.username))
                                .font(FontId::proportional(13.5))
                                .color(palette.secondary),
                        );
                    }
                    ui.add_space(SPACE_SM);

                    let resp = ui.add(
                        egui::TextEdit::singleline(&mut app.pin_input_buffer)
                            .password(true)
                            .hint_text("Nhập mật khẩu tài khoản...")
                            .desired_width(340.0),
                    );
                    let enter_pressed =
                        resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                    ui.add_space(SPACE_MD);

                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Xác nhận & Đăng nhập")
                                    .font(FontId::proportional(14.5))
                                    .strong()
                                    .color(Color32::WHITE),
                            )
                            .fill(palette.primary)
                            .rounding(Rounding::same(RADIUS_MD))
                            .min_size(Vec2::new(260.0, 40.0)),
                        )
                        .clicked()
                        || enter_pressed
                    {
                        app.try_unlock_with_password();
                    }

                    if !app.auth_error_msg.is_empty() {
                        ui.add_space(SPACE_SM);
                        ui.label(
                            RichText::new(&app.auth_error_msg)
                                .font(FontId::proportional(13.0))
                                .strong()
                                .color(palette.danger),
                        );
                    }
                });
            }
        });
}

fn render_camera_auth_box(
    app: &mut AirTeachApp,
    ui: &mut egui::Ui,
    palette: crate::ui::brand::ThemePalette,
    preview_w: f32,
    hint_label: &str,
) {
    let preview_h = preview_w * 0.68;
    egui::Frame::none()
        .fill(palette.bg_panel)
        .stroke(Stroke::new(1.2_f32, palette.border))
        .rounding(Rounding::same(RADIUS_MD))
        .inner_margin(egui::Margin::same(SPACE_SM))
        .show(ui, |ui| {
            let (rect, _) = ui.allocate_exact_size(Vec2::new(preview_w, preview_h), Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, Rounding::same(RADIUS_SM), palette.bg_canvas);

            let mut active_view_rect = rect;
            if app.camera_using_hardware {
                if let Some(tex) = &app.camera_texture {
                    let tex_size = tex.size_vec2();
                    let tex_aspect = if tex_size.y > 0.0 {
                        tex_size.x / tex_size.y
                    } else {
                        4.0 / 3.0
                    };
                    let box_aspect = rect.width() / rect.height().max(1.0);
                    let draw_rect = if (tex_aspect - box_aspect).abs() < 0.05 {
                        rect
                    } else if tex_aspect > box_aspect {
                        let h = rect.width() / tex_aspect;
                        Rect::from_center_size(rect.center(), Vec2::new(rect.width(), h))
                    } else {
                        let w = rect.height() * tex_aspect;
                        Rect::from_center_size(rect.center(), Vec2::new(w, rect.height()))
                    };
                    active_view_rect = draw_rect;

                    painter.image(
                        tex.id(),
                        draw_rect,
                        Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                        Color32::WHITE,
                    );
                }

                let ring_col = if app.last_face_bbox_norm.is_some() {
                    if app.camera_quality_gate_passed {
                        Color32::from_rgba_unmultiplied(
                            palette.success.r(),
                            palette.success.g(),
                            palette.success.b(),
                            170,
                        )
                    } else {
                        Color32::from_rgba_unmultiplied(
                            palette.warning.r(),
                            palette.warning.g(),
                            palette.warning.b(),
                            170,
                        )
                    }
                } else {
                    Color32::from_rgba_unmultiplied(6, 182, 212, 165)
                };

                // Visual biometric alignment ring
                painter.circle_stroke(
                    rect.center(),
                    rect.height() * 0.35,
                    Stroke::new(1.8_f32, ring_col),
                );

                // Top-left Adaptive Lighting pill + Luma
                let lighting_col = match app.camera_lighting_mode {
                    LightingAdaptationMode::LowLightBoost => Color32::from_rgb(0xf5, 0x9e, 0x0b),
                    LightingAdaptationMode::GlareSuppression => Color32::from_rgb(0x38, 0xbd, 0xf8),
                    LightingAdaptationMode::Normal => Color32::from_rgb(0x22, 0xc5, 0x5e),
                };
                draw_auth_hud_pill(
                    &painter,
                    Pos2::new(rect.min.x + 8.0, rect.min.y + 8.0),
                    false,
                    Color32::from_rgba_unmultiplied(10, 15, 28, 200),
                    lighting_col,
                    lighting_col,
                    &format!(
                        "{} · L:{:.0}",
                        app.camera_lighting_mode.label(),
                        app.camera_mean_luma
                    ),
                );

                // Top-right Laplacian Quality Gate + FPS pill
                let gate_col = if app.camera_quality_gate_passed {
                    palette.success
                } else {
                    palette.warning
                };
                let gate_txt = if app.camera_quality_gate_passed {
                    format!(
                        "S:{:.0} ĐẠT NÉT · {:.0}fps",
                        app.camera_sharpness, app.camera_fps
                    )
                } else {
                    format!(
                        "S:{:.0} MỜ · {:.0}fps",
                        app.camera_sharpness, app.camera_fps
                    )
                };
                draw_auth_hud_pill(
                    &painter,
                    Pos2::new(rect.max.x - 8.0, rect.min.y + 8.0),
                    true,
                    Color32::from_rgba_unmultiplied(10, 15, 28, 200),
                    gate_col,
                    gate_col,
                    &gate_txt,
                );

                // 2D Eye-level horizon & pupil crosshairs
                if let Some([lx, ly, rx, ry]) = app.last_face_eyes_norm {
                    let lp = Pos2::new(
                        active_view_rect.min.x + lx * active_view_rect.width(),
                        active_view_rect.min.y + ly * active_view_rect.height(),
                    );
                    let rp = Pos2::new(
                        active_view_rect.min.x + rx * active_view_rect.width(),
                        active_view_rect.min.y + ry * active_view_rect.height(),
                    );
                    let eye_col = Color32::from_rgb(0x00, 0xf5, 0xd4);
                    painter.line_segment(
                        [lp, rp],
                        Stroke::new(
                            1.2_f32,
                            Color32::from_rgba_unmultiplied(0x00, 0xf5, 0xd4, 150),
                        ),
                    );
                    for ep in [lp, rp] {
                        painter.line_segment(
                            [Pos2::new(ep.x - 5.0, ep.y), Pos2::new(ep.x + 5.0, ep.y)],
                            Stroke::new(1.5_f32, eye_col),
                        );
                        painter.line_segment(
                            [Pos2::new(ep.x, ep.y - 5.0), Pos2::new(ep.x, ep.y + 5.0)],
                            Stroke::new(1.5_f32, eye_col),
                        );
                        painter.circle_stroke(ep, 4.0, Stroke::new(1.2_f32, eye_col));
                    }
                }

                // Real detected face bounding box + scanner corner brackets
                if let Some([bx, by, bw, bh]) = app.last_face_bbox_norm {
                    let face_rect = Rect::from_min_size(
                        Pos2::new(
                            active_view_rect.left() + bx * active_view_rect.width(),
                            active_view_rect.top() + by * active_view_rect.height(),
                        ),
                        Vec2::new(
                            bw * active_view_rect.width(),
                            bh * active_view_rect.height(),
                        ),
                    );
                    let status_col = if app.camera_quality_gate_passed {
                        palette.success
                    } else {
                        palette.warning
                    };
                    painter.rect_stroke(
                        face_rect,
                        Rounding::same(8.0),
                        Stroke::new(2.0_f32, status_col),
                    );
                    let bracket = (face_rect.width().min(face_rect.height()) * 0.16).clamp(8.0, 18.0);
                    let corners = [
                        (face_rect.left_top(), Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)),
                        (face_rect.right_top(), Vec2::new(-1.0, 0.0), Vec2::new(0.0, 1.0)),
                        (face_rect.left_bottom(), Vec2::new(1.0, 0.0), Vec2::new(0.0, -1.0)),
                        (face_rect.right_bottom(), Vec2::new(-1.0, 0.0), Vec2::new(0.0, -1.0)),
                    ];
                    for (c, dx, dy) in corners {
                        painter.line_segment([c, c + dx * bracket], Stroke::new(3.0_f32, status_col));
                        painter.line_segment([c, c + dy * bracket], Stroke::new(3.0_f32, status_col));
                    }
                    let face_tag = if app.camera_quality_gate_passed {
                        "FACE ALIGNED · READY"
                    } else {
                        "KHUNG HÌNH MỜ · GIỮ YÊN CAMERA"
                    };
                    painter.text(
                        Pos2::new(face_rect.center().x, (face_rect.bottom() + 14.0).min(rect.bottom() - 6.0)),
                        Align2::CENTER_CENTER,
                        face_tag,
                        FontId::monospace(10.0),
                        status_col,
                    );
                } else {
                    painter.text(
                        Pos2::new(rect.center().x, rect.bottom() - 10.0),
                        Align2::CENTER_BOTTOM,
                        hint_label,
                        FontId::proportional(11.5),
                        palette.warning,
                    );
                }
            } else {
                draw_auth_hud_pill(
                    &painter,
                    Pos2::new(rect.min.x + 8.0, rect.min.y + 8.0),
                    false,
                    Color32::from_rgba_unmultiplied(10, 15, 28, 200),
                    palette.border,
                    palette.text_muted,
                    "CAMERA STANDBY",
                );
                let err_display = app
                    .camera_error_message
                    .as_deref()
                    .unwrap_or("Không tìm thấy Webcam hoặc đang chờ cấp quyền Camera");
                painter.text(
                    rect.center(),
                    Align2::CENTER_CENTER,
                    err_display,
                    FontId::proportional(12.5),
                    palette.warning,
                );
            }
        });
}

fn draw_auth_hud_pill(
    painter: &egui::Painter,
    anchor: Pos2,
    align_right: bool,
    bg: Color32,
    border: Color32,
    dot_col: Color32,
    label: &str,
) {
    let galley = painter.layout_no_wrap(label.to_string(), FontId::monospace(10.0), Color32::WHITE);
    let pad_x = 7.0;
    let dot_space = 9.0;
    let w = galley.size().x + pad_x * 2.0 + dot_space;
    let h = 19.0;
    let min_x = if align_right { anchor.x - w } else { anchor.x };
    let rect = Rect::from_min_size(Pos2::new(min_x, anchor.y), Vec2::new(w, h));
    painter.rect_filled(rect, Rounding::same(9.5), bg);
    painter.rect_stroke(
        rect,
        Rounding::same(9.5),
        Stroke::new(
            1.0_f32,
            Color32::from_rgba_unmultiplied(border.r(), border.g(), border.b(), 170),
        ),
    );
    painter.circle_filled(
        Pos2::new(rect.min.x + pad_x + 2.5, rect.center().y),
        2.8,
        dot_col,
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
