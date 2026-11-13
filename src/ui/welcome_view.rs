//! Branded Welcome & First-Run Onboarding View (`src/ui/welcome_view.rs`):
//! - Presents bespoke vector identity for `AirTeach AI` by `DNT - Victer`
//! - Clean Slate Policy: on fresh installs (`users` table empty), routes straight into First-Run Account Setup
//!   without auto-logging in or seeding personal/school/mock data.

use crate::app::{AirTeachApp, AppScreen};
use crate::ui::brand::{
    render_about_modal, render_brand_corner_footer, ui_brand_logo, APP_BRAND_NAME, APP_TAGLINE_VI,
    ORG_BRAND_NAME, RADIUS_LG, RADIUS_MD, RADIUS_SM, SPACE_LG, SPACE_MD, SPACE_SM,
};
use egui::{Color32, FontId, RichText, Rounding, Stroke, Vec2};

pub fn render_welcome_view(app: &mut AirTeachApp, ctx: &egui::Context) {
    let palette = app.config.theme_mode.palette();
    let is_fresh_install = app.registered_users.is_empty();

    egui::CentralPanel::default()
        .frame(
            egui::Frame::none()
                .fill(palette.bg_canvas)
                .inner_margin(egui::Margin::same(SPACE_LG)),
        )
        .show(ctx, |ui| {
            // Top bar: Theme toggle + About DNT - Victer
            ui.horizontal(|ui| {
                egui::Frame::none()
                    .fill(palette.surface)
                    .stroke(Stroke::new(1.0_f32, palette.border))
                    .rounding(Rounding::same(RADIUS_SM))
                    .inner_margin(egui::Margin::symmetric(10.0, 4.0))
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new(format!("DNT - VICTER · EDUCATION TECH"))
                                .font(FontId::monospace(11.0))
                                .strong()
                                .color(palette.secondary),
                        );
                    });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Giới thiệu")
                                    .font(FontId::proportional(12.5))
                                    .color(palette.text_primary),
                            )
                            .fill(palette.surface)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(RADIUS_SM))
                            .min_size(Vec2::new(88.0, 30.0)),
                        )
                        .clicked()
                    {
                        app.about_modal_open = true;
                    }

                    let theme_btn_label = format!("Giao diện: {}", app.config.theme_mode.label());
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(theme_btn_label)
                                    .font(FontId::proportional(12.5))
                                    .color(palette.secondary),
                            )
                            .fill(palette.surface)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(RADIUS_SM))
                            .min_size(Vec2::new(165.0, 30.0)),
                        )
                        .clicked()
                    {
                        app.toggle_theme_mode(ctx);
                    }
                });
            });

            ui.add_space(SPACE_LG);

            // Centered Onboarding Card
            ui.vertical_centered(|ui| {
                egui::Frame::none()
                    .fill(palette.surface)
                    .stroke(Stroke::new(1.2_f32, palette.border))
                    .rounding(Rounding::same(RADIUS_LG))
                    .inner_margin(egui::Margin::same(SPACE_LG))
                    .show(ui, |ui| {
                        ui.set_max_width(760.0);

                        ui.horizontal(|ui| {
                            ui_brand_logo(ui, 88.0);
                            ui.add_space(SPACE_LG);
                            ui.vertical(|ui| {
                                ui.label(
                                    RichText::new(APP_BRAND_NAME)
                                        .font(FontId::proportional(32.0))
                                        .strong()
                                        .color(palette.text_primary),
                                );
                                ui.label(
                                    RichText::new(format!("Sản phẩm công nghệ giáo dục bởi {ORG_BRAND_NAME}"))
                                        .font(FontId::proportional(14.5))
                                        .strong()
                                        .color(palette.secondary),
                                );
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new(APP_TAGLINE_VI)
                                        .font(FontId::proportional(13.5))
                                        .color(palette.text_secondary),
                                );
                            });
                        });

                        ui.add_space(SPACE_MD);
                        ui.separator();
                        ui.add_space(SPACE_MD);

                        // 3 Pillars
                        ui.columns(3, |cols| {
                            render_pillar(
                                &mut cols[0],
                                palette,
                                "1. Không chạm Chuẩn xác",
                                "Lọc 1-Euro trên đủ 21 khớp tay + chuẩn hóa tỷ lệ lòng bàn tay bất biến theo khoảng cách.",
                                palette.secondary,
                            );
                            render_pillar(
                                &mut cols[1],
                                palette,
                                "2. Bảo mật Cục bộ",
                                "SQLite Local-First, mật khẩu Argon2id, chỉ lưu vector khuôn mặt số hóa (không lưu ảnh thô).",
                                palette.primary,
                            );
                            render_pillar(
                                &mut cols[2],
                                palette,
                                "3. Trợ lý Giảng dạy AI",
                                "Bảng viết vô cực 60 FPS, ghi âm bài giảng, mở PDF/Ảnh cục bộ & phân tích với Gemini AI.",
                                palette.success,
                            );
                        });

                        ui.add_space(SPACE_MD);

                        // Clean Slate Status Box
                        let status_banner = if is_fresh_install {
                            "Thiết lập Lần đầu (Clean Slate): Cơ sở dữ liệu tài khoản đang trống. Vui lòng tạo tài khoản giảng viên đầu tiên để bắt đầu."
                        } else {
                            "Hệ thống đã sẵn sàng với hồ sơ cục bộ được bảo vệ bằng Face ID / Mật khẩu Argon2id."
                        };
                        egui::Frame::none()
                            .fill(palette.bg_panel)
                            .stroke(Stroke::new(
                                1.0_f32,
                                if is_fresh_install {
                                    palette.secondary
                                } else {
                                    palette.border
                                },
                            ))
                            .rounding(Rounding::same(RADIUS_MD))
                            .inner_margin(egui::Margin::same(SPACE_SM + 4.0))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.label(
                                    RichText::new(status_banner)
                                        .font(FontId::proportional(13.0))
                                        .color(palette.text_secondary),
                                );
                            });

                        ui.add_space(SPACE_MD);

                        // Primary Action Buttons
                        ui.horizontal(|ui| {
                            if is_fresh_install {
                                if ui
                                    .add(
                                        egui::Button::new(
                                            RichText::new("Thiết lập Tài khoản Đầu tiên ->")
                                                .font(FontId::proportional(15.0))
                                                .strong()
                                                .color(Color32::WHITE),
                                        )
                                        .fill(palette.primary)
                                        .rounding(Rounding::same(RADIUS_MD))
                                        .min_size(Vec2::new(280.0, 44.0)),
                                    )
                                    .clicked()
                                {
                                    app.enter_auth_screen(true);
                                }
                            } else {
                                if ui
                                    .add(
                                        egui::Button::new(
                                            RichText::new("Đăng nhập để Giảng dạy ->")
                                                .font(FontId::proportional(15.0))
                                                .strong()
                                                .color(Color32::WHITE),
                                        )
                                        .fill(palette.primary)
                                        .rounding(Rounding::same(RADIUS_MD))
                                        .min_size(Vec2::new(240.0, 44.0)),
                                    )
                                    .clicked()
                                {
                                    app.enter_auth_screen(false);
                                }

                                ui.add_space(SPACE_SM);

                                if ui
                                    .add(
                                        egui::Button::new(
                                            RichText::new("Đăng ký Tài khoản mới")
                                                .font(FontId::proportional(14.0))
                                                .color(palette.text_primary),
                                        )
                                        .fill(palette.surface_elevated)
                                        .stroke(Stroke::new(1.0_f32, palette.border))
                                        .rounding(Rounding::same(RADIUS_MD))
                                        .min_size(Vec2::new(200.0, 44.0)),
                                    )
                                    .clicked()
                                {
                                    app.enter_auth_screen(true);
                                }
                            }
                        });
                    });
            });

            ui.add_space(SPACE_MD);
            render_brand_corner_footer(ui, palette, &mut app.about_modal_open);
        });

    render_about_modal(ctx, &mut app.about_modal_open, palette);
}

fn render_pillar(
    ui: &mut egui::Ui,
    palette: crate::ui::brand::ThemePalette,
    title: &str,
    desc: &str,
    accent: Color32,
) {
    egui::Frame::none()
        .fill(palette.bg_panel)
        .stroke(Stroke::new(1.0_f32, palette.border))
        .rounding(Rounding::same(RADIUS_MD))
        .inner_margin(egui::Margin::same(SPACE_MD))
        .show(ui, |ui| {
            ui.set_min_height(112.0);
            ui.label(
                RichText::new(title)
                    .font(FontId::proportional(14.0))
                    .strong()
                    .color(accent),
            );
            ui.add_space(6.0);
            ui.label(
                RichText::new(desc)
                    .font(FontId::proportional(12.2))
                    .color(palette.text_secondary),
            );
        });
}
