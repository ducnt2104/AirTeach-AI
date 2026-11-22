//! Right Dock (`Trợ Lý Sư Phạm Gemini AI`, Collapsible, `Width = 330px` — MODULE 2.4 & MODULE 4):
//! - Compact API Key setup (auto-hides once key is saved, showing only `[Đổi API Key]`).
//! - Clean AI analysis output view (`ScrollArea`).
//! - 3 clear action buttons:
//!   * `[1. Giải & Chữa Bài Trên Bảng]`
//!   * `[2. Tóm Tắt & Tạo Trắc Nghiệm]`
//!   * `[3. Hỏi Đáp Sư Phạm]`

use crate::ai::gemini_chat::{
    mask_api_key, ApiConnectionStatus, GeminiChatClient, PedagogicalAnalysisMode,
    SubjectAssistantMode, DEFAULT_GEMINI_MODEL,
};
use crate::app::AirTeachApp;
use crate::ui::brand::ThemeMode;
use crate::utils::document_loader::DocumentLoader;
use egui::{Color32, FontId, RichText, Rounding, ScrollArea, Stroke, Vec2};

fn format_model_badge(model_id: &str) -> String {
    match model_id.trim() {
        "gemini-3.8-flash" => "Gemini 3.8 Flash".to_string(),
        "gemini-3.5-flash" => "Gemini 3.5 Flash".to_string(),
        "gemini-3.5-flash-lite" => "Gemini 3.5 Flash-Lite".to_string(),
        "gemini-2.5-flash" => "Gemini 2.5 Flash".to_string(),
        other if !other.is_empty() => other.to_string(),
        _ => DEFAULT_GEMINI_MODEL.to_string(),
    }
}

pub fn render_chat_drawer(app: &mut AirTeachApp, ctx: &egui::Context) {
    if !app.chat_drawer_open {
        return;
    }

    let palette = app.config.theme_mode.palette();
    let active_model_id = GeminiChatClient::resolve_model_id(&app.config.gemini_model);
    let model_badge = format_model_badge(&active_model_id);
    let saved_plain_key = app.storage.get_decrypted_api_key(&app.config);
    let (resolved_key, key_source) = GeminiChatClient::resolve_api_key_with_source(&saved_plain_key);
    let has_env_or_cfg_key = !resolved_key.is_empty();

    egui::SidePanel::right("airteach_modern_ai_copilot_panel")
        .resizable(false)
        .exact_width(340.0)
        .frame(
            egui::Frame::none()
                .fill(palette.bg_panel)
                .stroke(Stroke::new(1.0_f32, palette.border))
                .inner_margin(egui::Margin::same(10.0)),
        )
        .show(ctx, |ui| {
            // =========================================================================
            // 1. HEADER, 3-STATE CONNECTION BADGE & COMPACT API KEY SETUP
            // =========================================================================
            let has_saved_key = !app.config.encrypted_api_key_b64.trim().is_empty();
            let (badge_text, badge_color) = if !has_env_or_cfg_key {
                ("Chưa cấu hình", palette.text_muted)
            } else {
                match app.api_connection_status {
                    ApiConnectionStatus::Connected => ("Đã kết nối", palette.success),
                    ApiConnectionStatus::Error => ("Lỗi Key/Quota", palette.danger),
                    ApiConnectionStatus::Unconfigured => ("Chưa xác thực", palette.warning),
                }
            };
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new("Trợ Lý Sư Phạm AI")
                            .font(FontId::proportional(13.5))
                            .strong()
                            .color(palette.text_primary),
                    );
                    ui.label(
                        RichText::new(format!("{} • {}", model_badge, badge_text))
                            .font(FontId::proportional(10.5))
                            .strong()
                            .color(badge_color),
                    );
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Đóng").size(11.5).color(palette.text_muted),
                            )
                            .fill(palette.surface)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(5.0)),
                        )
                        .clicked()
                    {
                        app.chat_drawer_open = false;
                    }

                    let test_lbl = if app.api_test_in_flight { "..." } else { "Test API" };
                    if ui
                        .add_enabled(
                            !app.api_test_in_flight,
                            egui::Button::new(
                                RichText::new(test_lbl).size(11.0).color(Color32::WHITE).strong(),
                            )
                            .fill(palette.secondary)
                            .rounding(Rounding::same(5.0)),
                        )
                        .clicked()
                    {
                        app.trigger_gemini_health_check();
                    }

                    if has_saved_key
                        && ui
                            .add(
                                egui::Button::new(
                                    RichText::new("Xóa Key")
                                        .size(11.0)
                                        .color(palette.danger),
                                )
                                .fill(palette.surface)
                                .stroke(Stroke::new(1.0_f32, palette.border))
                                .rounding(Rounding::same(5.0)),
                            )
                            .clicked()
                    {
                        app.clear_ui_gemini_api_key();
                    }
                });
            });

            if has_env_or_cfg_key {
                ui.label(
                    RichText::new(format!(
                        "Key: {} ({})",
                        mask_api_key(&resolved_key),
                        key_source.label_vi()
                    ))
                    .font(FontId::proportional(10.0))
                    .color(palette.text_muted),
                );
            }
            if !app.api_status_detail_vi.trim().is_empty() {
                ui.label(
                    RichText::new(&app.api_status_detail_vi)
                        .font(FontId::proportional(10.0))
                        .color(badge_color),
                );
            }

            if !has_saved_key {
                ui.add_space(4.0);
                egui::Frame::none()
                    .fill(palette.surface)
                    .stroke(Stroke::new(1.0_f32, palette.secondary))
                    .rounding(Rounding::same(6.0))
                    .inner_margin(egui::Margin::same(6.0))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::TextEdit::singleline(&mut app.api_key_edit_buffer)
                                    .password(true)
                                    .hint_text("Nhập GEMINI_API_KEY...")
                                    .desired_width(215.0),
                            );
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new("Lưu")
                                            .size(12.0)
                                            .color(Color32::WHITE)
                                            .strong(),
                                    )
                                    .fill(palette.primary)
                                    .rounding(Rounding::same(5.0)),
                                )
                                .clicked()
                                && !app.api_key_edit_buffer.trim().is_empty()
                            {
                                app.save_ui_gemini_api_key();
                            }
                        });
                    });
            }

            ui.add_space(6.0);

            // Compact Subject Pill Strip
            ui.horizontal(|ui| {
                for mode in [
                    SubjectAssistantMode::Math,
                    SubjectAssistantMode::Physics,
                    SubjectAssistantMode::Chemistry,
                    SubjectAssistantMode::General,
                ] {
                    let active = app.subject_mode == mode;
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(mode.short_badge())
                                    .font(FontId::proportional(10.5))
                                    .color(if active { Color32::WHITE } else { palette.text_secondary })
                                    .strong(),
                            )
                            .fill(if active { palette.primary } else { palette.surface })
                            .stroke(Stroke::new(1.0_f32, if active { palette.primary } else { palette.border }))
                            .rounding(Rounding::same(5.0)),
                        )
                        .on_hover_text(mode.label())
                        .clicked()
                    {
                        app.subject_mode = mode;
                    }
                }
            });

            ui.add_space(6.0);

            // =========================================================================
            // 2. THREE CLEAR PEDAGOGICAL ACTION BUTTONS (Structured JSON + RAG + Vision)
            // =========================================================================
            let mut selected_mode: Option<PedagogicalAnalysisMode> = None;
            for (mode, label) in [
                (
                    PedagogicalAnalysisMode::SolveAndCheckError,
                    "1. Giải & Chữa Bài Trên Bảng (JSON)",
                ),
                (
                    PedagogicalAnalysisMode::SummarizeAndQuiz,
                    "2. Tóm Tắt & Tạo Trắc Nghiệm",
                ),
                (
                    PedagogicalAnalysisMode::PedagogicalInsights,
                    "3. Gợi Ý Sư Phạm & Tương Tác",
                ),
            ] {
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new(label)
                                .font(FontId::proportional(12.2))
                                .color(palette.text_primary)
                                .strong(),
                        )
                        .fill(palette.surface)
                        .stroke(Stroke::new(1.0_f32, palette.primary))
                        .rounding(Rounding::same(6.0))
                        .min_size(Vec2::new(ui.available_width(), 28.0)),
                    )
                    .clicked()
                {
                    selected_mode = Some(mode);
                }
                ui.add_space(3.0);
            }

            if let Some(mode) = selected_mode {
                app.analyze_board_and_lecture(mode);
            }

            ui.separator();
            ui.add_space(4.0);

            // =========================================================================
            // 3. CLEAN AI ANALYSIS OUTPUT VIEW
            // =========================================================================
            let avail_h = (ui.available_height() - 84.0).max(120.0);

            ScrollArea::vertical()
                .id_salt("gemini_chat_history_scroll")
                .max_height(avail_h)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for msg in &app.chat_messages {
                        let is_user = msg.role == "user";
                        let card_fill = if is_user {
                            match palette.mode {
                                ThemeMode::Dark => Color32::from_rgb(0x23, 0x2f, 0x46),
                                ThemeMode::Light => Color32::from_rgb(0xEE, 0xF2, 0xFF),
                            }
                        } else {
                            palette.surface
                        };
                        let border_col = if is_user {
                            palette.primary
                        } else {
                            palette.border
                        };
                        let clean_text = DocumentLoader::format_latex_to_unicode_math(&msg.text);

                        egui::Frame::none()
                            .fill(card_fill)
                            .stroke(Stroke::new(1.0_f32, border_col))
                            .rounding(Rounding::same(8.0))
                            .inner_margin(egui::Margin::same(9.0))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.horizontal_wrapped(|ui| {
                                    let role_lbl = if is_user {
                                        "Giáo viên".to_string()
                                    } else {
                                        model_badge.clone()
                                    };
                                    ui.label(
                                        RichText::new(role_lbl)
                                            .font(FontId::proportional(11.5))
                                            .strong()
                                            .color(if is_user {
                                                palette.primary
                                            } else {
                                                palette.secondary
                                            }),
                                    );
                                    if msg.has_image_attachment {
                                        let dim_txt = match msg.image_dimensions {
                                            Some((w, h)) => format!("[Ảnh {w}x{h} PNG]"),
                                            None => "[Ảnh PNG]".to_string(),
                                        };
                                        ui.label(
                                            RichText::new(dim_txt)
                                                .font(FontId::proportional(10.5))
                                                .color(palette.danger),
                                        );
                                    }
                                    if let Some(ref badge) = msg.rag_slide_badge {
                                        ui.label(
                                            RichText::new(format!("[{badge}]"))
                                                .font(FontId::proportional(10.5))
                                                .color(palette.success),
                                        );
                                    }
                                });
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new(&clean_text)
                                        .font(FontId::proportional(12.5))
                                        .color(palette.text_primary),
                                );

                                if !is_user && !clean_text.trim().is_empty() {
                                    ui.add_space(4.0);
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                RichText::new("Sao chép lời giải")
                                                    .font(FontId::proportional(11.0))
                                                    .color(palette.text_secondary),
                                            )
                                            .fill(palette.surface_elevated)
                                            .stroke(Stroke::new(1.0_f32, palette.border))
                                            .rounding(Rounding::same(5.0)),
                                        )
                                        .clicked()
                                    {
                                        ctx.copy_text(clean_text.clone());
                                    }
                                }
                            });
                        ui.add_space(5.0);
                    }

                    if app.chat_streaming_active {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label(
                                RichText::new(format!("Đang phân tích với {model_badge}..."))
                                    .font(FontId::proportional(11.5))
                                    .color(palette.secondary),
                            );
                        });
                    }
                });

            ui.add_space(4.0);
            ui.separator();

            // Compact Question Input
            let te_resp = ui.add(
                egui::TextEdit::singleline(&mut app.chat_input_draft)
                    .hint_text("Nhập câu hỏi sư phạm...")
                    .desired_width(f32::INFINITY),
            );

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let send_clicked = ui
                    .add(
                        egui::Button::new(
                            RichText::new("Gửi câu hỏi")
                                .font(FontId::proportional(12.0))
                                .color(Color32::WHITE)
                                .strong(),
                        )
                        .fill(palette.primary)
                        .rounding(Rounding::same(6.0))
                        .min_size(Vec2::new(ui.available_width(), 28.0)),
                    )
                    .clicked();

                let enter_pressed =
                    te_resp.has_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

                if (send_clicked || enter_pressed) && !app.chat_input_draft.trim().is_empty() {
                    let prompt = app.chat_input_draft.trim().to_string();
                    app.chat_input_draft.clear();
                    app.send_chat_message(prompt, false);
                }
            });
        });
}
