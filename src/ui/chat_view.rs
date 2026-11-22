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
use crate::ai::local_ai_manager::AiRoutingMode;
use crate::ai::whiteboard_agent::WhiteboardAction;
use crate::app::AirTeachApp;
use crate::ui::brand::ThemeMode;
use crate::utils::output_sanitizer::OutputSanitizer;
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

            // Smart Hybrid Routing + Local Model Auto-Discovery Bar
            ui.horizontal(|ui| {
                for rmode in [AiRoutingMode::Hybrid, AiRoutingMode::Local, AiRoutingMode::Cloud] {
                    let active = app.config.ai_routing_mode == rmode;
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(rmode.label())
                                    .font(FontId::proportional(10.0))
                                    .color(if active { Color32::WHITE } else { palette.text_muted })
                                    .strong(),
                            )
                            .fill(if active { palette.secondary } else { palette.surface })
                            .rounding(Rounding::same(5.0)),
                        )
                        .clicked()
                    {
                        app.config.ai_routing_mode = rmode;
                        app.perf_pipeline.telemetry.ai_mode = rmode.label().to_string();
                        app.storage.save_config(&app.config);
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Quét Local")
                                    .font(FontId::proportional(10.0))
                                    .color(palette.secondary),
                            )
                            .fill(palette.surface)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(5.0)),
                        )
                        .clicked()
                    {
                        app.discover_local_engines();
                    }
                });
            });
            if !app.discovered_local_servers.is_empty() {
                let mut picked_local: Option<(String, String, String)> = None;
                ui.horizontal_wrapped(|ui| {
                    for srv in &app.discovered_local_servers {
                        for mdl in srv.models.iter().take(4) {
                            let is_sel = app.config.local_model == *mdl;
                            if ui
                                .add(
                                    egui::Button::new(
                                        RichText::new(format!("{}: {}", srv.provider, mdl))
                                            .font(FontId::proportional(9.5))
                                            .color(if is_sel { Color32::WHITE } else { palette.text_secondary }),
                                    )
                                    .fill(if is_sel { palette.primary } else { palette.surface })
                                    .rounding(Rounding::same(4.0)),
                                )
                                .clicked()
                            {
                                picked_local = Some((srv.provider.clone(), srv.base_url.clone(), mdl.clone()));
                            }
                        }
                    }
                });
                if let Some((prov, url, mdl)) = picked_local {
                    app.select_local_model(prov, url, mdl);
                }
            }

            ui.add_space(4.0);

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

            let mut push_board_text: Option<String> = None;
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
                        let clean_text = OutputSanitizer::sanitize_plain_text(&msg.text);

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
                                    ui.horizontal(|ui| {
                                        if ui
                                            .add(
                                                egui::Button::new(
                                                    RichText::new("Sao chép")
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
                                        if ui
                                            .add(
                                                egui::Button::new(
                                                    RichText::new("Đưa lên bảng")
                                                        .font(FontId::proportional(11.0))
                                                        .color(palette.secondary)
                                                        .strong(),
                                                )
                                                .fill(palette.surface_elevated)
                                                .stroke(Stroke::new(1.0_f32, palette.secondary))
                                                .rounding(Rounding::same(5.0)),
                                            )
                                            .clicked()
                                        {
                                            push_board_text = Some(clean_text.clone());
                                        }
                                    });
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

            if let Some(txt) = push_board_text {
                let steps: Vec<String> = txt
                    .lines()
                    .map(|l| l.trim().to_string())
                    .filter(|l| !l.is_empty())
                    .take(6)
                    .collect();
                if steps.len() == 1 {
                    app.execute_whiteboard_actions(
                        &[WhiteboardAction::WriteOnBoard {
                            content: steps[0].clone(),
                            is_latex: true,
                            position: None,
                            font_size: 16.0,
                            color_rgba: None,
                        }],
                        None,
                    );
                } else if !steps.is_empty() {
                    app.execute_whiteboard_actions(
                        &[WhiteboardAction::SolveOnBoard {
                            steps,
                            target_region: None,
                        }],
                        None,
                    );
                }
            }

            ui.add_space(4.0);
            ui.separator();

            // Compact Question & Agentic Command Input (`ghi lên bảng hộ tôi 2x + 1 = 3 đi`)
            let te_resp = ui.add(
                egui::TextEdit::singleline(&mut app.chat_input_draft)
                    .hint_text("Hỏi AI hoặc gõ: ghi lên bảng 2x + 1 = 3...")
                    .desired_width(f32::INFINITY),
            );

            ui.add_space(4.0);
            ui.horizontal(|ui| {
                let send_clicked = ui
                    .add(
                        egui::Button::new(
                            RichText::new("Gửi / Lệnh Bảng")
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
                    app.submit_user_chat_or_command(prompt);
                }
            });
        });
}
