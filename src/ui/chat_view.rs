//! Right Dock (`Trợ Lý Sư Phạm Gemini AI`, Collapsible, `Width = 330px` — MODULE 2.4 & MODULE 4):
//! - Compact API Key setup (auto-hides once key is saved, showing only `[Đổi API Key]`).
//! - Clean AI analysis output view (`ScrollArea`).
//! - 3 clear action buttons:
//!   * `[1. Giải & Chữa Bài Trên Bảng]`
//!   * `[2. Tóm Tắt & Tạo Trắc Nghiệm]`
//!   * `[3. Hỏi Đáp Sư Phạm]`

use crate::ai::gemini_chat::PedagogicalAnalysisMode;
use crate::app::AirTeachApp;
use crate::ui::toolbar_view::{
    ALERT_ROSE, BG_DARK, BORDER_SLATE, PRIMARY_INDIGO, SECONDARY_TEAL, SURFACE_DARK, TEXT_LIGHT,
    TEXT_MUTED,
};
use crate::utils::document_loader::DocumentLoader;
use egui::{Color32, FontId, RichText, Rounding, ScrollArea, Stroke, Vec2};

pub fn render_chat_drawer(app: &mut AirTeachApp, ctx: &egui::Context) {
    if !app.chat_drawer_open {
        return;
    }

    egui::SidePanel::right("airteach_modern_ai_copilot_panel")
        .resizable(false)
        .exact_width(330.0)
        .frame(
            egui::Frame::none()
                .fill(SURFACE_DARK)
                .stroke(Stroke::new(1.0_f32, BORDER_SLATE))
                .inner_margin(egui::Margin::same(10.0)),
        )
        .show(ctx, |ui| {
            // =========================================================================
            // 1. HEADER & COMPACT API KEY SETUP (AUTO-HIDES WHEN SAVED)
            // =========================================================================
            let has_saved_key = !app.config.encrypted_api_key_b64.trim().is_empty();
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Trợ Lý Sư Phạm Gemini AI")
                        .font(FontId::proportional(13.5))
                        .strong()
                        .color(TEXT_LIGHT),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("[Đóng]").size(11.5).color(TEXT_MUTED),
                            )
                            .fill(BG_DARK)
                            .rounding(Rounding::same(5.0)),
                        )
                        .clicked()
                    {
                        app.chat_drawer_open = false;
                    }

                    if has_saved_key
                        && ui
                            .add(
                                egui::Button::new(
                                    RichText::new("[Đổi API Key]")
                                        .size(11.0)
                                        .color(SECONDARY_TEAL),
                                )
                                .fill(BG_DARK)
                                .rounding(Rounding::same(5.0)),
                            )
                            .clicked()
                    {
                        app.config.encrypted_api_key_b64.clear();
                        app.api_key_edit_buffer.clear();
                    }
                });
            });

            if !has_saved_key {
                ui.add_space(4.0);
                egui::Frame::none()
                    .fill(BG_DARK)
                    .stroke(Stroke::new(1.0_f32, SECONDARY_TEAL))
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
                                            .color(TEXT_LIGHT)
                                            .strong(),
                                    )
                                    .fill(PRIMARY_INDIGO)
                                    .rounding(Rounding::same(5.0)),
                                )
                                .clicked()
                                && !app.api_key_edit_buffer.trim().is_empty()
                            {
                                app.storage.set_encrypted_api_key(
                                    &mut app.config,
                                    &app.api_key_edit_buffer,
                                );
                            }
                        });
                    });
            }

            ui.add_space(6.0);

            // =========================================================================
            // 2. THREE CLEAR PEDAGOGICAL ACTION BUTTONS (MODULE 2.4)
            // =========================================================================
            let mut selected_mode: Option<PedagogicalAnalysisMode> = None;
            for (mode, label) in [
                (
                    PedagogicalAnalysisMode::SolveAndCheckError,
                    "[1. Giải & Chữa Bài Trên Bảng]",
                ),
                (
                    PedagogicalAnalysisMode::SummarizeAndQuiz,
                    "[2. Tóm Tắt & Tạo Trắc Nghiệm]",
                ),
                (
                    PedagogicalAnalysisMode::PedagogicalInsights,
                    "[3. Hỏi Đáp Sư Phạm]",
                ),
            ] {
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new(label)
                                .font(FontId::proportional(12.5))
                                .color(TEXT_LIGHT)
                                .strong(),
                        )
                        .fill(BG_DARK)
                        .stroke(Stroke::new(1.0_f32, PRIMARY_INDIGO))
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
                            Color32::from_rgb(0x23, 0x2f, 0x46)
                        } else {
                            BG_DARK
                        };
                        let border_col = if is_user {
                            PRIMARY_INDIGO
                        } else {
                            BORDER_SLATE
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
                                        "[Giáo viên]"
                                    } else {
                                        "[Gemini 2.5 Flash]"
                                    };
                                    ui.label(
                                        RichText::new(role_lbl)
                                            .font(FontId::proportional(11.5))
                                            .strong()
                                            .color(if is_user {
                                                PRIMARY_INDIGO
                                            } else {
                                                SECONDARY_TEAL
                                            }),
                                    );
                                    if msg.has_image_attachment {
                                        let dim_txt = match msg.image_dimensions {
                                            Some((w, h)) => format!("[Ảnh vùng chọn {w}x{h} PNG]"),
                                            None => "[Ảnh vùng chọn PNG]".to_string(),
                                        };
                                        ui.label(
                                            RichText::new(dim_txt)
                                                .font(FontId::proportional(10.5))
                                                .color(ALERT_ROSE),
                                        );
                                    }
                                });
                                ui.add_space(4.0);
                                ui.label(
                                    RichText::new(&clean_text)
                                        .font(FontId::proportional(12.5))
                                        .color(TEXT_LIGHT),
                                );

                                if !is_user && !clean_text.trim().is_empty() {
                                    ui.add_space(4.0);
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                RichText::new("[Sao chép lời giải]")
                                                    .font(FontId::proportional(11.0))
                                                    .color(TEXT_LIGHT),
                                            )
                                            .fill(SURFACE_DARK)
                                            .stroke(Stroke::new(1.0_f32, BORDER_SLATE))
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
                                RichText::new("Đang phân tích nét vẽ vùng chọn với Gemini 2.5 Flash...")
                                    .font(FontId::proportional(11.5))
                                    .color(SECONDARY_TEAL),
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
                            RichText::new("[Gửi câu hỏi]")
                                .font(FontId::proportional(12.0))
                                .color(TEXT_LIGHT)
                                .strong(),
                        )
                        .fill(PRIMARY_INDIGO)
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
