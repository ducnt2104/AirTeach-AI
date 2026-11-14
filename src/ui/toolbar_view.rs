//! Minimalist Studio Header Bar & Bottom Drawing Dock (`src/ui/toolbar_view.rs` — MODULE 2.1):
//! - Single Header Bar (`Height: 46px`, zero 2nd Sub-Toolbar!):
//!   * Left: `[Đăng xuất]` | Tên Giáo viên đã đăng nhập | `[Mở PDF/Slide]` | `[ < ] Slide 1/4 [ > ]` | `[Bật/Tắt Lưới Caro]`
//!   * Middle: Flexible stretch spacer so left and right controls never overlap
//!   * Right: `[Ghi Âm Bài Giảng]` | `[Ẩn/Hiện Camera]` | `[Trợ Lý AI (Đóng/Mở)]`
//! - Bottom Drawing Dock (`Height: 54px`):
//!   * Group 1: `[Phân Tích Vùng]` | `[Bút vẽ]` | `[Tẩy]` | `[Laser]` | `[Nắn hình: TẮT/BẬT]`
//!   * Group 2: 6 circular color buttons (`22x22px`) + thickness slider (`2px - 14px`)
//!   * Group 3: `[Hoàn tác (Undo)]` | `[Làm lại (Redo)]` | `[Xóa Bảng]`

use crate::app::{ActiveBoardTool, AirTeachApp};
use crate::core::audio_recorder::RecordingState;
use crate::ui::brand::{render_about_modal, ui_brand_logo, ThemeMode, ThemePalette};
use crate::utils::storage::BoardLayoutMode;
use egui::{Color32, FontId, RichText, Rounding, Sense, Stroke, Vec2};

pub const BG_DARK: Color32 = Color32::from_rgb(0x0f, 0x17, 0x2a); // Slate-900 (#0f172a)
pub const SURFACE_DARK: Color32 = Color32::from_rgb(0x1e, 0x29, 0x3b); // Slate-800 (#1e293b)
pub const BORDER_SLATE: Color32 = Color32::from_rgb(0x33, 0x41, 0x55); // Slate-700 (#334155)
pub const PRIMARY_INDIGO: Color32 = Color32::from_rgb(0x4f, 0x46, 0xe5); // Indigo-600 (#4f46e5)
pub const ALERT_ROSE: Color32 = Color32::from_rgb(0xf4, 0x3f, 0x5e); // Rose-500 (#f43f5e)
pub const SECONDARY_TEAL: Color32 = Color32::from_rgb(0x06, 0xb6, 0xd4); // Teal-500 (#06b6d4)
pub const TEXT_LIGHT: Color32 = Color32::from_rgb(0xf8, 0xfa, 0xfc); // Slate-50 (#f8fafc)
pub const TEXT_MUTED: Color32 = Color32::from_rgb(0x94, 0xa3, 0xb8); // Slate-400 (#94a3b8)

pub fn render_top_toolbar(app: &mut AirTeachApp, ctx: &egui::Context) {
    // =========================================================================
    // SINGLE COMPACT HEADER BAR (Height: 46px, Zero Sub-Toolbar — MODULE 2.1)
    // =========================================================================
    let palette = app.config.theme_mode.palette();

    egui::TopBottomPanel::top("airteach_studio_header_bar")
        .exact_height(46.0)
        .frame(
            egui::Frame::none()
                .fill(palette.bg_panel)
                .stroke(Stroke::new(1.0_f32, palette.border))
                .inner_margin(egui::Margin::symmetric(12.0, 6.0)),
        )
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            ui.horizontal_centered(|ui| {
                // Brand Logo + Quick About trigger
                if ui_brand_logo(ui, 26.0)
                    .on_hover_text("AirTeach AI — Sản phẩm của DNT - Victer (Nhấn để xem Giới thiệu)")
                    .interact(Sense::click())
                    .clicked()
                {
                    app.about_modal_open = true;
                }

                // LEFT GROUP: [Đăng xuất] | Tên Giáo viên | [Mở PDF/Slide] | [ < ] Slide 1/n [ > ] | [Bật/Tắt Lưới Caro]
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("[Đăng xuất]")
                                .color(palette.danger)
                                .font(FontId::proportional(12.5))
                                .strong(),
                        )
                        .fill(palette.surface)
                        .stroke(Stroke::new(1.0_f32, palette.danger))
                        .rounding(Rounding::same(6.0))
                        .min_size(Vec2::new(0.0, 30.0)),
                    )
                    .on_hover_text("Đăng xuất, xóa phiên làm việc & giải phóng Camera")
                    .clicked()
                {
                    app.logout_to_auth_gate();
                }

                let teacher_display = if app.logged_in_teacher_name.trim().is_empty() {
                    "Giảng viên"
                } else {
                    app.logged_in_teacher_name.as_str()
                };
                ui.label(
                    RichText::new(format!("GV: {teacher_display}"))
                        .font(FontId::proportional(13.0))
                        .color(palette.secondary)
                        .strong(),
                );

                ui.separator();

                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("[Mở PDF/Slide]")
                                .font(FontId::proportional(12.5))
                                .color(palette.text_primary)
                                .strong(),
                        )
                        .fill(palette.surface)
                        .stroke(Stroke::new(1.0_f32, palette.border))
                        .rounding(Rounding::same(6.0))
                        .min_size(Vec2::new(0.0, 30.0)),
                    )
                    .clicked()
                {
                    app.open_slide_file_dialog();
                }

                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("<")
                                .font(FontId::proportional(13.5))
                                .color(palette.text_primary)
                                .strong(),
                        )
                        .fill(palette.surface)
                        .stroke(Stroke::new(1.0_f32, palette.border))
                        .rounding(Rounding::same(6.0))
                        .min_size(Vec2::new(28.0, 30.0)),
                    )
                    .clicked()
                    && app.current_slide_idx > 0
                {
                    app.current_slide_idx -= 1;
                    app.invalidate_slide_texture();
                }

                ui.label(
                    RichText::new(format!(
                        "Slide {}/{}",
                        app.current_slide_idx + 1,
                        app.slides.len().max(1)
                    ))
                    .font(FontId::proportional(13.0))
                    .color(palette.text_primary)
                    .strong(),
                );

                if ui
                    .add(
                        egui::Button::new(
                            RichText::new(">")
                                .font(FontId::proportional(13.5))
                                .color(palette.text_primary)
                                .strong(),
                        )
                        .fill(palette.surface)
                        .stroke(Stroke::new(1.0_f32, palette.border))
                        .rounding(Rounding::same(6.0))
                        .min_size(Vec2::new(28.0, 30.0)),
                    )
                    .clicked()
                    && app.current_slide_idx + 1 < app.slides.len()
                {
                    app.current_slide_idx += 1;
                    app.invalidate_slide_texture();
                }

                let grid_active = app.config.canvas_grid_enabled;
                let (grid_fill, grid_text_col) = if grid_active {
                    (palette.primary, Color32::WHITE)
                } else {
                    (palette.surface, palette.text_primary)
                };
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("[Bật/Tắt Lưới Caro]")
                                .font(FontId::proportional(12.5))
                                .color(grid_text_col)
                                .strong(),
                        )
                        .fill(grid_fill)
                        .stroke(Stroke::new(1.0_f32, palette.border))
                        .rounding(Rounding::same(6.0))
                        .min_size(Vec2::new(0.0, 30.0)),
                    )
                    .clicked()
                {
                    app.config.canvas_grid_enabled = !app.config.canvas_grid_enabled;
                    app.storage.save_config(&app.config);
                }

                // MIDDLE FLEXIBLE STRETCH + RIGHT GROUP:
                // [Ghi Âm Bài Giảng] | [Ẩn/Hiện Camera] | [Trợ Lý AI (Đóng/Mở)]
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (ai_fill, ai_text_col) = if app.chat_drawer_open {
                        (palette.primary, Color32::WHITE)
                    } else {
                        (palette.surface, palette.text_primary)
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("[Trợ Lý AI (Đóng/Mở)]")
                                    .font(FontId::proportional(12.5))
                                    .color(ai_text_col)
                                    .strong(),
                            )
                            .fill(ai_fill)
                            .stroke(Stroke::new(1.0_f32, palette.primary))
                            .rounding(Rounding::same(6.0))
                            .min_size(Vec2::new(0.0, 30.0)),
                        )
                        .clicked()
                    {
                        app.chat_drawer_open = !app.chat_drawer_open;
                        app.config.layout_mode = if app.chat_drawer_open {
                            BoardLayoutMode::SplitScreenAi
                        } else {
                            BoardLayoutMode::SlidePresentation
                        };
                    }

                    let (cam_fill, cam_text_col) = if app.left_camera_dock_open {
                        let on_sec = match palette.mode {
                            ThemeMode::Dark => Color32::from_rgb(0x0f, 0x17, 0x2a),
                            ThemeMode::Light => Color32::WHITE,
                        };
                        (palette.secondary, on_sec)
                    } else {
                        (palette.surface, palette.text_primary)
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("[Ẩn/Hiện Camera]")
                                    .font(FontId::proportional(12.5))
                                    .color(cam_text_col)
                                    .strong(),
                            )
                            .fill(cam_fill)
                            .stroke(Stroke::new(1.0_f32, palette.secondary))
                            .rounding(Rounding::same(6.0))
                            .min_size(Vec2::new(0.0, 30.0)),
                        )
                        .clicked()
                    {
                        app.left_camera_dock_open = !app.left_camera_dock_open;
                        if app.left_camera_dock_open {
                            app.ensure_camera_acquired();
                        } else {
                            app.release_camera_hardware();
                        }
                    }

                    render_audio_recorder_widget(app, ui, ctx, palette);
                });
            });
        });

    render_about_modal(ctx, &mut app.about_modal_open, palette);
}

/// Non-Overlapping Bottom Drawing Dock (`Height: 54px`, theme-aware).
pub fn render_bottom_drawing_dock(app: &mut AirTeachApp, ctx: &egui::Context) {
    let palette = app.config.theme_mode.palette();

    egui::TopBottomPanel::bottom("airteach_bottom_drawing_dock")
        .exact_height(54.0)
        .frame(
            egui::Frame::none()
                .fill(palette.bg_panel)
                .stroke(Stroke::new(1.0_f32, palette.border))
                .inner_margin(egui::Margin::symmetric(12.0, 8.0)),
        )
        .show(ctx, |ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            ui.horizontal_centered(|ui| {
                // GROUP 1: [Phân Tích Vùng] | [Bút vẽ] | [Tẩy] | [Laser] | [Nắn hình: TẮT/BẬT]
                for (tool, label) in [
                    (ActiveBoardTool::SelectRegion, "[Phân Tích Vùng]"),
                    (ActiveBoardTool::Pen, "[Bút vẽ]"),
                    (ActiveBoardTool::Eraser, "[Tẩy]"),
                    (ActiveBoardTool::LaserPointer, "[Laser]"),
                ] {
                    let selected = app.active_tool == tool;
                    let (fill, text_col) = if selected {
                        let bg = if tool == ActiveBoardTool::SelectRegion {
                            palette.danger
                        } else {
                            palette.primary
                        };
                        (bg, Color32::WHITE)
                    } else {
                        (palette.surface, palette.text_primary)
                    };
                    let border_col = if selected {
                        palette.text_primary
                    } else if tool == ActiveBoardTool::SelectRegion {
                        palette.danger
                    } else {
                        palette.border
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new(label)
                                    .color(text_col)
                                    .font(FontId::proportional(12.5))
                                    .strong(),
                            )
                            .fill(fill)
                            .stroke(Stroke::new(1.2_f32, border_col))
                            .rounding(Rounding::same(7.0))
                            .min_size(Vec2::new(0.0, 32.0)),
                        )
                        .clicked()
                    {
                        app.active_tool = tool;
                    }
                }

                let (snap_fill, snap_text_col) = if app.config.auto_shape_snap {
                    (Color32::from_rgb(13, 148, 136), Color32::WHITE)
                } else {
                    (palette.surface, palette.text_primary)
                };
                let snap_label = if app.config.auto_shape_snap {
                    "[Nắn hình: BẬT]"
                } else {
                    "[Nắn hình: TẮT]"
                };
                if ui
                    .add(
                        egui::Button::new(
                            RichText::new(snap_label)
                                .color(snap_text_col)
                                .font(FontId::proportional(12.5))
                                .strong(),
                        )
                        .fill(snap_fill)
                        .stroke(Stroke::new(1.0_f32, palette.secondary))
                        .rounding(Rounding::same(7.0))
                        .min_size(Vec2::new(0.0, 32.0)),
                    )
                    .clicked()
                {
                    app.config.auto_shape_snap = !app.config.auto_shape_snap;
                    app.sync_stroke_engine_config();
                    app.storage.save_config(&app.config);
                }

                if app.config.auto_shape_snap {
                    let hold_fill = if app.config.hold_to_snap_350ms {
                        palette.primary
                    } else {
                        palette.surface
                    };
                    let hold_col = if app.config.hold_to_snap_350ms {
                        Color32::WHITE
                    } else {
                        palette.text_muted
                    };
                    if ui
                        .add(
                            egui::Button::new(
                                RichText::new("Giữ 350ms")
                                    .color(hold_col)
                                    .font(FontId::proportional(11.5))
                                    .strong(),
                            )
                            .fill(hold_fill)
                            .stroke(Stroke::new(1.0_f32, palette.border))
                            .rounding(Rounding::same(7.0))
                            .min_size(Vec2::new(0.0, 32.0)),
                        )
                        .on_hover_text("Bật: vẽ xong giữ tay 350ms mới nắn chuẩn hình")
                        .clicked()
                    {
                        app.config.hold_to_snap_350ms = !app.config.hold_to_snap_350ms;
                        app.storage.save_config(&app.config);
                    }
                }

                ui.separator();

                // GROUP 2: 6 Circular Color Buttons (22x22px) + Thickness Slider (2px - 14px)
                let ink_swatches: [([u8; 4], &str); 6] = [
                    ([0x00, 0xf5, 0xd4, 255], "Cyan (#00F5D4)"),
                    ([0xfe, 0xe4, 0x40, 255], "Vàng (#FEE440)"),
                    ([0xff, 0xff, 0xff, 255], "Trắng (#FFFFFF)"),
                    ([0x38, 0xb0, 0x00, 255], "Xanh lá (#38B000)"),
                    ([0xff, 0x00, 0x7f, 255], "Hồng (#FF007F)"),
                    ([0xff, 0x6d, 0x00, 255], "Cam (#FF6D00)"),
                ];

                for (rgba, tooltip) in ink_swatches {
                    let c = Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3]);
                    let is_active = app.current_ink_color == rgba;
                    let stroke = if is_active {
                        Stroke::new(2.4_f32, palette.text_primary)
                    } else {
                        Stroke::new(1.0_f32, palette.border)
                    };
                    let btn = egui::Button::new("")
                        .fill(c)
                        .stroke(stroke)
                        .rounding(Rounding::same(11.0))
                        .min_size(Vec2::new(22.0, 22.0));
                    if ui.add(btn).on_hover_text(tooltip).clicked() {
                        app.current_ink_color = rgba;
                        if app.active_tool == ActiveBoardTool::Eraser {
                            app.active_tool = ActiveBoardTool::Pen;
                        }
                    }
                }

                ui.add_space(2.0);
                ui.label(
                    RichText::new(format!("{:.0}px", app.current_stroke_thickness))
                        .font(FontId::proportional(12.0))
                        .color(palette.secondary)
                        .strong(),
                );
                ui.spacing_mut().slider_width = 88.0;
                ui.add(
                    egui::Slider::new(&mut app.current_stroke_thickness, 2.0..=14.0)
                        .show_value(false),
                );

                ui.separator();

                // GROUP 3: [Hoàn tác (Undo)] | [Làm lại (Redo)] | [Xóa Bảng]
                    if ui
                    .add(
                        egui::Button::new(
                            RichText::new("[Hoàn tác (Undo)]")
                                .color(palette.text_primary)
                                .font(FontId::proportional(12.5))
                                .strong(),
                        )
                        .fill(palette.surface)
                        .stroke(Stroke::new(1.0_f32, palette.border))
                        .rounding(Rounding::same(7.0))
                        .min_size(Vec2::new(0.0, 32.0)),
                    )
                    .on_hover_text("Ctrl+Z hoặc xòe 3 ngón (cooldown 400ms)")
                    .clicked()
                {
                    app.perform_undo_with_cooldown();
                }

                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("[Làm lại (Redo)]")
                                .color(palette.text_primary)
                                .font(FontId::proportional(12.5))
                                .strong(),
                        )
                        .fill(palette.surface)
                        .stroke(Stroke::new(1.0_f32, palette.border))
                        .rounding(Rounding::same(7.0))
                        .min_size(Vec2::new(0.0, 32.0)),
                    )
                    .clicked()
                {
                    if let Some(restored) = app.redo_strokes.pop() {
                        app.strokes.push(restored);
                    }
                }

                if ui
                    .add(
                        egui::Button::new(
                            RichText::new("[Xóa Bảng]")
                                .color(palette.danger)
                                .font(FontId::proportional(12.5))
                                .strong(),
                        )
                        .fill(palette.surface)
                        .stroke(Stroke::new(1.0_f32, palette.danger))
                        .rounding(Rounding::same(7.0))
                        .min_size(Vec2::new(0.0, 32.0)),
                    )
                    .clicked()
                {
                    app.strokes.clear();
                    app.redo_strokes.clear();
                    app.annotations.clear();
                }
            });
        });
}

fn render_audio_recorder_widget(
    app: &mut AirTeachApp,
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    palette: ThemePalette,
) {
    let rec_state = app.audio_recorder.state();
    let duration_s = app.audio_recorder.recorded_duration_secs();
    let mins = (duration_s as u32) / 60;
    let secs = (duration_s as u32) % 60;

    match rec_state {
        RecordingState::Stopped => {
            if ui
                .add(
                    egui::Button::new(
                        RichText::new("[Ghi Âm Bài Giảng]")
                            .font(FontId::proportional(12.5))
                            .color(Color32::WHITE)
                            .strong(),
                    )
                    .fill(palette.danger)
                    .rounding(Rounding::same(6.0))
                    .min_size(Vec2::new(0.0, 30.0)),
                )
                .clicked()
            {
                app.start_synchronized_lecture_recording();
            }
        }
        RecordingState::Recording => {
            let t = ctx.input(|i| i.time) as f32;
            let pulse = (t * 5.0).sin() * 0.5 + 0.5;
            let _ = pulse;
            if ui
                .add(
                    egui::Button::new(
                        RichText::new(format!("[Đang Ghi Âm {mins:02}:{secs:02}]"))
                            .font(FontId::proportional(12.5))
                            .color(Color32::WHITE)
                            .strong(),
                    )
                    .fill(palette.danger)
                    .rounding(Rounding::same(6.0))
                    .min_size(Vec2::new(0.0, 30.0)),
                )
                .clicked()
            {
                app.audio_recorder.pause();
            }
        }
        RecordingState::Paused => {
            if ui
                .add(
                    egui::Button::new(
                        RichText::new(format!("[Tiếp tục Ghi {mins:02}:{secs:02}]"))
                            .font(FontId::proportional(12.5))
                            .color(Color32::WHITE)
                            .strong(),
                    )
                    .fill(palette.primary)
                    .rounding(Rounding::same(6.0))
                    .min_size(Vec2::new(0.0, 30.0)),
                )
                .clicked()
            {
                app.audio_recorder.start();
            }
        }
    }
}
