//! Brand Identity & Design System Tokens (`src/ui/brand.rs`):
//! - Bespoke vector brand mark for `AirTeach AI` by `DNT - Victer` (mirrored from `assets/branding/airteach_logo.svg`)
//! - Unified Dark & Light Design System tokens (colors, spacing, radii, typography)
//! - Crisp SSAA rasterizer for Windows taskbar/titlebar `IconData`
//! - Reusable Brand Badge, Corner Attribution Footer & About Modal ("Giới thiệu")

use egui::{Color32, FontId, Pos2, Rect, RichText, Rounding, Stroke, Vec2};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

pub const APP_BRAND_NAME: &str = "AirTeach AI";
pub const ORG_BRAND_NAME: &str = "DNT - Victer";
pub const APP_TAGLINE_VI: &str = "Bảng tương tác không chạm & Trợ lý sư phạm AI";
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Embedded vector SVG logo source (`assets/branding/airteach_logo.svg`).
pub const EMBEDDED_LOGO_SVG: &str = include_str!("../../assets/branding/airteach_logo.svg");

// Design System Spacing & Radius Tokens
pub const SPACE_XS: f32 = 4.0;
pub const SPACE_SM: f32 = 8.0;
pub const SPACE_MD: f32 = 16.0;
pub const SPACE_LG: f32 = 24.0;
pub const SPACE_XL: f32 = 32.0;

pub const RADIUS_SM: f32 = 6.0;
pub const RADIUS_MD: f32 = 10.0;
pub const RADIUS_LG: f32 = 16.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThemeMode {
    Dark,
    Light,
}

impl Default for ThemeMode {
    fn default() -> Self {
        Self::Dark
    }
}

impl ThemeMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Dark => "Tối (Studio Dark)",
            Self::Light => "Sáng (Daylight Clean)",
        }
    }

    pub const fn toggle(self) -> Self {
        match self {
            Self::Dark => Self::Light,
            Self::Light => Self::Dark,
        }
    }

    pub const fn palette(self) -> ThemePalette {
        match self {
            Self::Dark => ThemePalette {
                mode: Self::Dark,
                bg_canvas: Color32::from_rgb(0x0B, 0x11, 0x20),
                bg_panel: Color32::from_rgb(0x0F, 0x17, 0x2A),
                surface: Color32::from_rgb(0x1E, 0x29, 0x3B),
                surface_elevated: Color32::from_rgb(0x27, 0x35, 0x49),
                border: Color32::from_rgb(0x33, 0x41, 0x55),
                primary: Color32::from_rgb(0x63, 0x66, 0xF1),
                secondary: Color32::from_rgb(0x06, 0xB6, 0xD4),
                success: Color32::from_rgb(0x22, 0xC5, 0x5E),
                warning: Color32::from_rgb(0xF5, 0x9E, 0x0B),
                danger: Color32::from_rgb(0xF4, 0x3F, 0x5E),
                text_primary: Color32::from_rgb(0xF8, 0xFA, 0xFC),
                text_secondary: Color32::from_rgb(0xCB, 0xD5, 0xE1),
                text_muted: Color32::from_rgb(0x94, 0xA3, 0xB8),
            },
            Self::Light => ThemePalette {
                mode: Self::Light,
                bg_canvas: Color32::from_rgb(0xF1, 0xF5, 0xF9),
                bg_panel: Color32::from_rgb(0xFF, 0xFF, 0xFF),
                surface: Color32::from_rgb(0xFF, 0xFF, 0xFF),
                surface_elevated: Color32::from_rgb(0xE2, 0xE8, 0xF0),
                border: Color32::from_rgb(0xCB, 0xD5, 0xE1),
                primary: Color32::from_rgb(0x4F, 0x46, 0xE5),
                secondary: Color32::from_rgb(0x08, 0x91, 0xB2),
                success: Color32::from_rgb(0x16, 0xA3, 0x4A),
                warning: Color32::from_rgb(0xD9, 0x77, 0x06),
                danger: Color32::from_rgb(0xE1, 0x1D, 0x48),
                text_primary: Color32::from_rgb(0x0F, 0x17, 0x2A),
                text_secondary: Color32::from_rgb(0x33, 0x41, 0x55),
                text_muted: Color32::from_rgb(0x64, 0x74, 0x8B),
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ThemePalette {
    pub mode: ThemeMode,
    pub bg_canvas: Color32,
    pub bg_panel: Color32,
    pub surface: Color32,
    pub surface_elevated: Color32,
    pub border: Color32,
    pub primary: Color32,
    pub secondary: Color32,
    pub success: Color32,
    pub warning: Color32,
    pub danger: Color32,
    pub text_primary: Color32,
    pub text_secondary: Color32,
    pub text_muted: Color32,
}

impl ThemePalette {
    /// Apply coherent egui visuals so standard widgets (scrollbars, inputs, buttons) match the active theme.
    pub fn apply_to_egui(self, ctx: &egui::Context) {
        let mut visuals = match self.mode {
            ThemeMode::Dark => egui::Visuals::dark(),
            ThemeMode::Light => egui::Visuals::light(),
        };
        visuals.panel_fill = self.bg_panel;
        visuals.window_fill = self.surface;
        visuals.extreme_bg_color = self.bg_canvas;
        visuals.faint_bg_color = self.surface_elevated;
        visuals.widgets.noninteractive.bg_fill = self.surface;
        visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, self.border);
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, self.text_primary);
        visuals.widgets.inactive.bg_fill = self.surface_elevated;
        visuals.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, self.border);
        visuals.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, self.text_primary);
        visuals.widgets.hovered.bg_fill = self.primary;
        visuals.widgets.hovered.fg_stroke = Stroke::new(1.5_f32, Color32::WHITE);
        visuals.widgets.active.bg_fill = self.primary;
        visuals.widgets.active.fg_stroke = Stroke::new(1.5_f32, Color32::WHITE);
        visuals.selection.bg_fill = self.primary.linear_multiply(0.35);
        visuals.selection.stroke = Stroke::new(1.5_f32, self.primary);
        visuals.window_rounding = Rounding::same(RADIUS_MD);
        visuals.widgets.noninteractive.rounding = Rounding::same(RADIUS_SM);
        visuals.widgets.inactive.rounding = Rounding::same(RADIUS_SM);
        visuals.widgets.hovered.rounding = Rounding::same(RADIUS_SM);
        visuals.widgets.active.rounding = Rounding::same(RADIUS_SM);
        ctx.set_visuals(visuals);
    }
}

/// Resolution-independent vector rendering of the AirTeach AI (DNT - Victer) logo
/// mirroring `assets/branding/airteach_logo.svg` in any egui UI container.
pub fn paint_brand_logo(painter: &egui::Painter, center: Pos2, size: f32) {
    let s = size / 128.0;
    let p = |x: f32, y: f32| Pos2::new(center.x + (x - 64.0) * s, center.y + (y - 64.0) * s);

    // 1. Rounded Smartboard badge background + cyan/indigo stroke
    let badge_rect = Rect::from_center_size(center, Vec2::splat(112.0 * s));
    painter.rect(
        badge_rect,
        Rounding::same(26.0 * s),
        Color32::from_rgb(0x0F, 0x17, 0x2A),
        Stroke::new((3.5 * s).max(1.2), Color32::from_rgb(0x38, 0xBD, 0xF8)),
    );

    // 2. Inner interactive board frame highlight
    if size >= 28.0 {
        let inner_rect = Rect::from_min_max(p(20.0, 20.0), p(108.0, 80.0));
        painter.rect_stroke(
            inner_rect,
            Rounding::same(10.0 * s),
            Stroke::new((1.6 * s).max(0.8), Color32::from_rgba_unmultiplied(0x47, 0x55, 0x69, 170)),
        );
    }

    // 3. Touchless concentric optical waves around index tip (64, 42)
    let arc_pts = |cx: f32, cy: f32, r: f32, start_deg: f32, end_deg: f32| -> Vec<Pos2> {
        let steps = 14usize;
        (0..=steps)
            .map(|i| {
                let t = i as f32 / steps as f32;
                let deg = start_deg + (end_deg - start_deg) * t;
                let rad = deg * PI / 180.0;
                p(cx + r * rad.cos(), cy + r * rad.sin())
            })
            .collect()
    };

    painter.add(egui::Shape::line(
        arc_pts(64.0, 42.0, 16.0, -180.0, 0.0),
        Stroke::new((2.8 * s).max(1.2), Color32::from_rgb(0x06, 0xB6, 0xD4)),
    ));
    painter.add(egui::Shape::line(
        arc_pts(64.0, 42.0, 25.0, -172.0, -8.0),
        Stroke::new((2.4 * s).max(1.0), Color32::from_rgb(0x63, 0x66, 0xF1)),
    ));

    // 4. Stylized raised index finger + palm silhouette
    let palm_rect = Rect::from_min_max(p(48.0, 66.0), p(88.0, 108.0));
    painter.rect_filled(
        palm_rect,
        Rounding::same(12.0 * s),
        Color32::from_rgb(0x22, 0xD3, 0xEE),
    );
    let finger_rect = Rect::from_min_max(p(56.0, 36.0), p(72.0, 74.0));
    painter.rect_filled(
        finger_rect,
        Rounding::same(8.0 * s),
        Color32::from_rgb(0xE0, 0xF2, 0xFE),
    );

    // Thumb gesture accent
    painter.line_segment(
        [p(41.0, 75.0), p(51.0, 84.0)],
        Stroke::new((4.0 * s).max(1.4), Color32::from_rgb(0xE0, 0xF2, 0xFE)),
    );

    // 5. Focal fingertip spark + top-right AI star node
    painter.circle_filled(p(64.0, 42.0), (5.0 * s).max(1.6), Color32::from_rgb(0xFD, 0xE0, 0x47));
    if size >= 24.0 {
        let star = vec![
            p(94.0, 26.0),
            p(96.0, 31.0),
            p(101.0, 33.0),
            p(96.0, 35.0),
            p(94.0, 40.0),
            p(92.0, 35.0),
            p(87.0, 33.0),
            p(92.0, 31.0),
        ];
        painter.add(egui::Shape::convex_polygon(
            star,
            Color32::from_rgb(0xFD, 0xE0, 0x47),
            Stroke::NONE,
        ));
    }
}

/// Draw vector logo widget inside egui layout and return its response (e.g. for click/tooltip).
pub fn ui_brand_logo(ui: &mut egui::Ui, size: f32) -> egui::Response {
    let (rect, response) = ui.allocate_exact_size(Vec2::splat(size), egui::Sense::hover());
    if ui.is_rect_visible(rect) {
        paint_brand_logo(ui.painter(), rect.center(), size);
    }
    response
}

/// High-resolution 2x SSAA rasterized RGBA icon matching `assets/branding/airteach_logo.svg`
/// for the native Windows taskbar / window titlebar icon.
pub fn rasterize_brand_logo_icon(size: u32) -> egui::IconData {
    let size = size.max(16);
    let mut rgba = vec![0u8; (size * size * 4) as usize];

    for py in 0..size {
        for px in 0..size {
            let mut acc_r = 0.0_f32;
            let mut acc_g = 0.0_f32;
            let mut acc_b = 0.0_f32;
            let mut acc_a = 0.0_f32;

            // 2x2 subpixel supersampling in normalized [0..128] SVG coordinate space
            for sy in 0..2 {
                for sx in 0..2 {
                    let u = ((px as f32 + (sx as f32 + 0.5) * 0.5) / size as f32) * 128.0;
                    let v = ((py as f32 + (sy as f32 + 0.5) * 0.5) / size as f32) * 128.0;
                    let [r, g, b, a] = sample_brand_svg_pixel(u, v);
                    acc_r += r;
                    acc_g += g;
                    acc_b += b;
                    acc_a += a;
                }
            }

            let idx = ((py * size + px) * 4) as usize;
            rgba[idx] = (acc_r * 0.25).round().clamp(0.0, 255.0) as u8;
            rgba[idx + 1] = (acc_g * 0.25).round().clamp(0.0, 255.0) as u8;
            rgba[idx + 2] = (acc_b * 0.25).round().clamp(0.0, 255.0) as u8;
            rgba[idx + 3] = (acc_a * 0.25).round().clamp(0.0, 255.0) as u8;
        }
    }

    egui::IconData {
        rgba,
        width: size,
        height: size,
    }
}

fn sd_rounded_rect(px: f32, py: f32, cx: f32, cy: f32, half_w: f32, half_h: f32, r: f32) -> f32 {
    let qx = (px - cx).abs() - (half_w - r);
    let qy = (py - cy).abs() - (half_h - r);
    (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt() + qx.max(qy).min(0.0) - r
}

fn sample_brand_svg_pixel(x: f32, y: f32) -> [f32; 4] {
    let outer_sd = sd_rounded_rect(x, y, 64.0, 64.0, 56.0, 56.0, 26.0);
    if outer_sd > 1.0 {
        return [0.0, 0.0, 0.0, 0.0];
    }

    let t = ((x + y) / 256.0).clamp(0.0, 1.0);
    let mut r = 15.0 * (1.0 - t) + 49.0 * t;
    let mut g = 23.0 * (1.0 - t) + 46.0 * t;
    let mut b = 42.0 * (1.0 - t) + 129.0 * t;

    // Outer gradient border ring
    if outer_sd > -3.8 {
        r = 56.0 * (1.0 - t) + 99.0 * t;
        g = 189.0 * (1.0 - t) + 102.0 * t;
        b = 248.0 * (1.0 - t) + 241.0 * t;
    }

    // Concentric tracking waves centered at (64, 42)
    let dx = x - 64.0;
    let dy = y - 42.0;
    let dist = (dx * dx + dy * dy).sqrt();
    if y <= 43.0 {
        if (dist - 16.0).abs() <= 1.8 {
            r = 6.0;
            g = 182.0;
            b = 212.0;
        } else if (dist - 25.0).abs() <= 1.5 {
            r = 99.0;
            g = 102.0;
            b = 241.0;
        }
    }

    // Stylized palm + raised index finger
    let palm_sd = sd_rounded_rect(x, y, 68.0, 87.0, 20.0, 21.0, 11.0);
    let finger_sd = sd_rounded_rect(x, y, 64.0, 55.0, 8.0, 19.0, 8.0);
    if palm_sd <= 0.0 {
        r = 34.0;
        g = 211.0;
        b = 238.0;
    }
    if finger_sd <= 0.0 {
        r = 224.0;
        g = 242.0;
        b = 254.0;
    }

    // Focal fingertip orb + AI star highlight
    if dist <= 5.0 || ((x - 94.0).abs() + (y - 33.0).abs()) <= 5.0 {
        r = 253.0;
        g = 224.0;
        b = 71.0;
    }

    let alpha = if outer_sd > -0.8 {
        (1.0 - outer_sd.max(0.0)).clamp(0.0, 1.0) * 255.0
    } else {
        255.0
    };
    [r, g, b, alpha]
}

/// Subtle bottom/corner organization badge for Auth & Welcome screens.
pub fn render_brand_corner_footer(ui: &mut egui::Ui, palette: ThemePalette, about_open: &mut bool) {
    ui.horizontal(|ui| {
        ui_brand_logo(ui, 20.0);
        ui.label(
            RichText::new(format!("{APP_BRAND_NAME} · Developed by {ORG_BRAND_NAME} · v{APP_VERSION}"))
                .font(FontId::proportional(11.5))
                .color(palette.text_muted),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui
                .add(
                    egui::Button::new(
                        RichText::new("Giới thiệu")
                            .font(FontId::proportional(11.5))
                            .color(palette.secondary),
                    )
                    .fill(Color32::TRANSPARENT)
                    .stroke(Stroke::new(1.0_f32, palette.border))
                    .rounding(Rounding::same(RADIUS_SM)),
                )
                .clicked()
            {
                *about_open = true;
            }
        });
    });
}

/// About / Giới thiệu dialog showcasing AirTeach AI & DNT - Victer identity.
pub fn render_about_modal(ctx: &egui::Context, open: &mut bool, palette: ThemePalette) {
    if !*open {
        return;
    }
    egui::Window::new(format!("Giới thiệu — {APP_BRAND_NAME}"))
        .open(open)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
        .fixed_size(Vec2::new(460.0, 320.0))
        .frame(
            egui::Frame::window(&ctx.style())
                .fill(palette.surface)
                .stroke(Stroke::new(1.2_f32, palette.border))
                .rounding(Rounding::same(RADIUS_LG))
                .inner_margin(egui::Margin::same(SPACE_LG)),
        )
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui_brand_logo(ui, 64.0);
                ui.add_space(SPACE_MD);
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(APP_BRAND_NAME)
                            .font(FontId::proportional(22.0))
                            .strong()
                            .color(palette.text_primary),
                    );
                    ui.label(
                        RichText::new(format!("Tổ chức phát triển: {ORG_BRAND_NAME}"))
                            .font(FontId::proportional(13.5))
                            .strong()
                            .color(palette.secondary),
                    );
                    ui.label(
                        RichText::new(format!("Phiên bản: v{APP_VERSION} (Local-First Desktop)"))
                            .font(FontId::monospace(11.5))
                            .color(palette.text_muted),
                    );
                });
            });
            ui.add_space(SPACE_MD);
            ui.separator();
            ui.add_space(SPACE_SM);
            ui.label(
                RichText::new(APP_TAGLINE_VI)
                    .font(FontId::proportional(14.0))
                    .strong()
                    .color(palette.text_primary),
            );
            ui.add_space(SPACE_XS);
            for bullet in [
                "• Nhận diện 21 điểm khớp bàn tay với 1-Euro Filter chuẩn hóa tỷ lệ lòng bàn tay & FSM chống rung.",
                "• Bảo mật Local-First: SQLite cục bộ, mật khẩu Argon2id, chỉ lưu vector khuôn mặt chuẩn hóa (không lưu ảnh thô).",
                "• Tích hợp bảng tương tác không chạm, ghi âm bài giảng & Trợ lý Sư phạm Gemini AI.",
            ] {
                ui.label(
                    RichText::new(bullet)
                        .font(FontId::proportional(12.5))
                        .color(palette.text_secondary),
                );
            }
            ui.add_space(SPACE_MD);
            ui.label(
                RichText::new(format!("© {ORG_BRAND_NAME} — Educational Technology Initiative"))
                    .font(FontId::monospace(11.0))
                    .color(palette.text_muted),
            );
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_svg_logo_is_valid_and_attributes_dnt_victer() {
        assert!(EMBEDDED_LOGO_SVG.contains("<svg"));
        assert!(EMBEDDED_LOGO_SVG.contains("AirTeach AI"));
        assert!(EMBEDDED_LOGO_SVG.contains("DNT - Victer"));
    }

    #[test]
    fn logo_rasterizer_renders_sharp_non_empty_icon_at_multiple_sizes() {
        for size in [32u32, 64u32, 128u32] {
            let icon = rasterize_brand_logo_icon(size);
            assert_eq!(icon.width, size);
            assert_eq!(icon.height, size);
            assert_eq!(icon.rgba.len(), (size * size * 4) as usize);
            let opaque_pixels = icon.rgba.chunks_exact(4).filter(|px| px[3] > 200).count();
            assert!(
                opaque_pixels > ((size * size) as usize) / 2,
                "Expected logo fill across >50% of icon canvas at {size}px"
            );
        }
    }

    #[test]
    fn design_tokens_provide_distinct_dark_and_light_palettes() {
        let dark = ThemeMode::Dark.palette();
        let light = ThemeMode::Light.palette();
        assert_ne!(dark.bg_canvas, light.bg_canvas);
        assert_ne!(dark.text_primary, light.text_primary);
        assert_ne!(dark.surface, light.surface);
    }
}
