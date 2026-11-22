//! Agentic Whiteboard Protocol (`src/ai/whiteboard_agent.rs`):
//! - Fast Local Intent Parser (< 1ms) for Vietnamese/English commands like
//!   `"ghi lên bảng hộ tôi 2x + 1 = 3 đi"`, `"viết công thức Pytago lên bảng"`, `"vẽ trục tọa độ"`, `"xóa bảng"`.
//! - Structured JSON Tool Action extraction (`write_on_board`, `solve_on_board`, `draw_shape`, `clear_board`, `undo_last_ai_action`).
//! - Collision-free empty-region finder (`find_free_board_region`) & crisp vector polylines (`build_vector_shape_strokes`).

use crate::core::stroke_engine::Point2D;
use crate::utils::output_sanitizer::OutputSanitizer;
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoardRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WhiteboardAction {
    WriteOnBoard {
        content: String,
        is_latex: bool,
        position: Option<Point2D>,
        font_size: f32,
        color_rgba: Option<[u8; 4]>,
    },
    SolveOnBoard {
        steps: Vec<String>,
        target_region: Option<BoardRect>,
    },
    DrawShape {
        shape_type: String,
        origin: Option<Point2D>,
        w: f32,
        h: f32,
        radius: f32,
        color_rgba: Option<[u8; 4]>,
    },
    DrawText {
        x: f32,
        y: f32,
        text: String,
        color_rgba: Option<[u8; 4]>,
        font_size: f32,
        normalized: bool,
    },
    DrawShapeNorm {
        shape_type: String,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        radius: f32,
        x2: Option<f32>,
        y2: Option<f32>,
        points: Option<Vec<Point2D>>,
        color_rgba: Option<[u8; 4]>,
        stroke_width: f32,
        label: Option<String>,
        normalized: bool,
    },
    DrawPolyline {
        points: Vec<Point2D>,
        color_rgba: Option<[u8; 4]>,
        stroke_width: f32,
        smooth: bool,
        normalized: bool,
    },
    HighlightRegion {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color_rgba: Option<[u8; 4]>,
        label: Option<String>,
        normalized: bool,
    },
    ClearRegion {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        normalized: bool,
    },
    ClearBoard,
    UndoLastAiAction,
}

pub fn parse_hex_color(s: &str) -> Option<[u8; 4]> {
    let hex = s.trim().trim_start_matches('#');
    if hex.len() == 6 {
        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
        return Some([r, g, b, 255]);
    }
    None
}

fn clamp_norm(val: Option<&Value>, default: f32) -> f32 {
    match val.and_then(|v| v.as_f64()) {
        Some(f) if f.is_finite() => (f as f32).clamp(0.0, 1000.0),
        _ => default,
    }
}

fn looks_like_math(s: &str) -> bool {
    ['=', '+', '-', '*', '/', '^', '_', '\\', '√', '²', '³', 'Δ', 'π', '∫']
        .iter()
        .any(|c| s.contains(*c))
}

fn strip_fillers_for_write(raw: &str) -> Option<String> {
    let low = raw.trim().to_lowercase();
    let triggers = [
        "ghi lên bảng hộ tôi",
        "ghi lên bảng giúp tôi",
        "viết lên bảng hộ tôi",
        "viết lên bảng giúp tôi",
        "ghi hộ tôi lên bảng",
        "viết hộ tôi lên bảng",
        "ghi ra bảng",
        "viết ra bảng",
        "ghi lên bảng",
        "viết lên bảng",
        "write on board",
        "put on board",
    ];
    let mut extracted: Option<&str> = None;
    for trig in triggers {
        if let Some(pos) = low.find(trig) {
            let after_byte = pos + trig.len();
            if after_byte <= raw.trim().len() {
                extracted = Some(&raw.trim()[after_byte..]);
                break;
            }
        }
    }
    let mut candidate = if let Some(s) = extracted {
        s.trim().to_string()
    } else if (low.starts_with("ghi ") || low.starts_with("viết ")) && low.contains("lên bảng") {
        let trimmed = raw.trim();
        let after_verb = if low.starts_with("ghi ") {
            &trimmed["ghi ".len()..]
        } else {
            &trimmed["viết ".len()..]
        };
        let mut t = after_verb.to_string();
        for suf in ["lên bảng hộ tôi", "lên bảng đi", "lên bảng nhé", "lên bảng"] {
            let t_low = t.to_lowercase();
            if let Some(idx) = t_low.rfind(suf) {
                t = t[..idx].to_string();
                break;
            }
        }
        t
    } else {
        return None;
    };

    for prefix in [":", "-", "công thức", "phương trình", "chữ", "giùm tôi", "hộ tôi", "hộ mình"]
    {
        let c_low = candidate.to_lowercase();
        if c_low.starts_with(prefix) {
            candidate = candidate[prefix.len()..].trim().to_string();
        }
    }
    loop {
        let c_low = candidate.to_lowercase();
        let mut trimmed_tail = false;
        for tail in [" đi", " nhé", " nha", " hộ tôi", " giúp tôi", " lên bảng", "!", "."] {
            if c_low.ends_with(tail) {
                candidate = candidate[..candidate.len() - tail.len()].trim().to_string();
                trimmed_tail = true;
                break;
            }
        }
        if !trimmed_tail {
            break;
        }
    }
    let final_str = candidate.trim_matches(['"', '\'', ' ']).to_string();
    if final_str.is_empty() {
        None
    } else {
        Some(final_str)
    }
}

pub struct WhiteboardAgent;

impl WhiteboardAgent {
    pub fn tool_protocol_prompt() -> &'static str {
        "Nếu giáo viên yêu cầu vẽ hình, viết công thức, giải bài hoặc chú thích lên bảng, hãy đính kèm block JSON trong thẻ ```whiteboard_actions\n\
         {\"speech_or_text\":\"Giải thích ngắn gọn\",\"actions\":[{\"type\":\"draw_shape\",\"shape\":\"triangle|line|arrow|rect|circle|ellipse|axes\",\"x\":150,\"y\":200,\"w\":220,\"h\":180,\"color\":\"#00F5D4\",\"strokeWidth\":3,\"label\":\"ABC\"},{\"type\":\"draw_text\",\"x\":150,\"y\":420,\"text\":\"a² + b² = c²\",\"color\":\"#FEE440\",\"fontSize\":18},{\"type\":\"draw_polyline\",\"points\":[[100,500],[200,450],[300,520]],\"color\":\"#00F5D4\",\"smooth\":true},{\"type\":\"highlight_region\",\"x\":120,\"y\":180,\"w\":280,\"h\":240,\"color\":\"#FEE440\",\"label\":\"Chú ý\"}]}\n\
         ``` (hệ tọa độ chuẩn hóa [0..1000], hỗ trợ thêm `clear_region`, `clear_all`)."
    }

    pub fn tool_protocol_prompt_with_context(
        canvas_w: f32,
        canvas_h: f32,
        free_regions_norm: &[[u32; 4]],
    ) -> String {
        let regions_str: Vec<String> = free_regions_norm
            .iter()
            .map(|r| format!("[{},{},{},{}]", r[0], r[1], r[2], r[3]))
            .collect();
        format!(
            "{} Kích thước bảng hiện tại: {}x{}px. Các vùng trống ưu tiên [x,y,w,h] hệ [0..1000]: [{}].",
            Self::tool_protocol_prompt(),
            canvas_w as i32,
            canvas_h as i32,
            regions_str.join(", ")
        )
    }

    /// Instant (<1ms) intent recognition avoiding LLM roundtrips for direct whiteboard commands.
    pub fn parse_fast_intent(user_text: &str) -> Option<WhiteboardAction> {
        let trimmed = user_text.trim();
        if trimmed.is_empty() {
            return None;
        }
        let low = trimmed.to_lowercase();

        if (low == "xóa bảng" || low == "clear board" || low == "xóa toàn bộ bảng")
            && !low.contains("giải")
        {
            return Some(WhiteboardAction::ClearBoard);
        }
        if low == "hoàn tác ai" || low == "hoàn tác thao tác ai" || low == "undo ai" {
            return Some(WhiteboardAction::UndoLastAiAction);
        }

        if (low.starts_with("vẽ ") || low.starts_with("draw ") || low.starts_with("hãy vẽ "))
            && !low.contains("giải")
        {
            let shape_type = if low.contains("trục tọa độ")
                || low.contains("hệ trục")
                || low.contains("oxy")
                || low.contains("axis")
            {
                "axis"
            } else if low.contains("mũi tên") || low.contains("arrow") {
                "arrow"
            } else if low.contains("hình tròn") || low.contains("circle") {
                "circle"
            } else if low.contains("hình vuông")
                || low.contains("chữ nhật")
                || low.contains("rect")
            {
                "rect"
            } else if low.contains("đường thẳng") || low.contains("line") {
                "line"
            } else {
                ""
            };
            if !shape_type.is_empty() {
                return Some(WhiteboardAction::DrawShape {
                    shape_type: shape_type.to_string(),
                    origin: None,
                    w: 190.0,
                    h: 120.0,
                    radius: 65.0,
                    color_rgba: None,
                });
            }
        }

        let has_solve = ["giải ", "tìm x", "chứng minh", "giải thích", "tóm tắt"]
            .iter()
            .any(|k| low.contains(k));
        if !has_solve && (low.contains("lên bảng") || low.contains("ra bảng")) {
            let known_formulas: [(&[&str], &str); 6] = [
                (&["pytago", "py-ta-go", "pythagor"], r"a^2 + b^2 = c^2"),
                (
                    &["công thức nghiệm", "bậc 2", "bậc hai"],
                    r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}",
                ),
                (&["delta", "biệt thức"], r"\Delta = b^2 - 4ac"),
                (&["vi-ét", "viet", "vieta"], r"x_1 + x_2 = -b/a, \; x_1 x_2 = c/a"),
                (&["newton"], r"\vec{F} = m \cdot \vec{a}"),
                (&["einstein"], r"E = mc^2"),
            ];
            for (keys, formula) in known_formulas {
                if keys.iter().any(|k| low.contains(k)) {
                    return Some(WhiteboardAction::WriteOnBoard {
                        content: formula.to_string(),
                        is_latex: true,
                        position: None,
                        font_size: 18.0,
                        color_rgba: None,
                    });
                }
            }

            if let Some(payload) = strip_fillers_for_write(trimmed) {
                let clean = OutputSanitizer::normalize_delimiters_and_tags(&payload)
                    .trim_matches('$')
                    .trim()
                    .to_string();
                let is_ltx = looks_like_math(&clean);
                return Some(WhiteboardAction::WriteOnBoard {
                    content: clean,
                    is_latex: is_ltx,
                    position: None,
                    font_size: 18.0,
                    color_rgba: None,
                });
            }
        }

        None
    }

    /// Extract JSON Whiteboard Tool calls from LLM markdown fences or standalone JSON and return sanitized chat text.
    pub fn extract_actions_from_response(raw_text: &str) -> (Vec<WhiteboardAction>, String) {
        let mut actions = Vec::new();
        let mut cleaned = raw_text.to_string();
        let mut extracted_speech: Option<String> = None;

        fn ingest_value(
            val: &Value,
            actions: &mut Vec<WhiteboardAction>,
            extracted_speech: &mut Option<String>,
        ) {
            if let Some(arr) = val.get("actions").and_then(|a| a.as_array()) {
                if let Some(sp) = val.get("speech_or_text").and_then(|s| s.as_str()) {
                    if !sp.trim().is_empty() {
                        *extracted_speech = Some(sp.trim().to_string());
                    }
                }
                for item in arr {
                    if let Some(act) = WhiteboardAgent::parse_json_action(item) {
                        actions.push(act);
                    }
                }
                return;
            }
            let list: Vec<&Value> = match val {
                Value::Array(arr) => arr.iter().collect(),
                Value::Object(_) => vec![val],
                _ => Vec::new(),
            };
            for item in list {
                if let Some(act) = WhiteboardAgent::parse_json_action(item) {
                    actions.push(act);
                }
            }
        }

        for fence in ["```whiteboard_actions", "```json"] {
            while let Some(start) = cleaned.find(fence) {
                let body_start = start + fence.len();
                let Some(rel_end) = cleaned[body_start..].find("```") else {
                    break;
                };
                let json_str = cleaned[body_start..body_start + rel_end].trim();
                let parsed_before = actions.len();
                let speech_before = extracted_speech.is_some();
                if let Ok(val) = serde_json::from_str::<Value>(json_str) {
                    ingest_value(&val, &mut actions, &mut extracted_speech);
                }
                if actions.len() > parsed_before || (!speech_before && extracted_speech.is_some()) {
                    cleaned.replace_range(start..body_start + rel_end + 3, "");
                } else {
                    break;
                }
            }
        }

        if actions.is_empty() && cleaned.trim_start().starts_with('{') {
            if let Ok(val) = serde_json::from_str::<Value>(cleaned.trim()) {
                ingest_value(&val, &mut actions, &mut extracted_speech);
                if !actions.is_empty() || extracted_speech.is_some() {
                    cleaned = extracted_speech.clone().unwrap_or_default();
                }
            }
        }

        if let Some(sp) = extracted_speech {
            if cleaned.trim().is_empty() {
                cleaned = sp;
            }
        }

        (actions, OutputSanitizer::sanitize_plain_text(&cleaned))
    }

    fn parse_json_action(v: &Value) -> Option<WhiteboardAction> {
        // 1. Normalized [0..1000] {"type": "draw_text"|"draw_shape"|"draw_polyline"|"highlight_region"|"clear_region"|"clear_all"}
        if let Some(act_type) = v.get("type").and_then(|t| t.as_str()) {
            match act_type.trim().to_lowercase().as_str() {
                "clear_all" => return Some(WhiteboardAction::ClearBoard),
                "clear_region" => {
                    return Some(WhiteboardAction::ClearRegion {
                        x: clamp_norm(v.get("x"), 0.0),
                        y: clamp_norm(v.get("y"), 0.0),
                        w: clamp_norm(v.get("w"), 200.0).max(20.0),
                        h: clamp_norm(v.get("h"), 200.0).max(20.0),
                        normalized: true,
                    });
                }
                "draw_text" => {
                    let text = v
                        .get("text")
                        .or_else(|| v.get("content"))
                        .and_then(|s| s.as_str())?
                        .trim()
                        .to_string();
                    if text.is_empty() {
                        return None;
                    }
                    let color_rgba = v.get("color").and_then(|c| c.as_str()).and_then(parse_hex_color);
                    let font_size = v
                        .get("fontSize")
                        .or_else(|| v.get("font_size"))
                        .and_then(|f| f.as_f64())
                        .unwrap_or(16.0) as f32;
                    return Some(WhiteboardAction::DrawText {
                        x: clamp_norm(v.get("x"), 120.0),
                        y: clamp_norm(v.get("y"), 160.0),
                        text,
                        color_rgba,
                        font_size: font_size.clamp(11.0, 36.0),
                        normalized: true,
                    });
                }
                "highlight_region" => {
                    let color_rgba = v.get("color").and_then(|c| c.as_str()).and_then(parse_hex_color);
                    let label = v
                        .get("label")
                        .and_then(|s| s.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty());
                    return Some(WhiteboardAction::HighlightRegion {
                        x: clamp_norm(v.get("x"), 100.0),
                        y: clamp_norm(v.get("y"), 100.0),
                        w: clamp_norm(v.get("w"), 220.0).max(30.0),
                        h: clamp_norm(v.get("h"), 140.0).max(30.0),
                        color_rgba,
                        label,
                        normalized: true,
                    });
                }
                "draw_polyline" => {
                    let arr = v.get("points")?.as_array()?;
                    let pts: Vec<Point2D> = arr
                        .iter()
                        .filter_map(|p| {
                            let pa = p.as_array()?;
                            if pa.len() >= 2 {
                                Some(Point2D::new(
                                    clamp_norm(pa.first(), 0.0),
                                    clamp_norm(pa.get(1), 0.0),
                                ))
                            } else {
                                None
                            }
                        })
                        .collect();
                    if pts.len() < 2 {
                        return None;
                    }
                    let color_rgba = v.get("color").and_then(|c| c.as_str()).and_then(parse_hex_color);
                    let stroke_width = v
                        .get("strokeWidth")
                        .or_else(|| v.get("stroke_width"))
                        .and_then(|f| f.as_f64())
                        .unwrap_or(3.2) as f32;
                    let smooth = v.get("smooth").and_then(|b| b.as_bool()).unwrap_or(true);
                    return Some(WhiteboardAction::DrawPolyline {
                        points: pts,
                        color_rgba,
                        stroke_width: stroke_width.clamp(1.5, 14.0),
                        smooth,
                        normalized: true,
                    });
                }
                "draw_shape" => {
                    let shape_type = v
                        .get("shape")
                        .or_else(|| v.get("shape_type"))
                        .or_else(|| v.get("kind"))
                        .and_then(|t| t.as_str())
                        .unwrap_or("rect")
                        .trim()
                        .to_lowercase();
                    let bbox = v.get("bbox").and_then(|b| b.as_array());
                    let x = clamp_norm(v.get("x").or_else(|| bbox.and_then(|b| b.first())), 160.0);
                    let y = clamp_norm(v.get("y").or_else(|| bbox.and_then(|b| b.get(1))), 180.0);
                    let w = clamp_norm(v.get("w").or_else(|| bbox.and_then(|b| b.get(2))), 200.0).max(20.0);
                    let h = clamp_norm(v.get("h").or_else(|| bbox.and_then(|b| b.get(3))), 150.0).max(20.0);
                    let radius = clamp_norm(v.get("radius"), 70.0).max(15.0);
                    let x2 = v.get("x2").map(|val| clamp_norm(Some(val), 360.0));
                    let y2 = v.get("y2").map(|val| clamp_norm(Some(val), 180.0));
                    let points = v.get("points").and_then(|arr| arr.as_array()).map(|arr| {
                        arr.iter()
                            .filter_map(|p| {
                                let pa = p.as_array()?;
                                if pa.len() >= 2 {
                                    Some(Point2D::new(
                                        clamp_norm(pa.first(), 0.0),
                                        clamp_norm(pa.get(1), 0.0),
                                    ))
                                } else {
                                    None
                                }
                            })
                            .collect()
                    });
                    let color_rgba = v.get("color").and_then(|c| c.as_str()).and_then(parse_hex_color);
                    let stroke_width = v
                        .get("strokeWidth")
                        .or_else(|| v.get("stroke_width"))
                        .and_then(|f| f.as_f64())
                        .unwrap_or(3.2) as f32;
                    let label = v
                        .get("label")
                        .and_then(|s| s.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty());
                    return Some(WhiteboardAction::DrawShapeNorm {
                        shape_type,
                        x,
                        y,
                        w,
                        h,
                        radius,
                        x2,
                        y2,
                        points,
                        color_rgba,
                        stroke_width: stroke_width.clamp(1.5, 14.0),
                        label,
                        normalized: true,
                    });
                }
                _ => {}
            }
        }

        // 2. Legacy {"tool": "...", "args": {...}}
        let tool = v.get("tool")?.as_str()?;
        let args = v.get("args").cloned().unwrap_or(Value::Null);
        match tool {
            "clear_board" => Some(WhiteboardAction::ClearBoard),
            "undo_last_ai_action" => Some(WhiteboardAction::UndoLastAiAction),
            "write_on_board" => {
                let content = args.get("content")?.as_str()?.trim().to_string();
                if content.is_empty() {
                    return None;
                }
                let fmt = args.get("format").and_then(|f| f.as_str()).unwrap_or("latex");
                let font_size = args
                    .get("fontSize")
                    .and_then(|f| f.as_f64())
                    .unwrap_or(17.0) as f32;
                let pos = args.get("position").and_then(|p| {
                    let x = p.get("x")?.as_f64()? as f32;
                    let y = p.get("y")?.as_f64()? as f32;
                    Some(Point2D::new(x, y))
                });
                Some(WhiteboardAction::WriteOnBoard {
                    content,
                    is_latex: fmt == "latex",
                    position: pos,
                    font_size,
                    color_rgba: None,
                })
            }
            "solve_on_board" => {
                let arr = args.get("steps")?.as_array()?;
                let steps: Vec<String> = arr
                    .iter()
                    .filter_map(|s| s.as_str().map(|t| t.trim().to_string()))
                    .filter(|s| !s.is_empty())
                    .collect();
                if steps.is_empty() {
                    return None;
                }
                Some(WhiteboardAction::SolveOnBoard {
                    steps,
                    target_region: None,
                })
            }
            "draw_shape" => {
                let shape_type = args
                    .get("type")
                    .and_then(|t| t.as_str())
                    .unwrap_or("arrow")
                    .to_string();
                let w = args.get("w").and_then(|v| v.as_f64()).unwrap_or(170.0) as f32;
                let h = args.get("h").and_then(|v| v.as_f64()).unwrap_or(110.0) as f32;
                let radius = args.get("radius").and_then(|v| v.as_f64()).unwrap_or(65.0) as f32;
                let origin = match (args.get("x").and_then(|v| v.as_f64()), args.get("y").and_then(|v| v.as_f64())) {
                    (Some(x), Some(y)) => Some(Point2D::new(x as f32, y as f32)),
                    _ => None,
                };
                Some(WhiteboardAction::DrawShape {
                    shape_type,
                    origin,
                    w,
                    h,
                    radius,
                    color_rgba: None,
                })
            }
            _ => None,
        }
    }
}

/// Find a clean collision-free `(x, y)` region on board avoiding existing strokes/slides/annotations.
pub fn find_free_board_region(
    occupied: &[BoardRect],
    canvas_w: f32,
    canvas_h: f32,
    needed_w: f32,
    needed_h: f32,
    preferred_anchor: Option<Point2D>,
) -> Point2D {
    let margin = 28.0_f32;
    let cw = canvas_w.max(420.0);
    let ch = canvas_h.max(320.0);
    let nw = needed_w.clamp(120.0, (cw - 2.0 * margin).max(120.0));
    let nh = needed_h.clamp(42.0, (ch - 2.0 * margin).max(42.0));

    let overlap_area = |cx: f32, cy: f32| -> f32 {
        let mut tot = 0.0_f32;
        for r in occupied {
            let ix0 = (cx - 10.0).max(r.x);
            let iy0 = (cy - 10.0).max(r.y);
            let ix1 = (cx + nw + 10.0).min(r.x + r.w);
            let iy1 = (cy + nh + 10.0).min(r.y + r.h);
            if ix1 > ix0 && iy1 > iy0 {
                tot += (ix1 - ix0) * (iy1 - iy0);
            }
        }
        tot
    };

    let step_x = ((cw - nw - 2.0 * margin) / 7.0).max(48.0);
    let step_y = ((ch - nh - 2.0 * margin) / 6.0).max(40.0);
    let mut best = Point2D::new(margin, 64.0);
    let mut best_cost = f32::INFINITY;

    let mut cy = 58.0_f32;
    while cy + nh <= ch - margin + 1.0 {
        let mut cx = margin;
        while cx + nw <= cw - margin + 1.0 {
            let ov = overlap_area(cx, cy);
            let dist_pen = if let Some(pref) = preferred_anchor {
                0.08 * Point2D::new(cx, cy).dist(pref)
            } else {
                0.03 * cy + 0.015 * cx
            };
            let cost = ov * 100.0 + dist_pen;
            if cost < best_cost {
                best_cost = cost;
                best = Point2D::new(cx, cy);
                if ov == 0.0 && preferred_anchor.is_none() {
                    return best;
                }
            }
            cx += step_x;
        }
        cy += step_y;
    }
    best
}

/// Return top `max_regions` collision-free `[x, y, w, h]` regions in `[0..1000]` normalized coordinates.
pub fn find_free_regions_normalized(
    occupied: &[BoardRect],
    canvas_w: f32,
    canvas_h: f32,
    max_regions: usize,
) -> Vec<[u32; 4]> {
    let cw = canvas_w.max(420.0);
    let ch = canvas_h.max(320.0);
    let mut scored: Vec<(f32, [u32; 4])> = Vec::new();
    for gy in [80_u32, 380, 660] {
        for gx in [50_u32, 360, 660] {
            let rw = 290_u32;
            let rh = 250_u32;
            let px = (gx as f32 / 1000.0) * cw;
            let py = (gy as f32 / 1000.0) * ch;
            let pw = (rw as f32 / 1000.0) * cw;
            let ph = (rh as f32 / 1000.0) * ch;
            let mut ov = 0.0_f32;
            for r in occupied {
                let ix0 = px.max(r.x);
                let iy0 = py.max(r.y);
                let ix1 = (px + pw).min(r.x + r.w);
                let iy1 = (py + ph).min(r.y + r.h);
                if ix1 > ix0 && iy1 > iy0 {
                    ov += (ix1 - ix0) * (iy1 - iy0);
                }
            }
            scored.push((ov, [gx, gy, rw, rh]));
        }
    }
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(max_regions.max(1))
        .map(|(_, rect)| rect)
        .collect()
}

/// Build clean vector strokes for `draw_shape` (`arrow`, `circle`, `ellipse`, `triangle`, `rect`, `line`, `axis`/`axes`).
pub fn build_vector_shape_strokes(
    shape_type: &str,
    origin: Point2D,
    w: f32,
    h: f32,
    radius: f32,
) -> Vec<Vec<Point2D>> {
    match shape_type.to_lowercase().as_str() {
        "circle" => {
            let r = radius.max(24.0);
            let cx = origin.x + r;
            let cy = origin.y + r;
            let pts: Vec<Point2D> = (0..=48)
                .map(|i| {
                    let th = std::f32::consts::TAU * i as f32 / 48.0;
                    Point2D::new(cx + r * th.cos(), cy + r * th.sin())
                })
                .collect();
            vec![pts]
        }
        "ellipse" => {
            let rx = (w * 0.5).max(20.0);
            let ry = (h * 0.5).max(14.0);
            let cx = origin.x + rx;
            let cy = origin.y + ry;
            let pts: Vec<Point2D> = (0..=48)
                .map(|i| {
                    let th = std::f32::consts::TAU * i as f32 / 48.0;
                    Point2D::new(cx + rx * th.cos(), cy + ry * th.sin())
                })
                .collect();
            vec![pts]
        }
        "triangle" => {
            let ww = w.max(70.0);
            let hh = h.max(60.0);
            let p_a = Point2D::new(origin.x, origin.y + hh);
            let p_b = Point2D::new(origin.x, origin.y);
            let p_c = Point2D::new(origin.x + ww, origin.y + hh);
            vec![vec![p_a, p_b, p_c, p_a]]
        }
        "rect" => {
            let ww = w.max(60.0);
            let hh = h.max(40.0);
            vec![vec![
                origin,
                Point2D::new(origin.x + ww, origin.y),
                Point2D::new(origin.x + ww, origin.y + hh),
                Point2D::new(origin.x, origin.y + hh),
                origin,
            ]]
        }
        "arrow" => {
            let x2 = origin.x + w.max(120.0);
            let y2 = origin.y;
            let head = 15.0_f32;
            vec![
                vec![origin, Point2D::new(x2, y2)],
                vec![
                    Point2D::new(x2 - head, y2 - 8.0),
                    Point2D::new(x2, y2),
                    Point2D::new(x2 - head, y2 + 8.0),
                ],
            ]
        }
        "axis" | "axes" => {
            let size = w.clamp(140.0, 260.0);
            let cx = origin.x + size * 0.5;
            let cy = origin.y + size * 0.5;
            let half = size * 0.5;
            let mut out = vec![
                vec![Point2D::new(cx - half, cy), Point2D::new(cx + half, cy)],
                vec![
                    Point2D::new(cx + half - 10.0, cy - 6.0),
                    Point2D::new(cx + half, cy),
                    Point2D::new(cx + half - 10.0, cy + 6.0),
                ],
                vec![Point2D::new(cx, cy + half), Point2D::new(cx, cy - half)],
                vec![
                    Point2D::new(cx - 6.0, cy - half + 10.0),
                    Point2D::new(cx, cy - half),
                    Point2D::new(cx + 6.0, cy - half + 10.0),
                ],
            ];
            for k in [-2.0_f32, -1.0, 1.0, 2.0] {
                let tx = cx + k * half * 0.35;
                let ty = cy - k * half * 0.35;
                out.push(vec![Point2D::new(tx, cy - 4.0), Point2D::new(tx, cy + 4.0)]);
                out.push(vec![Point2D::new(cx - 4.0, ty), Point2D::new(cx + 4.0, ty)]);
            }
            out
        }
        _ => vec![vec![origin, Point2D::new(origin.x + w.max(120.0), origin.y)]],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_intent_write_2x_plus_1_eq_3() {
        let act = WhiteboardAgent::parse_fast_intent("ghi lên bảng hộ tôi 2x + 1 = 3 đi")
            .expect("Must detect direct write command");
        match act {
            WhiteboardAction::WriteOnBoard {
                content, is_latex, ..
            } => {
                assert_eq!(content, "2x + 1 = 3");
                assert!(is_latex);
            }
            other => panic!("Unexpected {other:?}"),
        }
    }

    #[test]
    fn test_free_region_avoids_occupied_boxes() {
        let occupied = vec![BoardRect {
            x: 10.0,
            y: 40.0,
            w: 420.0,
            h: 320.0,
        }];
        let p = find_free_board_region(&occupied, 1000.0, 700.0, 220.0, 60.0, None);
        assert!(p.x >= 430.0 || p.y >= 360.0, "Got overlapping point {p:?}");
    }
}
