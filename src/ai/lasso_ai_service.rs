//! Smart Lasso-to-AI Service (`src/ai/lasso_ai_service.rs`):
//! - Polygon ray-casting containment & tight bounding box computation with `18px` padding.
//! - Generates targeted prompts for the 4 Floating Action Bar buttons:
//!   `[Giải bằng AI]`, `[Viết lời giải lên bảng]`, `[Làm đẹp chữ/LaTeX]`, `[Giải thích nhanh]`.

use crate::core::stroke_engine::Point2D;
use serde::{Deserialize, Serialize};

pub const DEFAULT_LASSO_PADDING_PX: f32 = 18.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LassoQuickAction {
    SolveAi,
    WriteSolutionToBoard,
    BeautifyLatex,
    ExplainQuick,
}

impl LassoQuickAction {
    pub const fn prompt(self) -> &'static str {
        match self {
            Self::SolveAi => {
                "Đọc chính xác đề bài/công thức viết tay trong ảnh vùng khoanh này và giải chi tiết từng bước bằng tiếng Việt. Không dùng $$$ hay thẻ <arrow>, dùng ký hiệu toán chuẩn."
            }
            Self::WriteSolutionToBoard => {
                "Đọc bài toán trong ảnh vùng khoanh, giải ngắn gọn từng bước và đính kèm khối JSON ```whiteboard_actions\n[{\"tool\":\"solve_on_board\",\"args\":{\"steps\":[\"Bước 1: ...\",\"Bước 2: ...\"]}}]\n``` để đưa thẳng lên bảng."
            }
            Self::BeautifyLatex => {
                "Hãy nhận diện chính xác duy nhất biểu thức toán học hoặc dòng chữ trong ảnh vùng khoanh này. Chỉ trả về đúng biểu thức công thức chuẩn (ví dụ 2x + 1 = 3 hoặc \\frac{-b \\pm \\sqrt{\\Delta}}{2a}), không giải thích thừa."
            }
            Self::ExplainQuick => {
                "Tóm tắt và giải thích ngắn gọn khái niệm/công thức trong vùng khoanh này thật dễ hiểu."
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct LassoSelectionRegion {
    pub polygon: Vec<Point2D>,
    pub min_pt: Point2D,
    pub max_pt: Point2D,
}

impl LassoSelectionRegion {
    pub fn from_points(points: &[Point2D], padding: f32) -> Option<Self> {
        if points.len() < 2 {
            return None;
        }
        let mut min_x = f32::INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut max_y = f32::NEG_INFINITY;
        for p in points {
            min_x = min_x.min(p.x);
            min_y = min_y.min(p.y);
            max_x = max_x.max(p.x);
            max_y = max_y.max(p.y);
        }
        let min_pt = Point2D::new((min_x - padding).max(0.0), (min_y - padding).max(0.0));
        let max_pt = Point2D::new(max_x + padding, max_y + padding);
        if (max_pt.x - min_pt.x) < 16.0 || (max_pt.y - min_pt.y) < 16.0 {
            return None;
        }
        Some(Self {
            polygon: points.to_vec(),
            min_pt,
            max_pt,
        })
    }

    pub fn point_inside(&self, pt: Point2D) -> bool {
        if self.polygon.len() < 5 {
            return pt.x >= self.min_pt.x
                && pt.x <= self.max_pt.x
                && pt.y >= self.min_pt.y
                && pt.y <= self.max_pt.y;
        }
        let mut inside = false;
        let n = self.polygon.len();
        let mut j = n - 1;
        for i in 0..n {
            let pi = self.polygon[i];
            let pj = self.polygon[j];
            let intersect = ((pi.y > pt.y) != (pj.y > pt.y))
                && (pt.x < (pj.x - pi.x) * (pt.y - pi.y) / (pj.y - pi.y).max(1e-6) + pi.x);
            if intersect {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    pub fn contains_stroke(&self, pts: &[Point2D]) -> bool {
        if pts.is_empty() {
            return false;
        }
        let hits = pts.iter().filter(|&&p| self.point_inside(p)).count();
        hits * 5 >= pts.len() * 2
    }
}

pub fn extract_clean_beautified_formula(reply: &str) -> String {
    let mut s = reply.trim().to_string();
    for fence in ["```latex", "```tex", "```text", "```"] {
        if s.starts_with(fence) {
            s = s[fence.len()..].trim().to_string();
        }
    }
    if s.ends_with("```") {
        s = s[..s.len() - 3].trim().to_string();
    }
    s.lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim()
        .trim_matches('$')
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lasso_polygon_bbox_and_containment() {
        let poly = vec![
            Point2D::new(100.0, 100.0),
            Point2D::new(220.0, 100.0),
            Point2D::new(220.0, 180.0),
            Point2D::new(100.0, 180.0),
            Point2D::new(100.0, 105.0),
        ];
        let reg = LassoSelectionRegion::from_points(&poly, DEFAULT_LASSO_PADDING_PX).unwrap();
        assert!((reg.min_pt.x - 82.0).abs() < 1e-3);
        assert!(reg.contains_stroke(&[Point2D::new(150.0, 140.0), Point2D::new(180.0, 140.0)]));
        assert!(!reg.contains_stroke(&[Point2D::new(400.0, 400.0), Point2D::new(420.0, 420.0)]));
    }
}
