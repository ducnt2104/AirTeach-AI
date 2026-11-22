//! Output Sanitizer & Math Formatter (`src/utils/output_sanitizer.rs`):
//! - Eliminates broken `$`, `$$`, `$$$` delimiters, raw HTML/XML leaks, and pseudo-tags
//!   (`<arrow>`, `<right_arrow>`, `<=>`, `=>`, `->`, `<bullet>`, `<check>`) across Chat & Canvas.
//! - Renders crisp Unicode Mathematical notation (nested `\frac`, `\sqrt`, superscripts, subscripts,
//!   Greek letters, set operators, real vector arrows `→ ⇒ ⇔`).
//! - Provides `StreamingSanitizer` buffering unclosed `$` / `<tag` fragments during SSE streaming.

use serde::{Deserialize, Serialize};

fn find_braced_arg(s: &str, start_brace: usize) -> Option<(&str, usize)> {
    let bytes = s.as_bytes();
    if start_brace >= bytes.len() || bytes[start_brace] != b'{' {
        return None;
    }
    let mut depth = 0i32;
    for (rel_idx, ch) in s[start_brace..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    let abs_idx = start_brace + rel_idx;
                    return Some((&s[start_brace + 1..abs_idx], abs_idx + 1));
                }
            }
            _ => {}
        }
    }
    None
}

fn map_superscript(ch: char) -> Option<char> {
    match ch {
        '0' => Some('⁰'),
        '1' => Some('¹'),
        '2' => Some('²'),
        '3' => Some('³'),
        '4' => Some('⁴'),
        '5' => Some('⁵'),
        '6' => Some('⁶'),
        '7' => Some('⁷'),
        '8' => Some('⁸'),
        '9' => Some('⁹'),
        '+' => Some('⁺'),
        '-' => Some('⁻'),
        '=' => Some('⁼'),
        '(' => Some('⁽'),
        ')' => Some('⁾'),
        'n' => Some('ⁿ'),
        'i' => Some('ⁱ'),
        'k' => Some('ᵏ'),
        'x' => Some('ˣ'),
        'y' => Some('ʸ'),
        _ => None,
    }
}

fn map_subscript(ch: char) -> Option<char> {
    match ch {
        '0' => Some('₀'),
        '1' => Some('₁'),
        '2' => Some('₂'),
        '3' => Some('₃'),
        '4' => Some('₄'),
        '5' => Some('₅'),
        '6' => Some('₆'),
        '7' => Some('₇'),
        '8' => Some('₈'),
        '9' => Some('₉'),
        '+' => Some('₊'),
        '-' => Some('₋'),
        '=' => Some('₌'),
        '(' => Some('₍'),
        ')' => Some('₎'),
        'n' => Some('ₙ'),
        'i' => Some('ᵢ'),
        'j' => Some('ⱼ'),
        'k' => Some('ₖ'),
        'x' => Some('ₓ'),
        _ => None,
    }
}

fn convert_script(inner: &str, sup: bool, fallback_prefix: char) -> String {
    let trimmed = inner.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let mut mapped = String::with_capacity(trimmed.len() * 2);
    for ch in trimmed.chars() {
        let m = if sup {
            map_superscript(ch)
        } else {
            map_subscript(ch)
        };
        if let Some(c) = m {
            mapped.push(c);
        } else {
            return format!("{fallback_prefix}{trimmed}");
        }
    }
    mapped
}

fn replace_case_insensitive_tag(s: &str, patterns: &[&str], replacement: &str) -> String {
    let mut out = s.to_string();
    let lower_patterns: Vec<String> = patterns.iter().map(|p| p.to_lowercase()).collect();
    loop {
        let low = out.to_lowercase();
        let mut matched_range: Option<(usize, usize)> = None;
        for lp in &lower_patterns {
            if let Some(idx) = low.find(lp.as_str()) {
                matched_range = Some((idx, idx + lp.len()));
                break;
            }
        }
        if let Some((start, end)) = matched_range {
            out.replace_range(start..end, replacement);
        } else {
            break;
        }
    }
    out
}

pub struct OutputSanitizer;

impl OutputSanitizer {
    /// Normalize `$`, `$$`, `$$$` delimiters, pseudo-tags (`<arrow>`, `<=>`, `->`),
    /// and leaked HTML/XML wrapper tags before rendering.
    pub fn normalize_delimiters_and_tags(raw: &str) -> String {
        if raw.is_empty() {
            return String::new();
        }
        let mut s = raw.replace("\r\n", "\n").replace('\r', "\n");
        s = s.replace(r"\[", "\n$$\n").replace(r"\]", "\n$$\n");
        s = s.replace(r"\(", "$").replace(r"\)", "$");

        // Collapse 3+ consecutive `$` ($$$, $$$$) -> $$
        while s.contains("$$$") {
            s = s.replace("$$$", "$$");
        }

        // Convert HTML <br> / strip wrapper tags
        s = replace_case_insensitive_tag(&s, &["<br>", "<br/>", "<br />"], "\n");
        for tag in [
            "<p>", "</p>", "<div>", "</div>", "<span>", "</span>", "<think>", "</think>",
            "<thought>", "</thought>", "<reasoning>", "</reasoning>", "<answer>", "</answer>",
            "<math>", "</math>", "<katex>", "</katex>",
        ] {
            s = replace_case_insensitive_tag(&s, &[tag], "");
        }

        // Replace text/XML pseudo-arrows and bullets with crisp Unicode vectors
        s = replace_case_insensitive_tag(
            &s,
            &["<arrow>", "<arrow/>", "<right_arrow>", "<right_arrow/>", "< arrow >"],
            " → ",
        );
        s = replace_case_insensitive_tag(&s, &["<left_arrow>", "<left_arrow/>"], " ← ");
        s = replace_case_insensitive_tag(
            &s,
            &["<bi_arrow>", "<double_arrow>", "<biarrow>"],
            " ⇔ ",
        );
        s = replace_case_insensitive_tag(&s, &["<implies>", "<implies/>"], " ⇒ ");
        s = replace_case_insensitive_tag(&s, &["<bullet>", "<bullet/>"], "• ");
        s = replace_case_insensitive_tag(&s, &["<check>", "<checkmark>", "<check/>"], "✓ ");
        s = replace_case_insensitive_tag(&s, &["<cross>", "<cross/>"], "✕ ");

        s = s.replace("<=>", " ⇔ ");
        s = s.replace("=>", " ⇒ ");
        s = s.replace("->", " → ");
        s = s.replace("<-", " ← ");
        s
    }

    /// Convert a LaTeX expression into clean human-readable Unicode mathematical notation.
    pub fn latex_expr_to_unicode(expr: &str) -> String {
        let mut s = Self::normalize_delimiters_and_tags(expr);
        // Protect escaped `\$`
        const SENTINEL: char = '\u{E000}';
        s = s.replace(r"\$", &SENTINEL.to_string());
        s = s.replace('$', "");
        s = s.replace(SENTINEL, "$");
        s = s.replace(r"\{", "{").replace(r"\}", "}");

        // Convert \dfrac{a}{b}, \tfrac{a}{b}, \frac{a}{b} (supports nesting)
        for cmd in [r"\dfrac", r"\tfrac", r"\frac"] {
            let needle = format!("{cmd}{{");
            while let Some(pos) = s.find(&needle) {
                let first_start = pos + cmd.len();
                let Some((num_raw, after_num)) = find_braced_arg(&s, first_start) else {
                    break;
                };
                let mut second_start = after_num;
                while second_start < s.len() && s.as_bytes()[second_start].is_ascii_whitespace() {
                    second_start += 1;
                }
                let Some((den_raw, after_den)) = find_braced_arg(&s, second_start) else {
                    break;
                };
                let num_c = Self::latex_expr_to_unicode(num_raw).trim().to_string();
                let den_c = Self::latex_expr_to_unicode(den_raw).trim().to_string();
                let num_fmt = if (num_c.contains(' ') || num_c.contains('+') || num_c.contains('-'))
                    && !(num_c.starts_with('(') && num_c.ends_with(')'))
                {
                    format!("({num_c})")
                } else {
                    num_c
                };
                let den_fmt = if (den_c.contains(' ') || den_c.contains('+') || den_c.contains('-'))
                    && !(den_c.starts_with('(') && den_c.ends_with(')'))
                {
                    format!("({den_c})")
                } else {
                    den_c
                };
                s.replace_range(pos..after_den, &format!("{num_fmt}/{den_fmt}"));
            }
        }

        // Convert \sqrt[n]{x} and \sqrt{x}
        while let Some(pos) = s.find(r"\sqrt") {
            let mut rest = pos + 5;
            let mut root_n = String::new();
            if s[rest..].starts_with('[') {
                if let Some(end_b) = s[rest + 1..].find(']') {
                    root_n = s[rest + 1..rest + 1 + end_b].trim().to_string();
                    rest += end_b + 2;
                }
            }
            if let Some((inner, after_end)) = find_braced_arg(&s, rest) {
                let inner_c = Self::latex_expr_to_unicode(inner).trim().to_string();
                let rad = match root_n.as_str() {
                    "3" => "∛".to_string(),
                    "4" => "∜".to_string(),
                    "" => "√".to_string(),
                    other => format!("{}√", convert_script(other, true, '^')),
                };
                s.replace_range(pos..after_end, &format!("{rad}({inner_c})"));
            } else {
                s.replace_range(pos..pos + 5, "√");
            }
        }

        for cmd in [r"\text", r"\mathrm", r"\mathbf", r"\operatorname"] {
            let needle = format!("{cmd}{{");
            while let Some(pos) = s.find(&needle) {
                if let Some((inner, after_end)) = find_braced_arg(&s, pos + cmd.len()) {
                    let owned = inner.to_string();
                    s.replace_range(pos..after_end, &owned);
                } else {
                    break;
                }
            }
        }

        while let Some(pos) = s.find(r"\vec{") {
            if let Some((inner, after_end)) = find_braced_arg(&s, pos + 4) {
                let owned = format!("{}⃗", inner.trim());
                s.replace_range(pos..after_end, &owned);
            } else {
                break;
            }
        }

        let symbol_table: [(&str, &str); 58] = [
            (r"\Longleftrightarrow", " ⟺ "),
            (r"\Leftrightarrow", " ⇔ "),
            (r"\Longrightarrow", " ⟹ "),
            (r"\Rightarrow", " ⇒ "),
            (r"\longrightarrow", " → "),
            (r"\rightarrow", " → "),
            (r"\leftarrow", " ← "),
            (r"\mapsto", " ↦ "),
            (r"\to", " → "),
            (r"\qquad", "   "),
            (r"\quad", "  "),
            (r"\,", " "),
            (r"\;", " "),
            (r"\:", " "),
            (r"\!", ""),
            (r"\left(", "("),
            (r"\right)", ")"),
            (r"\left[", "["),
            (r"\right]", "]"),
            (r"\left|", "|"),
            (r"\right|", "|"),
            (r"\infty", "∞"),
            (r"\iint", "∬"),
            (r"\int", "∫"),
            (r"\sum", "∑"),
            (r"\prod", "∏"),
            (r"\partial", "∂"),
            (r"\nabla", "∇"),
            (r"\Delta", "Δ"),
            (r"\delta", "δ"),
            (r"\alpha", "α"),
            (r"\beta", "β"),
            (r"\gamma", "γ"),
            (r"\theta", "θ"),
            (r"\lambda", "λ"),
            (r"\mu", "μ"),
            (r"\pi", "π"),
            (r"\sigma", "σ"),
            (r"\omega", "ω"),
            (r"\Omega", "Ω"),
            (r"\phi", "φ"),
            (r"\Psi", "Ψ"),
            (r"\pm", "±"),
            (r"\mp", "∓"),
            (r"\times", "×"),
            (r"\cdot", "·"),
            (r"\div", "÷"),
            (r"\leq", "≤"),
            (r"\geq", "≥"),
            (r"\neq", "≠"),
            (r"\approx", "≈"),
            (r"\equiv", "≡"),
            (r"\in", "∈"),
            (r"\notin", "∉"),
            (r"\forall", "∀"),
            (r"\exists", "∃"),
            (r"\cup", "∪"),
            (r"\cap", "∩"),
        ];
        for (pat, rep) in symbol_table {
            s = s.replace(pat, rep);
        }
        s = s
            .replace(r"\mathbb{R}", "ℝ")
            .replace(r"\mathbb{Z}", "ℤ")
            .replace(r"\mathbb{N}", "ℕ")
            .replace(r"\mathbb{Q}", "ℚ")
            .replace(r"\emptyset", "∅")
            .replace(r"\circ", "°")
            .replace(r"\angle", "∠")
            .replace(r"\perp", "⊥")
            .replace(r"\parallel", "∥")
            .replace(r"\triangle", "△")
            .replace(r"\sin", "sin")
            .replace(r"\cos", "cos")
            .replace(r"\tan", "tan")
            .replace(r"\log", "log")
            .replace(r"\ln", "ln")
            .replace(r"\lim", "lim");

        // Braced superscripts ^{...} & subscripts _{...}
        while let Some(pos) = s.find("^{") {
            if let Some((inner, after_end)) = find_braced_arg(&s, pos + 1) {
                let rep = convert_script(inner, true, '^');
                s.replace_range(pos..after_end, &rep);
            } else {
                break;
            }
        }
        while let Some(pos) = s.find("_{") {
            if let Some((inner, after_end)) = find_braced_arg(&s, pos + 1) {
                let rep = convert_script(inner, false, '_');
                s.replace_range(pos..after_end, &rep);
            } else {
                break;
            }
        }

        // Single-char ^2, _1 etc.
        let mut out = String::with_capacity(s.len());
        let mut chars = s.chars().peekable();
        while let Some(ch) = chars.next() {
            if ch == '^' {
                if let Some(&next_ch) = chars.peek() {
                    if let Some(sup) = map_superscript(next_ch) {
                        chars.next();
                        out.push(sup);
                        continue;
                    }
                }
            } else if ch == '_' {
                if let Some(&next_ch) = chars.peek() {
                    if let Some(sub) = map_subscript(next_ch) {
                        chars.next();
                        out.push(sub);
                        continue;
                    }
                }
            }
            out.push(ch);
        }

        out.replace(r"\begin{aligned}", "")
            .replace(r"\end{aligned}", "")
            .replace(r"\begin{cases}", "{ ")
            .replace(r"\end{cases}", " }")
            .replace(r"\\", "\n")
            .trim()
            .to_string()
    }

    /// Full sanitization pass guaranteeing zero `$`, `$$$`, raw LaTeX, or `<arrow>` strings appear.
    pub fn sanitize_plain_text(raw: &str) -> String {
        if raw.trim().is_empty() {
            return String::new();
        }
        let s = Self::latex_expr_to_unicode(raw);
        // Strip remaining unescaped `$`
        const SENTINEL: char = '\u{E000}';
        let cleaned = s
            .replace(r"\$", &SENTINEL.to_string())
            .replace('$', "")
            .replace(SENTINEL, "$");
        let mut collapsed = String::with_capacity(cleaned.len());
        let mut empty_nl = 0usize;
        for line in cleaned.lines() {
            let trimmed_end = line.trim_end();
            if trimmed_end.is_empty() {
                empty_nl += 1;
                if empty_nl <= 1 {
                    collapsed.push('\n');
                }
            } else {
                empty_nl = 0;
                collapsed.push_str(trimmed_end);
                collapsed.push('\n');
            }
        }
        collapsed.trim().to_string()
    }
}

/// Incremental token-stream sanitizer buffering unclosed `$` / `$$` / `<tag` chunks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamingSanitizer {
    buffer: String,
}

impl StreamingSanitizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }

    pub fn feed(&mut self, token_delta: &str) -> String {
        if token_delta.is_empty() {
            return String::new();
        }
        self.buffer.push_str(token_delta);
        let normalized = OutputSanitizer::normalize_delimiters_and_tags(&self.buffer);
        self.buffer = normalized.clone();

        let mut hold_idx = normalized.len();
        if let Some(last_lt) = normalized.rfind('<') {
            let last_gt = normalized.rfind('>').unwrap_or(0);
            if last_lt >= last_gt && (normalized.len() - last_lt) <= 24 {
                hold_idx = hold_idx.min(last_lt);
            }
        }
        if let Some(last_bs) = normalized.rfind('\\') {
            let tail = &normalized[last_bs..];
            if tail.len() <= 16 && !tail.contains(' ') && !tail.contains('}') {
                hold_idx = hold_idx.min(last_bs);
            }
        }

        let prefix = &normalized[..hold_idx];
        let double_positions: Vec<usize> = prefix.match_indices("$$").map(|(i, _)| i).collect();
        if double_positions.len() % 2 == 1 {
            hold_idx = hold_idx.min(*double_positions.last().unwrap());
        } else {
            let masked = prefix.replace("$$", "__");
            let single_positions: Vec<usize> = masked
                .match_indices('$')
                .filter(|(i, _)| *i == 0 || masked.as_bytes()[*i - 1] != b'\\')
                .map(|(i, _)| i)
                .collect();
            if single_positions.len() % 2 == 1 {
                hold_idx = hold_idx.min(*single_positions.last().unwrap());
            }
        }

        while !normalized.is_char_boundary(hold_idx) && hold_idx > 0 {
            hold_idx -= 1;
        }
        if hold_idx == 0 {
            return String::new();
        }
        let ready = normalized[..hold_idx].to_string();
        self.buffer = normalized[hold_idx..].to_string();
        OutputSanitizer::sanitize_plain_text(&ready)
    }

    pub fn flush(&mut self) -> String {
        let out = OutputSanitizer::sanitize_plain_text(&self.buffer);
        self.buffer.clear();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitizer_fixes_broken_dollars_arrows_and_latex() {
        let dirty = r"Giải: $$$2x + 1 = 3$$$ <arrow> $x = \frac{-(-2) \pm \sqrt{4}}{2}$ => x = 1$";
        let clean = OutputSanitizer::sanitize_plain_text(dirty);
        assert!(!clean.contains('$'), "Found stray '$' in {clean}");
        assert!(!clean.contains("<arrow>"), "Found '<arrow>' in {clean}");
        assert!(clean.contains('→'));
        assert!(clean.contains('⇒'));
        assert!(clean.contains("√(4)"));
    }

    #[test]
    fn test_streaming_sanitizer_holds_incomplete_tokens_then_flushes() {
        let mut st = StreamingSanitizer::new();
        let c1 = st.feed("Kết quả <arr");
        assert!(!c1.contains("<arr"));
        let c2 = st.feed("ow> $x^2");
        assert!(c2.contains('→'));
        assert!(!c2.contains('$'));
        let c3 = st.feed(" + 1 = 0$");
        assert!(c3.contains("x²"));
        let tail = st.flush();
        assert!(!tail.contains('$'));
    }
}
