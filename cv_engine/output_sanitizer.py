#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output Sanitizer & Math Renderer (`cv_engine/output_sanitizer.py`)
for AirTeach AI (DNT - Victer):
1. `OutputSanitizer`: fixes broken `$`, `$$`, `$$$` delimiters, eliminates stray raw `$`
   and leaked XML/HTML/pseudo-tags (`<arrow>`, `<right_arrow>`, `<=>`, `=>`, `->`, `<bullet>`, `<check>`),
   and converts LaTeX syntax into either:
   - Crisp Unicode Mathematical text (`to_unicode_math`)
   - High-DPI PNG math expressions rendered 100% offline via `matplotlib.mathtext` (with LRU cache)
   - Rich Qt/KaTeX-compatible HTML (`to_rich_html` / `to_katex_html`) with inline vector SVG arrows.
2. `StreamingSanitizer`: stateful token-stream buffer holding incomplete `$...$` / `$$...$$`
   spans or partial tags across streamed chunks so live UI never flickers with broken syntax.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import lru_cache
import html
import io
import re
from typing import Dict, List, Optional, Tuple

import importlib.util

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
Figure = None  # type: ignore[assignment]
FigureCanvasAgg = None  # type: ignore[assignment]


def _ensure_matplotlib() -> bool:
    """Lazy-load matplotlib on first math rasterization to keep GUI startup < 1.5s."""
    global Figure, FigureCanvasAgg, HAS_MATPLOTLIB
    if Figure is not None and FigureCanvasAgg is not None:
        return True
    if not HAS_MATPLOTLIB:
        return False
    try:
        from matplotlib.figure import Figure as _MplFigure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as _MplAgg

        Figure = _MplFigure
        FigureCanvasAgg = _MplAgg
        HAS_MATPLOTLIB = True
        return True
    except Exception:
        HAS_MATPLOTLIB = False
        return False


_PSEUDO_TAG_REPLACEMENTS: Tuple[Tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"<\s*(?:right_)?arrow\s*/?\s*>", re.IGNORECASE), " → "),
    (re.compile(r"<\s*left_arrow\s*/?\s*>", re.IGNORECASE), " ← "),
    (re.compile(r"<\s*(?:bi|double)_?arrow\s*/?\s*>", re.IGNORECASE), " ⇔ "),
    (re.compile(r"<\s*implies\s*/?\s*>", re.IGNORECASE), " ⇒ "),
    (re.compile(r"<\s*bullet\s*/?\s*>", re.IGNORECASE), "• "),
    (re.compile(r"<\s*check(?:mark)?\s*/?\s*>", re.IGNORECASE), "✓ "),
    (re.compile(r"<\s*cross\s*/?\s*>", re.IGNORECASE), "✕ "),
    (re.compile(r"<=>"), " ⇔ "),
    (re.compile(r"=>"), " ⇒ "),
    (re.compile(r"->"), " → "),
    (re.compile(r"<-"), " ← "),
)

_ALLOWED_HTML_EMPTY_STRIP = re.compile(
    r"</?(?:p|div|span|font|think|thought|reasoning|answer|output|step|math|katex)[^>]*>",
    re.IGNORECASE,
)
_BR_TAG_RE = re.compile(r"<\s*br\s*/?\s*>", re.IGNORECASE)

_SUP_MAP: Dict[str, str] = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "n": "ⁿ",
    "i": "ⁱ",
    "k": "ᵏ",
    "x": "ˣ",
    "y": "ʸ",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
}

_SUB_MAP: Dict[str, str] = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "n": "ₙ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "x": "ₓ",
}

_LATEX_SYMBOL_TABLE: Tuple[Tuple[str, str], ...] = (
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
    (r"\left\{", "{"),
    (r"\right\}", "}"),
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
    (r"\sigma", "σ"),
    (r"\omega", "ω"),
    (r"\Omega", "Ω"),
    (r"\phi", "φ"),
    (r"\Phi", "Φ"),
    (r"\psi", "ψ"),
    (r"\Psi", "Ψ"),
    (r"\pi", "π"),
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
    (r"\propto", "∝"),
    (r"\in", "∈"),
    (r"\notin", "∉"),
    (r"\forall", "∀"),
    (r"\exists", "∃"),
    (r"\subset", "⊂"),
    (r"\subseteq", "⊆"),
    (r"\cup", "∪"),
    (r"\cap", "∩"),
    (r"\emptyset", "∅"),
    (r"\mathbb{R}", "ℝ"),
    (r"\mathbb{Z}", "ℤ"),
    (r"\mathbb{N}", "ℕ"),
    (r"\mathbb{Q}", "ℚ"),
    (r"\mathbb{C}", "ℂ"),
    (r"\degree", "°"),
    (r"\circ", "°"),
    (r"\angle", "∠"),
    (r"\perp", "⊥"),
    (r"\parallel", "∥"),
    (r"\triangle", "△"),
    (r"\sin", "sin"),
    (r"\cos", "cos"),
    (r"\tan", "tan"),
    (r"\cot", "cot"),
    (r"\log", "log"),
    (r"\ln", "ln"),
    (r"\lim", "lim"),
)


def _find_braced_arg(s: str, start_brace: int) -> Optional[Tuple[str, int]]:
    """Find matching `{...}` block starting at `start_brace`. Returns `(inner, end_exclusive)`."""
    if start_brace >= len(s) or s[start_brace] != "{":
        return None
    depth = 0
    for idx in range(start_brace, len(s)):
        ch = s[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start_brace + 1 : idx], idx + 1
    return None


def _map_script(chars: str, mapping: Dict[str, str], prefix_if_complex: str) -> str:
    stripped = chars.strip()
    if stripped and all(c in mapping for c in stripped):
        return "".join(mapping[c] for c in stripped)
    return f"{prefix_if_complex}{stripped}"


def arrow_svg_data_uri(direction: str = "right", color_hex: str = "#06B6D4", size_px: int = 16) -> str:
    """Return an inline SVG data URI for crisp UI arrows (`right`, `left`, `double`, `implies`)."""
    col = color_hex if color_hex.startswith("#") else f"#{color_hex}"
    if direction == "double":
        body = (
            f'<path d="M5 12H19M5 12L9 8M5 12L9 16M19 12L15 8M19 12L15 16" '
            f'stroke="{col}" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    elif direction == "left":
        body = (
            f'<path d="M19 12H5M5 12L10 7M5 12L10 17" '
            f'stroke="{col}" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    elif direction == "implies":
        body = (
            f'<path d="M4 9.5H15M4 14.5H15M13 6.5L19.5 12L13 17.5" '
            f'stroke="{col}" stroke-width="2.0" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    else:
        body = (
            f'<path d="M4 12H19M19 12L14 7M19 12L14 17" '
            f'stroke="{col}" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_px}" height="{size_px}" '
        f'viewBox="0 0 24 24" fill="none">{body}</svg>'
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


@lru_cache(maxsize=256)
def render_latex_to_png_bytes(
    latex_expr: str,
    color_hex: str = "#F8FAFC",
    fontsize: int = 14,
    dpi: int = 135,
) -> Optional[bytes]:
    """
    Render a clean LaTeX math expression 100% offline using `matplotlib.mathtext`
    into a transparent crisp PNG (`bytes`). Results are LRU-cached for zero-lag redraws.
    """
    if not _ensure_matplotlib() or Figure is None or FigureCanvasAgg is None:
        return None
    cleaned = OutputSanitizer.clean_inner_latex_for_mathtext(latex_expr)
    if not cleaned:
        return None

    try:
        fig = Figure(figsize=(0.1, 0.1), dpi=dpi)
        fig.patch.set_alpha(0.0)
        canvas = FigureCanvasAgg(fig)
        txt = fig.text(
            0.5,
            0.5,
            f"${cleaned}$",
            color=color_hex,
            fontsize=fontsize,
            ha="center",
            va="center",
        )
        canvas.draw()
        bbox = txt.get_window_extent(renderer=canvas.get_renderer())
        w_in = max(0.2, (bbox.width + 10.0) / float(dpi))
        h_in = max(0.18, (bbox.height + 8.0) / float(dpi))
        fig.set_size_inches(w_in, h_in)
        txt.set_position((0.5, 0.5))
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0.03,
        )
        data = buf.getvalue()
        return data if data else None
    except Exception:
        return None


class OutputSanitizer:
    """Sanitizes AI responses & math formulas so `$`, `$$$`, and `<arrow>` never leak to UI."""

    @staticmethod
    def clean_inner_latex_for_mathtext(raw_expr: str) -> str:
        """Prepare a LaTeX expression safely for matplotlib `mathtext` (single `$` wrapped later)."""
        s = (raw_expr or "").strip()
        while s.startswith("$"):
            s = s[1:].strip()
        while s.endswith("$"):
            s = s[:-1].strip()
        s = s.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "")
        s = s.replace(r"\displaystyle", "").replace(r"\textstyle", "")
        s = s.replace(r"\Longleftrightarrow", r"\Leftrightarrow")
        s = s.replace(r"\Longrightarrow", r"\Rightarrow")
        s = s.replace(r"\longrightarrow", r"\rightarrow")
        # Convert \text{...} -> \mathrm{...} which mathtext handles reliably
        s = re.sub(r"\\text\s*\{", r"\\mathrm{", s)
        s = re.sub(r"\\textbf\s*\{", r"\\mathbf{", s)
        s = re.sub(r"\\mathbb\s*\{([A-Za-z])\}", r"\1", s)
        return s.strip()

    @staticmethod
    def normalize_delimiters_and_tags(raw: str) -> str:
        r"""Fix broken `$$$`, `\[...\]`, `\(...\)`, `<arrow>` pseudo-tags, and stray HTML tags."""
        if not raw:
            return ""
        s = str(raw).replace("\r\n", "\n").replace("\r", "\n")

        # Normalize LaTeX brackets \( \) and \[ \]
        s = s.replace(r"\[", "\n$$\n").replace(r"\]", "\n$$\n")
        s = s.replace(r"\(", "$").replace(r"\)", "$")

        # Fix 3+ consecutive dollar signs ($$$, $$$$) -> $$
        s = re.sub(r"\${3,}", "$$", s)

        # Strip disallowed/raw pseudo XML/HTML tags while preserving line breaks
        s = _BR_TAG_RE.sub("\n", s)
        s = _ALLOWED_HTML_EMPTY_STRIP.sub("", s)

        # Replace pseudo-tags (<arrow>, ->, =>, <=>, <bullet>, <check>)
        for pattern, rep in _PSEUDO_TAG_REPLACEMENTS:
            s = pattern.sub(rep, s)

        # Normalize spaces around inline math blocks, fixing trailing unpaired single '$'
        # First collapse `$$ ... $$` spacing gently
        s = re.sub(r"\$\$\s*\$\$", "", s)
        return s

    @staticmethod
    def strip_all_stray_dollars(text: str) -> str:
        """
        After extracting balanced `$$...$$` / `$...$` spans or converting to Unicode,
        ensure zero stray literal `$` signs remain except legitimate currency like `\\$100`.
        """
        # Protect escaped literal currency `\$` temporarily
        sentinel = "\uE000"
        s = text.replace(r"\$", sentinel)
        # Strip any remaining `$` delimiters cleanly
        s = s.replace("$", "")
        s = s.replace(sentinel, "$")
        return s

    @classmethod
    def latex_expr_to_unicode(cls, expr: str) -> str:
        """Convert one raw LaTeX formula string into clean Unicode Mathematical notation."""
        s = cls.normalize_delimiters_and_tags(expr)
        s = s.replace(r"\$", "\uE000").replace("$", "").replace("\uE000", "$")
        s = s.replace(r"\{", "{").replace(r"\}", "}")

        # Nested \frac{num}{den} and \dfrac{num}{den} / \tfrac{num}{den}
        for cmd in (r"\dfrac", r"\tfrac", r"\frac"):
            while True:
                pos = s.find(cmd + "{")
                if pos < 0:
                    break
                first = _find_braced_arg(s, pos + len(cmd))
                if not first:
                    break
                num, after_num = first
                # Allow whitespace between numerator and denominator
                next_idx = after_num
                while next_idx < len(s) and s[next_idx].isspace():
                    next_idx += 1
                second = _find_braced_arg(s, next_idx)
                if not second:
                    break
                den, after_den = second
                num_c = cls.latex_expr_to_unicode(num).strip()
                den_c = cls.latex_expr_to_unicode(den).strip()
                num_fmt = f"({num_c})" if any(op in num_c for op in (" ", "+", "-")) and not (num_c.startswith("(") and num_c.endswith(")")) else num_c
                den_fmt = f"({den_c})" if any(op in den_c for op in (" ", "+", "-")) and not (den_c.startswith("(") and den_c.endswith(")")) else den_c
                s = s[:pos] + f"{num_fmt}/{den_fmt}" + s[after_den:]

        # \sqrt[n]{x} and \sqrt{x}
        while True:
            pos = s.find(r"\sqrt")
            if pos < 0:
                break
            rest_idx = pos + 5
            root_n = ""
            if rest_idx < len(s) and s[rest_idx] == "[":
                end_b = s.find("]", rest_idx + 1)
                if end_b > rest_idx:
                    root_n = s[rest_idx + 1 : end_b].strip()
                    rest_idx = end_b + 1
            arg = _find_braced_arg(s, rest_idx)
            if not arg:
                s = s[:pos] + "√" + s[pos + 5 :]
                continue
            inner, after_end = arg
            inner_c = cls.latex_expr_to_unicode(inner).strip()
            rad = "∛" if root_n == "3" else ("∜" if root_n == "4" else (f"{_map_script(root_n, _SUP_MAP, '')}√" if root_n else "√"))
            s = s[:pos] + f"{rad}({inner_c})" + s[after_end:]

        # \text{...}, \mathrm{...}, \mathbf{...}, \vec{...}, \hat{...}
        for cmd in (r"\text", r"\mathrm", r"\mathbf", r"\operatorname"):
            while True:
                pos = s.find(cmd + "{")
                if pos < 0:
                    break
                arg = _find_braced_arg(s, pos + len(cmd))
                if not arg:
                    break
                inner, after_end = arg
                s = s[:pos] + inner + s[after_end:]

        while True:
            pos = s.find(r"\vec{")
            if pos < 0:
                break
            arg = _find_braced_arg(s, pos + 4)
            if not arg:
                break
            inner, after_end = arg
            s = s[:pos] + f"{inner.strip()}⃗" + s[after_end:]

        while True:
            pos = s.find(r"\overline{")
            if pos < 0:
                break
            arg = _find_braced_arg(s, pos + 9)
            if not arg:
                break
            inner, after_end = arg
            s = s[:pos] + f"{inner.strip()}̄" + s[after_end:]

        # Known commands -> Unicode
        for pat, rep in _LATEX_SYMBOL_TABLE:
            s = s.replace(pat, rep)

        # Braced exponents & subscripts: ^{...}, _{...}
        while True:
            pos = s.find("^{")
            if pos < 0:
                break
            arg = _find_braced_arg(s, pos + 1)
            if not arg:
                break
            inner, after_end = arg
            s = s[:pos] + _map_script(inner, _SUP_MAP, "^") + s[after_end:]

        while True:
            pos = s.find("_{")
            if pos < 0:
                break
            arg = _find_braced_arg(s, pos + 1)
            if not arg:
                break
            inner, after_end = arg
            s = s[:pos] + _map_script(inner, _SUB_MAP, "_") + s[after_end:]

        # Single-char exponents & subscripts: ^2, ^n, _1
        s = re.sub(r"\^([0-9a-zA-Z+\-])", lambda m: _map_script(m.group(1), _SUP_MAP, "^"), s)
        s = re.sub(r"_([0-9a-zA-Z+\-])", lambda m: _map_script(m.group(1), _SUB_MAP, "_"), s)

        # Strip leftover unknown backslash noise like `\begin{cases}` cleanly
        s = s.replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "")
        s = s.replace(r"\begin{cases}", "{ ").replace(r"\end{cases}", " }")
        s = s.replace(r"\\", "\n").replace("&", " ")
        s = re.sub(r"\s{3,}", "  ", s)
        return s.strip()

    @classmethod
    def sanitize_plain_text(cls, raw: str) -> str:
        """
        Convert full mixed AI response (text + inline `$...$` + display `$$...$$` + `<arrow>`)
        into zero-garbage Unicode math & prose suitable for Canvas annotations, Copy, and Rust UI.
        """
        s = cls.normalize_delimiters_and_tags(raw)
        if not s:
            return ""

        # Replace paired $$...$$ blocks
        s = re.sub(
            r"\$\$(.+?)\$\$",
            lambda m: cls.latex_expr_to_unicode(m.group(1)),
            s,
            flags=re.DOTALL,
        )
        # Replace paired $...$ spans
        s = re.sub(
            r"\$([^$\n]+?)\$",
            lambda m: cls.latex_expr_to_unicode(m.group(1)),
            s,
        )
        # Clean any remaining LaTeX commands & unpaired '$' signs outside delimiters
        s = cls.latex_expr_to_unicode(s)
        s = cls.strip_all_stray_dollars(s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    @classmethod
    def _split_fenced_code_blocks(cls, raw: str) -> List[Tuple[str, str, str]]:
        """
        Split raw Markdown into `("text", "", payload)` and `("code", lang, code_body)` segments
        BEFORE LaTeX/tag normalization so code syntax (`->`, `=>`, `_`, `&`, indentation, `<tags>`)
        inside fenced blocks is preserved verbatim.
        """
        normalized_nl = str(raw).replace("\r\n", "\n").replace("\r", "\n")
        lines = normalized_nl.split("\n")
        segments: List[Tuple[str, str, str]] = []
        text_buf: List[str] = []
        i = 0
        n = len(lines)
        while i < n:
            stripped = lines[i].strip()
            if stripped.startswith("```"):
                if text_buf:
                    segments.append(("text", "", "\n".join(text_buf)))
                    text_buf = []
                lang = stripped[3:].strip()
                code_lines: List[str] = []
                i += 1
                while i < n and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if i < n:
                    i += 1
                segments.append(("code", lang, "\n".join(code_lines)))
                continue
            text_buf.append(lines[i])
            i += 1
        if text_buf:
            segments.append(("text", "", "\n".join(text_buf)))
        return segments

    @classmethod
    def _render_code_block_html(
        cls,
        lang: str,
        code_body: str,
        *,
        code_bg: str,
        code_color: str,
        border_col: str,
        lang_color: str,
    ) -> str:
        lang_hdr = (
            f'<div style="color:{lang_color};font-size:10px;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">'
            f'{html.escape(lang)}</div>'
            if lang
            else ""
        )
        code_esc = html.escape(code_body)
        return (
            f'<div style="margin:6px 0;padding:8px 10px;background:{code_bg};'
            f'border:1px solid {border_col};border-radius:8px;">'
            f'{lang_hdr}<pre style="margin:0;white-space:pre-wrap;'
            f'font-family:Consolas,monospace;font-size:12px;color:{code_color};">'
            f'<code style="font-family:Consolas,monospace;color:{code_color};'
            f'background:transparent;">{code_esc}</code></pre></div>'
        )

    @classmethod
    def _render_inline_with_math(
        cls,
        raw_line: str,
        *,
        accent_color: str,
        arrow_color: str,
        code_bg: str,
        render_math_images: bool = True,
    ) -> str:
        """Render a single Markdown line/cell supporting inline code, inline `$...$` math, and arrows."""
        if not raw_line:
            return ""
        inline_pattern = re.compile(r"`([^`\n]+?)`|\$([^$\n]+?)\$")
        out: List[str] = []
        pos = 0
        for m in inline_pattern.finditer(raw_line):
            if m.start() > pos:
                chunk = raw_line[pos : m.start()]
                lead = " " if chunk[:1].isspace() else ""
                trail = " " if chunk[-1:].isspace() else ""
                cleaned = cls.strip_all_stray_dollars(cls.latex_expr_to_unicode(chunk))
                if cleaned:
                    out.append(
                        lead
                        + cls._inline_markdown_to_html(cleaned, accent_color, arrow_color)
                        + trail
                    )
                elif chunk:
                    out.append(" ")

            if m.group(1) is not None:
                code_esc = html.escape(m.group(1))
                out.append(
                    f'<code style="background:{code_bg};color:{accent_color};'
                    f'font-family:Consolas,monospace;font-weight:700;padding:1px 4px;'
                    f'border-radius:4px;">{code_esc}</code>'
                )
            else:
                payload = m.group(2) or ""
                needs_raster = any(
                    k in payload for k in (r"\frac", r"\dfrac", r"\sqrt", r"\int", r"\sum", r"\lim")
                )
                rendered_png = (
                    render_latex_to_png_bytes(payload, color_hex=accent_color, fontsize=11, dpi=120)
                    if (render_math_images and needs_raster)
                    else None
                )
                if rendered_png:
                    b64 = base64.b64encode(rendered_png).decode("ascii")
                    alt = html.escape(cls.latex_expr_to_unicode(payload))
                    out.append(
                        f'<img style="vertical-align:middle;" src="data:image/png;base64,{b64}" alt="{alt}"/>'
                    )
                else:
                    uni = html.escape(cls.latex_expr_to_unicode(payload))
                    out.append(
                        f'<span style="color:{accent_color}; font-weight:700;">{uni}</span>'
                    )
            pos = m.end()

        if pos < len(raw_line):
            chunk = raw_line[pos:]
            lead = " " if (pos > 0 and chunk[:1].isspace()) else ""
            trail = " " if chunk[-1:].isspace() else ""
            cleaned = cls.strip_all_stray_dollars(cls.latex_expr_to_unicode(chunk))
            if cleaned:
                out.append(
                    lead
                    + cls._inline_markdown_to_html(cleaned, accent_color, arrow_color)
                    + (trail if pos > 0 else "")
                )
            elif chunk and pos > 0:
                out.append(" ")

        return "".join(out)

    @classmethod
    def to_rich_html(
        cls,
        raw: str,
        dark_mode: bool = True,
        render_math_images: bool = True,
    ) -> str:
        """
        Sanitize AI text into crisp Qt-ready Rich HTML:
        - Fenced code blocks (` ```lang ... ``` `) rendered verbatim in styled `<pre><code>` blocks
        - Markdown tables (`| Header | ... |`) rendered as clean bordered HTML `<table>`s
          (preserving inline math inside cells) in both Dark and Light mode
        - `$$...$$` rendered to crisp offline `matplotlib.mathtext` PNG `<img>` (with Unicode fallback)
        - Inline complex math (`$...$`) rendered sharply via PNG or Unicode
        - `<arrow>`, `->`, `=>`, `<=>` rendered with crisp inline SVG vector arrows
        - Zero literal `$`, `$$$`, or raw HTML tags ever displayed outside code blocks.
        """
        text_color = "#F8FAFC" if dark_mode else "#0F172A"
        accent_color = "#00F5D4" if dark_mode else "#4F46E5"
        arrow_color = "#06B6D4" if dark_mode else "#2563EB"
        card_bg = "rgba(30, 41, 59, 0.72)" if dark_mode else "rgba(238, 242, 255, 0.9)"
        code_bg = "#0F172A" if dark_mode else "#F1F5F9"
        code_color = "#E2E8F0" if dark_mode else "#0F172A"
        table_hdr_bg = "#1E293B" if dark_mode else "#E2E8F0"
        table_alt_bg = "rgba(15, 23, 42, 0.45)" if dark_mode else "#F8FAFC"
        lang_color = "#94A3B8" if dark_mode else "#64748B"
        border_col = "#334155" if dark_mode else "#CBD5E1"

        if not raw or not str(raw).strip():
            return ""

        top_segments = cls._split_fenced_code_blocks(raw)
        html_parts: List[str] = []

        block_math_re = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)

        for seg_kind, seg_lang, seg_payload in top_segments:
            if seg_kind == "code":
                html_parts.append(
                    cls._render_code_block_html(
                        seg_lang,
                        seg_payload,
                        code_bg=code_bg,
                        code_color=code_color,
                        border_col=border_col,
                        lang_color=lang_color,
                    )
                )
                continue

            s = cls.normalize_delimiters_and_tags(seg_payload)
            if not s:
                continue

            pos = 0
            for m in block_math_re.finditer(s):
                if m.start() > pos:
                    html_parts.append(
                        cls._format_markdown_segment_html(
                            s[pos : m.start()],
                            text_color=text_color,
                            accent_color=accent_color,
                            arrow_color=arrow_color,
                            card_bg=card_bg,
                            border_col=border_col,
                            code_bg=code_bg,
                            code_color=code_color,
                            table_hdr_bg=table_hdr_bg,
                            table_alt_bg=table_alt_bg,
                            lang_color=lang_color,
                            render_math_images=render_math_images,
                        )
                    )
                payload = m.group(1)
                rendered_png = (
                    render_latex_to_png_bytes(payload, color_hex=accent_color, fontsize=14, dpi=140)
                    if render_math_images
                    else None
                )
                if rendered_png:
                    b64 = base64.b64encode(rendered_png).decode("ascii")
                    alt = html.escape(cls.latex_expr_to_unicode(payload))
                    html_parts.append(
                        f'<div style="margin:6px 0; padding:6px 10px; background:{card_bg}; '
                        f'border:1px solid {border_col}; border-radius:8px; text-align:center;">'
                        f'<img src="data:image/png;base64,{b64}" alt="{alt}"/></div>'
                    )
                else:
                    uni = html.escape(cls.latex_expr_to_unicode(payload))
                    html_parts.append(
                        f'<div style="margin:6px 0; padding:6px 10px; background:{card_bg}; '
                        f'border:1px solid {border_col}; border-radius:8px; color:{accent_color}; '
                        f'font-weight:700; text-align:center;">{uni}</div>'
                    )
                pos = m.end()

            if pos < len(s):
                html_parts.append(
                    cls._format_markdown_segment_html(
                        s[pos:],
                        text_color=text_color,
                        accent_color=accent_color,
                        arrow_color=arrow_color,
                        card_bg=card_bg,
                        border_col=border_col,
                        code_bg=code_bg,
                        code_color=code_color,
                        table_hdr_bg=table_hdr_bg,
                        table_alt_bg=table_alt_bg,
                        lang_color=lang_color,
                        render_math_images=render_math_images,
                    )
                )

        return (
            f'<div style="color:{text_color}; font-family:Segoe UI,Arial,sans-serif; '
            f'font-size:13px; line-height:1.5;">{"".join(html_parts)}</div>'
        )

    @classmethod
    def _inline_markdown_to_html(
        cls,
        raw_line: str,
        accent_color: str,
        arrow_color: str,
    ) -> str:
        esc = html.escape(raw_line)
        esc = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", esc)
        esc = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"<i>\1</i>", esc)
        esc = re.sub(
            r"`([^`]+?)`",
            f'<code style="color:{accent_color};font-weight:700;padding:1px 4px;border-radius:4px;">\\1</code>',
            esc,
        )
        esc = esc.replace("→", f'<img style="vertical-align:middle;" src="{arrow_svg_data_uri("right", arrow_color, 13)}" alt="→"/>')
        esc = esc.replace("⇒", f'<img style="vertical-align:middle;" src="{arrow_svg_data_uri("implies", arrow_color, 13)}" alt="⇒"/>')
        esc = esc.replace("⇔", f'<img style="vertical-align:middle;" src="{arrow_svg_data_uri("double", arrow_color, 14)}" alt="⇔"/>')
        esc = esc.replace("←", f'<img style="vertical-align:middle;" src="{arrow_svg_data_uri("left", arrow_color, 13)}" alt="←"/>')
        return esc

    @classmethod
    def _format_markdown_segment_html(
        cls,
        segment: str,
        *,
        text_color: str,
        accent_color: str,
        arrow_color: str,
        card_bg: str,
        border_col: str,
        code_bg: Optional[str] = None,
        code_color: Optional[str] = None,
        table_hdr_bg: Optional[str] = None,
        table_alt_bg: Optional[str] = None,
        lang_color: str = "#94A3B8",
        render_math_images: bool = True,
    ) -> str:
        if not segment:
            return ""
        eff_code_bg = code_bg or card_bg
        eff_code_color = code_color or text_color
        eff_tbl_hdr = table_hdr_bg or card_bg
        eff_tbl_alt = table_alt_bg or "transparent"

        lines = segment.splitlines()
        out: List[str] = []
        i = 0
        n = len(lines)

        def _is_table_sep(line: str) -> bool:
            s = line.strip()
            return bool("|" in s and "-" in s and re.fullmatch(r"[\|\s:\-]+", s))

        def _split_table_cells(line: str) -> List[str]:
            s = line.strip()
            if s.startswith("|"):
                s = s[1:]
            if s.endswith("|"):
                s = s[:-1]
            return [c.strip() for c in s.split("|")]

        def _render_inline(text: str) -> str:
            return cls._render_inline_with_math(
                text,
                accent_color=accent_color,
                arrow_color=arrow_color,
                code_bg=eff_code_bg,
                render_math_images=render_math_images,
            )

        while i < n:
            line = lines[i]
            stripped = line.strip()

            # Fenced code block ``` (fallback if called directly)
            if stripped.startswith("```"):
                lang = stripped[3:].strip()
                code_lines: List[str] = []
                i += 1
                while i < n and not lines[i].strip().startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                if i < n:
                    i += 1
                out.append(
                    cls._render_code_block_html(
                        lang,
                        "\n".join(code_lines),
                        code_bg=eff_code_bg,
                        code_color=eff_code_color,
                        border_col=border_col,
                        lang_color=lang_color,
                    )
                )
                continue

            # Markdown table | col1 | col2 | (supports both piped and pipe-separated GFM rows)
            if "|" in stripped and i + 1 < n and _is_table_sep(lines[i + 1]):
                header_cells = _split_table_cells(stripped)
                col_count = max(1, len(header_cells))
                i += 2
                body_rows: List[List[str]] = []
                while i < n:
                    row_s = lines[i].strip()
                    if not row_s or "|" not in row_s or row_s.startswith("```"):
                        break
                    cells = _split_table_cells(row_s)
                    while len(cells) < col_count:
                        cells.append("")
                    body_rows.append(cells[:col_count])
                    i += 1
                ths = "".join(
                    f'<th style="padding:6px 8px;border:1px solid {border_col};background:{eff_tbl_hdr};'
                    f'color:{accent_color};text-align:left;font-weight:700;">'
                    f'{_render_inline(c)}</th>'
                    for c in header_cells
                )
                trs = []
                for r_idx, r in enumerate(body_rows):
                    row_bg = eff_tbl_alt if (r_idx % 2 == 1) else "transparent"
                    tds = "".join(
                        f'<td style="padding:5px 8px;border:1px solid {border_col};'
                        f'background:{row_bg};color:{text_color};">'
                        f'{_render_inline(c)}</td>'
                        for c in r
                    )
                    trs.append(f"<tr>{tds}</tr>")
                out.append(
                    f'<table border="1" cellspacing="0" style="border-collapse:collapse;margin:8px 0;'
                    f'width:100%;font-size:12px;border:1px solid {border_col};">'
                    f'<thead><tr>{ths}</tr></thead><tbody>{"".join(trs)}</tbody></table>'
                )
                continue

            # Markdown headings # .. ####
            h_match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
            if h_match:
                lvl = len(h_match.group(1))
                fsize = {1: 16, 2: 15, 3: 14, 4: 13}.get(lvl, 13)
                txt_html = _render_inline(h_match.group(2))
                out.append(
                    f'<div style="font-size:{fsize}px;font-weight:700;color:{accent_color};margin:8px 0 3px 0;">{txt_html}</div>'
                )
                i += 1
                continue

            # Horizontal rule ---
            if re.fullmatch(r"[-*_]{3,}", stripped):
                out.append(f'<hr style="border:none;border-top:1px solid {border_col};margin:6px 0;"/>')
                i += 1
                continue

            # Bullet or numbered list line
            b_match = re.match(r"^[-*•]\s+(.+)$", stripped)
            n_match = re.match(r"^(\d+[\.\)])\s+(.+)$", stripped)
            if b_match:
                txt_html = _render_inline(b_match.group(1))
                out.append(f'<div style="margin:2px 0 2px 10px;"><span style="color:{accent_color};font-weight:700;">•</span> {txt_html}</div>')
                i += 1
                continue
            if n_match:
                num_lbl = html.escape(n_match.group(1))
                txt_html = _render_inline(n_match.group(2))
                out.append(f'<div style="margin:2px 0 2px 10px;"><span style="color:{accent_color};font-weight:700;">{num_lbl}</span> {txt_html}</div>')
                i += 1
                continue

            if not stripped:
                out.append("<br/>")
            else:
                out.append(_render_inline(line) + ("<br/>" if i < n - 1 else ""))
            i += 1

        return "".join(out)

    @classmethod
    def to_katex_html(cls, raw: str, dark_mode: bool = True) -> str:
        """
        Produce a complete standalone HTML fragment compatible with KaTeX auto-render
        (using normalized `$...$` / `$$...$$` delimiters, SVG arrows, and zero broken `$$$`).
        """
        s = cls.normalize_delimiters_and_tags(raw)
        arrow_color = "#06B6D4" if dark_mode else "#2563EB"
        esc = html.escape(s)
        esc = esc.replace("→", f'<img class="at-arrow" src="{arrow_svg_data_uri("right", arrow_color, 14)}" alt="→"/>')
        esc = esc.replace("⇒", f'<img class="at-arrow" src="{arrow_svg_data_uri("implies", arrow_color, 14)}" alt="⇒"/>')
        esc = esc.replace("⇔", f'<img class="at-arrow" src="{arrow_svg_data_uri("double", arrow_color, 14)}" alt="⇔"/>')
        return esc.replace("\n", "<br/>")


@dataclass
class StreamingSanitizer:
    """
    Incremental token buffer holding unclosed `$...$` / `$$...$$` / `<tag>` markers
    so live token-by-token streaming never displays torn syntax.
    When `preserve_latex=True`, emits clean Markdown + balanced `$...$`/`$$...$$` for rich HTML rendering.
    """

    buffer: str = ""
    preserve_latex: bool = False

    def reset(self) -> None:
        self.buffer = ""

    def feed(self, token_delta: str) -> str:
        if not token_delta:
            return ""
        self.buffer += token_delta
        normalized = OutputSanitizer.normalize_delimiters_and_tags(self.buffer)
        self.buffer = normalized

        # Hold back incomplete trailing `$` / `$$` block or trailing `<partial_tag`
        hold_idx = len(normalized)

        # Check unpaired `<` tag near tail (< 24 chars)
        last_lt = normalized.rfind("<")
        last_gt = normalized.rfind(">")
        if last_lt > last_gt and (len(normalized) - last_lt) <= 24:
            hold_idx = min(hold_idx, last_lt)

        # Check trailing `$` delimiter parity before hold_idx
        prefix = normalized[:hold_idx]
        # Count `$$` blocks
        double_positions = [m.start() for m in re.finditer(r"\$\$", prefix)]
        if len(double_positions) % 2 == 1:
            hold_idx = min(hold_idx, double_positions[-1])
        else:
            # Remove `$$` temporarily to check single inline `$` parity
            masked = re.sub(r"\$\$", "__", prefix)
            single_positions = [m.start() for m in re.finditer(r"(?<!\\)\$", masked)]
            if len(single_positions) % 2 == 1:
                hold_idx = min(hold_idx, single_positions[-1])

        if hold_idx <= 0:
            return ""
        ready = normalized[:hold_idx]
        self.buffer = normalized[hold_idx:]
        if self.preserve_latex:
            return ready
        return OutputSanitizer.sanitize_plain_text(ready)

    def flush(self) -> str:
        if self.preserve_latex:
            rem = OutputSanitizer.normalize_delimiters_and_tags(self.buffer)
        else:
            rem = OutputSanitizer.sanitize_plain_text(self.buffer)
        self.buffer = ""
        return rem
