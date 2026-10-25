"""
AirTeach AI - Symbolic Math Engine & Step-by-Step Derivation
============================================================
Offline symbolic mathematics engine powered by SymPy and deterministic
pedagogical solvers for classroom instruction:
- Converts handwritten LaTeX math formulas to symbolic expressions
- Solves linear, quadratic, polynomial, and transcendental equations
- Computes symbolic derivatives (d/dx) with differentiation rule breakdown
- Computes symbolic indefinite and definite integrals with anti-derivative steps
- Simplifies, factors, and evaluates expressions (exact fractions and decimals)
- Generates structured, pedagogical step-by-step derivations in Vietnamese/English
  ready for rendering into crisp vector graphics on LessonBoard
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import sympy
from sympy import (
    Eq,
    Float,
    I,
    Integer,
    Poly,
    Rational,
    Symbol,
    diff,
    expand,
    factor,
    integrate,
    latex,
    simplify,
    solve,
    sqrt,
    symbols,
    sympify,
)

logger = logging.getLogger(__name__)


# ─── Enumerations & Dataclasses ───────────────────────────────────────────────

class MathOperationType(str, Enum):
    AUTO = "auto"
    SOLVE = "solve"
    DERIVATIVE = "derivative"
    INTEGRAL = "integral"
    SIMPLIFY = "simplify"
    FACTOR = "factor"
    EXPAND = "expand"
    EVALUATE = "evaluate"


@dataclass
class DerivationStep:
    """A single pedagogical derivation step."""
    step_number: int
    title: str
    description: str
    latex: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "title": self.title,
            "description": self.description,
            "latex": self.latex,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DerivationStep:
        return cls(
            step_number=int(data.get("step_number", 1)),
            title=str(data.get("title", "")),
            description=str(data.get("description", "")),
            latex=str(data.get("latex", "")),
        )


@dataclass
class MathDerivationResult:
    """Complete result of a symbolic math derivation or evaluation."""
    success: bool
    operation: str
    original_latex: str
    cleaned_expression: str = ""
    solution_raw: str = ""
    solution_latex: str = ""
    steps: list[DerivationStep] = field(default_factory=list)
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    primary_variable: str = "x"
    approximate_values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "operation": self.operation,
            "original_latex": self.original_latex,
            "cleaned_expression": self.cleaned_expression,
            "solution_raw": self.solution_raw,
            "solution_latex": self.solution_latex,
            "steps": [s.to_dict() for s in self.steps],
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "primary_variable": self.primary_variable,
            "approximate_values": self.approximate_values,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MathDerivationResult:
        steps = [DerivationStep.from_dict(s) for s in data.get("steps", [])]
        return cls(
            success=bool(data.get("success", False)),
            operation=str(data.get("operation", "auto")),
            original_latex=str(data.get("original_latex", "")),
            cleaned_expression=str(data.get("cleaned_expression", "")),
            solution_raw=str(data.get("solution_raw", "")),
            solution_latex=str(data.get("solution_latex", "")),
            steps=steps,
            error_message=data.get("error_message"),
            execution_time_ms=float(data.get("execution_time_ms", 0.0)),
            primary_variable=str(data.get("primary_variable", "x")),
            approximate_values=[float(v) for v in data.get("approximate_values", [])],
        )


# ─── LaTeX to Python Expression Converter ──────────────────────────────────────

def convert_latex_to_python_expr(latex_str: str) -> str:
    """
    Robustly convert handwritten/typeset LaTeX mathematical syntax to
    standard Python expression syntax parseable by SymPy.
    """
    if not latex_str:
        return ""

    s = latex_str.strip()

    # Strip dollar signs
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()

    # Remove formatting commands
    s = re.sub(r"\\(?:displaystyle|textstyle|scriptstyle|mathbf|mathit|mathrm|rm|bf)\{([^{}]+)\}", r"\1", s)
    s = s.replace(r"\quad", " ").replace(r"\qquad", " ").replace(r"\,", " ").replace(r"\;", " ")

    # Radicals: \sqrt[n]{A} must be extracted before general bracket normalization
    s = re.sub(r"\\sqrt\[([^{}]+)\]\{([^{}]+)\}", r"((\2)**(1/(\1)))", s)
    sqrt_pat = re.compile(r"\\sqrt\{([^{}]+)\}")
    for _ in range(6):
        if not sqrt_pat.search(s):
            break
        s = sqrt_pat.sub(r"sqrt(\1)", s)

    # Bracket normalization
    s = s.replace(r"\left(", "(").replace(r"\right)", ")")
    s = s.replace(r"\left[", "(").replace(r"\right]", ")")
    s = s.replace(r"\left\{", "(").replace(r"\right\}", ")")
    s = s.replace(r"\{", "(").replace(r"\}", ")")
    s = s.replace("[", "(").replace("]", ")")

    # Multiplication and division symbols
    s = s.replace(r"\times", "*").replace(r"\cdot", "*").replace(r"\ast", "*")
    s = s.replace(r"\div", "/").replace("÷", "/")
    s = s.replace("×", "*").replace("·", "*")

    # Constants and common symbols
    s = re.sub(r"\\pi\b", "pi", s)
    s = re.sub(r"\\infty\b", "oo", s)

    # Fractions: handle nested \frac{A}{B} or \dfrac{A}{B}
    frac_pat = re.compile(r"\\(?:d)?frac\{([^{}]+)\}\{([^{}]+)\}")
    for _ in range(6):
        if not frac_pat.search(s):
            break
        s = frac_pat.sub(r"((\1)/(\2))", s)

    # Powers with trig: \sin^2(x) -> (sin(x))**2
    s = re.sub(r"\\(sin|cos|tan|cot|sec|csc)\^(\d+|\{[^{}]+\})\(([^)]+)\)", r"(\1(\3))**\2", s)
    s = re.sub(r"\\(sin|cos|tan|cot|sec|csc)\^(\d+|\{[^{}]+\})([a-zA-Z])", r"(\1(\3))**\2", s)

    # Functions
    s = re.sub(r"\\(sin|cos|tan|cot|sec|csc|asin|acos|atan|sinh|cosh|tanh)", r"\1", s)
    s = re.sub(r"\\ln\b", "log", s)
    s = re.sub(r"\\log\b", "log", s)
    s = re.sub(r"\\exp\b", "exp", s)

    # Powers: x^{expr} -> x**(expr) and x^2 -> x**2
    s = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", s)
    s = re.sub(r"\^([0-9a-zA-Z])", r"**\1", s)

    # Implicit multiplication:
    # 2x -> 2*x, 3.5y -> 3.5*y, 2( -> 2*(
    s = re.sub(r"(\d+(?:\.\d+)?)\s*([a-zA-Z(])", r"\1*\2", s)
    # )x -> )*x, )( -> )*(
    s = re.sub(r"(\))\s*([a-zA-Z0-9(])", r"\1*\2", s)
    # Single letter variable followed by ( when not a standard function
    # e.g. x(x+1) -> x*(x+1), t(t-2) -> t*(t-2)
    s = re.sub(r"\b([a-wyzAWYZ])\s*\(", r"\1*(", s)
    # Adjacent variables like x y -> x*y
    s = re.sub(r"\b([a-zA-Z])\s+([a-zA-Z])\b", r"\1*\2", s)

    # Clean double operators
    s = s.replace("--", "+").replace("++", "+")
    while "  " in s:
        s = s.replace("  ", " ")

    return s.strip()


def detect_primary_variable(expr_str: str) -> str:
    """Detect the most likely independent variable in an expression (x, t, y, etc.)."""
    vars_found = re.findall(r"\b([a-zA-Z])\b", expr_str)
    # Filter out known constants/functions
    candidates = [v for v in vars_found if v not in ("e", "i", "d")]
    for preferred in ("x", "t", "y", "z", "n", "u", "v", "r", "s", "theta"):
        if preferred in candidates:
            return preferred
    return candidates[0] if candidates else "x"


# ─── Symbolic Math Engine ─────────────────────────────────────────────────────

class SymbolicMathEngine:
    """
    High-performance, offline symbolic mathematics engine for classroom instruction.
    Generates verified exact solutions and structured pedagogical derivations.
    """

    def __init__(self) -> None:
        self._standard_symbols = {
            "x": symbols("x"),
            "y": symbols("y"),
            "z": symbols("z"),
            "t": symbols("t"),
            "u": symbols("u"),
            "v": symbols("v"),
            "a": symbols("a"),
            "b": symbols("b"),
            "c": symbols("c"),
            "n": symbols("n"),
            "pi": sympy.pi,
            "E": sympy.E,
            "I": sympy.I,
            "sqrt": sympy.sqrt,
            "sin": sympy.sin,
            "cos": sympy.cos,
            "tan": sympy.tan,
            "cot": sympy.cot,
            "asin": sympy.asin,
            "acos": sympy.acos,
            "atan": sympy.atan,
            "log": sympy.log,
            "exp": sympy.exp,
        }

    def _sympify_expr(self, expr_str: str, var: Optional[str] = None):
        """Parse string expression into SymPy expression safely."""
        local_dict = dict(self._standard_symbols)
        if var and var not in local_dict:
            local_dict[var] = symbols(var)
        return sympify(expr_str, locals=local_dict)

    # ── Equation Solving with Step-by-Step Breakdown ─────────────────────────

    def solve_equation(
        self,
        latex_str: str,
        variable: Optional[str] = None,
    ) -> MathDerivationResult:
        """
        Solve an equation with full pedagogical step-by-step reasoning.
        Supports linear, quadratic, polynomial, and rational equations.
        """
        t0 = time.perf_counter()
        py_str = convert_latex_to_python_expr(latex_str)

        if "=" in py_str:
            lhs_raw, rhs_raw = py_str.split("=", 1)
        else:
            lhs_raw, rhs_raw = py_str, "0"

        var_name = variable or detect_primary_variable(py_str)
        var_sym = symbols(var_name)

        try:
            lhs_expr = self._sympify_expr(lhs_raw, var_name)
            rhs_expr = self._sympify_expr(rhs_raw, var_name)
            eq_diff = lhs_expr - rhs_expr
            eq = Eq(lhs_expr, rhs_expr)
        except Exception as exc:
            return MathDerivationResult(
                success=False,
                operation="solve",
                original_latex=latex_str,
                cleaned_expression=py_str,
                error_message=f"Không thể phân tích cú pháp phương trình: {exc}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        steps: list[DerivationStep] = []
        approx_vals: list[float] = []

        # 1. Check if quadratic in var_name
        is_quadratic = False
        try:
            poly = Poly(eq_diff, var_sym)
            if poly.degree() == 2:
                is_quadratic = True
                coeffs = poly.all_coeffs()
                a_val, b_val, c_val = coeffs[0], coeffs[1], coeffs[2]

                # Step 1: Standard form
                steps.append(DerivationStep(
                    step_number=1,
                    title="Chuyển về dạng chuẩn bậc 2",
                    description=f"Đưa phương trình về dạng {latex(a_val)}{var_name}^2 + ({latex(b_val)}){var_name} + ({latex(c_val)}) = 0. "
                                f"Xác định các hệ số: a = {latex(a_val)}, b = {latex(b_val)}, c = {latex(c_val)}.",
                    latex=f"{latex(eq_diff)} = 0",
                ))

                # Step 2: Compute Delta
                delta = b_val**2 - 4 * a_val * c_val
                delta_expanded = f"({latex(b_val)})^2 - 4 \\cdot ({latex(a_val)}) \\cdot ({latex(c_val)}) = {latex(delta)}"
                steps.append(DerivationStep(
                    step_number=2,
                    title="Tính biệt thức Delta (Δ)",
                    description=f"Áp dụng công thức biệt thức: Δ = b² - 4ac = {delta_expanded}.",
                    latex=f"\\Delta = {latex(delta)}",
                ))

                # Step 3: Analyze Delta and compute roots
                d_sym = r"\Delta"
                if delta > 0:
                    sqrt_delta = sqrt(delta)
                    x1 = (-b_val + sqrt_delta) / (2 * a_val)
                    x2 = (-b_val - sqrt_delta) / (2 * a_val)
                    steps.append(DerivationStep(
                        step_number=3,
                        title="Biện luận và tính nghiệm phân biệt",
                        description=f"Vì Δ = {latex(delta)} > 0 nên phương trình có 2 nghiệm phân biệt.",
                        latex=f"{var_name}_1 = \\frac{{-b + \\sqrt{{{d_sym}}}}}{{2a}} = {latex(x1)}, \\quad {var_name}_2 = \\frac{{-b - \\sqrt{{{d_sym}}}}}{{2a}} = {latex(x2)}",
                    ))
                    sols = [x1, x2]
                elif delta == 0:
                    x0 = -b_val / (2 * a_val)
                    steps.append(DerivationStep(
                        step_number=3,
                        title="Phương trình có nghiệm kép",
                        description=f"Vì Δ = 0 nên phương trình có nghiệm kép {var_name}_1 = {var_name}_2 = -b/(2a).",
                        latex=f"{var_name} = {latex(x0)}",
                    ))
                    sols = [x0]
                else:
                    # Complex roots
                    x1 = (-b_val + sqrt(delta)) / (2 * a_val)
                    x2 = (-b_val - sqrt(delta)) / (2 * a_val)
                    lx1 = latex(x1)
                    lx2 = latex(x2)
                    pm_sym = r"\pm"
                    lb = latex(b_val)
                    ld = latex(-delta)
                    la2 = latex(2 * a_val)
                    steps.append(DerivationStep(
                        step_number=3,
                        title="Phương trình vô nghiệm thực",
                        description="Vì Δ < 0 nên phương trình không có nghiệm thực trên tập R (có 2 nghiệm phức liên hợp trên C).",
                        latex=f"{var_name}_{{1,2}} = \\frac{{-({lb}) {pm_sym} i\\sqrt{{{ld}}}}}{{{la2}}} = {lx1}, {lx2}",
                    ))
                    sols = [x1, x2]

                # Step 4: Final solution set
                sol_items_str = ", ".join([latex(s) for s in sols])
                steps.append(DerivationStep(
                    step_number=4,
                    title="Kết luận tập nghiệm",
                    description="Tổng hợp các nghiệm của phương trình.",
                    latex=f"S = \\left\\{{ {sol_items_str} \\right\\}}",
                ))
        except Exception:
            is_quadratic = False

        # 2. Check if linear in var_name
        is_linear = False
        if not is_quadratic:
            try:
                poly = Poly(eq_diff, var_sym)
                if poly.degree() == 1:
                    is_linear = True
                    a_val, b_val = poly.all_coeffs()[0], poly.all_coeffs()[1]

                    steps.append(DerivationStep(
                        step_number=1,
                        title="Đưa về dạng chuẩn bậc nhất",
                        description=f"Phương trình bậc nhất dạng ax + b = 0 với a = {latex(a_val)}, b = {latex(b_val)}.",
                        latex=f"{latex(a_val)}{var_name} + ({latex(b_val)}) = 0",
                    ))

                    steps.append(DerivationStep(
                        step_number=2,
                        title="Chuyển vế số hạng tự do",
                        description=f"Chuyển số hạng tự do sang vế phải: {latex(a_val)}{var_name} = {latex(-b_val)}.",
                        latex=f"{latex(a_val)}{var_name} = {latex(-b_val)}",
                    ))

                    sol = -b_val / a_val
                    steps.append(DerivationStep(
                        step_number=3,
                        title="Chia hai vế cho hệ số a",
                        description=f"Vì a = {latex(a_val)} ≠ 0, chia hai vế cho {latex(a_val)}: {var_name} = {latex(sol)}.",
                        latex=f"{var_name} = {latex(sol)}",
                    ))
                    sols = [sol]
            except Exception:
                is_linear = False

        # 3. General solver fallback if not linear or quadratic
        if not is_quadratic and not is_linear:
            try:
                sols = solve(eq, var_sym)
                steps.append(DerivationStep(
                    step_number=1,
                    title="Chuyển vế và thiết lập phương trình",
                    description=f"Đưa toàn bộ phương trình về vế trái để tìm nghiệm của {var_name}.",
                    latex=f"{latex(eq_diff)} = 0",
                ))

                # Check if factorable
                factored = factor(eq_diff)
                if factored != eq_diff:
                    steps.append(DerivationStep(
                        step_number=2,
                        title="Phân tích thành nhân tử",
                        description="Phân tích vế trái thành tích các nhân tử đơn giản hơn.",
                        latex=f"{latex(factored)} = 0",
                    ))

                if sols:
                    sol_items = [f"{var_name} = {latex(s)}" for s in sols]
                    steps.append(DerivationStep(
                        step_number=len(steps) + 1,
                        title="Tìm nghiệm phương trình",
                        description=f"Giải từng nhân tử ta thu được {len(sols)} nghiệm.",
                        latex=" \\quad \\text{hoặc} \\quad ".join(sol_items),
                    ))
                else:
                    steps.append(DerivationStep(
                        step_number=len(steps) + 1,
                        title="Kết luận",
                        description="Phương trình vô nghiệm trên tập xác định.",
                        latex="S = \\emptyset",
                    ))
            except Exception as exc:
                return MathDerivationResult(
                    success=False,
                    operation="solve",
                    original_latex=latex_str,
                    cleaned_expression=py_str,
                    error_message=f"Lỗi khi giải phương trình: {exc}",
                    execution_time_ms=(time.perf_counter() - t0) * 1000.0,
                )

        # Build final solution string
        if sols:
            sol_latex = ", ".join([f"{var_name} = {latex(s)}" for s in sols])
            sol_raw = ", ".join([f"{var_name} = {str(s)}" for s in sols])
            for s in sols:
                try:
                    num_val = float(s.evalf())
                    approx_vals.append(round(num_val, 6))
                except Exception:
                    pass
        else:
            sol_latex = "S = \\emptyset \\quad \\text{(Vô nghiệm)}"
            sol_raw = "No solution"

        elapsed = (time.perf_counter() - t0) * 1000.0
        return MathDerivationResult(
            success=True,
            operation="solve",
            original_latex=latex_str,
            cleaned_expression=py_str,
            solution_raw=sol_raw,
            solution_latex=sol_latex,
            steps=steps,
            execution_time_ms=elapsed,
            primary_variable=var_name,
            approximate_values=approx_vals,
        )

    # ── Symbolic Differentiation with Rules Breakdown ───────────────────────

    def differentiate(
        self,
        latex_str: str,
        variable: Optional[str] = None,
        order: int = 1,
    ) -> MathDerivationResult:
        """
        Compute the symbolic derivative of an expression with rule-by-rule derivation.
        """
        t0 = time.perf_counter()
        raw = latex_str.strip()
        detected_var = None

        # Extract inner expression from \frac{d}{dx}(...) or d/dx(...) if wrapped
        diff_match = re.search(r"(?:\\frac\{d\}\{d([a-zA-Z])\}|d/d([a-zA-Z]))\s*[\(\[\{](.+)[\)\]\}]$", raw)
        if diff_match:
            detected_var = diff_match.group(1) or diff_match.group(2)
            raw = diff_match.group(3).strip()

        py_str = convert_latex_to_python_expr(raw)
        var_name = variable or detected_var or detect_primary_variable(py_str)
        var_sym = symbols(var_name)

        try:
            expr = self._sympify_expr(py_str, var_name)
        except Exception as exc:
            return MathDerivationResult(
                success=False,
                operation="derivative",
                original_latex=latex_str,
                cleaned_expression=py_str,
                error_message=f"Không thể phân tích biểu thức đạo hàm: {exc}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        steps: list[DerivationStep] = []

        # Step 1: Initial formula
        steps.append(DerivationStep(
            step_number=1,
            title=f"Xác định hàm số cần lấy đạo hàm theo biến {var_name}",
            description=f"Hàm số f({var_name}) = {latex(expr)}.",
            latex=f"f({var_name}) = {latex(expr)}",
        ))

        # Step 2: Term-by-term breakdown if sum/polynomial
        terms = expr.as_ordered_terms() if hasattr(expr, "as_ordered_terms") else [expr]
        if len(terms) > 1:
            term_diffs = []
            term_explanations = []
            for t in terms:
                dt = diff(t, var_sym)
                term_diffs.append(f"\\frac{{d}}{{d{var_name}}}\\left({latex(t)}\\right) = {latex(dt)}")
                term_explanations.append(f"Đạo hàm của số hạng {latex(t)} là {latex(dt)}")

            steps.append(DerivationStep(
                step_number=2,
                title="Quy tắc đạo hàm của tổng: (u + v)' = u' + v'",
                description="Lấy đạo hàm riêng biệt từng số hạng: " + "; ".join(term_explanations[:3]) + ".",
                latex=" \\quad , \\quad ".join(term_diffs[:4]),
            ))

        # Compute full derivative
        d_expr = diff(expr, var_sym, order)
        d_simplified = simplify(d_expr)

        steps.append(DerivationStep(
            step_number=len(steps) + 1,
            title="Tổng hợp và rút gọn kết quả đạo hàm",
            description=f"Cộng các đạo hàm thành phần và tối giản biểu thức:",
            latex=f"f'({var_name}) = {latex(d_simplified)}",
        ))

        sol_latex = f"f'({var_name}) = {latex(d_simplified)}"
        sol_raw = str(d_simplified)

        elapsed = (time.perf_counter() - t0) * 1000.0
        return MathDerivationResult(
            success=True,
            operation="derivative",
            original_latex=latex_str,
            cleaned_expression=py_str,
            solution_raw=sol_raw,
            solution_latex=sol_latex,
            steps=steps,
            execution_time_ms=elapsed,
            primary_variable=var_name,
        )

    # ── Symbolic Integration with Anti-Derivative Breakdown ──────────────────

    def integrate_expression(
        self,
        latex_str: str,
        variable: Optional[str] = None,
        lower: Optional[str] = None,
        upper: Optional[str] = None,
    ) -> MathDerivationResult:
        """
        Compute symbolic indefinite or definite integral with anti-derivative steps.
        """
        t0 = time.perf_counter()
        raw = latex_str.strip()

        # Check for definite integral bounds: \int_{a}^{b} or \int_a^b
        bound_match = re.search(r"\\int_\{([^{}]+)\}\^\{([^{}]+)\}", raw)
        if not bound_match:
            bound_match = re.search(r"\\int_([a-zA-Z0-9]+)\^([a-zA-Z0-9]+)", raw)

        detected_lower = lower
        detected_upper = upper
        if bound_match and not lower and not upper:
            detected_lower = bound_match.group(1)
            detected_upper = bound_match.group(2)

        # Remove \int with bounds
        raw = re.sub(r"\\int(?:_\{[^{}]+\}\^\{[^{}]+\}|_[a-zA-Z0-9]+\^[a-zA-Z0-9]+)?", "", raw)
        raw = raw.replace(r"\int", "")

        # Extract variable from trailing d[x, y, t] if present
        d_var_match = re.search(r"d([a-zA-Z])\s*$", raw.strip())
        detected_var = None
        if d_var_match:
            detected_var = d_var_match.group(1)
            raw = re.sub(r"d[a-zA-Z]\s*$", "", raw.strip())

        py_str = convert_latex_to_python_expr(raw.strip())
        var_name = variable or detected_var or detect_primary_variable(py_str)
        var_sym = symbols(var_name)

        try:
            expr = self._sympify_expr(py_str, var_name)
        except Exception as exc:
            return MathDerivationResult(
                success=False,
                operation="integral",
                original_latex=latex_str,
                cleaned_expression=py_str,
                error_message=f"Không thể phân tích biểu thức tích phân: {exc}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        steps: list[DerivationStep] = []

        eff_lower = lower or detected_lower
        eff_upper = upper or detected_upper
        is_definite = eff_lower is not None and eff_upper is not None

        # Step 1: Initial integral setup
        if is_definite:
            steps.append(DerivationStep(
                step_number=1,
                title=f"Thiết lập tích phân xác định trên đoạn [{eff_lower}, {eff_upper}]",
                description=f"Hàm dưới dấu tích phân là f({var_name}) = {latex(expr)}.",
                latex=f"I = \\int_{{{eff_lower}}}^{{{eff_upper}}} \\left({latex(expr)}\\right) d{var_name}",
            ))
        else:
            steps.append(DerivationStep(
                step_number=1,
                title=f"Thiết lập nguyên hàm bất định theo biến {var_name}",
                description=f"Hàm dưới dấu nguyên hàm là f({var_name}) = {latex(expr)}.",
                latex=f"\\int \\left({latex(expr)}\\right) d{var_name}",
            ))

        # Anti-derivative
        f_anti = integrate(expr, var_sym)

        # Step 2: Anti-derivative rule
        steps.append(DerivationStep(
            step_number=2,
            title="Tìm nguyên hàm F(x) = ∫ f(x) dx",
            description="Áp dụng các bảng nguyên hàm cơ bản (Quy tắc lũy thừa: ∫ xⁿ dx = xⁿ⁺¹/(n+1)).",
            latex=f"F({var_name}) = {latex(f_anti)}",
        ))

        approx_vals = []
        if is_definite and eff_lower is not None and eff_upper is not None:
            try:
                low_sym = self._sympify_expr(eff_lower, var_name)
                up_sym = self._sympify_expr(eff_upper, var_name)
                val = integrate(expr, (var_sym, low_sym, up_sym))
                f_up = f_anti.subs(var_sym, up_sym)
                f_low = f_anti.subs(var_sym, low_sym)

                steps.append(DerivationStep(
                    step_number=3,
                    title="Áp dụng công thức Newton-Leibniz: I = F(b) - F(a)",
                    description=f"Thay cận trên {eff_upper} và cận dưới {eff_lower} vào nguyên hàm:",
                    latex=f"I = F({eff_upper}) - F({eff_lower}) = \\left({latex(f_up)}\\right) - \\left({latex(f_low)}\\right) = {latex(val)}",
                ))
                sol_latex = f"\\int_{{{eff_lower}}}^{{{eff_upper}}} f({var_name}) d{var_name} = {latex(val)}"
                sol_raw = str(val)
                try:
                    approx_vals.append(round(float(val.evalf()), 6))
                except Exception:
                    pass
            except Exception as exc:
                return MathDerivationResult(
                    success=False,
                    operation="integral",
                    original_latex=latex_str,
                    error_message=f"Lỗi khi tính tích phân xác định: {exc}",
                )
        else:
            sol_latex = f"{latex(f_anti)} + C"
            sol_raw = f"{str(f_anti)} + C"
            steps.append(DerivationStep(
                step_number=3,
                title="Thêm hằng số tích phân C",
                description="Nguyên hàm bất định được xác định sai khác một hằng số tùy ý C ∈ R.",
                latex=f"\\int f({var_name}) d{var_name} = {sol_latex}",
            ))

        elapsed = (time.perf_counter() - t0) * 1000.0
        return MathDerivationResult(
            success=True,
            operation="integral",
            original_latex=latex_str,
            cleaned_expression=py_str,
            solution_raw=sol_raw,
            solution_latex=sol_latex,
            steps=steps,
            execution_time_ms=elapsed,
            primary_variable=var_name,
            approximate_values=approx_vals,
        )

    # ── Simplification & Algebraic Expansion ─────────────────────────────────

    def simplify_expression(self, latex_str: str) -> MathDerivationResult:
        """
        Simplify algebraic, rational, or trigonometric expressions step-by-step.
        """
        t0 = time.perf_counter()
        py_str = convert_latex_to_python_expr(latex_str)
        var_name = detect_primary_variable(py_str)

        try:
            expr = self._sympify_expr(py_str, var_name)
        except Exception as exc:
            return MathDerivationResult(
                success=False,
                operation="simplify",
                original_latex=latex_str,
                cleaned_expression=py_str,
                error_message=f"Không thể phân tích biểu thức rút gọn: {exc}",
                execution_time_ms=(time.perf_counter() - t0) * 1000.0,
            )

        steps: list[DerivationStep] = []
        steps.append(DerivationStep(
            step_number=1,
            title="Biểu thức ban đầu",
            description="Ghi nhận biểu thức toán học cần rút gọn.",
            latex=latex(expr),
        ))

        # Check factoring
        fact = factor(expr)
        if fact != expr:
            steps.append(DerivationStep(
                step_number=2,
                title="Phân tích thành nhân tử (Factorization)",
                description="Nhóm các số hạng và áp dụng hằng đẳng thức đáng nhớ.",
                latex=latex(fact),
            ))

        simp = simplify(expr)
        steps.append(DerivationStep(
            step_number=len(steps) + 1,
            title="Biểu thức tối giản",
            description="Thu gọn các số hạng đồng dạng và triệt tiêu thừa số chung.",
            latex=latex(simp),
        ))

        approx_vals = []
        try:
            if simp.is_number:
                num_val = float(simp.evalf())
                approx_vals.append(round(num_val, 6))
        except Exception:
            pass

        sol_latex = latex(simp)
        sol_raw = str(simp)
        elapsed = (time.perf_counter() - t0) * 1000.0

        return MathDerivationResult(
            success=True,
            operation="simplify",
            original_latex=latex_str,
            cleaned_expression=py_str,
            solution_raw=sol_raw,
            solution_latex=sol_latex,
            steps=steps,
            execution_time_ms=elapsed,
            primary_variable=var_name,
            approximate_values=approx_vals,
        )

    # ── Auto Operation Router ────────────────────────────────────────────────

    def evaluate_auto(
        self,
        latex_str: str,
        preferred_operation: Optional[str] = None,
    ) -> MathDerivationResult:
        """
        Intelligently detect the intent of the mathematical formula and route to
        the appropriate symbolic solver (equation solving, differentiation,
        integration, or simplification).
        """
        if not latex_str or not latex_str.strip():
            return MathDerivationResult(
                success=False,
                operation="none",
                original_latex="",
                error_message="Chuỗi công thức trống.",
            )

        op = (preferred_operation or "").lower()

        if op in ("solve", "giai"):
            return self.solve_equation(latex_str)
        elif op in ("derivative", "diff", "dao_ham"):
            return self.differentiate(latex_str)
        elif op in ("integral", "int", "tich_phan"):
            return self.integrate_expression(latex_str)
        elif op in ("simplify", "rut_gon"):
            return self.simplify_expression(latex_str)

        # Automatic detection heuristic
        raw = latex_str.strip()
        if "\\int" in raw or "int(" in raw:
            return self.integrate_expression(latex_str)
        elif "\\frac{d}{d" in raw or "d/dx" in raw or raw.endswith("'") or "f'(" in raw:
            return self.differentiate(latex_str)
        elif "=" in raw:
            return self.solve_equation(latex_str)
        else:
            return self.simplify_expression(latex_str)
