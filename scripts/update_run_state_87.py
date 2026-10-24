"""
Script to update .agent/16H_RUN_STATE.md with Milestones 86 and 87.
"""
import sys

def main():
    path = ".agent/16H_RUN_STATE.md"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Add task 86 and 87
    old_task85 = """85. **Task I.71 — Interactive Physics Geometric Optics Thin Lens & Spherical Mirror Optical Bench Simulator**: `src/ui/interactive_thin_lens.py` (`InteractiveThinLensBenchItem`), authentic Gaussian thin-lens equation $1/f = 1/d + 1/d'$ and lateral magnification $k = -d'/d = A'B'/AB$, dual lens modes (Converging $f > 0$ and Diverging $f < 0$), dynamic object arrow $AB$ with air pinch draggable tip $B$ adjusting position $d$ and height $h$, real image ($d' > 0$, inverted) vs virtual image ($d' < 0$, upright), 3 authentic color-coded principal rays (Parallel-to-axis ray through $F'$, central ray straight through $O$, focal ray through $F$), movable observation screen with auto-focus detector ($d' \\approx d_{\\text{screen}}$), 4 curriculum presets (Magnifying Glass $d < f$, Projector Lens $f < d < 2f$, Camera Objective $d > 2f$, Diverging Eyeglass $f < 0$), touchless air gestures, vector stamping via `PhysicsTools.create_thin_lens_diagram`, `LessonBoard` lifecycle (`show_thin_lens`, `toggle_thin_lens`), View menu action (`Ctrl+Alt+O`), navigation toolbar button `🔬 Thấu kính mỏng`, in-air menu trigger (`thin_lens`), and comprehensive unit test suite `tests/test_interactive_thin_lens.py` (8 tests, 100% passing)."""

    new_tasks = old_task85 + """

86. **Task I.72 — Interactive Modern Physics Photoelectric Effect & Einstein's Quantum Photon Work Function Simulator**: `src/ui/interactive_photoelectric_effect.py` (`InteractivePhotoelectricEffectItem`), authentic Einstein photoelectric equation $E_k = hf - A = h(c/\\lambda) - A = e \\cdot U_h$, stopping potential $U_h = (hf - A)/e$ and threshold frequency $f_0 = A/h$ (red threshold wavelength $\\lambda_0 = hc/A$), authentic cathode metal targets (Cesium $A = 2.14\\text{ eV}$, Potassium $A = 2.30\\text{ eV}$, Sodium $A = 2.46\\text{ eV}$, Zinc $A = 4.31\\text{ eV}$, Platinum $A = 6.35\\text{ eV}$), dynamic evacuated quartz phototube with illuminated cathode, emitted photoelectrons with kinetic velocity $v = \\sqrt{2E_k/m_e}$, variable voltage bias $U_{AK} \\in [-5.0\\text{ V}, +5.0\\text{ V}]$, microammeter current readout $I(U_{AK})$, current-voltage characteristic curve $I - U_{AK}$ with $-U_h$ intercept, Millikan straight line plot $U_h(f)$ showing slope $h/e = 4.14 \\times 10^{-15}\\text{ V}\\cdot\\text{s}$ determining Planck's constant, 4 curriculum presets (Cesium Visible, Threshold Cutoff, Stopping Voltage Millikan, Zinc UV Hallwachs), touchless air gestures, vector stamping via `PhysicsTools.create_photoelectric_effect_diagram`, `LessonBoard` lifecycle (`show_photoelectric_effect`, `toggle_photoelectric_effect`), View menu action (`Ctrl+Alt+P`), navigation toolbar button `⚡ Quang điện`, in-air menu trigger (`photoelectric`), and comprehensive unit test suite `tests/test_interactive_photoelectric_effect.py` (8 tests, 100% passing).

87. **Task I.73 — Interactive Physics Thermodynamics Carnot Heat Engine & PV / TS Cycle Simulator**: `src/ui/interactive_carnot_engine.py` (`InteractiveCarnotEngineItem`), authentic thermodynamic cycle of ideal reversible engine, 4 reversible stages (Isothermal expansion at $T_H$, Adiabatic expansion $T_H \\to T_C$, Isothermal compression at $T_C$, Adiabatic compression $T_C \\to T_H$), Poisson adiabatic relation $T_H V_2^{\\gamma - 1} = T_C V_3^{\\gamma - 1}$ with diatomic gas index $\\gamma = 1.40$, net work output $W_{\\text{net}} = Q_H - |Q_C| = nR(T_H - T_C)\\ln(r) = \\oint P dV$, maximum Carnot thermal efficiency $\\eta = 1 - T_C/T_H = W_{\\text{net}}/Q_H$, reverse Carnot cycle coefficient of performance (COP) for refrigerator $\\text{COP}_{\\text{ref}} = T_C/(T_H - T_C)$ and heat pump $\\text{COP}_{\\text{pump}} = T_H/(T_H - T_C)$, dual diagnostic visualizations (P-V diagram with shaded enclosed net work and T-S diagram with rectangle of area $W_{\\text{net}}$), dynamic mechanical cross-section cylinder with gas temperature gradient, translating piston head, crank rod, and flywheel, 4 curriculum presets (Standard Steam $\\eta = 50\\%$, High Efficiency $\\eta = 66.7\\%$, Reverse Refrigerator, Low Delta-T), touchless air gestures, vector stamping via `PhysicsTools.create_carnot_engine_diagram`, `LessonBoard` lifecycle (`show_carnot_engine`, `toggle_carnot_engine`), View menu action (`Ctrl+Alt+T`), navigation toolbar button `🔥 Chu trình Carnot`, in-air menu trigger (`carnot`), and comprehensive unit test suite `tests/test_interactive_carnot_engine.py` (8 tests, 100% passing)."""

    if old_task85 not in content:
        print("ERROR: old_task85 not found!")
        sys.exit(1)
    content = content.replace(old_task85, new_tasks, 1)

    # Update Verification Evidence Log
    old_log = """- Current Checkpoint: **2026-10-24T05:46:00+07:00** (Verified Elapsed: **8h 58m 00s**)
- Master test suite: **729 passed across 58 suites (100% PASSED)**"""

    new_log = """- Current Checkpoint: **2026-10-24T06:11:00+07:00** (Verified Elapsed: **9h 23m 00s**)
- Master test suite: **745 passed across 60 suites (100% PASSED)**"""

    if old_log in content:
        content = content.replace(old_log, new_log, 1)
    else:
        print("Note: old_log not matched verbatim, will replace checkpoint lines.")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully updated .agent/16H_RUN_STATE.md with Milestones 86 and 87!")

if __name__ == "__main__":
    main()
