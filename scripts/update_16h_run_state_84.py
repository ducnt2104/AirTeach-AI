"""
Update .agent/16H_RUN_STATE.md to checkpoint 8h 53m (721 tests passing, Milestones 83 and 84).
"""

def main():
    with open('.agent/16H_RUN_STATE.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update header
    content = content.replace(
        "**Current Checkpoint Timestamp:** `2026-10-24T05:32:00+07:00`",
        "**Current Checkpoint Timestamp:** `2026-10-24T05:41:00+07:00`"
    )
    content = content.replace(
        "**Verified Elapsed Time:** `8h 44m 00s`",
        "**Verified Elapsed Time:** `8h 53m 00s`"
    )
    content = content.replace(
        "**Master Test Suite Status:** **705 / 705 unit tests PASSED (100%) across 55 test suites**",
        "**Master Test Suite Status:** **721 / 721 unit tests PASSED (100%) across 57 test suites**"
    )

    # 2. Add milestones 83 and 84
    milestone_83_84 = '''83. **Task I.69 — Interactive Optics Wave Optics & Young's Double-Slit Diffraction / Multi-Slit Interference Grating Simulator**: `src/ui/interactive_young_interference.py` (`InteractiveYoungInterferenceItem`), authentic wave optics engine with fringe spacing formula $i = \\lambda D / a$, physical intensity distribution $I(x) = I_0 \\cos^2(\\pi ax / (\\lambda D)) \\cdot \\text{sinc}^2(\\pi bx / (\\lambda D))$ including single-slit diffraction envelope, multi-slit grating extension ($N=2..10$), physical spectral wavelength to sRGB conversion, white light dispersion model revealing central white fringe and outer rainbow dispersion bands, interactive measurement crosshair probe ($x$, order $k$, path difference $\\Delta d$, intensity $I$), 5 authentic presets (Red He-Ne 633nm, Green DPSS 532nm, Violet 405nm, White Light Rainbow, Diffraction Grating $N=5$), touchless air gestures, vector stamping via `PhysicsTools.create_wave_interference_diagram`, `LessonBoard` lifecycle (`show_young_interference`, `toggle_young_interference`), View menu action (`Ctrl+Alt+Y`), navigation toolbar button `🌈 Giao thoa sóng`, in-air menu trigger (`young_interference`), and comprehensive unit test suite `tests/test_interactive_young_interference.py` (8 tests, 100% passing).
84. **Task I.70 — Interactive Physics Electricity & Alternating Current (AC) Series RLC Circuit & Phasor Diagram Simulator**: `src/ui/interactive_rlc_circuit.py` (`InteractiveRlcCircuitItem`), authentic AC series RLC circuit impedance $Z = \\sqrt{R^2 + (Z_L - Z_C)^2}$, inductive reactance $Z_L = 2\\pi f L$, capacitive reactance $Z_C = 1/(2\\pi f C)$, resonance condition $Z_L = Z_C \\implies f_0 = 1/(2\\pi\\sqrt{LC})$, phase shift $\\tan\\varphi = (Z_L - Z_C)/R$, power factor $\\cos\\varphi = R/Z$, active power $P = UI\\cos\\varphi$, Quality factor $Q = (1/R)\\sqrt{L/C}$, circuit schematic with AC source and dynamic oscillating current, rotating Fresnel Phasor diagram ($U_R, U_L, U_C, U$), dual-trace digital oscilloscope showing $u(t)$ and $i(t)$, instant resonance snap button, 4 curriculum presets (Resonance 50Hz, Inductive Circuit, Capacitive Circuit, High-Q Resonance), touchless air gestures, vector stamping via `PhysicsTools.create_rlc_circuit_diagram`, `LessonBoard` lifecycle (`show_rlc_circuit`, `toggle_rlc_circuit`), View menu action (`Ctrl+Alt+E`), navigation toolbar button `⚡ Mạch RLC`, in-air menu trigger (`rlc_circuit`), and comprehensive unit test suite `tests/test_interactive_rlc_circuit.py` (8 tests, 100% passing).
'''

    target_milestones = '82. **Task I.68 — Interactive Special Relativity Spacetime Minkowski Diagram & Lorentz Transformation / Time Dilation Simulator**'
    assert target_milestones in content, "target_milestones not found"
    idx = content.find(target_milestones)
    end_line_idx = content.find('\n\n---', idx)
    content = content[:end_line_idx] + '\n' + milestone_83_84 + content[end_line_idx:]

    # 3. Update Backlog
    backlog_text = '''## 4. PHASE I ACTIVE BACKLOG (15h – 16h)

1. **Continuous 16-Hour Execution & Long-Horizon Stability Monitoring**:
   - Maintain standing cron triggers until minimum 16 real elapsed hours are achieved (`2026-10-24T12:48:00+07:00`).
   - Monitor test regressions and system performance across all 57 suites.
2. **Task I.71 — Interactive Physics Geometric Optics Thin Lens & Spherical Mirror Optical Bench Simulator (`InteractiveThinLensBenchItem`)**:
   - Authentic Gaussian thin-lens equation $1/f = 1/d + 1/d'$ and lateral magnification $k = -d'/d = A'B'/AB$.
   - Converging lens ($f > 0$), Diverging lens ($f < 0$), Concave spherical mirror ($R > 0$), Convex spherical mirror ($R < 0$).
   - Dynamic object handle $AB$ with draggable height and position $d$, focal length adjustment slider $f$.
   - 3 authentic principal rays (Parallel-to-axis ray passing through focus $F'$, central ray passing straight through optical center $O$, and focal ray passing through $F$ emerging parallel to axis).
   - Real image ($d' > 0$) vs virtual image ($d' < 0$) with upright/inverted classification.
   - Vector stamping via `PhysicsTools.create_thin_lens_diagram`, LessonBoard lifecycle, and main window actions.
'''
    backlog_start = content.find('## 4. PHASE I ACTIVE BACKLOG (15h – 16h)')
    backlog_end = content.find('## 5. VERIFICATION EVIDENCE LOG')
    assert backlog_start != -1 and backlog_end != -1, "backlog boundaries not found"
    content = content[:backlog_start] + backlog_text + '\n---\n\n' + content[backlog_end:]

    # 4. Update Verification evidence log
    content = content.replace(
        "- Current Checkpoint: **2026-10-24T05:32:00+07:00** (Verified Elapsed: **8h 44m 00s**)",
        "- Current Checkpoint: **2026-10-24T05:41:00+07:00** (Verified Elapsed: **8h 53m 00s**)"
    )
    content = content.replace(
        "- Master test suite: **705 passed across 55 suites (100% PASSED)**",
        "- Master test suite: **721 passed across 57 suites (100% PASSED)**"
    )

    with open('.agent/16H_RUN_STATE.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated .agent/16H_RUN_STATE.md to checkpoint 8h 53m!")

if __name__ == '__main__':
    main()
