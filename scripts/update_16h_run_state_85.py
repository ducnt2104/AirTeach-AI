"""
Update .agent/16H_RUN_STATE.md to checkpoint 8h 58m (729 tests passing, Milestone 85).
"""

def main():
    with open('.agent/16H_RUN_STATE.md', 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Update header
    content = content.replace(
        "**Current Checkpoint Timestamp:** `2026-10-24T05:41:00+07:00`",
        "**Current Checkpoint Timestamp:** `2026-10-24T05:46:00+07:00`"
    )
    content = content.replace(
        "**Verified Elapsed Time:** `8h 53m 00s`",
        "**Verified Elapsed Time:** `8h 58m 00s`"
    )
    content = content.replace(
        "**Master Test Suite Status:** **721 / 721 unit tests PASSED (100%) across 57 test suites**",
        "**Master Test Suite Status:** **729 / 729 unit tests PASSED (100%) across 58 test suites**"
    )

    # 2. Add milestone 85
    milestone_85 = '''85. **Task I.71 — Interactive Physics Geometric Optics Thin Lens & Spherical Mirror Optical Bench Simulator**: `src/ui/interactive_thin_lens.py` (`InteractiveThinLensBenchItem`), authentic Gaussian thin-lens equation $1/f = 1/d + 1/d'$ and lateral magnification $k = -d'/d = A'B'/AB$, dual lens modes (Converging $f > 0$ and Diverging $f < 0$), dynamic object arrow $AB$ with air pinch draggable tip $B$ adjusting position $d$ and height $h$, real image ($d' > 0$, inverted) vs virtual image ($d' < 0$, upright), 3 authentic color-coded principal rays (Parallel-to-axis ray through $F'$, central ray straight through $O$, focal ray through $F$), movable observation screen with auto-focus detector ($d' \\approx d_{\\text{screen}}$), 4 curriculum presets (Magnifying Glass $d < f$, Projector Lens $f < d < 2f$, Camera Objective $d > 2f$, Diverging Eyeglass $f < 0$), touchless air gestures, vector stamping via `PhysicsTools.create_thin_lens_diagram`, `LessonBoard` lifecycle (`show_thin_lens`, `toggle_thin_lens`), View menu action (`Ctrl+Alt+O`), navigation toolbar button `🔬 Thấu kính mỏng`, in-air menu trigger (`thin_lens`), and comprehensive unit test suite `tests/test_interactive_thin_lens.py` (8 tests, 100% passing).
'''

    target_milestones = '84. **Task I.70 — Interactive Physics Electricity & Alternating Current (AC) Series RLC Circuit & Phasor Diagram Simulator**'
    assert target_milestones in content, "target_milestones not found"
    idx = content.find(target_milestones)
    end_line_idx = content.find('\n\n---', idx)
    content = content[:end_line_idx] + '\n' + milestone_85 + content[end_line_idx:]

    # 3. Update Backlog
    backlog_text = '''## 4. PHASE I ACTIVE BACKLOG (15h – 16h)

1. **Continuous 16-Hour Execution & Long-Horizon Stability Monitoring**:
   - Maintain standing cron triggers until minimum 16 real elapsed hours are achieved (`2026-10-24T12:48:00+07:00`).
   - Monitor test regressions and system performance across all 58 suites.
2. **Task I.72 — Interactive Modern Physics Photoelectric Effect & Einstein's Quantum Photon Work Function Simulator (`InteractivePhotoelectricEffectItem`)**:
   - Authentic Einstein photoelectric equation $E_k = hf - A = h(c/\\lambda) - A = e \\cdot U_h$.
   - Stopping potential $U_h = (hf - A)/e$ and threshold frequency $f_0 = A/h$ (red threshold wavelength $\\lambda_0 = hc/A$).
   - Authentic cathode metal targets (Cesium $A = 2.14\\text{ eV}$, Potassium $A = 2.30\\text{ eV}$, Sodium $A = 2.46\\text{ eV}$, Zinc $A = 4.31\\text{ eV}$, Platinum $A = 6.35\\text{ eV}$).
   - Dynamic evacuated quartz phototube with illuminated cathode, emitted photoelectrons with kinetic velocity $v = \\sqrt{2E_k/m_e}$, variable voltage bias $U_{AK} \\in [-5.0\\text{ V}, +5.0\\text{ V}]$, and microammeter current readout $I(U_{AK})$.
   - Current-voltage characteristic curve $I - U_{AK}$ and stopping voltage vs frequency $U_h(f)$ straight line ($h/e$ slope) determining Planck's constant $h = 6.626 \\times 10^{-34}\\text{ J}\\cdot\\text{s}$.
   - Vector stamping via `PhysicsTools.create_photoelectric_effect_diagram`, LessonBoard lifecycle, and main window actions.
'''
    backlog_start = content.find('## 4. PHASE I ACTIVE BACKLOG (15h – 16h)')
    backlog_end = content.find('## 5. VERIFICATION EVIDENCE LOG')
    assert backlog_start != -1 and backlog_end != -1, "backlog boundaries not found"
    content = content[:backlog_start] + backlog_text + '\n---\n\n' + content[backlog_end:]

    # 4. Update Verification evidence log
    content = content.replace(
        "- Current Checkpoint: **2026-10-24T05:41:00+07:00** (Verified Elapsed: **8h 53m 00s**)",
        "- Current Checkpoint: **2026-10-24T05:46:00+07:00** (Verified Elapsed: **8h 58m 00s**)"
    )
    content = content.replace(
        "- Master test suite: **721 passed across 57 suites (100% PASSED)**",
        "- Master test suite: **729 passed across 58 suites (100% PASSED)**"
    )

    with open('.agent/16H_RUN_STATE.md', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated .agent/16H_RUN_STATE.md to checkpoint 8h 58m!")

if __name__ == '__main__':
    main()
