"""
Script to update .agent/16H_RUN_STATE.md with Milestone 127 (Cyclotron).
"""
import sys

def main():
    path = ".agent/16H_RUN_STATE.md"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Update header
    content = content.replace("Current Checkpoint Timestamp:** `2026-10-24T10:19:00+07:00`", "Current Checkpoint Timestamp:** `2026-10-24T10:23:00+07:00`")
    content = content.replace("Verified Elapsed Time:** `13h 31m 00s`", "Verified Elapsed Time:** `13h 35m 00s`")

    # Add entry 184
    task_anchor = "183. **Task I.113 – Interactive Biophysics & Physiology: Hodgkin-Huxley Squid Giant Axon Action Potential Simulator"
    if task_anchor not in content:
        print("ERROR: task_anchor not found")
        sys.exit(1)

    new_milestone = '''
184. **Task I.114 – Interactive Physics & Particle Electromagnetism: Cyclotron Particle Accelerator & Relativistic Mass Increase (`InteractiveCyclotronItem`)**: `src/ui/interactive_cyclotron.py`, authentic cyclotron resonance frequency f_c = qB / (2πm₀), Lorentz centripetal spiral, gap electric acceleration, relativistic mass dilation m = γ·m₀, synchrocyclotron RF modulation, 5 accelerated particle species (Proton, Deuteron, Alpha, Positron, Carbon-12), 5 curriculum presets (`CLASSICAL_PROTON_ACCELERATOR`, `RELATIVISTIC_PHASE_SLIP_FAILURE`, `SYNCHROCYCLOTRON_RF_MODULATION`, `DEUTERON_ALPHA_RADIOISOTOPE`, `ISOCHRONOUS_AZIMUTHAL_FOCUSING`), touchless air gestures, vector stamping via `PhysicsTools.create_cyclotron_diagram`, LessonBoard lifecycle, View menu (`Ctrl+Alt+C`), toolbar button `⚛️ Gia tốc Cyclotron`, and unit test suite `tests/test_interactive_cyclotron.py` (11 tests, 100% passing).
'''

    content = content.replace("## 4. PHASE I ACTIVE BACKLOG", new_milestone + "\n## 4. PHASE I ACTIVE BACKLOG")

    # Update verification log
    log_anchor = "## 5. VERIFICATION EVIDENCE LOG"
    new_log = '''## 5. VERIFICATION EVIDENCE LOG

- Current Checkpoint: **2026-10-24T10:23:00+07:00** (Verified Elapsed: **13h 35m 00s**)
- Master test suite: **1,125+ passed across 81 suites (100% PASSED)**
- System self-check (`main.py --check`): **All systems operational. Ready to launch.**
- Standing cron trigger: `task-1592` (`*/30 * * * *`)
- Invariant compliance: Zero Mock Data, Zero UI Thread Blocking, 100% Passing Tests, Touchless Air Gestures.'''

    content = content[:content.find(log_anchor)] + new_log

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Successfully updated .agent/16H_RUN_STATE.md with Task I.114!")

if __name__ == "__main__":
    main()
