import sys

with open('.agent/16H_RUN_STATE.md', 'r', encoding='utf-8') as f:
    text = f.read()

# Update Checkpoint & Elapsed
text = text.replace(
    '**Current Checkpoint Timestamp:** `2026-10-24T10:23:00+07:00`',
    '**Current Checkpoint Timestamp:** `2026-10-24T10:45:00+07:00`'
)
text = text.replace(
    '**Verified Elapsed Time:** `13h 35m 00s`',
    '**Verified Elapsed Time:** `13h 57m 00s`'
)

new_milestones = '''
185. **Task I.115 – Interactive Computer Science & Automata Theory: Aho-Corasick Multi-Pattern Exact String Matching Automaton (`InteractiveAhoCorasickItem`)**: `src/ui/interactive_aho_corasick.py`, authentic Bell Laboratories (1975) multi-pattern exact string matching algorithm, Goto Trie tree construction, BFS failure function f(u) resolution, dictionary suffix output propagation, linear scan complexity O(n + m + z), 5 curriculum presets (`CLASSIC_BELL_LABS_HE_SHE_HIS_HERS`, `DNA_BIOINFORMATICS_RESTRICTION_SITES`, `NATURAL_LANGUAGE_KEYWORD_FILTER`, `OVERLAPPING_PREFIX_SUFFIX_STRESS`, `CYBERSECURITY_SNORT_IDS_SIGNATURES`), touchless air gestures, vector stamping via `InformaticsTools.create_aho_corasick_diagram`, LessonBoard lifecycle, View menu (`Ctrl+Alt+A`), toolbar button `🔤 Otomat Aho-Corasick`, and unit test suite `tests/test_interactive_aho_corasick.py` (12 tests, 100% passing).

186. **Task I.116 – Interactive Physical Chemistry & Thermodynamics: Phase Diagrams, Triple Point, Critical Point, Supercritical Fluid & Clausius-Clapeyron Vapor Pressure Simulator (`InteractivePhaseDiagramItem`)**: `src/ui/interactive_phase_diagram.py`, authentic thermodynamics, Clausius-Clapeyron vapor pressure ln(P/P_ref) = -(ΔH_vap/R)(1/T - 1/T_ref), sublimation curve, anomalous negative melting slope of Water Ice Ih (dP/dT < 0, regelation ice skating), CO2 dry ice sublimation at 1 atm (P_triple = 5.11 atm > 1 atm), supercritical fluid phase transitions, 24-molecule microscopic particle lattice and fluid dynamics, 5 curriculum presets (`WATER_H2O_ANOMALOUS_MELTING`, `CARBON_DIOXIDE_DRY_ICE_SUBLIMATION`, `ARGON_NOBLE_GAS_CLASSICAL`, `SULFUR_ALLOTROPIC_POLYMORPHISM`, `SUPERCRITICAL_EXTRACTION_INDUSTRIAL`), automated isobaric heating and isothermal compression trajectories, vector stamping via `ChemistryTools.create_phase_diagram_diagram`, LessonBoard lifecycle, View menu (`Ctrl+Alt+P`), toolbar button `🌡️ Giản đồ Pha P-T`, and unit test suite `tests/test_interactive_phase_diagram.py` (12 tests, 100% passing).

187. **Task I.117 – Interactive Physics & Electrodynamics: Maxwell's Transverse Electromagnetic Waves, Poynting Vector & Radiation Pressure Simulator (`InteractiveElectromagneticWaveItem`)**: `src/ui/interactive_em_wave.py`, authentic Maxwell equations in vacuum, transverse wave propagation (E ⊥ B ⊥ k), Poynting energy flux vector S = (E x B) / μ₀, radiation pressure P_rad = <S>/c, Stokes polarization parameters (S0, S1, S2, S3) with invariant S0² = S1² + S2² + S3², Linear 45°, Circular RCP/LCP, Elliptical, Quarter-Wave Plate (λ/4) birefringent retarder, 5 curriculum presets (`LINEAR_POLARIZATION_45_DEG`, `CIRCULAR_RIGHT_HANDED_RCP`, `CIRCULAR_LEFT_HANDED_LCP`, `ELLIPTICAL_GENERAL_POLARIZATION`, `QUARTER_WAVE_RETARDER_CRYSTAL`), 3D isometric vector field visualizer, transverse polarization screen, vector stamping via `PhysicsTools.create_electromagnetic_wave_diagram`, LessonBoard lifecycle, View menu (`Ctrl+Alt+E`), toolbar button `📡 Sóng Điện từ`, and unit test suite `tests/test_interactive_em_wave.py` (11 tests, 100% passing).

188. **Task I.118 – Interactive Biology & Population Genetics: Hardy-Weinberg Equilibrium, Genetic Drift, Natural Selection & Bottleneck Simulator (`InteractiveHardyWeinbergItem`)**: `src/ui/interactive_hardy_weinberg.py`, authentic population genetics, p + q = 1 and p² + 2pq + q² = 1 parabolas, Wright-Fisher stochastic binomial genetic drift, natural selection with differential relative fitness (w_AA, w_Aa, w_aa), sickle-cell anemia overdominance balanced polymorphism (HbA/HbS), population bottlenecks (N=25), inbreeding coefficient F, Chi-square goodness-of-fit test (df=1, α=0.05, critical value 3.841), 100-organism visual grid, 5 curriculum presets (`CLASSIC_IDEAL_EQUILIBRIUM`, `SICKLE_CELL_OVERDOMINANCE`, `RECESSIVE_LETHAL_SELECTION`, `GENETIC_DRIFT_BOTTLENECK`, `INBREEDING_CONSANGUINITY_F`), vector stamping via `BiologyTools.create_hardy_weinberg_diagram`, LessonBoard lifecycle, View menu (`Ctrl+Alt+H`), toolbar button `🧬 Hardy-Weinberg`, and unit test suite `tests/test_interactive_hardy_weinberg.py` (12 tests, 100% passing).
'''

pos = text.find('## 4. PHASE I ACTIVE BACKLOG')
if pos == -1:
    print('Failed to find PHASE I ACTIVE BACKLOG')
    sys.exit(1)

new_text = text[:pos] + new_milestones + '\n' + text[pos:]
with open('.agent/16H_RUN_STATE.md', 'w', encoding='utf-8') as f:
    f.write(new_text)

print('Successfully updated 16H_RUN_STATE.md!')
