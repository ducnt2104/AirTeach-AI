"""
Script to update .agent/16H_RUN_STATE.md with Milestones 88-94.
"""
import re

path = ".agent/16H_RUN_STATE.md"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Update Checkpoint Timestamp & Verified Elapsed
content = re.sub(
    r"\*\*Current Checkpoint Timestamp:\*\* `[^`]+`",
    "**Current Checkpoint Timestamp:** `2026-10-24T07:26:00+07:00`",
    content,
)
content = re.sub(
    r"\*\*Verified Elapsed Time:\*\* `[^`]+`",
    "**Verified Elapsed Time:** `10h 38m 00s`",
    content,
)

new_milestones = """
148. **Task I.74 — Interactive Quantum Physics Bohr Hydrogen Atom & Spectral Lines Simulator**: `src/ui/interactive_bohr_atom.py` (`InteractiveBohrAtomItem`), authentic Bohr quantum postulates, stationary energy levels $E_n = -13.6/n^2$ eV, photon emission energy $\\varepsilon = |E_m - E_n| = hc/\\lambda$, Rydberg formula for Lyman ($n=1$), Balmer ($n=2$, visible 4-line series $H_\\alpha, H_\\beta, H_\\gamma, H_\\delta$), Paschen ($n=3$), and Brackett ($n=4$), concentric quantum orbits $r_n = n^2 r_0$ ($r_0 = 0.53$ Å), orbiting electron animation, emitted photon wave packet vector animation with true spectral color, vector stamping via `PhysicsTools.create_bohr_atom_diagram`, LessonBoard lifecycle (`show_bohr_atom`, `toggle_bohr_atom`), View menu action (`Ctrl+Alt+B`), navigation toolbar button `⚛️ Nguyên tử Bohr`, and comprehensive unit tests (100% passing).
149. **Task I.75 — Interactive Computer Science Formal Languages & Automata (DFA / NFA) State Machine Simulator**: `src/ui/interactive_automata.py` (`InteractiveAutomataItem`), formal 5-tuple machine $M = \\langle Q, \\Sigma, \\delta, q_0, F \\rangle$, authentic discrete transition function $\\delta(q, c)$, input tape string reader, animated head cursor, state transition diagram with circular nodes, double-ring accepting states, curved transition arrows with symbol badges, 5 curriculum presets (Binary Even Parity, Substring Match, Modulo-3 Divisibility, Float Number Regex, NFA with $\\varepsilon$-transitions), step-by-step and auto-run animation, vector stamping via `InformaticsTools.create_automata_diagram`, LessonBoard lifecycle (`show_automata`, `toggle_automata`), View menu action (`Ctrl+Alt+A`), navigation toolbar button `🤖 Máy trạng thái`, and comprehensive unit tests (100% passing).
150. **Task I.76 — Interactive Informatics Sorting Algorithms & Computational Complexity Simulator**: `src/ui/interactive_sorting.py` (`InteractiveSortingItem`), authentic discrete step-by-step sorting generator engine for 5 algorithms (Bubble Sort $O(n^2)$, Insertion Sort $O(n^2)$, Selection Sort $O(n^2)$, Quick Sort $O(n \\log n)$, Merge Sort $O(n \\log n)$), real-time comparison and swap counters, bar chart visualization with color-coded active pointers ($i, j, \\text{pivot}$), step-by-step and continuous animation, vector stamping via `InformaticsTools.create_sorting_diagram`, LessonBoard lifecycle (`show_sorting`, `toggle_sorting`), View menu action (`Ctrl+Alt+S`), navigation toolbar button `📊 Thuật toán sắp xếp`, and comprehensive unit tests (100% passing).
151. **Task I.77 — Interactive Biology Genetics Mendel & Punnett Square Simulator**: `src/ui/interactive_genetics_punnett.py` (`InteractiveGeneticsPunnettItem`), authentic Mendelian genetics engine with Monohybrid ($Aa \\times Aa$, 3:1), Dihybrid ($AaBb \\times AaBb$, 9:3:3:1), ABO Blood Group, and X-linked Colorblindness presets, Chi-Square goodness-of-fit test ($\\chi^2 = \\sum (O - E)^2 / E$), 2x2 and 4x4 interactive Punnett grids, genotype badges and phenotype descriptions, vector stamping via `BiologyTools.create_punnett_genetics_diagram`, LessonBoard lifecycle (`show_punnett_genetics`, `toggle_punnett_genetics`), View menu action (`Ctrl+Alt+G`), navigation toolbar button `🧬 Di truyền Mendel`, and comprehensive unit tests (100% passing).
152. **Task I.78 — Interactive Organic Chemistry Hydrocarbons & Isomer Simulator**: `src/ui/interactive_organic_chemistry.py` (`InteractiveOrganicChemistryItem`), authentic IUPAC chemical physics engine with degree of unsaturation $k = (2C + 2 - H + N - X) / 2$, valence rules validation, 5 curriculum presets (Ethanol vs Dimethyl Ether $\\Delta T_s = 102.37^\\circ\\text{C}$, But-1-ene vs But-2-ene, cis vs trans-But-2-ene, Propanal vs Acetone, Butanoic Acid vs Ethyl Acetate), Lewis structural and skeletal formula views, draggable atom nodes, vector stamping via `ChemistryTools.create_organic_isomer_diagram`, LessonBoard lifecycle (`show_organic_chemistry`, `toggle_organic_chemistry`), View menu action (`Ctrl+Alt+H`), navigation toolbar button `🧪 Đồng phân Hóa học`, and comprehensive unit test suite `tests/test_interactive_organic_chemistry.py` (14 tests, 100% passing).
153. **Task I.79 — Interactive Physics Fluid Dynamics Bernoulli & Venturi Tube Airflow Simulator**: `src/ui/interactive_fluid_venturi.py` (`InteractiveFluidVenturiItem`), authentic continuity equation $A_1 v_1 = A_2 v_2 = Q$, horizontal Bernoulli static pressure drop $\\Delta P = \\frac{1}{2}\\rho (v_2^2 - v_1^2)$, U-tube differential manometer liquid height difference $\\Delta h = \\frac{\\Delta P}{\\rho_m g}$, Lagrangian flow tracer particles with continuous bell-curve acceleration through throat, velocity vectors, 5 curriculum presets (Venturi Flowmeter, Carburetor Choke, Airplane Wing Lift, Perfume Aspirator, Water Pipe with Mercury), draggable throat constriction handle, vector stamping via `PhysicsTools.create_fluid_venturi_diagram`, LessonBoard lifecycle (`show_fluid_venturi`, `toggle_fluid_venturi`), View menu action (`Ctrl+Alt+F`), navigation toolbar button `🌊 Cơ học chất lưu`, and comprehensive unit test suite `tests/test_interactive_fluid_venturi.py` (17 tests, 100% passing).
154. **Task I.80 — Interactive Informatics Binary Tree & Huffman Encoding Simulator**: `src/ui/interactive_binary_tree.py` (`InteractiveBinaryTreeItem`), authentic Binary Search Tree (BST) with $O(\\log n)$ balanced vs $O(n)$ degenerate worst-case layout, 4 tree traversals (In-order sorted keys, Pre-order prefix, Post-order postfix/RPN, Level-order BFS), BST search with active search path highlighting and comparison counter, Huffman optimal prefix coding tree via priority queue min-heap, Shannon information entropy $H(X)$, average codeword length $\\bar{L}$, Huffman efficiency $\\eta$, compression ratio vs 8-bit ASCII, Arithmetic expression tree $((A+B) \\times (C-D) / E)$, touchless air gestures, vector stamping via `InformaticsTools.create_binary_tree_diagram`, LessonBoard lifecycle (`show_binary_tree`, `toggle_binary_tree`), View menu action (`Ctrl+Alt+U`), navigation toolbar button `🌳 Cây nhị phân`, and comprehensive unit test suite `tests/test_interactive_binary_tree.py` (16 tests, 100% passing).
"""

target_sec = "## 4. PHASE I ACTIVE BACKLOG (15h – 16h)"
if target_sec in content and "Task I.80" not in content:
    content = content.replace(target_sec, new_milestones + "\n---\n\n" + target_sec, 1)

# Update Verification Evidence Log
content = re.sub(
    r"- Current Checkpoint: \*\*([^\*]+)\*\* \(Verified Elapsed: \*\*([^\*]+)\*\*\)",
    "- Current Checkpoint: **2026-10-24T07:26:00+07:00** (Verified Elapsed: **10h 38m 00s**)",
    content,
)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully updated .agent/16H_RUN_STATE.md!")
