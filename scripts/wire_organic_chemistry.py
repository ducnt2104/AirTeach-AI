"""
Script to cleanly wire Organic Chemistry Isomers into:
  1. ChemistryTools in src/ui/stem_tools.py (create_organic_isomer_diagram)
  2. LessonBoard in src/ui/lesson_board.py (lifecycle & exclusion)
  3. MainWindow in src/ui/main_window.py (view menu, toolbar, in-air dispatch, slot)
"""
import sys

def wire_stem_tools():
    path = "src/ui/stem_tools.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "create_organic_isomer_diagram" in content:
        print("create_organic_isomer_diagram already in stem_tools.py")
        return

    target = """        # Inner aromatic delocalized electron circle
        r_inner = radius * 0.58
        inner_circle = QGraphicsEllipseItem(cx - r_inner, cy - r_inner, 2 * r_inner, 2 * r_inner)
        inner_circle.setPen(pen)
        inner_circle.setBrush(Qt.BrushStyle.NoBrush)
        inner_circle.setZValue(11.0)
        items.append(inner_circle)

        return items"""

    method_code = '''

    @staticmethod
    def create_organic_isomer_diagram(
        center: tuple[float, float],
        molecule: Optional[Any] = None,
        preset: Optional[Any] = None,
        is_skeletal: bool = False,
    ) -> list[QGraphicsItem]:
        """Creates an authentic 2D vector structural formula and properties diagram for organic chemistry."""
        items: list[QGraphicsItem] = []
        cx, cy = center
        w, h = 480.0, 320.0
        x0 = cx - w / 2.0
        y0 = cy - h / 2.0

        # 1. Background Card
        bg = QGraphicsRectItem(x0, y0, w, h)
        bg.setBrush(QBrush(QColor(15, 23, 42, 235)))
        bg.setPen(QPen(QColor(56, 189, 248, 160), 1.5))
        bg.setZValue(10.0)
        items.append(bg)

        # Fallback if molecule not passed
        if molecule is None:
            from src.ui.interactive_organic_chemistry import PRESETS
            pr = preset or PRESETS["C2H6O_ALCOHOL_ETHER"]
            mol = pr.molecule_a
        else:
            mol = molecule

        # 2. Header text
        title_item = QGraphicsTextItem()
        title_item.setPlainText(f"🧪 {mol.name_vi} ({mol.iupac_name})")
        title_item.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title_item.setDefaultTextColor(QColor(248, 250, 252))
        title_item.setPos(x0 + 16.0, y0 + 12.0)
        title_item.setZValue(12.0)
        items.append(title_item)

        sub_item = QGraphicsTextItem()
        sub_item.setPlainText(
            f"CTPT: {mol.formula} | M = {mol.molecular_mass:.2f} g/mol | Độ bất bão hòa k = {mol.degree_of_unsaturation:.0f}"
        )
        sub_item.setFont(QFont("Segoe UI", 8))
        sub_item.setDefaultTextColor(QColor(148, 163, 184))
        sub_item.setPos(x0 + 16.0, y0 + 36.0)
        sub_item.setZValue(12.0)
        items.append(sub_item)

        # 3. Canvas Offset Calculation
        atom_xs = [a.x for a in mol.atoms]
        atom_ys = [a.y for a in mol.atoms]
        if atom_xs and atom_ys:
            min_x, max_x = min(atom_xs), max(atom_xs)
            min_y, max_y = min(atom_ys), max(atom_ys)
            mol_cx = (min_x + max_x) / 2.0
            mol_cy = (min_y + max_y) / 2.0
            target_cx = x0 + w * 0.45
            target_cy = y0 + 140.0
            dx = target_cx - mol_cx
            dy = target_cy - mol_cy
        else:
            dx, dy = 0.0, 0.0

        cpk_stamps = {
            "C": (QColor(51, 65, 85), QColor(248, 250, 252)),
            "H": (QColor(226, 232, 240), QColor(15, 23, 42)),
            "O": (QColor(239, 68, 68), QColor(255, 255, 255)),
            "N": (QColor(59, 130, 246), QColor(255, 255, 255)),
            "Cl": (QColor(16, 185, 129), QColor(255, 255, 255)),
        }

        # 4. Bonds
        for bond in mol.bonds:
            a1 = mol.get_atom(bond.atom1_id)
            a2 = mol.get_atom(bond.atom2_id)
            if not a1 or not a2:
                continue

            if is_skeletal and (a1.symbol == "H" or a2.symbol == "H"):
                continue

            p1 = (a1.x + dx, a1.y + dy)
            p2 = (a2.x + dx, a2.y + dy)
            b_color = QColor(56, 189, 248) if is_skeletal else QColor(148, 163, 184)
            pen = QPen(b_color, 2.5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)

            if bond.order == 2:
                norm_dx = p2[0] - p1[0]
                norm_dy = p2[1] - p1[1]
                dist = math.hypot(norm_dx, norm_dy) or 1.0
                nx = -norm_dy / dist * 3.0
                ny = norm_dx / dist * 3.0
                l1 = QGraphicsLineItem(p1[0] + nx, p1[1] + ny, p2[0] + nx, p2[1] + ny)
                l2 = QGraphicsLineItem(p1[0] - nx, p1[1] - ny, p2[0] - nx, p2[1] - ny)
                l1.setPen(pen)
                l2.setPen(pen)
                l1.setZValue(11.0)
                l2.setZValue(11.0)
                items.extend([l1, l2])
            else:
                line = QGraphicsLineItem(p1[0], p1[1], p2[0], p2[1])
                line.setPen(pen)
                line.setZValue(11.0)
                items.append(line)

        # 5. Atoms
        for atom in mol.atoms:
            if is_skeletal and (atom.symbol == "C" or atom.symbol == "H"):
                continue

            ax = atom.x + dx
            ay = atom.y + dy
            r = 12.0 if atom.symbol != "H" else 9.0

            bg_c, txt_c = cpk_stamps.get(atom.symbol, (QColor(100, 116, 139), QColor(255, 255, 255)))
            circ = QGraphicsEllipseItem(ax - r, ay - r, r * 2.0, r * 2.0)
            circ.setBrush(QBrush(bg_c))
            circ.setPen(QPen(QColor(248, 250, 252), 1.2))
            circ.setZValue(12.0)
            items.append(circ)

            txt = QGraphicsTextItem()
            txt.setPlainText(atom.symbol)
            txt.setFont(QFont("Segoe UI", 8 if atom.symbol != "H" else 7, QFont.Weight.Bold))
            txt.setDefaultTextColor(txt_c)
            txt.setPos(ax - r + 3.0, ay - r - 1.0)
            txt.setZValue(13.0)
            items.append(txt)

        # 6. Properties footer banner
        footer = QGraphicsRectItem(x0 + 12.0, y0 + h - 85.0, w - 24.0, 72.0)
        footer.setBrush(QBrush(QColor(30, 41, 59, 220)))
        footer.setPen(QPen(QColor(71, 85, 105), 1.0))
        footer.setZValue(11.0)
        items.append(footer)

        f_txt = QGraphicsTextItem()
        info_lines = [
            f"• Nhiệt độ sôi: {mol.boiling_point_c:+.1f} °C   |   Momen lưỡng cực: {mol.dipole_moment_debye:.2f} D",
            f"• Lực liên kết: {mol.intermolecular_forces}",
            f"• Trạng thái & tính chất: {mol.physical_state_vi[:58]}...",
        ]
        f_txt.setPlainText("\\n".join(info_lines))
        f_txt.setFont(QFont("Segoe UI", 7))
        f_txt.setDefaultTextColor(QColor(226, 232, 240))
        f_txt.setPos(x0 + 18.0, y0 + h - 82.0)
        f_txt.setZValue(12.0)
        items.append(f_txt)

        return items'''

    if target not in content:
        print("ERROR: Target for stem_tools not found!")
        sys.exit(1)

    content = content.replace(target, target + method_code, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("Added create_organic_isomer_diagram to ChemistryTools in stem_tools.py")


def wire_lesson_board():
    path = "src/ui/lesson_board.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Attribute in __init__
    init_target = "        self._punnett_genetics_item: Optional[QGraphicsItem] = None"
    init_add = "\n        self._organic_chemistry_item: Optional[QGraphicsItem] = None"
    if init_target in content and "_organic_chemistry_item" not in content:
        content = content.replace(init_target, init_target + init_add, 1)
        print("Added _organic_chemistry_item to LessonBoard.__init__")

    # 2. Lifecycle methods
    punnett_methods = """    def get_punnett_genetics_item(self) -> Optional[QGraphicsItem]:
        \"\"\"Return interactive Punnett genetics instance if created.\"\"\"
        return self._punnett_genetics_item"""

    organic_methods = '''

    def show_organic_chemistry(
        self,
        visible: bool = True,
        pos: Optional[tuple[float, float]] = None,
    ) -> Optional[QGraphicsItem]:
        """Show or hide the interactive Organic Chemistry Hydrocarbon & Isomer Simulator."""
        if self._organic_chemistry_item is None:
            from src.ui.interactive_organic_chemistry import InteractiveOrganicChemistryItem
            spawn_pos = pos if pos is not None else (310.0, 130.0)
            self._organic_chemistry_item = InteractiveOrganicChemistryItem(pos=spawn_pos)
            self.addItem(self._organic_chemistry_item)

        self._organic_chemistry_item.setVisible(visible)
        if visible and pos is not None:
            self._organic_chemistry_item.setPos(pos[0], pos[1])
        self.update()
        return self._organic_chemistry_item if visible else None

    def toggle_organic_chemistry(self) -> bool:
        """Toggle interactive organic chemistry simulator visibility; returns new state."""
        if self._organic_chemistry_item is None or not self._organic_chemistry_item.isVisible():
            self.show_organic_chemistry(True)
            return True
        else:
            self.show_organic_chemistry(False)
            return False

    def is_organic_chemistry_visible(self) -> bool:
        """Return True if interactive organic chemistry simulator is active on board."""
        return self._organic_chemistry_item is not None and self._organic_chemistry_item.isVisible()

    def get_organic_chemistry(self) -> Optional[QGraphicsItem]:
        """Return interactive organic chemistry instance if active and visible."""
        if self._organic_chemistry_item is not None and self._organic_chemistry_item.isVisible():
            return self._organic_chemistry_item
        return None

    def get_organic_chemistry_item(self) -> Optional[QGraphicsItem]:
        """Return interactive organic chemistry instance if created."""
        return self._organic_chemistry_item'''

    if punnett_methods in content and "def show_organic_chemistry" not in content:
        content = content.replace(punnett_methods, punnett_methods + organic_methods, 1)
        print("Added lifecycle methods to LessonBoard")

    # 3. Exclusion in get_items_in_rect
    excl_target = "            getattr(self, \"_punnett_genetics_item\", None),"
    excl_add = "\n            getattr(self, \"_organic_chemistry_item\", None),"
    if excl_target in content and "_organic_chemistry_item" not in content.split("get_items_in_rect")[1]:
        content = content.replace(excl_target, excl_target + excl_add, 1)
        print("Added _organic_chemistry_item to exclusion set in get_items_in_rect")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def wire_main_window():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. View menu action
    sorting_menu = """        sorting_act = QAction("📊 Thuật toán sắp xếp & Độ phức tạp Big-O (Sorting Algorithms)", self)
        sorting_act.setShortcut(QKeySequence("Ctrl+Alt+S"))
        sorting_act.triggered.connect(self._on_toggle_sorting)
        view_menu.addAction(sorting_act)"""

    organic_menu = """
        punnett_menu_act = QAction("🧬 Di truyền Mendel & Khung Punnett Square (Mendel Genetics)", self)
        punnett_menu_act.setShortcut(QKeySequence("Ctrl+Alt+G"))
        punnett_menu_act.triggered.connect(self._on_toggle_punnett_genetics)
        view_menu.addAction(punnett_menu_act)

        organic_act = QAction("🧪 Đồng phân Hóa học Hữu cơ & Nhóm chức (Organic Isomerism)", self)
        organic_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        organic_act.triggered.connect(self._on_toggle_organic_chemistry)
        view_menu.addAction(organic_act)"""

    if sorting_menu in content and "organic_act" not in content:
        content = content.replace(sorting_menu, sorting_menu + organic_menu, 1)
        print("Added organic_act to View menu")

    # 2. Navigation toolbar action
    punnett_tb = """        punnett_tb_act = QAction("🧬 Di truyền Mendel", self)
        punnett_tb_act.setToolTip("Bật/Tắt mô phỏng Di truyền Mendel & Khung Punnett Square (Ctrl+Alt+G)")
        punnett_tb_act.setShortcut(QKeySequence("Ctrl+Alt+G"))
        punnett_tb_act.triggered.connect(self._on_toggle_punnett_genetics)
        nav_tb.addAction(punnett_tb_act)"""

    organic_tb = """
        organic_tb_act = QAction("🧪 Đồng phân Hóa học", self)
        organic_tb_act.setToolTip("Bật/Tắt mô phỏng Đồng phân Hóa học Hữu cơ & Nhóm chức (Ctrl+Alt+H)")
        organic_tb_act.setShortcut(QKeySequence("Ctrl+Alt+H"))
        organic_tb_act.triggered.connect(self._on_toggle_organic_chemistry)
        nav_tb.addAction(organic_tb_act)"""

    if punnett_tb in content and "organic_tb_act" not in content:
        content = content.replace(punnett_tb, punnett_tb + organic_tb, 1)
        print("Added organic_tb_act to Navigation Toolbar")

    # 3. In-air Tool Dispatch
    punnett_dispatch = """            elif val in ("punnett", "genetics", "mendel", "di_truyen", "khung_punnett", "lai_giong"):
                self._on_toggle_punnett_genetics()"""

    organic_dispatch = """            elif val in ("punnett", "genetics", "mendel", "di_truyen", "khung_punnett", "lai_giong"):
                self._on_toggle_punnett_genetics()
            elif val in ("organic", "organic_chem", "isomer", "dong_phan", "hoa_huu_co", "hydrocarbon"):
                self._on_toggle_organic_chemistry()"""

    if punnett_dispatch in content and "organic_chem" not in content:
        content = content.replace(punnett_dispatch, organic_dispatch, 1)
        print("Added organic dispatch in-air")

    # 4. Slot
    punnett_slot = """    @Slot()
    def _on_toggle_punnett_genetics(self) -> bool:
        \"\"\"Toggle interactive Biology Mendel Genetics & Punnett Square visualizer on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_punnett_genetics()
        self._status.showMessage(
            f"Di truyền Mendel & Khung Punnett: {'Hiển thị (Ctrl+Alt+G)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    organic_slot = """    @Slot()
    def _on_toggle_punnett_genetics(self) -> bool:
        \"\"\"Toggle interactive Biology Mendel Genetics & Punnett Square visualizer on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_punnett_genetics()
        self._status.showMessage(
            f"Di truyền Mendel & Khung Punnett: {'Hiển thị (Ctrl+Alt+G)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

    @Slot()
    def _on_toggle_organic_chemistry(self) -> bool:
        \"\"\"Toggle interactive Organic Chemistry Isomer simulator on LessonBoard.\"\"\"
        is_visible = self._scene.toggle_organic_chemistry()
        self._status.showMessage(
            f"Đồng phân Hóa học Hữu cơ: {'Hiển thị (Ctrl+Alt+H)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible"""

    if punnett_slot in content and "_on_toggle_organic_chemistry" not in content:
        content = content.replace(punnett_slot, organic_slot, 1)
        print("Added _on_toggle_organic_chemistry slot to MainWindow")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    wire_stem_tools()
    wire_lesson_board()
    wire_main_window()
    print("ALL WIRE TASKS COMPLETED!")
