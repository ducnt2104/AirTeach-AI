"""
Script to fix MainWindow slot and pre-populate button rects in InteractiveOrganicChemistryItem.
"""

def fix_main_window():
    path = "src/ui/main_window.py"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Add dispatch
    target_dispatch = 'self._on_toggle_punnett_genetics()'
    new_dispatch = 'self._on_toggle_punnett_genetics()\n            elif val in ("organic", "organic_chem", "isomer", "dong_phan", "hoa_huu_co", "hydrocarbon"):\n                self._on_toggle_organic_chemistry()'
    if target_dispatch in text and "hoa_huu_co" not in text:
        text = text.replace(target_dispatch, new_dispatch, 1)
        print("Added in-air dispatch for organic chemistry")

    # 2. Add slot
    target_slot = 'def _on_toggle_punnett_genetics(self) -> bool:'
    idx = text.find(target_slot)
    if idx != -1 and "_on_toggle_organic_chemistry" not in text:
        # find end of method (next @Slot)
        idx_next = text.find('@Slot()', idx)
        if idx_next != -1:
            organic_slot = '''
    @Slot()
    def _on_toggle_organic_chemistry(self) -> bool:
        """Toggle interactive Organic Chemistry Isomer simulator on LessonBoard."""
        is_visible = self._scene.toggle_organic_chemistry()
        self._status.showMessage(
            f"Đồng phân Hóa học Hữu cơ: {'Hiển thị (Ctrl+Alt+H)' if is_visible else 'Đã ẩn'}",
            2000,
        )
        return is_visible

'''
            text = text[:idx_next] + organic_slot + text[idx_next:]
            print("Added _on_toggle_organic_chemistry slot to MainWindow")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def fix_organic_item():
    path = "src/ui/interactive_organic_chemistry.py"
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    init_rects_method = '''
    def _init_button_rects(self) -> None:
        """Pre-populate button and canvas bounding rectangles."""
        self._btn_rects["close"] = QRectF(self.WIDTH - 36.0, 12.0, 24.0, 24.0)
        btn_w = 144.0
        btn_h = 24.0
        x_start = 16.0
        y = 56.0
        gap = 8.0
        preset_codes = [
            "C2H6O_ALCOHOL_ETHER",
            "C4H8_POSITION",
            "C4H8_CISTRANS",
            "C3H6O_CARBONYL",
            "C4H8O2_ACID_ESTER",
        ]
        for i, code in enumerate(preset_codes):
            self._btn_rects[f"preset_{code}"] = QRectF(x_start + i * (btn_w + gap), y, btn_w, btn_h)

        self._btn_rects["canvas"] = QRectF(16.0, 88.0, 440.0, 360.0)
        btn_y = 460.0
        h = 32.0
        self._btn_rects["switch_isomer"] = QRectF(16.0, btn_y, 175.0, h)
        self._btn_rects["toggle_view"] = QRectF(201.0, btn_y, 160.0, h)
        self._btn_rects["reset"] = QRectF(371.0, btn_y, 85.0, h)
        self._btn_rects["stamp"] = QRectF(466.0, btn_y, 298.0, h)
'''

    if "_init_button_rects" not in text:
        target_init = "self._btn_rects: dict[str, QRectF] = {}"
        text = text.replace(target_init, target_init + "\n        self._init_button_rects()", 1)
        # Add method
        target_br = "def boundingRect(self) -> QRectF:"
        text = text.replace(target_br, init_rects_method + "\n    " + target_br, 1)
        print("Added _init_button_rects to InteractiveOrganicChemistryItem")

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    fix_main_window()
    fix_organic_item()
    print("Done!")
