path = "src/ui/stem_tools.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

target = """    @staticmethod
    def create_geometric_triangle(
        v0: tuple[float, float],
        v1: tuple[float, float],
        v2: tuple[float, float],
        labels: tuple[str, str, str] = ("A", "B", "C"),
        color: QColor = QColor("#1a73e8"),
        fill_color: QColor = QColor("#e8f0fe"),
        show_right_angle: bool = False,
        right_angle_vertex_idx: int = 0,
    ) -> list[QGraphicsItem]:"""

replacement = """    @staticmethod
    def create_geometric_triangle(
        v0: tuple[float, float],
        v1: tuple[float, float],
        v2: tuple[float, float],
        labels: tuple[str, str, str] = ("A", "B", "C"),
        color: Optional[QColor] = None,
        fill_color: QColor = QColor("#e8f0fe"),
        show_right_angle: bool = False,
        right_angle_vertex_idx: int = 0,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        color = pen_color or color or QColor("#1a73e8")"""

if target in text:
    text = text.replace(target, replacement, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Updated create_geometric_triangle signature!")
else:
    print("Target not found!")
