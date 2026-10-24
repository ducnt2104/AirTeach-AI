path = "src/ui/stem_tools.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Fix create_circle_with_radius
old_circle_sig = """    @staticmethod
    def create_circle_with_radius(
        center: tuple[float, float],
        radius: float = 80.0,
        label: str = "R",
        color: QColor = QColor("#1a73e8"),
        fill_color: QColor = QColor("#e8f0fe"),
    ) -> list[QGraphicsItem]:"""

new_circle_sig = """    @staticmethod
    def create_circle_with_radius(
        center: tuple[float, float],
        radius: float = 80.0,
        label: str = "R",
        color: Optional[QColor] = None,
        fill_color: QColor = QColor("#e8f0fe"),
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        color = pen_color or color or QColor("#1a73e8")"""

if old_circle_sig in text:
    text = text.replace(old_circle_sig, new_circle_sig, 1)
    print("Updated create_circle_with_radius signature!")

# Fix create_angle_arc
old_angle_sig = """    @staticmethod
    def create_angle_arc(
        vertex: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        radius: float = 40.0,
        label: str = "",
        color: QColor = QColor("#22c55e"),
    ) -> list[QGraphicsItem]:"""

new_angle_sig = """    @staticmethod
    def create_angle_arc(
        vertex: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        radius: float = 40.0,
        label: str = "",
        color: Optional[QColor] = None,
        pen_color: Optional[QColor] = None,
    ) -> list[QGraphicsItem]:
        color = pen_color or color or QColor("#22c55e")"""

if old_angle_sig in text:
    text = text.replace(old_angle_sig, new_angle_sig, 1)
    print("Updated create_angle_arc signature!")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
