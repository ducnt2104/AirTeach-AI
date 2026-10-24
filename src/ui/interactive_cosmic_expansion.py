"""
AirTeach AI — Interactive Astrophysics & Cosmology: Hubble-Lemaître Law,
Cosmological Redshift & Cosmic Expansion Rate Simulator.

Simulates:
  - Hubble-Lemaître Law of Cosmic Metric Expansion:
      v = H_0 * d
  - Cosmological Redshift z and Relativistic Recessional Velocity:
      z = (λ_obs - λ_emit) / λ_emit = Δλ / λ_0
      1 + z = a(t_obs) / a(t_emit)
      v_rel / c = ((1 + z)² - 1) / ((1 + z)² + 1)
  - Hubble Time and Hubble Distance (Cosmic Horizon):
      t_H = 1 / H_0 ≈ 13.8 - 14.5 Gyr
      d_H = c / H_0 ≈ 4200 - 4400 Mpc (Galaxies beyond d_H recede faster than c)
  - Hydrogen Balmer Absorption Spectral Line Redshifts:
      Hα (656.3 nm), Hβ (486.1 nm), Hγ (434.0 nm), Hδ (410.2 nm)
  - Friedmann-Lemaître-Robertson-Walker (FLRW) Cosmic Scale Factor a(t) expansion.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional

from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QTimer
from PySide6.QtGui import (
    QColor,
    QPainter,
    QPen,
    QBrush,
    QFont,
    QPainterPath,
    QLinearGradient,
    QRadialGradient,
)
from PySide6.QtWidgets import QGraphicsObject, QGraphicsItem


# Standard Physical & Cosmological Constants
SPEED_OF_LIGHT_KM_S: float = 299792.458  # km/s
MPC_TO_KM: float = 3.085677581e19        # 1 Mpc in km
SEC_PER_YEAR: float = 3.15576e7          # Seconds per Julian year

# Laboratory Rest-Frame Hydrogen Balmer Absorption Lines (in nanometers)
BALMER_LINES_NM: dict[str, float] = {
    "Hδ": 410.2,   # Violet
    "Hγ": 434.0,   # Blue-Violet
    "Hβ": 486.1,   # Cyan
    "Hα": 656.3,   # Crimson Red
}


@dataclass(frozen=True)
class CosmicPreset:
    """Represents a curriculum scenario for Cosmic Expansion & Hubble's Law."""
    code: str
    title_vi: str
    badge_label: str
    target_name: str
    hubble_constant_km_s_mpc: float  # H_0 in km/s/Mpc
    default_distance_mpc: float      # Distance in Megaparsecs
    default_redshift_z: float        # Cosmological redshift z
    omega_matter: float              # Matter density parameter Ω_m
    omega_lambda: float              # Dark energy density parameter Ω_Λ
    description: str


COSMIC_PRESETS: dict[str, CosmicPreset] = {
    "MODERN_PLANCK_LAMBDA_CDM": CosmicPreset(
        code="MODERN_PLANCK_LAMBDA_CDM",
        title_vi="Mô Hình Chuẩn ΛCDM — Đo Đạc Vệ Tinh Planck (Bức Xạ Nền CMB)",
        badge_label="PLANCK ΛCDM",
        target_name="Thiên hà Chuẩn Cụm Coma",
        hubble_constant_km_s_mpc=67.4,
        default_distance_mpc=100.0,    # 100 Mpc
        default_redshift_z=0.0225,
        omega_matter=0.315,
        omega_lambda=0.685,
        description="Tham số vũ trụ học chuẩn xác nhất từ vệ tinh Planck ESA đo đạc bức xạ tàn dư Big Bang (CMB). Với H₀ = 67.4 km/s/Mpc, tuổi của vũ trụ là 13.797 tỷ năm.",
    ),
    "SH0ES_SUPERNOVAE_TENSION": CosmicPreset(
        code="SH0ES_SUPERNOVAE_TENSION",
        title_vi="Thang Đo Khoảng Cách Siêu Tân Tinh Ia — Bí Ẩn Khủng Hoảng Hubble (Tension)",
        badge_label="SH0ES TENSION",
        target_name="Siêu Tân Tinh SN Ia trong NGC 5584",
        hubble_constant_km_s_mpc=73.04,
        default_distance_mpc=25.0,     # 25 Mpc
        default_redshift_z=0.0061,
        omega_matter=0.30,
        omega_lambda=0.70,
        description="Đo đạc vũ trụ cục bộ bằng kính Hubble (HST) qua sao biến quang Cepheid và siêu tân tinh Ia cho H₀ = 73.04 km/s/Mpc. Chênh lệch 5σ so với Planck là một trong những bí ẩn lớn nhất của vật lý hiện đại.",
    ),
    "EDWIN_HUBBLE_1929_ORIGINAL": CosmicPreset(
        code="EDWIN_HUBBLE_1929_ORIGINAL",
        title_vi="Khám Phá Lịch Sử 1929 — Edwin Hubble & Vũ Trụ Giãn Nở",
        badge_label="HUBBLE 1929",
        target_name="Thiên hà Tiên Nữ M31 & Cụm Xử Nữ",
        hubble_constant_km_s_mpc=70.0,
        default_distance_mpc=2.0,      # 2 Mpc
        default_redshift_z=0.00047,
        omega_matter=0.30,
        omega_lambda=0.70,
        description="Công bố chấn động năm 1929 chứng minh vũ trụ không tĩnh tại mà đang giãn nở đồng nhất: các thiên hà càng ở xa càng rời xa chúng ta với vận tốc tỷ lệ thuận khoảng cách (v = H₀·d).",
    ),
    "HIGH_REDSHIFT_JWST_GALAXY": CosmicPreset(
        code="HIGH_REDSHIFT_JWST_GALAXY",
        title_vi="Kính Thiên Văn Không Gian James Webb (JWST) — Vũ Trụ Sơ Khai z = 14",
        badge_label="JWST DEEP FIELD",
        target_name="Thiên hà Sơ khai JADES-GS-z14-0",
        hubble_constant_km_s_mpc=70.0,
        default_distance_mpc=4100.0,   # ~4.1 Gpc proper distance
        default_redshift_z=14.32,
        omega_matter=0.315,
        omega_lambda=0.685,
        description="Thiên hà xa nhất từng quan sát được bởi JWST với độ dịch chuyển đỏ kỷ lục z = 14.32. Ánh sáng phát ra chỉ 290 triệu năm sau Vụ Nổ Lớn, vạch phổ phát xạ Hα bị kéo dãn vào dải hồng ngoại trung.",
    ),
    "SUPERLUMINAL_HUBBLE_HORIZON": CosmicPreset(
        code="SUPERLUMINAL_HUBBLE_HORIZON",
        title_vi="Chân Trời Hubble — Vận Tốc Rời Xa Vượt Tốc Độ Ánh Sáng (v > c)",
        badge_label="SUPERLUMINAL HORIZON",
        target_name="Chân trời Vũ trụ Học (Hubble Sphere)",
        hubble_constant_km_s_mpc=70.0,
        default_distance_mpc=4500.0,   # > d_H = c/H_0 ≈ 4286 Mpc
        default_redshift_z=1.50,
        omega_matter=0.30,
        omega_lambda=0.70,
        description="Ở khoảng cách vượt quá bán kính Hubble d_H = c / H₀ ≈ 4286 Mpc, không gian giãn nở làm các thiên hà rời xa chúng ta với vận tốc v > c mà không vi phạm thuyết tương đối hẹp vì vật thể không chuyển động xuyên qua không gian.",
    ),
}


@dataclass
class ExpandingGalaxyNode:
    """Represents a simulated galaxy in the expanding cosmic web."""
    id_num: int
    base_x: float          # In normalized coordinates [-1.0, 1.0]
    base_y: float
    luminosity: float
    color: QColor
    name: str


class InteractiveCosmicExpansionItem(QGraphicsObject):
    """
    Interactive PySide6 Graphics Item simulating the Hubble-Lemaître Law,
    Cosmological Redshift of Hydrogen Balmer lines, and Cosmic Metric Expansion.
    """

    cosmic_computed = Signal(float, float, float, float)  # v_km_s, z, age_gyr, horizon_mpc
    stamp_requested = Signal(dict)

    CARD_W = 740.0
    CARD_H = 650.0

    def __init__(self, pos: tuple[float, float] = (80.0, 40.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation state
        self._active_preset_code: str = "MODERN_PLANCK_LAMBDA_CDM"
        self._preset: CosmicPreset = COSMIC_PRESETS[self._active_preset_code]

        self._hubble_constant: float = self._preset.hubble_constant_km_s_mpc
        self._distance_mpc: float = self._preset.default_distance_mpc
        self._redshift_z: float = self._preset.default_redshift_z

        # Cosmic scale factor animation
        self._scale_factor: float = 1.0
        self._scale_direction: float = 0.003
        self._is_running: bool = True

        # Galaxies in cosmological field
        self._galaxies: list[ExpandingGalaxyNode] = []
        self._init_cosmic_field(num_galaxies=24)

        # UI & Touchless buttons
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # Timer for 60fps real-time physics expansion
        self._timer = QTimer()
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start(16)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    # -------------------------------------------------------------------------
    # Physical & Cosmological Properties (Hubble-Lemaître & Special Relativity)
    # -------------------------------------------------------------------------
    @property
    def recessional_velocity_km_s(self) -> float:
        """
        Hubble-Lemaître law linear recessional velocity:
        v = H_0 * d
        """
        return self._hubble_constant * self._distance_mpc

    @property
    def relativistic_velocity_fraction_c(self) -> float:
        """
        Relativistic recessional velocity fraction v/c from cosmological redshift z:
        v/c = ((1 + z)² - 1) / ((1 + z)² + 1)
        """
        z = max(0.0, self.effective_redshift_z)
        factor = (1.0 + z) ** 2
        return (factor - 1.0) / (factor + 1.0)

    @property
    def effective_redshift_z(self) -> float:
        """
        Cosmological redshift z computed from Hubble velocity or current preset:
        For small distances, z ≈ v / c. For high redshift, z is explicitly tracked.
        """
        v = self.recessional_velocity_km_s
        c = SPEED_OF_LIGHT_KM_S
        if v / c < 0.2:
            return v / c
        else:
            # Use preset default or calculate relativistic z
            beta = min(0.999, v / (c * 2.5))
            return math.sqrt((1.0 + beta) / (1.0 - beta)) - 1.0

    @property
    def hubble_time_Gyr(self) -> float:
        """
        Hubble time t_H = 1 / H_0 in Gigayears.
        Approximates the age of the expanding universe.
        """
        h0 = max(10.0, self._hubble_constant)
        # H0 in s^-1: (H0 * 1000) / (MPC_TO_KM * 1000)
        h0_si = (h0) / MPC_TO_KM
        seconds_age = 1.0 / h0_si
        return seconds_age / (SEC_PER_YEAR * 1e9)

    @property
    def hubble_horizon_distance_Mpc(self) -> float:
        """
        Hubble sphere distance d_H = c / H_0 in Megaparsecs.
        Galaxies beyond d_H recede faster than c due to expansion of space.
        """
        h0 = max(10.0, self._hubble_constant)
        return SPEED_OF_LIGHT_KM_S / h0

    @property
    def hubble_horizon_distance_Gly(self) -> float:
        """Hubble horizon in Billion Light Years (Gly)."""
        # 1 Mpc ≈ 3.26156 million light years
        return (self.hubble_horizon_distance_Mpc * 3.26156) / 1000.0

    def get_observed_wavelength_nm(self, rest_wavelength_nm: float) -> float:
        """
        Calculates observed redshifted wavelength:
        λ_obs = λ_emit * (1 + z)
        """
        z = self.effective_redshift_z
        return rest_wavelength_nm * (1.0 + z)

    # -------------------------------------------------------------------------
    # Cosmic Field Initialization & Animation
    # -------------------------------------------------------------------------
    def _init_cosmic_field(self, num_galaxies: int = 24) -> None:
        """Distributes galaxies in isotropic cosmic field."""
        self._galaxies.clear()
        colors = [
            QColor("#38bdf8"),
            QColor("#818cf8"),
            QColor("#c084fc"),
            QColor("#f472b6"),
            QColor("#fbbf24"),
        ]
        names = ["NGC 224", "M87", "NGC 1365", "Centaurus A", "M104", "NGC 4486", "M33", "NGC 5128"]
        for i in range(num_galaxies):
            angle = random.uniform(0.0, 2.0 * math.pi)
            dist = math.sqrt(random.uniform(0.04, 0.90))
            bx = dist * math.cos(angle)
            by = dist * math.sin(angle)
            lum = random.uniform(3.0, 6.0)
            col = random.choice(colors)
            nm = random.choice(names) + f"-{i+1}"
            self._galaxies.append(ExpandingGalaxyNode(i, bx, by, lum, col, nm))

    def _on_sim_tick(self) -> None:
        """Animation tick for expanding metric scale factor a(t)."""
        if not self._is_running:
            return

        self._scale_factor += self._scale_direction
        if self._scale_factor > 1.35:
            self._scale_direction = -0.003
        elif self._scale_factor < 0.75:
            self._scale_direction = 0.003

        self.update()

    # -------------------------------------------------------------------------
    # Simulation Control & Presets
    # -------------------------------------------------------------------------
    def set_preset(self, code: str) -> None:
        """Switches to a curriculum scenario preset."""
        if code not in COSMIC_PRESETS:
            return
        self._active_preset_code = code
        self._preset = COSMIC_PRESETS[code]
        self._hubble_constant = self._preset.hubble_constant_km_s_mpc
        self._distance_mpc = self._preset.default_distance_mpc
        self._redshift_z = self._preset.default_redshift_z

        self.cosmic_computed.emit(
            self.recessional_velocity_km_s,
            self.effective_redshift_z,
            self.hubble_time_Gyr,
            self.hubble_horizon_distance_Mpc,
        )
        self.update()

    def set_distance_mpc(self, d_mpc: float) -> None:
        """Sets galaxy luminosity distance d in Mpc [0.1, 10000.0]."""
        self._distance_mpc = max(0.1, min(10000.0, d_mpc))
        self.cosmic_computed.emit(
            self.recessional_velocity_km_s,
            self.effective_redshift_z,
            self.hubble_time_Gyr,
            self.hubble_horizon_distance_Mpc,
        )
        self.update()

    def set_hubble_constant(self, h0: float) -> None:
        """Sets Hubble constant H_0 in km/s/Mpc [40.0, 100.0]."""
        self._hubble_constant = max(40.0, min(100.0, h0))
        self.cosmic_computed.emit(
            self.recessional_velocity_km_s,
            self.effective_redshift_z,
            self.hubble_time_Gyr,
            self.hubble_horizon_distance_Mpc,
        )
        self.update()

    def reset_defaults(self) -> None:
        """Resets parameters to active preset defaults."""
        self._hubble_constant = self._preset.hubble_constant_km_s_mpc
        self._distance_mpc = self._preset.default_distance_mpc
        self._redshift_z = self._preset.default_redshift_z
        self.cosmic_computed.emit(
            self.recessional_velocity_km_s,
            self.effective_redshift_z,
            self.hubble_time_Gyr,
            self.hubble_horizon_distance_Mpc,
        )
        self.update()

    # -------------------------------------------------------------------------
    # Layout Buttons
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Defines touch button bounding boxes at the bottom."""
        by = self.CARD_H - 46.0
        bw = 88.0
        bh = 32.0
        spacing = 10.0
        start_x = 18.0

        keys = [
            ("btn_preset", 130.0),
            ("btn_d_down", 72.0),
            ("btn_d_up", 72.0),
            ("btn_h0_down", 72.0),
            ("btn_h0_up", 72.0),
            ("btn_reset", 66.0),
            ("btn_stamp", 112.0),
        ]

        curr_x = start_x
        for key, width in keys:
            self._buttons[key] = QRectF(curr_x, by, width, bh)
            curr_x += width + spacing

    # -------------------------------------------------------------------------
    # Rendering Pipeline
    # -------------------------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Main Card Container
        card_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor("#080c14"))
        grad.setColorAt(1.0, QColor("#0f172a"))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor("#1e293b"), 2.0))
        painter.drawRoundedRect(card_rect, 14.0, 14.0)

        # 2. Header
        self._paint_header(painter)

        # 3. Expanding Universe Field & Scale Factor a(t)
        self._paint_cosmic_expansion_field(painter)

        # 4. Hydrogen Balmer Redshift Spectrum Visualizer
        self._paint_spectral_redshift_panel(painter)

        # 5. Hubble Diagram & Metrics Panel
        self._paint_metrics_panel(painter)

        # 6. Bottom Touchless Buttons
        self._paint_buttons(painter)

    def _paint_header(self, painter: QPainter) -> None:
        # Badge
        badge_rect = QRectF(20.0, 16.0, 185.0, 24.0)
        painter.setBrush(QBrush(QColor("#0284c7")))
        painter.setPen(QPen(QColor("#38bdf8"), 1.0))
        painter.drawRoundedRect(badge_rect, 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, f"🌌 {self._preset.badge_label}")

        # Title
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(218.0, 33.0, self._preset.title_vi)

        # Subtitle
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(
            22.0,
            58.0,
            f"Mục tiêu: {self._preset.target_name} | Hằng số Hubble H₀ = {self._hubble_constant:.1f} km/s/Mpc | Khoảng cách d = {self._distance_mpc:.1f} Mpc",
        )

    def _paint_cosmic_expansion_field(self, painter: QPainter) -> None:
        """Renders the 2D expanding metric cosmological grid with receding galaxies."""
        px = 22.0
        py = 74.0
        pw = 350.0
        ph = 250.0

        # Background box
        painter.setBrush(QBrush(QColor("#04070d")))
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 12.0, py + 20.0, f"1. Vũ Trụ Giãn Nở (Hệ Số Co Giãn a(t) = {self._scale_factor:.2f})")

        cx = px + pw / 2.0
        cy = py + ph / 2.0 + 10.0
        radius_max = (ph / 2.0) - 26.0

        # Concentric distance circles
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for ring_i in [0.33, 0.66, 1.0]:
            r_px = ring_i * radius_max
            ring_pen = QPen(QColor("#1e293b"), 1.0, Qt.PenStyle.DashLine)
            painter.setPen(ring_pen)
            painter.drawEllipse(QPointF(cx, cy), r_px, r_px)

        # Observer at center (Milky Way)
        painter.setBrush(QBrush(QColor("#fbbf24")))
        painter.setPen(QPen(QColor("#ffffff"), 1.5))
        painter.drawEllipse(QPointF(cx, cy), 5.0, 5.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#fbbf24")))
        painter.drawText(cx - 28.0, cy - 8.0, "Ngân Hà (r=0)")

        # Expanding galaxies
        scale = self._scale_factor
        for g in self._galaxies:
            gx = cx + g.base_x * radius_max * scale
            gy = cy + g.base_y * radius_max * scale

            # If inside field
            if (gx - cx) ** 2 + (gy - cy) ** 2 <= radius_max ** 2:
                # Recessional vector line from center to galaxy
                vec_pen = QPen(QColor("#f43f5e"), 1.0, Qt.PenStyle.DotLine)
                painter.setPen(vec_pen)
                painter.drawLine(QPointF(cx, cy), QPointF(gx, gy))

                # Galaxy dot with halo
                painter.setBrush(QBrush(QColor(g.color.red(), g.color.green(), g.color.blue(), 60)))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(gx, gy), g.luminosity + 2.0, g.luminosity + 2.0)

                painter.setBrush(QBrush(g.color))
                painter.setPen(QPen(QColor("#ffffff"), 1.0))
                painter.drawEllipse(QPointF(gx, gy), g.luminosity / 2.0, g.luminosity / 2.0)

        # Cosmological Principle Caption
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#64748b")))
        painter.drawText(px + 12.0, py + ph - 8.0, "Nguyên lý Vũ trụ học: Không gian dãn nở đẳng hướng, mọi điểm đều thấy các thiên hà rời xa.")

    def _paint_spectral_redshift_panel(self, painter: QPainter) -> None:
        """Renders rest-frame vs observed redshifted Hydrogen Balmer absorption lines."""
        px = 384.0
        py = 74.0
        pw = 334.0
        ph = 250.0

        # Background box
        painter.setBrush(QBrush(QColor("#04070d")))
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 12.0, py + 20.0, "2. Phổ Vạch Hấp Thụ Hydro (Balmer Redshift)")

        # Spectrum band 1: Laboratory Rest Frame (z = 0)
        s1_y = py + 52.0
        s_w = pw - 36.0
        s_h = 32.0
        s_x0 = px + 18.0

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(s_x0, s1_y - 6.0, "Phổ Chuẩn Phòng Thí Nghiệm (Rest Frame, z = 0):")

        # Color spectrum gradient from 390 nm (violet) to 700 nm (red)
        spec_grad = QLinearGradient(s_x0, 0.0, s_x0 + s_w, 0.0)
        spec_grad.setColorAt(0.00, QColor("#7c3aed"))   # 390 nm violet
        spec_grad.setColorAt(0.18, QColor("#2563eb"))   # 445 nm blue
        spec_grad.setColorAt(0.35, QColor("#06b6d4"))   # 500 nm cyan
        spec_grad.setColorAt(0.55, QColor("#22c55e"))   # 550 nm green
        spec_grad.setColorAt(0.75, QColor("#eab308"))   # 600 nm yellow
        spec_grad.setColorAt(1.00, QColor("#ef4444"))   # 700 nm red

        painter.setBrush(QBrush(spec_grad))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRect(QRectF(s_x0, s1_y, s_w, s_h))

        # Draw Balmer rest lines (dark absorption lines)
        wl_min = 390.0
        wl_max = 700.0
        for name, wl in BALMER_LINES_NM.items():
            rel_pos = (wl - wl_min) / (wl_max - wl_min)
            lx = s_x0 + rel_pos * s_w
            painter.setPen(QPen(QColor("#000000"), 2.2))
            painter.drawLine(QPointF(lx, s1_y), QPointF(lx, s1_y + s_h))

            # Label
            painter.setFont(QFont("Consolas", 7))
            painter.setPen(QPen(QColor("#ffffff")))
            painter.drawText(lx - 6.0, s1_y + s_h + 12.0, name)

        # Spectrum band 2: Observed Redshifted Galaxy Spectrum
        s2_y = s1_y + 72.0
        z = self.effective_redshift_z
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(s_x0, s2_y - 6.0, f"Phổ Quan Sát Được Từ Thiên Hà (Dịch Đỏ z = {z:.4f}):")

        painter.setBrush(QBrush(spec_grad))
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.drawRect(QRectF(s_x0, s2_y, s_w, s_h))

        # Redshifted lines
        for name, wl_rest in BALMER_LINES_NM.items():
            wl_obs = self.get_observed_wavelength_nm(wl_rest)
            # Rest line position in top spectrum
            lx_rest = s_x0 + ((wl_rest - wl_min) / (wl_max - wl_min)) * s_w

            if wl_obs <= wl_max:
                lx_obs = s_x0 + ((wl_obs - wl_min) / (wl_max - wl_min)) * s_w
                # Draw redshift displacement leader line
                dash_pen = QPen(QColor("#f43f5e"), 1.0, Qt.PenStyle.DotLine)
                painter.setPen(dash_pen)
                painter.drawLine(QPointF(lx_rest, s1_y + s_h), QPointF(lx_obs, s2_y))

                # Absorption line
                painter.setPen(QPen(QColor("#000000"), 2.2))
                painter.drawLine(QPointF(lx_obs, s2_y), QPointF(lx_obs, s2_y + s_h))

                painter.setFont(QFont("Consolas", 7))
                painter.setPen(QPen(QColor("#fca5a5")))
                painter.drawText(lx_obs - 6.0, s2_y + s_h + 12.0, f"{wl_obs:.0f}")
            else:
                # Line shifted beyond visible spectrum into infrared (e.g. JWST)
                painter.setFont(QFont("Segoe UI", 7))
                painter.setPen(QPen(QColor("#f43f5e")))
                painter.drawText(s_x0 + s_w - 95.0, s2_y + s_h / 2.0 + 4.0, f"➔ Vào dải Hồng ngoại (IR)")
                break

        # Redshift formula badge
        f_box = QRectF(s_x0, s2_y + s_h + 18.0, s_w, 24.0)
        painter.setBrush(QBrush(QColor(15, 23, 42, 220)))
        painter.setPen(QPen(QColor("#f43f5e"), 1.0))
        painter.drawRoundedRect(f_box, 4.0, 4.0)

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#f43f5e")))
        painter.drawText(f_box, Qt.AlignmentFlag.AlignCenter, f"z = Δλ / λ₀ = {z:.4f}  |  1 + z = a_obs / a_emit")

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """Renders live quantitative parameters, Hubble law, and cosmic horizon metrics."""
        px = 22.0
        py = 336.0
        pw = self.CARD_W - 44.0
        ph = 142.0

        # Background container
        painter.setBrush(QBrush(QColor("#09101c")))
        painter.setPen(QPen(QColor("#1e293b"), 1.2))
        painter.drawRoundedRect(QRectF(px, py, pw, ph), 8.0, 8.0)

        # Header of metrics
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#38bdf8")))
        painter.drawText(px + 14.0, py + 20.0, "3. Định Luật Hubble-Lemaître: v = H₀ · d  |  Phương Trình Trắc Địa FLRW")

        # Column 1
        painter.setFont(QFont("Segoe UI", 8))
        painter.setPen(QPen(QColor("#e2e8f0")))
        col1_x = px + 14.0
        line1_y = py + 44.0
        dy = 22.0

        v_rec = self.recessional_velocity_km_s
        v_frac_c = v_rec / SPEED_OF_LIGHT_KM_S
        painter.drawText(col1_x, line1_y, f"• Khoảng cách thiên hà d: {self._distance_mpc:.1f} Mpc ({self._distance_mpc * 3.26156:.1f} triệu năm ánh sáng)")
        painter.drawText(col1_x, line1_y + dy, f"• Vận tốc rời xa v = H₀·d: {v_rec:.1f} km/s ({v_frac_c:.3f} c)")
        painter.drawText(col1_x, line1_y + 2 * dy, f"• Độ dịch chuyển đỏ vũ trụ: z = {self.effective_redshift_z:.5f}")
        painter.drawText(col1_x, line1_y + 3 * dy, f"• Tuổi Vũ Trụ Hubble: t_H = 1/H₀ = {self.hubble_time_Gyr:.2f} tỷ năm (Gyr)")

        # Column 2
        col2_x = px + 360.0
        painter.drawText(col2_x, line1_y, f"• Hằng số Hubble H₀: {self._hubble_constant:.2f} km/s/Mpc")
        painter.drawText(col2_x, line1_y + dy, f"• Chân trời vũ trụ (Hubble Sphere): d_H = c/H₀ = {self.hubble_horizon_distance_Mpc:.0f} Mpc ({self.hubble_horizon_distance_Gly:.1f} Gly)")
        painter.drawText(col2_x, line1_y + 2 * dy, f"• Tham số mật độ: Ω_m = {self._preset.omega_matter:.3f} (Vật chất) | Ω_Λ = {self._preset.omega_lambda:.3f} (Năng lượng tối)")
        status_horizon = "VƯỢT CHÂN TRỜI (v > c)" if v_rec > SPEED_OF_LIGHT_KM_S else "TRONG BÁN KÍNH HUBBLE (v < c)"
        status_col = QColor("#ef4444") if v_rec > SPEED_OF_LIGHT_KM_S else QColor("#10b981")
        painter.setPen(QPen(status_col))
        painter.drawText(col2_x, line1_y + 3 * dy, f"• Trạng thái chân trời: {status_horizon}")

        # Explanatory Educational Text at bottom
        desc_rect = QRectF(px + 12.0, py + ph + 8.0, pw - 24.0, 95.0)
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor("#94a3b8")))
        painter.drawText(desc_rect, Qt.TextFlag.TextWordWrap, self._preset.description)

    def _paint_buttons(self, painter: QPainter) -> None:
        """Renders interactive touch/mouse control buttons."""
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))

        labels = {
            "btn_preset": f"📋 Kịch Bản ({self._preset.badge_label})",
            "btn_d_down": "d - 25%",
            "btn_d_up": "d + 25%",
            "btn_h0_down": "H₀ - 5%",
            "btn_h0_up": "H₀ + 5%",
            "btn_reset": "🔄 Đặt Lại",
            "btn_stamp": "📌 Đóng Dấu Bảng",
        }

        for key, rect in self._buttons.items():
            is_hover = (key == self._hovered_btn)
            text = labels.get(key, key)

            if key == "btn_stamp":
                bg = QColor("#0284c7") if not is_hover else QColor("#0369a1")
                border = QColor("#38bdf8")
            elif key == "btn_preset":
                bg = QColor("#4338ca") if not is_hover else QColor("#3730a3")
                border = QColor("#818cf8")
            elif "up" in key:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#38bdf8")
            elif "down" in key:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#f43f5e")
            else:
                bg = QColor("#1e293b") if not is_hover else QColor("#334155")
                border = QColor("#64748b")

            painter.setBrush(QBrush(bg))
            painter.setPen(QPen(border, 1.2))
            painter.drawRoundedRect(rect, 5.0, 5.0)

            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    # -------------------------------------------------------------------------
    # Mouse & Touch Event Handling
    # -------------------------------------------------------------------------
    def hoverMoveEvent(self, event) -> None:
        pos = event.pos()
        old_btn = self._hovered_btn
        self._hovered_btn = None
        for key, rect in self._buttons.items():
            if rect.contains(pos):
                self._hovered_btn = key
                break
        if self._hovered_btn != old_btn:
            self.update()
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        if self._hovered_btn is not None:
            self._hovered_btn = None
            self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.pos()
            for key, rect in self._buttons.items():
                if rect.contains(pos):
                    self._on_button_clicked(key)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def _on_button_clicked(self, key: str) -> None:
        """Dispatches button actions."""
        if key == "btn_preset":
            keys = list(COSMIC_PRESETS.keys())
            idx = keys.index(self._active_preset_code)
            next_code = keys[(idx + 1) % len(keys)]
            self.set_preset(next_code)
        elif key == "btn_d_down":
            self.set_distance_mpc(self._distance_mpc * 0.75)
        elif key == "btn_d_up":
            self.set_distance_mpc(self._distance_mpc * 1.25)
        elif key == "btn_h0_down":
            self.set_hubble_constant(self._hubble_constant * 0.95)
        elif key == "btn_h0_up":
            self.set_hubble_constant(self._hubble_constant * 1.05)
        elif key == "btn_reset":
            self.reset_defaults()
        elif key == "btn_stamp":
            self.request_stamp()

    def request_stamp(self) -> None:
        """Emits signal to stamp vector diagram onto whiteboard canvas."""
        data = {
            "preset_code": self._active_preset_code,
            "target_name": self._preset.target_name,
            "hubble_constant": self._hubble_constant,
            "distance_mpc": self._distance_mpc,
            "recessional_velocity_km_s": self.recessional_velocity_km_s,
            "redshift_z": self.effective_redshift_z,
            "hubble_time_Gyr": self.hubble_time_Gyr,
            "hubble_horizon_distance_Mpc": self.hubble_horizon_distance_Mpc,
        }
        self.stamp_requested.emit(data)
