"""
AirTeach AI - Interactive Thermodynamics & Heat Transfer Simulator
===================================================================
Authentic 2D heat conduction, convection, and thermal boundary layer simulation:
- 2D Transient Heat Diffusion Equation (Fourier's Second Law):
    ρ * c_p * ∂T/∂t = ∇ · (k ∇T) + q_dot
    ∂T/∂t = α * (∂²T/∂x² + ∂²T/∂y²) + q_dot / (ρ * c_p)
    where:
      α = k / (ρ * c_p) is thermal diffusivity (m²/s)
      k is thermal conductivity (W/(m·K))
      ρ is material mass density (kg/m³)
      c_p is specific heat capacity (J/(kg·K))
      q_dot is volumetric heat generation rate (W/m³)
- Finite-Difference Time-Domain (FTCS - Forward-Time Central-Space) Formulation:
    T^{n+1}_{i,j} = T^n_{i,j} + Δt * [ α * ( (T_{i+1,j} - 2*T_{i,j} + T_{i-1,j})/Δx²
                                           + (T_{i,j+1} - 2*T_{i,j} + T_{i,j-1})/Δy² )
                                     + q_dot_{i,j} / (ρ * c_p) ]
    Numerical Stability (Courant-Friedrichs-Lewy / von Neumann condition):
      Fo = α * Δt / min(Δx², Δy²) <= 0.25
- Convective Boundary Conditions (Newton's Law of Cooling):
    -k * (∂T/∂n)|_boundary = h * (T_surface - T_inf)
    where h is heat transfer coefficient (W/(m²·K)), T_inf is ambient temperature.
- Dimensionless Thermal Numbers:
    - Biot Number: Bi = h * L_c / k  (Internal conduction vs surface convection)
    - Fourier Number: Fo = α * t / L_c² (Dimensionless diffusion time)
    - Rayleigh Number: Ra = Gr * Pr = (g * β * (T_s - T_inf) * L³) / (ν * α)
    - Nusselt Number: Nu = h * L / k_fluid (Convective enhancement factor)
- Heat Flux Vector Field:
    q''_x = -k * ∂T/∂x,  q''_y = -k * ∂T/∂y
- 5 Curriculum Presets:
    1. TRANSIENT_ROD_CONDUCTION: 1D Fourier thermal penetration transitioning to linear gradient.
    2. CPU_COOLING_FINS: Concentrated hotspot dissipating through aluminum fin convective array.
    3. NATURAL_CONVECTION_PLUME: Heated base inducing buoyancy-driven thermal boundary layer.
    4. INSULATED_WALL_PULSE: Transient thermal wave through composite wall showing phase lag.
    5. BIMETALLIC_HEAT_SPREADER: Heat flux refraction across Copper-Steel high/low conductivity interface.
- Touchless webcam air gestures (air-tap, slider dragging), bimanual manipulation,
  isothermal contour line extraction, canvas stamping, and .airteach packaging.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)

# Standard Material Thermal Constants (k in W/(m·K), rho in kg/m³, cp in J/(kg·K))
MATERIAL_COPPER = {"name": "Đồng (Copper)", "k": 398.0, "rho": 8940.0, "cp": 385.0}
MATERIAL_ALUMINUM = {"name": "Nhôm (Aluminum)", "k": 205.0, "rho": 2700.0, "cp": 900.0}
MATERIAL_STEEL = {"name": "Thép carbon (Steel)", "k": 50.0, "rho": 7850.0, "cp": 460.0}
MATERIAL_INSULATOR = {"name": "Bông thủy tinh (Fiberglass)", "k": 0.04, "rho": 32.0, "cp": 840.0}
MATERIAL_SILICON = {"name": "Silicon (CPU Die)", "k": 149.0, "rho": 2330.0, "cp": 712.0}


@dataclass
class HeatTransferPreset:
    """Preset configuration for educational heat transfer experiments."""
    key: str
    name: str
    desc: str
    k_default: float          # W/(m·K)
    h_default: float          # W/(m²·K)
    heat_source_w: float      # Watts
    t_ambient: float          # °C
    t_source: float           # °C (Dirichlet condition if applied)


PRESETS: List[HeatTransferPreset] = [
    HeatTransferPreset(
        key="rod_conduction",
        name="Thanh dẫn nhiệt Fourier",
        desc="Dẫn nhiệt 1 chiều từ nguồn nóng 130°C qua thanh kim loại, đối lưu nhẹ ở bề mặt",
        k_default=398.0,
        h_default=15.0,
        heat_source_w=0.0,
        t_ambient=25.0,
        t_source=130.0,
    ),
    HeatTransferPreset(
        key="cpu_fins",
        name="Tản nhiệt CPU vây nhôm",
        desc="Điểm nóng chip CPU 75W tỏa nhiệt qua khối đế và các vây tản nhiệt làm mát cưỡng bức",
        k_default=205.0,
        h_default=85.0,
        heat_source_w=75.0,
        t_ambient=25.0,
        t_source=85.0,
    ),
    HeatTransferPreset(
        key="convection_plume",
        name="Đối lưu & Cột khói nhiệt",
        desc="Điểm gia nhiệt đáy tạo dòng đối lưu tự nhiên nổi lên trên (Rayleigh Ra ~ 3.5e5)",
        k_default=60.0,
        h_default=30.0,
        heat_source_w=90.0,
        t_ambient=20.0,
        t_source=140.0,
    ),
    HeatTransferPreset(
        key="insulated_wall",
        name="Tường cách nhiệt đa lớp",
        desc="Sóng nhiệt bên ngoài truyền qua lớp gạch và bông cách nhiệt, thể hiện độ trễ pha",
        k_default=0.8,
        h_default=12.0,
        heat_source_w=0.0,
        t_ambient=22.0,
        t_source=95.0,
    ),
    HeatTransferPreset(
        key="bimetallic_spreader",
        name="Tản nhiệt lưỡng kim Cu-Fe",
        desc="Khúc xạ dòng nhiệt qua mặt phân giới giữa Đồng (k=398) và Thép (k=50)",
        k_default=398.0,
        h_default=20.0,
        heat_source_w=65.0,
        t_ambient=24.0,
        t_source=115.0,
    ),
]


class InteractiveHeatTransferItem(QGraphicsObject):
    """
    QGraphicsObject widget for authentic 2D thermodynamics, Fourier heat diffusion,
    Newtonian convective cooling, and thermal boundary layer simulation.
    """
    preset_changed = Signal(str)
    property_changed = Signal(str, float)

    WIDTH: float = 900.0
    HEIGHT: float = 600.0

    # Grid resolution for 2D finite difference
    NX: int = 60
    NY: int = 40

    def __init__(
        self,
        board: Optional[LessonBoard] = None,
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.board = board
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        # Simulation state
        self._preset_index: int = 0
        self._is_paused: bool = False
        self._sim_time: float = 0.0
        self._sim_speed: float = 1.0

        # Physical domain dimensions (meters)
        self._lx: float = 0.30  # 30 cm
        self._ly: float = 0.20  # 20 cm
        self._dx: float = self._lx / (self.NX - 1)
        self._dy: float = self._ly / (self.NY - 1)

        # Configurable thermal parameters
        self._k: float = PRESETS[0].k_default          # W/(m·K)
        self._h: float = PRESETS[0].h_default          # W/(m²·K)
        self._heat_source_w: float = PRESETS[0].heat_source_w  # Watts
        self._t_ambient: float = PRESETS[0].t_ambient  # °C
        self._t_source: float = PRESETS[0].t_source    # °C

        # Material fields (default homogeneous aluminum/copper)
        self._rho: float = 2700.0   # kg/m³
        self._cp: float = 900.0     # J/(kg·K)
        self._k_map = np.full((self.NY, self.NX), self._k, dtype=np.float64)
        self._alpha_map = np.full((self.NY, self.NX), self._k / (self._rho * self._cp), dtype=np.float64)

        # Temperature field (NY x NX) initialized to ambient
        self._temp = np.full((self.NY, self.NX), self._t_ambient, dtype=np.float64)
        self._temp_prev = self._temp.copy()

        # Fixed boundary masks (Dirichlet)
        self._fixed_mask = np.zeros((self.NY, self.NX), dtype=bool)
        self._fixed_temp = np.full((self.NY, self.NX), self._t_ambient, dtype=np.float64)

        # Heat source volumetric field (W/m³)
        self._q_dot = np.zeros((self.NY, self.NX), dtype=np.float64)

        # Display modes
        self._show_flux_vectors: bool = True
        self._show_isotherms: bool = True
        self._probe_active: bool = False
        self._probe_pos: QPointF = QPointF(self.WIDTH * 0.45, 230.0)

        # Air gesture interaction tracking
        self._dragging_slider: Optional[str] = None
        self._dragging_probe: bool = False
        self._hover_pos: Optional[QPointF] = None

        # Precomputed colormap lookup table (256 RGB tuples)
        self._colormap_lut = self._build_colormap_lut()

        # Render cache
        self._image_cache: Optional[QImage] = None

        # Apply initial preset geometry and boundary conditions
        self._apply_preset_geometry()

    @property
    def current_preset(self) -> HeatTransferPreset:
        return PRESETS[self._preset_index]

    def _build_colormap_lut(self) -> List[Tuple[int, int, int]]:
        """Generate high-contrast scientific thermal colormap (Blue -> Cyan -> Green -> Yellow -> Orange -> Red -> White)."""
        lut: List[Tuple[int, int, int]] = []
        # Keypoints: normalized t in [0, 1]
        pts = [
            (0.00, (15, 23, 42)),      # Deep cold slate/navy
            (0.15, (30, 90, 190)),     # Blue
            (0.35, (16, 185, 129)),    # Teal/Cyan-green
            (0.55, (234, 179, 8)),     # Yellow-gold
            (0.75, (249, 115, 22)),    # Amber-orange
            (0.90, (239, 68, 68)),     # Hot red
            (1.00, (255, 255, 255)),   # Incandescent white
        ]
        for i in range(256):
            val = i / 255.0
            for idx in range(len(pts) - 1):
                t0, c0 = pts[idx]
                t1, c1 = pts[idx + 1]
                if t0 <= val <= t1:
                    u = (val - t0) / (t1 - t0) if t1 > t0 else 0.0
                    r = int(round(c0[0] + u * (c1[0] - c0[0])))
                    g = int(round(c0[1] + u * (c1[1] - c0[1])))
                    b = int(round(c0[2] + u * (c1[2] - c0[2])))
                    lut.append((r, g, b))
                    break
        while len(lut) < 256:
            lut.append((255, 255, 255))
        return lut

    def _apply_preset_geometry(self) -> None:
        """Configure material maps, heat sources, and Dirichlet boundaries according to preset."""
        preset = self.current_preset
        self._k = preset.k_default
        self._h = preset.h_default
        self._heat_source_w = preset.heat_source_w
        self._t_ambient = preset.t_ambient
        self._t_source = preset.t_source

        # Reset fields
        self._temp.fill(self._t_ambient)
        self._temp_prev = self._temp.copy()
        self._fixed_mask.fill(False)
        self._fixed_temp.fill(self._t_ambient)
        self._q_dot.fill(0.0)

        # Cell volume (assume unit depth 0.01 m)
        dz = 0.01
        cell_vol = self._dx * self._dy * dz

        if preset.key == "rod_conduction":
            # Pure Copper 1D-like conduction
            self._k_map.fill(self._k)
            self._rho = MATERIAL_COPPER["rho"]
            self._cp = MATERIAL_COPPER["cp"]
            # Left edge held fixed at T_source
            self._fixed_mask[:, 0:2] = True
            self._fixed_temp[:, 0:2] = self._t_source
            self._temp[:, 0:2] = self._t_source

        elif preset.key == "cpu_fins":
            # Aluminum base with vertical fins
            self._k_map.fill(self._k)
            self._rho = MATERIAL_ALUMINUM["rho"]
            self._cp = MATERIAL_ALUMINUM["cp"]
            # Central heat generation at bottom (CPU Die: 20mm x 20mm)
            cx = self.NX // 2
            cy = self.NY - 4
            num_cells = 4 * 4
            q_vol = (self._heat_source_w / (num_cells * cell_vol)) if cell_vol > 0 else 1e6
            self._q_dot[cy:cy + 4, cx - 2:cx + 2] = q_vol
            # Air gaps between fins: top half has 5 fins, lower conductivity in gaps
            fin_width = 3
            gap_width = 4
            for x in range(self.NX):
                # Pattern fins in top 20 rows
                if (x % (fin_width + gap_width)) >= fin_width:
                    self._k_map[0:self.NY // 2, x] = 0.026  # Air gap

        elif preset.key == "convection_plume":
            # Fluid / low-k medium with intense localized bottom heater
            self._k_map.fill(self._k)
            self._rho = 1000.0
            self._cp = 1200.0
            # Bottom localized hot heater
            cx = self.NX // 2
            self._fixed_mask[self.NY - 3:self.NY, cx - 5:cx + 5] = True
            self._fixed_temp[self.NY - 3:self.NY, cx - 5:cx + 5] = self._t_source
            self._temp[self.NY - 3:self.NY, cx - 5:cx + 5] = self._t_source

        elif preset.key == "insulated_wall":
            # Multi-layer wall: Brick (left 15 cols), Fiberglass insulation (middle 15 cols), Drywall (right 30 cols)
            self._k_map[:, 0:15] = 0.72       # Brick
            self._k_map[:, 15:30] = 0.038     # Fiberglass
            self._k_map[:, 30:] = 0.17        # Drywall
            self._rho = 1800.0
            self._cp = 880.0
            # Left exterior exposed to intense thermal boundary
            self._fixed_mask[:, 0:2] = True
            self._fixed_temp[:, 0:2] = self._t_source
            self._temp[:, 0:2] = self._t_source

        elif preset.key == "bimetallic_spreader":
            # Left half: Copper (k=398), Right half: Stainless Steel (k=50)
            mid = self.NX // 2
            self._k_map[:, :mid] = MATERIAL_COPPER["k"]
            self._k_map[:, mid:] = MATERIAL_STEEL["k"]
            self._rho = 8400.0
            self._cp = 420.0
            # Concentrated point source near interface
            self._fixed_mask[self.NY // 2 - 3:self.NY // 2 + 3, mid - 4:mid] = True
            self._fixed_temp[self.NY // 2 - 3:self.NY // 2 + 3, mid - 4:mid] = self._t_source
            self._temp[self.NY // 2 - 3:self.NY // 2 + 3, mid - 4:mid] = self._t_source

        # Update thermal diffusivity alpha = k / (rho * cp)
        self._alpha_map = np.clip(self._k_map / (self._rho * self._cp), 1e-9, 1e-2)
        self._sim_time = 0.0

    def set_preset(self, index: int) -> None:
        """Switch active educational heat transfer preset."""
        if 0 <= index < len(PRESETS):
            self._preset_index = index
            self._apply_preset_geometry()
            self.preset_changed.emit(self.current_preset.key)
            self.update()

    def set_thermal_conductivity(self, val: float) -> None:
        """Update bulk thermal conductivity (W/(m·K))."""
        self._k = max(0.01, min(val, 500.0))
        # Scale k_map proportionally
        curr_mean = float(np.mean(self._k_map))
        if curr_mean > 0:
            scale = self._k / curr_mean
            self._k_map *= scale
            self._alpha_map = np.clip(self._k_map / (self._rho * self._cp), 1e-9, 1e-2)
        self.property_changed.emit("k", self._k)
        self.update()

    def set_convection_coeff(self, val: float) -> None:
        """Update surface convective heat transfer coefficient h (W/(m²·K))."""
        self._h = max(1.0, min(val, 200.0))
        self.property_changed.emit("h", self._h)
        self.update()

    def set_heat_source_power(self, val: float) -> None:
        """Update volumetric heat source power (Watts)."""
        self._heat_source_w = max(0.0, min(val, 200.0))
        # Re-apply source to q_dot if preset has internal generation
        if self.current_preset.key == "cpu_fins":
            dz = 0.01
            cell_vol = self._dx * self._dy * dz
            cx = self.NX // 2
            cy = self.NY - 4
            num_cells = 4 * 4
            q_vol = (self._heat_source_w / (num_cells * cell_vol)) if cell_vol > 0 else 0.0
            self._q_dot.fill(0.0)
            self._q_dot[cy:cy + 4, cx - 2:cx + 2] = q_vol
        self.property_changed.emit("heat_source_w", self._heat_source_w)
        self.update()

    def set_ambient_temp(self, val: float) -> None:
        """Update ambient environment fluid temperature (°C)."""
        self._t_ambient = max(-20.0, min(val, 60.0))
        self.property_changed.emit("t_ambient", self._t_ambient)
        self.update()

    def toggle_pause(self) -> None:
        """Toggle simulation pause/resume state."""
        self._is_paused = not self._is_paused
        self.update()

    def reset_simulation(self) -> None:
        """Reset thermal fields to initial preset conditions."""
        self._apply_preset_geometry()
        self.update()

    def step_simulation(self, dt: float) -> None:
        """
        Advance 2D transient Fourier heat conduction by dt seconds using FTCS.
        Ensures unconditional numerical stability via sub-stepping.
        """
        if self._is_paused or dt <= 0.0:
            return

        effective_dt = dt * self._sim_speed

        # Calculate max allowable stable time step: dt_cfl <= 0.25 * min(dx^2, dy^2) / max(alpha)
        max_alpha = float(np.max(self._alpha_map))
        min_dxy2 = min(self._dx**2, self._dy**2)
        dt_max_stable = (0.22 * min_dxy2 / max_alpha) if max_alpha > 0 else 0.01

        # Perform sub-stepping to ensure strict FTCS stability
        steps = max(1, int(math.ceil(effective_dt / dt_max_stable)))
        sub_dt = effective_dt / steps

        dx2 = self._dx**2
        dy2 = self._dy**2
        inv_dx2 = 1.0 / dx2
        inv_dy2 = 1.0 / dy2

        for _ in range(steps):
            T = self._temp
            # 2nd derivative in X (interior)
            d2T_dx2 = (T[:, 2:] - 2.0 * T[:, 1:-1] + T[:, :-2]) * inv_dx2
            # 2nd derivative in Y (interior)
            d2T_dy2 = (T[2:, :] - 2.0 * T[1:-1, :] + T[:-2, :]) * inv_dy2

            # Laplacian on interior (NY-2, NX-2)
            laplacian = d2T_dx2[1:-1, :] + d2T_dy2[:, 1:-1]

            # Temperature change rate
            dT_dt = (
                self._alpha_map[1:-1, 1:-1] * laplacian
                + self._q_dot[1:-1, 1:-1] / (self._rho * self._cp)
            )

            # Update interior temperature
            self._temp[1:-1, 1:-1] += sub_dt * dT_dt

            # Apply Newton's Law of Cooling Convective Boundary Conditions on edges
            # Top boundary (y = 0): -k * dT/dy = h * (T - T_inf) => T[0, :] = T[1, :] - (h*dy/k)*(T - T_inf)
            bi_y = (self._h * self._dy) / np.maximum(self._k_map[0, :], 0.01)
            self._temp[0, :] = (self._temp[1, :] + bi_y * self._t_ambient) / (1.0 + bi_y)

            # Bottom boundary (y = NY-1)
            bi_y_b = (self._h * self._dy) / np.maximum(self._k_map[-1, :], 0.01)
            self._temp[-1, :] = (self._temp[-2, :] + bi_y_b * self._t_ambient) / (1.0 + bi_y_b)

            # Left boundary (x = 0)
            bi_x_l = (self._h * self._dx) / np.maximum(self._k_map[:, 0], 0.01)
            self._temp[:, 0] = (self._temp[:, 1] + bi_x_l * self._t_ambient) / (1.0 + bi_x_l)

            # Right boundary (x = NX-1)
            bi_x_r = (self._h * self._dx) / np.maximum(self._k_map[:, -1], 0.01)
            self._temp[:, -1] = (self._temp[:, -2] + bi_x_r * self._t_ambient) / (1.0 + bi_x_r)

            # Enforce fixed Dirichlet temperature constraints
            if np.any(self._fixed_mask):
                self._temp[self._fixed_mask] = self._fixed_temp[self._fixed_mask]

            # Numerical sanity clamp to prevent physical divergence
            np.clip(self._temp, -50.0, 1000.0, out=self._temp)

        self._sim_time += effective_dt
        self.update()

    def get_probe_reading(self) -> Tuple[float, float, float, float]:
        """
        Return local thermodynamic metrics at probe position:
        (temperature_c, grad_t_magnitude, heat_flux_w_m2, biot_number)
        """
        canvas_rect = self._canvas_rect()
        px = self._probe_pos.x()
        py = self._probe_pos.y()
        if not canvas_rect.contains(px, py):
            return (self._t_ambient, 0.0, 0.0, 0.0)

        norm_x = (px - canvas_rect.left()) / canvas_rect.width()
        norm_y = (py - canvas_rect.top()) / canvas_rect.height()

        gx = int(round(norm_x * (self.NX - 1)))
        gy = int(round(norm_y * (self.NY - 1)))
        gx = max(0, min(gx, self.NX - 1))
        gy = max(0, min(gy, self.NY - 1))

        temp_c = float(self._temp[gy, gx])

        # Compute spatial gradient dT/dx, dT/dy
        if 0 < gx < self.NX - 1:
            dT_dx = (self._temp[gy, gx + 1] - self._temp[gy, gx - 1]) / (2.0 * self._dx)
        else:
            dT_dx = 0.0

        if 0 < gy < self.NY - 1:
            dT_dy = (self._temp[gy + 1, gx] - self._temp[gy - 1, gx]) / (2.0 * self._dy)
        else:
            dT_dy = 0.0

        grad_mag = math.sqrt(dT_dx**2 + dT_dy**2)
        local_k = float(self._k_map[gy, gx])
        heat_flux = local_k * grad_mag

        # Biot number: Bi = h * L_c / k
        l_c = self._lx * 0.5
        biot = (self._h * l_c) / local_k if local_k > 0 else 0.0

        return (temp_c, grad_mag, heat_flux, biot)

    # -------------------------------------------------------------------------
    # QGraphicsItem Overrides & Layout Geometry
    # -------------------------------------------------------------------------

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.WIDTH, self.HEIGHT)

    def _canvas_rect(self) -> QRectF:
        """Bounding rect of the 2D heat diffusion canvas."""
        return QRectF(25.0, 115.0, 560.0, 360.0)

    def _hud_rect(self) -> QRectF:
        """Bounding rect of the scientific HUD panel on the right."""
        return QRectF(600.0, 115.0, 275.0, 360.0)

    def _controls_rect(self) -> QRectF:
        """Bounding rect of the bottom slider control panel."""
        return QRectF(25.0, 490.0, 850.0, 95.0)

    def _preset_button_rects(self) -> List[Tuple[QRectF, HeatTransferPreset]]:
        """Compute hit rectangles for preset selector tabs along the top."""
        rects = []
        btn_w = 162.0
        btn_h = 32.0
        start_x = 25.0
        y = 65.0
        spacing = 10.0
        for i, preset in enumerate(PRESETS):
            x = start_x + i * (btn_w + spacing)
            rects.append((QRectF(x, y, btn_w, btn_h), preset))
        return rects

    # -------------------------------------------------------------------------
    # Painting & Rendering
    # -------------------------------------------------------------------------

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # 1. Main Glassmorphic Container
        self._paint_background(painter)

        # 2. Header & Title Bar
        self._paint_header(painter)

        # 3. Preset Buttons
        self._paint_preset_buttons(painter)

        # 4. 2D Heat Diffusion Canvas & Isotherms
        self._paint_heat_canvas(painter)

        # 5. Scientific Analytics HUD
        self._paint_hud(painter)

        # 6. Bottom Parameter Sliders
        self._paint_controls(painter)

        # 7. Virtual Temperature Probe (if active)
        self._paint_probe(painter)

        painter.restore()

    def _paint_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        bg_grad = QLinearGradient(0, 0, 0, self.HEIGHT)
        bg_grad.setColorAt(0.0, QColor(11, 15, 25, 250))
        bg_grad.setColorAt(1.0, QColor(15, 23, 42, 252))
        painter.setBrush(QBrush(bg_grad))
        painter.setPen(QPen(QColor(56, 189, 248, 80), 1.5))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        # Title
        painter.setPen(QColor(241, 245, 249))
        font_title = QFont("Segoe UI", 13, QFont.Bold)
        painter.setFont(font_title)
        painter.drawText(28, 38, "NHIỆT ĐỘNG LỰC HỌC: TRUYỀN NHIỆT & ĐỐI LƯU 2D (FOURIER & NEWTON)")

        # Subtitle
        painter.setPen(QColor(148, 163, 184))
        font_sub = QFont("Segoe UI", 9)
        painter.setFont(font_sub)
        painter.drawText(28, 54, "Phương trình vi phân dẫn nhiệt Fourier 2D, đối lưu làm mát bề mặt & lớp biên nhiệt")

        # Status badge (Live / Paused)
        badge_rect = QRectF(self.WIDTH - 150.0, 22.0, 122.0, 26.0)
        if self._is_paused:
            painter.setBrush(QColor(239, 68, 68, 40))
            painter.setPen(QPen(QColor(239, 68, 68), 1.2))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)
            painter.setPen(QColor(252, 165, 165))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, "⏸ TẠM DỪNG")
        else:
            painter.setBrush(QColor(16, 185, 129, 40))
            painter.setPen(QPen(QColor(16, 185, 129), 1.2))
            painter.drawRoundedRect(badge_rect, 6.0, 6.0)
            painter.setPen(QColor(110, 231, 183))
            painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
            painter.drawText(badge_rect, Qt.AlignCenter, "▶ ĐANG MÔ PHỎNG")

    def _paint_preset_buttons(self, painter: QPainter) -> None:
        font_btn = QFont("Segoe UI", 8, QFont.Bold)
        painter.setFont(font_btn)
        for i, (btn_rect, preset) in enumerate(self._preset_button_rects()):
            is_active = (i == self._preset_index)
            if is_active:
                painter.setBrush(QColor(14, 165, 233, 220))
                painter.setPen(QPen(QColor(186, 230, 253), 1.2))
            else:
                painter.setBrush(QColor(30, 41, 59, 180))
                painter.setPen(QPen(QColor(71, 85, 105), 1.0))
            painter.drawRoundedRect(btn_rect, 6.0, 6.0)

            painter.setPen(QColor(255, 255, 255) if is_active else QColor(203, 213, 225))
            painter.drawText(btn_rect, Qt.AlignCenter, preset.name)

    def _paint_heat_canvas(self, painter: QPainter) -> None:
        """Render 2D thermal color field, material structures, and heat flux arrows."""
        c_rect = self._canvas_rect()

        # Canvas frame
        painter.setBrush(QColor(5, 8, 15))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(c_rect, 8.0, 8.0)

        # Build QImage representation of temperature matrix
        t_min = float(self._t_ambient)
        t_max = max(t_min + 10.0, float(np.max(self._temp)))

        # Create 2D RGB buffer
        img = QImage(self.NX, self.NY, QImage.Format_RGB32)
        norm_t = np.clip((self._temp - t_min) / (t_max - t_min), 0.0, 1.0)
        lut_indices = (norm_t * 255.0).astype(np.int32)

        for y in range(self.NY):
            for x in range(self.NX):
                idx = lut_indices[y, x]
                r, g, b = self._colormap_lut[idx]
                img.setPixelColor(x, y, QColor(r, g, b))

        # Render smoothed temperature heatmap into canvas
        painter.save()
        painter.setClipRect(c_rect)
        painter.drawImage(c_rect, img)

        # Overlay Material Boundaries / Air gaps if present (e.g. CPU fins)
        if self.current_preset.key == "cpu_fins":
            painter.setPen(QPen(QColor(0, 0, 0, 160), 1.5))
            fin_w_px = (3.0 / self.NX) * c_rect.width()
            gap_w_px = (4.0 / self.NX) * c_rect.width()
            period_px = fin_w_px + gap_w_px
            fin_h_px = c_rect.height() * 0.5
            for x_idx in range(self.NX):
                if (x_idx % 7) >= 3:
                    # Gap column
                    gx = c_rect.left() + (x_idx / self.NX) * c_rect.width()
                    gw = c_rect.width() / self.NX
                    painter.fillRect(QRectF(gx, c_rect.top(), gw, fin_h_px), QColor(10, 15, 26, 210))

        elif self.current_preset.key == "bimetallic_spreader":
            # Draw interface boundary line between Copper & Steel
            mid_x = c_rect.left() + c_rect.width() * 0.5
            painter.setPen(QPen(QColor(255, 255, 255, 180), 1.5, Qt.DashLine))
            painter.drawLine(QPointF(mid_x, c_rect.top()), QPointF(mid_x, c_rect.bottom()))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(253, 186, 116))
            painter.drawText(QRectF(c_rect.left() + 10, c_rect.top() + 8, 120, 20), Qt.AlignLeft, "Đồng (k=398)")
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(QRectF(mid_x + 10, c_rect.top() + 8, 120, 20), Qt.AlignLeft, "Thép (k=50)")

        # Render Isotherms (Contour Lines)
        if self._show_isotherms:
            self._paint_isotherms(painter, c_rect, t_min, t_max)

        # Render Heat Flux Arrows (q'' = -k grad T)
        if self._show_flux_vectors:
            self._paint_flux_arrows(painter, c_rect)

        # Canvas Border
        painter.restore()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(56, 189, 248, 100), 1.5))
        painter.drawRoundedRect(c_rect, 8.0, 8.0)

        # Colorbar legend at bottom of canvas
        self._paint_colorbar(painter, c_rect, t_min, t_max)

    def _paint_colorbar(self, painter: QPainter, c_rect: QRectF, t_min: float, t_max: float) -> None:
        bar_w = 260.0
        bar_h = 10.0
        bar_x = c_rect.right() - bar_w - 12.0
        bar_y = c_rect.bottom() - bar_h - 12.0
        bar_rect = QRectF(bar_x, bar_y, bar_w, bar_h)

        cbar_grad = QLinearGradient(bar_x, 0, bar_x + bar_w, 0)
        cbar_grad.setColorAt(0.00, QColor(15, 23, 42))
        cbar_grad.setColorAt(0.15, QColor(30, 90, 190))
        cbar_grad.setColorAt(0.35, QColor(16, 185, 129))
        cbar_grad.setColorAt(0.55, QColor(234, 179, 8))
        cbar_grad.setColorAt(0.75, QColor(249, 115, 22))
        cbar_grad.setColorAt(0.90, QColor(239, 68, 68))
        cbar_grad.setColorAt(1.00, QColor(255, 255, 255))

        painter.setBrush(QBrush(cbar_grad))
        painter.setPen(QPen(QColor(255, 255, 255, 140), 1.0))
        painter.drawRoundedRect(bar_rect, 3.0, 3.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Bold))
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(QRectF(bar_x - 60, bar_y - 2, 55, 14), Qt.AlignRight, f"{t_min:.0f}°C")
        painter.drawText(QRectF(bar_x + bar_w + 5, bar_y - 2, 55, 14), Qt.AlignLeft, f"{t_max:.0f}°C")

    def _paint_isotherms(self, painter: QPainter, c_rect: QRectF, t_min: float, t_max: float) -> None:
        """Draw isothermal temperature contour lines."""
        span = t_max - t_min
        if span <= 5.0:
            return

        step = 15.0 if span > 60 else (10.0 if span > 30 else 5.0)
        start_level = math.ceil(t_min / step) * step
        levels = np.arange(start_level, t_max, step)

        painter.setPen(QPen(QColor(255, 255, 255, 75), 1.0, Qt.DotLine))
        cell_w = c_rect.width() / (self.NX - 1)
        cell_h = c_rect.height() / (self.NY - 1)

        # Simplified marching segments across grid cells
        for level in levels:
            for y in range(0, self.NY - 1, 2):
                for x in range(0, self.NX - 1, 2):
                    t00 = self._temp[y, x]
                    t10 = self._temp[y, x + 1]
                    t01 = self._temp[y + 1, x]
                    if (min(t00, t10) <= level <= max(t00, t10)) or (min(t00, t01) <= level <= max(t00, t01)):
                        px = c_rect.left() + x * cell_w
                        py = c_rect.top() + y * cell_h
                        painter.drawPoint(QPointF(px, py))

    def _paint_flux_arrows(self, painter: QPainter, c_rect: QRectF) -> None:
        """Render sparse heat flux vectors q'' = -k grad(T) pointing down temperature gradients."""
        cell_w = c_rect.width() / (self.NX - 1)
        cell_h = c_rect.height() / (self.NY - 1)

        painter.setPen(QPen(QColor(253, 224, 71, 160), 1.2))
        step_x = 4
        step_y = 4

        for y in range(2, self.NY - 2, step_y):
            for x in range(2, self.NX - 2, step_x):
                dT_dx = (self._temp[y, x + 1] - self._temp[y, x - 1]) / (2.0 * self._dx)
                dT_dy = (self._temp[y + 1, x] - self._temp[y - 1, x]) / (2.0 * self._dy)
                grad_mag = math.sqrt(dT_dx**2 + dT_dy**2)
                if grad_mag > 5.0:
                    local_k = float(self._k_map[y, x])
                    # Heat flux is opposite to gradient
                    qx = -local_k * dT_dx
                    qy = -local_k * dT_dy
                    flux_mag = math.sqrt(qx**2 + qy**2)
                    if flux_mag > 0:
                        norm_len = min(16.0, max(5.0, math.log10(flux_mag + 1.0) * 3.5))
                        vx = (qx / flux_mag) * norm_len
                        vy = (qy / flux_mag) * norm_len

                        px = c_rect.left() + x * cell_w
                        py = c_rect.top() + y * cell_h
                        p1 = QPointF(px, py)
                        p2 = QPointF(px + vx, py + vy)
                        painter.drawLine(p1, p2)

                        # Arrowhead
                        angle = math.atan2(vy, vx)
                        head_len = 3.5
                        ha1 = angle + math.pi * 0.82
                        ha2 = angle - math.pi * 0.82
                        painter.drawLine(p2, QPointF(p2.x() + head_len * math.cos(ha1), p2.y() + head_len * math.sin(ha1)))
                        painter.drawLine(p2, QPointF(p2.x() + head_len * math.cos(ha2), p2.y() + head_len * math.sin(ha2)))

    def _paint_hud(self, painter: QPainter) -> None:
        """Render scientific telemetry HUD: Fourier metrics, Biot number, convection rate."""
        h_rect = self._hud_rect()

        # HUD Panel background
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(h_rect, 8.0, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
        painter.setPen(QColor(56, 189, 248))
        painter.drawText(int(h_rect.left() + 14), int(h_rect.top() + 24), "THÔNG SỐ NHIỆT THỰC (HUD)")

        # Divider
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawLine(
            QPointF(h_rect.left() + 14, h_rect.top() + 32),
            QPointF(h_rect.right() - 14, h_rect.top() + 32)
        )

        # Physical calculations
        t_min = float(np.min(self._temp))
        t_max = float(np.max(self._temp))
        t_avg = float(np.mean(self._temp))

        # Total heat dissipated via surface convection: Q_out = integral(h * (T - T_inf) dA)
        # Perimeter edges sum
        p_cells = np.concatenate([self._temp[0, :], self._temp[-1, :], self._temp[:, 0], self._temp[:, -1]])
        q_dissipated_w = float(np.sum(self._h * (p_cells - self._t_ambient) * self._dx * 0.01))

        # Biot number: Bi = h * L_c / k
        l_c = self._lx * 0.5
        biot_avg = (self._h * l_c) / self._k if self._k > 0 else 0.0

        # Fourier number: Fo = alpha * t / L_c^2
        alpha_eff = float(np.mean(self._alpha_map))
        fo_num = (alpha_eff * self._sim_time) / (l_c**2) if l_c > 0 else 0.0

        # Rayleigh number for natural convection (buoyancy)
        # Ra = g * beta * deltaT * L^3 / (nu * alpha)
        delta_t = max(0.1, t_max - self._t_ambient)
        ra_num = (9.81 * (1.0 / (self._t_ambient + 273.15)) * delta_t * (self._ly**3)) / (1.5e-5 * 2.2e-5)

        # Metrics rows
        y_cursor = h_rect.top() + 52.0
        line_spacing = 24.0

        metrics = [
            ("Thời gian mô phỏng (t):", f"{self._sim_time:.2f} s"),
            ("Nhiệt độ cực đại (T_max):", f"{t_max:.1f} °C"),
            ("Nhiệt độ trung bình (T_avg):", f"{t_avg:.1f} °C"),
            ("Nhiệt độ môi trường (T_inf):", f"{self._t_ambient:.1f} °C"),
            ("Nhiệt lượng tỏa ra (Q_out):", f"{max(0.0, q_dissipated_w):.1f} W"),
            ("Độ dẫn nhiệt k:", f"{self._k:.1f} W/(m·K)"),
            ("Hệ số đối lưu h:", f"{self._h:.1f} W/(m²·K)"),
            ("Số Biot (Bi = h·L/k):", f"{biot_avg:.3f}"),
            ("Số Fourier (Fo = α·t/L²):", f"{fo_num:.3f}"),
            ("Số Rayleigh (Ra):", f"{ra_num:.2e}"),
        ]

        painter.setFont(QFont("Segoe UI", 8))
        for label, val_str in metrics:
            painter.setPen(QColor(148, 163, 184))
            painter.drawText(int(h_rect.left() + 14), int(y_cursor), label)
            painter.setPen(QColor(241, 245, 249))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(
                QRectF(h_rect.left() + 14, y_cursor - 12, h_rect.width() - 28, 16),
                Qt.AlignRight,
                val_str
            )
            painter.setFont(QFont("Segoe UI", 8))
            y_cursor += line_spacing

        # Regime diagnostic badge
        y_cursor += 6.0
        regime_rect = QRectF(h_rect.left() + 14, y_cursor, h_rect.width() - 28, 26)
        if biot_avg < 0.1:
            painter.setBrush(QColor(16, 185, 129, 30))
            painter.setPen(QPen(QColor(16, 185, 129), 1.0))
            painter.drawRoundedRect(regime_rect, 5.0, 5.0)
            painter.setPen(QColor(110, 231, 183))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(regime_rect, Qt.AlignCenter, "Bi < 0.1: Điện dung nhiệt gộp (Lumped)")
        else:
            painter.setBrush(QColor(56, 189, 248, 30))
            painter.setPen(QPen(QColor(56, 189, 248), 1.0))
            painter.drawRoundedRect(regime_rect, 5.0, 5.0)
            painter.setPen(QColor(186, 230, 253))
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.drawText(regime_rect, Qt.AlignCenter, "Bi >= 0.1: Gradient nhiệt nội tại chiếm ưu thế")

    def _paint_controls(self, painter: QPainter) -> None:
        """Render interactive bottom sliders for thermal conductivity k, convection h, source power, ambient."""
        ctrl_rect = self._controls_rect()

        # Background container
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.setPen(QPen(QColor(51, 65, 85), 1.0))
        painter.drawRoundedRect(ctrl_rect, 8.0, 8.0)

        # 4 Sliders arranged horizontally
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        sliders = [
            ("k", "Độ dẫn nhiệt k", f"{self._k:.1f} W/(m·K)", self._k, 0.1, 400.0),
            ("h", "Hệ số đối lưu h", f"{self._h:.1f} W/(m²·K)", self._h, 1.0, 150.0),
            ("p", "Công suất nhiệt P", f"{self._heat_source_w:.1f} W", self._heat_source_w, 0.0, 150.0),
            ("t_inf", "Nhiệt độ môi trường T_∞", f"{self._t_ambient:.1f} °C", self._t_ambient, 0.0, 50.0),
        ]

        for i, (key, label, val_text, val, min_v, max_v) in enumerate(sliders):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0

            # Slider Title & Value
            painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
            painter.setPen(QColor(203, 213, 225))
            painter.drawText(int(cx), int(cy), label)

            painter.setPen(QColor(56, 189, 248))
            painter.drawText(QRectF(cx, cy - 12, col_w - 20, 14), Qt.AlignRight, val_text)

            # Track
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx, track_y, track_w, 6.0)
            painter.setBrush(QColor(30, 41, 59))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(track_rect, 3.0, 3.0)

            # Active fill
            norm = (val - min_v) / (max_v - min_v) if max_v > min_v else 0.0
            fill_rect = QRectF(cx, track_y, track_w * norm, 6.0)
            painter.setBrush(QColor(14, 165, 233))
            painter.drawRoundedRect(fill_rect, 3.0, 3.0)

            # Thumb handle
            thumb_x = cx + track_w * norm
            thumb_rect = QRectF(thumb_x - 6.0, track_y - 4.0, 12.0, 14.0)
            painter.setBrush(QColor(241, 245, 249))
            painter.setPen(QPen(QColor(14, 165, 233), 1.2))
            painter.drawRoundedRect(thumb_rect, 3.0, 3.0)

        # Bottom Action Bar: Toggle Probe & Toggle Flux Vectors
        action_y = ctrl_rect.top() + 62.0
        btn_probe_rect = QRectF(ctrl_rect.left() + 20, action_y, 160, 24)
        painter.setBrush(QColor(56, 189, 248, 40) if self._probe_active else QColor(30, 41, 59))
        painter.setPen(QPen(QColor(56, 189, 248) if self._probe_active else QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(btn_probe_rect, 4.0, 4.0)
        painter.setPen(QColor(241, 245, 249))
        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.drawText(btn_probe_rect, Qt.AlignCenter, "🔍 NHIỆT KẾ ẢO (PROBE)")

        btn_flux_rect = QRectF(ctrl_rect.left() + 190, action_y, 160, 24)
        painter.setBrush(QColor(234, 179, 8, 40) if self._show_flux_vectors else QColor(30, 41, 59))
        painter.setPen(QPen(QColor(234, 179, 8) if self._show_flux_vectors else QColor(71, 85, 105), 1.0))
        painter.drawRoundedRect(btn_flux_rect, 4.0, 4.0)
        painter.setPen(QColor(241, 245, 249))
        painter.drawText(btn_flux_rect, Qt.AlignCenter, "⚡ VECTƠ DÒNG NHIỆT (q'')")

    def _paint_probe(self, painter: QPainter) -> None:
        """Render virtual temperature probe crosshair & reading tooltip."""
        if not self._probe_active:
            return

        c_rect = self._canvas_rect()
        px = max(c_rect.left(), min(self._probe_pos.x(), c_rect.right()))
        py = max(c_rect.top(), min(self._probe_pos.y(), c_rect.bottom()))

        # Target crosshair
        painter.setPen(QPen(QColor(239, 68, 68), 1.5))
        painter.drawEllipse(QPointF(px, py), 8.0, 8.0)
        painter.drawLine(QPointF(px - 14.0, py), QPointF(px + 14.0, py))
        painter.drawLine(QPointF(px, py - 14.0), QPointF(px, py + 14.0))

        # Probe floating telemetry tooltip
        t_c, grad_m, flux_w, biot = self.get_probe_reading()
        tip_w = 170.0
        tip_h = 60.0
        tip_x = min(c_rect.right() - tip_w - 5.0, px + 12.0)
        tip_y = max(c_rect.top() + 5.0, py - tip_h - 10.0)
        tip_rect = QRectF(tip_x, tip_y, tip_w, tip_h)

        painter.setBrush(QColor(15, 23, 42, 235))
        painter.setPen(QPen(QColor(239, 68, 68), 1.2))
        painter.drawRoundedRect(tip_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Bold))
        painter.setPen(QColor(254, 202, 202))
        painter.drawText(int(tip_x + 8), int(tip_y + 16), f"Nhiệt độ T: {t_c:.1f} °C")
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor(226, 232, 240))
        painter.drawText(int(tip_x + 8), int(tip_y + 32), f"Gradient |∇T|: {grad_m:.1f} K/m")
        painter.drawText(int(tip_x + 8), int(tip_y + 48), f"Dòng nhiệt q'': {flux_w:.1f} W/m²")

    # -------------------------------------------------------------------------
    # Mouse & Touchless Air Gesture Interaction
    # -------------------------------------------------------------------------

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()

        # Check Preset buttons
        for i, (btn_rect, _) in enumerate(self._preset_button_rects()):
            if btn_rect.contains(pos):
                self.set_preset(i)
                event.accept()
                return

        # Check Action buttons
        ctrl_rect = self._controls_rect()
        action_y = ctrl_rect.top() + 62.0
        btn_probe_rect = QRectF(ctrl_rect.left() + 20, action_y, 160, 24)
        if btn_probe_rect.contains(pos):
            self._probe_active = not self._probe_active
            self.update()
            event.accept()
            return

        btn_flux_rect = QRectF(ctrl_rect.left() + 190, action_y, 160, 24)
        if btn_flux_rect.contains(pos):
            self._show_flux_vectors = not self._show_flux_vectors
            self.update()
            event.accept()
            return

        # Check Status badge click (pause / resume)
        badge_rect = QRectF(self.WIDTH - 150.0, 22.0, 122.0, 26.0)
        if badge_rect.contains(pos):
            self.toggle_pause()
            event.accept()
            return

        # Check Probe drag
        if self._probe_active:
            c_rect = self._canvas_rect()
            if c_rect.contains(pos):
                self._probe_pos = pos
                self._dragging_probe = True
                self.update()
                event.accept()
                return

        # Check Sliders
        col_w = (ctrl_rect.width() - 40.0) / 4.0
        for i, key in enumerate(["k", "h", "p", "t_inf"]):
            cx = ctrl_rect.left() + 20.0 + i * col_w
            cy = ctrl_rect.top() + 18.0
            track_y = cy + 18.0
            track_w = col_w - 25.0
            track_rect = QRectF(cx - 5.0, track_y - 8.0, track_w + 10.0, 22.0)
            if track_rect.contains(pos):
                self._dragging_slider = key
                self._update_slider_from_pos(key, pos.x(), cx, track_w)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        pos = event.pos()
        if self._dragging_probe:
            c_rect = self._canvas_rect()
            self._probe_pos = QPointF(
                max(c_rect.left(), min(pos.x(), c_rect.right())),
                max(c_rect.top(), min(pos.y(), c_rect.bottom()))
            )
            self.update()
            event.accept()
            return

        if self._dragging_slider:
            ctrl_rect = self._controls_rect()
            col_w = (ctrl_rect.width() - 40.0) / 4.0
            slider_keys = ["k", "h", "p", "t_inf"]
            if self._dragging_slider in slider_keys:
                idx = slider_keys.index(self._dragging_slider)
                cx = ctrl_rect.left() + 20.0 + idx * col_w
                track_w = col_w - 25.0
                self._update_slider_from_pos(self._dragging_slider, pos.x(), cx, track_w)
                event.accept()
                return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        self._dragging_slider = None
        self._dragging_probe = False
        super().mouseReleaseEvent(event)

    def _update_slider_from_pos(self, key: str, px: float, cx: float, track_w: float) -> None:
        norm = max(0.0, min(1.0, (px - cx) / track_w))
        if key == "k":
            self.set_thermal_conductivity(0.1 + norm * (400.0 - 0.1))
        elif key == "h":
            self.set_convection_coeff(1.0 + norm * (150.0 - 1.0))
        elif key == "p":
            self.set_heat_source_power(0.0 + norm * 150.0)
        elif key == "t_inf":
            self.set_ambient_temp(0.0 + norm * 50.0)

    # -------------------------------------------------------------------------
    # Serialization & Stamping
    # -------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize simulator state for .airteach lesson format."""
        return {
            "type": "InteractiveHeatTransferItem",
            "preset_key": self.current_preset.key,
            "k": self._k,
            "h": self._h,
            "heat_source_w": self._heat_source_w,
            "t_ambient": self._t_ambient,
            "t_source": self._t_source,
            "sim_time": self._sim_time,
            "is_paused": self._is_paused,
            "probe_active": self._probe_active,
            "probe_pos": [self._probe_pos.x(), self._probe_pos.y()],
            "pos": [self.pos().x(), self.pos().y()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], board: Optional[LessonBoard] = None) -> InteractiveHeatTransferItem:
        """Instantiate simulator from serialized dictionary."""
        item = cls(board=board)
        preset_key = data.get("preset_key", "rod_conduction")
        for i, p in enumerate(PRESETS):
            if p.key == preset_key:
                item.set_preset(i)
                break
        if "k" in data:
            item.set_thermal_conductivity(float(data["k"]))
        if "h" in data:
            item.set_convection_coeff(float(data["h"]))
        if "heat_source_w" in data:
            item.set_heat_source_power(float(data["heat_source_w"]))
        if "t_ambient" in data:
            item.set_ambient_temp(float(data["t_ambient"]))
        if "is_paused" in data:
            item._is_paused = bool(data["is_paused"])
        if "probe_active" in data:
            item._probe_active = bool(data["probe_active"])
        if "probe_pos" in data:
            coords = data["probe_pos"]
            item._probe_pos = QPointF(coords[0], coords[1])
        if "pos" in data:
            item.setPos(data["pos"][0], data["pos"][1])
        return item
