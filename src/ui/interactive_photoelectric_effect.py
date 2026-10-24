"""
AirTeach AI — Interactive Modern Physics: Photoelectric Effect & Einstein's Light Quanta.
Simulates Einstein's photoelectric equation E_k = hν - Φ = e·U_h, cathode metal work functions,
threshold frequency ν_0 = Φ/h, stopping potential U_h(ν), current-voltage I - U_AK characteristics,
and Millikan's experimental slope determination of Planck's constant h.
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


# Fundamental Physical Constants (CODATA 2018)
H_PLANCK_J_S = 6.62607015e-34      # J·s
H_PLANCK_EV_S = 4.135667696e-15    # eV·s
C_LIGHT = 299792458.0              # m/s
E_CHARGE = 1.602176634e-19         # C
M_ELECTRON_KG = 9.1093837e-31      # kg
HC_CONSTANT_EV_NM = 1239.8419      # hc in eV·nm


@dataclass(frozen=True)
class MetalTarget:
    name_vi: str
    symbol: str
    work_function_ev: float  # Work function Φ in eV
    color: QColor            # Surface metal tint


@dataclass(frozen=True)
class PhotoelectricPreset:
    code: str
    title_vi: str
    badge_label: str
    metal: MetalTarget
    wavelength_nm: float     # Incident light wavelength λ in nm
    intensity_mw: float      # Light intensity in milliwatts
    voltage_uak_v: float     # Applied voltage U_AK in Volts
    description: str


METALS: dict[str, MetalTarget] = {
    "CESIUM": MetalTarget("Xesi (Cesium)", "Cs", 2.14, QColor(250, 204, 21)),
    "POTASSIUM": MetalTarget("Kali (Potassium)", "K", 2.30, QColor(192, 132, 252)),
    "SODIUM": MetalTarget("Natri (Sodium)", "Na", 2.46, QColor(245, 158, 11)),
    "ZINC": MetalTarget("Kẽm (Zinc)", "Zn", 4.31, QColor(148, 163, 184)),
    "PLATINUM": MetalTarget("Bạch kim (Platinum)", "Pt", 6.35, QColor(226, 232, 240)),
}


PHOTOELECTRIC_PRESETS: dict[str, PhotoelectricPreset] = {
    "CESIUM_VISIBLE_LIGHT_THRESHOLD": PhotoelectricPreset(
        code="CESIUM_VISIBLE_LIGHT_THRESHOLD",
        title_vi="Catot Xesi (Cs, Φ = 2.14 eV) — Quang điện Trong Vùng Ánh sáng Nhìn thấy",
        badge_label="CESIUM VISIBLE",
        metal=METALS["CESIUM"],
        wavelength_nm=532.0,  # Green laser (2.33 eV > 2.14 eV)
        intensity_mw=40.0,
        voltage_uak_v=1.0,
        description="Xesi có công thoát thấp nhất trong các kim loại kiềm (2.14 eV, giới hạn quang điện λ₀ = 579 nm). Ánh sáng xanh lục 532 nm (2.33 eV) dễ dàng bứt electron với thế hãm U_h = 0.19 V.",
    ),
    "SODIUM_MILLIKAN_PLANCK_MEASUREMENT": PhotoelectricPreset(
        code="SODIUM_MILLIKAN_PLANCK_MEASUREMENT",
        title_vi="Thí nghiệm Lịch sử Millikan (1916) với Natri (Na) — Xác định Hằng số Planck h",
        badge_label="MILLIKAN SODIUM",
        metal=METALS["SODIUM"],
        wavelength_nm=365.0,  # UV line (3.40 eV > 2.46 eV)
        intensity_mw=50.0,
        voltage_uak_v=0.5,
        description="Robert Millikan đo thế hãm U_h theo tần số ν trên bề mặt Natri sạch trong chân không cao. Độ dốc đường thẳng h/e = 4.136×10⁻¹⁵ V·s chứng minh tuyệt đối phương trình lượng tử của Einstein.",
    ),
    "ZINC_ULTRAVIOLET_HALLWACHS": PhotoelectricPreset(
        code="ZINC_ULTRAVIOLET_HALLWACHS",
        title_vi="Thí nghiệm Hallwachs (1888) với Kẽm (Zn) — Đòi hỏi Bức xạ Tử ngoại UV",
        badge_label="ZINC UV HALLWACHS",
        metal=METALS["ZINC"],
        wavelength_nm=254.0,  # UV-C lamp (4.88 eV > 4.31 eV)
        intensity_mw=60.0,
        voltage_uak_v=0.0,
        description="Kẽm có công thoát cao (4.31 eV, λ₀ = 288 nm). Ánh sáng nhìn thấy dù cường độ cực mạnh cũng không thể giải phóng electron (I = 0). Chỉ tia tử ngoại UV mới làm tấm kẽm tích điện dương.",
    ),
    "LIGHT_INTENSITY_INDEPENDENCE": PhotoelectricPreset(
        code="LIGHT_INTENSITY_INDEPENDENCE",
        title_vi="Độc lập Cường độ Sáng — Thế hãm U_h Không Đổi khi Cường độ Tăng Gấp 5",
        badge_label="INTENSITY TEST",
        metal=METALS["SODIUM"],
        wavelength_nm=405.0,  # Violet (3.06 eV > 2.46 eV -> U_h = 0.60 V)
        intensity_mw=100.0,
        voltage_uak_v=0.8,
        description="Tăng cường độ sáng làm tăng số photon đập vào catot, do đó tăng dòng bão hòa I_s, nhưng động năng ban đầu cực đại E_k và thế hãm U_h hoàn toàn không đổi. Mâu thuẫn trực tiếp với thuyết sóng cổ điển.",
    ),
    "PLATINUM_HIGH_WORK_FUNCTION": PhotoelectricPreset(
        code="PLATINUM_HIGH_WORK_FUNCTION",
        title_vi="Bạch kim (Pt, Φ = 6.35 eV) — Công thoát Rất Cao đòi hỏi Photon Chân không UV",
        badge_label="PLATINUM VUV",
        metal=METALS["PLATINUM"],
        wavelength_nm=185.0,  # Vacuum UV (6.70 eV > 6.35 eV -> U_h = 0.35 V)
        intensity_mw=30.0,
        voltage_uak_v=0.2,
        description="Kim loại quý Bạch kim liên kết electron cực kỳ chặt chẽ (6.35 eV, λ₀ = 195 nm). Cần photon tử ngoại chân không (VUV) năng lượng cao để kích hoạt hiệu ứng quang điện ngoài.",
    ),
}


@dataclass
class Photoelectron:
    x: float
    y: float
    vx: float
    vy: float
    lifetime: float


class InteractivePhotoelectricEffectItem(QGraphicsObject):
    """
    Interactive Photoelectric Effect & Einstein's Quantum Photon Simulator.
    Simulates photoelectric emission from metal cathodes, stopping potential U_h,
    work function Φ, current-voltage I-U_AK curves, and Millikan slope h/e.
    """

    preset_changed = Signal(str)
    physics_computed = Signal(float, float, float)  # photon_ev, uh_v, current_ua
    stamp_requested = Signal(dict)

    CARD_W = 950.0
    CARD_H = 640.0

    def __init__(self, pos: tuple[float, float] = (100.0, 50.0), parent: Optional[QGraphicsItem] = None) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(150.0)

        # Simulation Preset & Parameters
        self._active_preset_code: str = "CESIUM_VISIBLE_LIGHT_THRESHOLD"
        self._preset: PhotoelectricPreset = PHOTOELECTRIC_PRESETS[self._active_preset_code]

        self._metal: MetalTarget = self._preset.metal
        self._lambda_nm: float = self._preset.wavelength_nm
        self._intensity_mw: float = self._preset.intensity_mw
        self._voltage_uak_v: float = self._preset.voltage_uak_v

        # Animated Photoelectrons in vacuum phototube
        self._electrons: list[Photoelectron] = []
        self._is_running: bool = True

        # UI & Touchless Interaction
        self._buttons: dict[str, QRectF] = {}
        self._layout_buttons()
        self._hovered_btn: Optional[str] = None
        self._is_dragging: bool = False
        self._last_mouse_pos = QPointF()

        # 30 fps animation timer
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

    # -------------------------------------------------------------------------
    # Modern Physics Properties & Einstein Quantum Equations
    # -------------------------------------------------------------------------
    @property
    def active_preset(self) -> PhotoelectricPreset:
        return self._preset

    @property
    def metal(self) -> MetalTarget:
        return self._metal

    @property
    def wavelength_nm(self) -> float:
        return self._lambda_nm

    @property
    def light_frequency_hz(self) -> float:
        """Frequency ν = c / λ in Hz."""
        return C_LIGHT / (self._lambda_nm * 1e-9)

    @property
    def photon_energy_ev(self) -> float:
        """Photon energy E = hν = hc / λ in eV."""
        return HC_CONSTANT_EV_NM / self._lambda_nm

    @property
    def work_function_ev(self) -> float:
        """Cathode metal work function Φ in eV."""
        return self._metal.work_function_ev

    @property
    def red_threshold_wavelength_nm(self) -> float:
        """Red threshold wavelength λ_0 = hc / Φ in nm."""
        return HC_CONSTANT_EV_NM / self._metal.work_function_ev

    @property
    def threshold_frequency_hz(self) -> float:
        """Threshold frequency ν_0 = Φ / h in Hz."""
        return (self._metal.work_function_ev * E_CHARGE) / H_PLANCK_J_S

    @property
    def is_photoelectric_active(self) -> bool:
        """True if incident photon energy exceeds work function (E >= Φ)."""
        return self.photon_energy_ev >= self.work_function_ev

    @property
    def max_kinetic_energy_ev(self) -> float:
        """Maximum photoelectron kinetic energy E_k = hν - Φ in eV."""
        if not self.is_photoelectric_active:
            return 0.0
        return self.photon_energy_ev - self.work_function_ev

    @property
    def stopping_potential_uh_v(self) -> float:
        """Stopping potential U_h = (hν - Φ) / e in Volts."""
        return self.max_kinetic_energy_ev

    @property
    def max_velocity_m_s(self) -> float:
        """Maximum photoelectron velocity v_max = √(2 E_k / m_e) in m/s."""
        if not self.is_photoelectric_active:
            return 0.0
        ek_joules = self.max_kinetic_energy_ev * E_CHARGE
        return math.sqrt((2.0 * ek_joules) / M_ELECTRON_KG)

    @property
    def saturation_current_ua(self) -> float:
        """Saturation photocurrent I_s proportional to light intensity."""
        if not self.is_photoelectric_active:
            return 0.0
        # ~1.2 μA per 10 mW of light above threshold
        return (self._intensity_mw / 10.0) * 1.25

    @property
    def measured_photocurrent_ua(self) -> float:
        """
        Calculates photocurrent I(U_AK) based on applied voltage U_AK:
        - If U_AK <= -U_h: I = 0 (all electrons retarded)
        - If -U_h < U_AK < 0: I = I_s * (1 + U_AK / U_h)² (retarded region)
        - If U_AK >= 0: I = I_s (saturation region)
        """
        if not self.is_photoelectric_active:
            return 0.0
        uh = self.stopping_potential_uh_v
        uak = self._voltage_uak_v
        i_sat = self.saturation_current_ua

        if uh < 1e-4:
            return i_sat if uak >= 0.0 else 0.0

        if uak <= -uh:
            return 0.0
        elif uak < 0.0:
            frac = 1.0 + (uak / uh)
            return i_sat * (frac ** 2)
        else:
            return i_sat

    def calc_photocurrent_at_voltage(self, v_test: float) -> float:
        """Calculates photocurrent I at arbitrary voltage v_test."""
        if not self.is_photoelectric_active:
            return 0.0
        uh = self.stopping_potential_uh_v
        i_sat = self.saturation_current_ua
        if uh < 1e-4:
            return i_sat if v_test >= 0.0 else 0.0
        if v_test <= -uh:
            return 0.0
        elif v_test < 0.0:
            frac = 1.0 + (v_test / uh)
            return i_sat * (frac ** 2)
        else:
            return i_sat

    # -------------------------------------------------------------------------
    # Animation & Electron Particle Dynamics
    # -------------------------------------------------------------------------
    def _on_sim_tick(self) -> None:
        if self._is_running:
            self._update_electrons()
        self.update()

    def _update_electrons(self) -> None:
        """Simulates electron motion in vacuum tube under electric field E = -U_AK / d."""
        cathode_x = 75.0
        anode_x = 385.0
        d_px = anode_x - cathode_x

        # Electric acceleration: a_x ∝ U_AK
        # Positive U_AK accelerates electrons towards anode; negative retards
        accel_x = self._voltage_uak_v * 0.18

        # Spawn new electrons if photoelectric effect is active
        if self.is_photoelectric_active and len(self._electrons) < 40:
            # Number of spawns proportional to intensity
            n_spawn = 1 if random.random() < (self._intensity_mw / 80.0) else 0
            for _ in range(n_spawn):
                # Initial kinetic velocity
                v0 = math.sqrt(self.max_kinetic_energy_ev) * 3.5 * random.uniform(0.6, 1.0)
                angle = random.uniform(-0.5, 0.5)
                self._electrons.append(
                    Photoelectron(
                        x=cathode_x + 5.0,
                        y=175.0 + random.uniform(-40.0, 40.0),
                        vx=v0 * math.cos(angle),
                        vy=v0 * math.sin(angle),
                        lifetime=0.0,
                    )
                )

        # Update existing electrons
        survivors: list[Photoelectron] = []
        for e in self._electrons:
            e.vx += accel_x
            e.x += e.vx
            e.y += e.vy
            e.lifetime += 1.0

            # Check boundaries
            # Turned back to cathode?
            if e.x < cathode_x and e.vx < 0:
                continue
            # Reached anode?
            if e.x >= anode_x:
                continue
            # Out of tube bounds?
            if e.y < 110.0 or e.y > 240.0 or e.lifetime > 80.0:
                continue
            survivors.append(e)

        self._electrons = survivors

    # -------------------------------------------------------------------------
    # Presets & Interactive Controls
    # -------------------------------------------------------------------------
    def load_preset(self, preset_code: str) -> None:
        """Loads a curriculum photoelectric effect preset."""
        if preset_code not in PHOTOELECTRIC_PRESETS:
            return
        self._active_preset_code = preset_code
        self._preset = PHOTOELECTRIC_PRESETS[preset_code]
        self._metal = self._preset.metal
        self._lambda_nm = self._preset.wavelength_nm
        self._intensity_mw = self._preset.intensity_mw
        self._voltage_uak_v = self._preset.voltage_uak_v
        self._electrons.clear()

        self.preset_changed.emit(preset_code)
        self.physics_computed.emit(
            self.photon_energy_ev, self.stopping_potential_uh_v, self.measured_photocurrent_ua
        )
        self.update()

    def set_metal(self, metal_key: str) -> None:
        """Sets cathode metal target."""
        if metal_key in METALS:
            self._metal = METALS[metal_key]
            self.physics_computed.emit(
                self.photon_energy_ev, self.stopping_potential_uh_v, self.measured_photocurrent_ua
            )
            self.update()

    def step_wavelength_nm(self, delta_nm: float) -> None:
        """Steps incident wavelength λ in nanometers."""
        self._lambda_nm = max(150.0, min(800.0, self._lambda_nm + delta_nm))
        self.physics_computed.emit(
            self.photon_energy_ev, self.stopping_potential_uh_v, self.measured_photocurrent_ua
        )
        self.update()

    def step_voltage_v(self, delta_v: float) -> None:
        """Steps applied voltage U_AK in Volts."""
        self._voltage_uak_v = max(-5.0, min(5.0, round(self._voltage_uak_v + delta_v, 2)))
        self.physics_computed.emit(
            self.photon_energy_ev, self.stopping_potential_uh_v, self.measured_photocurrent_ua
        )
        self.update()

    def step_intensity_mw(self, delta_mw: float) -> None:
        """Steps incident light power in milliwatts."""
        self._intensity_mw = max(10.0, min(200.0, self._intensity_mw + delta_mw))
        self.physics_computed.emit(
            self.photon_energy_ev, self.stopping_potential_uh_v, self.measured_photocurrent_ua
        )
        self.update()

    def toggle_animation(self) -> None:
        """Toggles particle animation."""
        self._is_running = not self._is_running
        self.update()

    # -------------------------------------------------------------------------
    # UI Buttons & Touchless Air Gestures
    # -------------------------------------------------------------------------
    def _layout_buttons(self) -> None:
        """Initializes touchless air gesture button bounds."""
        bar_y = 574.0
        self._buttons["btn_preset"] = QRectF(20.0, bar_y, 160.0, 34.0)
        self._buttons["btn_w_down"] = QRectF(188.0, bar_y, 45.0, 34.0)
        self._buttons["btn_w_up"] = QRectF(238.0, bar_y, 45.0, 34.0)
        self._buttons["btn_v_down"] = QRectF(290.0, bar_y, 45.0, 34.0)
        self._buttons["btn_v_up"] = QRectF(340.0, bar_y, 45.0, 34.0)
        self._buttons["btn_int_down"] = QRectF(392.0, bar_y, 45.0, 34.0)
        self._buttons["btn_int_up"] = QRectF(442.0, bar_y, 45.0, 34.0)
        self._buttons["btn_m_cs"] = QRectF(494.0, bar_y, 42.0, 34.0)
        self._buttons["btn_m_na"] = QRectF(540.0, bar_y, 42.0, 34.0)
        self._buttons["btn_m_zn"] = QRectF(586.0, bar_y, 42.0, 34.0)
        self._buttons["btn_m_pt"] = QRectF(632.0, bar_y, 42.0, 34.0)
        self._buttons["btn_p_intensity"] = QRectF(680.0, bar_y, 82.0, 34.0)
        self._buttons["btn_play"] = QRectF(768.0, bar_y, 70.0, 34.0)
        self._buttons["btn_stamp"] = QRectF(844.0, bar_y, 86.0, 34.0)

    def _handle_button_click(self, key: str) -> None:
        preset_keys = list(PHOTOELECTRIC_PRESETS.keys())
        if key == "btn_preset":
            curr_idx = preset_keys.index(self._active_preset_code)
            next_idx = (curr_idx + 1) % len(preset_keys)
            self.load_preset(preset_keys[next_idx])
        elif key == "btn_w_down":
            self.step_wavelength_nm(-20.0)
        elif key == "btn_w_up":
            self.step_wavelength_nm(20.0)
        elif key == "btn_v_down":
            self.step_voltage_v(-0.2)
        elif key == "btn_v_up":
            self.step_voltage_v(0.2)
        elif key == "btn_int_down":
            self.step_intensity_mw(-20.0)
        elif key == "btn_int_up":
            self.step_intensity_mw(20.0)
        elif key == "btn_m_cs":
            self.set_metal("CESIUM")
        elif key == "btn_m_na":
            self.set_metal("SODIUM")
        elif key == "btn_m_zn":
            self.set_metal("ZINC")
        elif key == "btn_m_pt":
            self.set_metal("PLATINUM")
        elif key == "btn_p_intensity":
            self.load_preset("LIGHT_INTENSITY_INDEPENDENCE")
        elif key == "btn_play":
            self.toggle_animation()
        elif key == "btn_stamp":
            self.stamp_diagram()
        self.update()

    def handle_air_pinch(self, pos: QPointF) -> bool:
        """Handles touchless air pinch gesture targeting interactive UI buttons."""
        local_pos = self.mapFromScene(pos)
        for key, rect in self._buttons.items():
            if rect.contains(local_pos):
                self._handle_button_click(key)
                return True
        return False

    def handle_air_drag(self, delta: QPointF) -> None:
        """Translates item across whiteboard when dragged by air pinch."""
        self.setPos(self.pos() + delta)

    def stamp_diagram(self) -> None:
        """Emits signal requesting permanent vector stamping of current diagram."""
        stamp_data = {
            "center": (self.pos().x() + self.CARD_W / 2.0, self.pos().y() + self.CARD_H / 2.0),
            "width": 680.0,
            "height": 380.0,
            "preset_code": self._active_preset_code,
            "metal_name": self._metal.name_vi,
            "work_function_ev": self.work_function_ev,
            "wavelength_nm": self._lambda_nm,
            "photon_energy_ev": self.photon_energy_ev,
            "stopping_potential_v": self.stopping_potential_uh_v,
            "voltage_uak_v": self._voltage_uak_v,
            "photocurrent_ua": self.measured_photocurrent_ua,
            "red_threshold_nm": self.red_threshold_wavelength_nm,
        }
        self.stamp_requested.emit(stamp_data)

    # -------------------------------------------------------------------------
    # QGraphicsItem Rendering
    # -------------------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.CARD_W, self.CARD_H)

    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_card_background(painter)
        self._paint_header(painter)
        self._paint_phototube_panel(painter)
        self._paint_iv_curve_panel(painter)
        self._paint_millikan_curve_panel(painter)
        self._paint_metrics_panel(painter)
        self._paint_controls_bar(painter)

    def _paint_card_background(self, painter: QPainter) -> None:
        bg_rect = self.boundingRect()
        grad = QLinearGradient(0.0, 0.0, 0.0, self.CARD_H)
        grad.setColorAt(0.0, QColor(9, 14, 25, 252))
        grad.setColorAt(1.0, QColor(5, 8, 16, 255))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(56, 189, 248, 140), 1.8))
        painter.drawRoundedRect(bg_rect, 14.0, 14.0)

    def _paint_header(self, painter: QPainter) -> None:
        title_font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(248, 250, 252)))
        painter.drawText(22, 30, f"HIỆU ỨNG QUANG ĐIỆN EINSTEIN & THUYẾT LƯỢNG TỬ ÁNH SÁNG: {self._preset.badge_label}")

        sub_font = QFont("Segoe UI", 8)
        painter.setFont(sub_font)
        painter.setPen(QPen(QColor(148, 163, 184)))
        status_txt = "PHÁT XẠ ELECTRON (E ≥ Φ)" if self.is_photoelectric_active else "KHÔNG PHÁT XẠ (E < Φ, Dưới Ngưỡng)"
        status_col = "#22c55e" if self.is_photoelectric_active else "#ef4444"
        painter.drawText(
            22,
            48,
            f"Catot: {self._metal.name_vi} (Φ = {self.work_function_ev:.2f} eV) | λ = {self._lambda_nm:.1f} nm (E = {self.photon_energy_ev:.2f} eV) | U_h = {self.stopping_potential_uh_v:.2f} V | U_AK = {self._voltage_uak_v:+.2f} V | I = {self.measured_photocurrent_ua:.2f} μA | {status_txt}",
        )

    def _paint_phototube_panel(self, painter: QPainter) -> None:
        """Renders evacuated quartz glass bulb, cathode, anode ring, light rays, and electrons."""
        tx = 20.0
        ty = 60.0
        tw = 440.0
        th = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(tx, ty, tw, th, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(tx + 12.0, ty + 20.0, "TẾ BÀO QUANG ĐIỆN CHÂN KHÔNG & QUỸ ĐẠO ELECTRON:")

        # Quartz glass bulb
        bulb_rect = QRectF(tx + 40.0, ty + 40.0, tw - 80.0, th - 70.0)
        painter.setBrush(QBrush(QColor(15, 23, 42, 140)))
        painter.setPen(QPen(QColor(56, 189, 248, 180), 1.5))
        painter.drawRoundedRect(bulb_rect, 24.0, 24.0)

        # Cathode Plate (Left)
        cat_x = bulb_rect.left() + 25.0
        cat_y = bulb_rect.center().y() - 45.0
        cat_w = 12.0
        cat_h = 90.0

        painter.setBrush(QBrush(self._metal.color))
        painter.setPen(QPen(QColor(248, 250, 252), 1.4))
        painter.drawRoundedRect(cat_x, cat_y, cat_w, cat_h, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(226, 232, 240)))
        painter.drawText(cat_x - 18.0, cat_y - 6.0, f"Catot ({self._metal.symbol})")

        # Anode Collector Ring (Right)
        an_x = bulb_rect.right() - 35.0
        an_y = cat_y
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(148, 163, 184), 2.5))
        painter.drawEllipse(an_x, an_y + 10.0, 16.0, cat_h - 20.0)
        painter.drawText(an_x - 10.0, cat_y - 6.0, "Anot (A)")

        # Incident Photons (colored by wavelength)
        light_color = self._get_wavelength_color(self._lambda_nm)
        painter.setPen(QPen(light_color, 2.0, Qt.PenStyle.DashLine))
        for offset in [-25.0, 0.0, 25.0]:
            p_ray = QPainterPath()
            p_ray.moveTo(tx + 5.0, cat_y + 45.0 + offset - 40.0)
            p_ray.lineTo(cat_x, cat_y + 45.0 + offset)
            painter.drawPath(p_ray)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(light_color))
        painter.drawText(tx + 12.0, cat_y + 8.0, f"hν = {self.photon_energy_ev:.2f} eV")

        # Draw Flying Photoelectrons
        painter.setBrush(QBrush(QColor(56, 189, 248)))
        painter.setPen(Qt.PenStyle.NoPen)
        for e in self._electrons:
            painter.drawEllipse(QPointF(e.x, e.y), 3.0, 3.0)

        # Microammeter display at bottom right
        meter_rect = QRectF(tx + tw - 120.0, ty + th - 40.0, 105.0, 26.0)
        painter.setBrush(QBrush(QColor(9, 14, 25)))
        painter.setPen(QPen(QColor(56, 189, 248), 1.0))
        painter.drawRoundedRect(meter_rect, 4.0, 4.0)

        curr_ua = self.measured_photocurrent_ua
        curr_col = QColor(34, 197, 94) if curr_ua > 0 else QColor(148, 163, 184)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(curr_col))
        painter.drawText(meter_rect, Qt.AlignmentFlag.AlignCenter, f"μA: {curr_ua:.2f} μA")

        # Electric Field arrow between plates
        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QPen(QColor(250, 204, 21)))
        ef_dir = "→ E_điện" if self._voltage_uak_v < 0 else "← E_điện"
        painter.drawText(cat_x + 60.0, ty + th - 22.0, f"U_AK = {self._voltage_uak_v:+.2f} V ({ef_dir})")

    def _paint_iv_curve_panel(self, painter: QPainter) -> None:
        """Plots the experimental current-voltage I - U_AK characteristic curve."""
        sx = 480.0
        sy = 60.0
        sw = 450.0
        sh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(sx, sy, sw, sh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(sx + 12.0, sy + 20.0, "ĐƯỜNG ĐẶC TRƯNG VÔN-AMPE I = f(U_AK):")

        # Axes
        px = sx + 45.0
        py = sy + 35.0
        pw = sw - 65.0
        ph = sh - 65.0

        # X-axis spans from -3.0 V to +3.0 V
        # Center voltage 0 V is at px + pw / 2
        v_min = -3.0
        v_max = 3.0
        v_span = v_max - v_min
        zero_x = px + (-v_min / v_span) * pw

        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawLine(px, py + ph, px + pw, py + ph)  # X-axis (voltage)
        painter.drawLine(zero_x, py, zero_x, py + ph)    # Y-axis (current at U=0)

        # Plot I - U_AK curve
        iv_path = QPainterPath()
        steps = int(pw)
        max_i = max(0.1, self.saturation_current_ua * 1.25)

        for s in range(steps + 1):
            v_curr = v_min + (s / float(steps)) * v_span
            i_curr = self.calc_photocurrent_at_voltage(v_curr)
            gx = px + s
            gy = py + ph - (i_curr / max_i) * ph
            if s == 0:
                iv_path.moveTo(gx, gy)
            else:
                iv_path.lineTo(gx, gy)

        # Fill under curve
        fill_path = QPainterPath(iv_path)
        fill_path.lineTo(px + pw, py + ph)
        fill_path.lineTo(px, py + ph)
        fill_path.closeSubpath()
        grad = QLinearGradient(px, py, px, py + ph)
        grad.setColorAt(0.0, QColor(56, 189, 248, 120))
        grad.setColorAt(1.0, QColor(56, 189, 248, 10))
        painter.setBrush(QBrush(grad))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(fill_path)

        # Stroke
        painter.setPen(QPen(QColor(56, 189, 248), 2.0))
        painter.drawPath(iv_path)

        # Stopping potential marker (-U_h)
        uh = self.stopping_potential_uh_v
        if self.is_photoelectric_active and uh > 0:
            uh_x = px + ((-uh - v_min) / v_span) * pw
            painter.setPen(QPen(QColor(244, 63, 94), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(uh_x, py, uh_x, py + ph)
            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(244, 63, 94)))
            painter.drawText(uh_x - 14.0, py + ph + 13.0, f"-U_h ({uh:.2f}V)")

        # Saturation current marker I_s
        is_y = py + ph - (self.saturation_current_ua / max_i) * ph
        painter.setPen(QPen(QColor(250, 204, 21), 1.0, Qt.PenStyle.DotLine))
        painter.drawLine(zero_x, is_y, px + pw, is_y)
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(250, 204, 21)))
        painter.drawText(px + pw - 55.0, is_y - 4.0, f"I_s = {self.saturation_current_ua:.2f}μA")

        # Current Operating Point
        curr_v = max(v_min, min(v_max, self._voltage_uak_v))
        curr_x = px + ((curr_v - v_min) / v_span) * pw
        curr_i = self.measured_photocurrent_ua
        curr_y = py + ph - (curr_i / max_i) * ph

        painter.setBrush(QBrush(QColor(34, 197, 94)))
        painter.setPen(QPen(QColor(255, 255, 255), 1.4))
        painter.drawEllipse(QPointF(curr_x, curr_y), 5.0, 5.0)

        # X-axis Voltage labels
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(px, py + ph + 14.0, "-3V")
        painter.drawText(zero_x - 6.0, py + ph + 14.0, "0V")
        painter.drawText(px + pw - 18.0, py + ph + 14.0, "+3V")

    def _paint_millikan_curve_panel(self, painter: QPainter) -> None:
        """Plots the linear Millikan curve U_h vs frequency ν, verifying h/e slope."""
        mx = 20.0
        my = 315.0
        mw = 440.0
        mh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(mx, my, mw, mh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(mx + 12.0, my + 20.0, "THỰC NGHIỆM MILLIKAN: U_h = (h/e)ν - Φ/e:")

        # Plot axes
        px = mx + 45.0
        py = my + 35.0
        pw = mw - 65.0
        ph = mh - 65.0

        painter.setPen(QPen(QColor(51, 65, 85), 1.2))
        painter.drawLine(px, py + ph, px + pw, py + ph)  # X-axis: frequency ν (0 to 1.5e15 Hz)
        painter.drawLine(px, py, px, py + ph)            # Y-axis: stopping potential U_h (0 to 4 V)

        # Frequency range: 0.0 to 1.6e15 Hz
        nu_max = 1.6e15
        uh_max = 4.0

        # Plot Millikan line: U_h = (h/e) * nu - Phi
        nu_0 = self.threshold_frequency_hz
        phi = self.work_function_ev

        # Point at nu_0: U_h = 0
        nu0_x = px + (nu_0 / nu_max) * pw
        # Point at nu_max
        uh_end = (H_PLANCK_EV_S * nu_max) - phi
        end_x = px + pw
        end_y = py + ph - max(0.0, min(1.0, uh_end / uh_max)) * ph

        if nu0_x < px + pw:
            painter.setPen(QPen(QColor(245, 158, 11), 2.2))
            painter.drawLine(nu0_x, py + ph, end_x, end_y)

            # X-intercept marker ν_0
            painter.setBrush(QBrush(QColor(244, 63, 94)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(nu0_x, py + ph), 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(244, 63, 94)))
            painter.drawText(nu0_x - 14.0, py + ph + 13.0, f"ν_0 ({self.red_threshold_wavelength_nm:.0f}nm)")

        # Slope label h/e
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(245, 158, 11)))
        painter.drawText(px + pw * 0.45, py + 25.0, "Độ dốc k = h/e = 4.136 × 10⁻¹⁵ V·s")

        # Current operating frequency point
        nu_curr = self.light_frequency_hz
        uh_curr = self.stopping_potential_uh_v
        pt_x = px + (nu_curr / nu_max) * pw
        pt_y = py + ph - (uh_curr / uh_max) * ph

        if px <= pt_x <= px + pw:
            painter.setBrush(QBrush(QColor(34, 197, 94)))
            painter.setPen(QPen(QColor(255, 255, 255), 1.4))
            painter.drawEllipse(QPointF(pt_x, pt_y), 5.0, 5.0)

            painter.setFont(QFont("Segoe UI", 6, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(34, 197, 94)))
            painter.drawText(pt_x + 6.0, pt_y + 4.0, f"ν = {nu_curr/1e14:.2f} × 10¹⁴ Hz")

        # X-axis labels
        painter.setFont(QFont("Segoe UI", 6))
        painter.setPen(QPen(QColor(148, 163, 184)))
        painter.drawText(px, py + ph + 14.0, "0")
        painter.drawText(px + pw / 2.0 - 15.0, py + ph + 14.0, "0.8 × 10¹⁵ Hz")
        painter.drawText(px + pw - 30.0, py + ph + 14.0, "1.6 × 10¹⁵ Hz")

    def _paint_metrics_panel(self, painter: QPainter) -> None:
        """Displays rigorous analytical quantum mechanics equations and physical metrics."""
        mx = 480.0
        my = 315.0
        mw = 450.0
        mh = 240.0

        painter.setBrush(QBrush(QColor(3, 7, 15)))
        painter.setPen(QPen(QColor(30, 41, 59), 1.2))
        painter.drawRoundedRect(mx, my, mw, mh, 8.0)

        # Header
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QPen(QColor(56, 189, 248)))
        painter.drawText(mx + 12.0, my + 20.0, "PHƯƠNG TRÌNH LƯỢNG TỬ & ĐẠI LƯỢNG VẬT LÝ:")

        metrics = [
            ("Năng lượng Photon (Photon Energy):", f"E = hc/λ = {self.photon_energy_ev:.3f} eV ({self.photon_energy_ev * E_CHARGE * 1e19:.2f} × 10⁻¹⁹ J)"),
            ("Công thoát Kim loại (Work Function):", f"Φ = {self.work_function_ev:.2f} eV ({self._metal.name_vi})"),
            ("Giới hạn Quang điện (Red Threshold):", f"λ₀ = hc/Φ = {self.red_threshold_wavelength_nm:.1f} nm (ν₀ = {self.threshold_frequency_hz/1e14:.2f} × 10¹⁴ Hz)"),
            ("Động năng Ban đầu Cực đại (Max E_k):", f"E_k = hν - Φ = {self.max_kinetic_energy_ev:.3f} eV"),
            ("Vận tốc Ban đầu Cực đại (v_max):", f"v_max = √(2E_k/m_e) = {self.max_velocity_m_s/1e5:.2f} × 10⁵ m/s"),
            ("Điện thế Hãm (Stopping Potential):", f"U_h = E_k / e = {self.stopping_potential_uh_v:.3f} V"),
            ("Điện thế Đặt vào Catot-Anot (U_AK):", f"U_AK = {self._voltage_uak_v:+.2f} V"),
            ("Cường độ Dòng Quang điện Hiện tại:", f"I = {self.measured_photocurrent_ua:.2f} μA (Bão hòa I_s = {self.saturation_current_ua:.2f} μA)"),
        ]

        row_y = my + 44.0
        line_h = 21.0
        for name, val in metrics:
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QPen(QColor(148, 163, 184)))
            painter.drawText(mx + 14.0, row_y, name)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QPen(QColor(226, 232, 240)))
            painter.drawText(mx + 225.0, row_y, val)
            row_y += line_h

    def _paint_controls_bar(self, painter: QPainter) -> None:
        """Renders interactive touchless air gesture buttons."""
        buttons_info = [
            ("btn_preset", f"Preset: {self._preset.badge_label}", QColor(14, 165, 233)),
            ("btn_w_down", "λ -", QColor(51, 65, 85)),
            ("btn_w_up", "λ +", QColor(51, 65, 85)),
            ("btn_v_down", "U -", QColor(51, 65, 85)),
            ("btn_v_up", "U +", QColor(51, 65, 85)),
            ("btn_int_down", "P -", QColor(51, 65, 85)),
            ("btn_int_up", "P +", QColor(51, 65, 85)),
            ("btn_m_cs", "Cs", QColor(30, 41, 59)),
            ("btn_m_na", "Na", QColor(30, 41, 59)),
            ("btn_m_zn", "Zn", QColor(30, 41, 59)),
            ("btn_m_pt", "Pt", QColor(30, 41, 59)),
            ("btn_p_intensity", "Cường độ P", QColor(30, 41, 59)),
            ("btn_play", "Dừng" if self._is_running else "Chạy", QColor(245, 158, 11)),
            ("btn_stamp", "Ghim", QColor(16, 185, 129)),
        ]

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        for key, text, color in buttons_info:
            rect = self._buttons[key]
            hovered = self._hovered_btn == key
            btn_col = color.lighter(125) if hovered else color

            painter.setBrush(QBrush(btn_col))
            painter.setPen(QPen(QColor(255, 255, 255, 160) if hovered else QColor(0, 0, 0, 80), 1.0))
            painter.drawRoundedRect(rect, 6.0, 6.0)

            painter.setPen(QPen(QColor(255, 255, 255)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

    @staticmethod
    def _get_wavelength_color(lambda_nm: float) -> QColor:
        """Converts optical wavelength λ (nm) to representative RGB color."""
        if lambda_nm < 380:
            return QColor(168, 85, 247)   # Ultraviolet (false purple)
        elif lambda_nm < 440:
            return QColor(138, 43, 226)   # Violet
        elif lambda_nm < 490:
            return QColor(0, 191, 255)    # Blue
        elif lambda_nm < 560:
            return QColor(34, 197, 94)    # Green
        elif lambda_nm < 595:
            return QColor(234, 179, 8)    # Yellow
        elif lambda_nm < 630:
            return QColor(249, 115, 22)   # Orange
        elif lambda_nm <= 750:
            return QColor(239, 68, 68)    # Red
        else:
            return QColor(148, 163, 184)  # Infrared
