"""
AirTeach AI - Interactive Cryptography, RSA Public-Key & Modular Arithmetic Circle Simulator
==============================================================================================
Provides an authentic discrete mathematics and cryptography simulation:
  - RSA Public-Key Cryptography (Hệ mật mã khóa công khai RSA - Rivest, Shamir, Adleman):
      1. Primes p, q => Modulus n = p * q
      2. Euler's totient function: φ(n) = (p - 1) * (q - 1)
      3. Public exponent e: 1 < e < φ(n) such that gcd(e, φ(n)) = 1
      4. Private exponent d: Extended Euclidean Algorithm
         e * d ≡ 1 (mod φ(n)) <=> d = e^(-1) mod φ(n)
      5. Encryption: C = M^e mod n (Bản mã)
      6. Decryption: M = C^d mod n (Bản rõ)
      7. Mathematical Proof: Euler's Totient Theorem
         M^(e*d) ≡ M^(k*φ(n)+1) ≡ M (mod n)
  - Diffie-Hellman Key Exchange (Trao đổi khóa Diffie-Hellman):
      Public parameters: prime modulus p, primitive root generator g.
      Alice chooses secret a => A = g^a mod p
      Bob chooses secret b => B = g^b mod p
      Shared secret: K_Alice = B^a mod p == K_Bob = A^b mod p == g^(a*b) mod p
  - Modular Arithmetic Clock & Chord Web Visualizer:
      Circle of n points (0 to n-1).
      Connecting chords i -> (i * k) mod n produces stunning fractal envelope geometry:
        - k = 2: Cardioid (Hình tim)
        - k = 3: Nephroid (Hình thận)
        - k = e: RSA modular permutation chord web!
  - Touchless air gestures and vector diagram stamping via InformaticsTools.create_rsa_crypto_diagram.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass
import logging
import math
from typing import Optional, TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsObject,
    QGraphicsSceneMouseEvent,
    QStyleOptionGraphicsItem,
    QWidget,
)

from src.ui.stem_tools import InformaticsTools

if TYPE_CHECKING:
    from src.ui.lesson_board import LessonBoard

logger = logging.getLogger(__name__)


# Standard Math Helpers for Modular Arithmetic & Number Theory

def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean Algorithm: returns (gcd, x, y) such that a*x + b*y = gcd."""
    if b == 0:
        return a, 1, 0
    gcd_val, x1, y1 = extended_gcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return gcd_val, x, y


def modular_inverse(e: int, phi: int) -> Optional[int]:
    """Computes modular inverse d such that (e * d) % phi == 1."""
    gcd_val, x, _ = extended_gcd(e, phi)
    if gcd_val != 1:
        return None  # No modular inverse if not coprime
    return (x % phi + phi) % phi


def fast_modular_pow(base: int, exp: int, mod: int) -> int:
    """Computes (base^exp) % mod using binary exponentiation."""
    return pow(base, exp, mod)


# RSA Preset Pairs: (name_vi, p, q, e)
RSA_PRESETS: dict[str, tuple[str, int, int, int]] = {
    "PAIR_7_11": ("Cặp Nguyên Tố Nhỏ (p=7, q=11, n=77)", 7, 11, 7),
    "PAIR_11_13": ("Cặp Cổ Điển (p=11, q=13, n=143)", 11, 13, 7),
    "PAIR_13_17": ("Cặp Trung Bình (p=13, q=17, n=221)", 13, 17, 5),
    "PAIR_17_19": ("Cặp Mở Rộng (p=17, q=19, n=323)", 17, 19, 11),
}


class InteractiveCryptographyItem(QGraphicsObject):
    """
    Interactive Cryptography, RSA Public-Key & Modular Arithmetic Circle Simulator item.
    """

    crypto_updated = Signal(int, int, int, int)  # (m, c, e, d)
    mode_changed = Signal(str)                   # ("RSA", "DIFFIE_HELLMAN", "MODULAR_CHORDS")
    stamp_requested = Signal()

    def __init__(
        self,
        pos: tuple[float, float] = (450.0, 240.0),
        width: float = 780.0,
        height: float = 500.0,
        preset: str = "PAIR_7_11",
        mode: str = "RSA",
        parent: Optional[QGraphicsItem] = None,
    ) -> None:
        super().__init__(parent)
        self.setPos(pos[0], pos[1])
        self.setZValue(95.0)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)

        self._width = float(width)
        self._height = float(height)

        # Operational Mode: RSA, DIFFIE_HELLMAN, MODULAR_CHORDS
        self._mode = mode.upper() if mode.upper() in ("RSA", "DIFFIE_HELLMAN", "MODULAR_CHORDS") else "RSA"

        # Active RSA Parameters
        self._preset = preset.upper() if preset.upper() in RSA_PRESETS else "PAIR_7_11"
        spec = RSA_PRESETS[self._preset]
        self._p = spec[1]
        self._q = spec[2]
        self._e = spec[3]
        self._recompute_rsa_keys()

        # Messages
        self._plaintext_m = 9             # Plaintext integer M
        self._ciphertext_c = self.encrypt(self._plaintext_m)

        # Modular Chord Web Multiplier (for MODULAR_CHORDS mode)
        self._multiplier_k = 2.0          # Cardioid (k=2), Nephroid (k=3)
        self._chord_anim_active = False

        # Diffie-Hellman Parameters
        self._dh_p = 23                   # Prime modulus
        self._dh_g = 5                    # Primitive root generator
        self._dh_alice_secret = 6         # a
        self._dh_bob_secret = 15          # b

        # Animation timer (30 FPS) for chord spinning
        self._timer = QTimer()
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._on_sim_tick)
        self._timer.start()

        # Layout Geometry Constants
        self._dial_x = 24.0
        self._dial_y = 52.0
        self._dial_w = 280.0
        self._dial_h = 320.0

        self._work_x = 320.0
        self._work_y = 52.0
        self._work_w = self._width - self._work_x - 24.0  # 436.0 px
        self._work_h = 320.0

        # Window Controls
        self._rect_btn_close = QRectF(self._width - 34.0, 10.0, 24.0, 24.0)

        # Control Deck Rectangles (y = 388 to 480)
        # Row 1: Mode Selectors (y = 388, h = 26)
        self._rect_btn_mode_rsa = QRectF(24.0, 388.0, 220.0, 26.0)
        self._rect_btn_mode_dh = QRectF(254.0, 388.0, 240.0, 26.0)
        self._rect_btn_mode_chords = QRectF(504.0, 388.0, 252.0, 26.0)

        # Row 2: Parameters & Action Steppers (y = 422, h = 26)
        # Plaintext M stepper [-] [+]
        self._rect_btn_m_dec = QRectF(24.0, 422.0, 28.0, 26.0)
        self._rect_btn_m_inc = QRectF(118.0, 422.0, 28.0, 26.0)
        # Preset cycler
        self._rect_btn_cycle_primes = QRectF(154.0, 422.0, 185.0, 26.0)
        # Actions
        self._rect_btn_encrypt = QRectF(348.0, 422.0, 95.0, 26.0)
        self._rect_btn_decrypt = QRectF(450.0, 422.0, 95.0, 26.0)
        self._rect_btn_reset = QRectF(554.0, 422.0, 95.0, 26.0)
        self._rect_btn_stamp = QRectF(658.0, 422.0, 98.0, 26.0)

    # ─── Public Properties & Cryptographic Solvers ───────────────────────────

    @property
    def prime_p(self) -> int:
        return self._p

    @property
    def prime_q(self) -> int:
        return self._q

    @property
    def modulus_n(self) -> int:
        """Modulus n = p * q."""
        return self._p * self._q

    @property
    def euler_phi(self) -> int:
        """Euler's totient function φ(n) = (p - 1) * (q - 1)."""
        return (self._p - 1) * (self._q - 1)

    @property
    def public_exponent_e(self) -> int:
        return self._e

    @property
    def private_exponent_d(self) -> int:
        return self._d

    @property
    def plaintext_m(self) -> int:
        return self._plaintext_m

    @property
    def ciphertext_c(self) -> int:
        return self._ciphertext_c

    @property
    def mode(self) -> str:
        return self._mode

    def _recompute_rsa_keys(self) -> None:
        """Computes private exponent d = e^(-1) mod φ(n)."""
        phi = self.euler_phi
        d_val = modular_inverse(self._e, phi)
        if d_val is None:
            # Fallback to finding coprime e if not coprime
            for cand_e in (3, 5, 7, 11, 13, 17):
                if math.gcd(cand_e, phi) == 1:
                    self._e = cand_e
                    d_val = modular_inverse(self._e, phi)
                    break
        self._d = d_val if d_val is not None else 1

    def encrypt(self, message: int) -> int:
        """RSA Encryption: C = M^e mod n."""
        n = self.modulus_n
        m_clamped = max(0, min(n - 1, int(message)))
        return fast_modular_pow(m_clamped, self._e, n)

    def decrypt(self, ciphertext: int) -> int:
        """RSA Decryption: M = C^d mod n."""
        n = self.modulus_n
        c_clamped = max(0, min(n - 1, int(ciphertext)))
        return fast_modular_pow(c_clamped, self._d, n)

    # ─── Controls & Setters ──────────────────────────────────────────────────

    def set_mode(self, mode_name: str) -> None:
        """Switches active visualization mode."""
        name_upper = mode_name.upper()
        if name_upper in ("RSA", "DIFFIE_HELLMAN", "MODULAR_CHORDS") and name_upper != self._mode:
            self._mode = name_upper
            self.mode_changed.emit(self._mode)
            self.update()

    def set_preset(self, preset_name: str) -> None:
        """Sets prime pair preset and re-encrypts."""
        name_upper = preset_name.upper()
        if name_upper in RSA_PRESETS:
            self._preset = name_upper
            spec = RSA_PRESETS[self._preset]
            self._p = spec[1]
            self._q = spec[2]
            self._e = spec[3]
            self._recompute_rsa_keys()
            self._plaintext_m = min(self._plaintext_m, self.modulus_n - 1)
            self._ciphertext_c = self.encrypt(self._plaintext_m)
            self.crypto_updated.emit(self._plaintext_m, self._ciphertext_c, self._e, self._d)
            self.update()

    def set_plaintext(self, message: int) -> None:
        """Sets plaintext integer M [1 to n-1] and encrypts."""
        n = self.modulus_n
        self._plaintext_m = max(1, min(n - 1, int(message)))
        self._ciphertext_c = self.encrypt(self._plaintext_m)
        self.crypto_updated.emit(self._plaintext_m, self._ciphertext_c, self._e, self._d)
        self.update()

    def cycle_preset(self) -> None:
        """Cycles through RSA prime presets."""
        keys = list(RSA_PRESETS.keys())
        idx = keys.index(self._preset)
        next_key = keys[(idx + 1) % len(keys)]
        self.set_preset(next_key)

    def reset_simulation(self) -> None:
        """Resets to default message and keys."""
        self._plaintext_m = 9
        self._ciphertext_c = self.encrypt(self._plaintext_m)
        self._multiplier_k = 2.0
        self.crypto_updated.emit(self._plaintext_m, self._ciphertext_c, self._e, self._d)
        self.update()

    def _on_sim_tick(self) -> None:
        """Simulation tick callback."""
        if self._mode == "MODULAR_CHORDS" and self._chord_anim_active:
            self._multiplier_k = (self._multiplier_k + 0.02) % 24.0
            self.update()

    # ─── Stamping to LessonBoard ─────────────────────────────────────────────

    def stamp_to_canvas(self, board: Optional[LessonBoard] = None) -> list[QGraphicsItem]:
        """
        Stamps RSA public-key cryptographic workflow and modular circle vector diagram onto LessonBoard.
        """
        cx = self.x() + self._width / 2.0
        cy = self.y() + self._height / 2.0

        items = InformaticsTools.create_rsa_crypto_diagram(
            center=(cx, cy),
            width=440.0,
            height=260.0,
            prime_p=self._p,
            prime_q=self._q,
            public_e=self._e,
            private_d=self._d,
            plaintext_m=self._plaintext_m,
            ciphertext_c=self._ciphertext_c,
        )

        target_board = board
        if target_board is None:
            scene = self.scene()
            if scene is not None and hasattr(scene, "add_item_with_undo"):
                target_board = scene  # type: ignore

        if target_board is not None and hasattr(target_board, "add_item_with_undo"):
            for item in items:
                target_board.add_item_with_undo(item)  # type: ignore
        elif self.scene() is not None:
            for item in items:
                self.scene().addItem(item)

        self.stamp_requested.emit()
        logger.info(
            "Stamped RSA Cryptography diagram to LessonBoard (n=%d, e=%d, d=%d, M=%d, C=%d)",
            self.modulus_n, self._e, self._d, self._plaintext_m, self._ciphertext_c
        )
        return items

    # ─── Air Gesture & Interaction Handlers ──────────────────────────────────

    def handle_air_pinch(self, scene_pos: QPointF, prev_pos: Optional[QPointF] = None) -> bool:
        """
        Handles touchless air pinch gesture.
        If pinching the modular dial area, adjusts message M or multiplier k.
        If pinching title area and prev_pos is provided, translates the widget.
        """
        local_pos = self.mapFromScene(scene_pos)

        # Check modular circle area
        dial_center = QPointF(self._dial_x + self._dial_w / 2.0, self._dial_y + self._dial_h / 2.0)
        dx = local_pos.x() - dial_center.x()
        dy = local_pos.y() - dial_center.y()
        dist = math.hypot(dx, dy)

        if dist <= 120.0:
            # Angle around dial maps to message M
            angle = math.atan2(dy, dx) + math.pi / 2.0
            if angle < 0.0:
                angle += 2.0 * math.pi
            n = self.modulus_n
            val = int(round((angle / (2.0 * math.pi)) * n))
            val = max(1, min(n - 1, val))
            self.set_plaintext(val)
            return True

        # Window drag via air pinch if prev_pos is provided
        if prev_pos is not None:
            delta = scene_pos - prev_pos
            self.setPos(self.pos() + delta)
            return True

        return False

    def handle_air_tap(self, scene_pos: QPointF, board: Optional[LessonBoard] = None) -> bool:
        """Handles touchless air tap gesture on buttons."""
        local_pos = self.mapFromScene(scene_pos)

        if self._rect_btn_close.contains(local_pos):
            self.close()
            return True
        elif self._rect_btn_mode_rsa.contains(local_pos):
            self.set_mode("RSA")
            return True
        elif self._rect_btn_mode_dh.contains(local_pos):
            self.set_mode("DIFFIE_HELLMAN")
            return True
        elif self._rect_btn_mode_chords.contains(local_pos):
            self.set_mode("MODULAR_CHORDS")
            return True
        elif self._rect_btn_m_dec.contains(local_pos):
            self.set_plaintext(self._plaintext_m - 1)
            return True
        elif self._rect_btn_m_inc.contains(local_pos):
            self.set_plaintext(self._plaintext_m + 1)
            return True
        elif self._rect_btn_cycle_primes.contains(local_pos):
            self.cycle_preset()
            return True
        elif self._rect_btn_encrypt.contains(local_pos):
            self._ciphertext_c = self.encrypt(self._plaintext_m)
            self.update()
            return True
        elif self._rect_btn_decrypt.contains(local_pos):
            decrypted = self.decrypt(self._ciphertext_c)
            self.set_plaintext(decrypted)
            return True
        elif self._rect_btn_reset.contains(local_pos):
            self.reset_simulation()
            return True
        elif self._rect_btn_stamp.contains(local_pos):
            self.stamp_to_canvas(board)
            return True

        return False

    def close(self) -> None:
        """Stops timer and removes item from scene."""
        self._timer.stop()
        scene = self.scene()
        if scene:
            scene.removeItem(self)

    # ─── Mouse Event Handlers ────────────────────────────────────────────────

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        local_pos = event.pos()

        # Check Close Button
        if self._rect_btn_close.contains(local_pos):
            self.close()
            event.accept()
            return

        # Check Mode Buttons
        if self._rect_btn_mode_rsa.contains(local_pos):
            self.set_mode("RSA")
            event.accept()
            return
        elif self._rect_btn_mode_dh.contains(local_pos):
            self.set_mode("DIFFIE_HELLMAN")
            event.accept()
            return
        elif self._rect_btn_mode_chords.contains(local_pos):
            self.set_mode("MODULAR_CHORDS")
            event.accept()
            return

        # Check Steppers & Actions
        if self._rect_btn_m_dec.contains(local_pos):
            self.set_plaintext(self._plaintext_m - 1)
            event.accept()
            return
        elif self._rect_btn_m_inc.contains(local_pos):
            self.set_plaintext(self._plaintext_m + 1)
            event.accept()
            return
        elif self._rect_btn_cycle_primes.contains(local_pos):
            self.cycle_preset()
            event.accept()
            return
        elif self._rect_btn_encrypt.contains(local_pos):
            self._ciphertext_c = self.encrypt(self._plaintext_m)
            self.update()
            event.accept()
            return
        elif self._rect_btn_decrypt.contains(local_pos):
            decrypted = self.decrypt(self._ciphertext_c)
            self.set_plaintext(decrypted)
            event.accept()
            return
        elif self._rect_btn_reset.contains(local_pos):
            self.reset_simulation()
            event.accept()
            return
        elif self._rect_btn_stamp.contains(local_pos):
            self.stamp_to_canvas()
            event.accept()
            return

        # Check click on circle to pick message M
        dial_center = QPointF(self._dial_x + self._dial_w / 2.0, self._dial_y + self._dial_h / 2.0)
        dx = local_pos.x() - dial_center.x()
        dy = local_pos.y() - dial_center.y()
        dist = math.hypot(dx, dy)

        if dist <= 120.0:
            angle = math.atan2(dy, dx) + math.pi / 2.0
            if angle < 0.0:
                angle += 2.0 * math.pi
            n = self.modulus_n
            val = int(round((angle / (2.0 * math.pi)) * n))
            val = max(1, min(n - 1, val))
            self.set_plaintext(val)
            event.accept()
            return

        super().mousePressEvent(event)

    # ─── Bounding Rect & Painting ────────────────────────────────────────────

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, self._height)

    def paint(
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: Optional[QWidget] = None,
    ) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        # 1. Outer Dark STEM Dashboard Panel
        self._paint_dashboard_frame(painter)

        # 2. Left Column: Modular Arithmetic Circle & Chord Webs
        self._paint_modular_circle(painter)

        # 3. Right Column: RSA Workflow or Diffie-Hellman Key Exchange HUD
        if self._mode == "DIFFIE_HELLMAN":
            self._paint_diffie_hellman_workflow(painter)
        else:
            self._paint_rsa_workflow(painter)

        # 4. Bottom Interactive Control Deck
        self._paint_control_deck(painter)

        painter.restore()

    def _paint_dashboard_frame(self, painter: QPainter) -> None:
        """Paints high-tech dark STEM HUD background container."""
        bg_rect = self.boundingRect()
        painter.setPen(QPen(QColor("#334155"), 1.5))
        painter.setBrush(QBrush(QColor(15, 23, 42, 245)))  # Slate 900
        painter.drawRoundedRect(bg_rect, 10.0, 10.0)

        # Header Title Bar
        header_rect = QRectF(0.0, 0.0, self._width, 42.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(30, 41, 59, 220)))  # Slate 800
        painter.drawRoundedRect(header_rect, 10.0, 10.0)
        painter.drawRect(QRectF(0.0, 20.0, self._width, 22.0))

        # Title
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        painter.setPen(QColor("#f8fafc"))
        painter.drawText(
            16.0, 26.0,
            "🔐 MẬT MÃ HỌC RSA & SỐ HỌC ĐỒNG DƯ (CRYPTOGRAPHY & MODULAR ARITHMETIC)"
        )

        n = self.modulus_n
        phi = self.euler_phi
        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(
            self._width - 340.0, 26.0,
            f"n = {self._p}×{self._q} = {n} | φ(n) = {phi} | (e, d) = ({self._e}, {self._d})"
        )

        # Close button [X]
        painter.setPen(QPen(QColor("#94a3b8"), 1.2))
        painter.setBrush(QBrush(QColor(239, 68, 68, 180)))
        painter.drawRoundedRect(self._rect_btn_close, 4.0, 4.0)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.drawText(self._rect_btn_close, Qt.AlignmentFlag.AlignCenter, "✕")

    def _paint_modular_circle(self, painter: QPainter) -> None:
        """Paints modulo clock circle with chords and message ray."""
        dx = self._dial_x
        dy = self._dial_y
        dw = self._dial_w
        dh = self._dial_h

        dial_rect = QRectF(dx, dy, dw, dh)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(10, 15, 29, 230)))
        painter.drawRoundedRect(dial_rect, 6.0, 6.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#a855f7"))
        painter.drawText(dx + 12.0, dy + 18.0, "VÒNG TRÒN ĐỒNG DƯ (MODULO DIAL)")

        # Dial center and radius
        cx = dx + dw / 2.0
        cy = dy + dh / 2.0 + 4.0
        r = 100.0

        # Base circle
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(cx, cy), r, r)

        # Determine points around dial
        # For visual aesthetics: sample min(n, 36) points
        n = self.modulus_n
        num_pts = min(n, 36)
        coords: list[tuple[float, float]] = []

        for i in range(num_pts):
            angle = (2.0 * math.pi * i) / float(num_pts) - math.pi / 2.0
            px_val = cx + r * math.cos(angle)
            py_val = cy + r * math.sin(angle)
            coords.append((px_val, py_val))

            # Dot
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor("#64748b")))
            painter.drawEllipse(QPointF(px_val, py_val), 2.0, 2.0)

        # Chords rendering
        if self._mode == "MODULAR_CHORDS":
            # Multiplier k fractal chords
            k_mult = self._multiplier_k
            chord_pen = QPen(QColor(234, 179, 8, 120), 1.0)
            painter.setPen(chord_pen)
            for i in range(num_pts):
                dest = int(round(i * k_mult)) % num_pts
                x1, y1 = coords[i]
                x2, y2 = coords[dest]
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#eab308"))
            k_name = "Cardioid (Hình tim)" if abs(k_mult - 2.0) < 0.1 else (
                "Nephroid (Hình thận)" if abs(k_mult - 3.0) < 0.1 else f"Hệ số k = {k_mult:.1f}"
            )
            painter.drawText(dx + 12.0, dy + dh - 10.0, f"Dây cung: x ➔ {k_name}")

        else:
            # RSA / Exponentiation mapping chords: x -> (x * e) % num_pts
            chord_pen = QPen(QColor(168, 85, 247, 90), 1.0)
            painter.setPen(chord_pen)
            for i in range(num_pts):
                dest = (i * self._e) % num_pts
                x1, y1 = coords[i]
                x2, y2 = coords[dest]
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

            # Highlight Active Plaintext M -> Ciphertext C Chord
            m_idx = self._plaintext_m % num_pts
            c_idx = self._ciphertext_c % num_pts
            mx, my = coords[m_idx]
            cx_pt, cy_pt = coords[c_idx]

            # Glowing ray from M to C
            painter.setPen(QPen(QColor("#ef4444"), 2.2))
            painter.drawLine(QPointF(mx, my), QPointF(cx_pt, cy_pt))

            # Green dot for M
            painter.setPen(QPen(QColor("#ffffff"), 1.2))
            painter.setBrush(QBrush(QColor("#10b981")))
            painter.drawEllipse(QPointF(mx, my), 4.5, 4.5)

            # Red dot for C
            painter.setBrush(QBrush(QColor("#ef4444")))
            painter.drawEllipse(QPointF(cx_pt, cy_pt), 4.5, 4.5)

            # Text labels near dots
            painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#10b981"))
            painter.drawText(mx + 6.0, my - 4.0, f"M={self._plaintext_m}")
            painter.setPen(QColor("#ef4444"))
            painter.drawText(cx_pt + 6.0, cy_pt + 12.0, f"C={self._ciphertext_c}")

            # Note at bottom
            painter.setFont(QFont("Segoe UI", 7))
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(dx + 12.0, dy + dh - 10.0, f"✦ Phép ánh xạ: M ➔ C = M^{self._e} mod {n}")

    def _paint_rsa_workflow(self, painter: QPainter) -> None:
        """Paints RSA mathematical workflow and step cards."""
        wx = self._work_x
        wy = self._work_y
        ww = self._work_w
        wh = self._work_h

        work_rect = QRectF(wx, wy, ww, wh)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(10, 15, 29, 230)))
        painter.drawRoundedRect(work_rect, 6.0, 6.0)

        # Title
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(wx + 12.0, wy + 18.0, "QUY TRÌNH MẬT MÃ HÓA KHÓA CÔNG KHAI RSA")

        n = self.modulus_n
        phi = self.euler_phi
        e = self._e
        d = self._d
        m = self._plaintext_m
        c = self._ciphertext_c

        # Card 1: Key Generation (Khởi tạo khóa)
        c1_y = wy + 26.0
        c1_h = 74.0
        c1_rect = QRectF(wx + 10.0, c1_y, ww - 20.0, c1_h)
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor(30, 41, 59, 160)))
        painter.drawRoundedRect(c1_rect, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#0284c7"))
        painter.drawText(wx + 16.0, c1_y + 14.0, "BƯỚC 1: KHỞI TẠO CẶP KHÓA (KEY GENERATION)")

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(wx + 16.0, c1_y + 28.0, f"● Hai số nguyên tố: p = {self._p}, q = {self._q}")
        painter.drawText(wx + 16.0, c1_y + 42.0, f"● Modulus n = p×q = {n} | Hàm Euler φ(n) = (p-1)(q-1) = {phi}")

        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(wx + 16.0, c1_y + 58.0, f"✦ Khóa công khai (Public Key):  (e, n) = ({e}, {n})")
        painter.setPen(QColor("#f43f5e"))
        painter.drawText(wx + 220.0, c1_y + 58.0, f"✦ Khóa bí mật (Private Key): (d, n) = ({d}, {n})")

        # Card 2: Encryption (Mã hóa)
        c2_y = c1_y + c1_h + 8.0
        c2_h = 68.0
        c2_rect = QRectF(wx + 10.0, c2_y, ww - 20.0, c2_h)
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor(30, 41, 59, 160)))
        painter.drawRoundedRect(c2_rect, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(wx + 16.0, c2_y + 14.0, "BƯỚC 2: MÃ HÓA BẢN RÕ (ENCRYPTION): C = M^e mod n")

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(wx + 16.0, c2_y + 30.0, f"Bản rõ gốc (Plaintext): M = {m}")
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(wx + 16.0, c2_y + 50.0, f"C = {m}^{e} mod {n} = {c}  (Bản mã gửi qua mạng)")

        # Card 3: Decryption (Giải mã)
        c3_y = c2_y + c2_h + 8.0
        c3_h = 68.0
        c3_rect = QRectF(wx + 10.0, c3_y, ww - 20.0, c3_h)
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor(30, 41, 59, 160)))
        painter.drawRoundedRect(c3_rect, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#a855f7"))
        painter.drawText(wx + 16.0, c3_y + 14.0, "BƯỚC 3: GIẢI MÃ BẢN MÃ (DECRYPTION): M = C^d mod n")

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(wx + 16.0, c3_y + 30.0, f"Người nhận dùng khóa bí mật: d = {d}")
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        decrypted_m = self.decrypt(c)
        painter.drawText(wx + 16.0, c3_y + 50.0, f"M = {c}^{d} mod {n} = {decrypted_m}  ✓ Khôi phục thành công!")

        # Bottom Invariant Note
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#94a3b8"))
        ed_val = e * d
        painter.drawText(
            wx + 10.0, wy + wh - 10.0,
            f"Chứng minh: e·d = {e}×{d} = {ed_val} ≡ 1 (mod {phi}) ➔ C^d ≡ M^(ed) ≡ M (mod {n})"
        )

    def _paint_diffie_hellman_workflow(self, painter: QPainter) -> None:
        """Paints Diffie-Hellman Key Exchange steps."""
        wx = self._work_x
        wy = self._work_y
        ww = self._work_w
        wh = self._work_h

        work_rect = QRectF(wx, wy, ww, wh)
        painter.setPen(QPen(QColor("#334155"), 1.2))
        painter.setBrush(QBrush(QColor(10, 15, 29, 230)))
        painter.drawRoundedRect(work_rect, 6.0, 6.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#f59e0b"))
        painter.drawText(wx + 12.0, wy + 18.0, "GIAO THỨC TRAO ĐỔI KHÓA DIFFIE-HELLMAN (KEY EXCHANGE)")

        p = self._dh_p
        g = self._dh_g
        a = self._dh_alice_secret
        b = self._dh_bob_secret

        # Public values
        A = fast_modular_pow(g, a, p)
        B = fast_modular_pow(g, b, p)

        # Shared secret
        K_alice = fast_modular_pow(B, a, p)
        K_bob = fast_modular_pow(A, b, p)

        # Public params card
        c1 = QRectF(wx + 10.0, wy + 28.0, ww - 20.0, 48.0)
        painter.setPen(QPen(QColor("#334155"), 1.0))
        painter.setBrush(QBrush(QColor(30, 41, 59, 160)))
        painter.drawRoundedRect(c1, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#0284c7"))
        painter.drawText(wx + 16.0, wy + 42.0, "THÔNG SỐ CÔNG KHAI (PUBLIC PARAMETERS)")
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(wx + 16.0, wy + 58.0, f"Số nguyên tố p = {p} | Phần tử sinh nguyên thủy g = {g}")

        # Alice and Bob columns
        col_w = (ww - 28.0) / 2.0
        # Alice
        c_alice = QRectF(wx + 10.0, wy + 84.0, col_w, 140.0)
        painter.drawRoundedRect(c_alice, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ec4899"))
        painter.drawText(wx + 16.0, wy + 100.0, "ALICE (Bên A)")
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(wx + 16.0, wy + 118.0, f"Khóa bí mật: a = {a}")
        painter.drawText(wx + 16.0, wy + 134.0, f"Giá trị công khai:")
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(wx + 16.0, wy + 150.0, f"A = {g}^{a} mod {p} = {A}")
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(wx + 16.0, wy + 172.0, f"Khóa chung tính được:")
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(wx + 16.0, wy + 192.0, f"K = {B}^{a} mod {p} = {K_alice}")

        # Bob
        c_bob = QRectF(wx + 18.0 + col_w, wy + 84.0, col_w, 140.0)
        painter.drawRoundedRect(c_bob, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#38bdf8"))
        painter.drawText(wx + 24.0 + col_w, wy + 100.0, "BOB (Bên B)")
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(wx + 24.0 + col_w, wy + 118.0, f"Khóa bí mật: b = {b}")
        painter.drawText(wx + 24.0 + col_w, wy + 134.0, f"Giá trị công khai:")
        painter.setFont(QFont("Consolas", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(wx + 24.0 + col_w, wy + 150.0, f"B = {g}^{b} mod {p} = {B}")
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(wx + 24.0 + col_w, wy + 172.0, f"Khóa chung tính được:")
        painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#fbbf24"))
        painter.drawText(wx + 24.0 + col_w, wy + 192.0, f"K = {A}^{b} mod {p} = {K_bob}")

        # Secret agreement result
        c_res = QRectF(wx + 10.0, wy + 232.0, ww - 20.0, 42.0)
        painter.setPen(QPen(QColor("#10b981"), 1.2))
        painter.setBrush(QBrush(QColor(16, 185, 129, 30)))
        painter.drawRoundedRect(c_res, 4.0, 4.0)

        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#10b981"))
        painter.drawText(wx + 16.0, wy + 258.0, f"✓ KHÓA PHIÊN CHUNG ĐỒNG THUẬN: K = {K_alice} (Bảo mật tuyệt đối!)")

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(
            wx + 10.0, wy + wh - 10.0,
            f"Kẻ nghe lén (Eve) chỉ biết p={p}, g={g}, A={A}, B={B}; bài toán Logarit Rời rạc bảo vệ a và b."
        )

    def _paint_control_deck(self, painter: QPainter) -> None:
        """Paints interactive button deck."""
        # Row 1: Mode Selectors
        def draw_mode_btn(rect: QRectF, text: str, is_active: bool, col_hex: str) -> None:
            col = QColor(col_hex) if is_active else QColor("#1e293b")
            border = col.lighter(130) if is_active else QColor("#475569")
            painter.setPen(QPen(border, 1.2))
            painter.setBrush(QBrush(col))
            painter.drawRoundedRect(rect, 4.0, 4.0)

            painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
            painter.setPen(QColor("#ffffff") if is_active else QColor("#94a3b8"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)

        draw_mode_btn(self._rect_btn_mode_rsa, "🔐 Mật mã RSA (Public-Key)", self._mode == "RSA", "#0284c7")
        draw_mode_btn(self._rect_btn_mode_dh, "🤝 Trao đổi khóa Diffie-Hellman", self._mode == "DIFFIE_HELLMAN", "#7c3aed")
        draw_mode_btn(self._rect_btn_mode_chords, "🌀 Vòng tròn Dây cung Modulo (Cardioid)", self._mode == "MODULAR_CHORDS", "#d97706")

        # Row 2: Action Buttons
        # Plaintext M stepper [-] [M=9] [+]
        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.setBrush(QBrush(QColor("#334155")))
        painter.drawRoundedRect(self._rect_btn_m_dec, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_m_dec, Qt.AlignmentFlag.AlignCenter, "−")

        m_mid = QRectF(self._rect_btn_m_dec.right() + 2.0, self._rect_btn_m_dec.top(), 62.0, 26.0)
        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor("#cbd5e1"))
        painter.drawText(m_mid, Qt.AlignmentFlag.AlignCenter, f"M = {self._plaintext_m}")

        painter.setPen(QPen(QColor("#475569"), 1.0))
        painter.setBrush(QBrush(QColor("#334155")))
        painter.drawRoundedRect(self._rect_btn_m_inc, 3.0, 3.0)
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_m_inc, Qt.AlignmentFlag.AlignCenter, "+")

        # Preset cycler
        painter.setPen(QPen(QColor("#0284c7"), 1.0))
        painter.setBrush(QBrush(QColor("#0369a1")))
        painter.drawRoundedRect(self._rect_btn_cycle_primes, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        spec_name = RSA_PRESETS[self._preset][0].split("(")[1].replace(")", "")
        painter.drawText(self._rect_btn_cycle_primes, Qt.AlignmentFlag.AlignCenter, f"🔢 Cặp (p, q): {spec_name}")

        # Encrypt & Decrypt buttons
        painter.setPen(QPen(QColor("#059669"), 1.0))
        painter.setBrush(QBrush(QColor("#047857")))
        painter.drawRoundedRect(self._rect_btn_encrypt, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_encrypt, Qt.AlignmentFlag.AlignCenter, "🔒 Mã hóa (C)")

        painter.setPen(QPen(QColor("#7c3aed"), 1.0))
        painter.setBrush(QBrush(QColor("#6d28d9")))
        painter.drawRoundedRect(self._rect_btn_decrypt, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_decrypt, Qt.AlignmentFlag.AlignCenter, "🔓 Giải mã (M)")

        # Reset & Stamp
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.setBrush(QBrush(QColor("#334155")))
        painter.drawRoundedRect(self._rect_btn_reset, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_reset, Qt.AlignmentFlag.AlignCenter, "🔄 Đặt lại")

        painter.setPen(QPen(QColor("#059669"), 1.2))
        painter.setBrush(QBrush(QColor("#047857")))
        painter.drawRoundedRect(self._rect_btn_stamp, 4.0, 4.0)
        painter.setFont(QFont("Segoe UI", 7, QFont.Weight.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(self._rect_btn_stamp, Qt.AlignmentFlag.AlignCenter, "🖊️ Đóng dấu")
