"""
Unit and integration tests for Interactive Cryptography, RSA Public-Key & Modular Arithmetic Circle Engine.
Verifies Extended Euclidean Algorithm, modular inverse, RSA encryption/decryption roundtrip,
Diffie-Hellman key exchange, interactive item UI/air-gestures, and LessonBoard integration.
"""

import pytest
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QApplication

from src.ui.interactive_cryptography import (
    InteractiveCryptographyItem,
    RSA_PRESETS,
    extended_gcd,
    fast_modular_pow,
    modular_inverse,
)
from src.ui.lesson_board import LessonBoard


@pytest.fixture(scope="module")
def qapp():
    """Ensure QApplication instance exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_extended_euclidean_and_modular_inverse(qapp):
    """Test Extended Euclidean Algorithm and modular inverse computation."""
    # Test gcd(35, 15) = 5
    gcd_val, x, y = extended_gcd(35, 15)
    assert gcd_val == 5
    assert 35 * x + 15 * y == 5

    # Test modular inverse of e=7 mod phi=60
    # 7 * 43 = 301 = 5 * 60 + 1 = 1 mod 60
    d = modular_inverse(7, 60)
    assert d == 43
    assert (7 * d) % 60 == 1

    # Test non-coprime case (e.g. 6 and 60 share factor 6) -> should return None
    assert modular_inverse(6, 60) is None


def test_rsa_key_generation_and_presets(qapp):
    """Verify RSA key parameters across all standard prime presets."""
    item = InteractiveCryptographyItem(preset="PAIR_7_11")
    assert item.prime_p == 7
    assert item.prime_q == 11
    assert item.modulus_n == 77
    assert item.euler_phi == 60
    assert item.public_exponent_e == 7
    assert item.private_exponent_d == 43
    assert (item.public_exponent_e * item.private_exponent_d) % item.euler_phi == 1

    # Cycle preset to PAIR_11_13
    item.cycle_preset()
    assert item.prime_p == 11
    assert item.prime_q == 13
    assert item.modulus_n == 143
    assert item.euler_phi == 120
    assert (item.public_exponent_e * item.private_exponent_d) % item.euler_phi == 1

    # Explicit preset change to PAIR_13_17
    item.set_preset("PAIR_13_17")
    assert item.prime_p == 13
    assert item.prime_q == 17
    assert item.modulus_n == 221
    assert item.euler_phi == 192
    assert (item.public_exponent_e * item.private_exponent_d) % item.euler_phi == 1


def test_rsa_encryption_decryption_roundtrip(qapp):
    """Verify RSA encryption and decryption exact mathematical inversion: M = (M^e)^d mod n."""
    item = InteractiveCryptographyItem(preset="PAIR_7_11")  # n=77
    test_messages = [2, 5, 9, 15, 23, 42, 65, 75]

    for m in test_messages:
        c = item.encrypt(m)
        decrypted = item.decrypt(c)
        assert decrypted == m, f"Failed roundtrip for M={m}: got C={c}, M'={decrypted}"

    # Verify set_plaintext updates internal state
    item.set_plaintext(19)
    assert item.plaintext_m == 19
    assert item.ciphertext_c == item.encrypt(19)
    assert item.decrypt(item.ciphertext_c) == 19


def test_diffie_hellman_shared_secret(qapp):
    """Verify Diffie-Hellman Key Agreement protocol mathematics."""
    p = 23
    g = 5
    alice_secret_a = 6
    bob_secret_b = 15

    # Alice public key: A = g^a mod p
    alice_public_A = fast_modular_pow(g, alice_secret_a, p)  # 5^6 mod 23 = 8
    # Bob public key: B = g^b mod p
    bob_public_B = fast_modular_pow(g, bob_secret_b, p)      # 5^15 mod 23 = 19

    # Alice calculates shared secret: S_A = B^a mod p
    shared_secret_alice = fast_modular_pow(bob_public_B, alice_secret_a, p)
    # Bob calculates shared secret: S_B = A^b mod p
    shared_secret_bob = fast_modular_pow(alice_public_A, bob_secret_b, p)

    assert shared_secret_alice == shared_secret_bob == 2


def test_interactive_item_instantiation_and_modes(qapp):
    """Verify item geometry, mode switching, and signal emission."""
    item = InteractiveCryptographyItem(pos=(100.0, 150.0), mode="RSA")
    assert item.x() == 100.0
    assert item.y() == 150.0
    assert item.boundingRect().width() == 780.0
    assert item.boundingRect().height() == 500.0
    assert item.mode == "RSA"

    # Switch mode to DIFFIE_HELLMAN
    item.set_mode("DIFFIE_HELLMAN")
    assert item.mode == "DIFFIE_HELLMAN"

    # Switch mode to MODULAR_CHORDS
    item.set_mode("MODULAR_CHORDS")
    assert item.mode == "MODULAR_CHORDS"


def test_air_tap_and_pinch_interactions(qapp):
    """Verify touchless air-tap and air-pinch gesture triggers on item control buttons."""
    item = InteractiveCryptographyItem(pos=(0.0, 0.0), preset="PAIR_7_11")
    initial_m = item.plaintext_m

    # Air tap on [+] plaintext button (rect: 118, 422, 28, 26)
    btn_inc_center = QPointF(118.0 + 14.0, 422.0 + 13.0)
    handled = item.handle_air_tap(btn_inc_center)
    assert handled is True
    assert item.plaintext_m == initial_m + 1

    # Air tap on [-] plaintext button (rect: 24, 422, 28, 26)
    btn_dec_center = QPointF(24.0 + 14.0, 422.0 + 13.0)
    handled = item.handle_air_tap(btn_dec_center)
    assert handled is True
    assert item.plaintext_m == initial_m

    # Air tap on Diffie-Hellman mode button (rect: 254, 388, 240, 26)
    btn_dh_center = QPointF(254.0 + 50.0, 388.0 + 13.0)
    handled = item.handle_air_tap(btn_dh_center)
    assert handled is True
    assert item.mode == "DIFFIE_HELLMAN"

    # Air pinch to drag item
    init_x, init_y = item.x(), item.y()
    handled_pinch = item.handle_air_pinch(QPointF(70.0, 35.0), QPointF(50.0, 20.0))
    assert handled_pinch is True
    assert item.x() == init_x + 20.0
    assert item.y() == init_y + 15.0


def test_stamp_to_canvas(qapp):
    """Verify stamping RSA Cryptography vector diagram to LessonBoard."""
    board = LessonBoard()
    item = InteractiveCryptographyItem(pos=(200.0, 200.0), preset="PAIR_7_11")
    board.addItem(item)

    initial_item_count = len(board.items())
    stamped_items = item.stamp_to_canvas(board)

    assert len(stamped_items) > 0
    assert len(board.items()) > initial_item_count
    item.close()


def test_lesson_board_integration(qapp):
    """Verify LessonBoard show/toggle/get methods and eraser immunity for cryptography item."""
    board = LessonBoard()
    assert board.is_cryptography_visible() is False
    assert board.get_cryptography() is None

    # Toggle to show
    state = board.toggle_cryptography()
    assert state is True
    assert board.is_cryptography_visible() is True
    crypto_item = board.get_cryptography()
    assert crypto_item is not None
    assert isinstance(crypto_item, InteractiveCryptographyItem)

    # Verify eraser exclusion: get_items_in_rect should not include crypto_item
    from PySide6.QtCore import QRectF
    overlap_rect = QRectF(crypto_item.x(), crypto_item.y(), 100.0, 100.0)
    items_in_rect = board.get_items_in_rect(overlap_rect)
    assert crypto_item not in items_in_rect

    # Toggle to hide
    state = board.toggle_cryptography()
    assert state is False
    assert board.is_cryptography_visible() is False
    assert board.get_cryptography() is None
