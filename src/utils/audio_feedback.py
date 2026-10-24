"""
AirTeach AI - Acoustic Gesture Feedback & Synthetic Audio Engine
================================================================
Provides non-blocking, latency-free auditory feedback for in-air gesture
interactions (Air Tap, Dwell click, Shape Snapping, Eraser, Page Turns,
and Countdown Timer Chime) using synthesized PCM audio in memory.

Zero external audio files required: all waveforms are generated on the fly
via mathematical harmonic synthesis with exponential decay envelopes.
"""
from __future__ import annotations

import io
import logging
import math
import struct
import wave
from enum import Enum, auto
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import winsound
    _HAS_WINSOUND = True
except ImportError:
    winsound = None  # type: ignore[assignment]
    _HAS_WINSOUND = False


class SoundEffect(Enum):
    """Categorized auditory interaction chimes."""
    CLICK = auto()         # Crisp in-air tap or menu item selection
    SNAP = auto()          # Harmonic major triad for geometric shape snapping
    PAGE_TURN = auto()     # Soft pitch-swept woosh for page navigation
    ERASE = auto()         # Low resonant sweep for stroke erasure
    TIMER_CHIME = auto()   # Celebratory alert triad for countdown timer completion
    UNDO = auto()          # Descending tone for undo action
    METRONOME_TICK = auto() # Regular metronome beat woodblock tick
    METRONOME_BELL = auto() # High accent bell for measure downbeat


def _synthesize_pcm_wav(
    frequencies: List[float],
    duration_sec: float = 0.08,
    sample_rate: int = 22050,
    volume: float = 0.8,
    decay_rate: float = 5.0,
    attack_rate: float = 35.0,
    pitch_bend_rate: float = 0.0,
) -> bytes:
    """
    Synthesize an in-memory 16-bit mono PCM WAV buffer for harmonic tones.
    
    Args:
        frequencies: List of component frequencies in Hz.
        duration_sec: Tone duration in seconds.
        sample_rate: Audio sampling frequency (Hz).
        volume: Output scale [0.0, 1.0].
        decay_rate: Exponential envelope decay coefficient.
        attack_rate: Attack envelope smoothing coefficient.
        pitch_bend_rate: Rate of linear pitch modulation per second.
        
    Returns:
        Raw bytes formatted as standard RIFF/WAVE container.
    """
    num_samples = int(sample_rate * duration_sec)
    buf = io.BytesIO()

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()

        for i in range(num_samples):
            t = float(i) / sample_rate
            # Smooth attack & natural exponential decay envelope
            attack = 1.0 - math.exp(-attack_rate * t) if t > 0.0 else 0.0
            decay = math.exp(-decay_rate * (t / duration_sec))
            envelope = attack * decay

            # Harmonic superposition with optional pitch sweep
            val = 0.0
            for f in frequencies:
                effective_f = max(20.0, f + pitch_bend_rate * t)
                val += math.sin(2.0 * math.pi * effective_f * t)
            
            val = (val / max(1, len(frequencies))) * envelope * volume
            sample_val = int(max(-32767, min(32767, val * 32767.0)))
            frames.extend(struct.pack("<h", sample_val))

        wf.writeframes(frames)

    return buf.getvalue()


class AudioFeedbackManager:
    """
    Central manager for touchless auditory feedback in AirTeach AI.
    Pre-synthesizes and caches audio buffers for instant zero-latency playback.
    """

    _instance: Optional[AudioFeedbackManager] = None

    def __init__(self, enabled: bool = True, volume: float = 0.75) -> None:
        self._enabled = enabled
        self._volume = max(0.0, min(1.0, volume))
        self._cache: Dict[SoundEffect, bytes] = {}
        self._build_waveforms()

    @classmethod
    def instance(cls) -> AudioFeedbackManager:
        """Get the singleton audio feedback manager instance."""
        if cls._instance is None:
            cls._instance = AudioFeedbackManager()
        return cls._instance

    @property
    def enabled(self) -> bool:
        """Whether audio feedback chimes are active."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = bool(value)

    @property
    def volume(self) -> float:
        """Master audio feedback volume [0.0, 1.0]."""
        return self._volume

    @volume.setter
    def volume(self, val: float) -> None:
        new_vol = max(0.0, min(1.0, float(val)))
        if abs(new_vol - self._volume) > 1e-4:
            self._volume = new_vol
            self._build_waveforms()

    def _build_waveforms(self) -> None:
        """Pre-render and cache all synthesized audio buffers at current volume."""
        vol = self._volume
        # 1. Click / Air Tap: High crisp pop (1150 Hz, 18ms)
        self._cache[SoundEffect.CLICK] = _synthesize_pcm_wav(
            frequencies=[1150.0],
            duration_sec=0.022,
            volume=vol * 0.9,
            decay_rate=12.0,
            attack_rate=120.0,
        )
        # 2. Shape Snap: Bright ascending major triad (C5=523.25, E5=659.25, G5=783.99 Hz, 75ms)
        self._cache[SoundEffect.SNAP] = _synthesize_pcm_wav(
            frequencies=[523.25, 659.25, 783.99],
            duration_sec=0.085,
            volume=vol * 0.85,
            decay_rate=4.5,
            attack_rate=45.0,
        )
        # 3. Page Turn: Low downward pitch sweep (420 Hz -> 260 Hz, 60ms)
        self._cache[SoundEffect.PAGE_TURN] = _synthesize_pcm_wav(
            frequencies=[420.0],
            duration_sec=0.065,
            volume=vol * 0.7,
            decay_rate=6.0,
            attack_rate=40.0,
            pitch_bend_rate=-2400.0,
        )
        # 4. Erase: Soft low tone (240 Hz, 50ms)
        self._cache[SoundEffect.ERASE] = _synthesize_pcm_wav(
            frequencies=[240.0, 310.0],
            duration_sec=0.055,
            volume=vol * 0.65,
            decay_rate=7.0,
            attack_rate=35.0,
        )
        # 5. Timer Chime: Resonant celebratory triad chime (E5=659.25, G#5=830.61, B5=987.77 Hz, 350ms)
        self._cache[SoundEffect.TIMER_CHIME] = _synthesize_pcm_wav(
            frequencies=[659.25, 830.61, 987.77],
            duration_sec=0.38,
            volume=vol * 1.0,
            decay_rate=2.2,
            attack_rate=30.0,
        )
        # 6. Undo: Subtle descending step (540 Hz -> 360 Hz, 55ms)
        self._cache[SoundEffect.UNDO] = _synthesize_pcm_wav(
            frequencies=[540.0],
            duration_sec=0.06,
            volume=vol * 0.75,
            decay_rate=5.5,
            attack_rate=50.0,
            pitch_bend_rate=-3000.0,
        )
        # 7. Metronome Regular Beat: Dry woodblock click (680 Hz, 16ms)
        self._cache[SoundEffect.METRONOME_TICK] = _synthesize_pcm_wav(
            frequencies=[680.0],
            duration_sec=0.018,
            volume=vol * 0.85,
            decay_rate=14.0,
            attack_rate=150.0,
        )
        # 8. Metronome Downbeat Bell: High resonant accent ping (1250 Hz, 28ms)
        self._cache[SoundEffect.METRONOME_BELL] = _synthesize_pcm_wav(
            frequencies=[1250.0, 2500.0],
            duration_sec=0.032,
            volume=vol * 1.0,
            decay_rate=8.0,
            attack_rate=100.0,
        )

    def play(self, effect: SoundEffect) -> None:
        """
        Play a sound effect asynchronously.
        Never blocks the caller and catches all audio subsystem exceptions safely.
        """
        if not self._enabled or self._volume <= 0.001:
            return

        wav_bytes = self._cache.get(effect)
        if not wav_bytes:
            return

        if _HAS_WINSOUND and winsound is not None:
            try:
                # SND_ASYNC | SND_MEMORY plays instantly without stalling the GUI
                winsound.PlaySound(
                    wav_bytes,
                    winsound.SND_MEMORY | winsound.SND_ASYNC | winsound.SND_NODEFAULT,
                )
            except Exception as e:
                logger.debug("winsound audio playback failed safely: %s", e)

    def play_click(self) -> None:
        """Play crisp air tap / click confirmation."""
        self.play(SoundEffect.CLICK)

    def play_snap(self) -> None:
        """Play geometric shape snapping chime."""
        self.play(SoundEffect.SNAP)

    def play_page_turn(self) -> None:
        """Play page turn woosh."""
        self.play(SoundEffect.PAGE_TURN)

    def play_erase(self) -> None:
        """Play eraser confirmation sound."""
        self.play(SoundEffect.ERASE)

    def play_timer_chime(self) -> None:
        """Play countdown timer alert buzzer / chime."""
        self.play(SoundEffect.TIMER_CHIME)

    def play_undo(self) -> None:
        """Play undo/redo tone."""
        self.play(SoundEffect.UNDO)

    def play_metronome_tick(self) -> None:
        """Play standard woodblock metronome tick."""
        self.play(SoundEffect.METRONOME_TICK)

    def play_metronome_bell(self) -> None:
        """Play accented high bell downbeat chime for measure start."""
        self.play(SoundEffect.METRONOME_BELL)
