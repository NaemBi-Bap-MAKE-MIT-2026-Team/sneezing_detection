"""
output_feature/speaker_output.py
---------------------------------
Example module for Raspberry Pi audio / alert output.

This module provides a simple interface for producing an audible or
text-based alert when a sneeze is detected. Replace / extend the
implementation with your actual RPi output hardware (e.g. a speaker
connected through the 3.5 mm jack, a buzzer on GPIO, an LED, etc.).
"""

import time
from typing import Optional

import numpy as np

# sounddevice is used here for speaker playback.
# On an RPi without a display it is often simpler to use aplay / subprocess,
# but sounddevice keeps the dependency list consistent with the microphone module.
try:
    import sounddevice as sd
    _SD_AVAILABLE = True
except ImportError:
    _SD_AVAILABLE = False


class SpeakerOutput:
    """Simple audio / console output for sneeze-detection alerts.

    Usage
    -----
    output = SpeakerOutput(playback_sr=16000, device=None)
    output.alert("bless you!")        # print + optional beep
    output.play_wav(audio_array)      # play a numpy audio array

    Parameters
    ----------
    playback_sr : Sample rate used for audio playback (Hz).
    device      : sounddevice output device index, or None for system default.
    volume      : Playback volume scaling factor (0.0 – 1.0).
    """

    def __init__(
        self,
        playback_sr: int = 16000,
        device: Optional[int] = None,
        volume: float = 1.0,
    ):
        self.playback_sr = playback_sr
        self.device      = device
        self.volume      = float(np.clip(volume, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def alert(self, message: str = "bless you!") -> None:
        """Print a detection message and play a short beep.

        Replace the beep with your preferred alert (GPIO buzzer, LED flash,
        TTS synthesis, etc.).
        """
        print(message)
        self._beep(duration_ms=200, freq_hz=880)

    def play_wav(self, audio: np.ndarray) -> None:
        """Play a float32 mono audio array through the system speaker.

        Parameters
        ----------
        audio : 1-D float32 numpy array, values in [-1, 1].
        """
        if not _SD_AVAILABLE:
            print("[SpeakerOutput] sounddevice not available — skipping playback.")
            return
        data = (audio * self.volume).astype(np.float32)
        sd.play(data, samplerate=self.playback_sr, device=self.device)
        sd.wait()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _beep(self, duration_ms: int = 200, freq_hz: float = 880) -> None:
        """Generate and play a simple sine-wave beep."""
        if not _SD_AVAILABLE:
            return
        t = np.linspace(0, duration_ms / 1000.0, int(self.playback_sr * duration_ms / 1000.0), endpoint=False)
        tone = (np.sin(2 * np.pi * freq_hz * t) * 0.5 * self.volume).astype(np.float32)
        sd.play(tone, samplerate=self.playback_sr, device=self.device)
        sd.wait()
