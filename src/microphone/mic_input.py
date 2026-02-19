"""
microphone/mic_input.py
-----------------------
Example module for Raspberry Pi microphone input via sounddevice.

Replace / extend this module with your actual RPi microphone setup
(e.g. USB mic, I2S MEMS mic, ReSpeaker HAT, etc.).
"""

import queue
import collections
from typing import Optional

import numpy as np
import sounddevice as sd


class MicrophoneStream:
    """Continuous microphone capture with a pre-roll buffer.

    Usage
    -----
    stream = MicrophoneStream(capture_sr=48000, frame_sec=0.10, pre_seconds=0.5)
    with stream:
        while True:
            chunk = stream.read()      # blocking; returns one frame (numpy array)
            pre   = stream.pre_buffer  # snapshot of recent frames before trigger

    Parameters
    ----------
    capture_sr  : Microphone sample rate (Hz). 48000 recommended for RPi USB mics.
    frame_sec   : Duration of each audio chunk returned by read() (seconds).
    pre_seconds : Length of the rolling pre-buffer (seconds).
    device      : sounddevice device index or None for the system default.
    """

    def __init__(
        self,
        capture_sr: int = 48000,
        frame_sec: float = 0.10,
        pre_seconds: float = 0.5,
        device: Optional[int] = None,
    ):
        self.capture_sr  = capture_sr
        self.frame_sec   = frame_sec
        self.device      = device

        self._frame_samples = int(capture_sr * frame_sec)
        pre_chunks = max(1, int(pre_seconds / frame_sec))
        self.pre_buffer: collections.deque = collections.deque(maxlen=pre_chunks)

        self._queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self):
        """Open the sounddevice input stream."""
        self._stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.capture_sr,
            blocksize=self._frame_samples,
            callback=self._callback,
        )
        self._stream.start()

    def close(self):
        """Stop and close the stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read(self) -> np.ndarray:
        """Block until the next audio frame is available and return it.

        Returns
        -------
        np.ndarray of shape (frame_samples,), dtype float32.
        """
        return self._queue.get()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(self, indata, frames, time_info, status):
        chunk = indata[:, 0].astype(np.float32).copy()
        self._queue.put(chunk)
