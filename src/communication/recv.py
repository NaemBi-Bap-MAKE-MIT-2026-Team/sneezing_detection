"""
communication/recv.py
----------------------
NetworkMicStream: receives UDP audio from send.py and exposes the
exact same interface as microphone.MicrophoneStream.

  read()       — blocking; returns one frame as float32 numpy array
  pre_buffer   — rolling deque of recent frames (for pre-roll)
  open/close   — start/stop the background receiver thread
  __enter__/__exit__ — context manager

Tiny UDP packets (10 ms each) from send.py are accumulated in a background
thread. Once enough samples fill one frame_sec window they are yielded from
read() — indistinguishable from a local MicrophoneStream frame.

Usage
-----
from communication.recv import NetworkMicStream

mic = NetworkMicStream(
    host       = "0.0.0.0",   # listen on all interfaces
    port       = 12345,
    capture_sr = 48000,
    frame_sec  = 0.10,
    pre_seconds= 0.5,
)
with mic:
    while True:
        chunk = mic.read()          # same as MicrophoneStream.read()
        pre   = mic.pre_buffer      # same as MicrophoneStream.pre_buffer
"""

import collections
import queue
import socket
import threading
from typing import Optional

import numpy as np


# Maximum UDP payload we will ever receive.
# 10 ms @ 48 kHz × 4 bytes/sample = 1 920 bytes, but leave headroom.
_UDP_BUFSIZE = 65535


class NetworkMicStream:
    """Drop-in replacement for MicrophoneStream that reads audio over UDP.

    Parameters
    ----------
    host        : IP address to bind (``"0.0.0.0"`` = all interfaces).
    port        : UDP port to listen on (must match send.py --port).
    capture_sr  : Expected sample rate of the incoming audio (Hz).
    frame_sec   : Duration of each chunk returned by read() (seconds).
    pre_seconds : Length of the rolling pre-buffer (seconds).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 12345,
        capture_sr: int = 48000,
        frame_sec: float = 0.10,
        pre_seconds: float = 0.5,
    ):
        self.capture_sr = capture_sr
        self.frame_sec  = frame_sec

        self._frame_samples = int(capture_sr * frame_sec)
        pre_chunks = max(1, int(pre_seconds / frame_sec))
        self.pre_buffer: collections.deque = collections.deque(maxlen=pre_chunks)

        self._queue: queue.Queue = queue.Queue()

        self._sock: Optional[socket.socket] = None
        self._host = host
        self._port = port
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Public API  (identical to MicrophoneStream)
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Bind the UDP socket and start the background receiver thread."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(0.5)          # lets _recv_loop check _running
        self._sock.bind((self._host, self._port))

        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Stop the receiver thread and close the socket."""
        self._running = False
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def read(self) -> np.ndarray:
        """Block until the next frame is available and return it.

        Returns
        -------
        np.ndarray of shape (frame_samples,), dtype float32.
        """
        return self._queue.get()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        """Background thread: receive UDP packets → assemble → enqueue frames."""
        buf = np.zeros((0,), dtype=np.float32)

        while self._running:
            try:
                data, _ = self._sock.recvfrom(_UDP_BUFSIZE)
            except socket.timeout:
                continue
            except OSError:
                break

            # Append raw float32 samples to accumulation buffer
            chunk = np.frombuffer(data, dtype=np.float32)
            buf   = np.concatenate((buf, chunk))

            # Yield complete frames as soon as they are ready
            while len(buf) >= self._frame_samples:
                frame = buf[:self._frame_samples].copy()
                buf   = buf[self._frame_samples:]
                self._queue.put(frame)
