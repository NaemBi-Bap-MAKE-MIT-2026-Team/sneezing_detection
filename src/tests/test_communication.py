"""
tests/test_communication.py
----------------------------
Standalone tests for the communication module (UDP loopback).

No physical microphone or send.py process required — a background thread
mimics send.py by pushing raw float32 packets to 127.0.0.1.

Run from src/:
    python tests/test_communication.py

Results:
    PASS  — assertion satisfied
    FAIL  — unexpected error
"""

import socket
import sys
import threading
import time
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


_HOST = "127.0.0.1"
_PORT = 14700          # arbitrary port unlikely to clash with real usage
_CAPTURE_SR   = 48000
_FRAME_SEC    = 0.10
_PRE_SECONDS  = 0.5
_SEND_BLOCK   = 480    # 10 ms @ 48 kHz  (same as send.py default)


def _pass(name: str) -> None:
    print(f"  PASS  {name}")


def _fail(name: str, err: Exception) -> None:
    print(f"  FAIL  {name}  — {err}")


# ---------------------------------------------------------------------- #
# Helper: fake sender                                                     #
# ---------------------------------------------------------------------- #

def _run_fake_sender(dest: tuple, n_packets: int, block: int, delay: float):
    """Send n_packets of silence (float32 zeros) over UDP, then stop."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    data = np.zeros(block, dtype=np.float32).tobytes()
    for _ in range(n_packets):
        sock.sendto(data, dest)
        time.sleep(delay)
    sock.close()


# ---------------------------------------------------------------------- #
# Tests                                                                   #
# ---------------------------------------------------------------------- #

def test_loopback_sample_count():
    """Transmitted sample count must equal received sample count."""
    name = "loopback — transmitted == received samples"
    try:
        from communication.recv import NetworkMicStream

        n_packets   = 30                         # 30 × 10 ms = 300 ms of audio
        total_sent  = n_packets * _SEND_BLOCK    # = 14 400 samples
        frame_size  = int(_CAPTURE_SR * _FRAME_SEC)  # = 4 800 samples
        # We expect floor(total_sent / frame_size) complete frames
        expected_frames = total_sent // frame_size

        mic = NetworkMicStream(
            host        = _HOST,
            port        = _PORT,
            capture_sr  = _CAPTURE_SR,
            frame_sec   = _FRAME_SEC,
            pre_seconds = _PRE_SECONDS,
        )

        received_samples = 0

        with mic:
            # Start fake sender after a short delay so recv socket is ready
            t = threading.Thread(
                target=_run_fake_sender,
                args=((_HOST, _PORT), n_packets, _SEND_BLOCK, 0.008),
                daemon=True,
            )
            t.start()

            for _ in range(expected_frames):
                frame = mic.read()
                received_samples += len(frame)

        assert received_samples == expected_frames * frame_size, \
            f"expected {expected_frames * frame_size}, got {received_samples}"
        _pass(f"{name}  ({received_samples} samples in {expected_frames} frames)")

    except Exception as e:
        _fail(name, e)


def test_pre_buffer_updates():
    """pre_buffer should accumulate frames as main.py expects."""
    name = "pre_buffer — deque length grows up to maxlen"
    try:
        from communication.recv import NetworkMicStream

        n_packets = 15
        frame_size = int(_CAPTURE_SR * _FRAME_SEC)
        expected_frames = (n_packets * _SEND_BLOCK) // frame_size

        mic = NetworkMicStream(
            host        = _HOST,
            port        = _PORT + 1,
            capture_sr  = _CAPTURE_SR,
            frame_sec   = _FRAME_SEC,
            pre_seconds = _PRE_SECONDS,
        )

        with mic:
            t = threading.Thread(
                target=_run_fake_sender,
                args=((_HOST, _PORT + 1), n_packets, _SEND_BLOCK, 0.008),
                daemon=True,
            )
            t.start()

            for _ in range(expected_frames):
                frame = mic.read()
                mic.pre_buffer.append(frame)

        pre_chunks = max(1, int(_PRE_SECONDS / _FRAME_SEC))  # = 5
        assert len(mic.pre_buffer) <= pre_chunks, \
            f"deque exceeded maxlen: {len(mic.pre_buffer)} > {pre_chunks}"
        assert len(mic.pre_buffer) > 0, "pre_buffer is empty"
        _pass(f"{name}  (len={len(mic.pre_buffer)}, maxlen={pre_chunks})")

    except Exception as e:
        _fail(name, e)


def test_context_manager_clean_exit():
    """open() / close() via context manager must not raise."""
    name = "context manager — clean open and close"
    try:
        from communication.recv import NetworkMicStream

        mic = NetworkMicStream(
            host        = _HOST,
            port        = _PORT + 2,
            capture_sr  = _CAPTURE_SR,
            frame_sec   = _FRAME_SEC,
            pre_seconds = _PRE_SECONDS,
        )
        with mic:
            pass  # enter and exit immediately without reading

        _pass(name)
    except Exception as e:
        _fail(name, e)


def test_frame_dtype_and_shape():
    """Each frame returned by read() must be float32, 1-D, frame_samples long."""
    name = "read() — dtype=float32, shape=(frame_samples,)"
    try:
        from communication.recv import NetworkMicStream

        n_packets  = 15
        frame_size = int(_CAPTURE_SR * _FRAME_SEC)

        mic = NetworkMicStream(
            host        = _HOST,
            port        = _PORT + 3,
            capture_sr  = _CAPTURE_SR,
            frame_sec   = _FRAME_SEC,
            pre_seconds = _PRE_SECONDS,
        )

        with mic:
            t = threading.Thread(
                target=_run_fake_sender,
                args=((_HOST, _PORT + 3), n_packets, _SEND_BLOCK, 0.008),
                daemon=True,
            )
            t.start()

            frame = mic.read()

        assert frame.ndim == 1,              f"expected 1-D, got {frame.ndim}"
        assert len(frame) == frame_size,     f"expected {frame_size}, got {len(frame)}"
        assert frame.dtype == np.float32,    f"expected float32, got {frame.dtype}"
        _pass(f"{name}  (shape={frame.shape}, dtype={frame.dtype})")

    except Exception as e:
        _fail(name, e)


# ---------------------------------------------------------------------- #
# Runner                                                                  #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=== test_communication ===")
    test_loopback_sample_count()
    test_pre_buffer_updates()
    test_context_manager_clean_exit()
    test_frame_dtype_and_shape()
    print("==========================")
