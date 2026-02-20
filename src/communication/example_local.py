"""
communication/example_local.py
--------------------------------
Example: read directly from the local microphone via MicrophoneStream.

Run from src/:
    python communication/example_local.py

This is the simplest case — no network involved.
MicrophoneStream is re-exported through the communication package so
main.py can import everything from a single place.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from communication import MicrophoneStream

CAPTURE_SR   = 48000
FRAME_SEC    = 0.10   # 100 ms per frame = 4 800 samples
PRE_SECONDS  = 0.5
N_FRAMES     = 30     # read 30 frames (~3 seconds) then stop


def main():
    mic = MicrophoneStream(
        capture_sr  = CAPTURE_SR,
        frame_sec   = FRAME_SEC,
        pre_seconds = PRE_SECONDS,
    )

    print(f"Local mic  sr={CAPTURE_SR} Hz  frame={int(CAPTURE_SR * FRAME_SEC)} samples")
    print(f"Reading {N_FRAMES} frames ({N_FRAMES * FRAME_SEC:.1f} s) ...")

    with mic:
        for i in range(N_FRAMES):
            frame = mic.read()
            rms   = float(np.sqrt(np.mean(frame ** 2)))
            print(f"  frame {i+1:02d}  shape={frame.shape}  dtype={frame.dtype}  rms={rms:.5f}")

    print("Done.")


if __name__ == "__main__":
    main()
