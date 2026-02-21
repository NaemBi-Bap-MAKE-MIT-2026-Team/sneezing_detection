"""
communication/example_network.py
----------------------------------
Example: stream audio from send.py and consume it via NetworkMicStream.

Two modes depending on the --role argument:

  sender   — captures the local microphone and sends frames over UDP
  receiver — receives UDP frames and prints RMS values (like a local mic)

Run from src/ in two separate terminals:

  Terminal A (sender):
      python communication/example_network.py --role sender --host 127.0.0.1

  Terminal B (receiver):
      python communication/example_network.py --role receiver

The receiver output is identical to example_local.py, demonstrating that
NetworkMicStream is a transparent drop-in for MicrophoneStream.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from communication import AudioSender, NetworkMicStream

CAPTURE_SR  = 48000
FRAME_SEC   = 0.10
PRE_SECONDS = 0.5
PORT        = 12345
N_FRAMES    = 30     # receiver reads this many frames then stops


def run_sender(host: str, port: int) -> None:
    print(f"[sender] Streaming mic → {host}:{port}  (Ctrl+C to stop)")
    sender = AudioSender(
        host       = host,
        port       = port,
        capture_sr = CAPTURE_SR,
        block_ms   = 10.0,
    )
    sender.run()
    print("[sender] STOP")


def run_receiver(host: str, port: int) -> None:
    mic = NetworkMicStream(
        host        = host,
        port        = port,
        capture_sr  = CAPTURE_SR,
        frame_sec   = FRAME_SEC,
        pre_seconds = PRE_SECONDS,
    )

    print(f"[receiver] Listening on {host}:{port}  "
          f"frame={int(CAPTURE_SR * FRAME_SEC)} samples")
    print(f"[receiver] Reading {N_FRAMES} frames ({N_FRAMES * FRAME_SEC:.1f} s) ...")

    with mic:
        for i in range(N_FRAMES):
            frame = mic.read()                          # identical to MicrophoneStream.read()
            rms   = float(np.sqrt(np.mean(frame ** 2)))
            print(f"  frame {i+1:02d}  shape={frame.shape}  dtype={frame.dtype}  rms={rms:.5f}")

    print("[receiver] Done.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Network mic example")
    ap.add_argument("--role", choices=["sender", "receiver"], required=True)
    ap.add_argument("--host", default="127.0.0.1",
                    help="sender: destination IP  |  receiver: bind address")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    if args.role == "sender":
        run_sender(args.host, args.port)
    else:
        run_receiver("0.0.0.0", args.port)


if __name__ == "__main__":
    main()
