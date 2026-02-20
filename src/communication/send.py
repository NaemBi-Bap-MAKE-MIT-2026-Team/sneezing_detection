"""
communication/send.py
----------------------
AudioSender: captures microphone audio via MicrophoneStream and streams
each frame immediately to recv.py (NetworkMicStream) over UDP.

Internally this module uses the same MicrophoneStream class that main.py
uses for local capture, so capture behaviour is fully consistent.

Usage — as a class
------------------
from communication.send import AudioSender

sender = AudioSender(host="192.168.1.42", port=12345)
sender.run()   # blocks; Ctrl+C to stop

Usage — as a script (CLI)
--------------------------
# Stream to recv.py on the same machine:
python send.py

# Stream to a remote Raspberry Pi:
python send.py --host 192.168.1.42 --port 12345

# Custom sample rate and block size:
python send.py --capture-sr 48000 --block-ms 10

CLI arguments
-------------
--host          Destination IP (recv.py device)      default: 127.0.0.1
--port          UDP port                              default: 12345
--capture-sr    Microphone sample rate (Hz)           default: 48000
--block-ms      Packet size in ms (latency knob)      default: 10
--device        sounddevice input device index/name   default: system default
"""

import argparse
import socket
import sys
from pathlib import Path
from typing import Optional

# Allow running as a standalone script from the src/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from microphone import MicrophoneStream


class AudioSender:
    """Capture microphone audio and stream it over UDP to a NetworkMicStream receiver.

    Each frame produced by MicrophoneStream is immediately sent as a raw
    float32 UDP packet — no extra buffering, no encoding overhead.

    Parameters
    ----------
    host       : Destination IP address of the recv.py device.
    port       : Destination UDP port (must match NetworkMicStream port).
    capture_sr : Microphone sample rate (Hz).
    block_ms   : Duration of each sent packet (ms). Smaller = lower latency.
                 Default 10 ms → 480 samples @ 48 kHz → ~100 packets/s.
    device     : sounddevice input device index or name. None = system default.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12345,
        capture_sr: int = 48000,
        block_ms: float = 10.0,
        device=None,
    ):
        self._dest = (host, port)
        self._mic  = MicrophoneStream(
            capture_sr  = capture_sr,
            frame_sec   = block_ms / 1000.0,
            pre_seconds = 0.0,   # no pre-roll needed on the sender side
            device      = device,
        )
        self._sock: Optional[socket.socket] = None

    def run(self) -> None:
        """Open the microphone and start streaming. Blocks until KeyboardInterrupt."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            with self._mic:
                while True:
                    frame = self._mic.read()
                    self._sock.sendto(frame.tobytes(), self._dest)
        except KeyboardInterrupt:
            pass
        finally:
            self._sock.close()
            self._sock = None


# ---------------------------------------------------------------------- #
# CLI entry point                                                         #
# ---------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Stream microphone audio over UDP")
    ap.add_argument("--host",       default="127.0.0.1",
                    help="Destination IP address of the recv.py device")
    ap.add_argument("--port",       type=int, default=12345)
    ap.add_argument("--capture-sr", type=int, default=48000,
                    help="Microphone sample rate in Hz")
    ap.add_argument("--block-ms",   type=float, default=10.0,
                    help="Packet size in ms (default 10 ms → ~100 pkt/s)")
    ap.add_argument("--device",     default=None,
                    help="sounddevice input device index or name")
    args = ap.parse_args()

    device = args.device
    if device is not None:
        try:
            device = int(device)
        except ValueError:
            pass  # keep as string (device name)

    blocksize = max(1, int(args.capture_sr * args.block_ms / 1000.0))
    print(f"Streaming mic → {args.host}:{args.port}  "
          f"(sr={args.capture_sr} Hz, block={blocksize} samples / {args.block_ms:.0f} ms)")
    print("Ctrl+C to stop")

    sender = AudioSender(
        host       = args.host,
        port       = int(args.port),
        capture_sr = int(args.capture_sr),
        block_ms   = float(args.block_ms),
        device     = device,
    )
    sender.run()
    print("STOP")


if __name__ == "__main__":
    main()
