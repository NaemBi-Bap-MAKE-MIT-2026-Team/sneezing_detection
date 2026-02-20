"""
src/main.py
-----------
Real-time sneeze detection entry point for Raspberry Pi.

Wires together:
  - communication   : unified audio-input layer (local mic OR network stream)
  - ml_model        : TFLite model + v4 preprocessing
  - output_feature  : ST7789 LCD animation + speaker alert

Asset directory layout (matches raspi/sneeze-detection/):
  ~/Documents/sneeze-detection/
    images/  idle.png  detect1.png  detect2.png  detect3.png
    sounds/  bless_you.wav
    weights/ v4_model.tflite  v4_norm_stats.npz

Usage
-----
# Local microphone (default):
python main.py

# Receive audio from send.py running on another device:
python main.py --network --recv-host 0.0.0.0 --recv-port 12345

# Disable LCD (e.g. no hardware attached):
python main.py --no-lcd
"""

import argparse
import time
from pathlib import Path

import numpy as np

from ml_model import LiteModel, load_stats, preproc, resample_to_model_sr, rms
from ml_model import config as cfg
from communication import MicrophoneStream, NetworkMicStream
from output_feature import SpeakerOutput, LCD, LCDAnimator

# ---------------------------------------------------------------------- #
# Asset paths                                                             #
# ---------------------------------------------------------------------- #
BASE_DIR   = Path.home() / "Documents" / "sneeze-detection"
IMAGES_DIR = BASE_DIR / "images"
ANIM_FPS   = 12.0


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #

def _build_mic(args):
    """Return a MicrophoneStream or NetworkMicStream depending on --network."""
    if args.network:
        return NetworkMicStream(
            host        = args.recv_host,
            port        = args.recv_port,
            capture_sr  = cfg.CAPTURE_SR,
            frame_sec   = cfg.FRAME_SEC,
            pre_seconds = cfg.PRE_SECONDS,
        )
    return MicrophoneStream(
        capture_sr  = cfg.CAPTURE_SR,
        frame_sec   = cfg.FRAME_SEC,
        pre_seconds = cfg.PRE_SECONDS,
        device      = cfg.INPUT_DEVICE,
    )


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time sneeze detector")
    ap.add_argument(
        "--network",
        action="store_true",
        help="Receive audio from send.py over UDP instead of using a local mic",
    )
    ap.add_argument(
        "--recv-host", default="0.0.0.0",
        help="[--network] UDP bind address  (default: 0.0.0.0)",
    )
    ap.add_argument(
        "--recv-port", type=int, default=12345,
        help="[--network] UDP port          (default: 12345)",
    )
    ap.add_argument(
        "--no-lcd",
        action="store_true",
        help="Disable the ST7789 LCD driver (useful when no display is attached)",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    # Model + normalisation stats                                         #
    # ------------------------------------------------------------------ #
    print("Loading model and stats...")
    mu, sdv = load_stats(cfg.STATS_PATH)
    model   = LiteModel(cfg.TFLITE_PATH)

    # ------------------------------------------------------------------ #
    # Microphone source                                                   #
    # ------------------------------------------------------------------ #
    mic  = _build_mic(args)
    mode = (f"network  bind={args.recv_host}:{args.recv_port}"
            if args.network else
            f"local  device={cfg.INPUT_DEVICE}  sr={cfg.CAPTURE_SR}")
    print(f"Microphone: {mode}")

    # ------------------------------------------------------------------ #
    # Output: speaker + LCD                                               #
    # ------------------------------------------------------------------ #
    speaker  = SpeakerOutput(playback_sr=cfg.MODEL_SR)

    if args.no_lcd:
        animator = None
        print("LCD: disabled (--no-lcd)")
    else:
        lcd      = LCD()
        animator = LCDAnimator(
            lcd         = lcd,
            idle_path   = IMAGES_DIR / "idle.png",
            frame_paths = [
                IMAGES_DIR / "detect1.png",
                IMAGES_DIR / "detect2.png",
                IMAGES_DIR / "detect3.png",
            ],
            fps = ANIM_FPS,
        )
        animator.start()  # show idle frame on LCD

    # ------------------------------------------------------------------ #
    # Detection loop                                                      #
    # ------------------------------------------------------------------ #
    target_samples = int(cfg.CAPTURE_SR * cfg.CLIP_SECONDS)

    capturing    = False
    captured     = []
    captured_len = 0
    ignore_until = 0.0

    print("STREAM START (Ctrl+C to stop)")

    with mic:
        try:
            while True:
                x   = mic.read()
                now = time.time()

                # Cooldown: keep pre-buffer warm but skip inference
                if now < ignore_until:
                    mic.pre_buffer.append(x)
                    continue

                x_rms = rms(x)

                if not capturing:
                    mic.pre_buffer.append(x)
                    if x_rms >= cfg.RMS_TRIGGER_TH:
                        capturing    = True
                        captured     = list(mic.pre_buffer)
                        captured.append(x)
                        captured_len = sum(len(c) for c in captured)
                    else:
                        continue
                else:
                    captured.append(x)
                    captured_len += len(x)

                if capturing and captured_len >= target_samples:
                    y48 = np.concatenate(captured, axis=0)

                    if len(y48) > target_samples:
                        y48 = y48[:target_samples]
                    elif len(y48) < target_samples:
                        y48 = np.pad(y48, (0, target_samples - len(y48)))

                    y16  = resample_to_model_sr(y48, cfg.CAPTURE_SR)
                    x_in = preproc(y16, mu, sdv)
                    p    = model.predict_proba(x_in)

                    if p >= cfg.PROB_TH:
                        print("Bless you!")
                        speaker.alert("Bless you!")   # beep + print
                        if animator:
                            animator.trigger()         # detect1→2→3 → idle (daemon thread)
                        ignore_until = time.time() + cfg.COOLDOWN_SEC

                    capturing    = False
                    captured     = []
                    captured_len = 0
                    mic.pre_buffer.clear()

        except KeyboardInterrupt:
            print("STOP")


if __name__ == "__main__":
    main()
