"""
src/main.py
-----------
Real-time sneeze detection entry point for Raspberry Pi.

Wires together:
  - ml_model      : TFLite model + preprocessing (v4)
  - microphone    : sounddevice-based microphone capture
  - output_feature: speaker alert + ST7789 LCD animation

Asset directory layout (matches raspi/sneeze-detection/):
  ~/Documents/sneeze-detection/
    images/  idle.png  detect1.png  detect2.png  detect3.png
    sounds/  bless_you.wav
    weights/ v4_model.tflite  v4_norm_stats.npz
"""

import time
from pathlib import Path

import numpy as np

from ml_model import LiteModel, load_stats, preproc, resample_to_model_sr, rms
from ml_model import config as cfg
from microphone import MicrophoneStream
from output_feature import SpeakerOutput, LCD, LCDAnimator

# ---------------------------------------------------------------------- #
# Asset paths (same layout as raspi/sneeze-detection/)                   #
# ---------------------------------------------------------------------- #
BASE_DIR   = Path.home() / "Documents" / "sneeze-detection"
IMAGES_DIR = BASE_DIR / "images"
SOUNDS_DIR = BASE_DIR / "sounds"
ANIM_FPS   = 12.0

print(f"STARTING SNEEZE DETECTOR (model: {cfg.TFLITE_PATH})")


def main():
    # ------------------------------------------------------------------ #
    # Model + normalisation stats                                         #
    # ------------------------------------------------------------------ #
    print("Loading model and stats...")
    mu, sdv = load_stats(cfg.STATS_PATH)
    model   = LiteModel(cfg.TFLITE_PATH)

    # ------------------------------------------------------------------ #
    # Microphone                                                          #
    # ------------------------------------------------------------------ #
    print("Starting microphone...")
    mic = MicrophoneStream(
        capture_sr  = cfg.CAPTURE_SR,
        frame_sec   = cfg.FRAME_SEC,
        pre_seconds = cfg.PRE_SECONDS,
        device      = cfg.INPUT_DEVICE,
    )
    print(f"Configuration: CAPTURE_SR={cfg.CAPTURE_SR}, FRAME_SEC={cfg.FRAME_SEC}, "
          f"PRE_SECONDS={cfg.PRE_SECONDS}, INPUT_DEVICE={cfg.INPUT_DEVICE}")

    # ------------------------------------------------------------------ #
    # Output: speaker + LCD                                               #
    # ------------------------------------------------------------------ #
    speaker = SpeakerOutput(playback_sr=cfg.MODEL_SR)

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
    target_samples_capture = int(cfg.CAPTURE_SR * cfg.CLIP_SECONDS)

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

                if capturing and captured_len >= target_samples_capture:
                    y48 = np.concatenate(captured, axis=0)

                    if len(y48) > target_samples_capture:
                        y48 = y48[:target_samples_capture]
                    elif len(y48) < target_samples_capture:
                        y48 = np.pad(y48, (0, target_samples_capture - len(y48)))

                    y16  = resample_to_model_sr(y48, cfg.CAPTURE_SR)
                    x_in = preproc(y16, mu, sdv)
                    p    = model.predict_proba(x_in)

                    if p >= cfg.PROB_TH:
                        print("Bless you!")
                        speaker.alert("Bless you!")   # beep + print
                        animator.trigger()             # detect1→2→3 → idle (daemon thread)
                        ignore_until = time.time() + cfg.COOLDOWN_SEC

                    capturing    = False
                    captured     = []
                    captured_len = 0
                    mic.pre_buffer.clear()

        except KeyboardInterrupt:
            print("STOP")


if __name__ == "__main__":
    main()
