"""
src/main.py
-----------
Real-time sneeze detection entry point for Raspberry Pi.

Wires together:
  - ml_model   : TFLite model + preprocessing (v4)
  - microphone : sounddevice-based microphone capture
  - output_feature : speaker / alert output
"""

import time
import numpy as np

from ml_model import LiteModel, load_stats, preproc, resample_to_model_sr, rms
from ml_model import config as cfg
from microphone import MicrophoneStream
from output_feature import SpeakerOutput


def main():
    # ------------------------------------------------------------------ #
    # Initialise model and normalisation stats                            #
    # ------------------------------------------------------------------ #
    mu, sdv = load_stats(cfg.STATS_PATH)
    model   = LiteModel(cfg.TFLITE_PATH)

    # ------------------------------------------------------------------ #
    # Initialise microphone and output modules                            #
    # ------------------------------------------------------------------ #
    mic    = MicrophoneStream(
        capture_sr  = cfg.CAPTURE_SR,
        frame_sec   = cfg.FRAME_SEC,
        pre_seconds = cfg.PRE_SECONDS,
        device      = cfg.INPUT_DEVICE,
    )
    output = SpeakerOutput(playback_sr=cfg.MODEL_SR)

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
                        output.alert("bless you!")
                        ignore_until = time.time() + cfg.COOLDOWN_SEC

                    capturing    = False
                    captured     = []
                    captured_len = 0
                    mic.pre_buffer.clear()

        except KeyboardInterrupt:
            print("STOP")


if __name__ == "__main__":
    main()
