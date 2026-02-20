"""
src/main.py
-----------
RPi inference + LCD + speaker pipeline (Hybrid Burst mode)

Directory layout (actual):
  ~/sneezing_detection/src/
    main.py
    v4_model.tflite
    v4_norm_stats.npz
    output_feature/
      images/ idle.png detect1.png detect2.png detect3.png
      sounds/ bless_you.wav

Run
---
# Network audio (from laptop send.py):
python -m src.main --network --recv-host 0.0.0.0 --recv-port 12345

# Local mic:
python -m src.main
"""

import argparse
import time
import threading
import subprocess
from pathlib import Path

import numpy as np

from ml_model import LiteModel, load_stats, preproc, resample_to_model_sr, rms
from ml_model import config as cfg
from communication import MicrophoneStream, NetworkMicStream
from output_feature import LCD, LCDAnimator

# ---------------------------------------------------------------------- #
# Paths (fixed to your current repo layout)                               #
# ---------------------------------------------------------------------- #
SRC_DIR = Path(__file__).resolve().parent  # ~/sneezing_detection/src

TFLITE_PATH = SRC_DIR / "v4_model.tflite"
STATS_PATH  = SRC_DIR / "v4_norm_stats.npz"

OUTPUT_DIR = SRC_DIR / "output_feature"
IMAGES_DIR = OUTPUT_DIR / "images"
SOUNDS_DIR = OUTPUT_DIR / "sounds"
BLESS_WAV  = SOUNDS_DIR / "bless_you.wav"

ANIM_FPS = 12.0


def require(p: Path, label: str):
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")


def play_bless_wav_async(wav_path: Path) -> None:
    wav_path = Path(wav_path)

    def _run():
        try:
            subprocess.run(["aplay", "-q", str(wav_path)], check=False)
        except Exception as e:
            print(f"[WARN] aplay failed: {e}")

    if wav_path.exists():
        threading.Thread(target=_run, daemon=True).start()
    else:
        print(f"[WARN] bless wav missing: {wav_path}")


def _build_mic(args):
    if args.network:
        return NetworkMicStream(
            host=args.recv_host,
            port=args.recv_port,
            capture_sr=cfg.CAPTURE_SR,
            frame_sec=cfg.FRAME_SEC,
            pre_seconds=0.0,  # hybrid burst does not need pre_buffer
        )
    return MicrophoneStream(
        capture_sr=cfg.CAPTURE_SR,
        frame_sec=cfg.FRAME_SEC,
        pre_seconds=0.0,   # hybrid burst does not need pre_buffer
        device=cfg.INPUT_DEVICE,
    )


class HybridBurstDetector:
    """
    Hybrid Burst (Guard RMS + Burst Sliding Inference)

    - Always keeps a 2s ring buffer (48k domain).
    - IDLE: compute RMS on guard frames only.
    - Trigger: enter BURST for BURST_SECONDS.
    - BURST: every HOP_SEC, run inference on current 2s window.
    - Detect: call on_detect(prob), enter cooldown.
    """

    def __init__(
        self,
        *,
        model,
        mu: np.ndarray,
        sdv: np.ndarray,
        capture_sr: int,
        clip_seconds: float,
        guard_frame_sec: float,
        rms_trigger_th: float,
        burst_seconds: float,
        hop_sec: float,
        prob_th: float,
        cooldown_sec: float,
    ):
        self.model = model
        self.mu = mu
        self.sdv = sdv

        self.capture_sr = int(capture_sr)
        self.clip_seconds = float(clip_seconds)

        self.guard_frame_sec = float(guard_frame_sec)
        self.rms_trigger_th = float(rms_trigger_th)

        self.burst_seconds = float(burst_seconds)
        self.hop_sec = float(hop_sec)

        self.prob_th = float(prob_th)
        self.cooldown_sec = float(cooldown_sec)

        self.win_samples = int(self.capture_sr * self.clip_seconds)
        self.ring = np.zeros(self.win_samples, dtype=np.float32)
        self.filled = 0

        self.guard_samples = max(1, int(self.capture_sr * self.guard_frame_sec))
        self.guard_acc = np.zeros(0, dtype=np.float32)

        self.hop_samples = max(1, int(self.capture_sr * self.hop_sec))
        self.since_last_infer = 0

        self.STATE_IDLE = 0
        self.STATE_BURST = 1
        self.state = self.STATE_IDLE

        self.burst_until = 0.0
        self.ignore_until = 0.0

    def feed(self, x: np.ndarray, now: float, on_detect):
        x = np.asarray(x, dtype=np.float32).reshape(-1)

        # always update ring
        self._push_ring(x)
        self.since_last_infer += len(x)

        # cooldown
        if now < self.ignore_until:
            self.state = self.STATE_IDLE
            self.guard_acc = np.zeros(0, dtype=np.float32)
            return

        # IDLE: RMS guard only
        if self.state == self.STATE_IDLE:
            self.guard_acc = np.concatenate([self.guard_acc, x])
            while len(self.guard_acc) >= self.guard_samples:
                g = self.guard_acc[:self.guard_samples]
                self.guard_acc = self.guard_acc[self.guard_samples:]

                g_rms = rms(g)
                if g_rms >= self.rms_trigger_th:
                    self.state = self.STATE_BURST
                    self.burst_until = now + self.burst_seconds
                    self.since_last_infer = self.hop_samples  # force immediate infer
                    self.guard_acc = np.zeros(0, dtype=np.float32)
                    break

        # BURST: sliding inference
        if self.state == self.STATE_BURST:
            if now >= self.burst_until:
                self.state = self.STATE_IDLE
                self.since_last_infer = 0
                return

            if self.filled < self.win_samples:
                return

            if self.since_last_infer >= self.hop_samples:
                self.since_last_infer = 0
                p = self._infer_current_window()

                if p >= self.prob_th:
                    on_detect(p, now)
                    self.ignore_until = time.time() + self.cooldown_sec
                    self.state = self.STATE_IDLE
                    self.since_last_infer = 0
                    self.guard_acc = np.zeros(0, dtype=np.float32)

    def _push_ring(self, x: np.ndarray):
        n = len(x)
        if n >= self.win_samples:
            self.ring[:] = x[-self.win_samples:]
            self.filled = self.win_samples
            return
        self.ring = np.roll(self.ring, -n)
        self.ring[-n:] = x
        self.filled = min(self.win_samples, self.filled + n)

    def _infer_current_window(self) -> float:
        y48 = self.ring.copy()
        y16 = resample_to_model_sr(y48, self.capture_sr)
        x_in = preproc(y16, self.mu, self.sdv)
        return float(self.model.predict_proba(x_in))


def main() -> None:
    ap = argparse.ArgumentParser(description="Real-time sneeze detector (hybrid burst)")
    ap.add_argument("--network", action="store_true")
    ap.add_argument("--recv-host", default="0.0.0.0")
    ap.add_argument("--recv-port", type=int, default=12345)
    ap.add_argument("--no-lcd", action="store_true")
    args = ap.parse_args()

    # fail-fast asset checks
    require(TFLITE_PATH, "TFLite model")
    require(STATS_PATH, "Norm stats")
    require(IMAGES_DIR, "Images dir")
    require(SOUNDS_DIR, "Sounds dir")
    require(BLESS_WAV, "Bless wav")

    require(IMAGES_DIR / "idle.png", "idle image")
    require(IMAGES_DIR / "detect1.png", "detect1 image")
    require(IMAGES_DIR / "detect2.png", "detect2 image")
    require(IMAGES_DIR / "detect3.png", "detect3 image")

    # model + stats
    print("Loading model and stats...")
    mu, sdv = load_stats(STATS_PATH)
    model = LiteModel(TFLITE_PATH)

    # mic
    mic = _build_mic(args)
    mode = (f"network bind={args.recv_host}:{args.recv_port}"
            if args.network else
            f"local device={cfg.INPUT_DEVICE} sr={cfg.CAPTURE_SR}")
    print(f"Microphone: {mode}")

    # LCD
    animator = None
    if args.no_lcd:
        print("LCD: disabled (--no-lcd)")
    else:
        lcd = LCD()
        animator = LCDAnimator(
            lcd=lcd,
            idle_path=IMAGES_DIR / "idle.png",
            frame_paths=[
                IMAGES_DIR / "detect1.png",
                IMAGES_DIR / "detect2.png",
                IMAGES_DIR / "detect3.png",
            ],
            fps=ANIM_FPS,
        )
        animator.start()

    # hybrid burst params (논의 결론 기준)
    BURST_SECONDS = 3.0
    HOP_SEC = 0.5

    detector = HybridBurstDetector(
        model=model,
        mu=mu,
        sdv=sdv,
        capture_sr=cfg.CAPTURE_SR,
        clip_seconds=cfg.CLIP_SECONDS,
        guard_frame_sec=cfg.FRAME_SEC,
        rms_trigger_th=cfg.RMS_TRIGGER_TH,
        burst_seconds=BURST_SECONDS,
        hop_sec=HOP_SEC,
        prob_th=cfg.PROB_TH,
        cooldown_sec=cfg.COOLDOWN_SEC,
    )

    def on_detect(p: float, now: float):
        print(f"Bless you! p={p:.3f}")
        # 동시에 시작
        play_bless_wav_async(BLESS_WAV)
        if animator:
            animator.trigger()

    print("STREAM START (Ctrl+C to stop)")
    print(f"mode: hybrid burst, burst={BURST_SECONDS}s, hop={HOP_SEC}s")

    with mic:
        try:
            while True:
                x = mic.read()
                now = time.time()
                detector.feed(x, now, on_detect)
        except KeyboardInterrupt:
            print("STOP")


if __name__ == "__main__":
    main()