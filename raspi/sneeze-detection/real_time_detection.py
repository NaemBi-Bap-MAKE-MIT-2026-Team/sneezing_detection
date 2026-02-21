#!/usr/bin/env python3
# real_time_detection.py
# Requirements:
# - 2-second window + 0.5-second hop for real-time inference
# - On detection, print "Bless you!" to terminal
# - LCD: idle frame (default) + on detection play detect1->detect2->detect3 sequentially (required)
# - Sound: play sounds/bless_you.wav (required)
#
# Directory layout:
# ~/Documents/sneeze-detection/
#   images/idle.png detect1.png detect2.png detect3.png
#   sounds/bless_you.wav
#   weights/v4_model.tflite v4_norm_stats.npz

import argparse
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import librosa
import sounddevice as sd
from PIL import Image


# =========================
# Model / preprocessing config (based on bench_e2e.py)
# =========================
MODEL_SR = 16000
CLIP_SECONDS = 2.0

N_MELS = 64
N_FFT = 400
HOP = 160
CENTER = False
TARGET_RMS = 0.1


def rms(x, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + eps))


def normalize_rms(x, target=TARGET_RMS):
    x = np.asarray(x, dtype=np.float32)
    r = rms(x)
    if r > 1e-6:
        x = x * (target / (r + 1e-8))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def logmel(y_16k_2s: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y_16k_2s,
        sr=MODEL_SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
        center=CENTER,
    )
    return np.log(S + 1e-6).T.astype(np.float32)  # (frames, mels)


def load_stats(stats_path: Path):
    st = np.load(str(stats_path), allow_pickle=True)
    mu = st["mu"].astype(np.float32)
    sdv = st["sd"].astype(np.float32)

    if mu.ndim == 3 and mu.shape[0] == 1 and mu.shape[1] == 1:
        mu = mu.reshape(-1)
    if sdv.ndim == 3 and sdv.shape[0] == 1 and sdv.shape[1] == 1:
        sdv = sdv.reshape(-1)

    if mu.ndim == 3 and mu.shape[0] == 1:
        mu = mu[0]
    if sdv.ndim == 3 and sdv.shape[0] == 1:
        sdv = sdv[0]

    mu = mu.reshape(-1).astype(np.float32)
    sdv = sdv.reshape(-1).astype(np.float32)
    return mu, sdv


def preproc(y16: np.ndarray, mu: np.ndarray, sdv: np.ndarray) -> np.ndarray:
    target = int(MODEL_SR * CLIP_SECONDS)
    y16 = np.asarray(y16, dtype=np.float32)

    if len(y16) > target:
        y16 = y16[:target]
    elif len(y16) < target:
        y16 = np.pad(y16, (0, target - len(y16)))

    y16 = normalize_rms(y16)
    f = logmel(y16)  # (frames, mels)
    fn = (f - mu[None, :]) / (sdv[None, :] + 1e-6)
    return fn[None, :, :, None].astype(np.float32)


def load_interpreter(model_path: Path):
    # Priority: ai_edge_litert -> tflite_runtime -> tensorflow
    try:
        from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
        interp = TFLiteInterpreter(model_path=str(model_path))
        return interp
    except Exception:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
        interp = TFLiteInterpreter(model_path=str(model_path))
        return interp
    except Exception:
        pass

    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=str(model_path))
        return interp
    except Exception as e:
        raise RuntimeError(f"Failed to load TFLite Interpreter: {e}") from e


# =========================
# LCD (required)
# =========================
class LCD:
    def __init__(self):
        try:
            import st7789
        except Exception as e:
            raise RuntimeError(f"LCD driver (st7789) import failed: {e}")

        self.disp = st7789.ST7789(
            rotation=90,
            port=0,
            cs=1,
            dc=9,
            backlight=13,
            spi_speed_hz=80_000_000,
        )

    def show(self, img_240: Image.Image):
        self.disp.display(img_240)


def load_frame(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB").resize((240, 240))


# =========================
# Sound playback (required)
# =========================
def play_wav_aplay(wav_path: Path, device: str | None):
    if not wav_path.exists():
        raise FileNotFoundError(wav_path)
    cmd = ["aplay"]
    if device:
        cmd += ["-D", device]
    cmd += [str(wav_path)]
    subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# =========================
# Ring buffer (mono)
# =========================
class RingBuffer:
    def __init__(self, capacity_samples: int):
        self.buf = np.zeros((capacity_samples,), dtype=np.float32)
        self.cap = int(capacity_samples)
        self.w = 0
        self.full = False
        self.lock = threading.Lock()

    def push(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = x.size
        if n <= 0:
            return
        if n >= self.cap:
            x = x[-self.cap:]
            n = x.size

        with self.lock:
            end = self.w + n
            if end <= self.cap:
                self.buf[self.w:end] = x
            else:
                k = self.cap - self.w
                self.buf[self.w:] = x[:k]
                self.buf[:end - self.cap] = x[k:]
            self.w = (self.w + n) % self.cap
            if not self.full and end >= self.cap:
                self.full = True

    def read_last(self, n: int) -> np.ndarray:
        n = int(n)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)

        with self.lock:
            available = self.cap if self.full else self.w
            if available <= 0:
                return np.zeros((0,), dtype=np.float32)
            if n > available:
                n = available

            start = (self.w - n) % self.cap
            if start < self.w or not self.full:
                if start < self.w:
                    return self.buf[start:self.w].copy()
                return self.buf[:n].copy()

            part1 = self.buf[start:].copy()
            part2 = self.buf[:self.w].copy()
            return np.concatenate([part1, part2], axis=0)


# =========================
# Main
# =========================
def main():
    base_dir = Path.home() / "Documents" / "sneeze-detection"
    weights_dir = base_dir / "weights"
    images_dir = base_dir / "images"
    sounds_dir = base_dir / "sounds"

    ap = argparse.ArgumentParser()
    ap.add_argument("--input-device", default=None, help="sounddevice input device index or name (system default if omitted)")
    ap.add_argument("--capture-sr", type=int, default=48000)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--block-sec", type=float, default=0.02)   # 20ms
    ap.add_argument("--hop-sec", type=float, default=0.50)     # 0.5s hop
    ap.add_argument("--rms-gate", type=float, default=0.008)   # skip inference if too quiet
    ap.add_argument("--prob-th", type=float, default=0.90)
    ap.add_argument("--cooldown-sec", type=float, default=1.5)

    ap.add_argument("--aplay-device", default=None, help="aplay -D device (e.g. hw:0,0). System default if omitted.")
    ap.add_argument("--anim-fps", type=float, default=12.0)
    args = ap.parse_args()

    # Required files
    tflite_path = weights_dir / "v4_model.tflite"
    stats_path = weights_dir / "v4_norm_stats.npz"
    bless_path = sounds_dir / "bless_you.wav"

    idle_path = images_dir / "idle.png"
    d1_path = images_dir / "detect1.png"
    d2_path = images_dir / "detect2.png"
    d3_path = images_dir / "detect3.png"

    for p in [tflite_path, stats_path, bless_path, idle_path, d1_path, d2_path, d3_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    # Loading
    mu, sdv = load_stats(stats_path)
    interp = load_interpreter(tflite_path)
    interp.allocate_tensors()
    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    lcd = LCD()
    idle = load_frame(idle_path)
    detect_frames = [load_frame(d1_path), load_frame(d2_path), load_frame(d3_path)]

    # Show idle frame once
    lcd.show(idle)

    # Audio input setup
    input_device = args.input_device
    if input_device is not None:
        try:
            input_device = int(input_device)
        except Exception:
            input_device = str(input_device)

    capture_sr = int(args.capture_sr)
    channels = int(args.channels)
    blocksize = int(max(1, round(capture_sr * float(args.block_sec))))

    need_samples = int(capture_sr * CLIP_SECONDS)
    ring_cap = int(capture_sr * (CLIP_SECONDS + 1.0))  # 3-second buffer
    ring = RingBuffer(ring_cap)

    q = queue.Queue(maxsize=80)

    def audio_cb(indata, frames_count, time_info, status):
        try:
            if channels == 1:
                x = indata[:, 0].astype(np.float32, copy=True)
            else:
                x = np.mean(indata.astype(np.float32), axis=1)
            q.put_nowait(x)
        except queue.Full:
            pass

    # Prevent duplicate event execution
    event_lock = threading.Lock()
    event_active = False

    def run_event():
        nonlocal event_active
        with event_lock:
            if event_active:
                return
            event_active = True

        try:
            # LCD animation: detect1 -> detect2 -> detect3 (based on fps)
            dt = 1.0 / max(1.0, float(args.anim_fps))
            for fr in detect_frames:
                lcd.show(fr)
                time.sleep(dt)

            # Sound playback
            play_wav_aplay(bless_path, args.aplay_device)

        finally:
            # Return to idle
            lcd.show(idle)
            with event_lock:
                event_active = False

    # Scheduling
    hop_sec = float(args.hop_sec)
    next_infer_t = time.time() + hop_sec
    ignore_until = 0.0

    try:
        with sd.InputStream(
            device=input_device,
            channels=channels,
            samplerate=capture_sr,
            blocksize=blocksize,
            dtype="float32",
            callback=audio_cb,
        ):
            while True:
                x = q.get()
                ring.push(x)

                now = time.time()
                if now < next_infer_t:
                    continue
                while next_infer_t <= now:
                    next_infer_t += hop_sec

                if now < ignore_until:
                    continue

                y = ring.read_last(need_samples)
                if y.size < need_samples:
                    continue

                if rms(y) < float(args.rms_gate):
                    continue

                # 48k -> 16k
                y16 = librosa.resample(y.astype(np.float32), orig_sr=capture_sr, target_sr=MODEL_SR).astype(np.float32)

                x_in = preproc(y16, mu, sdv)
                interp.set_tensor(in_det["index"], x_in.astype(in_det["dtype"], copy=False))
                interp.invoke()
                p = float(interp.get_tensor(out_det["index"]).reshape(-1)[0])

                if p >= float(args.prob_th):
                    print("Bless you!")
                    threading.Thread(target=run_event, daemon=True).start()
                    ignore_until = time.time() + float(args.cooldown_sec)

    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
