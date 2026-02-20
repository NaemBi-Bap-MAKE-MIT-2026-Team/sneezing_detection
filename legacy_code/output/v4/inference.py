#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
inference.py (Laptop)
- Idle: RMS guard only
- On RMS trigger: burst sliding inference for BURST_SECONDS
  - 2s ring buffer
  - run inference every HOP_SEC (0.5s default)
  - detect if prob >= PROB_TH
- On detect: play bless_you.wav (optional) + send UDP to Pi (optional)

Requirements:
  pip install numpy librosa sounddevice soundfile
  - For TFLite:
      pip install tflite-runtime
    or
      pip install tensorflow

Usage:
  python inference.py --model v4_model.tflite --stats v4_norm_stats.npz
"""

import argparse
import collections
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import sounddevice as sd

try:
    import soundfile as sf
except Exception:
    sf = None


# -------------------------
# TFLite interpreter
# -------------------------
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except Exception:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter


class LiteModel:
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(path)
        self.interp = TFLiteInterpreter(model_path=str(path))
        self.interp.allocate_tensors()
        self.in_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]

    def predict_proba(self, x: np.ndarray) -> float:
        self.interp.set_tensor(self.in_det["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out_det["index"]).reshape(-1)[0]
        return float(y)


# -------------------------
# Audio + feature pipeline (v4 compatible)
# -------------------------
def rms(x: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + eps))


def normalize_rms(x: np.ndarray, target: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    r = rms(x)
    if r > 1e-6:
        x = x * (target / (r + 1e-8))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


def load_stats(stats_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    st = np.load(str(stats_path), allow_pickle=True)
    mu = st["mu"].astype(np.float32)
    sdv = st["sd"].astype(np.float32)

    # 여러 포맷 대응
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


def logmel(
    y_16k_2s: np.ndarray,
    model_sr: int,
    n_fft: int,
    hop: int,
    n_mels: int,
    center: bool,
) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y_16k_2s,
        sr=model_sr,
        n_fft=n_fft,
        hop_length=hop,
        n_mels=n_mels,
        power=2.0,
        center=center,
    )
    return np.log(S + 1e-6).T.astype(np.float32)  # (frames, mels)


def preproc_v4(
    y16: np.ndarray,
    mu: np.ndarray,
    sdv: np.ndarray,
    *,
    model_sr: int,
    clip_seconds: float,
    target_rms: float,
    n_mels: int,
    n_fft: int,
    hop: int,
    center: bool,
) -> np.ndarray:
    target = int(model_sr * clip_seconds)
    y16 = np.asarray(y16, dtype=np.float32)

    if len(y16) > target:
        y16 = y16[:target]
    elif len(y16) < target:
        y16 = np.pad(y16, (0, target - len(y16)))

    y16 = normalize_rms(y16, target_rms)
    f = logmel(y16, model_sr, n_fft, hop, n_mels, center)  # (frames, mels)
    fn = (f - mu[None, :]) / (sdv[None, :] + 1e-6)
    return fn[None, :, :, None].astype(np.float32)  # (1, frames, mels, 1)


# -------------------------
# UDP notifier (optional)
# -------------------------
class UdpNotifier:
    def __init__(self, host: Optional[str], port: int):
        self.host = host
        self.port = port
        self.sock = None
        if host:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, msg: str) -> None:
        if not self.sock or not self.host:
            return
        try:
            self.sock.sendto(msg.encode("utf-8"), (self.host, self.port))
        except Exception:
            pass


# -------------------------
# WAV player (optional)
# -------------------------
def play_wav_async(path: Path) -> None:
    if sf is None:
        return
    if not path.exists():
        return

    def _run():
        try:
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
            y = data[:, 0]
            sd.play(y, sr, blocking=True)
        except Exception:
            pass

    th = threading.Thread(target=_run, daemon=True)
    th.start()


# -------------------------
# Main state machine
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("v4_model.tflite"))
    p.add_argument("--stats", type=Path, default=Path("v4_norm_stats.npz"))
    p.add_argument("--input-device", type=int, default=None)

    p.add_argument("--capture-sr", type=int, default=48000)
    p.add_argument("--model-sr", type=int, default=16000)
    p.add_argument("--clip-seconds", type=float, default=2.0)

    # v4 params
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--n-fft", type=int, default=400)
    p.add_argument("--hop", type=int, default=160)
    p.add_argument("--center", action="store_true", default=False)
    p.add_argument("--target-rms", type=float, default=0.1)

    # Guard + burst
    p.add_argument("--guard-frame-sec", type=float, default=0.10)
    p.add_argument("--rms-trigger-th", type=float, default=0.008)

    p.add_argument("--burst-seconds", type=float, default=3.0)
    p.add_argument("--chunk-sec", type=float, default=0.10)
    p.add_argument("--hop-sec", type=float, default=0.5)

    # Decision
    p.add_argument("--prob-th", type=float, default=0.90)
    p.add_argument("--cooldown-sec", type=float, default=1.5)

    # Actions
    p.add_argument("--play-wav", type=Path, default=Path("bless_you.wav"))
    p.add_argument("--no-audio", action="store_true", default=False)

    p.add_argument("--udp-host", type=str, default=None)
    p.add_argument("--udp-port", type=int, default=5005)
    p.add_argument("--udp-message", type=str, default="SNEEZE")

    args = p.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(args.model)
    if not args.stats.exists():
        raise FileNotFoundError(args.stats)

    mu, sdv = load_stats(args.stats)
    model = LiteModel(args.model)
    udp = UdpNotifier(args.udp_host, args.udp_port)

    CAPTURE_SR = int(args.capture_sr)
    MODEL_SR = int(args.model_sr)

    win_samples = int(CAPTURE_SR * args.clip_seconds)   # 2초(48k) 링버퍼 크기
    chunk_samples = int(CAPTURE_SR * args.chunk_sec)
    guard_samples = int(CAPTURE_SR * args.guard_frame_sec)
    hop_samples = int(CAPTURE_SR * args.hop_sec)

    if chunk_samples <= 0 or guard_samples <= 0 or hop_samples <= 0:
        raise ValueError("chunk/guard/hop size must be > 0")

    # 링버퍼(48k) 유지
    ring = np.zeros(win_samples, dtype=np.float32)
    filled = 0
    since_last_infer = 0

    # 스트림 콜백 -> 큐
    q: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        # (frames, channels)
        x = indata[:, 0].astype(np.float32).copy()
        q.put(x)

    # 상태
    STATE_IDLE = 0
    STATE_BURST = 1

    state = STATE_IDLE
    burst_until = 0.0
    ignore_until = 0.0

    # 가드 RMS는 guard_frame 단위로만 계산
    guard_buf = collections.deque()
    guard_buf_len = 0

    def push_ring(x: np.ndarray):
        nonlocal ring, filled
        n = len(x)
        if n >= win_samples:
            ring[:] = x[-win_samples:]
            filled = win_samples
            return
        ring = np.roll(ring, -n)
        ring[-n:] = x
        filled = min(win_samples, filled + n)

    def run_infer_on_current_window() -> float:
        # 48k(2초) -> 16k -> preproc -> tflite
        y48 = ring.copy()
        y16 = librosa.resample(y48, orig_sr=CAPTURE_SR, target_sr=MODEL_SR).astype(np.float32)
        x_in = preproc_v4(
            y16, mu, sdv,
            model_sr=MODEL_SR,
            clip_seconds=args.clip_seconds,
            target_rms=args.target_rms,
            n_mels=args.n_mels,
            n_fft=args.n_fft,
            hop=args.hop,
            center=args.center,
        )
        return model.predict_proba(x_in)

    print("STREAM START (Ctrl+C to stop)")
    print(f"mode: guard RMS -> burst sliding {args.burst_seconds}s (hop={args.hop_sec}s)")
    if args.udp_host:
        print(f"udp: {args.udp_host}:{args.udp_port} msg='{args.udp_message}'")

    with sd.InputStream(
        device=args.input_device,
        channels=1,
        samplerate=CAPTURE_SR,
        blocksize=chunk_samples,
        callback=callback,
    ):
        try:
            while True:
                x = q.get()
                now = time.time()

                # 링버퍼 갱신은 항상 수행
                push_ring(x)
                since_last_infer += len(x)

                # 쿨다운 중이면, 버스트 시작/감지를 하지 않음
                if now < ignore_until:
                    state = STATE_IDLE
                    guard_buf.clear()
                    guard_buf_len = 0
                    continue

                # -----------------------
                # IDLE: RMS guard only
                # -----------------------
                if state == STATE_IDLE:
                    # guard_frame_sec 만큼 모아서 RMS 계산
                    guard_buf.append(x)
                    guard_buf_len += len(x)

                    while guard_buf_len >= guard_samples:
                        # guard_samples만큼 꺼내서 rms
                        need = guard_samples
                        parts = []
                        while need > 0 and guard_buf:
                            a = guard_buf[0]
                            if len(a) <= need:
                                parts.append(a)
                                need -= len(a)
                                guard_buf.popleft()
                            else:
                                parts.append(a[:need])
                                guard_buf[0] = a[need:]
                                need = 0

                        guard_buf_len -= guard_samples
                        g = np.concatenate(parts) if parts else np.zeros(guard_samples, np.float32)
                        g_rms = rms(g)

                        if g_rms >= args.rms_trigger_th:
                            # 버스트 시작
                            state = STATE_BURST
                            burst_until = now + args.burst_seconds
                            since_last_infer = hop_samples  # 즉시 1회 추론하도록
                            guard_buf.clear()
                            guard_buf_len = 0
                            print(f"[BURST START] t={now:.3f} rms={g_rms:.6f}")
                            break

                # -----------------------
                # BURST: sliding inference
                # -----------------------
                if state == STATE_BURST:
                    if now >= burst_until:
                        state = STATE_IDLE
                        since_last_infer = 0
                        print(f"[BURST END] t={now:.3f} (timeout)")
                        continue

                    if filled < win_samples:
                        # 아직 2초 윈도우가 안 찬 경우(시작 직후)
                        continue

                    if since_last_infer >= hop_samples:
                        since_last_infer = 0

                        p = run_infer_on_current_window()
                        print(f"[INFER] t={now:.3f} p={p:.4f}")

                        if p >= args.prob_th:
                            print("[DETECT] bless you!")

                            # 액션
                            if not args.no_audio:
                                play_wav_async(args.play_wav)

                            udp.send(args.udp_message)

                            # 쿨다운
                            ignore_until = time.time() + args.cooldown_sec

                            # 버스트 종료
                            state = STATE_IDLE
                            since_last_infer = 0
                            guard_buf.clear()
                            guard_buf_len = 0

        except KeyboardInterrupt:
            print("STOP")


if __name__ == "__main__":
    main()