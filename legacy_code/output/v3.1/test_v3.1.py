import time
import queue
from pathlib import Path

import numpy as np
import librosa
import sounddevice as sd

TFLITE_PATH = Path(r".\v3_1_model.tflite")
STATS_PATH  = Path(r".\norm_stats.npz")
INPUT_DEVICE = 1

CAPTURE_SR = 48000
HOP_SECONDS = 0.10

# 트리거 후 슬립
SLEEP_AFTER_TRIGGER = 1.0

# 게이트/이벤트 홀드 (감지 우선으로 완화)
SILENCE_RMS_TH = 0.006
EVENT_RMS_TH   = 0.008
EVENT_HOLD_SECONDS = 2.5

# 트리거 기준 (감지 우선으로 완화)
ABS_TH = 0.60

# spike는 일단 옵션으로만 출력 (나중에 오탐 줄일 때 사용)
USE_SPIKE = False
DELTA_TH = 0.10
BASELINE_SECONDS = 5.0

PRINT_EVERY = 1

MODEL_SR = 16000
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(MODEL_SR * CLIP_SECONDS)

N_MELS = 64
N_FFT = 400
HOP = 160
TARGET_RMS = 0.1

THRESHOLD = float(open("v3_1_threshold.txt").read())

Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter as _I
    Interpreter = _I
except Exception:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


def rms(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(np.mean(x * x) + eps))


def normalize_rms(y: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    r = rms(y)
    if r > 1e-6:
        y = y * (target_rms / (r + 1e-8))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def logmel(y_2s_16k: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y_2s_16k, sr=MODEL_SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    return np.log(S + 1e-6).T.astype(np.float32)


class Preproc:
    def __init__(self, stats_path: Path):
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

        self.mu = mu
        self.sdv = sdv

    def __call__(self, y_2s_16k: np.ndarray) -> np.ndarray:
        y = np.nan_to_num(y_2s_16k.astype(np.float32))
        y = normalize_rms(y)
        feat = logmel(y)
        feat = (feat - self.mu) / (self.sdv + 1e-6)
        return feat[None, ..., None].astype(np.float32)


class LiteModel:
    def __init__(self, tflite_path: Path):
        self.interp = Interpreter(model_path=str(tflite_path))
        self.interp.allocate_tensors()
        self.in_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]

    def predict(self, x: np.ndarray) -> float:
        self.interp.set_tensor(self.in_det["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out_det["index"]).reshape(-1)[0]
        return float(y)


def main():
    model = LiteModel(TFLITE_PATH)
    pre = Preproc(STATS_PATH)

    hop_samples = int(CAPTURE_SR * HOP_SECONDS)
    q = queue.Queue()

    ring_16k = np.zeros(CLIP_SAMPLES, dtype=np.float32)

    baseline_hist = []
    baseline_maxlen = max(1, int(BASELINE_SECONDS / HOP_SECONDS))

    in_event = False
    event_until = 0.0
    p_peak = 0.0
    p_last = 0.0
    ignore_until = 0.0

    def callback(indata, frames, time_info, status):
        if status:
            pass
        q.put(indata[:, 0].astype(np.float32).copy())

    print("STREAM START")
    print("ABS_TH:", ABS_TH, "USE_SPIKE:", USE_SPIKE, "DELTA_TH:", DELTA_TH)
    print("EVENT_RMS_TH:", EVENT_RMS_TH, "EVENT_HOLD_SECONDS:", EVENT_HOLD_SECONDS, "SILENCE_RMS_TH:", SILENCE_RMS_TH)

    with sd.InputStream(
        device=INPUT_DEVICE,
        channels=1,
        samplerate=CAPTURE_SR,
        blocksize=hop_samples,
        callback=callback
    ):
        step = 0
        try:
            while True:
                x48 = q.get()
                now = time.time()

                if now < ignore_until:
                    step += 1
                    if step % PRINT_EVERY == 0:
                        print("ignore...", "left:", round(ignore_until - now, 2), "s")
                    continue

                x16 = librosa.resample(x48, orig_sr=CAPTURE_SR, target_sr=MODEL_SR).astype(np.float32)

                n = len(x16)
                if n >= CLIP_SAMPLES:
                    ring_16k[:] = x16[-CLIP_SAMPLES:]
                else:
                    ring_16k = np.roll(ring_16k, -n)
                    ring_16k[-n:] = x16

                ring_rms = rms(ring_16k)

                if ring_rms < SILENCE_RMS_TH:
                    in_event = False
                    p_peak = 0.0
                    step += 1
                    if step % PRINT_EVERY == 0:
                        print("silent", "rms:", round(ring_rms, 6))
                    continue

                x_in = pre(ring_16k)
                p = model.predict(x_in)
                p_last = p

                baseline_hist.append(p)
                if len(baseline_hist) > baseline_maxlen:
                    baseline_hist.pop(0)
                baseline = float(np.median(baseline_hist)) if baseline_hist else p

                if (not in_event) and (ring_rms >= EVENT_RMS_TH):
                    in_event = True
                    event_until = now + EVENT_HOLD_SECONDS
                    p_peak = p

                if in_event:
                    if p > p_peak:
                        p_peak = p

                    if now >= event_until:
                        spike = p_peak - baseline

                        if USE_SPIKE:
                            fired = (p_peak >= ABS_TH) and (spike >= DELTA_TH)
                        else:
                            fired = (p_peak >= ABS_TH)

                        if fired:
                            print("EVENT END",
                                  "p_peak:", round(p_peak, 3),
                                  "base:", round(baseline, 3),
                                  "spike:", round(spike, 3),
                                  "rms:", round(ring_rms, 6),
                                  "-> bless you!")
                            ignore_until = time.time() + SLEEP_AFTER_TRIGGER
                            in_event = False
                            p_peak = 0.0
                            baseline_hist.clear()
                            time.sleep(SLEEP_AFTER_TRIGGER)
                            continue
                        else:
                            print("EVENT END",
                                  "p_peak:", round(p_peak, 3),
                                  "base:", round(baseline, 3),
                                  "spike:", round(spike, 3),
                                  "rms:", round(ring_rms, 6),
                                  "-> no trigger")
                            in_event = False
                            p_peak = 0.0

                step += 1
                if step % PRINT_EVERY == 0:
                    if in_event:
                        print("event",
                              "p:", round(p_last, 3),
                              "p_peak:", round(p_peak, 3),
                              "base:", round(baseline, 3),
                              "rms:", round(ring_rms, 6))
                    else:
                        print("idle",
                              "p:", round(p_last, 3),
                              "base:", round(baseline, 3),
                              "rms:", round(ring_rms, 6))

        except KeyboardInterrupt:
            print("STOP")


if __name__ == "__main__":
    main()
