# test_saving_v4.py
# 목적: 재채기 감지 시 "bless you!" 출력 + 해당 2초 wav 저장
# 핵심: log-mel center=False로 학습(v4)과 프레임 수 일치

import time
import queue
import collections
from pathlib import Path

import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf


# =========================
# 사용자 설정
# =========================
TFLITE_PATH = Path(r".\v4_model.tflite")
STATS_PATH  = Path(r".\v4_norm_stats.npz")

INPUT_DEVICE = None
CAPTURE_SR   = 48000
MODEL_SR     = 16000

CLIP_SECONDS = 2.0
PRE_SECONDS  = 0.5
FRAME_SEC    = 0.10

RMS_TRIGGER_TH = 0.008
PROB_TH        = 0.90
COOLDOWN_SEC   = 1.5

# v4 학습과 동일해야 함
N_MELS = 64
N_FFT  = 400
HOP    = 160
CENTER = False
TARGET_RMS = 0.1

SAVE_DIR = Path("./detected_wavs")
# =========================


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
        center=CENTER,   # 중요
    )
    return np.log(S + 1e-6).T.astype(np.float32)


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

    def expected_input_shape(self):
        return self.in_det.get("shape", None)

    def predict_proba(self, x: np.ndarray) -> float:
        self.interp.set_tensor(self.in_det["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out_det["index"]).reshape(-1)[0]
        return float(y)


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    if not TFLITE_PATH.exists():
        raise FileNotFoundError(TFLITE_PATH)
    if not STATS_PATH.exists():
        raise FileNotFoundError(STATS_PATH)

    mu, sdv = load_stats(STATS_PATH)
    model = LiteModel(TFLITE_PATH)
    
    frame_samples = int(CAPTURE_SR * FRAME_SEC)
    target_samples_capture = int(CAPTURE_SR * CLIP_SECONDS)

    pre_chunks = max(1, int(PRE_SECONDS / FRAME_SEC))
    pre_buffer = collections.deque(maxlen=pre_chunks)

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            pass
        q.put(indata[:, 0].astype(np.float32).copy())

    capturing = False
    captured = []
    captured_len = 0
    ignore_until = 0.0
    save_idx = 0

    print("STREAM START (Ctrl+C to stop)")

    with sd.InputStream(
        device=INPUT_DEVICE,
        channels=1,
        samplerate=CAPTURE_SR,
        blocksize=frame_samples,
        callback=callback,
    ):
        try:
            while True:
                x = q.get()
                now = time.time()

                if now < ignore_until:
                    pre_buffer.append(x)
                    continue

                if not capturing:
                    pre_buffer.append(x)
                    if rms(x) >= RMS_TRIGGER_TH:
                        capturing = True
                        captured = list(pre_buffer)
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

                    y16 = librosa.resample(y48, orig_sr=CAPTURE_SR, target_sr=MODEL_SR).astype(np.float32)

                    x_in = preproc(y16, mu, sdv)
                    p = model.predict_proba(x_in)

                    if p >= PROB_TH:
                        save_idx += 1
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        fname = SAVE_DIR / f"sneeze_{ts}_{save_idx:04d}_p{p:.3f}.wav"
                        sf.write(str(fname), y16, MODEL_SR)

                        print("bless you!")
                        ignore_until = time.time() + COOLDOWN_SEC

                    capturing = False
                    captured = []
                    captured_len = 0
                    pre_buffer.clear()

        except KeyboardInterrupt:
            print("STOP")


if __name__ == "__main__":
    main()
