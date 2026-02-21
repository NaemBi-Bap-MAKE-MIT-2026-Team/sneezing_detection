# test.py
# 목적: 임계값 넘으면 "bless you!"만 출력 (다른 출력 없음)

import time
import queue
import collections
from pathlib import Path

import numpy as np
import librosa
import sounddevice as sd

# =========================
# 사용자 설정
# =========================
TFLITE_PATH = Path(r".\v3_1_model.tflite")
STATS_PATH  = Path(r".\norm_stats.npz")

INPUT_DEVICE = 1          # 본인 장치 번호로 수정
CAPTURE_SR   = 48000      # Windows 안정용. 마이크가 16000을 안정적으로 지원하면 16000으로 바꿔도 됨
MODEL_SR     = 16000

CLIP_SECONDS = 2.0
PRE_SECONDS  = 0.5        # 트리거 전 버퍼(0으로 해도 됨)
FRAME_SEC    = 0.10       # 입력 콜백 프레임 크기(0.1초)

RMS_TRIGGER_TH = 0.008    # 이벤트 캡처 트리거(환경에 따라 조정)
PROB_TH        = 0.90     # 재채기 판정 임계값(원하면 올리기)
COOLDOWN_SEC   = 2      # 한 번 감지 후 재감지 방지

# log-mel 파라미터 (학습과 동일해야 함)
N_MELS = 64
N_FFT  = 400
HOP    = 160
TARGET_RMS = 0.1
# =========================


def rms(x: np.ndarray, eps: float = 1e-8) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + eps))


def normalize_rms(x: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
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
    )
    return np.log(S + 1e-6).T.astype(np.float32)  # (frames, mels)


def load_stats(stats_path: Path):
    st = np.load(str(stats_path), allow_pickle=True)
    mu = st["mu"].astype(np.float32)
    sdv = st["sd"].astype(np.float32)

    # (1,1,mels)->(mels,)
    if mu.ndim == 3 and mu.shape[0] == 1 and mu.shape[1] == 1:
        mu = mu.reshape(-1)
    if sdv.ndim == 3 and sdv.shape[0] == 1 and sdv.shape[1] == 1:
        sdv = sdv.reshape(-1)

    # (1,frames,mels)->(frames,mels)
    if mu.ndim == 3 and mu.shape[0] == 1:
        mu = mu[0]
    if sdv.ndim == 3 and sdv.shape[0] == 1:
        sdv = sdv[0]

    return mu, sdv


def preproc(y16_2s: np.ndarray, mu: np.ndarray, sdv: np.ndarray) -> np.ndarray:
    y = np.asarray(y16_2s, dtype=np.float32)
    target = int(MODEL_SR * CLIP_SECONDS)
    if len(y) > target:
        y = y[:target]
    elif len(y) < target:
        y = np.pad(y, (0, target - len(y)))

    y = normalize_rms(y)
    f = logmel(y)  # (frames,mels)
    fn = (f - mu) / (sdv + 1e-6)
    return fn[None, ..., None].astype(np.float32)  # (1,frames,mels,1)


# TFLite interpreter
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except Exception:
    import tensorflow as tf
    TFLiteInterpreter = tf.lite.Interpreter


class LiteModel:
    def __init__(self, path: Path):
        self.interp = TFLiteInterpreter(model_path=str(path))
        self.interp.allocate_tensors()
        self.in_det = self.interp.get_input_details()[0]
        self.out_det = self.interp.get_output_details()[0]

    def predict_proba(self, x: np.ndarray) -> float:
        self.interp.set_tensor(self.in_det["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out_det["index"]).reshape(-1)[0]
        return float(y)


def main():
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
        x = indata[:, 0].astype(np.float32).copy()
        q.put(x)

    capturing = False
    captured = []
    captured_len = 0
    ignore_until = 0.0

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

                # 쿨다운 중에는 버퍼만 업데이트
                if now < ignore_until:
                    pre_buffer.append(x)
                    continue

                x_rms = rms(x)

                if not capturing:
                    pre_buffer.append(x)
                    if x_rms >= RMS_TRIGGER_TH:
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

                    # 정확히 2초
                    if len(y48) > target_samples_capture:
                        y48 = y48[:target_samples_capture]
                    elif len(y48) < target_samples_capture:
                        y48 = np.pad(y48, (0, target_samples_capture - len(y48)))

                    # 48k -> 16k
                    y16 = librosa.resample(y48, orig_sr=CAPTURE_SR, target_sr=MODEL_SR).astype(np.float32)

                    x_in = preproc(y16, mu, sdv)
                    p = model.predict_proba(x_in)

                    if p >= PROB_TH:
                        print("bless you!")
                        ignore_until = time.time() + COOLDOWN_SEC
                        time.sleep(COOLDOWN_SEC)

                    capturing = False
                    captured = []
                    captured_len = 0
                    pre_buffer.clear()

        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
