# event_capture_tflite_filtered.py
# 목적: "대화 오탐"을 줄이기 위해 2초 캡처 후 모델 추론 전에 간단한 물리 필터(대화-유사 제거)를 적용
# 구조: RMS 트리거 -> (0.5s 프리버퍼 + 2.0s 캡처) -> [필터] -> (log-mel + norm_stats) -> TFLite 추론 -> bless you -> 쿨다운

import time
import queue
import collections
from pathlib import Path

import numpy as np
import librosa
import sounddevice as sd

# -----------------------------
# 효민님이 여기만 수정
# -----------------------------
TFLITE_PATH = Path(r".\sneeze_ds_cnn_dynamic.tflite")
STATS_PATH  = Path(r".\norm_stats.npz")

# probe에서 OK 뜬 장치 사용 권장: 0,1,4,5,9
INPUT_DEVICE = 1

# 1차 RMS 트리거(환경에 따라 조정)
RMS_TRIGGER_TH = 0.008

# 프리버퍼(트리거 직전 보관)
PRE_SECONDS = 0.5

# 모델 입력 규격(고정)
TOTAL_SECONDS = 2.0

# 감지 후 대기(중복 감지 방지)
COOLDOWN_SECONDS = 1.5

# 최종 판정 threshold
PROB_TH = 0.60

# Windows 안정성: 48k로 캡처 후 16k로 리샘플
CAPTURE_SR = 48000
MODEL_SR = 16000

# 한 프레임 길이
FRAME_SEC = 0.10

# 로그 빈도
PRINT_EVERY = 1

# -----------------------------
# 대화 오탐 줄이기용 "필터" 파라미터
# -----------------------------
# 재채기: 피크가 날카롭고(crest 큼), 고주파 비율이 높고(highband 큼), 2초 내내 지속되지 않음(active 낮음)
# 대화: crest 낮고, highband 낮고, active 높음

CREST_TH = 6.0          # 피크/평균 비율 최소
HIGHBAND_TH = 0.35      # 2kHz 이상 에너지 비율 최소
ACTIVE_MAX = 0.60       # "소리 있는 프레임" 비율 최대
ACTIVE_FRAME_MS = 20.0  # active 계산 프레임 크기
ACTIVE_RMS_TH = 0.02    # active 프레임을 '소리 있음'으로 볼 RMS 임계(환경에 따라 조정)

# 필터를 끄고 싶으면 False
USE_FILTER = True
# -----------------------------


# -----------------------------
# 학습과 동일해야 하는 특징 파라미터
# -----------------------------
N_MELS = 64
N_FFT = 400
HOP = 160
TARGET_RMS = 0.1
# -----------------------------


def list_input_devices():
    devs = sd.query_devices()
    out = []
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0:
            out.append((i, d["name"], d["max_input_channels"]))
    return out


def rms(x: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.sqrt(np.mean(x * x) + eps))


def normalize_rms(y: np.ndarray, target_rms: float = TARGET_RMS) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    r = rms(y)
    if r > 1e-8:
        y = y * (target_rms / (r + 1e-8))
    return np.clip(y, -1.0, 1.0).astype(np.float32)


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


# -----------------------------
# 대화-유사 제거용 필터 함수들
# -----------------------------
def crest_factor(x: np.ndarray, eps: float = 1e-8) -> float:
    # 피크/평균 비율. 재채기는 보통 큼.
    x = np.asarray(x, dtype=np.float32)
    return float(np.max(np.abs(x)) / (rms(x) + eps))


def highband_ratio(x: np.ndarray, sr: int, fmin: float = 2000.0) -> float:
    # 2kHz 이상 에너지 비율. 재채기는 보통 큼.
    X = librosa.stft(x, n_fft=512, hop_length=256, win_length=512)
    mag2 = (np.abs(X) ** 2)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    hi = mag2[freqs >= fmin].sum()
    tot = mag2.sum() + 1e-8
    return float(hi / tot)


def active_ratio(x: np.ndarray, sr: int, frame_ms: float, thr: float) -> float:
    # 2초 중 '소리 있는' 프레임 비율. 대화는 보통 큼.
    frame_len = int(sr * frame_ms / 1000.0)
    frame_len = max(frame_len, 160)  # 안전
    n = len(x) // frame_len
    if n <= 0:
        return 0.0
    xs = x[:n * frame_len].reshape(n, frame_len)
    fr_rms = np.sqrt(np.mean(xs * xs, axis=1) + 1e-8)
    return float(np.mean(fr_rms > thr))


def passes_voice_filter(y16: np.ndarray) -> tuple[bool, dict]:
    # True면 "재채기 후보"로 통과, False면 "대화/지속 소리"로 보고 스킵
    c = crest_factor(y16)
    h = highband_ratio(y16, sr=MODEL_SR)
    a = active_ratio(y16, sr=MODEL_SR, frame_ms=ACTIVE_FRAME_MS, thr=ACTIVE_RMS_TH)

    ok = (c >= CREST_TH) and (h >= HIGHBAND_TH) and (a <= ACTIVE_MAX)
    info = {"crest": c, "high": h, "active": a}
    return ok, info


class Preproc:
    def __init__(self, stats_path: Path):
        st = np.load(str(stats_path), allow_pickle=True)
        mu = st["mu"].astype(np.float32)
        sdv = st["sd"].astype(np.float32)

        # (1,1,mels) -> (mels,)
        if mu.ndim == 3 and mu.shape[0] == 1 and mu.shape[1] == 1:
            mu = mu.reshape(-1)
        if sdv.ndim == 3 and sdv.shape[0] == 1 and sdv.shape[1] == 1:
            sdv = sdv.reshape(-1)

        # (1,frames,mels) -> (frames,mels)
        if mu.ndim == 3 and mu.shape[0] == 1:
            mu = mu[0]
        if sdv.ndim == 3 and sdv.shape[0] == 1:
            sdv = sdv[0]

        self.mu = mu
        self.sdv = sdv

    def __call__(self, y_16k_2s: np.ndarray) -> np.ndarray:
        y = np.nan_to_num(y_16k_2s.astype(np.float32))
        y = normalize_rms(y)  # 학습과 정합
        feat = logmel(y)      # (frames, mels)
        feat = (feat - self.mu) / (self.sdv + 1e-6)
        return feat[None, ..., None].astype(np.float32)  # (1,frames,mels,1)


# TFLite Interpreter (Windows: tensorflow fallback)
Interpreter = None
try:
    from tflite_runtime.interpreter import Interpreter as _I
    Interpreter = _I
except Exception:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter


class LiteModel:
    def __init__(self, tflite_path: Path):
        self.interp = Interpreter(model_path=str(tflite_path))
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
        raise FileNotFoundError(f"TFLITE not found: {TFLITE_PATH}")
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"STATS not found: {STATS_PATH}")

    # 장치 목록 출력(원하면 주석)
    devs = list_input_devices()
    print("INPUT DEVICES:")
    for i, name, ch in devs:
        print(f" {i}: {name} (max_in_ch={ch})")
    print("")

    model = LiteModel(TFLITE_PATH)
    pre = Preproc(STATS_PATH)

    frame_samples = int(CAPTURE_SR * FRAME_SEC)
    target_samples_capture = int(CAPTURE_SR * TOTAL_SECONDS)

    pre_chunks = int(PRE_SECONDS / FRAME_SEC)
    pre_buffer = collections.deque(maxlen=max(1, pre_chunks))

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            pass
        x = indata[:, 0].astype(np.float32).copy()
        q.put(x)

    print("SYSTEM START")
    print(f"device={INPUT_DEVICE} capture_sr={CAPTURE_SR} frame_sec={FRAME_SEC}")
    print(f"RMS_TRIGGER_TH={RMS_TRIGGER_TH} PRE_SECONDS={PRE_SECONDS} TOTAL_SECONDS={TOTAL_SECONDS}")
    print(f"PROB_TH={PROB_TH} COOLDOWN_SECONDS={COOLDOWN_SECONDS}")
    print(f"FILTER use={USE_FILTER} crest>={CREST_TH} high>={HIGHBAND_TH} active<={ACTIVE_MAX} (active_thr={ACTIVE_RMS_TH})")
    print("")

    capturing = False
    captured = []
    captured_len = 0
    ignore_until = 0.0

    step = 0

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
                step += 1

                # 쿨다운: 감지 직후 일정 시간은 트리거 금지(프리버퍼만 유지)
                if now < ignore_until:
                    pre_buffer.append(x)
                    if step % max(1, PRINT_EVERY) == 0:
                        print("cooldown...", "left:", round(ignore_until - now, 2), "s", "rms:", round(rms(x), 6))
                    continue

                x_rms = rms(x)

                if not capturing:
                    pre_buffer.append(x)

                    if x_rms >= RMS_TRIGGER_TH:
                        capturing = True
                        captured = list(pre_buffer)
                        captured.append(x)
                        captured_len = sum(len(c) for c in captured)
                        print("TRIGGER", "rms:", round(x_rms, 6), "-> capture start")
                    else:
                        if step % max(1, PRINT_EVERY) == 0:
                            print("idle", "rms:", round(x_rms, 6))
                        continue
                else:
                    captured.append(x)
                    captured_len += len(x)

                # 2초 캡처 완료 시 처리
                if capturing and captured_len >= target_samples_capture:
                    y48 = np.concatenate(captured, axis=0)

                    # 정확히 2초로 고정
                    if len(y48) > target_samples_capture:
                        y48 = y48[:target_samples_capture]
                    elif len(y48) < target_samples_capture:
                        y48 = np.pad(y48, (0, target_samples_capture - len(y48)))

                    # 48k -> 16k 리샘플
                    y16 = librosa.resample(y48, orig_sr=CAPTURE_SR, target_sr=MODEL_SR).astype(np.float32)

                    # 16k 기준 정확히 2초로 고정(안전)
                    target_16 = int(MODEL_SR * TOTAL_SECONDS)
                    if len(y16) > target_16:
                        y16 = y16[:target_16]
                    elif len(y16) < target_16:
                        y16 = np.pad(y16, (0, target_16 - len(y16)))

                    # -----------------------------
                    # (중요) 대화-유사 제거 필터
                    # -----------------------------
                    if USE_FILTER:
                        ok, info = passes_voice_filter(y16)
                        if not ok:
                            print("SKIP (voice-like)",
                                  "crest:", round(info["crest"], 2),
                                  "high:", round(info["high"], 3),
                                  "active:", round(info["active"], 3))
                            # 상태 초기화 후 감시 복귀
                            capturing = False
                            captured = []
                            captured_len = 0
                            pre_buffer.clear()
                            print("BACK TO WATCHING\n")
                            continue
                        else:
                            print("PASS FILTER",
                                  "crest:", round(info["crest"], 2),
                                  "high:", round(info["high"], 3),
                                  "active:", round(info["active"], 3))

                    # 전처리 + 추론
                    x_in = pre(y16)
                    prob = model.predict_proba(x_in)

                    # 결과
                    if prob >= PROB_TH:
                        print("RESULT: SNEEZE", "prob:", round(prob, 4), "ACTION: Bless you!")
                        ignore_until = time.time() + COOLDOWN_SECONDS
                        time.sleep(COOLDOWN_SECONDS)
                    else:
                        print("RESULT: NEG", "prob:", round(prob, 4))

                    # 상태 초기화
                    capturing = False
                    captured = []
                    captured_len = 0
                    pre_buffer.clear()

                    print("BACK TO WATCHING\n")

        except KeyboardInterrupt:
            print("STOP")


if __name__ == "__main__":
    main()
