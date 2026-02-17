import argparse
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

# -------------------------
# TFLite Interpreter 로드
# -------------------------
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

# -------------------------
# 전처리 (v3.x와 동일 전제)
# -------------------------
MODEL_SR = 16000
CLIP_SECONDS = 2.0
CLIP_SAMPLES = int(MODEL_SR * CLIP_SECONDS)

N_MELS = 64
N_FFT = 400
HOP = 160
TARGET_RMS = 0.1

def rms(x: np.ndarray) -> float:
    x = np.asarray(x, np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-8))

def normalize_rms(x: np.ndarray, target: float = TARGET_RMS) -> np.ndarray:
    x = np.asarray(x, np.float32)
    r = rms(x)
    if r > 1e-6:
        x = x * (target / (r + 1e-8))
    return np.clip(x, -1.0, 1.0).astype(np.float32)

def logmel(y: np.ndarray) -> np.ndarray:
    # center 옵션은 학습 파이프라인과 다를 수 있습니다.
    # 아래는 가장 흔한 설정(center=True). mismatch는 frames 강제 보정으로 해결합니다.
    S = librosa.feature.melspectrogram(
        y=y,
        sr=MODEL_SR,
        n_fft=N_FFT,
        hop_length=HOP,
        n_mels=N_MELS,
        power=2.0,
        center=True,
    )
    return np.log(S + 1e-6).T.astype(np.float32)  # (frames, 64)

def load_stats(stats_path: Path):
    st = np.load(str(stats_path), allow_pickle=True)
    mu = st["mu"].astype(np.float32)
    sdv = st["sd"].astype(np.float32)

    # (1,1,64) -> (64,)
    if mu.ndim == 3 and mu.shape[0] == 1 and mu.shape[1] == 1:
        mu = mu.reshape(-1)
    if sdv.ndim == 3 and sdv.shape[0] == 1 and sdv.shape[1] == 1:
        sdv = sdv.reshape(-1)

    # (1,frames,64) -> (frames,64)
    if mu.ndim == 3 and mu.shape[0] == 1:
        mu = mu[0]
    if sdv.ndim == 3 and sdv.shape[0] == 1:
        sdv = sdv[0]

    return mu, sdv

def fix_2s(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, np.float32)
    if len(y) >= CLIP_SAMPLES:
        return y[:CLIP_SAMPLES]
    return np.pad(y, (0, CLIP_SAMPLES - len(y))).astype(np.float32)

def preproc_2s(y_2s_16k: np.ndarray, mu: np.ndarray, sdv: np.ndarray, expected_frames: int) -> np.ndarray:
    y = normalize_rms(fix_2s(y_2s_16k))
    f = logmel(y)  # (frames, 64)

    # frames를 모델 입력에 강제로 맞춤
    cur = f.shape[0]
    if cur > expected_frames:
        f = f[:expected_frames, :]
    elif cur < expected_frames:
        f = np.pad(f, ((0, expected_frames - cur), (0, 0)), mode="constant")

    # mu/sdv가 (64,)이면 자동 브로드캐스트로 (frames,64)에 적용됨
    f = (f - mu) / (sdv + 1e-6)

    x = f[np.newaxis, :, :, np.newaxis].astype(np.float32)  # (1,frames,64,1)
    return x

class LiteModel:
    def __init__(self, tflite_path: Path):
        self.interp = Interpreter(model_path=str(tflite_path))
        self.interp.allocate_tensors()
        self.in_detail = self.interp.get_input_details()[0]
        self.out_detail = self.interp.get_output_details()[0]
        self.i = self.in_detail["index"]
        self.o = self.out_detail["index"]
        self.in_shape = tuple(self.in_detail["shape"])  # (1, frames, 64, 1)

    def predict_proba(self, x: np.ndarray) -> float:
        self.interp.set_tensor(self.i, x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.o).reshape(-1)[0]
        return float(y)

def sliding_windows(y: np.ndarray, hop_seconds: float):
    hop = int(MODEL_SR * hop_seconds)
    if hop <= 0:
        hop = int(MODEL_SR * 0.5)

    if len(y) <= CLIP_SAMPLES:
        yield 0.0, fix_2s(y)
        return

    max_start = len(y) - CLIP_SAMPLES
    s = 0
    while s <= max_start:
        seg = y[s:s+CLIP_SAMPLES]
        yield s / MODEL_SR, seg
        s += hop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", type=str, default="./v3_2_model.tflite")
    ap.add_argument("--stats", type=str, default="./norm_stats_v3.2.npz")
    ap.add_argument("--dir", type=str, default=".")
    ap.add_argument("--files", type=str, default="sneeze.wav,test1.wav,test2.wav,test3.wav,test4.wav")
    ap.add_argument("--thr", type=float, default=0.90)
    ap.add_argument("--hop", type=float, default=0.5)
    ap.add_argument("--save_hits", action="store_true")
    ap.add_argument("--save_dir", type=str, default="./offline_hits")
    args = ap.parse_args()

    tflite_path = Path(args.tflite)
    stats_path = Path(args.stats)
    base_dir = Path(args.dir)
    save_dir = Path(args.save_dir)

    if not tflite_path.exists():
        raise FileNotFoundError(tflite_path)
    if not stats_path.exists():
        raise FileNotFoundError(stats_path)

    mu, sdv = load_stats(stats_path)
    model = LiteModel(tflite_path)

    expected_frames = int(model.in_shape[1])
    if model.in_shape[2] != 64 or model.in_shape[3] != 1:
        print("경고: 모델 입력의 mel/채널 차원이 예상과 다릅니다:", model.in_shape)

    files = [f.strip() for f in args.files.split(",") if f.strip()]
    paths = [base_dir / f for f in files]

    print("tflite:", tflite_path)
    print("stats :", stats_path)
    print("model input shape:", model.in_shape)
    print("threshold:", args.thr, "hop_seconds:", args.hop)
    print("-" * 60)

    if args.save_hits:
        save_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        if not p.exists():
            print("MISSING:", p)
            continue

        y, _ = librosa.load(str(p), sr=MODEL_SR, mono=True)
        y = y.astype(np.float32)

        best_p = -1.0
        best_t = 0.0
        best_seg = None

        for t0, seg in sliding_windows(y, hop_seconds=args.hop):
            x_in = preproc_2s(seg, mu, sdv, expected_frames)
            prob = model.predict_proba(x_in)
            if prob > best_p:
                best_p = prob
                best_t = t0
                best_seg = seg

        verdict = "SNEEZE" if best_p >= args.thr else "NEG"
        print(f"{p.name:15s}  max_p={best_p:.4f}  at={best_t:.2f}s  -> {verdict}")

        if args.save_hits and best_p >= args.thr and best_seg is not None:
            out_wav = save_dir / f"{p.stem}_hit_p{best_p:.3f}_t{best_t:.2f}.wav"
            sf.write(str(out_wav), best_seg, MODEL_SR)

if __name__ == "__main__":
    main()
