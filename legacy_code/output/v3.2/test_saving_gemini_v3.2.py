import time
import queue
from pathlib import Path
import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf

# =========================
# 사용자 설정 및 경로
# =========================
TFLITE_PATH = Path("./v3_2_model.tflite")
STATS_PATH  = Path("./norm_stats_v3.2.npz")
SAVE_DIR    = Path("./detected_wavs")

# 노트북의 기본 마이크를 사용하려면 None 또는 장치 인덱스 입력
# sd.query_devices()로 확인 가능합니다.
INPUT_DEVICE = None 
MODEL_SR     = 16000

CLIP_SECONDS = 2.0   
HOP_SECONDS  = 0.5    

RMS_TRIGGER_TH = 0.008
PROB_TH        = 0.90
COOLDOWN_SEC   = 2.0

N_MELS = 64
N_FFT  = 400
HOP    = 160
TARGET_RMS = 0.1

# =========================
# 오디오 전처리 및 모델 클래스
# =========================
def rms(x):
    return float(np.sqrt(np.mean(x * x) + 1e-8))

def normalize_rms(x, target=TARGET_RMS):
    r = rms(x)
    if r > 1e-6:
        x = x * (target / (r + 1e-8))
    return np.clip(x, -1.0, 1.0)

def logmel(y):
    # Mel Spectrogram 계산: 주파수 성분을 멜 스케일로 변환
    # $S = |FFT(y)|^2$ 과정을 거쳐 멜 필터뱅크 적용
    S = librosa.feature.melspectrogram(
        y=y, sr=MODEL_SR, n_fft=N_FFT,
        hop_length=HOP, n_mels=N_MELS, power=2.0
    )
    return np.log(S + 1e-6).T

def preproc(y, mu, sdv):
    y = normalize_rms(y)
    f = logmel(y)
    f = (f - mu) / (sdv + 1e-6)
    
    # [None, ..., None] 대신 명확하게 차원을 지정합니다.
    # 결과 형태: (1, 시간_프레임_수, 멜_개수, 1) -> 4차원
    return f[np.newaxis, :, :, np.newaxis].astype(np.float32)

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

class LiteModel:
    def __init__(self, path):
        if not path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {path}")
        self.interp = Interpreter(model_path=str(path))
        self.interp.allocate_tensors()
        self.i = self.interp.get_input_details()[0]["index"]
        self.o = self.interp.get_output_details()[0]["index"]

    def predict(self, x):
        self.interp.set_tensor(self.i, x)
        self.interp.invoke()
        return float(self.interp.get_tensor(self.o)[0][0])

# =========================
# 메인 루프
# =========================
def main():
    model = LiteModel(TFLITE_PATH)
    SAVE_DIR.mkdir(exist_ok=True)
    print("--- Sneeze Detector Ready ---")
    
    if not STATS_PATH.exists():
        print(f"에러: {STATS_PATH} 파일이 없습니다.")
        return

    st = np.load(str(STATS_PATH), allow_pickle=True)
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

    ring_buffer = np.zeros(int(MODEL_SR * CLIP_SECONDS))
    hop_samples = int(MODEL_SR * HOP_SECONDS)
    new_samples_count = 0

    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        q.put(indata[:, 0].copy())

    with sd.InputStream(device=INPUT_DEVICE, channels=1, samplerate=MODEL_SR, callback=callback):
        try:
            while True:
                chunk = q.get()
                ring_buffer = np.roll(ring_buffer, -len(chunk))
                ring_buffer[-len(chunk):] = chunk
                new_samples_count += len(chunk)

                if new_samples_count >= hop_samples:
                    new_samples_count = 0
                    
                    recent_audio = ring_buffer[-hop_samples:]
                    current_rms = rms(recent_audio)
                    
                    # 모니터링용 출력 (필요 시 주석 해제)
                    # print(f"RMS: {current_rms:.4f}", end='\r')

                    if current_rms >= RMS_TRIGGER_TH:
                        x_in = preproc(ring_buffer, mu, sdv)
                        prob = model.predict(x_in)

                        if prob >= PROB_TH:
                            print("\n" + "="*30)
                            print(f"!!! BLESS YOU !!! (Prob: {prob:.3f})")
                            print("="*30 + "\n")
                            
                            timestamp = time.strftime("%Y%m%d-%H%M%S")
                            fname = SAVE_DIR / f"sneeze_{timestamp}.wav"
                            sf.write(str(fname), ring_buffer, MODEL_SR)
                            
                            time.sleep(COOLDOWN_SEC)
                            print("Ready again...")
                            # 쿨다운 동안 쌓인 큐 비우기
                            while not q.empty(): q.get()

        except KeyboardInterrupt:
            print("\nDetector Stopped.")

if __name__ == "__main__":
    main()