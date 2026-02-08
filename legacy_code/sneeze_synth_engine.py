import os
import librosa
import numpy as np
import soundfile as sf
from glob import glob
import random
from tqdm import tqdm
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

class SneezeIntegratedEngine:
    def __init__(self, sr=16000, duration=2.0):
        self.sr = sr
        self.duration = duration
        self.num_samples = int(sr * duration)
        self.target_rms = 0.1

    def load_audio_standard(self, path):
        # 16kHz 리샘플링 및 모노 로드 (scipy 리샘플러 사용)
        y, _ = librosa.load(path, sr=self.sr, mono=True, res_type='scipy')
        
        # 길이 정규화 (2초)
        if len(y) < self.num_samples:
            pad = self.num_samples - len(y)
            y = np.pad(y, (pad//2, pad - pad//2), 'constant')
        else:
            y = y[:self.num_samples]
            
        # RMS 진폭 정규화 (0.1)
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            y = y * (self.target_rms / rms)
        return y

    def step1_prepare_background_pool(self, bg_dirs, output_dir):
        print("\n# 1단계: 배경 및 채터링 슬라이싱 시작")
        os.makedirs(output_dir, exist_ok=True)
        hop = int(self.sr * 0.25)
        count = 0
        
        for d in bg_dirs:
            files = glob(os.path.join(d, "*.wav"))
            for f in tqdm(files, desc=f"{os.path.basename(d)} 처리 중"):
                try:
                    y, _ = librosa.load(f, sr=self.sr, mono=True)
                    for start in range(0, len(y) - self.num_samples, hop):
                        chunk = y[start:start + self.num_samples]
                        # 각 조각 정규화
                        rms = np.sqrt(np.mean(chunk**2))
                        if rms > 1e-4:
                            chunk = chunk * (self.target_rms / rms)
                            save_name = f"bg_{count}.wav"
                            sf.write(os.path.join(output_dir, save_name), chunk, self.sr)
                            count += 1
                except Exception as e:
                    print(f"파일 로드 에러 {f}: {e}")
        print(f"결과: {count}개의 배경 조각이 풀에 생성되었습니다.")
        return count

    def step2_prepare_esc50_sources(self, esc_root, output_dir):
        print("\n# 2단계: ESC-50 재채기 및 하드 네거티브 추출")
        csv_path = os.path.join(esc_root, "meta/esc50.csv")
        audio_dir = os.path.join(esc_root, "audio")
        df = pd.read_csv(csv_path)
        
        # 선별된 하드 네거티브 카테고리
        hn_cats = ['clapping', 'door_knock', 'mouse_click', 'glass_breaking', 'coughing', 'laughing']
        
        s_out = os.path.join(output_dir, "sneeze_esc50")
        hn_out = os.path.join(output_dir, "hard_neg_esc50")
        os.makedirs(s_out, exist_ok=True)
        os.makedirs(hn_out, exist_ok=True)

        for _, row in tqdm(df.iterrows(), total=len(df), desc="ESC-50 필터링"):
            cat = row['category']
            if cat == 'sneezing' or cat in hn_cats:
                y = self.load_audio_standard(os.path.join(audio_dir, row['filename']))
                target_path = s_out if cat == 'sneezing' else hn_out
                sf.write(os.path.join(target_path, row['filename']), y, self.sr)

    def step3_final_synthesis(self, sneeze_dirs, bg_pool_dir, hn_pool_dir, final_dir, num_aug=10, neg_ratio=1.5):
        print("\n# 3단계: 최종 데이터셋 합성 및 증강")
        pos_out = os.path.join(final_dir, "class_1_sneeze")
        neg_out = os.path.join(final_dir, "class_0_noise")
        os.makedirs(pos_out, exist_ok=True)
        os.makedirs(neg_out, exist_ok=True)

        # 소스 파일 리스트
        s_files = []
        for d in sneeze_dirs:
            s_files.extend(glob(os.path.join(d, "*.wav")))
        bg_files = glob(os.path.join(bg_pool_dir, "*.wav"))
        hn_files = glob(os.path.join(hn_pool_dir, "*.wav"))

        # Positive 생성 (재채기 + 배경)
        p_count = 0
        for idx, s_path in enumerate(tqdm(s_files, desc="Positive 합성")):
            target = self.load_audio_standard(s_path)
            for i in range(num_aug):
                bg = self.load_audio_standard(random.choice(bg_files))
                snr = random.choice([5, 10, 15, 20])
                alpha = 1.0 / (10**(snr/20))
                # 시간축 롤링 적용
                mixed = np.roll(target, random.randint(0, self.num_samples)) + alpha * bg
                # Peak 정규화로 클리핑 방지
                mixed /= (np.max(np.abs(mixed)) + 1e-7)
                sf.write(os.path.join(pos_out, f"pos_{idx}_{i}.wav"), mixed, self.sr)
                p_count += 1

        # Negative 생성 (하드 네거티브 60%, 소프트 네거티브 40%)
        n_target = int(p_count * neg_ratio)
        for i in tqdm(range(n_target), desc="Negative 합성"):
            if random.random() < 0.4:
                # 소프트 네거티브: 배경 풀에서 무작위 추출
                res = self.load_audio_standard(random.choice(bg_files))
            else:
                # 하드 네거티브: ESC-50 충격음 + 배경 합성
                hn = self.load_audio_standard(random.choice(hn_files))
                bg = self.load_audio_standard(random.choice(bg_files))
                snr = random.choice([3, 7, 12])
                alpha = 1.0 / (10**(snr/20))
                mixed = hn + alpha * bg
                mixed /= (np.max(np.abs(mixed)) + 1e-7)
                res = mixed
            sf.write(os.path.join(neg_out, f"neg_{i}.wav"), res, self.sr)
        
        print(f"\n# 최종 보고")
        print(f"Positive 샘플: {p_count}개")
        print(f"Negative 샘플: {n_target}개")
        print(f"데이터 비율: 1 : {neg_ratio}")

if __name__ == "__main__":
    # 인스턴스 생성
    engine = SneezeIntegratedEngine()
    
    # 실행 경로 설정
    BG_SOURCES = ["raw_data/background", "raw_data/chattering"]
    ESC50_ROOT = "raw_data/esc-50"
    SNEEZE_RAW = "raw_data/sneeze"
    
    PROCESSED_BG = "processed/bg_pool"
    PROCESSED_ESC = "processed"
    FINAL_DATASET = "final_dataset"

    # 파이프라인 가동
    engine.step1_prepare_background_pool(BG_SOURCES, PROCESSED_BG)
    engine.step2_prepare_esc50_sources(ESC50_ROOT, PROCESSED_ESC)
    engine.step3_final_synthesis(
        sneeze_dirs=[SNEEZE_RAW, os.path.join(PROCESSED_ESC, "sneeze_esc50")],
        bg_pool_dir=PROCESSED_BG,
        hn_pool_dir=os.path.join(PROCESSED_ESC, "hard_neg_esc50"),
        final_dir=FINAL_DATASET,
        num_aug=10,
        neg_ratio=1.5
    )