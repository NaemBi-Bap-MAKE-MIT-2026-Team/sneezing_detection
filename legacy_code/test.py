import pyaudio
import numpy as np
import tensorflow as tf
import joblib
import librosa
import collections
import time

# 1. 시스템 설정 상수
RATE = 16000
CHUNK = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16
THRESHOLD = 0.05       # 환경에 따라 조정 (RMS 기준)
PRE_SECONDS = 0.5      # 트리거 발생 전 보관할 시간
TOTAL_SECONDS = 2.0    # 모델 입력 규격
COOLDOWN_SECONDS = 1.5 # 감지 후 대기 시간

# 2. 유틸리티 함수 정의
def normalize_audio(audio):
    # 입력 신호의 에너지를 학습 데이터 수준(RMS 0.1)으로 강제 조정
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        return audio / rms * 0.1
    return audio

# 3. 모델 및 스케일러 로드
# 파일 경로가 정확한지 확인하십시오.
model = tf.keras.models.load_model('sneeze_detection_model.keras')
scaler = joblib.load('sneeze_scaler.pkl')

# 4. 오디오 버퍼 및 스트림 초기화
pre_buffer_chunks = int(RATE / CHUNK * PRE_SECONDS)
pre_buffer = collections.deque(maxlen=pre_buffer_chunks)

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("시스템 가동: 링 버퍼 기반 실시간 감시 중...")

try:
    while True:
        # 데이터 읽기 및 정규화 (-1.0 ~ 1.0)
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        current_samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 에너지 계산: $RMS = \sqrt{\frac{1}{n}\sum_{i=1}^{n}x_i^2}$
        rms = np.sqrt(np.mean(current_samples**2))
        
        if rms > THRESHOLD:
            print(f"트리거 발생 (RMS: {rms:.4f}) - 분석 시작")
            
            # 링 버퍼에서 이전 데이터 확보
            captured_audio = list(pre_buffer)
            captured_audio.append(current_samples)
            
            # 남은 시간만큼 추가 녹음 수행
            remaining_time = TOTAL_SECONDS - (len(captured_audio) * CHUNK / RATE)
            remaining_chunks = int(remaining_time * RATE / CHUNK)
            
            for _ in range(remaining_chunks):
                post_data = stream.read(CHUNK)
                post_samples = np.frombuffer(post_data, dtype=np.int16).astype(np.float32) / 32768.0
                captured_audio.append(post_samples)
            
            # 데이터 통합 및 전처리 파이프라인
            full_audio = np.concatenate(captured_audio)
            
            # 1단계: 샘플 개수 강제 고정 (int 형변환 필수)
            target_size = int(RATE * TOTAL_SECONDS)
            full_audio = librosa.util.fix_length(full_audio, size=target_size)
            
            # 2단계: 에너지 레벨 정규화
            full_audio = normalize_audio(full_audio)
            
            # 3단계: MFCC 특징 추출
            mfcc = librosa.feature.mfcc(y=full_audio, sr=RATE, n_mfcc=20, hop_length=512)
            
            # 4단계: 스케일링 및 모델 입력 형태 변환
            feat_scaled = scaler.transform(mfcc.reshape(1, -1))
            final_input = feat_scaled.reshape(1, 20, 63, 1)
            
            # 5단계: 추론 수행
            prob = model.predict(final_input, verbose=0)[0][0]
            
            # 결과 판단: $P(y=1|x) > \tau$
            if prob > 0.85:
                print(f"결과: 재채기 감지 (확률: {prob:.4f})")
                print("ACTION: Bless you!")
                time.sleep(COOLDOWN_SECONDS)
                pre_buffer.clear()
            else:
                print(f"결과: 일반 소음 (확률: {prob:.4f})")
            
            print("감시 모드 복귀")
        else:
            # 트리거 미발생 시 최신 0.5초 유지
            pre_buffer.append(current_samples)

except KeyboardInterrupt:
    print("사용자에 의해 중단됨")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()