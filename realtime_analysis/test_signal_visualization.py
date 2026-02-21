import collections

import matplotlib.pyplot as plt
import numpy as np

import librosa
import pyaudio

from scipy.signal import butter, lfilter

# 1. 필터 및 정규화 함수 정의
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, b, a):
    return lfilter(b, a, data)

def normalize_audio(audio):
    rms = np.sqrt(np.mean(audio**2))
    if rms > 0:
        return audio / rms * 0.1
    return audio

# 2. 시스템 상수 및 설정
RATE = 16000
CHUNK = 2048 # 상시 추론 시 CPU 부하를 줄이기 위해 버퍼 크기를 키움
TOTAL_SAMPLES = int(RATE * 2.0)
LOWCUT = 500
HIGHCUT = 2000

# 3. 모델 및 스케일러 로드
# 파일명이 이전 단계에서 저장한 이름과 일치하는지 확인하십시오.
b, a = butter_bandpass(LOWCUT, HIGHCUT, RATE)

# 4. 2초 분량의 오디오 링 버퍼 초기화
audio_buffer = collections.deque(maxlen=int(TOTAL_SAMPLES))

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("상시 감지 시스템 가동 중... (Ctrl+C 종료)")

try:
    while True:
        # 오디오 데이터 읽기
        raw_data = stream.read(CHUNK, exception_on_overflow=False)
        current_samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 링 버퍼에 추가
        audio_buffer.extend(current_samples)
        
        # 버퍼가 2초치 가득 찼을 때만 추론 시작
        if len(audio_buffer) == TOTAL_SAMPLES:
            audio_data = np.array(audio_buffer)
            
            # 전처리 파이프라인: 필터링 -> 정규화
            filtered_audio = apply_filter(audio_data, b, a)
            normalized_audio = normalize_audio(filtered_audio)
            
            # MFCC 특징 추출
            mfcc = librosa.feature.mfcc(y=normalized_audio, sr=RATE, n_mfcc=20, hop_length=512)

            plt.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis') 
            plt.colorbar()
            plt.pause(0.25)
            plt.clf()
            plt.close()
            
except KeyboardInterrupt:
    print("시스템 중단")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()