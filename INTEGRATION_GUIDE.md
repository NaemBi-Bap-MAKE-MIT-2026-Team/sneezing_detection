# BlessYouFlow를 main.py에 통합하기

> **초안**: main.py와 BlessYouFlow 파이프라인 연동 가이드

---

## 📋 통합 준비 체크리스트

### 1. 환경 변수 설정 (필수)

```bash
# Gemini API 키 (필수)
export GEMINI_API_KEY="your_gemini_api_key"

# ElevenLabs API 키 (현재 미완성 - 나중에 추가)
export ELEVENLABS_API_KEY="your_elevenlabs_api_key"
```

### 2. 의존성 설치

```bash
# realtime_detection 의존성
cd realtime_detection
pip install -r requirements.txt

# BlessYouFlow 추가 의존성
pip install google-genai elevenlabs requests
```

### 3. 파일 확인

```
src/
├── connection/
│   ├── bless_you_flow.py          ✅ Ready
│   ├── gps/gps.py                 ✅ Ready
│   ├── weather/weather.py         ✅ Ready
│   ├── gemini/gemini_comment.py  ✅ Ready (개선됨)
│   └── elven_labs/tts_player.py   ⚠️  미완성
└── output_feature/sounds/
    └── bless_you.wav              ⚠️ 필요 (준비 필요)
```

---

## 🔌 통합 패턴 (3가지)

### 패턴 A: 비동기 실행 (권장 - 간단함)

**장점**: 재채기 감지 지연 없음, 구현 간단  
**단점**: 여러 재채기 시 음성 겹칠 수 있음

```python
# realtime_detection/main.py

from pathlib import Path
from src.connection.bless_you_flow import BlessYouFlow

class RealtimeSneezeDetector:
    def __init__(self, ...):
        # ... 기존 초기화 코드
        
        # BlessYouFlow 초기화
        try:
            self.bless_you_flow = BlessYouFlow(
                bless_wav_path=Path(__file__).resolve().parent.parent / "src" / "output_feature" / "sounds" / "bless_you.wav",
                language="en",  # 또는 "ko"
                enable_context=True,  # GPS/날씨 활성화
            )
            print("✓ BlessYouFlow 모듈 로드됨")
        except (ImportError, ValueError) as e:
            print(f"⚠ BlessYouFlow 로드 실패: {e}")
            self.bless_you_flow = None

    def on_sneeze_detected(self):
        """재채기 감지 콜백"""
        # 기존 OutputHandler 호출
        self.output_handler.handle_detection(...)
        
        # BlessYouFlow 백그라운드 실행
        if self.bless_you_flow:
            thread = self.bless_you_flow.run_async()
            # 스레드는 백그라운드에서 자동으로 실행됨
```

**호출 위치**:
```python
# main.py의 주 루프
is_sneeze, probability = self.model_inference.predict(mfcc_features)

if is_sneeze:
    self.output_handler.handle_detection(...)
    self.on_sneeze_detected()  # ← 추가
```

---

### 패턴 B: 동기 실행 (블로킹)

**장점**: 순차 처리, 구현 명확  
**단점**: ~15초 블로킹 (감지 지연)

```python
if self.bless_you_flow:
    self.bless_you_flow.run()  # 완료까지 대기 (블로킹)
```

---

### 패턴 C: 큐 기반 (최고 - 복잡)

**장점**: 감지 지연 없음 + 순차 처리 (음성 겹침 없음)  
**단점**: 코드 복잡도 증가

```python
import queue
import threading

class RealtimeSneezeDetector:
    def __init__(self, ...):
        # ... 기존 초기화
        
        # BlessYouFlow 초기화
        self.bless_you_flow = BlessYouFlow(...)
        
        # 응답 큐 및 워커 스레드
        self.response_queue = queue.Queue(maxsize=10)
        self.response_worker = threading.Thread(
            target=self._response_worker,
            daemon=True
        )
        self.response_worker.start()

    def _response_worker(self):
        """백그라운드: 큐에서 요청을 꺼내 순차 처리"""
        while True:
            try:
                # 1분 타임아웃 (CPU 낭비 방지)
                _ = self.response_queue.get(timeout=60)
                if self.bless_you_flow:
                    self.bless_you_flow.run()  # 순차 실행
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[응답 워커] 오류: {e}")

    def on_sneeze_detected(self):
        """재채기 감지 콜백"""
        try:
            # 큐에 넣기 (비블로킹)
            self.response_queue.put(None, block=False)
        except queue.Full:
            print("⚠ 응답 큐가 가득 찼습니다 (빨리 처리 중)")
```

---

## ✅ 최소 통합 코드 (패턴 A: 권장)

### 1단계: 임포트 추가

```python
# realtime_detection/main.py 상단

from pathlib import Path

try:
    from src.connection.bless_you_flow import BlessYouFlow
    _BLESS_YOU_AVAILABLE = True
except ImportError:
    _BLESS_YOU_AVAILABLE = False
    print("⚠ BlessYouFlow 모듈 없음 — 재채기 감지만 작동")
```

### 2단계: __init__ 수정

```python
class RealtimeSneezeDetector:
    def __init__(self, model_path=None, threshold=None, verbose=None):
        # ... 기존 모듈 초기화 ...
        
        # BlessYouFlow 초기화 (최상단)
        self.bless_you_flow = None
        if _BLESS_YOU_AVAILABLE:
            try:
                wav_path = Path(__file__).resolve().parent.parent / "src" / "output_feature" / "sounds" / "bless_you.wav"
                self.bless_you_flow = BlessYouFlow(
                    bless_wav_path=wav_path,
                    language="en",
                    enable_context=True
                )
                print("✓ BlessYouFlow 로드됨")
            except Exception as e:
                print(f"⚠ BlessYouFlow 실패: {e}")
```

### 3단계: 감지 루프 수정

```python
# 메인 루프 내에서
is_sneeze, probability = self.model_inference.predict(mfcc_features)

# Handle output
self.output_handler.handle_detection(is_sneeze, probability, audio_chunk, SAMPLE_RATE)

# BlessYouFlow 호출 (새로 추가)
if is_sneeze and self.bless_you_flow:
    self.bless_you_flow.run_async()  # 백그라운드 실행
```

---

## 🧪 테스트 방법

### 1. 단위 테스트 (모듈별)

```bash
cd src/connection

# GPS 테스트
python -m gps.gps
# 출력: [GPSLocator] ✓ ...

# 날씨 테스트
python -m weather.weather
# 출력: [WeatherFetcher] ✓ ...

# Gemini 테스트
python -m gemini.gemini_comment en 5
# 출력: 5개 메시지

# TTS 테스트 (ElevenLabs API 키 필요)
python -m elven_labs.tts_player "Test message"
```

### 2. 통합 테스트

```bash
# BlessYouFlow 단독 실행
cd src/connection
python bless_you_flow.py

# 예상 동작
# [1] WAV 재생 (aplay -q bless_you.wav)
# [2] GPS 조회 (IP 기반)
# [3] 날씨 조회 (Open-Meteo)
# [4] Gemini 멘트 생성
# [5] ElevenLabs TTS 재생
```

### 3. main.py 통합 테스트

```bash
cd realtime_detection

# 환경 변수 설정
export GEMINI_API_KEY="your_key"
export ELEVENLABS_API_KEY="your_key"

# 실행
python main.py --verbose

# 재채기 감지 후:
# [BlessYouFlow] Stage 1: WAV 재생 중...
# [GPSLocator] ✓ Seoul, South Korea
# [WeatherFetcher] ✓ 25°C, 60%, Good AQI
# [GeminiComment] ✓ 5개 메시지 생성 완료
# [ElevenLabsTTS] ✓ 재생 완료
```

---

## 🐛 트러블슈팅

### 문제 1: "requests 패키지 없음"
```
[GPSLocator] ⚠ requests 패키지 없음
```
**해결**:
```bash
pip install requests
```

---

### 문제 2: "Gemini API 키 없음"
```
ValueError: Gemini API 키가 없습니다
```
**해결**:
```bash
export GEMINI_API_KEY="sk-..."
```

---

### 문제 3: "google-genai 패키지 없음"
```
ImportError: google-genai 패키지가 설치되지 않았습니다
```
**해결**:
```bash
pip install google-genai
```

---

### 문제 4: "멘트를 생성할 수 없음"
```
[GeminiComment] ❌ 배치 생성 오류: ...
[BlessYouFlow] ⚠ 멘트 없음 — TTS 건너뜀
```
**원인**: Gemini API 속도, 네트워크 지연, 할당량 초과  
**해결**: Timeout 증가 또는 재시도 로직 추가

---

### 문제 5: "MP3 플레이어 없음"
```
[ElevenLabsTTS] ⚠ MP3 플레이어를 찾지 못했습니다
```
**해결** (RPi):
```bash
sudo apt install mpg123
```

**해결** (Mac):
```bash
brew install mpg123
```

**해결** (Linux):
```bash
sudo apt install mpg123
```

---

## 🔧 커튼마이제이션

### 언어 변경 (한국어)

```python
self.bless_you_flow = BlessYouFlow(
    bless_wav_path=wav_path,
    language="ko",  # ← 한국어로 변경
    enable_context=True
)
```

### 음성 변경

```python
from src.connection.elven_labs.tts_player import ElevenLabsTTSPlayer

flow = BlessYouFlow(...)
# 내부 TTS 음성 변경은 현재 지원 안 함
# (향후 패라미터 추가 필요)
```

### 메시지 캐시 크기 변경

```python
self.bless_you_flow = BlessYouFlow(
    ...,
    num_messages=50,  # 기본값: 30
)
```

### GPS/날씨 비활성화 (빠른 응답)

```python
self.bless_you_flow = BlessYouFlow(
    ...,
    enable_context=False,  # GPS/날씨 스킵
)
# → 기본 프롬프트만 사용 (빠름)
```

---

## 📊 성능 목표

| 항목 | 타겟 | 현재 |
|---|---|---|
| **WAV 재생** | 1-2초 | ✅ 1-2초 |
| **GPS 조회** | < 5초 | ✅ < 2초 (IP 기반) |
| **날씨 API** | < 5초 | ✅ < 1초 (Open-Meteo) |
| **Gemini 생성** | < 10초 | ⏳ 3-8초 (배치) |
| **TTS 재생** | < 5초 | ⏳ 2-5초 (네트워크) |
| **전체 파이프라인** | < 20초 | ⏳ ~12초 (병렬) |

> 병렬 실행으로 총 소요 시간은 max(WAV:2s, GPS:2s) + Gemini:5s + TTS:3s ≈ 10초

---

## ✨ 다음 단계

1. ✅ **검토 완료**: 파이프라인 구조 정상
2. ✅ **코드 개선**: Gemini 파싱 + 컨텍스트 필터링
3. ⏳ **통합**: main.py와 연동 (패턴 A 권장)
4. ⏳ **테스트**: 단위 테스트 → 통합 테스트
5. ⏳ **배포**: RPi에 배포 및 최종 테스트

---

**상태**: Ready for Integration  
**마지막 업데이트**: 2026-02-21
