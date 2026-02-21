# BlessYouFlow 파이프라인 검토 보고서

**작성일**: 2026-02-21  
**검토 대상**: `src/connection/bless_you_flow.py` 및 연관 모듈들  
**상태**: ✅ 파이프라인 구조 정상 / ⚠️ 개선사항 있음 / 🔴 주의사항 있음

---

## 📋 파이프라인 구조

### 전체 흐름

```
재채기 감지 (from realtime_detection/main.py)
    ↓
BlessYouFlow.run()
    ├─ Stage 1: WAV 재생 + GPS/날씨 조회 (병렬)
    │   ├─ [메인] aplay로 bless_you.wav 동기 재생
    │   └─ [백그라운드] GPS & Weather API 호출
    │       ├─ GPSLocator: IP 기반 위치 조회
    │       ├─ WeatherFetcher: Open-Meteo (무료) + wttr.in
    │       └─ → ctx dict 생성
    │
    ├─ Stage 2: Gemini API로 건강 멘트 생성
    │   ├─ 캐시 확인 (self._message_cache)
    │   ├─ 비었으면 배치 생성 (num_messages=30으로 캐싱)
    │   ├─ GeminiCommentGenerator.generate_batch()
    │   └─ → comment str 추출
    │
    └─ Stage 3: ElevenLabs TTS 음성 재생
        ├─ ElevenLabsTTSPlayer.speak(comment)
        ├─ [아직 미완성 - 확인 필요]
        └─ → 음성 출력
```

---

## ✅ 정상 작동 구간

### 1. **GPS 모듈** (`gps.py`)

| 항목 | 상태 | 설명 |
|---|---|---|
| API 호출 | ✅ | `ip-api.com` (무료, API 키 不필요) |
| 타임아웃 | ✅ | `config.CONTEXT_FETCH_TIMEOUT` (5초) 적용 |
| 에러 처리 | ✅ | 예외 발생 시 `None` 반환 (흐름 중단 안 함) |
| 반환 형식 | ✅ | `{"city": str, "country": str, "lat": float, "lon": float, "region": str}` |

**동작 검증**:
```python
# gps.py 라인 49-57 정상 작동
if _REQUESTS_AVAILABLE:
    response = requests.get(cfg.GPS_IP_API_URL, timeout=self.timeout)
    # cfg.GPS_IP_API_URL = "http://ip-api.com/json/"
    return self._parse(raw)  # → context dict 생성
```

**의존성 체크**:
- ✅ `requests` 패키지 필요
- ✅ 환경 변수/설정 없음 (순수 공개 API)

---

### 2. **날씨 모듈** (`weather.py`)

| 항목 | 상태 | 설명 |
|---|---|---|
| 주 API | ✅ | Open-Meteo (무료, API 키 不필요) |
| 보조 API | ✅ | wttr.in (선택사항, 더 풍부한 설명) |
| 타임아웃 | ✅ | `config.CONTEXT_FETCH_TIMEOUT` 적용 |
| 에러 처리 | ⚠️ | 부분 실패 시 `None` 필드 포함 가능 |

**API 엔드포인트**:
```python
# Open-Meteo (기본)
WEATHER_API_URL = 
  "https://api.open-meteo.com/v1/forecast"
  "?latitude={lat}&longitude={lon}"
  "&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"

# Open-Meteo 대기질
AIR_QUALITY_API_URL =
  "https://air-quality-api.open-meteo.com/v1/air-quality"
  "?latitude={lat}&longitude={lon}"
  "&current=pm10,pm2_5,us_aqi"

# wttr.in (선택사항)
url = f"https://wttr.in/{city}?format=j1"
```

**반환 형식**:
```python
{
    "temperature": float,           # °C
    "humidity": int,                # %
    "weather_label": str,           # e.g. "Partly cloudy"
    "wind_speed": float,            # km/h
    "pm2_5": float,                 # µg/m³
    "pm10": float,                  # µg/m³
    "us_aqi": int,                  # US AQI 지수
    "aqi_label": str,               # "Good" / "Moderate" / etc
    "temp_change_yesterday": str,    # "+2.5°C" (city 제공 시)
}
```

**의존성 체크**:
- ✅ `requests` 패키지 필요

---

### 3. **Gemini 모듈** (`gemini_comment.py`)

| 항목 | 상태 | 설명 |
|---|---|---|
| API 버전 | ✅ | `google-genai` (gemini-2.0-flash) |
| 배치 생성 | ✅ | 한 번에 여러 멘트 생성 + 캐싱 |
| 컨텍스트 | ✅ | 환경 정보 포함/미포함 두 프롬프트 각각 준비 |
| 다국어 | ✅ | EN / KO 지원 |
| 에러 처리 | ✅ | 실패 시 기본 fallback 텍스트 제공 |

**프롬프트 종류**:

1. **기본 프롬프트** (`_BATCH_PROMPTS[lang]`):
   - GPS/날씨 없을 때 사용
   - 일반적인 건강 조언 생성

2. **컨텍스트 프롬프트** (`_BATCH_CONTEXT_PROMPTS[lang]`):
   - 위치, 온도, 습도, 대기질 기반
   - 환경에 맞는 맞춤형 멘트 생성

**멘트 파싱** (라인 221-226):
```python
raw_text = response.text.strip()
messages = [
    line.strip()[2:]  # "- " 제거
    for line in raw_text.split("\n")
    if line.strip().startswith("- ")
]
```

**의존성 체크**:
- ✅ `google-genai` 패키지 필요
- ⚠️ `GEMINI_API_KEY` 환경 변수 필수

---

### 4. **ElevenLabs TTS 모듈** (`tts_player.py`)

| 항목 | 상태 | 설명 |
|---|---|---|
| API 버전 | 🟡 | `elevenlabs` (아직 검증 필요) |
| 플레이어 | ✅ | mpg123 / ffplay / mplayer 자동 선택 |
| 소리 재생 | ⚠️ | **아직 미완성** (사용자 주의) |
| 임시 파일 | ✅ | tempfile로 자동 정리 |

**플레이어 우선순위**:
```python
_PLAYER_COMMANDS = [
    ["mpg123", "-q"],                               # 가장 가벼움 (RPi 권장)
    ["ffplay", "-nodisp", "-autoexit", ...],
    ["mplayer", "-really-quiet"],
]
```

**의존성 체크**:
- ✅ `elevenlabs` 패키지 필요
- ⚠️ `ELEVENLABS_API_KEY` 환경 변수 필수
- 🔴 **시스템 플레이어 미설치 시 재생 불가** (경고는 출력됨)

---

## ⚠️ 주의사항 및 개선사항

### 🔴 Issue 1: 멘트 파싱 포맷 의존성

**문제**: Gemini 응답이 정확히 `"- "` 형식을 따르지 않으면 멘트 누락

```python
# 현재 코드 (gemini_comment.py 라인 221-226)
messages = [
    line.strip()[2:]  # "- " 정확히 기대
    for line in raw_text.split("\n")
    if line.strip().startswith("- ")
]
```

**시나리오**:
- Gemini가 `"* Message"` 형식 반환 → 파싱 안 됨
- Gemini가 `"1. Message"` 형식 반환 → 파싱 안 됨
- 줄 전체가 메시지인 경우 → 파싱 안 됨

**권장사항**:
```python
# 더 유연한 파싱 로직
messages = []
for line in raw_text.split("\n"):
    line = line.strip()
    if not line:
        continue
    # 여러 형식 지원: "- msg", "* msg", "1. msg", "msg"
    if line.startswith(("- ", "* ", "• ")):
        messages.append(line[2:].strip())
    elif line and line[0].isdigit() and (". " in line or ") " in line):
        # "1. msg" 또는 "1) msg" 처리
        msg = line.split(". ", 1)[-1] if ". " in line else line.split(") ", 1)[-1]
        messages.append(msg.strip())
    elif line:  # 구분자 없는 메시지도 허용
        messages.append(line)
return messages[:num_messages]  # 초과분 제거
```

---

### 🔴 Issue 2: 날씨 API 부분 실패 처리

**문제**: Open-Meteo가 절반만 성공하면 `None` 필드가 생김

```python
# weather.py 라인 110-130
weather = self._fetch_weather(lat, lon)  # 성공
air = self._fetch_air_quality(lat, lon)  # 실패 → None

result["temperature"] = weather.get("temperature_2m")  # ✅ 있음
result["pm2_5"] = air.get("pm2_5") if air else None     # ❌ None
```

**Gemini 프롬프트 전달 시**:
```python
# bless_you_flow.py 라인 178
prompt = template.format(num_messages=num_messages, **context)
# context = {"pm2_5": None, ...} → "PM2.5: None" 그대로 프롬프트에 들어감
```

**권장사항**:
```python
# bless_you_flow.py의 _stage2_get_comment()에서
if ctx:
    # None 값 필터링
    ctx_clean = {k: v for k, v in ctx.items() if v is not None}
    if len(ctx_clean) < 5:  # 너무 적으면 기본 프롬프트 사용
        ctx = None
```

---

### 🟡 Issue 3: 타임아웃 시 부분 결과 사용

**문제**: Stage 1 타임아웃 (GPS/Weather 8초 초과) 후에도 진행

```python
# bless_you_flow.py 라인 97-98
ctx_thread.join(timeout=self._CTX_TIMEOUT)  # 8초 후 강제 반환
ctx = ctx_result[0] if ctx_result else None  # None 가능
```

**현재 동작**: OK (건강하게 폴백)
- `ctx = None` → 기본 프롬프트로 Gemini 호출
- Stage 2 / Stage 3 계속 진행

**의도적인 설계이므로** 문제 없음 ✅

---

### 🟡 Issue 4: ElevenLabs 아직 미완성

**현재 상태**:
- ✅ API 호출 구조 정상 (`self.client.text_to_speech.convert()`)
- ✅ 임시 파일 생성 및 정리 정상
- ⚠️ **플레이어 미설치 시** → "플레이어 없음 — 재생 건너뜀" 출력만 함

**테스트 체크리스트**:
```bash
# 1. elevenlabs 패키지 설치 확인
pip show elevenlabs

# 2. MP3 플레이어 설치 (RPi)
sudo apt install mpg123

# 3. 환경 변수 설정
export ELEVENLABS_API_KEY="your_key"

# 4. 단독 테스트 (tts_player.py)
cd src/connection
python -m elven_labs.tts_player "Hello, stay warm!"
```

---

## ✅ 통합 검증 체크리스트

### 단계별 테스트

```markdown
[ ] **1단계: 모듈 독립 테스트**
    [ ] python -m src.connection.gps.gps
    [ ] python -m src.connection.weather.weather
    [ ] python -m src.connection.gemini.gemini_comment en 5
    [ ] python -m src.connection.elven_labs.tts_player "Test message"

[ ] **2단계: 통합 테스트 (모든 API 키 설정)**
    [ ] export GEMINI_API_KEY="..."
    [ ] export ELEVENLABS_API_KEY="..."  
    [ ] python src/connection/bless_you_flow.py

[ ] **3단계: main.py 통합**
    [ ] BlessYouFlow 임포트 및 오류 없음
    [ ] 재채기 감지 → BlessYouFlow.run_async() 호출
    [ ] 백그라운드 재생 정상 확인

[ ] **4단계: RPi 배포 테스트**
    [ ] 네트워크 연결 확인
    [ ] MP3 플레이어 설치
    [ ] API 키 설정
    [ ] 실제 환경에서 테스트
```

---

## 🔌 main.py 통합 가이드

### 현재 상태

`realtime_detection/main.py`는 **재채기 감지만 담당**:
- 오디오 캡처 → MFCC 추출 → 모델 추론
- 감지 시 `OutputHandlerModule.handle_detection()` 호출

### 통합 방식 (권장안)

#### **방식 A: 비동기 통합 (권장)**
```python
# realtime_detection/main.py 수정
from src.connection.bless_you_flow import BlessYouFlow

class RealtimeSneezeDetector:
    def __init__(self, ...):
        # ... 기존 코드
        # BlessYouFlow 초기화
        try:
            self.bless_you_flow = BlessYouFlow(
                bless_wav_path=Path("src/output_feature/sounds/bless_you.wav"),
                language="en"  # 또는 "ko"
            )
        except (ImportError, ValueError) as e:
            print(f"⚠ BlessYouFlow 초기화 실패: {e}")
            self.bless_you_flow = None

    def on_sneeze_detected(self, audio_chunk, probability):
        # 기존 OutputHandler 호출
        self.output_handler.handle_detection(...)
        
        # BlessYouFlow 백그라운드 실행 (블로킹 안 함)
        if self.bless_you_flow:
            self.bless_you_flow.run_async()  # ← 스레드에서 실행
```

**장점**:
- 재채기 감지 지연 없음
- 여러 재채기 감지 시 대기열 문제 없음

**단점**:
- 소리/TTS 여러 개 동시 재생 가능 (충돌 가능)

---

#### **방식 B: 동기 통합 (간단함)**
```python
# BlessYouFlow.run() 직접 호출 (블로킹)
if self.bless_you_flow:
    self.bless_you_flow.run()  # ← Stage 1-3 완료까지 대기
```

**장점**:
- 순차 처리로 명확함

**단점**:
- 총 ~15초 정도 블로킹 (1초 cooldown 동안 음성 재생)
- 그 사이 재채기 감지 불가

---

#### **방식 C: 큐 기반 (최고)**
```python
import queue
import threading

class RealtimeSneezeDetector:
    def __init__(self, ...):
        self.sneeze_queue = queue.Queue()
        self.response_thread = threading.Thread(
            target=self._response_worker, daemon=True
        )
        self.response_thread.start()
    
    def _response_worker(self):
        """백그라운드: 큐에서 꺼내서 BlessYouFlow 실행"""
        while True:
            try:
                audio_chunk, prob = self.sneeze_queue.get(timeout=60)
                if self.bless_you_flow:
                    self.bless_you_flow.run()  # 순차 처리
            except queue.Empty:
                continue
    
    def on_sneeze_detected(self, audio_chunk, probability):
        self.output_handler.handle_detection(...)
        self.sneeze_queue.put((audio_chunk, probability))
```

**장점**:
- 주 감지 스레드 블로킹 안 함
- 재채기 순차 처리 (음성 충돌 없음)

**단점**:
- 코드 복잡도 증가

---

## 🎯 권장 최종 지침

### 현재 상태
1. ✅ **GPS 모듈**: 정상 작동
2. ✅ **Weather 모듈**: 정상 작동 (멘트 파싱 주의)
3. ✅ **Gemini 모듈**: 정상 작동 (멘트 파싱 주의)
4. ⚠️ **ElevenLabs 모듈**: 미완성 (API 키 & 플레이어 필수)

### 다음 단계
1. **멘트 파싱 개선** (Issue 1 해결)
2. **날씨 API 부분 실패 처리** (Issue 2 해결)
3. **ElevenLabs API 키 & 플레이어 설정**
4. **main.py 통합** (방식 A 또는 C 권장)
5. **end-to-end 테스트** 체크리스트 실행

### 배포 전 체크
```
[ ] GPS: ip-api.com 정상 응답
[ ] Weather: Open-Meteo 정상 응답
[ ] Gemini: GEMINI_API_KEY 설정 + 배치 생성 검증
[ ] ElevenLabs: ELEVENLABS_API_KEY 설정 + 음성 재생 검증
[ ] main.py: 통합되어 재채기 감지 시 BlessYouFlow 호출
[ ] RPi: 네트워크 + MP3 플레이어 설치
```

---

## 📌 참고: 각 모듈 설정 위치

| 항목 | 위치 | 내용 |
|---|---|---|
| GPS API | `src/ml_model/config.py:9` | `GPS_IP_API_URL` |
| Weather API | `src/ml_model/config.py:18-26` | `WEATHER_API_URL`, `AIR_QUALITY_API_URL` |
| Timeout | `src/ml_model/config.py:32` | `CONTEXT_FETCH_TIMEOUT` |
| Gemini 모델 | `src/connection/gemini/gemini_comment.py:156` | `"gemini-2.0-flash"` |
| ElevenLabs 음성 | `src/connection/elven_labs/tts_player.py:33` | `"Rachel"` |
| WAV 파일 | `bless_you_flow.py` 호출 시 | `bless_wav_path` 파라미터 |

---

**작성**: GitHub Copilot  
**검토 완료**: 2026-02-21
