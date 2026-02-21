# connection

재채기 감지 후 GPS/날씨 정보를 수집하고, Gemini로 건강 멘트를 생성하여
ElevenLabs TTS로 재생하는 파이프라인 모듈입니다.

---

## 디렉토리 구조

```text
connection/
├── __init__.py                        # BlessYouFlow export
├── README.md                          ← 이 파일
├── bless_you_flow.py                  # 전체 파이프라인 오케스트레이터
├── gemini/
│   ├── __init__.py
│   └── gemini_comment.py              # Gemini API → 건강 멘트 배치 생성
├── elven_labs/
│   ├── __init__.py
│   └── tts_player.py                  # ElevenLabs API → TTS 재생
├── gps/
│   ├── __init__.py
│   └── gps.py                         # IP 기반 위치 조회 (ip-api.com, 무료)
└── weather/
    ├── __init__.py
    └── weather.py                     # 날씨 + 대기질 조회 (Open-Meteo, 무료)
```

---

## 파이프라인 흐름

```text
재채기 감지 (on_detect)
        │
        ├────────────────────── 병렬 실행 ──────────────────────┐
        │                                                        │
  [Stage 1a]                                              [Stage 1b]
bless_you.wav 재생                         GPS 위치 조회 (ip-api.com)
   (aplay)                                         +
        │                               날씨/대기질 조회 (Open-Meteo)
        │                                                        │
        └───────────────── 두 작업 완료 후 ─────────────────────┘
                                   │
                             [Stage 2]
                    Gemini API → 건강 멘트 생성
                    (위치 + 날씨 컨텍스트 포함)
                    (배치 30개 생성 후 캐시 활용)
                                   │
                             [Stage 3]
                    ElevenLabs TTS → 멘트 음성 재생
```

> **GPS / 날씨 API 키 불필요** — ip-api.com 과 Open-Meteo 는 완전 무료입니다.

---

## 빠른 시작

### 1. 패키지 설치

```bash
cd src
pip install -r requirements_rpi4.txt
```

Raspberry Pi에서 MP3 재생을 위한 시스템 패키지:

```bash
sudo apt update && sudo apt install mpg123
```

### 2. API 키 설정

`src/` 디렉토리에서 실행합니다:

```bash
cp .env.example .env
```

`.env` 파일을 열어 실제 키를 입력합니다:

```dotenv
GEMINI_API_KEY=AIza...여기에_실제_키_입력
ELEVENLABS_API_KEY=sk_...여기에_실제_키_입력
```

> `.env` 파일은 `.gitignore` 에 등록되어 있으므로 Git에 커밋되지 않습니다.

### 3. 실행

```bash
cd src

# 기본 실행 (영어 멘트)
python main.py

# 한국어 멘트
python main.py --lang ko

# LLM/TTS 비활성화 (WAV 재생만)
python main.py --no-llm
```

---

## API 키 발급

### Gemini API 키

1. [Google AI Studio](https://aistudio.google.com/app/apikey) 접속
2. **Create API key** 클릭
3. 생성된 키를 `.env` 의 `GEMINI_API_KEY=` 뒤에 입력

### ElevenLabs API 키

1. [ElevenLabs](https://elevenlabs.io) 회원가입
2. 우측 상단 프로필 → **My Account** → **API Keys**
3. **Create API Key** 클릭
4. 생성된 키를 `.env` 의 `ELEVENLABS_API_KEY=` 뒤에 입력

---

## API 키 설정 방법 비교

| 방법 | 적합한 환경 | 설명 |
| --- | --- | --- |
| `.env` 파일 | 개발 환경 (Mac/Linux PC) | `src/.env` 에 키 작성, 자동 로드 |
| 환경 변수 | Raspberry Pi 상시 실행 | `~/.bashrc` 에 export 추가 |
| systemd | RPi 서비스 자동 시작 | `Environment=` 지시어 사용 |

### 환경 변수로 설정 (Raspberry Pi)

```bash
# ~/.bashrc 또는 ~/.zshrc 에 추가
echo 'export GEMINI_API_KEY="AIza..."' >> ~/.bashrc
echo 'export ELEVENLABS_API_KEY="sk_..."' >> ~/.bashrc
source ~/.bashrc
```

### systemd 서비스로 설정 (RPi 자동 실행)

`/etc/systemd/system/sneezing.service`:

```ini
[Unit]
Description=Sneezing Detection Service
After=network.target

[Service]
User=pi
WorkingDirectory=/home/pi/sneezing_detection/src
Environment="GEMINI_API_KEY=AIza..."
Environment="ELEVENLABS_API_KEY=sk_..."
ExecStart=/usr/bin/python3 main.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

---

## 각 모듈 단독 테스트

```bash
cd src

# Gemini 멘트 생성 테스트 (영어 5개)
python -m connection.gemini.gemini_comment en 5

# Gemini 멘트 생성 테스트 (한국어 5개)
python -m connection.gemini.gemini_comment ko 5

# ElevenLabs TTS 재생 테스트
python -m connection.elven_labs.tts_player "Stay warm and healthy!"

# 전체 파이프라인 테스트 (WAV → GPS/날씨 → Gemini → ElevenLabs)
python -m connection.bless_you_flow
python -m connection.bless_you_flow ko

# GPS 위치 조회 테스트
python -m connection.gps.gps

# 날씨/대기질 조회 테스트 (서울 좌표)
python -m connection.weather.weather
```

### 자동화 테스트

```bash
cd src
python tests/test_connection.py              # API 키 있으면 실제 호출
python tests/test_connection.py --no-audio   # TTS 재생 없이 텍스트 생성만
```

---

## ElevenLabs 음성 변경

기본 음성은 `Rachel` (영어 여성)입니다.

### 사용 가능한 음성 목록 확인

```python
from elevenlabs.client import ElevenLabs
import os

client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
for voice in client.voices.get_all().voices:
    print(voice.voice_id, voice.name)
```

### 음성 변경 방법

`main.py` 의 `BlessYouFlow(...)` 호출에 `elevenlabs_voice_id` 파라미터를 추가합니다:

```python
bless_flow = BlessYouFlow(
    bless_wav_path=BLESS_WAV,
    language=args.lang,
    elevenlabs_voice_id="Bella",   # 원하는 음성 이름 또는 ID
)
```

---

## Gemini 배치 메시지 캐시

API 호출 횟수를 최소화하기 위해 배치 생성 + 캐시 방식을 사용합니다.

- **첫 번째 재채기** 감지 시: Gemini API 1회 호출 → **30개** 멘트를 한 번에 생성
- **이후 재채기** 감지 시: 캐시에서 하나씩 꺼내 사용 (API 호출 없음)
- **캐시가 비면**: 다시 30개 배치 생성

캐시 크기는 `BlessYouFlow(num_messages=30)` 으로 조정할 수 있습니다.

---

## 폴백 동작

| 상황 | 동작 |
| --- | --- |
| GPS 조회 실패 | 컨텍스트 없이 Gemini 기본 프롬프트 사용 |
| 날씨 조회 실패 | 컨텍스트 없이 Gemini 기본 프롬프트 사용 |
| Gemini API 실패 | 내장 기본 문구 반환 후 ElevenLabs 재생 |
| ElevenLabs API 실패 | 오류 출력 후 무음 처리 |
| mpg123 없음 | 오류 출력 후 무음 처리 |
| API 키 없음 | `ValueError` → `main.py` 가 WAV 재생 모드로 자동 폴백 |

---

## 언어 설정

| `--lang` | 프롬프트 언어 | 출력 예시 |
| --- | --- | --- |
| `en` (기본) | 영어 | "The PM2.5 levels are high today — try wearing a mask!" |
| `ko` | 한국어 | "오늘 미세먼지가 많으니 마스크를 꼭 착용하세요!" |
