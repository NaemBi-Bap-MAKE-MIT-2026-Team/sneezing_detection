# connection

재채기 감지 후 외부 API(Gemini, ElevenLabs)와 연동하여
건강 멘트를 생성하고 TTS로 재생하는 모듈입니다.

---

## 디렉토리 구조

```
connection/
├── __init__.py
├── README.md                          ← 이 파일
├── llm_command/
│   ├── __init__.py
│   ├── gemini_comment.py              # Gemini API → 건강 멘트 생성
│   └── bless_you_flow.py             # 전체 흐름 오케스트레이터
└── elven_labs/
    ├── __init__.py
    └── tts_player.py                  # ElevenLabs API → TTS 재생
```

---

## 재생 흐름

```
재채기 감지 (on_detect)
        │
        ├──────────────────────────────────────── 병렬 실행
        │                                          │
  bless_you.wav 재생 (aplay)          Gemini API → 건강 멘트 생성
        │                                          │
        └──────────────── 두 작업 완료 후 ─────────┘
                                   │
                    ElevenLabs TTS → 멘트 음성 재생
```

---

## 환경 변수 설정

두 API 모두 **환경 변수**로 키를 설정합니다.
코드에 API 키를 직접 입력하지 마세요.

### Gemini API 키

1. [Google AI Studio](https://aistudio.google.com/app/apikey) 에서 API 키 발급
2. 환경 변수 설정:

```bash
# 현재 세션에만 적용
export GEMINI_API_KEY="your-gemini-api-key-here"

# 영구 적용 (RPi의 경우 ~/.bashrc 또는 ~/.zshrc)
echo 'export GEMINI_API_KEY="your-gemini-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### ElevenLabs API 키

1. [ElevenLabs](https://elevenlabs.io) 회원가입 → My Account → API Keys
2. 환경 변수 설정:

```bash
# 현재 세션에만 적용
export ELEVENLABS_API_KEY="your-elevenlabs-api-key-here"

# 영구 적용
echo 'export ELEVENLABS_API_KEY="your-elevenlabs-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

### 설정 확인

```bash
echo $GEMINI_API_KEY       # 키가 출력되면 정상
echo $ELEVENLABS_API_KEY   # 키가 출력되면 정상
```

---

## 시스템 의존성 설치

ElevenLabs는 MP3 오디오를 반환합니다.
Raspberry Pi에서 MP3를 재생하려면 `mpg123`이 필요합니다.

```bash
sudo apt update
sudo apt install mpg123
```

> ffplay (ffmpeg) 또는 mplayer도 대안으로 지원됩니다.

---

## Python 패키지 설치

```bash
pip install -r requirements_rpi4.txt
```

또는 개별 설치:

```bash
pip install google-generativeai>=0.8.0
pip install elevenlabs>=1.0.0
```

---

## ElevenLabs 음성 ID 설정

기본 음성은 `Rachel` (영어 여성)입니다.
다른 음성을 사용하려면 `BlessYouFlow` 생성 시 `elevenlabs_voice_id`를 지정하거나,
`main.py`의 `BlessYouFlow(...)` 호출에 파라미터를 추가하세요.

사용 가능한 음성 목록 확인:

```python
from elevenlabs.client import ElevenLabs
import os

client = ElevenLabs(api_key=os.environ["ELEVENLABS_API_KEY"])
for voice in client.voices.get_all().voices:
    print(voice.voice_id, voice.name)
```

---

## 실행 방법

### main.py에서 사용 (권장)

```bash
cd src

# 영어 멘트 (기본값)
python -m src.main

# 한국어 멘트
python -m src.main --lang ko

# LLM/TTS 비활성화 (기존 WAV 재생만)
python -m src.main --no-llm
```

### 각 모듈 단독 테스트

```bash
cd src

# Gemini 멘트 생성 테스트 (영어)
python -m connection.llm_command.gemini_comment

# Gemini 멘트 생성 테스트 (한국어)
python -m connection.llm_command.gemini_comment ko

# ElevenLabs TTS 재생 테스트
python -m connection.elven_labs.tts_player "Stay warm and healthy!"

# 전체 흐름 테스트 (bless_you.wav → Gemini → ElevenLabs)
python -m connection.llm_command.bless_you_flow
python -m connection.llm_command.bless_you_flow ko
```

### 자동화 테스트 실행

```bash
cd src
python tests/test_connection.py
```

---

## 폴백 동작

API 키가 없거나 API 호출이 실패할 경우 자동으로 폴백합니다.

| 상황 | 동작 |
|------|------|
| API 키 없음 | 기존 `bless_you.wav` 재생으로 폴백 |
| Gemini API 실패 | 내장 기본 문구 사용 후 ElevenLabs 재생 |
| ElevenLabs API 실패 | 오류 메시지 출력 후 무음 처리 |
| mpg123 없음 | 오류 메시지 출력 후 무음 처리 |

---

## Gemini 멘트 생성 언어

| `--lang` | 예시 출력 |
|----------|-----------|
| `en` (기본) | "Drink plenty of water and get some rest!" |
| `ko` | "따뜻하게 입고 물 많이 드세요!" |
