"""
tests/test_connection.py
-------------------------
Standalone tests for the connection module (Gemini + ElevenLabs).

API 키가 있으면 실제 API를 호출하고, 없으면 SKIP합니다.
실제 오디오 재생은 --no-audio 플래그로 건너뛸 수 있습니다.

Run from src/:
    python tests/test_connection.py              # API 키 있으면 실제 호출
    python tests/test_connection.py --no-audio   # TTS 재생 없이 텍스트 생성만 확인

Results:
    PASS  — 정상 동작
    SKIP  — API 키 또는 패키지 없음
    FAIL  — 예상치 못한 오류
"""

import os
import sys
import time
from pathlib import Path

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

_NO_AUDIO = "--no-audio" in sys.argv

_GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "")
_ELEVENLABS_KEY = os.environ.get("ELEVENLABS_API_KEY", "")

SOUNDS_DIR = Path(__file__).resolve().parents[1] / "output_feature" / "sounds"
BLESS_WAV  = SOUNDS_DIR / "bless_you.wav"


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #

def _pass(name: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  PASS  {name}{suffix}")


def _skip(name: str, reason: str) -> None:
    print(f"  SKIP  {name}  ({reason})")


def _fail(name: str, err: Exception) -> None:
    print(f"  FAIL  {name}  — {err}")


# ---------------------------------------------------------------------- #
# 패키지 임포트 테스트                                                     #
# ---------------------------------------------------------------------- #

def test_import_gemini_package():
    """google-generativeai 패키지 임포트 확인."""
    name = "import — google-generativeai"
    try:
        import google.generativeai  # noqa: F401
        _pass(name)
    except ImportError as e:
        _fail(name, ImportError(f"pip install google-generativeai  →  {e}"))


def test_import_elevenlabs_package():
    """elevenlabs 패키지 임포트 확인."""
    name = "import — elevenlabs"
    try:
        from elevenlabs.client import ElevenLabs  # noqa: F401
        _pass(name)
    except ImportError as e:
        _fail(name, ImportError(f"pip install elevenlabs  →  {e}"))


def test_import_connection_module():
    """connection 패키지 전체 임포트 확인."""
    name = "import — connection.BlessYouFlow"
    try:
        from connection import BlessYouFlow  # noqa: F401
        _pass(name)
    except ImportError as e:
        _fail(name, e)


# ---------------------------------------------------------------------- #
# Gemini 테스트                                                            #
# ---------------------------------------------------------------------- #

def test_gemini_env_key_present():
    """GEMINI_API_KEY 환경 변수 존재 확인."""
    name = "env — GEMINI_API_KEY is set"
    if _GEMINI_KEY:
        _pass(name, f"length={len(_GEMINI_KEY)}")
    else:
        _fail(name, EnvironmentError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다."))


def test_gemini_generate_english():
    """Gemini API로 영어 건강 멘트 생성 확인."""
    name = "GeminiCommentGenerator.generate — language=en"
    if not _GEMINI_KEY:
        _skip(name, "GEMINI_API_KEY not set")
        return
    try:
        from connection.llm_command.gemini_comment import GeminiCommentGenerator
        gen = GeminiCommentGenerator()
        t0 = time.time()
        text = gen.generate("en")
        elapsed = time.time() - t0

        assert isinstance(text, str), f"반환 타입이 str이 아님: {type(text)}"
        assert len(text) > 0, "빈 문자열이 반환됨"
        assert len(text) < 500, f"멘트가 너무 김 (len={len(text)})"
        _pass(name, f"{elapsed:.1f}s | '{text[:60]}…'" if len(text) > 60 else f"{elapsed:.1f}s | '{text}'")
    except Exception as e:
        _fail(name, e)


def test_gemini_generate_korean():
    """Gemini API로 한국어 건강 멘트 생성 확인."""
    name = "GeminiCommentGenerator.generate — language=ko"
    if not _GEMINI_KEY:
        _skip(name, "GEMINI_API_KEY not set")
        return
    try:
        from connection.llm_command.gemini_comment import GeminiCommentGenerator
        gen = GeminiCommentGenerator()
        t0 = time.time()
        text = gen.generate("ko")
        elapsed = time.time() - t0

        assert isinstance(text, str), f"반환 타입이 str이 아님: {type(text)}"
        assert len(text) > 0, "빈 문자열이 반환됨"
        _pass(name, f"{elapsed:.1f}s | '{text[:60]}…'" if len(text) > 60 else f"{elapsed:.1f}s | '{text}'")
    except Exception as e:
        _fail(name, e)


def test_gemini_fallback_on_invalid_key():
    """잘못된 API 키 입력 시 fallback 문구를 반환하는지 확인."""
    name = "GeminiCommentGenerator.generate — fallback on invalid key"
    try:
        from connection.llm_command.gemini_comment import GeminiCommentGenerator, _DEFAULT_FALLBACKS
        gen = GeminiCommentGenerator(api_key="invalid-key-for-test")
        result = gen.generate("en")

        # 실패 시 fallback 문구가 반환되어야 함
        assert isinstance(result, str), f"반환 타입이 str이 아님: {type(result)}"
        assert len(result) > 0, "빈 문자열이 반환됨"
        _pass(name, f"fallback='{result}'")
    except Exception as e:
        _fail(name, e)


# ---------------------------------------------------------------------- #
# ElevenLabs 테스트                                                        #
# ---------------------------------------------------------------------- #

def test_elevenlabs_env_key_present():
    """ELEVENLABS_API_KEY 환경 변수 존재 확인."""
    name = "env — ELEVENLABS_API_KEY is set"
    if _ELEVENLABS_KEY:
        _pass(name, f"length={len(_ELEVENLABS_KEY)}")
    else:
        _fail(name, EnvironmentError("ELEVENLABS_API_KEY 환경 변수가 설정되지 않았습니다."))


def test_elevenlabs_generate_audio_bytes():
    """ElevenLabs API로 오디오 바이트 생성 확인 (재생 없이)."""
    name = "ElevenLabsTTSPlayer._generate_audio — returns non-empty bytes"
    if not _ELEVENLABS_KEY:
        _skip(name, "ELEVENLABS_API_KEY not set")
        return
    try:
        from connection.elven_labs.tts_player import ElevenLabsTTSPlayer
        player = ElevenLabsTTSPlayer()
        t0 = time.time()
        audio_bytes = player._generate_audio("Stay healthy!")
        elapsed = time.time() - t0

        assert isinstance(audio_bytes, bytes), f"반환 타입이 bytes가 아님: {type(audio_bytes)}"
        assert len(audio_bytes) > 0, "빈 오디오 바이트가 반환됨"
        # MP3 magic bytes: ID3 또는 0xFF 0xFB
        is_mp3 = audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb"
        assert is_mp3, f"MP3 형식이 아님 (첫 4바이트: {audio_bytes[:4].hex()})"
        _pass(name, f"{elapsed:.1f}s | {len(audio_bytes):,} bytes")
    except Exception as e:
        _fail(name, e)


def test_elevenlabs_speak_short_text():
    """ElevenLabs speak() 실제 재생 테스트."""
    name = "ElevenLabsTTSPlayer.speak — plays audio"
    if not _ELEVENLABS_KEY:
        _skip(name, "ELEVENLABS_API_KEY not set")
        return
    if _NO_AUDIO:
        _skip(name, "--no-audio 플래그로 재생 건너뜀")
        return
    try:
        from connection.elven_labs.tts_player import ElevenLabsTTSPlayer
        player = ElevenLabsTTSPlayer()
        t0 = time.time()
        player.speak("Stay warm and healthy!")
        elapsed = time.time() - t0

        _pass(name, f"{elapsed:.1f}s")
    except Exception as e:
        _fail(name, e)


def test_elevenlabs_player_found():
    """시스템에 MP3 플레이어(mpg123 등)가 설치되어 있는지 확인."""
    name = "system — MP3 player available (mpg123 / ffplay / mplayer)"
    try:
        from connection.elven_labs.tts_player import _find_player
        cmd = _find_player()
        if cmd is not None:
            _pass(name, f"player='{cmd[0]}'")
        else:
            _fail(name, FileNotFoundError(
                "mpg123, ffplay, mplayer 중 하나도 설치되지 않았습니다. "
                "sudo apt install mpg123 을 실행하세요."
            ))
    except Exception as e:
        _fail(name, e)


# ---------------------------------------------------------------------- #
# BlessYouFlow 통합 테스트                                                 #
# ---------------------------------------------------------------------- #

def test_bless_you_flow_init():
    """API 키가 있을 때 BlessYouFlow 초기화 성공 확인."""
    name = "BlessYouFlow.__init__ — initializes without error"
    if not _GEMINI_KEY or not _ELEVENLABS_KEY:
        _skip(name, "GEMINI_API_KEY 또는 ELEVENLABS_API_KEY not set")
        return
    try:
        from connection.llm_command.bless_you_flow import BlessYouFlow
        flow = BlessYouFlow(bless_wav_path=BLESS_WAV, language="en")
        assert flow is not None
        _pass(name)
    except Exception as e:
        _fail(name, e)


def test_bless_you_flow_run_no_audio():
    """BlessYouFlow.run() — WAV 없이 Gemini + ElevenLabs만 검증."""
    name = "BlessYouFlow.run — Gemini + ElevenLabs (no WAV playback)"
    if not _GEMINI_KEY or not _ELEVENLABS_KEY:
        _skip(name, "GEMINI_API_KEY 또는 ELEVENLABS_API_KEY not set")
        return
    if _NO_AUDIO:
        _skip(name, "--no-audio 플래그로 재생 건너뜀")
        return
    try:
        from connection.llm_command.bless_you_flow import BlessYouFlow

        # 존재하지 않는 WAV 경로로 WAV 재생을 건너뛰고 Gemini + TTS만 검증
        flow = BlessYouFlow(
            bless_wav_path=Path("/tmp/nonexistent_bless.wav"),
            language="en",
        )
        t0 = time.time()
        flow.run()
        elapsed = time.time() - t0

        _pass(name, f"{elapsed:.1f}s")
    except Exception as e:
        _fail(name, e)


def test_bless_you_flow_wav_exists():
    """bless_you.wav 파일이 실제로 존재하는지 확인."""
    name = "asset — bless_you.wav exists"
    if BLESS_WAV.exists():
        _pass(name, str(BLESS_WAV))
    else:
        _fail(name, FileNotFoundError(f"파일 없음: {BLESS_WAV}"))


# ---------------------------------------------------------------------- #
# GPSLocator 테스트                                                        #
# ---------------------------------------------------------------------- #

def test_gps_import():
    """GPSLocator 임포트 확인."""
    name = "import — connection.gps.GPSLocator"
    try:
        from connection.gps.gps import GPSLocator  # noqa: F401
        _pass(name)
    except ImportError as e:
        _fail(name, e)


def test_gps_get_location():
    """GPSLocator.get_location() — 실제 네트워크 호출."""
    name = "GPSLocator.get_location — returns location dict"
    try:
        from connection.gps.gps import GPSLocator
        locator = GPSLocator()
        t0 = time.time()
        loc = locator.get_location()
        elapsed = time.time() - t0

        if loc is None:
            _skip(name, "네트워크 없음 또는 ip-api.com 응답 실패")
            return
        assert "city" in loc and "lat" in loc and "lon" in loc, f"필드 누락: {loc}"
        _pass(name, f"{elapsed:.1f}s | {loc['city']}, {loc['country']}")
    except Exception as e:
        _fail(name, e)


def test_gps_network_failure_returns_none():
    """잘못된 URL로 GPSLocator 호출 시 None 반환 확인 (graceful degradation)."""
    name = "GPSLocator.get_location — returns None on failure"
    try:
        import ml_model.config as cfg
        from connection.gps.gps import GPSLocator
        original = cfg.GPS_IP_API_URL
        cfg.GPS_IP_API_URL = "http://invalid.local.nonexistent/json/"
        locator = GPSLocator(timeout=2)
        result = locator.get_location()
        cfg.GPS_IP_API_URL = original
        assert result is None, f"None이 반환되어야 하는데 {result}가 반환됨"
        _pass(name)
    except Exception as e:
        _fail(name, e)


# ---------------------------------------------------------------------- #
# WeatherFetcher 테스트                                                    #
# ---------------------------------------------------------------------- #

def test_weather_import():
    """WeatherFetcher 임포트 확인."""
    name = "import — connection.weather.WeatherFetcher"
    try:
        from connection.weather.weather import WeatherFetcher  # noqa: F401
        _pass(name)
    except ImportError as e:
        _fail(name, e)


def test_weather_get_context_seoul():
    """WeatherFetcher.get_context() — 서울 좌표로 실제 API 호출."""
    name = "WeatherFetcher.get_context — Seoul (37.56, 126.97)"
    try:
        from connection.weather.weather import WeatherFetcher
        fetcher = WeatherFetcher()
        t0 = time.time()
        data = fetcher.get_context(lat=37.56, lon=126.97)
        elapsed = time.time() - t0

        if data is None:
            _skip(name, "네트워크 없음 또는 Open-Meteo 응답 실패")
            return
        required = {"temperature", "humidity", "aqi_label", "pm2_5"}
        missing = required - set(data.keys())
        assert not missing, f"필드 누락: {missing}"
        _pass(name, f"{elapsed:.1f}s | {data['temperature']}°C, AQI={data.get('us_aqi')} ({data['aqi_label']})")
    except Exception as e:
        _fail(name, e)


# ---------------------------------------------------------------------- #
# Gemini + Context 테스트                                                  #
# ---------------------------------------------------------------------- #

def test_gemini_generate_with_context():
    """Gemini API — 날씨/위치 컨텍스트 포함 멘트 생성 확인."""
    name = "GeminiCommentGenerator.generate — with context (en)"
    if not _GEMINI_KEY:
        _skip(name, "GEMINI_API_KEY not set")
        return
    try:
        from connection.llm_command.gemini_comment import GeminiCommentGenerator
        gen = GeminiCommentGenerator()
        context = {
            "city": "Seoul", "country": "South Korea",
            "temperature": 5.0, "humidity": 70,
            "weather_label": "Partly cloudy", "wind_speed": 3.0,
            "aqi_label": "Moderate", "pm2_5": 35.0, "pm10": 45.0,
        }
        t0 = time.time()
        text = gen.generate("en", context=context)
        elapsed = time.time() - t0
        assert isinstance(text, str) and len(text) > 0
        _pass(name, f"{elapsed:.1f}s | '{text[:60]}…'" if len(text) > 60 else f"{elapsed:.1f}s | '{text}'")
    except Exception as e:
        _fail(name, e)


def test_gemini_generate_context_none_fallback():
    """context=None 시 기존 프롬프트 방식으로 정상 동작 확인."""
    name = "GeminiCommentGenerator.generate — context=None backward compat"
    if not _GEMINI_KEY:
        _skip(name, "GEMINI_API_KEY not set")
        return
    try:
        from connection.llm_command.gemini_comment import GeminiCommentGenerator
        gen = GeminiCommentGenerator()
        text = gen.generate("en", context=None)
        assert isinstance(text, str) and len(text) > 0
        _pass(name, f"'{text[:60]}'")
    except Exception as e:
        _fail(name, e)


# ---------------------------------------------------------------------- #
# Runner                                                                  #
# ---------------------------------------------------------------------- #

if __name__ == "__main__":
    print("=== test_connection ===")
    print(f"  GEMINI_API_KEY   : {'설정됨' if _GEMINI_KEY else '미설정'}")
    print(f"  ELEVENLABS_API_KEY: {'설정됨' if _ELEVENLABS_KEY else '미설정'}")
    print(f"  --no-audio       : {_NO_AUDIO}")
    print()

    print("[ 패키지 임포트 ]")
    test_import_gemini_package()
    test_import_elevenlabs_package()
    test_import_connection_module()

    print()
    print("[ 환경 변수 ]")
    test_gemini_env_key_present()
    test_elevenlabs_env_key_present()

    print()
    print("[ Gemini API ]")
    test_gemini_generate_english()
    test_gemini_generate_korean()
    test_gemini_fallback_on_invalid_key()

    print()
    print("[ ElevenLabs API ]")
    test_elevenlabs_player_found()
    test_elevenlabs_generate_audio_bytes()
    test_elevenlabs_speak_short_text()

    print()
    print("[ BlessYouFlow 통합 ]")
    test_bless_you_flow_wav_exists()
    test_bless_you_flow_init()
    test_bless_you_flow_run_no_audio()

    print()
    print("[ GPS ]")
    test_gps_import()
    test_gps_get_location()
    test_gps_network_failure_returns_none()

    print()
    print("[ Weather + Air Quality ]")
    test_weather_import()
    test_weather_get_context_seoul()

    print()
    print("[ Gemini + Context ]")
    test_gemini_generate_with_context()
    test_gemini_generate_context_none_fallback()

    print()
    print("======================")
