"""
connection/bless_you_flow.py
-----------------------------------------
재채기 감지 후 전체 응답 흐름을 조율하는 오케스트레이터.

Pipeline
--------
Stage 1  bless_you.wav 재생  +  GPS + 날씨/대기질 조회  (병렬)
Stage 2  Gemini API로 건강 멘트 선택/생성  (배치 캐시 활용)
Stage 3  ElevenLabs TTS로 멘트 재생

Usage
-----
flow = BlessYouFlow(
    bless_wav_path=Path("..."),
    gemini_api_key="...",
    elevenlabs_api_key="...",
    language="en",
)
flow.run()          # 블로킹 (전체 흐름 완료까지 대기)
flow.run_async()    # 논블로킹 (백그라운드 스레드)
"""

import subprocess
import threading
from pathlib import Path
from typing import Optional

from .gemini.gemini_comment import GeminiCommentGenerator
from .eleven_labs.tts_player import ElevenLabsTTSPlayer

try:
    from .gps.gps import GPSLocator
    from .weather.weather import WeatherFetcher
    _CONTEXT_AVAILABLE = True
except ImportError as _ctx_err:
    print(f"[BlessYouFlow] ⚠ GPS/Weather 모듈 불러오기 실패: {_ctx_err}")
    _CONTEXT_AVAILABLE = False


class BlessYouFlow:
    """재채기 감지 후 WAV → GPS/날씨 → Gemini → ElevenLabs TTS 파이프라인.

    Parameters
    ----------
    bless_wav_path      : bless_you.wav 파일 경로.
    gemini_api_key      : Gemini API 키 (None이면 환경 변수 GEMINI_API_KEY 사용).
    elevenlabs_api_key  : ElevenLabs API 키 (None이면 환경 변수 ELEVENLABS_API_KEY 사용).
    elevenlabs_voice_id : ElevenLabs 음성 ID.
    language            : 멘트 언어 ("en" 또는 "ko").
    enable_context      : GPS/날씨 컨텍스트 수집 활성화 여부 (기본값: True).
    num_messages        : Gemini 배치 메시지 수 (캐시 크기). 기본값 30.
    """

    _CTX_TIMEOUT = 8.0   # GPS+Weather 완료 대기 최대 시간 (초)
    _GEN_TIMEOUT = 10.0  # Gemini 생성 대기 최대 시간 (초)

    def __init__(
        self,
        bless_wav_path: Path,
        gemini_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        elevenlabs_voice_id: str = "Rachel",
        language: str = "en",
        enable_context: bool = True,
        num_messages: int = 30,
    ):
        self.bless_wav_path = Path(bless_wav_path)
        self.language = language
        self._enable_context = enable_context and _CONTEXT_AVAILABLE
        self._num_messages = num_messages
        self._message_cache: list[str] = []

        self._gemini = GeminiCommentGenerator(api_key=gemini_api_key)
        
        # TTS 저장 경로 설정
        tts_output_dir = Path(__file__).resolve().parent.parent / "output_feature" / "sounds"
        tts_output_dir.mkdir(parents=True, exist_ok=True)
        
        self._tts = ElevenLabsTTSPlayer(
            api_key=elevenlabs_api_key,
            output_dir=tts_output_dir,
        )

        if self._enable_context:
            self._gps = GPSLocator()
            self._weather = WeatherFetcher()
        else:
            self._gps = None
            self._weather = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """파이프라인 전체를 블로킹으로 실행합니다.

        Stage 1  WAV 재생 + GPS/날씨 조회 (병렬)
        Stage 2  Gemini 건강 멘트 선택/생성
        Stage 3  ElevenLabs TTS 재생
        """
        # Stage 1 ─ WAV 재생과 컨텍스트 조회를 병렬 실행
        ctx = self._stage1_wav_and_context()

        # Stage 2 ─ 캐시에서 멘트 꺼내거나 Gemini API로 생성
        comment = self._stage2_get_comment(ctx)

        # Stage 3 ─ ElevenLabs TTS 재생
        self._stage3_speak(comment)

    def run_async(self) -> threading.Thread:
        """파이프라인을 백그라운드 스레드에서 실행합니다.

        Returns
        -------
        threading.Thread
            실행 중인 스레드. join()으로 완료를 대기할 수 있습니다.
        """
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def _stage1_wav_and_context(self) -> Optional[dict]:
        """[Stage 1] WAV 재생과 GPS/날씨 조회를 병렬로 실행하고 컨텍스트를 반환합니다."""
        ctx_result: list[Optional[dict]] = []
        ctx_thread = threading.Thread(
            target=lambda: ctx_result.append(self._fetch_context()),
            daemon=True,
        )
        ctx_thread.start()

        self._play_wav(self.bless_wav_path)             # 메인 스레드 블로킹 (WAV 재생)
        ctx_thread.join(timeout=self._CTX_TIMEOUT)      # WAV 재생 후 컨텍스트 완료 대기

        ctx = ctx_result[0] if ctx_result else None
        if ctx:
            print(
                f"[BlessYouFlow] 📍 {ctx['city']}, {ctx['country']} "
                f"| {ctx['temperature']}°C | AQI {ctx['aqi_label']}"
            )
        return ctx

    def _stage2_get_comment(self, ctx: Optional[dict]) -> str:
        """[Stage 2] 캐시에서 멘트를 반환하거나, 비었으면 Gemini API로 배치 생성합니다."""
        # 컨텍스트 정제: None 값 필터링 및 완성도 확인
        if ctx:
            ctx_clean = {k: v for k, v in ctx.items() if v is not None}
            # 핵심 필드가 5개 이상 있어야 컨텍스트 사용
            if len(ctx_clean) < 5:
                print(
                    f"[BlessYouFlow] ⚠ 불완전한 컨텍스트 ({len(ctx_clean)}/8) "
                    "— 기본 프롬프트 사용"
                )
                ctx = None
            else:
                ctx = ctx_clean

        result: list[str] = []

        def _generate():
            if not self._message_cache:
                self._message_cache = self._gemini.generate_batch(
                    num_messages=self._num_messages,
                    language=self.language,
                    context=ctx,
                )
                if self._message_cache:
                    print(f"[BlessYouFlow] 🔄 {len(self._message_cache)}개 메시지 캐시됨")

            if self._message_cache:
                result.append(self._message_cache.pop(0))
            else:
                # 배치 생성 실패 시 단일 생성으로 폴백
                result.append(self._gemini.generate(self.language, context=ctx))

        gen_thread = threading.Thread(target=_generate, daemon=True)
        gen_thread.start()
        gen_thread.join(timeout=self._GEN_TIMEOUT)

        comment = result[0] if result else ""
        if comment:
            print(f"[BlessYouFlow] 💬 {comment}")
        return comment

    def _stage3_speak(self, comment: str) -> Optional[Path]:
        """[Stage 3] ElevenLabs TTS로 멘트를 생성 및 저장합니다."""
        if comment:
            # TTS 생성, 저장, 재생 (save=True, play=False로 저장만 진행)
            wav_path = self._tts.speak(comment, save=True, play=False)
            if wav_path:
                print(f"[BlessYouFlow] 🎵 WAV 저장: {wav_path}")
            return wav_path
        else:
            print("[BlessYouFlow] ⚠ 멘트 없음 — TTS 건너뜀.")
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_context(self) -> Optional[dict]:
        """GPS + 날씨/대기질 컨텍스트를 조회합니다.

        Returns
        -------
        dict | None
            Gemini 프롬프트에 전달할 컨텍스트 딕셔너리. 조회 실패 시 None.
        """
        if not self._enable_context:
            return None
        try:
            location = self._gps.get_location()
            if not location:
                print("[BlessYouFlow] ⚠ GPS 조회 실패 — 컨텍스트 없이 진행")
                return None
            weather = self._weather.get_context(location["lat"], location["lon"], city=location["city"])
            if not weather:
                print("[BlessYouFlow] ⚠ 날씨 조회 실패 — 컨텍스트 없이 진행")
                return None
            return {
                "city":          location["city"],
                "country":       location["country"],
                "temperature":   weather["temperature"],
                "humidity":      weather["humidity"],
                "weather_label": weather["weather_label"],
                "wind_speed":    weather["wind_speed"],
                "aqi_label":     weather["aqi_label"],
                "pm2_5":         weather["pm2_5"],
                "pm10":          weather["pm10"],
            }
        except Exception as e:
            print(f"[BlessYouFlow] ⚠ 컨텍스트 조회 오류: {e}")
            return None

    def _play_wav(self, wav_path: Path) -> None:
        """aplay로 WAV 파일을 동기 재생합니다 (Linux용)."""
        if not wav_path.exists():
            print(f"[BlessYouFlow] ⚠ WAV 없음: {wav_path}")
            return
        try:
            subprocess.run(["aplay", "-q", str(wav_path)], check=False)
        except FileNotFoundError:
            print("[BlessYouFlow] ⚠ aplay 없음 — WAV 건너뜀.")
        except Exception as e:
            print(f"[BlessYouFlow] ⚠ WAV 재생 오류: {e}")


if __name__ == "__main__":
    # 단독 실행 테스트
    import sys
    from pathlib import Path

    wav = Path(__file__).resolve().parents[2] / "output_feature" / "sounds" / "bless_you.wav"
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    print(f"[BlessYouFlow] 테스트 실행 (language={lang})")
    print(f"[BlessYouFlow] WAV: {wav}")

    try:
        flow = BlessYouFlow(bless_wav_path=wav, language=lang)
        flow.run()
        print("[BlessYouFlow] ✓ 완료")
    except (ValueError, ImportError) as e:
        print(f"[오류] {e}")
        print("GEMINI_API_KEY 와 ELEVENLABS_API_KEY 환경 변수를 설정하고 다시 실행하세요.")
