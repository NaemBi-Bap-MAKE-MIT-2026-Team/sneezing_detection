"""
connection/llm_command/bless_you_flow.py
-----------------------------------------
재채기 감지 후 전체 응답 흐름을 조율하는 오케스트레이터.

흐름
----
1. bless_you.wav 재생 (aplay, 기존 방식 유지)
   GPS + 날씨/대기질 조회 (병렬, WAV 재생 중 동시에)
2. 위치/날씨 컨텍스트를 포함하여 Gemini API로 건강 멘트 생성
3. ElevenLabs TTS로 멘트 재생

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
from .elven_labs.tts_player import ElevenLabsTTSPlayer

try:
    from .gps.gps import GPSLocator
    from .weather.weather import WeatherFetcher
    _CONTEXT_AVAILABLE = True
except ImportError as _ctx_err:
    print(f"[BlessYouFlow] ⚠ GPS/Weather 모듈 불러오기 실패: {_ctx_err}")
    _CONTEXT_AVAILABLE = False


class BlessYouFlow:
    """bless_you.wav + GPS/날씨 컨텍스트 + Gemini 멘트 생성 + ElevenLabs TTS 재생을 조율합니다.

    Parameters
    ----------
    bless_wav_path      : bless_you.wav 파일 경로.
    gemini_api_key      : Gemini API 키 (None이면 환경 변수 사용).
    elevenlabs_api_key  : ElevenLabs API 키 (None이면 환경 변수 사용).
    elevenlabs_voice_id : ElevenLabs 음성 ID.
    language            : 멘트 생성 언어 ("en" 또는 "ko").
    enable_context      : GPS/날씨 컨텍스트 수집 활성화 여부 (기본값: True).
    num_messages        : Gemini 배치 생성 메시지 수. 캐시로 저장되어 API 호출을 줄입니다.
    """

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
        self._tts = ElevenLabsTTSPlayer(
            api_key=elevenlabs_api_key,
            voice_id=elevenlabs_voice_id,
        )

        if self._enable_context:
            self._gps = GPSLocator()
            self._weather = WeatherFetcher()
        else:
            self._gps = None
            self._weather = None

    def run(self) -> None:
        """전체 흐름을 블로킹으로 실행합니다.

        1) bless_you.wav 재생과 GPS+날씨 조회를 병렬로 시작.
        2) 둘 다 완료된 후 컨텍스트를 포함하여 Gemini API 호출.
        3) ElevenLabs TTS로 멘트 재생.
        """
        comment_holder: list[str] = []
        context_holder: list[Optional[dict]] = []

        # --- GPS + 날씨/대기질 조회 스레드 (WAV 재생 중 병렬 실행) ---
        def _fetch_ctx():
            ctx = self._fetch_context()
            context_holder.append(ctx)
            if ctx:
                print(
                    f"[BlessYouFlow] 📍 {ctx['city']}, {ctx['country']} "
                    f"| {ctx['temperature']}°C "
                    f"| AQI {ctx['aqi_label']}"
                )

        context_thread = threading.Thread(target=_fetch_ctx, daemon=True)
        context_thread.start()

        # --- bless_you.wav 재생 (메인 스레드 블로킹) ---
        self._play_wav(self.bless_wav_path)

        # --- GPS/날씨 완료 대기 (WAV 재생 시간 내에 대부분 완료됨) ---
        context_thread.join(timeout=8.0)

        # --- Gemini 멘트 생성 (컨텍스트 포함) ---
        ctx = context_holder[0] if context_holder else None

        def _generate():
            try:
                # 캐시가 비었으면 배치 생성으로 채움
                if not self._message_cache:
                    self._message_cache = self._gemini.generate_batch(
                        num_messages=self._num_messages,
                        language=self.language,
                        context=ctx,
                    )
                    print(
                        f"[BlessYouFlow] 🔄 메시지 배치 생성 완료 "
                        f"({len(self._message_cache)}개 캐시됨)"
                    )

                if self._message_cache:
                    text = self._message_cache.pop(0)
                else:
                    # 배치 생성 실패 시 단일 생성으로 폴백
                    text = self._gemini.generate(self.language, context=ctx)

                comment_holder.append(text)
                print(f"[BlessYouFlow] 💬 생성된 멘트: {text}")
            except Exception as e:
                print(f"[BlessYouFlow] ❌ Gemini 오류: {e}")

        gemini_thread = threading.Thread(target=_generate, daemon=True)
        gemini_thread.start()
        gemini_thread.join(timeout=10.0)

        # --- ElevenLabs TTS 재생 ---
        if comment_holder:
            self._tts.speak(comment_holder[0])
        else:
            print("[BlessYouFlow] ⚠ 멘트 없음 — TTS 재생 건너뜀.")

    def run_async(self) -> threading.Thread:
        """전체 흐름을 백그라운드 스레드에서 비동기 실행합니다.

        Returns
        -------
        threading.Thread
            실행 중인 스레드 객체. join()으로 완료를 대기할 수 있습니다.
        """
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        return t

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_context(self) -> Optional[dict]:
        """GPS + 날씨 + 대기질 컨텍스트를 조회합니다.

        Returns
        -------
        dict | None
            Gemini 프롬프트에 전달할 컨텍스트 딕셔너리.
            조회 실패 시 None을 반환합니다.
        """
        if not self._enable_context:
            return None
        try:
            location = self._gps.get_location()
            if not location:
                print("[BlessYouFlow] ⚠ GPS 조회 실패 — 컨텍스트 없이 진행")
                return None
            weather = self._weather.get_context(location["lat"], location["lon"])
            if not weather:
                print("[BlessYouFlow] ⚠ 날씨 조회 실패 — 컨텍스트 없이 진행")
                return None
            return {
                "city": location["city"],
                "country": location["country"],
                "temperature": weather["temperature"],
                "humidity": weather["humidity"],
                "weather_label": weather["weather_label"],
                "wind_speed": weather["wind_speed"],
                "aqi_label": weather["aqi_label"],
                "pm2_5": weather["pm2_5"],
                "pm10": weather["pm10"],
            }
        except Exception as e:
            print(f"[BlessYouFlow] ⚠ 컨텍스트 조회 중 예외: {e}")
            return None

    def _play_wav(self, wav_path: Path) -> None:
        """aplay로 WAV 파일을 동기 재생합니다."""
        if not wav_path.exists():
            print(f"[BlessYouFlow] ⚠ WAV 없음: {wav_path}")
            return
        try:
            subprocess.run(
                ["aplay", "-q", str(wav_path)],
                check=False,
            )
        except FileNotFoundError:
            print("[BlessYouFlow] ⚠ aplay 없음 — WAV 재생 건너뜀.")
        except Exception as e:
            print(f"[BlessYouFlow] ⚠ WAV 재생 오류: {e}")


if __name__ == "__main__":
    # 단독 실행 테스트
    import sys
    from pathlib import Path

    # 기본 bless_you.wav 위치
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
