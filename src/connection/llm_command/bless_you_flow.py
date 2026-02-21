"""
connection/llm_command/bless_you_flow.py
-----------------------------------------
재채기 감지 후 전체 응답 흐름을 조율하는 오케스트레이터.

흐름
----
1. bless_you.wav 재생 (aplay, 기존 방식 유지)
2. Gemini API로 건강 멘트 생성  ─┐ 병렬 실행
   (bless_you.wav 재생 중 동시에)  ─┘
3. bless_you.wav 완료 후 ElevenLabs TTS로 멘트 재생

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

from .gemini_comment import GeminiCommentGenerator
from ..elven_labs.tts_player import ElevenLabsTTSPlayer


class BlessYouFlow:
    """bless_you.wav + Gemini 멘트 생성 + ElevenLabs TTS 재생을 조율합니다.

    Parameters
    ----------
    bless_wav_path      : bless_you.wav 파일 경로.
    gemini_api_key      : Gemini API 키 (None이면 환경 변수 사용).
    elevenlabs_api_key  : ElevenLabs API 키 (None이면 환경 변수 사용).
    elevenlabs_voice_id : ElevenLabs 음성 ID.
    language            : 멘트 생성 언어 ("en" 또는 "ko").
    """

    def __init__(
        self,
        bless_wav_path: Path,
        gemini_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        elevenlabs_voice_id: str = "Rachel",
        language: str = "en",
    ):
        self.bless_wav_path = Path(bless_wav_path)
        self.language = language

        self._gemini = GeminiCommentGenerator(api_key=gemini_api_key)
        self._tts = ElevenLabsTTSPlayer(
            api_key=elevenlabs_api_key,
            voice_id=elevenlabs_voice_id,
        )

    def run(self) -> None:
        """전체 흐름을 블로킹으로 실행합니다.

        1) bless_you.wav 재생과 Gemini API 호출을 병렬로 시작.
        2) 둘 다 완료된 후 ElevenLabs TTS로 멘트 재생.
        """
        comment_holder: list[str] = []
        error_holder: list[Exception] = []

        # --- Gemini 멘트 생성 스레드 ---
        def _generate():
            try:
                text = self._gemini.generate(self.language)
                comment_holder.append(text)
                print(f"[BlessYouFlow] 💬 생성된 멘트: {text}")
            except Exception as e:
                error_holder.append(e)
                print(f"[BlessYouFlow] ❌ Gemini 오류: {e}")

        gemini_thread = threading.Thread(target=_generate, daemon=True)
        gemini_thread.start()

        # --- bless_you.wav 재생 (메인 스레드에서 블로킹) ---
        self._play_wav(self.bless_wav_path)

        # --- Gemini 완료 대기 ---
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
