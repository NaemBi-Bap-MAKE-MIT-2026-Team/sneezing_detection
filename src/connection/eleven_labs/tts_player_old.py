"""
connection/elven_labs/tts_player.py
-------------------------------------
ElevenLabs API를 사용하여 텍스트를 음성으로 변환하고 재생합니다.

Raspberry Pi에서 mpg123 (MP3) 또는 aplay (WAV) 중 가용한 플레이어를
자동으로 선택하여 재생합니다.

Usage
-----
player = ElevenLabsTTSPlayer(api_key="YOUR_KEY", voice_id="Rachel")
player.speak("Stay warm and healthy!")

Environment
-----------
ELEVENLABS_API_KEY 환경 변수로 API 키를 설정할 수 있습니다.
"""

import io
import os
import subprocess
import tempfile
from typing import Optional

try:
    from elevenlabs.client import ElevenLabs
    _ELEVEN_AVAILABLE = True
except ImportError:
    _ELEVEN_AVAILABLE = False


# RPi에서 MP3 재생에 사용할 외부 커맨드 우선순위
_PLAYER_COMMANDS = [
    ["mpg123", "-q"],   # 가장 가벼운 MP3 플레이어
    ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
    ["mplayer", "-really-quiet"],
]

# ElevenLabs 기본 음성 ID (Rachel — 영어 여성 음성)
DEFAULT_VOICE_ID = "Rachel"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"


def _find_player() -> Optional[list]:
    """사용 가능한 MP3 플레이어 명령을 찾아 반환합니다."""
    for cmd in _PLAYER_COMMANDS:
        try:
            subprocess.run(
                [cmd[0], "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return cmd
        except FileNotFoundError:
            continue
    return None


class ElevenLabsTTSPlayer:
    """ElevenLabs API TTS 재생 클래스.

    Parameters
    ----------
    api_key   : ElevenLabs API 키. None이면 환경 변수 ELEVENLABS_API_KEY 사용.
    voice_id  : 사용할 음성 ID 또는 음성 이름. 기본값은 "Rachel".
    model_id  : 사용할 모델 ID. 다국어 지원은 eleven_multilingual_v2 권장.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
    ):
        if not _ELEVEN_AVAILABLE:
            raise ImportError(
                "elevenlabs 패키지가 설치되지 않았습니다. "
                "pip install elevenlabs 를 실행하세요."
            )

        resolved_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "ElevenLabs API 키가 없습니다. "
                "ElevenLabsTTSPlayer(api_key=...) 또는 "
                "ELEVENLABS_API_KEY 환경 변수를 설정하세요."
            )

        self.client = ElevenLabs(api_key=resolved_key)
        self.voice_id = voice_id
        self.model_id = model_id
        self._player_cmd = _find_player()

        if self._player_cmd is None:
            print(
                "[ElevenLabsTTS] ⚠ MP3 플레이어(mpg123/ffplay/mplayer)를 찾지 못했습니다. "
                "sudo apt install mpg123 을 실행하세요."
            )

    def speak(self, text: str) -> None:
        """텍스트를 ElevenLabs TTS로 변환하고 재생합니다.

        Parameters
        ----------
        text : 재생할 텍스트.
        """
        if not text.strip():
            return

        try:
            audio_bytes = self._generate_audio(text)
            self._play_mp3(audio_bytes)
        except Exception as e:
            print(f"[ElevenLabsTTS] ❌ 재생 실패: {e}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_audio(self, text: str) -> bytes:
        """ElevenLabs API로 MP3 오디오 바이트를 생성합니다."""
        audio_generator = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model_id,
        )
        return b"".join(audio_generator)

    def _play_mp3(self, audio_bytes: bytes) -> None:
        """MP3 바이트를 임시 파일에 저장하고 외부 플레이어로 재생합니다."""
        if self._player_cmd is None:
            print("[ElevenLabsTTS] 플레이어 없음 — 재생 건너뜀.")
            return

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            subprocess.run(
                self._player_cmd + [tmp_path],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    # 단독 실행 테스트
    import sys

    text = sys.argv[1] if len(sys.argv) > 1 else "Stay warm and drink plenty of water!"
    print(f"[ElevenLabsTTS] 테스트 재생: '{text}'")

    try:
        player = ElevenLabsTTSPlayer()
        player.speak(text)
        print("[ElevenLabsTTS] ✓ 재생 완료")
    except ValueError as e:
        print(f"[오류] {e}")
        print("환경 변수 ELEVENLABS_API_KEY 를 설정하고 다시 실행하세요.")
