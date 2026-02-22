"""
connection/elven_labs/tts_player.py
-------------------------------------
ElevenLabs REST API를 사용한 WAV 생성 및 저장 (통합 버전)

파이프라인:
1. GPS: 위치 조회
2. Weather: 날씨 정보 조회
3. Gemini: 건강 멘트 생성
4. ElevenLabs TTS: 음성 생성 및 WAV 저장

Usage
-----
# 생성 및 저장만
generator = ElevenLabsTTSGenerator()
wav_path = generator.generate_and_save("Stay warm!", "./sounds")

# 생성, 저장, 재생
player = ElevenLabsTTSPlayer(output_dir="./sounds")
player.speak("Stay warm and healthy!")
"""

import os
import json
import time
import hashlib
import requests
import tempfile
import subprocess
from pathlib import Path
from typing import Optional


# ===== ElevenLabs REST API 설정 =====
ELEVEN_BASE_URL = "https://api.elevenlabs.io"
ELEVEN_VOICE_ID = "hpp4J3VqNfWAUOO0d1Us"
DEFAULT_MODEL_ID = "eleven_multilingual_v2"

VOICE_SETTINGS = {
    "stability": 0.75,
    "similarity_boost": 0.90,
    "style": 0.10,
    "use_speaker_boost": True,
    "speed": 1.0
}

MAX_CHARS_PER_PROMPT = 800


def _text_hash(text: str) -> str:
    """텍스트의 짧은 해시값 반환"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def _atomic_write(path: Path, data: bytes) -> None:
    """원자적 파일 쓰기"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


class ElevenLabsTTSGenerator:
    """ElevenLabs REST API를 사용한 WAV 생성"""

    def __init__(self, api_key: Optional[str] = None, voice_id: str = ELEVEN_VOICE_ID):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "❌ ELEVENLABS_API_KEY가 없습니다. "
                "환경 변수를 설정하거나 .env 파일을 확인하세요."
            )
        self.voice_id = voice_id
        self.base_url = ELEVEN_BASE_URL

    def generate_wav(self, text: str, output_format: str = "wav_48000") -> bytes:
        """WAV 오디오 생성"""
        if len(text) > MAX_CHARS_PER_PROMPT:
            text = text[:MAX_CHARS_PER_PROMPT].rstrip()
            print(f"[TTSGenerator] ⚠ 텍스트 길이 제한 적용")

        url = f"{self.base_url}/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": DEFAULT_MODEL_ID,
            "voice_settings": VOICE_SETTINGS,
            "output_format": output_format,
        }

        try:
            print(f"[TTSGenerator] 🎤 API 호출 중... ({len(text)} chars)")
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            print(f"[TTSGenerator] ✓ WAV 생성됨 ({len(response.content)} bytes)")
            return response.content
        except Exception as e:
            if output_format == "wav_48000":
                print(f"[TTSGenerator] ⚠ 48kHz 실패 — 44.1kHz로 재시도")
                return self.generate_wav(text, "wav_44100")
            raise

    def generate_and_save(
        self,
        text: str,
        output_dir: Optional[Path] = None,
        save_as: Optional[Path] = None,
    ) -> Path:
        """WAV 생성 및 저장

        Parameters
        ----------
        save_as : 저장할 고정 경로. 지정 시 output_dir/타임스탬프 대신 해당 경로에 덮어씁니다.
        """
        if save_as is not None:
            output_path = Path(save_as)
        else:
            if output_dir is None:
                output_dir = Path("tts_output")
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            text_hash = _text_hash(text)
            output_path = output_dir / f"{timestamp}_{text_hash}.wav"

        audio_bytes = self.generate_wav(text)
        _atomic_write(output_path, audio_bytes)

        print(f"[TTSGenerator] ✓ 저장됨: {output_path}")
        return output_path


class ElevenLabsTTSPlayer:
    """ElevenLabs TTS 생성, 저장, 재생"""

    def __init__(self, api_key: Optional[str] = None, output_dir: Optional[Path] = None):
        self.generator = ElevenLabsTTSGenerator(api_key=api_key)
        self.output_dir = Path(output_dir) if output_dir else None
        self._player_cmd = self._find_player()

    def _find_player(self) -> Optional[list]:
        """오디오 플레이어 자동 감지"""
        player_commands = [
            ["aplay", "-q"],
            ["mpg123", "-q"],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
        ]
        for cmd in player_commands:
            try:
                subprocess.run(
                    [cmd[0], "--version"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                    timeout=2,
                )
                print(f"[TTSPlayer] ✓ 플레이어: {cmd[0]}")
                return cmd
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        return None

    def speak(
        self,
        text: str,
        save: bool = True,
        play: bool = True,
        save_as: Optional[Path] = None,
    ) -> Optional[Path]:
        """WAV 생성, 저장, 재생

        Parameters
        ----------
        save_as : 저장할 고정 경로. 지정 시 매번 같은 파일에 덮어씁니다.
        """
        if not text or not text.strip():
            return None

        try:
            # 1. 생성 및 저장
            if save:
                output_path = self.generator.generate_and_save(
                    text, self.output_dir, save_as=save_as
                )
            else:
                audio_bytes = self.generator.generate_wav(text)
                output_path = None

            # 2. 재생
            if play and output_path and output_path.exists():
                self._play_file(output_path)

            return output_path
        except Exception as e:
            print(f"[TTSPlayer] ❌ {e}")
            return None

    def _play_file(self, path: Path) -> None:
        """파일 재생"""
        if not self._player_cmd:
            print(f"[TTSPlayer] ⚠ 플레이어 없음")
            return
        try:
            subprocess.run(
                self._player_cmd + [str(path)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
            )
            print(f"[TTSPlayer] ✓ 재생됨")
        except Exception as e:
            print(f"[TTSPlayer] ⚠ 재생 실패: {e}")


if __name__ == "__main__":
    import sys
    text = sys.argv[1] if len(sys.argv) > 1 else "Stay warm and healthy!"
    
    try:
        generator = ElevenLabsTTSGenerator()
        wav_path = generator.generate_and_save(text, Path("tts_output"))
        print(f"✓ 완료: {wav_path}")
    except ValueError as e:
        print(f"❌ {e}")
