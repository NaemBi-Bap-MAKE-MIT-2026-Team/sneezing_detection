import os
import time
import json
import hashlib
from pathlib import Path
import requests
import pandas as pd

# ===== 설정 =====
OUTPUT_DIR = Path("output-feature/sounds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ELEVEN_API_KEY = os.environ["ELEVENLABS_API_KEY"]
VOICE_ID = "hpp4J3VqNfWAUOO0d1Us"
BASE_URL = "https://api.elevenlabs.io"

# 영어만이면 multilingual도 가능하지만, 효민님 환경에 맞게 그대로 두셔도 됩니다.
MODEL_ID = "eleven_multilingual_v2"

VOICE_SETTINGS = {
    "stability": 0.75,
    "similarity_boost": 0.90,
    "style": 0.10,
    "use_speaker_boost": True,
    "speed": 1.0
}

MAX_PROMPTS = 50
MAX_CHARS_PER_PROMPT = 800  # 너무 길면 품질/비용/지연이 흔들려서 안전 상한 권장

# ===== 유틸 =====
def text_hash8(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]

def atomic_write_bytes(final_path: Path, data: bytes) -> None:
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with open(tmp_path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)

def eleven_tts(text: str, output_format: str) -> bytes:
    # Convert endpoint: /v1/text-to-speech/{voice_id}
    # output_format을 바꿔가며 WAV를 받을 수 있습니다.
    # 공식 API 레퍼런스 참고 :contentReference[oaicite:6]{index=6}
    url = f"{BASE_URL}/v1/text-to-speech/{VOICE_ID}"
    headers = {
        "xi-api-key": ELEVEN_API_KEY,
        "Content-Type": "application/json",
        # WAV로 받겠다는 의도를 명확히 합니다(서버는 output_format이 더 중요)
        "Accept": "audio/wav",
    }
    payload = {
        "text": text,
        "model_id": MODEL_ID,
        "voice_settings": VOICE_SETTINGS,
        "output_format": output_format,
    }
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    return r.content

# ===== Gemini 프롬프트 생성(효민님 구현으로 교체) =====
def generate_prompts_gemini() -> list[str]:
    # TODO: 효민님이 사용하는 Gemini SDK/REST 호출로 교체
    # 반드시 영어만 나오도록 system prompt를 고정하시고,
    # 리스트 길이는 최대 50을 넘기지 않게 하십시오.
    return [
        "Short test prompt one.",
        "Short test prompt two.",
    ]

def main():
    prompts = generate_prompts_gemini()
    prompts = [p.strip() for p in prompts if p and p.strip()]
    prompts = prompts[:MAX_PROMPTS]

    run_ts = time.strftime("%Y%m%d_%H%M%S")

    saved = []
    for idx, text in enumerate(prompts, start=1):
        # 길이 상한
        if len(text) > MAX_CHARS_PER_PROMPT:
            text = text[:MAX_CHARS_PER_PROMPT].rstrip()

        h = text_hash8(text)
        filename = f"{run_ts}_{idx:02d}_{h}.wav"
        out_path = OUTPUT_DIR / filename

        # 48k 시도 후 44.1k 폴백
        try:
            audio = eleven_tts(text, output_format="wav_48000")
        except Exception:
            audio = eleven_tts(text, output_format="wav_44100")

        atomic_write_bytes(out_path, audio)
        saved.append({
            "index": idx,
            "file": str(out_path),
            "hash8": h,
            "text": text,
        })

    # 실행 결과 기록(나중에 재생/디버깅에 중요)
    manifest_path = OUTPUT_DIR / f"{run_ts}_manifest.json"
    atomic_write_bytes(manifest_path, json.dumps(saved, ensure_ascii=False, indent=2).encode("utf-8"))

if __name__ == "__main__":
    main()