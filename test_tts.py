#!/usr/bin/env python3
"""
TTS 생성 단위 테스트
"""

from dotenv import load_dotenv
from pathlib import Path
import os
import sys

# .env 로드
env_path = Path(__file__).parent / "src" / ".env"
load_dotenv(env_path)

# TTS 모듈 임포트
from src.connection.eleven_labs.tts_player import ElevenLabsTTSGenerator

print("\n" + "=" * 80)
print("🧪 TTS 생성 단위 테스트")
print("=" * 80)

# API 키 확인  
api_key = os.getenv("ELEVENLABS_API_KEY")
if api_key:
    print(f"✓ API 키 로드됨: {api_key[:15]}...{api_key[-10:]}")
else:
    print("❌ API 키 없음 (ELEVENLABS_API_KEY)")
    sys.exit(1)

try:
    # 생성기 초기화
    print("\n[1] ElevenLabsTTSGenerator 초기화")
    gen = ElevenLabsTTSGenerator()
    print("✓ 생성기 초기화 완료")
    
    # 테스트 텍스트
    text = "Stay warm and drink plenty of water today!"
    print(f"\n[2] WAV 생성")
    print(f"📝 텍스트: \"{text}\"")
    print()
    
    # 생성 및 저장
    output_dir = Path(__file__).parent / "tts_test_output"
    wav_path = gen.generate_and_save(text, output_dir)
    
    print()
    # 파일 검증
    if wav_path.exists():
        file_size = wav_path.stat().st_size
        print(f"✓ WAV 파일 생성 성공!")
        print(f"  📁 파일명: {wav_path.name}")
        print(f"  📊 파일 크기: {file_size:,} bytes")
        print(f"  📍 전체 경로: {wav_path.absolute()}")
        
        print("\n" + "=" * 80)
        print(f"✅ 테스트 완료! WAV 파일 생성됨")
        print("=" * 80 + "\n")
    else:
        print(f"❌ 파일이 존재하지 않음: {wav_path}")
        sys.exit(1)

except Exception as e:
    print(f"\n❌ 오류 발생: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
