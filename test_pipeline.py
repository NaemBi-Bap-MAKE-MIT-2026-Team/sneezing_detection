#!/usr/bin/env python3
"""
파이프라인 통합 테스트
GPS → Weather → Gemini → ElevenLabs TTS

Usage:
    python test_pipeline.py [language]
    python test_pipeline.py en
    python test_pipeline.py ko
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트 설정
sys.path.insert(0, str(Path(__file__).parent))

# .env 로드
from dotenv import load_dotenv
env_path = Path(__file__).parent / "src" / ".env"
load_dotenv(env_path)

from src.connection.bless_you_flow import BlessYouFlow


def main():
    language = sys.argv[1] if len(sys.argv) > 1 else "en"
    
    print("\n" + "=" * 80)
    print("🎯 파이프라인 통합 테스트")
    print("=" * 80)
    print(f"언어: {language}")
    print(f"API 키 확인:")
    print(f"  - GEMINI_API_KEY: {'✓' if os.getenv('GEMINI_API_KEY') else '❌'}")
    print(f"  - ELEVENLABS_API_KEY: {'✓' if os.getenv('ELEVENLABS_API_KEY') else '❌'}")

    # WAV 저장 경로 생성
    output_dir = Path(__file__).parent / "src" / "output_feature" / "sounds"
    output_dir.mkdir(parents=True, exist_ok=True)

    # bless_you.wav 경로
    bless_wav = output_dir / "bless_you.wav"
    
    try:
        # BlessYouFlow 초기화
        print("\n[초기화] BlessYouFlow 생성 중...")
        flow = BlessYouFlow(
            bless_wav_path=bless_wav,
            language=language,
            enable_context=True,
        )
        print("✓ BlessYouFlow 초기화 완료")
        
        # Stage 1: GPS + Weather
        print("\n" + "-" * 80)
        print("[1️⃣ Stage 1] WAV 재생 + GPS/Weather 조회")
        print("-" * 80)
        ctx = flow._stage1_wav_and_context()
        
        if ctx:
            print(f"\n✓ GPS/Weather 조회 성공!")
            print(f"  📍 위치: {ctx.get('city')}, {ctx.get('country')}")
            print(f"  🌡️  기온: {ctx.get('temperature')}°C")
            print(f"  💧 습도: {ctx.get('humidity')}%")
            print(f"  🌤️  날씨: {ctx.get('weather_label')}")
            print(f"  💨 풍속: {ctx.get('wind_speed')} km/h")
            print(f"  ⚡ AQI: {ctx.get('aqi_label')} ({ctx.get('us_aqi')})")
            print(f"  🌫️  PM2.5: {ctx.get('pm2_5')} µg/m³")
            print(f"  🌫️  PM10: {ctx.get('pm10')} µg/m³")
        else:
            print("⚠ GPS/Weather 조회 실패 — 기본 프롬프트로 진행")
        
        # Stage 2: Gemini
        print("\n" + "-" * 80)
        print("[2️⃣ Stage 2] Gemini 건강 멘트 생성")
        print("-" * 80)
        comment = flow._stage2_get_comment(ctx)
        
        if comment:
            print(f"✓ 멘트 생성 완료!")
            print(f"  💬 {comment}")
        else:
            print("❌ 멘트 생성 실패!")
            return 1
        
        # Stage 3: ElevenLabs TTS
        print("\n" + "-" * 80)
        print("[3️⃣ Stage 3] ElevenLabs WAV 생성 및 저장")
        print("-" * 80)
        wav_path = flow._stage3_speak(comment)
        
        if wav_path and wav_path.exists():
            file_size = wav_path.stat().st_size
            print(f"\n✓ WAV 생성 및 저장 완료!")
            print(f"  📁 파일: {wav_path.name}")
            print(f"  📊 크기: {file_size:,} bytes")
            print(f"  📍 경로: {wav_path.absolute()}")
            print(f"  ✅ 파일 존재: Yes")
        else:
            print("❌ WAV 생성 실패!")
            return 1
        
        # 성공 결과
        print("\n" + "=" * 80)
        print("✅ 파이프라인 테스트 완료!")
        print("=" * 80)
        print(f"생성된 WAV 파일: {wav_path}")
        print("=" * 80 + "\n")
        
        return 0

    except ValueError as e:
        print(f"\n❌ 값 오류: {e}")
        print("\n필수 설정:")
        print("  1. .env 파일 생성 (src/.env)")
        print("  2. API 키 입력:")
        print("     GEMINI_API_KEY=your_key")
        print("     ELEVENLABS_API_KEY=your_key")
        return 1
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
