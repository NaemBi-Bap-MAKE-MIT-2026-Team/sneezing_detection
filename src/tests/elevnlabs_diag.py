"""
elevenlabs_diagnostic.py
------------------------
ElevenLabs API 연결 상태 및 인증 문제를 진단하는 유틸리티.

Usage:
    python -m utils.elevenlabs_diagnostic
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print(f"❌ .env 파일을 찾을 수 없습니다: {env_path}")
        return False
    load_dotenv(env_path)
    return True


def check_api_key():
    """Check if API key exists and is in correct format."""
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    
    print("\n[1] API 키 존재 여부 확인")
    print("─" * 50)
    
    if not api_key:
        print("❌ ELEVENLABS_API_KEY가 설정되지 않았습니다.")
        return False
    
    if not api_key.startswith("sk_"):
        print(f"⚠️  API 키 형식이 올바르지 않습니다.")
        print(f"   예상: sk_xxxxxxxx...xxxxxxxx")
        print(f"   현재: {api_key[:20]}...")
        return False
    
    print(f"✓ API 키 형식 정상: {api_key[:15]}...{api_key[-5:]}")
    return True


def test_api_connection(api_key):
    """Test basic API connection."""
    print("\n[2] API 연결 테스트")
    print("─" * 50)
    
    headers = {"xi-api-key": api_key}
    
    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers=headers,
            timeout=10,
        )
        print(f"HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✓ API 연결 성공")
            voices = response.json().get("voices", [])
            print(f"   사용 가능한 음성: {len(voices)}개")
            return True
        elif response.status_code == 401:
            print("❌ 401 Unauthorized — API 키가 유효하지 않습니다.")
            print(f"   응답: {response.text[:200]}")
            return False
        elif response.status_code == 403:
            print("❌ 403 Forbidden — API 접근 권한이 없습니다.")
            print(f"   응답: {response.text[:200]}")
            return False
        else:
            print(f"⚠️  예상치 못한 상태: {response.status_code}")
            print(f"   응답: {response.text[:200]}")
            return False
    
    except requests.exceptions.Timeout:
        print("❌ 타임아웃: ElevenLabs 서버에 연결할 수 없습니다.")
        return False
    except requests.exceptions.ConnectionError:
        print("❌ 네트워크 오류: 인터넷 연결을 확인하세요.")
        return False
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        return False


def test_text_to_speech(api_key):
    """Test text-to-speech API call."""
    print("\n[3] Text-to-Speech API 테스트")
    print("─" * 50)
    
    # 기본 음성 ID (Bella)
    voice_id = "EXAVITQu4vr4xnSDxMaL"
    test_text = "Hello, this is a test message."
    
    headers = {"xi-api-key": api_key}
    data = {
        "text": test_text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
        }
    }
    
    try:
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            json=data,
            headers=headers,
            timeout=30,
        )
        print(f"HTTP Status: {response.status_code}")
        
        if response.status_code == 200:
            audio_size = len(response.content)
            print(f"✓ TTS 성공 — 오디오 크기: {audio_size} bytes")
            return True
        elif response.status_code == 401:
            print("❌ 401 Unauthorized — API 키가 유효하지 않습니다.")
            return False
        elif response.status_code == 402:
            print("❌ 402 Payment Required — 사용량 초과 또는 요금 미지불")
            return False
        else:
            print(f"⚠️  상태 코드: {response.status_code}")
            print(f"   응답: {response.text[:300]}")
            return False
    
    except requests.exceptions.Timeout:
        print("❌ 타임아웃: TTS 서버가 응답하지 않습니다.")
        return False
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        return False


def check_usage(api_key):
    """Check API usage and quota."""
    print("\n[4] API 사용량 확인")
    print("─" * 50)
    
    headers = {"xi-api-key": api_key}
    
    try:
        response = requests.get(
            "https://api.elevenlabs.io/v1/user/subscription",
            headers=headers,
            timeout=10,
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 구독 정보 조회 성공")
            print(f"   계획: {data.get('tier', 'Unknown')}")
            print(f"   문자 한도: {data.get('character_limit', 'N/A')}")
            print(f"   사용함: {data.get('character_count', 'N/A')}")
            print(f"   남은량: {data.get('character_limit', 0) - data.get('character_count', 0)}")
            return True
        else:
            print(f"⚠️  구독 정보를 조회할 수 없습니다. (Status: {response.status_code})")
            return False
    
    except Exception as e:
        print(f"❌ 오류: {str(e)}")
        return False


def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 50)
    print("🔍 ElevenLabs API 진단 도구")
    print("=" * 50)
    
    # Load .env
    if not load_env():
        sys.exit(1)
    
    api_key = os.getenv("ELEVENLABS_API_KEY", "").strip()
    
    # Run tests
    results = {
        "API 키 확인": check_api_key(),
        "API 연결": test_api_connection(api_key),
        "Text-to-Speech": test_text_to_speech(api_key),
        "사용량 확인": check_usage(api_key),
    }
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 진단 결과 요약")
    print("=" * 50)
    for test_name, result in results.items():
        status = "✓" if result else "❌"
        print(f"{status} {test_name}")
    
    print("\n💡 해결책:")
    if not results["API 키 확인"]:
        print("   1. .env 파일에서 ELEVENLABS_API_KEY를 확인하세요.")
        print("   2. https://elevenlabs.io/app/account/account 에서 새 키를 생성하세요.")
    
    if not results["API 연결"]:
        print("   1. API 키가 유효한지 다시 확인하세요.")
        print("   2. 인터넷 연결 상태를 확인하세요.")
    
    if results["API 연결"] and not results["Text-to-Speech"]:
        print("   1. 사용량이 초과되었을 수 있습니다.")
        print("   2. ElevenLabs 대시보드에서 구독을 확인하세요.")
    
    print()
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)