"""
connection/gemini/gemini_comment.py
-----------------------------------------
Google Gemini API를 사용하여 재채기 감지 후 건강 멘트를 생성합니다.

한 번의 API 호출로 여러 개의 메시지를 배치 생성하고, 환경 데이터
(위치/날씨/대기질)를 반영한 맞춤형 멘트를 제공합니다.

Usage
-----
gen = GeminiCommentGenerator(api_key="YOUR_KEY")
msgs = gen.generate_batch(num_messages=30)         # 30개 메시지 리스트
msgs = gen.generate_batch(num_messages=30, "ko")   # 한국어
comment = gen.generate()                           # 단일 메시지 (내부적으로 배치 1개)

Environment
-----------
GEMINI_API_KEY 환경 변수로 API 키를 설정할 수 있습니다.
"""

import os
from typing import Optional

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# 기본 배치 프롬프트 (위치/날씨 컨텍스트 없음)
# ---------------------------------------------------------------------------
_BATCH_PROMPTS = {
    "en": (
        "Role: You are a warm, witty, and caring health companion.\n\n"
        "Task: Generate a list of {num_messages} distinct, short, comforting messages "
        "for someone who has just sneezed. Each message should be 1-2 sentences long.\n\n"
        "Instructions:\n"
        "1. Tone: Friendly, casual, and supportive — like a kind senior or a close friend.\n"
        "2. Include a simple, warm piece of health advice (e.g., drinking warm water, staying warm, resting).\n"
        "3. Length: Keep each message concise (1-2 sentences).\n"
        "4. Output Format: List each message on a new line, prefixed with a hyphen and a space "
        "(e.g., - Message 1). Ensure there are exactly {num_messages} messages.\n\n"
        "Output Examples:\n"
        "- Stay hydrated with some warm tea — it will help soothe your throat!\n"
        "- Take care of yourself and stay warm today!"
    ),
    "ko": (
        "역할: 당신은 따뜻하고 재치 있는 건강 동반자입니다.\n\n"
        "과제: 방금 재채기를 한 사람을 위해 {num_messages}개의 짧고 위로가 되는 "
        "멘트를 생성해 주세요. 각 멘트는 1~2문장으로 작성해 주세요.\n\n"
        "지침:\n"
        "1. 어조: 친근하고 편안하며 따뜻하게 — 친한 선배나 가까운 친구처럼.\n"
        "2. 따뜻한 건강 조언을 포함해 주세요 (예: 따뜻한 물 마시기, 따뜻하게 입기, 휴식 등).\n"
        "3. 길이: 각 멘트는 간결하게 (1~2문장).\n"
        "4. 출력 형식: 각 멘트를 새 줄에 하이픈과 공백으로 시작하여 작성 "
        "(예: - 멘트 1). 정확히 {num_messages}개의 멘트를 작성해 주세요.\n\n"
        "출력 예시:\n"
        "- 따뜻한 차 한 잔 마시면 목이 좀 나아질 거예요!\n"
        "- 오늘 따뜻하게 입고 건강 잘 챙기세요!"
    ),
}

# ---------------------------------------------------------------------------
# 환경 컨텍스트 포함 배치 프롬프트 (GPS + 날씨/대기질)
# ---------------------------------------------------------------------------
_BATCH_CONTEXT_PROMPTS = {
    "en": (
        "Role: You are a warm, witty, and caring health companion.\n\n"
        "Task: Generate a list of {num_messages} distinct, short, comforting messages "
        "for someone in {city}, {country} who has just sneezed, "
        "considering the provided environmental factors. "
        "Each message should be 1-2 sentences long and focus on the most relevant environmental factor.\n\n"
        "Environmental Factors Today:\n"
        "- Current Temperature: {temperature}°C\n"
        "- Humidity: {humidity}%\n"
        "- Weather Condition: {weather_label}\n"
        "- Wind Speed: {wind_speed} km/h\n"
        "- Fine Dust (PM10): {pm10} µg/m³\n"
        "- Ultra-Fine Dust (PM2.5): {pm2_5} µg/m³\n"
        "- Air Quality: {aqi_label}\n\n"
        "Instructions:\n"
        "1. Tone: Friendly, casual, and supportive — like a kind senior or a close friend.\n"
        "2. Structure:\n"
        "    - Based on the 'Environmental Factors Today', identify the most probable cause for a sneeze.\n"
        "    - Mention that potential cause (e.g., high PM2.5, sudden temperature drop, humidity).\n"
        "    - Provide a simple, warm piece of advice directly related to the identified cause.\n"
        "3. Length: Keep each message concise (1-2 sentences).\n"
        "4. Focus on the single most relevant environmental factor per message.\n"
        "5. Output Format: List each message on a new line, prefixed with a hyphen and a space "
        "(e.g., - Message 1). Ensure there are exactly {num_messages} messages.\n\n"
        "Output Examples:\n"
        "- The PM2.5 levels are quite high today — try wearing a mask when going outside.\n"
        "- With the air quality this poor, a warm cup of water will help soothe your throat.\n"
        "- The temperature dropped suddenly, so keep yourself warm and cozy!"
    ),
    "ko": (
        "역할: 당신은 따뜻하고 재치 있는 건강 동반자입니다.\n\n"
        "과제: {country} {city}에서 방금 재채기를 한 사람을 위해 "
        "{num_messages}개의 짧고 위로가 되는 멘트를 생성해 주세요. "
        "제공된 환경 요인을 고려하여, 가장 관련성 높은 요인에 집중해 주세요. "
        "각 멘트는 1~2문장으로 작성해 주세요.\n\n"
        "오늘의 환경 요인:\n"
        "- 현재 기온: {temperature}°C\n"
        "- 습도: {humidity}%\n"
        "- 날씨 상태: {weather_label}\n"
        "- 풍속: {wind_speed} km/h\n"
        "- 미세먼지 (PM10): {pm10} µg/m³\n"
        "- 초미세먼지 (PM2.5): {pm2_5} µg/m³\n"
        "- 대기질: {aqi_label}\n\n"
        "지침:\n"
        "1. 어조: 친근하고 편안하며 따뜻하게 — 친한 선배나 가까운 친구처럼.\n"
        "2. 구성:\n"
        "    - '오늘의 환경 요인'을 바탕으로 재채기의 가장 가능성 높은 원인을 파악하세요.\n"
        "    - 해당 원인을 언급하세요 (예: 높은 PM2.5, 갑작스러운 기온 변화, 습도 등).\n"
        "    - 파악된 원인과 직접 관련된 따뜻한 건강 조언을 제공하세요.\n"
        "3. 길이: 각 멘트는 간결하게 (1~2문장).\n"
        "4. 멘트마다 가장 관련성 높은 환경 요인 하나에 집중하세요.\n"
        "5. 출력 형식: 각 멘트를 새 줄에 하이픈과 공백으로 시작하여 작성 "
        "(예: - 멘트 1). 정확히 {num_messages}개의 멘트를 작성해 주세요.\n\n"
        "출력 예시:\n"
        "- 오늘 초미세먼지가 매우 나쁘네요 — 외출 시 마스크를 꼭 착용하세요.\n"
        "- 대기질이 좋지 않으니 따뜻한 물 한 잔으로 목을 달래보세요.\n"
        "- 기온이 갑자기 내려갔으니 따뜻하게 입고 건강 챙기세요!"
    ),
}

_DEFAULT_FALLBACKS = {
    "en": "Take care of yourself and stay warm!",
    "ko": "따뜻하게 입고 건강 챙기세요!",
}


class GeminiCommentGenerator:
    """Gemini API를 사용해 재채기 후 건강 멘트를 배치 생성하는 클래스.

    Parameters
    ----------
    api_key          : Gemini API 키. None이면 환경 변수 GEMINI_API_KEY 사용.
    model_name       : 사용할 Gemini 모델 ID.
    temperature      : 생성 창의성 (0.0 결정론적 ~ 1.0 창의적). 기본값 0.9.
    max_output_tokens: 최대 출력 토큰 수. 배치 생성 시 충분히 크게 설정.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.9,
        max_output_tokens: int = 2048,
    ):
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "google-genai 패키지가 설치되지 않았습니다. "
                "pip install google-genai 를 실행하세요."
            )

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "Gemini API 키가 없습니다. "
                "GeminiCommentGenerator(api_key=...) 또는 "
                "GEMINI_API_KEY 환경 변수를 설정하세요."
            )

        self._client = genai.Client(api_key=resolved_key)
        self._model_name = model_name
        self._gen_config = genai_types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def generate_batch(
        self,
        num_messages: int = 30,
        language: str = "en",
        context: Optional[dict] = None,
    ) -> list[str]:
        """재채기 후 건강 멘트를 여러 개 배치 생성합니다.

        Parameters
        ----------
        num_messages : 생성할 메시지 개수.
        language     : 출력 언어 코드. "en" (영어) 또는 "ko" (한국어).
        context      : GPS/날씨/대기질 정보 딕셔너리 (선택). None이면 기본 프롬프트 사용.
                       필요 키: city, country, temperature, humidity, weather_label,
                                wind_speed, aqi_label, pm2_5, pm10

        Returns
        -------
        list[str]
            생성된 건강 멘트 텍스트 리스트.
            API 호출 실패 시 빈 리스트를 반환합니다.
        """
        if context:
            try:
                template = _BATCH_CONTEXT_PROMPTS.get(language, _BATCH_CONTEXT_PROMPTS["en"])
                prompt = template.format(num_messages=num_messages, **context)
            except KeyError as e:
                print(f"[GeminiComment] ⚠ 컨텍스트 키 누락({e}) — 기본 프롬프트 사용")
                template = _BATCH_PROMPTS.get(language, _BATCH_PROMPTS["en"])
                prompt = template.format(num_messages=num_messages)
        else:
            template = _BATCH_PROMPTS.get(language, _BATCH_PROMPTS["en"])
            prompt = template.format(num_messages=num_messages)

        try:
            print(f"[GeminiComment] Gemini API 호출 중 ({num_messages}개 메시지 생성)...")
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._gen_config,
            )
            raw_text = response.text.strip()
            messages = [
                line.strip()[2:]
                for line in raw_text.split("\n")
                if line.strip().startswith("- ")
            ]
            print(f"[GeminiComment] ✓ {len(messages)}개 메시지 생성 완료")
            return messages
        except Exception as e:
            print(f"[GeminiComment] ❌ 배치 생성 오류: {e}")
            return []

    def generate(self, language: str = "en", context: Optional[dict] = None) -> str:
        """재채기 후 건강 멘트를 단일 메시지로 생성합니다.

        Parameters
        ----------
        language : 출력 언어 코드. "en" (영어) 또는 "ko" (한국어).
        context  : GPS/날씨/대기질 정보 딕셔너리 (선택).

        Returns
        -------
        str
            생성된 건강 멘트 텍스트.
            API 호출 실패 시 기본 fallback 문구를 반환합니다.
        """
        messages = self.generate_batch(num_messages=1, language=language, context=context)
        if messages:
            return messages[0]
        return _DEFAULT_FALLBACKS.get(language, _DEFAULT_FALLBACKS["en"])


if __name__ == "__main__":
    # 단독 실행 테스트
    import sys

    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"[GeminiComment] 테스트 (language={lang}, num_messages={num})")

    try:
        gen = GeminiCommentGenerator()
        messages = gen.generate_batch(num_messages=num, language=lang)
        print(f"\n생성된 메시지 ({len(messages)}개):")
        for i, msg in enumerate(messages, 1):
            print(f"{i}. {msg}")
    except ValueError as e:
        print(f"[오류] {e}")
        print("환경 변수 GEMINI_API_KEY 를 설정하고 다시 실행하세요.")
