"""
connection/gemini/gemini_comment.py
-----------------------------------------
Generates health comments after sneeze detection using the Google Gemini API.

Batch-generates multiple messages in a single API call and supports optional
environmental data (location/weather/air quality) for contextual comments.

Usage
-----
gen = GeminiCommentGenerator(api_key="YOUR_KEY")
msgs = gen.generate_batch(num_messages=30)         # list of 30 messages
msgs = gen.generate_batch(num_messages=30, "ko")   # Korean
comment = gen.generate()                           # single message (internally batch of 1)

Environment
-----------
API key can be set via the GEMINI_API_KEY environment variable.
"""

import json
import os
import re
from typing import Optional

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default batch prompt (no location/weather context)
# ---------------------------------------------------------------------------
_BATCH_PROMPTS = {
    "en": (
        "Role: You are a warm, witty, and caring health companion.\n\n"
        "Task: Generate exactly {num_messages} distinct, short, comforting messages "
        "for someone who has just sneezed. Each message should be 1-2 sentences long.\n\n"
        "Instructions:\n"
        "1. Tone: Friendly, casual, and supportive — like a kind senior or a close friend.\n"
        "2. Include a simple, warm piece of health advice (e.g., drinking warm water, staying warm, resting).\n"
        "3. Length: Keep each message concise (1-2 sentences).\n"
        "4. Output Format: Respond with ONLY a JSON array of exactly {num_messages} strings. "
        "No explanation, no markdown, no extra text — just the raw JSON array.\n\n"
        'Example output for 3 messages:\n'
        '["Stay hydrated with some warm tea!", '
        '"Take care of yourself and stay warm today.", '
        '"Drink a glass of warm water to soothe your throat!"]'
    ),
    "ko": (
        "역할: 당신은 따뜻하고 재치 있는 건강 동반자입니다.\n\n"
        "과제: 방금 재채기를 한 사람을 위해 정확히 {num_messages}개의 짧고 위로가 되는 "
        "멘트를 생성해 주세요. 각 멘트는 1~2문장으로 작성해 주세요.\n\n"
        "지침:\n"
        "1. 어조: 친근하고 편안하며 따뜻하게 — 친한 선배나 가까운 친구처럼.\n"
        "2. 따뜻한 건강 조언을 포함해 주세요 (예: 따뜻한 물 마시기, 따뜻하게 입기, 휴식 등).\n"
        "3. 길이: 각 멘트는 간결하게 (1~2문장).\n"
        "4. 출력 형식: 정확히 {num_messages}개의 문자열을 담은 JSON 배열만 응답하세요. "
        "설명, 마크다운, 추가 텍스트 없이 순수 JSON 배열만 출력하세요.\n\n"
        "3개 예시 출력:\n"
        '["따뜻한 차 한 잔 마시면 목이 좀 나아질 거예요!", '
        '"오늘 따뜻하게 입고 건강 잘 챙기세요.", '
        '"따뜻한 물 한 잔으로 목을 달래보세요!"]'
    ),
}

# ---------------------------------------------------------------------------
# Context-enriched batch prompt (GPS + weather/air quality)
# ---------------------------------------------------------------------------
_BATCH_CONTEXT_PROMPTS = {
    "en": (
        "Role: You are a warm, witty, and caring health companion.\n\n"
        "Task: Generate exactly {num_messages} distinct, short, comforting messages "
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
        "5. Output Format: Respond with ONLY a JSON array of exactly {num_messages} strings. "
        "No explanation, no markdown, no extra text — just the raw JSON array.\n\n"
        'Example output for 3 messages:\n'
        '["The PM2.5 levels are quite high today — try wearing a mask when going outside.", '
        '"With the air quality this poor, a warm cup of water will help soothe your throat.", '
        '"The temperature dropped suddenly, so keep yourself warm and cozy!"]'
    ),
    "ko": (
        "역할: 당신은 따뜻하고 재치 있는 건강 동반자입니다.\n\n"
        "과제: {country} {city}에서 방금 재채기를 한 사람을 위해 "
        "정확히 {num_messages}개의 짧고 위로가 되는 멘트를 생성해 주세요. "
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
        "5. 출력 형식: 정확히 {num_messages}개의 문자열을 담은 JSON 배열만 응답하세요. "
        "설명, 마크다운, 추가 텍스트 없이 순수 JSON 배열만 출력하세요.\n\n"
        "3개 예시 출력:\n"
        '["오늘 초미세먼지가 매우 나쁘네요 — 외출 시 마스크를 꼭 착용하세요.", '
        '"대기질이 좋지 않으니 따뜻한 물 한 잔으로 목을 달래보세요.", '
        '"기온이 갑자기 내려갔으니 따뜻하게 입고 건강 챙기세요!"]'
    ),
}

_DEFAULT_FALLBACKS = {
    "en": "Take care of yourself and stay warm!",
    "ko": "따뜻하게 입고 건강 챙기세요!",
}


class GeminiCommentGenerator:
    """Batch-generates post-sneeze health comments using the Gemini API.

    Parameters
    ----------
    api_key          : Gemini API key. Uses GEMINI_API_KEY env var if None.
    model_name       : Gemini model ID to use.
    temperature      : Generation creativity (0.0 deterministic ~ 1.0 creative). Default 0.9.
    max_output_tokens: Max output tokens. Set large enough for batch generation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.9,
        max_output_tokens: int = 8192,
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
            response_mime_type="application/json",
        )

    def generate_batch(
        self,
        num_messages: int = 1,
        language: str = "en",
        context: Optional[dict] = None,
    ) -> list[str]:
        """Batch-generate post-sneeze health comments.

        Parameters
        ----------
        num_messages : Number of messages to generate.
        language     : Output language code. "en" (English) or "ko" (Korean).
        context      : GPS/weather/air quality dict (optional). Uses default prompt if None.
                       Required keys: city, country, temperature, humidity, weather_label,
                                      wind_speed, aqi_label, pm2_5, pm10

        Returns
        -------
        list[str]
            List of generated health comment strings.
            Returns an empty list on API call failure.
        """
        if context:
            try:
                template = _BATCH_CONTEXT_PROMPTS.get(language, _BATCH_CONTEXT_PROMPTS["en"])
                prompt = template.format(num_messages=num_messages, **context)
            except KeyError as e:
                print(f"[GeminiComment] ⚠ Missing context key ({e}) — using default prompt")
                template = _BATCH_PROMPTS.get(language, _BATCH_PROMPTS["en"])
                prompt = template.format(num_messages=num_messages)
        else:
            template = _BATCH_PROMPTS.get(language, _BATCH_PROMPTS["en"])
            prompt = template.format(num_messages=num_messages)

        try:
            print(f"[GeminiComment] Calling Gemini API (generating {num_messages} messages)...")
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._gen_config,
            )
            raw_text = response.text.strip()
            messages = self._parse_messages(raw_text, num_messages)
            print(f"[GeminiComment] ✓ {len(messages)} messages generated")
            return messages
        except Exception as e:
            print(f"[GeminiComment] ❌ Batch generation error: {e}")
            return []

    def _parse_messages(self, raw_text: str, expected_count: int) -> list[str]:
        """Parse messages from a Gemini JSON response.

        Primary path : JSON array of strings (matches response_mime_type="application/json").
        Fallback path: regex-based parsing for markdown/list-formatted responses.

        Parameters
        ----------
        raw_text : Raw Gemini API response text.
        expected_count : Expected number of messages (used for validation).

        Returns
        -------
        list[str]
            List of parsed message strings.
        """
        # --- Primary: JSON array parsing ---
        try:
            # Strip markdown code fences if the model wraps the JSON
            clean = re.sub(r"```(?:json)?\s*|\s*```", "", raw_text).strip()
            data = json.loads(clean)

            # Unwrap dict: some models return {"messages": [...]} instead of [...]
            if isinstance(data, dict):
                for key in ("messages", "items", "responses", "comments", "list"):
                    if key in data and isinstance(data[key], list):
                        data = data[key]
                        break
                else:
                    # Last resort: use first list value in the dict
                    for v in data.values():
                        if isinstance(v, list):
                            data = v
                            break

            if isinstance(data, list):
                messages = [str(m).strip() for m in data if str(m).strip()]
                if messages:
                    if len(messages) > expected_count:
                        messages = messages[:expected_count]
                    return messages
        except json.JSONDecodeError as e:
            print(f"[GeminiComment] ⚠ JSON decode error at pos {e.pos}: {e.msg} — raw response: {raw_text[:300]!r}")
        except ValueError:
            pass

        # --- Fallback: regex-based parsing ---
        print("[GeminiComment] ⚠ Falling back to regex parser")
        raw_text = re.sub(r"```[\w]*\n?", "", raw_text)

        _SKIP_PATTERN = re.compile(
            r"^(here are|below are|the following|these are|sure[,!]|of course|certainly|"
            r"i[''`]?ve generated|i[''`]?ve created|as requested|check out|enjoy)",
            re.IGNORECASE,
        )

        messages = []
        for line in raw_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            line_clean = re.sub(r"^(\*{1,2}|_{1,2})", "", line).strip()
            line_clean = re.sub(r"(\*{1,2}|_{1,2})$", "", line_clean).strip()

            if _SKIP_PATTERN.match(line_clean):
                continue

            m = re.match(r"^\d+[.)]\s+(.*)", line_clean)
            if m:
                msg = re.sub(r"^(\*{1,2}|_{1,2})|(\*{1,2}|_{1,2})$", "", m.group(1)).strip()
                if msg:
                    messages.append(msg)
                continue

            m = re.match(r"^[-*•]\s+(.*)", line_clean)
            if m:
                msg = re.sub(r"^(\*{1,2}|_{1,2})|(\*{1,2}|_{1,2})$", "", m.group(1)).strip()
                if msg:
                    messages.append(msg)

        if len(messages) > expected_count:
            messages = messages[:expected_count]

        return messages

    def generate(self, language: str = "en", context: Optional[dict] = None) -> str:
        """Generate a single post-sneeze health comment.

        Parameters
        ----------
        language : Output language code. "en" (English) or "ko" (Korean).
        context  : GPS/weather/air quality dict (optional).

        Returns
        -------
        str
            Generated health comment text.
            Returns a hardcoded fallback string on API call failure.
        """
        messages = self.generate_batch(num_messages=1, language=language, context=context)
        if messages:
            return messages[0]
        return _DEFAULT_FALLBACKS.get(language, _DEFAULT_FALLBACKS["en"])


if __name__ == "__main__":
    # Standalone test
    import sys

    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"[GeminiComment] Test run (language={lang}, num_messages={num})")

    try:
        gen = GeminiCommentGenerator()
        messages = gen.generate_batch(num_messages=num, language=lang)
        print(f"\nGenerated messages ({len(messages)}):")
        for i, msg in enumerate(messages, 1):
            print(f"{i}. {msg}")
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("Set GEMINI_API_KEY environment variable and try again.")
