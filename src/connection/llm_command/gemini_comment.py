"""
connection/llm_command/gemini_comment.py
-----------------------------------------
Google Gemini API를 사용하여 재채기 감지 후 건강 멘트를 생성합니다.

Usage
-----
gen = GeminiCommentGenerator(api_key="YOUR_KEY")
comment = gen.generate()          # 영어 멘트
comment = gen.generate("ko")      # 한국어 멘트

Environment
-----------
GEMINI_API_KEY 환경 변수로 API 키를 설정할 수 있습니다.
"""

import os
from typing import Optional

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# 언어별 프롬프트 템플릿
_PROMPTS = {
    "en": (
        "Someone just sneezed. Generate a short, warm, and caring health message "
        "to say immediately after saying 'Bless you!'. "
        "The message should be 1-2 sentences, health-conscious, and friendly. "
        "Do NOT include 'Bless you' in the response. "
        "Respond ONLY with the message text, no extra explanation."
    ),
    "ko": (
        "방금 재채기를 했습니다. '건강하세요!' 뒤에 이어서 말할 짧고 따뜻한 건강 멘트를 생성해 주세요. "
        "1~2문장으로, 건강에 관심을 담아 친근하게 작성해 주세요. "
        "'건강하세요'는 포함하지 마세요. "
        "멘트 텍스트만 응답하고, 다른 설명은 추가하지 마세요."
    ),
}

_DEFAULT_FALLBACKS = {
    "en": "Take care of yourself and stay warm!",
    "ko": "따뜻하게 입고 건강 챙기세요!",
}


class GeminiCommentGenerator:
    """Gemini API를 사용해 재채기 후 건강 멘트를 생성하는 클래스.

    Parameters
    ----------
    api_key    : Gemini API 키. None이면 환경 변수 GEMINI_API_KEY 사용.
    model_name : 사용할 Gemini 모델 ID.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
    ):
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai 패키지가 설치되지 않았습니다. "
                "pip install google-generativeai 를 실행하세요."
            )

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "Gemini API 키가 없습니다. "
                "GeminiCommentGenerator(api_key=...) 또는 "
                "GEMINI_API_KEY 환경 변수를 설정하세요."
            )

        genai.configure(api_key=resolved_key)
        self.model = genai.GenerativeModel(model_name)
        self._model_name = model_name

    def generate(self, language: str = "en") -> str:
        """재채기 후 건강 멘트를 생성합니다.

        Parameters
        ----------
        language : 출력 언어 코드. "en" (영어) 또는 "ko" (한국어).

        Returns
        -------
        str
            생성된 건강 멘트 텍스트.
            API 호출 실패 시 기본 fallback 문구를 반환합니다.
        """
        prompt = _PROMPTS.get(language, _PROMPTS["en"])
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            if text:
                return text
        except Exception as e:
            print(f"[GeminiComment] API 오류: {e}")

        return _DEFAULT_FALLBACKS.get(language, _DEFAULT_FALLBACKS["en"])


if __name__ == "__main__":
    # 단독 실행 테스트
    import sys

    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    print(f"[GeminiComment] 테스트 (language={lang})")

    try:
        gen = GeminiCommentGenerator()
        comment = gen.generate(lang)
        print(f"생성된 멘트: {comment}")
    except ValueError as e:
        print(f"[오류] {e}")
        print("환경 변수 GEMINI_API_KEY 를 설정하고 다시 실행하세요.")
