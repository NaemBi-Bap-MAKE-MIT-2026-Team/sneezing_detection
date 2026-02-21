import google.generativeai as genai
import os
import asyncio
# import requests # 실제 외부 API 호출 시 필요

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

def get_environmental_data():
    """
    외부 API에서 날씨 및 미세먼지 데이터를 가져오는 함수 (의사 코드).
    실제 구현 시, 각 서비스의 API 키와 엔드포인트를 사용해야 합니다.
    """
    # --- 실제 외부 API 호출 코드 여기에 작성 ---
    # 예시 데이터 (실제 데이터로 대체되어야 함)
    current_temp = 7
    weather_condition = "흐림"
    temp_change_vs_yesterday = "3도 하강"
    fine_dust_pm10 = "나쁨"
    ultra_fine_dust_pm25 = "매우 나쁨"
    pollen_index = "보통" # 해당 API가 없을 경우 N/A 또는 빈 값

    return {
        "current_temp": current_temp,
        "weather_condition": weather_condition,
        "temp_change_vs_yesterday": temp_change_vs_yesterday,
        "fine_dust_pm10": fine_dust_pm10,
        "ultra_fine_dust_pm25": ultra_fine_dust_pm25,
        "pollen_index": pollen_index
    }

def generate_multiple_comforting_messages(num_messages=30):
    env_data = get_environmental_data()

    # 모델 설정: 'temperature'를 높여 다양한 응답을 유도합니다.
    generation_config = genai.GenerationConfig(
        temperature=0.9,  # 0.0 (결정론적) ~ 1.0 (창의적)
        max_output_tokens=2048 # 더 많은 출력을 위해 충분히 큰 값 설정
    )
    model = genai.GenerativeModel('models/gemini-2.5-flash', generation_config=generation_config)

    # 환경 데이터를 포함하여 프롬프트 동적 생성
    # num_messages 개수만큼 메시지를 생성하고, 각 메시지를 - 로 시작하도록 지시
    prompt = (
        "Role: you are a warm, witty, and caring health companion.\n\n"
        "Task: Generate a list of " + str(num_messages) + " distinct, short, comforting messages for someone who has just sneezed, "
        "considering the provided environmental factors. Each message should be 1-2 sentences long and focus on the most relevant environmental factor.\n\n"
        "Environmental Factors Today:\n"
        f"- Current Temperature: {env_data['current_temp']}°C\n"
        f"- Weather Condition: {env_data['weather_condition']}\n"
        f"- Temperature Change (vs. yesterday): {env_data['temp_change_vs_yesterday']}\n"
        f"- Fine Dust (PM10) Level: {env_data['fine_dust_pm10']}\n"
        f"- Ultra-Fine Dust (PM2.5) Level: {env_data['ultra_fine_dust_pm25']}\n"
        f"- Pollen Index: {env_data['pollen_index']}\n\n"
        "Instructions:\n"
        "1. Tone: Friendly, casual, and supportive-like a kind senior or a close friend.\n"
        "2. Structure:\n"
        "    - Based on the 'Environmental Factors Today', identify the most probable cause for a sneeze.\n"
        "    - Mention that potential cause (e.g., high pollen index, fine dust, sudden temperature drop).\n"
        "    - Provide a simple, warm piece of advice (e.g., drinking warm water, wearing a mask, or resting), directly related to the identified cause.\n"
        "3. Length: Keep each message concise (1-2 sentences).\n"
        "4. Focus on the most relevant environmental factor to provide a targeted and empathetic message.\n"
        "5. Output Format: List each message on a new line, prefixed with a hyphen and a space (e.g., - Message 1). Ensure there are exactly " + str(num_messages) + " messages.\n\n"
        "Output Examples:\n"
        "- The pollen count is quite high today, so try to stay hydrated with some warm tea.\n"
        "- The fine dust seems a bit harsh today. A cup of warm water will definitely help soothe your throat.\n"
        "- The temperature dropped suddenly. Keep yourself warm and cozy!"
    )

    try:
        print(f"Gemini API 호출 중 ({num_messages}개 메시지 생성)...")
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        # 응답 문자열을 파싱하여 리스트로 저장
        messages = [line.strip()[2:] for line in raw_text.split('\n') if line.strip().startswith('- ')]

        print(f"\n생성된 메시지 ({len(messages)}개):")
        for i, msg in enumerate(messages):
            print(f"{i+1}. {msg}")

        return messages

    except Exception as e:
        print(f"메시지 생성 중 오류 발생: {e}")
        return []

if __name__ == "__main__":
    generated_messages = generate_multiple_comforting_messages(num_messages=30)
    print("\n--- 리스트에 저장된 최종 메시지 ---")
    print(generated_messages)
    print(f"총 {len(generated_messages)}개의 메시지가 생성되었습니다.")
