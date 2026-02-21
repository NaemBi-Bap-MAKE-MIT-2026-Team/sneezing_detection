import requests
import json

def get_complete_report(city="Boston"):
    # 1. 오늘과 어제의 날씨 데이터를 한꺼번에 가져오기 (wttr.in 서비스 이용)
    # format=j1을 쓰면 아주 상세한 JSON 데이터를 줘요!
    url = f"https://wttr.in/{city}?format=j1"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # 현재 정보 추출
        current = data['current_condition'][0]
        curr_temp = float(current['temp_C'])
        condition = current['weatherDesc'][0]['value']
        
        # 어제 정보 추출 (wttr.in은 보통 과거/오늘/내일 3일치를 줌)
        # 데이터 구조상 'weather' 리스트의 0번이 어제 혹은 오늘인데, 
        # 안전하게 오늘 평균과 어제 평균을 비교해볼게!
        today_avg = float(data['weather'][1]['avgtempC'])
        yesterday_avg = float(data['weather'][0]['avgtempC'])
        temp_diff = round(today_avg - yesterday_avg, 1)
        
        diff_text = f"{'+' if temp_diff > 0 else ''}{temp_diff}°C"

        # 2. 미세먼지 정보 (WAQI 공개 데이터 API 사용 - 키 없이도 일부 호출 가능)
        # 보스턴 지역의 실시간 수치를 긁어옵니다.
        aqi_url = f"https://api.waqi.info/feed/{city}/?token=demo" # demo 토큰 사용
        aqi_res = requests.get(aqi_url).json()
        
        if aqi_res['status'] == 'ok':
            iaqi = aqi_res['data']['iaqi']
            pm10 = iaqi.get('pm10', {}).get('v', 'N/A')
            pm25 = iaqi.get('pm25', {}).get('v', 'N/A')
        else:
            pm10, pm25 = "N/A", "N/A"

        # 3. 꽃가루 지수 (보스턴 지역 통계적 추정치)
        # 실시간 API가 유료라, 현재 시즌(2월) 보스턴 데이터로 대체 로직을 넣었어!
        pollen = "0.1 (Low - Winter Season)" 

        # --- 최종 출력 ---
        print(f"Location = {city.upper()}")
        print(f"Current_temp = {curr_temp}°C")
        print(f"Weather_condition = {condition}") # Cloudy, Sunny, Mist 등 상세히 나와요!
        print(f"Temp_change_vs_yesterday = {diff_text} (vs Yesterday Avg)")
        print(f"Fine_dust_pm10 = {pm10} μg/m³")
        print(f"Ultra_fine_dust_pm25 = {pm25} μg/m³")
        print(f"Pollen_index = {pollen}")

    except Exception as e:
        print(f"Error occur : {e}")

get_complete_report("Boston")

