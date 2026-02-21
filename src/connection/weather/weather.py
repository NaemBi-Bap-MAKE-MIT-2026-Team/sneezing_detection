"""
connection/weather/weather.py
------------------------------
날씨 및 대기질 데이터를 조회합니다.

- Open-Meteo  : 기온, 습도, 풍속, PM10, PM2.5, US AQI (무료, API 키 불필요)
- wttr.in     : 날씨 상태 설명 및 어제 대비 기온 변화 (무료, API 키 불필요)

Usage
-----
fetcher = WeatherFetcher()

# 기본 (Open-Meteo만)
data = fetcher.get_context(lat=37.56, lon=126.97)

# wttr.in 포함 (더 풍부한 날씨 설명 + 어제 대비 기온 변화)
data = fetcher.get_context(lat=37.56, lon=126.97, city="Seoul")
# {
#   "temperature": 5.0, "humidity": 70, "weather_label": "Partly cloudy",
#   "wind_speed": 3.2, "pm2_5": 35.0, "pm10": 45.0,
#   "us_aqi": 35, "aqi_label": "Good",
#   "temp_change_yesterday": "+2.5°C"   # city 제공 시 추가
# }
"""

import sys
import os
from typing import Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from ml_model import config as cfg
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from ml_model import config as cfg


# WMO 날씨 코드 → 설명 매핑 (주요 코드 포함)
_WMO_CODE_LABELS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Heavy freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers", 82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail", 99: "Heavy thunderstorm with hail",
}

# US AQI 구간 → 등급 레이블
_AQI_LABELS: list[tuple[float, str]] = [
    (50,   "Good"),
    (100,  "Moderate"),
    (150,  "Unhealthy for Sensitive Groups"),
    (200,  "Unhealthy"),
    (300,  "Very Unhealthy"),
    (float("inf"), "Hazardous"),
]


class WeatherFetcher:
    """Open-Meteo API로 날씨 및 대기질 정보를 조회하는 클래스.

    Parameters
    ----------
    timeout : HTTP 요청 타임아웃(초). 기본값은 config.CONTEXT_FETCH_TIMEOUT.
    """

    def __init__(self, timeout: int = cfg.CONTEXT_FETCH_TIMEOUT):
        self.timeout = timeout
        if not _REQUESTS_AVAILABLE:
            print("[WeatherFetcher] ⚠ requests 패키지 없음. pip install requests 를 실행하세요.")

    def get_context(
        self,
        lat: float,
        lon: float,
        city: Optional[str] = None,
    ) -> Optional[dict]:
        """위도/경도로 날씨 + 대기질 정보를 조회합니다.

        Parameters
        ----------
        lat  : 위도
        lon  : 경도
        city : 도시 이름 (선택). 제공 시 wttr.in으로 날씨 상태 설명과
               어제 대비 기온 변화를 추가로 조회합니다.

        Returns
        -------
        dict | None
            성공 시: {
                "temperature": float,           # 섭씨 온도
                "humidity": int,                # 상대습도 (%)
                "weather_label": str,           # 날씨 설명
                "wind_speed": float,            # 풍속 (km/h)
                "pm2_5": float,                 # PM2.5 (µg/m³)
                "pm10": float,                  # PM10 (µg/m³)
                "us_aqi": int,                  # US AQI 지수
                "aqi_label": str,               # AQI 등급 레이블
                "temp_change_yesterday": str,   # 어제 대비 기온 변화 (city 제공 시)
            }
            실패 시: None
        """
        if not _REQUESTS_AVAILABLE:
            return None

        weather = self._fetch_weather(lat, lon)
        air = self._fetch_air_quality(lat, lon)

        if weather is None and air is None:
            return None

        result: dict = {}

        if weather:
            result["temperature"] = weather.get("temperature_2m")
            result["humidity"] = weather.get("relative_humidity_2m")
            result["wind_speed"] = weather.get("wind_speed_10m")
            result["weather_label"] = self._weather_label(weather.get("weather_code", -1))
        else:
            result["temperature"] = None
            result["humidity"] = None
            result["wind_speed"] = None
            result["weather_label"] = "Unknown conditions"

        if air:
            us_aqi = air.get("us_aqi")
            result["pm2_5"] = air.get("pm2_5")
            result["pm10"] = air.get("pm10")
            result["us_aqi"] = us_aqi
            result["aqi_label"] = self._aqi_label(us_aqi) if us_aqi is not None else "Unknown"
        else:
            result["pm2_5"] = None
            result["pm10"] = None
            result["us_aqi"] = None
            result["aqi_label"] = "Unknown"

        # wttr.in 보강: 더 풍부한 날씨 설명 + 어제 대비 기온 변화
        if city:
            wttr = self._fetch_wttr(city)
            if wttr:
                result["weather_label"] = wttr["condition"]
                result["temp_change_yesterday"] = wttr["temp_change"]

        return result

    def _fetch_wttr(self, city: str) -> Optional[dict]:
        """wttr.in으로 날씨 상태 설명과 어제 대비 기온 변화를 조회합니다.

        Parameters
        ----------
        city : 도시 이름 (예: "Seoul", "Boston")

        Returns
        -------
        dict | None
            {"condition": str, "temp_change": str} 또는 None
        """
        url = f"https://wttr.in/{city}?format=j1"
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            condition = data["current_condition"][0]["weatherDesc"][0]["value"]
            today_avg = float(data["weather"][1]["avgtempC"])
            yesterday_avg = float(data["weather"][0]["avgtempC"])
            temp_diff = round(today_avg - yesterday_avg, 1)
            temp_change = f"{'+' if temp_diff > 0 else ''}{temp_diff}°C"

            return {"condition": condition, "temp_change": temp_change}
        except Exception as e:
            print(f"[WeatherFetcher] ⚠ wttr.in 오류: {e}")
            return None

    def _fetch_weather(self, lat: float, lon: float) -> Optional[dict]:
        """Open-Meteo 날씨 API를 호출합니다."""
        url = cfg.WEATHER_API_URL.format(lat=lat, lon=lon)
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("current", {})
        except Exception as e:
            print(f"[WeatherFetcher] ❌ 날씨 API 오류: {e}")
            return None

    def _fetch_air_quality(self, lat: float, lon: float) -> Optional[dict]:
        """Open-Meteo 대기질 API를 호출합니다."""
        url = cfg.AIR_QUALITY_API_URL.format(lat=lat, lon=lon)
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("current", {})
        except Exception as e:
            print(f"[WeatherFetcher] ❌ 대기질 API 오류: {e}")
            return None

    def _aqi_label(self, aqi: int) -> str:
        """US AQI 수치를 등급 문자열로 변환합니다."""
        for threshold, label in _AQI_LABELS:
            if aqi <= threshold:
                return label
        return "Hazardous"

    def _weather_label(self, wmo_code: int) -> str:
        """WMO 날씨 코드를 사람이 읽을 수 있는 문자열로 변환합니다."""
        return _WMO_CODE_LABELS.get(wmo_code, "Unknown conditions")


if __name__ == "__main__":
    # 서울 좌표로 셀프 테스트
    fetcher = WeatherFetcher()
    data = fetcher.get_context(lat=37.56, lon=126.97, city="Seoul")
    if data:
        print(
            f"[WeatherFetcher] ✓ "
            f"{data['temperature']}°C, "
            f"습도 {data['humidity']}%, "
            f"{data['weather_label']}, "
            f"AQI {data['us_aqi']} ({data['aqi_label']}), "
            f"PM2.5 {data['pm2_5']} µg/m³"
        )
        if "temp_change_yesterday" in data:
            print(f"[WeatherFetcher] 어제 대비 기온 변화: {data['temp_change_yesterday']}")
    else:
        print("[WeatherFetcher] ❌ 데이터 조회 실패")
