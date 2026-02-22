"""
connection/weather/weather.py
------------------------------
Fetches weather and air quality data.

- Open-Meteo  : temperature, humidity, wind speed, PM10, PM2.5, US AQI (free, no API key)
- wttr.in     : weather condition description and temperature change vs yesterday (free, no API key)

Usage
-----
fetcher = WeatherFetcher()

# Basic (Open-Meteo only)
data = fetcher.get_context(lat=37.56, lon=126.97)

# With wttr.in (richer weather description + temperature change vs yesterday)
data = fetcher.get_context(lat=37.56, lon=126.97, city="Seoul")
# {
#   "temperature": 5.0, "humidity": 70, "weather_label": "Partly cloudy",
#   "wind_speed": 3.2, "pm2_5": 35.0, "pm10": 45.0,
#   "us_aqi": 35, "aqi_label": "Good",
#   "temp_change_yesterday": "+2.5°C"   # included when city is provided
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
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from ml_model import config as cfg


# WMO weather code → description mapping (major codes included)
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

# US AQI range → grade label
_AQI_LABELS: list[tuple[float, str]] = [
    (50,   "Good"),
    (100,  "Moderate"),
    (150,  "Unhealthy for Sensitive Groups"),
    (200,  "Unhealthy"),
    (300,  "Very Unhealthy"),
    (float("inf"), "Hazardous"),
]


class WeatherFetcher:
    """Fetches weather and air quality information from the Open-Meteo API.

    Parameters
    ----------
    timeout : HTTP request timeout (seconds). Defaults to config.CONTEXT_FETCH_TIMEOUT.
    """

    def __init__(self, timeout: int = cfg.CONTEXT_FETCH_TIMEOUT):
        self.timeout = timeout
        if not _REQUESTS_AVAILABLE:
            print("[WeatherFetcher] ⚠ requests package not found. Run: pip install requests")

    def get_context(
        self,
        lat: float,
        lon: float,
        city: Optional[str] = None,
    ) -> Optional[dict]:
        """Fetch weather + air quality data by latitude/longitude.

        Parameters
        ----------
        lat  : Latitude.
        lon  : Longitude.
        city : City name (optional). When provided, augments the result with a
               weather condition description and temperature change vs yesterday from wttr.in.

        Returns
        -------
        dict | None
            On success: {
                "temperature": float,           # Temperature in Celsius
                "humidity": int,                # Relative humidity (%)
                "weather_label": str,           # Weather condition description
                "wind_speed": float,            # Wind speed (km/h)
                "pm2_5": float,                 # PM2.5 (µg/m³)
                "pm10": float,                  # PM10 (µg/m³)
                "us_aqi": int,                  # US AQI index
                "aqi_label": str,               # AQI grade label
                "temp_change_yesterday": str,   # Temp change vs yesterday (when city provided)
            }
            On failure: None
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

        # Augment with wttr.in: richer weather description + temperature change vs yesterday
        if city:
            wttr = self._fetch_wttr(city)
            if wttr:
                result["weather_label"] = wttr["condition"]
                result["temp_change_yesterday"] = wttr["temp_change"]

        return result

    def _fetch_wttr(self, city: str) -> Optional[dict]:
        """Fetch weather condition description and temperature change vs yesterday from wttr.in.

        Parameters
        ----------
        city : City name (e.g., "Seoul", "Boston").

        Returns
        -------
        dict | None
            {"condition": str, "temp_change": str} or None.
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
            print(f"[WeatherFetcher] ⚠ wttr.in error: {e}")
            return None

    def _fetch_weather(self, lat: float, lon: float) -> Optional[dict]:
        """Call the Open-Meteo weather API."""
        url = cfg.WEATHER_API_URL.format(lat=lat, lon=lon)
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("current", {})
        except Exception as e:
            print(f"[WeatherFetcher] ❌ Weather API error: {e}")
            return None

    def _fetch_air_quality(self, lat: float, lon: float) -> Optional[dict]:
        """Call the Open-Meteo air quality API."""
        url = cfg.AIR_QUALITY_API_URL.format(lat=lat, lon=lon)
        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
            return response.json().get("current", {})
        except Exception as e:
            print(f"[WeatherFetcher] ❌ Air quality API error: {e}")
            return None

    def _aqi_label(self, aqi: int) -> str:
        """Convert a US AQI value to a grade string."""
        for threshold, label in _AQI_LABELS:
            if aqi <= threshold:
                return label
        return "Hazardous"

    def _weather_label(self, wmo_code: int) -> str:
        """Convert a WMO weather code to a human-readable string."""
        return _WMO_CODE_LABELS.get(wmo_code, "Unknown conditions")


if __name__ == "__main__":
    # Self-test with Seoul coordinates
    fetcher = WeatherFetcher()
    data = fetcher.get_context(lat=37.56, lon=126.97, city="Seoul")
    if data:
        print(
            f"[WeatherFetcher] ✓ "
            f"{data['temperature']}°C, "
            f"humidity {data['humidity']}%, "
            f"{data['weather_label']}, "
            f"AQI {data['us_aqi']} ({data['aqi_label']}), "
            f"PM2.5 {data['pm2_5']} µg/m³"
        )
        if "temp_change_yesterday" in data:
            print(f"[WeatherFetcher] Temperature change vs yesterday: {data['temp_change_yesterday']}")
    else:
        print("[WeatherFetcher] ❌ Data fetch failed")
