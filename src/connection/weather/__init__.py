"""
connection/weather
------------------
Open-Meteo API를 사용한 날씨 및 대기질 조회 (무료, API 키 불필요).
"""

from .weather import WeatherFetcher

__all__ = ["WeatherFetcher"]
