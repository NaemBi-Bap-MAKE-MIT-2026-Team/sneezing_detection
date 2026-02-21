"""
connection/gps/gps.py
---------------------
IP 기반 위치 조회 모듈.

ip-api.com 의 무료 JSON 엔드포인트를 사용합니다 (API 키 불필요).
네트워크 오류 또는 응답 실패 시 None을 반환하여 흐름을 중단하지 않습니다.

Usage
-----
locator = GPSLocator()
location = locator.get_location()   # -> dict | None
# {"city": "Seoul", "country": "South Korea", "region": "Seoul",
#  "lat": 37.56, "lon": 126.97}
"""

import sys
import os
from typing import Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# config 경로 처리 (단독 실행 및 패키지 import 모두 지원)
try:
    from ml_model import config as cfg
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    from ml_model import config as cfg


class GPSLocator:
    """IP geolocation을 사용해 현재 위치 정보를 반환하는 클래스.

    Parameters
    ----------
    timeout : HTTP 요청 타임아웃(초). 기본값은 config.CONTEXT_FETCH_TIMEOUT.
    """

    def __init__(self, timeout: int = cfg.CONTEXT_FETCH_TIMEOUT):
        self.timeout = timeout
        if not _REQUESTS_AVAILABLE:
            print("[GPSLocator] ⚠ requests 패키지 없음. pip install requests 를 실행하세요.")

    def get_location(self) -> Optional[dict]:
        """현재 IP 기반 위치 정보를 반환합니다.

        Returns
        -------
        dict | None
            성공 시: {
                "city": str,
                "country": str,
                "region": str,
                "lat": float,
                "lon": float,
            }
            실패 시: None
        """
        if not _REQUESTS_AVAILABLE:
            return None
        raw = self._fetch()
        if raw is None:
            return None
        return self._parse(raw)

    def _fetch(self) -> Optional[dict]:
        """ip-api.com에 HTTP GET 요청을 보내고 원시 JSON을 반환합니다.

        실패 시 None을 반환합니다. 예외를 외부로 전파하지 않습니다.
        """
        try:
            response = requests.get(cfg.GPS_IP_API_URL, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[GPSLocator] ❌ 네트워크 오류: {e}")
            return None

    def _parse(self, raw: dict) -> Optional[dict]:
        """원시 API 응답에서 필요한 필드만 추출합니다.

        ip-api.com 응답에는 'status' 필드가 있으며,
        'fail'이면 None을 반환합니다.
        """
        try:
            if raw.get("status") != "success":
                print(f"[GPSLocator] ❌ API 응답 실패: {raw.get('message', 'unknown')}")
                return None
            return {
                "city": raw["city"],
                "country": raw["country"],
                "region": raw["regionName"],
                "lat": float(raw["lat"]),
                "lon": float(raw["lon"]),
            }
        except KeyError as e:
            print(f"[GPSLocator] ❌ 응답 필드 누락: {e}")
            return None


if __name__ == "__main__":
    locator = GPSLocator()
    loc = locator.get_location()
    if loc:
        print(f"[GPSLocator] ✓ {loc['city']}, {loc['country']} ({loc['lat']}, {loc['lon']})")
    else:
        print("[GPSLocator] ❌ 위치 조회 실패")
