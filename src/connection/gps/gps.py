"""
connection/gps/gps.py
---------------------
IP-based geolocation module.

Uses the free JSON endpoint of ip-api.com (no API key required).
Returns None on network errors or response failures without interrupting the pipeline.

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

# Resolve config import for both standalone execution and package usage
try:
    from ml_model import config as cfg
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from ml_model import config as cfg


class GPSLocator:
    """Returns current location information using IP geolocation.

    Parameters
    ----------
    timeout : HTTP request timeout (seconds). Defaults to config.CONTEXT_FETCH_TIMEOUT.
    """

    def __init__(self, timeout: int = cfg.CONTEXT_FETCH_TIMEOUT):
        self.timeout = timeout
        if not _REQUESTS_AVAILABLE:
            print("[GPSLocator] ⚠ requests package not found. Run: pip install requests")

    def get_location(self) -> Optional[dict]:
        """Return the current IP-based location.

        Returns
        -------
        dict | None
            On success: {
                "city": str,
                "country": str,
                "region": str,
                "lat": float,
                "lon": float,
            }
            On failure: None
        """
        if not _REQUESTS_AVAILABLE:
            return None
        raw = self._fetch()
        if raw is None:
            return None
        return self._parse(raw)

    def _fetch(self) -> Optional[dict]:
        """Send an HTTP GET request to ip-api.com and return the raw JSON.

        Returns None on failure. Does not propagate exceptions.
        """
        try:
            response = requests.get(cfg.GPS_IP_API_URL, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[GPSLocator] ❌ Network error: {e}")
            return None

    def _parse(self, raw: dict) -> Optional[dict]:
        """Extract required fields from the raw API response.

        The ip-api.com response includes a 'status' field.
        Returns None if status is 'fail'.
        """
        try:
            if raw.get("status") != "success":
                print(f"[GPSLocator] ❌ API response failed: {raw.get('message', 'unknown')}")
                return None
            return {
                "city": raw["city"],
                "country": raw["country"],
                "region": raw["regionName"],
                "lat": float(raw["lat"]),
                "lon": float(raw["lon"]),
            }
        except KeyError as e:
            print(f"[GPSLocator] ❌ Missing response field: {e}")
            return None


if __name__ == "__main__":
    locator = GPSLocator()
    loc = locator.get_location()
    if loc:
        print(f"[GPSLocator] ✓ {loc['city']}, {loc['country']} ({loc['lat']}, {loc['lon']})")
    else:
        print("[GPSLocator] ❌ Location lookup failed")
