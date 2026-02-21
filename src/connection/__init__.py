"""
connection
----------
External API connection layer.

  gemini     — Gemini API for health message generation
  elven_labs — ElevenLabs API for TTS playback
  gps        — IP-based geolocation
  weather    — Open-Meteo weather & air quality
"""

from .bless_you_flow import BlessYouFlow

__all__ = ["BlessYouFlow"]
