"""
communication
-------------
Unified audio-input layer.

Exports both input sources so callers (main.py, tests) can switch
between local microphone and network stream without changing any
downstream code.

  MicrophoneStream   — local sounddevice microphone (from microphone/)
  NetworkMicStream   — audio received over UDP from AudioSender
  AudioSender        — sends MicrophoneStream frames over UDP to NetworkMicStream
"""

from microphone import MicrophoneStream
from .recv import NetworkMicStream
from .send import AudioSender

__all__ = ["MicrophoneStream", "NetworkMicStream", "AudioSender"]
