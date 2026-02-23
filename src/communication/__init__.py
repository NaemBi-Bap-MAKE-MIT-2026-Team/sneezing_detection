"""
communication
-------------
Unified audio-input layer.

Exports all input sources so callers (main.py, tests) can switch
between local microphone, UDP stream, and WebSocket stream without
changing any downstream code.

  MicrophoneStream    — local sounddevice microphone (from microphone/)
  NetworkMicStream    — audio received over UDP from AudioSender / send.py
  WebSocketMicStream  — audio received over WebSocket from browser (app/index.html)
  AudioSender         — sends MicrophoneStream frames over UDP to NetworkMicStream
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from microphone import MicrophoneStream
from .recv import NetworkMicStream
from .send import AudioSender
from .ws_recv import WebSocketMicStream

__all__ = ["MicrophoneStream", "NetworkMicStream", "WebSocketMicStream", "AudioSender"]
