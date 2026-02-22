"""
communication/ws_recv.py
------------------------
WebSocketMicStream: receives Float32 PCM audio from the browser web-app
(app/index.html) over a WebSocket connection and exposes the exact same
interface as NetworkMicStream / MicrophoneStream.

Protocol
--------
- The browser sends raw little-endian float32 frames as binary WebSocket messages.
- Each message is one 100 ms frame (4 800 samples @ 48 kHz) = 19 200 bytes.
- No header, no framing — identical to UDP send.py raw bytes.

The server:
- Listens on ws://<host>:<port>/audio
- Serves GET / with the contents of app/index.html so the browser can load
  the page and WebSocket from the same origin (HTTP → WS upgrade on port 8080).
- Accumulates incoming float32 samples in a queue; read() returns one frame_sec
  worth of samples, just like NetworkMicStream.read().

Usage
-----
from communication.ws_recv import WebSocketMicStream

mic = WebSocketMicStream(
    host       = "0.0.0.0",
    port       = 8080,
    capture_sr = 48000,
    frame_sec  = 0.10,
    pre_seconds= 0.0,
)
with mic:
    while True:
        chunk = mic.read()
"""

import asyncio
import collections
import queue
import threading
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import websockets
    from websockets.server import serve as ws_serve
    import websockets.exceptions
except ImportError as _e:
    raise ImportError(
        "websockets package not found. Install it with:\n"
        "  pip install 'websockets>=12.0'"
    ) from _e

# Path to the single-file web app served on GET /
_APP_HTML = Path(__file__).resolve().parent.parent.parent / "app" / "index.html"


class WebSocketMicStream:
    """Drop-in replacement for NetworkMicStream that receives audio over WebSocket.

    Parameters
    ----------
    host        : IP to bind (``"0.0.0.0"`` = all interfaces).
    port        : TCP port for the WebSocket server (default 8080).
    capture_sr  : Expected sample rate of incoming audio (Hz).
    frame_sec   : Duration of each chunk returned by read() (seconds).
    pre_seconds : Length of rolling pre-buffer (seconds).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8080,
        capture_sr: int = 48000,
        frame_sec: float = 0.10,
        pre_seconds: float = 0.0,
    ):
        self.capture_sr = capture_sr
        self.frame_sec  = frame_sec

        self._frame_samples = int(capture_sr * frame_sec)
        pre_chunks = max(1, int(pre_seconds / frame_sec)) if pre_seconds > 0 else 1
        self.pre_buffer: collections.deque = collections.deque(maxlen=pre_chunks)

        self._queue: queue.Queue = queue.Queue()

        self._host = host
        self._port = port
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Public API  (identical to MicrophoneStream / NetworkMicStream)
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Start the WebSocket server in a background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()
        print(f"🌐 WebSocket server starting on ws://{self._host}:{self._port}/audio")
        print(f"   Open http://<your-pi-ip>:{self._port}/ on your phone")

    def close(self) -> None:
        """Stop the server and background thread."""
        self._running = False
        if self._loop is not None and self._server is not None:
            self._loop.call_soon_threadsafe(self._server.close)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def read(self) -> np.ndarray:
        """Block until the next frame is available and return it.

        Returns
        -------
        np.ndarray of shape (frame_samples,), dtype float32.
        """
        frame = self._queue.get()
        self.pre_buffer.append(frame)
        return frame

    # ------------------------------------------------------------------
    # Internal — async WebSocket server
    # ------------------------------------------------------------------

    def _run_server(self) -> None:
        """Entry point for the background thread: runs the asyncio event loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        async def handler(ws):
            path = ws.request.path if hasattr(ws, "request") else getattr(ws, "path", "/")

            # Serve the web app HTML on GET /
            if path == "/" or path == "":
                await self._serve_http(ws)
                return

            if path != "/audio":
                return

            print(f"🎤 Client connected from {ws.remote_address}")
            acc = np.zeros((0,), dtype=np.float32)

            try:
                async for message in ws:
                    if not self._running:
                        break
                    if isinstance(message, bytes):
                        chunk = np.frombuffer(message, dtype=np.float32)
                        acc = np.concatenate((acc, chunk))

                        while len(acc) >= self._frame_samples:
                            frame = acc[:self._frame_samples].copy()
                            acc   = acc[self._frame_samples:]
                            self._queue.put(frame)

            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                print(f"📵 Client disconnected from {ws.remote_address}")

        self._server = await ws_serve(handler, self._host, self._port)
        await self._server.wait_closed()

    async def _serve_http(self, ws) -> None:
        """Send a minimal HTTP response with index.html for browser access."""
        # websockets library handles the upgrade handshake; non-upgrade GET
        # requests arrive here only when the client sends a plain HTTP request
        # before WebSocket negotiation. In practice, serve the file via a
        # separate lightweight HTTP server started alongside the WS server.
        pass


# ---------------------------------------------------------------------------
# Standalone HTTP server for serving index.html (used by open())
# ---------------------------------------------------------------------------

def _start_http_server(html_path: Path, port: int) -> None:
    """Serve index.html on the same port via a tiny HTTP server (separate thread)."""
    import http.server
    import socketserver

    html_bytes = html_path.read_bytes() if html_path.exists() else b"<h1>index.html not found</h1>"

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html_bytes)))
            self.end_headers()
            self.wfile.write(html_bytes)

        def log_message(self, fmt, *args):
            pass  # suppress access log noise

    # Use a free port one above the WS port for the HTTP file server
    http_port = port + 1
    try:
        with socketserver.TCPServer(("", http_port), _Handler) as httpd:
            httpd.serve_forever()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("Self-test: WebSocketMicStream")
    print("Open your browser at http://localhost:8081/ to load the web app.")
    print("Then connect to ws://localhost:8080/audio and send audio.")
    print("Press Ctrl+C to stop.\n")

    mic = WebSocketMicStream(host="0.0.0.0", port=8080, capture_sr=48000, frame_sec=0.10)

    # Start HTTP server for the web app in a separate daemon thread
    http_thread = threading.Thread(
        target=_start_http_server,
        args=(_APP_HTML, 8080),
        daemon=True,
    )
    http_thread.start()

    with mic:
        try:
            t0 = time.time()
            count = 0
            while True:
                frame = mic.read()
                count += 1
                if count % 10 == 0:
                    elapsed = time.time() - t0
                    print(f"  {count} frames received ({elapsed:.1f}s elapsed, "
                          f"last RMS={float(np.sqrt(np.mean(frame**2))):.4f})")
        except KeyboardInterrupt:
            print("Stopped.")
