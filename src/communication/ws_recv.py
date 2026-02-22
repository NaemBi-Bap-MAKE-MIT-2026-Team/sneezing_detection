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
import http as _http_module
import queue
import ssl as _ssl_module
import subprocess
import threading
from pathlib import Path
from typing import Optional

import numpy as np

# Self-signed cert stored next to this file (generated on first --ssl run)
_CERT_PATH = Path(__file__).resolve().parent / "ssl_cert.pem"
_KEY_PATH  = Path(__file__).resolve().parent / "ssl_key.pem"


def _ensure_ssl_cert(cert_path: Path, key_path: Path) -> None:
    """Generate a self-signed certificate/key pair if they don't already exist."""
    if cert_path.exists() and key_path.exists():
        return
    print("[SSL] Generating self-signed certificate (valid 10 years)…")
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509",
                "-newkey", "rsa:2048",
                "-keyout", str(key_path),
                "-out",    str(cert_path),
                "-days",   "3650",
                "-nodes",
                "-subj",   "/CN=sneeze-detector",
            ],
            check=True,
            capture_output=True,
        )
        print(f"[SSL] Certificate written to {cert_path}")
    except FileNotFoundError:
        raise RuntimeError(
            "openssl not found. Install with: brew install openssl (macOS) "
            "or apt-get install openssl (Linux)"
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"openssl failed: {exc.stderr.decode()}")

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
        ssl: bool = False,
    ):
        self.capture_sr = capture_sr
        self.frame_sec  = frame_sec

        self._frame_samples = int(capture_sr * frame_sec)
        pre_chunks = max(1, int(pre_seconds / frame_sec)) if pre_seconds > 0 else 1
        self.pre_buffer: collections.deque = collections.deque(maxlen=pre_chunks)

        self._queue: queue.Queue = queue.Queue()

        self._host = host
        self._port = port
        self._ssl = ssl
        self._ssl_ctx: Optional[_ssl_module.SSLContext] = None
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
        """Start the WebSocket+HTTP server in a background thread."""
        if self._ssl:
            _ensure_ssl_cert(_CERT_PATH, _KEY_PATH)
            ctx = _ssl_module.SSLContext(_ssl_module.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(_CERT_PATH, _KEY_PATH)
            self._ssl_ctx = ctx
            scheme_ws, scheme_http = "wss", "https"
        else:
            self._ssl_ctx = None
            scheme_ws, scheme_http = "ws", "http"

        self._running = True
        self._thread = threading.Thread(target=self._run_server, daemon=True)
        self._thread.start()

        print(f"🌐 WebSocket  : {scheme_ws}://{self._host}:{self._port}/audio")
        print(f"📱 Web app    : {scheme_http}://localhost:{self._port}/  (open in browser)")

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
        # Read the HTML once at server start
        html_bytes = _APP_HTML.read_bytes() if _APP_HTML.exists() else b"<h1>index.html not found</h1>"

        async def _process_request(path, request_headers):
            """Legacy API: (path: str, headers) → None or (status, headers, body).

            Return None to allow WebSocket upgrade; return a tuple to send a
            plain HTTP response instead (used to serve index.html on GET /).
            """
            if path == "/":
                return (
                    _http_module.HTTPStatus.OK,
                    [
                        ("Content-Type", "text/html; charset=utf-8"),
                        ("Content-Length", str(len(html_bytes))),
                    ],
                    html_bytes,
                )
            return None  # allow WebSocket upgrade for any other path

        async def handler(ws):
            # Legacy API exposes the path as ws.path
            path = getattr(ws, "path", "/audio")
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

        self._server = await ws_serve(
            handler,
            self._host,
            self._port,
            ssl=self._ssl_ctx,
            process_request=_process_request,
        )
        await self._server.wait_closed()


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("Self-test: WebSocketMicStream")
    print("Open your browser at http://localhost:8080/ to load the web app.")
    print("Press Ctrl+C to stop.\n")

    mic = WebSocketMicStream(host="0.0.0.0", port=8080, capture_sr=48000, frame_sec=0.10)

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
