"""
connection/bless_you_flow.py
-----------------------------------------
Orchestrator for the full response pipeline triggered after a sneeze is detected.

Pipeline
--------
[On start] initialize()  — blocking
  GPS location lookup → Weather/air quality fetch
  → Gemini batch comment generation (N messages)
  → ElevenLabs TTS WAV pre-generation for all N messages
  → WAV queue ready

[On detection] run() / run_async()
  Pop next pre-generated WAV from queue → play immediately (no API call at detection time)
  When queue is exhausted → fall back to static bless_you.wav

Usage
-----
flow = BlessYouFlow(
    bless_wav_path=Path("..."),
    language="en",
    num_messages=10,
)
flow.initialize()   # Blocking call before detection loop (generates all N WAV files)
flow.run_async()    # Non-blocking call on detection (plays next WAV from queue)
"""

import threading
from pathlib import Path
from typing import Optional

from .gemini.gemini_comment import GeminiCommentGenerator
from .eleven_labs.tts_player import ElevenLabsTTSPlayer

try:
    from .gps.gps import GPSLocator
    from .weather.weather import WeatherFetcher
    _CONTEXT_AVAILABLE = True
except ImportError as _ctx_err:
    print(f"[BlessYouFlow] ⚠ GPS/Weather module import failed: {_ctx_err}")
    _CONTEXT_AVAILABLE = False


class BlessYouFlow:
    """Pre-generates N TTS WAV files at startup and plays them sequentially on sneeze detection.

    Parameters
    ----------
    bless_wav_path      : Path to bless_you.wav (static fallback when queue is empty).
    gemini_api_key      : Gemini API key (uses GEMINI_API_KEY env var if None).
    elevenlabs_api_key  : ElevenLabs API key (uses ELEVENLABS_API_KEY env var if None).
    elevenlabs_voice_id : ElevenLabs voice ID.
    language            : Comment language ("en" or "ko").
    enable_context      : Enable GPS/weather context collection (default True).
    num_messages        : Number of messages to pre-generate at startup. Default 10.
    """

    _CTX_TIMEOUT = 8.0   # Max wait time for GPS+Weather (seconds)

    def __init__(
        self,
        bless_wav_path: Path,
        gemini_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        elevenlabs_voice_id: str = "Rachel",
        language: str = "en",
        enable_context: bool = True,
        num_messages: int = 10,
    ):
        self.bless_wav_path = Path(bless_wav_path)
        self.language = language
        self._enable_context = enable_context and _CONTEXT_AVAILABLE
        self._num_messages = num_messages

        self._gemini = GeminiCommentGenerator(api_key=gemini_api_key)

        # Output directory for pre-generated TTS WAV files
        self._tts_output_dir = Path(__file__).resolve().parent.parent / "output_feature" / "sounds"
        self._tts_output_dir.mkdir(parents=True, exist_ok=True)

        self._tts = ElevenLabsTTSPlayer(
            api_key=elevenlabs_api_key,
            output_dir=self._tts_output_dir,
        )

        if self._enable_context:
            self._gps = GPSLocator()
            self._weather = WeatherFetcher()
        else:
            self._gps = None
            self._weather = None

        self._wav_queue: list[Path] = []
        self._queue_lock = threading.Lock()
        self._playing = False
        self._play_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def initialize(self) -> bool:
        """Pre-generate all TTS WAV files before the detection loop starts.

        Steps:
          1. Fetch GPS + weather context (once)
          2. Batch-generate num_messages comments via Gemini (one API call)
          3. Convert each comment to a WAV file via ElevenLabs

        Returns
        -------
        bool
            True if at least one WAV file was successfully generated.
        """
        print(
            f"[BlessYouFlow] 🔄 Initializing... "
            f"(GPS → Weather → Gemini×1 → ElevenLabs×{self._num_messages})"
        )

        # Stage 1: Fetch environmental context (once for the whole batch)
        ctx = self._fetch_context()

        # Stage 2: Batch-generate all messages in a single Gemini API call
        messages = self._gemini.generate_batch(
            num_messages=self._num_messages,
            language=self.language,
            context=ctx,
        )

        if not messages:
            print("[BlessYouFlow] ⚠ Gemini batch generation failed — no WAV files generated")
            return False

        print(f"[BlessYouFlow] ✓ {len(messages)} messages received — converting to WAV...")

        # Stage 3: Convert each message to a WAV file
        wav_queue: list[Path] = []
        for i, msg in enumerate(messages):
            wav_path = self._tts_output_dir / f"tts_bless_you_{i:02d}.wav"
            tts_text = f"Bless you! {msg}"
            result = self._tts.speak(tts_text, save=True, play=False, save_as=wav_path)
            if result:
                wav_queue.append(result)
                print(f"[BlessYouFlow]   [{i+1}/{len(messages)}] ✓ {wav_path.name}")
            else:
                print(f"[BlessYouFlow]   [{i+1}/{len(messages)}] ❌ WAV generation failed: {msg[:40]}...")

        with self._queue_lock:
            self._wav_queue = wav_queue

        ok = len(self._wav_queue) > 0
        if ok:
            print(f"[BlessYouFlow] ✓ Initialization complete — {len(self._wav_queue)} WAV files ready")
        else:
            print("[BlessYouFlow] ⚠ All WAV generations failed — will use static WAV fallback")
        return ok

    def run(self) -> None:
        """Play the next pre-generated WAV file from the queue.

        Falls back to the static bless_you.wav when the queue is empty.
        Skips playback if audio is already playing.
        """
        with self._play_lock:
            if self._playing:
                print("[BlessYouFlow] ⏭ Already playing — skipping")
                return
            self._playing = True

        try:
            with self._queue_lock:
                wav_to_play = self._wav_queue.pop(0) if self._wav_queue else None
                remaining = len(self._wav_queue)

            if wav_to_play and wav_to_play.exists():
                print(f"[BlessYouFlow] 🎵 Playing: {wav_to_play.name} ({remaining} remaining)")
                self._play_wav(wav_to_play)
            else:
                if wav_to_play:
                    print(f"[BlessYouFlow] ⚠ WAV file missing: {wav_to_play.name} — playing fallback")
                else:
                    print("[BlessYouFlow] ⚠ Queue empty — playing fallback WAV")
                self._play_wav(self.bless_wav_path)
        finally:
            with self._play_lock:
                self._playing = False

    def run_async(self) -> threading.Thread:
        """Run the pipeline in a background thread.

        Returns
        -------
        threading.Thread
            The running thread. Call join() to wait for completion.
        """
        t = threading.Thread(target=self.run, daemon=True)
        t.start()
        return t

    def queue_size(self) -> int:
        """Return the number of pre-generated WAV files remaining in the queue."""
        with self._queue_lock:
            return len(self._wav_queue)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_context(self) -> Optional[dict]:
        """Fetch GPS + weather/air quality context.

        Returns
        -------
        dict | None
            Context dictionary to pass to Gemini. None on fetch failure.
        """
        if not self._enable_context:
            return None
        try:
            location = self._gps.get_location()
            if not location:
                print("[BlessYouFlow] ⚠ GPS lookup failed — proceeding without context")
                return None
            weather = self._weather.get_context(location["lat"], location["lon"], city=location["city"])
            if not weather:
                print("[BlessYouFlow] ⚠ Weather lookup failed — proceeding without context")
                return None

            # Reject partial data: if any numeric field is None, the Gemini prompt
            # would contain "None°C" / "None%" which causes malformed or empty responses.
            _REQUIRED_NUMERIC = ("temperature", "humidity", "wind_speed", "pm2_5", "pm10")
            missing = [k for k in _REQUIRED_NUMERIC if weather.get(k) is None]
            if missing:
                print(f"[BlessYouFlow] ⚠ Incomplete weather data ({', '.join(missing)}) — proceeding without context")
                return None

            return {
                "city":          location["city"],
                "country":       location["country"],
                "temperature":   weather["temperature"],
                "humidity":      weather["humidity"],
                "weather_label": weather["weather_label"],
                "wind_speed":    weather["wind_speed"],
                "aqi_label":     weather["aqi_label"],
                "pm2_5":         weather["pm2_5"],
                "pm10":          weather["pm10"],
            }
        except Exception as e:
            print(f"[BlessYouFlow] ⚠ Context fetch error: {e}")
            return None

    def _play_wav(self, wav_path: Path) -> None:
        """Play a WAV file synchronously via the TTS player's detected command."""
        if not wav_path.exists():
            print(f"[BlessYouFlow] ⚠ WAV not found: {wav_path}")
            return
        self._tts._play_file(wav_path)


if __name__ == "__main__":
    # Standalone test: validate full pipeline in initialize() → run() order
    import sys
    from pathlib import Path

    wav = Path(__file__).resolve().parents[1] / "output_feature" / "sounds" / "bless_you.wav"
    lang = sys.argv[1] if len(sys.argv) > 1 else "en"
    num = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"[BlessYouFlow] Running test (language={lang}, num_messages={num})")
    print(f"[BlessYouFlow] WAV: {wav}")

    try:
        flow = BlessYouFlow(bless_wav_path=wav, language=lang, num_messages=num)
        flow.initialize()  # GPS → Weather → Gemini batch → ElevenLabs batch (blocking)
        print(f"[BlessYouFlow] Queue size: {flow.queue_size()}")
        flow.run()         # Play first pre-generated WAV
        print(f"[BlessYouFlow] Queue size after run: {flow.queue_size()}")
        print("[BlessYouFlow] ✓ Done")
    except (ValueError, ImportError) as e:
        print(f"[ERROR] {e}")
        print("Set GEMINI_API_KEY and ELEVENLABS_API_KEY environment variables and try again.")
