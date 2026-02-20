"""
output_feature/lcd_output.py
-----------------------------
LCD display module for Raspberry Pi (ST7789 240×240).

Extracted from raspi/sneeze-detection/real_time_detection.py.

Hardware assumed:
  - ST7789 240×240 SPI LCD
  - Port 0, CS 1, DC GPIO9, Backlight GPIO13
  - SPI speed 80 MHz, rotation 90°

Replace the constructor arguments to match your wiring.

GIF support
-----------
Pass a .gif file for idle_path or any entry in frame_paths to have
per-frame timing read and applied automatically. Static images (.png /
.jpg etc.) use the fps parameter for their display duration.
"""

import time
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageSequence

# Type alias: (PIL.Image, duration_sec) pair
_Frame = Tuple[Image.Image, float]


def load_frame(path: Path) -> Image.Image:
    """Load an image file and resize it to 240×240 RGB."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB").resize((240, 240))


def _load_path_frames(path: Path, fps: float,
                      override_sec: Optional[float] = None) -> List[_Frame]:
    """Return a list of (Image, duration_sec) pairs for the given path.

    - GIF    : uses each frame's embedded duration; override_sec fixes it.
    - Static : uses 1/fps as duration; override_sec fixes it.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".gif":
        gif = Image.open(path)
        result: List[_Frame] = []
        for frame in ImageSequence.Iterator(gif):
            img = frame.copy().convert("RGB").resize((240, 240))
            native_sec = frame.info.get("duration", 100) / 1000.0
            duration = override_sec if override_sec is not None else native_sec
            result.append((img, duration))
        return result
    else:
        img = Image.open(path).convert("RGB").resize((240, 240))
        duration = override_sec if override_sec is not None else 1.0 / fps
        return [(img, duration)]


class LCD:
    """Thin wrapper around the ST7789 driver.

    Usage
    -----
    lcd = LCD()
    idle = load_frame(images_dir / "idle.png")
    lcd.show(idle)
    """

    def __init__(self):
        try:
            import st7789
        except ImportError as e:
            raise RuntimeError(f"st7789 LCD driver import failed: {e}")

        self.disp = st7789.ST7789(
            rotation=90,
            port=0,
            cs=1,
            dc=9,
            backlight=13,
            spi_speed_hz=80_000_000,
        )

    def show(self, img: Image.Image) -> None:
        """Display a 240×240 PIL Image on the LCD."""
        self.disp.display(img)


class LCDAnimator:
    """Plays a detect animation then returns to idle; runs in a daemon thread.

    GIF support
    -----------
    Pass a .gif file for idle_path or any item in frame_paths to play
    frames with their embedded timing.
    - idle GIF    : loops in a background thread after start() is called.
    - detect GIF  : plays once on trigger(), then returns to idle.

    Usage
    -----
    images_dir = Path("images")
    animator = LCDAnimator(
        lcd        = LCD(),
        idle_path  = images_dir / "idle.gif",      # .png also accepted
        frame_paths = [
            images_dir / "detect.gif",              # .png also accepted
        ],
        fps = 12.0,
    )
    animator.start()        # show idle (GIF: start loop)
    # on detection:
    animator.trigger()      # play detect animation, then return to idle
    """

    def __init__(
        self,
        lcd: LCD,
        idle_path: Path,
        frame_paths: List[Path],
        fps: float = 12.0,
    ):
        self.lcd = lcd
        self.fps = max(1.0, float(fps))

        # idle: GIF -> multiple frames; static -> single frame
        self.idle_frames: List[_Frame] = _load_path_frames(idle_path, self.fps)

        # detect: concatenate frames from each path in order
        self.detect_frames: List[_Frame] = []
        for p in frame_paths:
            self.detect_frames.extend(_load_path_frames(p, self.fps))

        self._lock       = threading.Lock()
        self._active     = False
        self._stop_idle  = threading.Event()

    def start(self) -> None:
        """Start idle display.

        - Static image: shown once immediately.
        - GIF: starts looping in a background thread.
        """
        self._stop_idle.clear()
        if len(self.idle_frames) > 1:
            threading.Thread(target=self._idle_loop, daemon=True).start()
        else:
            self.lcd.show(self.idle_frames[0][0])

    def trigger(self) -> None:
        """Spawn a daemon thread that plays the detect animation once (non-blocking).

        Duplicate triggers while an animation is running are silently ignored.
        """
        threading.Thread(target=self._run_event, daemon=True).start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _idle_loop(self) -> None:
        """Loop idle GIF frames until the stop event is set."""
        while not self._stop_idle.is_set():
            for img, dur in self.idle_frames:
                if self._stop_idle.is_set():
                    return
                self.lcd.show(img)
                time.sleep(dur)

    def _run_event(self) -> None:
        with self._lock:
            if self._active:
                return
            self._active = True

        # stop idle loop
        self._stop_idle.set()

        try:
            for img, dur in self.detect_frames:
                self.lcd.show(img)
                time.sleep(dur)
        finally:
            with self._lock:
                self._active = False
            self.start()  # resume idle
