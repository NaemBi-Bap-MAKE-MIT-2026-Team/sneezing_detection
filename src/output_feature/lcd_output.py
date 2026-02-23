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
"""

import time
import threading
from pathlib import Path
from typing import List, Optional

from PIL import Image, ImageSequence


def load_frame(path: Path) -> Image.Image:
    """Load an image file and resize it to 240×240 RGB."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB").resize((240, 240))


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
            raise RuntimeError(f"LCD driver (st7789) import failed: {e}")

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

    Usage
    -----
    images_dir = Path("images")
    animator = LCDAnimator(
        lcd        = LCD(),
        idle_path  = images_dir / "idle.png",
        frame_paths = [
            images_dir / "detect1.png",
            images_dir / "detect2.png",
            images_dir / "detect3.png",
        ],
        fps = 12.0,
    )
    animator.start()        # show idle
    # on detection:
    animator.trigger()      # plays detect1→detect2→detect3, then reverts to idle
    """

    def __init__(
        self,
        lcd: LCD,
        idle_path: Path,
        frame_paths: List[Path],
        fps: float = 12.0,
    ):
        self.lcd    = lcd
        self.idle   = load_frame(idle_path)
        self.frames = [load_frame(p) for p in frame_paths]
        self.fps    = max(1.0, float(fps))

        self._lock   = threading.Lock()
        self._active = False

    def start(self) -> None:
        """Display the idle frame immediately."""
        self.lcd.show(self.idle)

    def trigger(self) -> None:
        """Spawn a daemon thread that plays the animation once (non-blocking).

        Duplicate triggers while an animation is running are silently ignored.
        """
        threading.Thread(target=self._run_event, daemon=True).start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_event(self) -> None:
        with self._lock:
            if self._active:
                return
            self._active = True

        try:
            dt = 1.0 / self.fps
            for frame in self.frames:
                self.lcd.show(frame)
                time.sleep(dt)
        finally:
            self.lcd.show(self.idle)
            with self._lock:
                self._active = False


class GifAnimator:
    """Plays a GIF file once on detection, then shows a black screen.

    Usage
    -----
    animator = GifAnimator(
        lcd      = LCD(),
        gif_path = Path("output_feature/images/bless_you.gif"),
    )
    animator.start()    # show black screen initially
    animator.trigger()  # play GIF once, then back to black (non-blocking)

    Parameters
    ----------
    lcd        : LCD instance.
    gif_path   : Path to the GIF file.
    fade       : Enable fade in/out effect (default False).
                 Enable only after verifying basic playback works.
    fade_steps : Number of frames over which fade in/out is applied.
    """

    def __init__(
        self,
        lcd: "LCD",
        gif_path: Path,
        fade: bool = False,
        fade_steps: int = 5,
    ):
        self.lcd        = lcd
        self.fade       = fade
        self.fade_steps = fade_steps

        self._black = Image.new("RGB", (240, 240), (0, 0, 0))

        # Load GIF frames with per-frame duration
        self._frames: List[tuple] = []
        gif = Image.open(gif_path)
        for frame in ImageSequence.Iterator(gif):
            img = frame.copy().convert("RGB").resize((240, 240))
            duration_ms = frame.info.get("duration", 100)
            self._frames.append((img, duration_ms / 1000.0))

        self._lock   = threading.Lock()
        self._active = False

    def start(self) -> None:
        """Show black screen initially."""
        self.lcd.show(self._black)

    def trigger(self) -> None:
        """Spawn a daemon thread that plays the GIF once (non-blocking).

        Duplicate triggers while the GIF is playing are silently ignored.
        After playback the LCD is set to black.
        """
        threading.Thread(target=self._run_event, daemon=True).start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _blend_fade(self, img: Image.Image, alpha: float) -> Image.Image:
        """Blend image with black (alpha=0 → black, alpha=1 → original)."""
        return Image.blend(self._black, img, max(0.0, min(1.0, alpha)))

    def _run_event(self) -> None:
        with self._lock:
            if self._active:
                return
            self._active = True

        try:
            total = len(self._frames)
            for i, (img, duration) in enumerate(self._frames):
                if self.fade:
                    if i < self.fade_steps:
                        img = self._blend_fade(img, (i + 1) / self.fade_steps)
                    elif i >= total - self.fade_steps:
                        img = self._blend_fade(img, (total - i) / self.fade_steps)
                self.lcd.show(img)
                time.sleep(duration)
        finally:
            self.lcd.show(self._black)
            with self._lock:
                self._active = False
