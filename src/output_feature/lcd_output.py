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

from PIL import Image


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
            raise RuntimeError(f"LCD 드라이버(st7789) import 실패: {e}")

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
