"""
tests/test_lcd.py
------------------
LCD display test script.

Displays images (jpg/png/gif) found in src/output_feature/images/
on the LCD in alphabetical order.

GIF handling
-------------
- GIF files use their embedded per-frame duration automatically.
- Specifying --interval overrides the native GIF timing.
- jpg/png always use --interval (default 0.3 s).

Run from src/:
    python tests/test_lcd.py

Arguments
---------
--interval   Frame interval in seconds.  GIF: native timing if omitted.
                                         jpg/png: default 0.3
--loops      Total repeat count.         default: 0 (infinite)
--images     List of file paths to show. default: auto-scan output_feature/images/
--once       Display each file once in order, then exit.

Examples
--------
# Default run (0.3 s interval, infinite loop)
python tests/test_lcd.py

# GIF file — use embedded timing
python tests/test_lcd.py --images output_feature/images/animation.gif

# Override GIF timing (fixed 0.2 s)
python tests/test_lcd.py --images output_feature/images/animation.gif --interval 0.2

# Mix jpg + gif
python tests/test_lcd.py --images output_feature/images/idle.jpg output_feature/images/anim.gif

# Display once then exit
python tests/test_lcd.py --once

# Repeat 3 times
python tests/test_lcd.py --loops 3
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# Allow running as script from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageSequence
from output_feature.lcd_output import LCD

# Default image directory (src/output_feature/images/)
_DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent.parent / "output_feature" / "images"

# Supported extensions
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# Type alias: (PIL.Image, duration_sec) pair
_Frame = tuple[Image.Image, float]


# ---------------------------------------------------------------------- #
# Image loaders                                                           #
# ---------------------------------------------------------------------- #

def _resize(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((240, 240))


def _load_gif(path: Path, override_sec: Optional[float]) -> list[_Frame]:
    """Open a GIF and return a list of (frame, duration_sec) pairs.

    If override_sec is given, it replaces the GIF's embedded timing.
    """
    gif = Image.open(path)
    result: list[_Frame] = []
    for frame in ImageSequence.Iterator(gif):
        img = _resize(frame.copy())
        native_sec = frame.info.get("duration", 100) / 1000.0
        duration   = override_sec if override_sec is not None else native_sec
        result.append((img, duration))
    return result


def _load_static(path: Path, interval_sec: float) -> list[_Frame]:
    """Open a static image (jpg/png etc.) and return a [(frame, duration_sec)] list."""
    if not path.exists():
        raise FileNotFoundError(path)
    img = _resize(Image.open(path))
    return [(img, interval_sec)]


def _build_playlist(
    paths: list[Path],
    interval_override: Optional[float],
    static_interval: float,
) -> list[tuple[str, int, int, _Frame]]:
    """Return a list of (filename, frame_no, total_frames, (img, duration_sec)) tuples."""
    playlist = []
    for path in paths:
        if path.suffix.lower() == ".gif":
            frames = _load_gif(path, interval_override)
        else:
            sec    = interval_override if interval_override is not None else static_interval
            frames = _load_static(path, sec)
        total = len(frames)
        for i, frame in enumerate(frames):
            playlist.append((path.name, i + 1, total, frame))
    return playlist


def _collect_images(image_dir: Path) -> list[Path]:
    """Collect supported image files from a directory, sorted by name."""
    return sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------- #
# Main                                                                    #
# ---------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="LCD image slideshow test")
    ap.add_argument(
        "--interval", type=float, default=None,
        help="Frame interval (seconds). GIF: uses native timing if omitted. "
             "jpg/png: 0.3 s if omitted.",
    )
    ap.add_argument(
        "--loops", type=int, default=0,
        help="Repeat count (0 = infinite).  default: 0",
    )
    ap.add_argument(
        "--images", nargs="+", default=None,
        help="List of file paths to display. Auto-scans output_feature/images/ if omitted.",
    )
    ap.add_argument(
        "--once", action="store_true",
        help="Display each file once in order, then exit.",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    # Build file list                                                     #
    # ------------------------------------------------------------------ #
    if args.images:
        paths = [Path(p) for p in args.images]
    else:
        paths = _collect_images(_DEFAULT_IMAGE_DIR)

    if not paths:
        print(f"[ERROR] No image files found in: {_DEFAULT_IMAGE_DIR}")
        sys.exit(1)

    print("File list:")
    for p in paths:
        tag = "GIF" if p.suffix.lower() == ".gif" else "img"
        print(f"  [{tag}] {p.name}")

    # ------------------------------------------------------------------ #
    # Build playlist                                                      #
    # ------------------------------------------------------------------ #
    static_interval = args.interval if args.interval is not None else 0.3

    print("Loading frames...")
    playlist = _build_playlist(paths, args.interval, static_interval)

    n_files  = len(paths)
    n_frames = len(playlist)
    timing   = f"interval={args.interval}s (override)" if args.interval is not None \
               else "GIF native timing / static images 0.3 s"
    print(f"Loaded.  {n_files} file(s) -> {n_frames} frame(s)  [{timing}]")

    # ------------------------------------------------------------------ #
    # Initialise LCD                                                      #
    # ------------------------------------------------------------------ #
    lcd = LCD()

    # ------------------------------------------------------------------ #
    # Slideshow                                                           #
    # ------------------------------------------------------------------ #
    def _show_frame(idx: int, total: int, name: str, fno: int, ftot: int,
                    img: Image.Image, dur: float) -> None:
        lcd.show(img)
        frame_info = f"frame {fno}/{ftot}" if ftot > 1 else ""
        print(f"  [{idx+1}/{total}] {name} {frame_info}  {dur:.3f}s    ", end="\r")
        time.sleep(dur)

    total = len(playlist)

    if args.once:
        print("Displaying once then exiting.")
        for i, (name, fno, ftot, (img, dur)) in enumerate(playlist):
            _show_frame(i, total, name, fno, ftot, img, dur)
        print("\nDone.")
        return

    loop_count  = 0
    frame_count = 0
    print("Press Ctrl+C to stop.")
    try:
        while args.loops == 0 or loop_count < args.loops:
            for i, (name, fno, ftot, (img, dur)) in enumerate(playlist):
                _show_frame(i, total, name, fno, ftot, img, dur)
                frame_count += 1
            loop_count += 1
    except KeyboardInterrupt:
        pass

    print(f"\nStopped.  {loop_count} loop(s) / {frame_count} frame(s) displayed.")


if __name__ == "__main__":
    main()
