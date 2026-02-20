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
from typing import List, Optional, Tuple

from PIL import Image, ImageSequence

# (PIL.Image, duration_sec) 쌍
_Frame = Tuple[Image.Image, float]

# GIF가 아닌 정지 이미지의 기본 프레임 간격 (초)
DEFAULT_FRAME_SEC = 0.05


def _resize(img: Image.Image) -> Image.Image:
    """240×240 RGB로 변환."""
    return img.convert("RGB").resize((240, 240))


def load_frames(path: Path, frame_sec: float = DEFAULT_FRAME_SEC) -> List[_Frame]:
    """이미지 파일을 (프레임, duration) 리스트로 로드합니다.

    - GIF  : 각 프레임을 순서대로 추출하고, frame_sec 을 모든 프레임에 적용합니다.
    - 정지 이미지(jpg/png 등): 단일 프레임으로 반환합니다.

    Parameters
    ----------
    path      : 이미지 파일 경로
    frame_sec : 프레임 표시 간격 (초). GIF·정지 이미지 모두 동일하게 적용됩니다.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    img = Image.open(path)

    if path.suffix.lower() == ".gif":
        frames: List[_Frame] = []
        for frame in ImageSequence.Iterator(img):
            frames.append((_resize(frame.copy()), frame_sec))
        return frames
    else:
        return [(_resize(img), frame_sec)]


# 하위 호환용 단일 프레임 로더
def load_frame(path: Path) -> Image.Image:
    """Load an image file and resize it to 240×240 RGB (정지 이미지 전용)."""
    return load_frames(path)[0][0]


class LCD:
    """Thin wrapper around the ST7789 driver.

    Usage
    -----
    lcd = LCD()
    idle_frames = load_frames(images_dir / "idle.gif")
    lcd.show(idle_frames[0][0])
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
    """Idle GIF 루프 재생 + 감지 시 detect GIF 한 번 재생 후 idle 복귀.

    GIF / 정지 이미지 모두 지원합니다. frame_sec(기본 0.05초) 간격으로
    각 프레임을 표시합니다.

    Usage
    -----
    images_dir = Path("images")
    animator = LCDAnimator(
        lcd           = LCD(),
        idle_path     = images_dir / "bless_you.gif",
        frame_paths   = [images_dir / "bless_you.gif"],
        frame_sec     = 0.05,
    )
    animator.start()   # idle GIF 루프 시작 (백그라운드 스레드)
    # 감지 시:
    animator.trigger() # detect GIF 1회 재생 후 idle 복귀
    """

    def __init__(
        self,
        lcd: LCD,
        idle_path: Path,
        frame_paths: List[Path],
        frame_sec: float = DEFAULT_FRAME_SEC,
        fps: float = 12.0,          # 하위 호환용 (미사용, frame_sec 우선)
    ):
        self.lcd        = lcd
        self.frame_sec  = frame_sec

        # idle 프레임 로드 (GIF 포함)
        self.idle_frames: List[_Frame] = load_frames(idle_path, frame_sec)

        # detect 프레임 로드 (여러 파일 연결)
        self.det_frames: List[_Frame] = []
        for p in frame_paths:
            self.det_frames.extend(load_frames(p, frame_sec))

        self._lock        = threading.Lock()
        self._detecting   = False      # detect 재생 중 여부
        self._stop_idle   = threading.Event()
        self._idle_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Idle GIF를 백그라운드에서 무한 루프 재생합니다."""
        self._stop_idle.clear()
        self._idle_thread = threading.Thread(
            target=self._run_idle_loop, daemon=True
        )
        self._idle_thread.start()

    def trigger(self) -> None:
        """Detect 애니메이션을 1회 재생 후 idle로 복귀 (비블로킹).

        이미 재생 중이면 무시합니다.
        """
        with self._lock:
            if self._detecting:
                return
            self._detecting = True

        threading.Thread(target=self._run_detect, daemon=True).start()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_idle_loop(self) -> None:
        """Idle 프레임을 무한 반복 표시합니다. _detecting 중에는 건너뜁니다."""
        while not self._stop_idle.is_set():
            for img, dur in self.idle_frames:
                if self._stop_idle.is_set():
                    return
                if not self._detecting:
                    self.lcd.show(img)
                time.sleep(dur)

    def _run_detect(self) -> None:
        """Detect 프레임을 1회 재생하고 idle 루프로 복귀합니다."""
        try:
            for img, dur in self.det_frames:
                self.lcd.show(img)
                time.sleep(dur)
        finally:
            with self._lock:
                self._detecting = False
