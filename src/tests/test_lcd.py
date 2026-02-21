"""
tests/test_lcd.py
------------------
LCD 디스플레이 테스트 스크립트.

src/output_feature/images/ 안의 이미지(jpg/png/gif)를 순서대로 LCD에 띄웁니다.

GIF 처리
--------
- GIF는 내장된 프레임별 duration을 자동으로 읽어 적용합니다.
- --interval 을 지정하면 GIF 원본 타이밍을 override합니다.
- jpg/png 는 항상 --interval (기본 0.3초) 을 사용합니다.

Run from src/:
    python tests/test_lcd.py

Arguments
---------
--interval   프레임 간격 (초)  GIF: 미지정 시 GIF 원본 타이밍 사용
                               jpg/png: default 0.3
--loops      전체 반복 횟수    default: 0 (무한)
--images     표시할 파일 경로 목록  default: output_feature/images/ 자동 탐색
--once       한 번씩만 순서대로 출력 후 종료

Examples
--------
# 기본 실행 (0.3초 간격, 무한 루프)
python tests/test_lcd.py

# GIF 파일 — 내장 타이밍 사용
python tests/test_lcd.py --images output_feature/images/animation.gif

# GIF 타이밍 override (0.2초 고정)
python tests/test_lcd.py --images output_feature/images/animation.gif --interval 0.2

# jpg + gif 혼합
python tests/test_lcd.py --images output_feature/images/idle.jpg output_feature/images/anim.gif

# 한 번만 출력하고 종료
python tests/test_lcd.py --once

# 3회 반복
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

# 기본 이미지 디렉터리 (src/output_feature/images/)
_DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent.parent / "output_feature" / "images"

# 지원 확장자
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# (PIL.Image, duration_sec) 쌍의 타입
_Frame = tuple[Image.Image, float]


# ---------------------------------------------------------------------- #
# 이미지 로더                                                             #
# ---------------------------------------------------------------------- #

def _resize(img: Image.Image) -> Image.Image:
    return img.convert("RGB").resize((240, 240))


def _load_gif(path: Path, override_sec: Optional[float]) -> list[_Frame]:
    """GIF를 열고 (프레임, duration) 리스트를 반환합니다.

    override_sec 가 주어지면 GIF 내장 타이밍 대신 해당 값을 사용합니다.
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
    """jpg/png 등 정지 이미지를 열고 (프레임, duration) 리스트를 반환합니다."""
    if not path.exists():
        raise FileNotFoundError(path)
    img = _resize(Image.open(path))
    return [(img, interval_sec)]


def _build_playlist(
    paths: list[Path],
    interval_override: Optional[float],
    static_interval: float,
) -> list[tuple[str, int, int, _Frame]]:
    """(파일명, 프레임번호, 전체프레임수, (img, duration)) 리스트를 반환합니다."""
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
    """디렉터리에서 지원 파일을 이름 순으로 수집합니다."""
    return sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )


# ---------------------------------------------------------------------- #
# 메인                                                                    #
# ---------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="LCD 이미지 슬라이드쇼 테스트")
    ap.add_argument(
        "--interval", type=float, default=None,
        help="프레임 간격 (초). GIF: 미지정 시 GIF 내장 타이밍 사용. "
             "jpg/png: 미지정 시 0.3초",
    )
    ap.add_argument(
        "--loops", type=int, default=0,
        help="반복 횟수 (0 = 무한)  default: 0",
    )
    ap.add_argument(
        "--images", nargs="+", default=None,
        help="표시할 파일 경로 목록. 미지정 시 output_feature/images/ 자동 탐색",
    )
    ap.add_argument(
        "--once", action="store_true",
        help="한 번씩 순서대로 출력하고 종료",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    # 파일 목록 구성                                                      #
    # ------------------------------------------------------------------ #
    if args.images:
        paths = [Path(p) for p in args.images]
    else:
        paths = _collect_images(_DEFAULT_IMAGE_DIR)

    if not paths:
        print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {_DEFAULT_IMAGE_DIR}")
        sys.exit(1)

    print("파일 목록:")
    for p in paths:
        tag = "GIF" if p.suffix.lower() == ".gif" else "img"
        print(f"  [{tag}] {p.name}")

    # ------------------------------------------------------------------ #
    # 플레이리스트 생성                                                   #
    # ------------------------------------------------------------------ #
    static_interval = args.interval if args.interval is not None else 0.3

    print("프레임 로드 중...")
    playlist = _build_playlist(paths, args.interval, static_interval)

    n_files  = len(paths)
    n_frames = len(playlist)
    timing   = f"interval={args.interval}s (override)" if args.interval is not None \
               else "GIF 원본 타이밍 / 정지이미지 0.3s"
    print(f"로드 완료.  {n_files}개 파일 → {n_frames} 프레임  [{timing}]")

    # ------------------------------------------------------------------ #
    # LCD 초기화                                                          #
    # ------------------------------------------------------------------ #
    lcd = LCD()

    # ------------------------------------------------------------------ #
    # 슬라이드쇼                                                          #
    # ------------------------------------------------------------------ #
    def _show_frame(idx: int, total: int, name: str, fno: int, ftot: int,
                    img: Image.Image, dur: float) -> None:
        lcd.show(img)
        frame_info = f"frame {fno}/{ftot}" if ftot > 1 else ""
        print(f"  [{idx+1}/{total}] {name} {frame_info}  {dur:.3f}s    ", end="\r")
        time.sleep(dur)

    total = len(playlist)

    if args.once:
        print("1회 출력 후 종료합니다.")
        for i, (name, fno, ftot, (img, dur)) in enumerate(playlist):
            _show_frame(i, total, name, fno, ftot, img, dur)
        print("\n완료.")
        return

    loop_count  = 0
    frame_count = 0
    print("Ctrl+C 로 종료합니다.")
    try:
        while args.loops == 0 or loop_count < args.loops:
            for i, (name, fno, ftot, (img, dur)) in enumerate(playlist):
                _show_frame(i, total, name, fno, ftot, img, dur)
                frame_count += 1
            loop_count += 1
    except KeyboardInterrupt:
        pass

    print(f"\n종료.  {loop_count}회 루프 / 총 {frame_count} 프레임 출력.")


if __name__ == "__main__":
    main()
