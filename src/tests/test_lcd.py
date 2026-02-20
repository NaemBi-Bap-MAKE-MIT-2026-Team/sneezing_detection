"""
tests/test_lcd.py
------------------
LCD 디스플레이 테스트 스크립트.

src/output_feature/images/ 안의 이미지를 순서대로 LCD에 띄웁니다.
프레임 간격과 반복 횟수를 CLI 인자로 조정할 수 있습니다.

Run from src/:
    python tests/test_lcd.py

Arguments
---------
--interval   프레임 간격 (초)         default: 0.3
--loops      전체 반복 횟수           default: 0 (무한)
--images     표시할 이미지 경로 목록  default: output_feature/images/ 자동 탐색
--once       이미지를 한 번씩만 순서대로 출력 후 종료

Examples
--------
# 기본 0.3초 간격 무한 루프
python tests/test_lcd.py

# 0.5초 간격, 3회 반복
python tests/test_lcd.py --interval 0.5 --loops 3

# 특정 이미지만 지정
python tests/test_lcd.py --images output_feature/images/detect1.jpg output_feature/images/detect2.jpg

# 한 번만 순서대로 출력하고 종료
python tests/test_lcd.py --once
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running as script from src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from output_feature.lcd_output import LCD, load_frame

# 기본 이미지 디렉터리 (src/output_feature/images/)
_DEFAULT_IMAGE_DIR = Path(__file__).resolve().parent.parent / "output_feature" / "images"

# 지원 확장자
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def _collect_images(image_dir: Path) -> list[Path]:
    """이미지 디렉터리에서 파일을 이름 순으로 수집합니다."""
    files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="LCD 이미지 슬라이드쇼 테스트")
    ap.add_argument(
        "--interval", type=float, default=0.3,
        help="프레임 간격 (초)  default: 0.3",
    )
    ap.add_argument(
        "--loops", type=int, default=0,
        help="반복 횟수 (0 = 무한)  default: 0",
    )
    ap.add_argument(
        "--images", nargs="+", default=None,
        help="표시할 이미지 경로 목록. 미지정 시 output_feature/images/ 자동 탐색",
    )
    ap.add_argument(
        "--once", action="store_true",
        help="이미지를 한 번씩 순서대로 출력하고 종료",
    )
    args = ap.parse_args()

    # ------------------------------------------------------------------ #
    # 이미지 목록 구성                                                    #
    # ------------------------------------------------------------------ #
    if args.images:
        image_paths = [Path(p) for p in args.images]
    else:
        image_paths = _collect_images(_DEFAULT_IMAGE_DIR)

    if not image_paths:
        print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {_DEFAULT_IMAGE_DIR}")
        sys.exit(1)

    print(f"이미지 {len(image_paths)}장 로드 중...")
    for p in image_paths:
        print(f"  {p.name}")

    frames = [load_frame(p) for p in image_paths]
    print(f"로드 완료.  interval={args.interval}s  "
          f"loops={'∞' if args.loops == 0 else args.loops}")

    # ------------------------------------------------------------------ #
    # LCD 초기화                                                          #
    # ------------------------------------------------------------------ #
    lcd = LCD()

    # ------------------------------------------------------------------ #
    # 슬라이드쇼                                                          #
    # ------------------------------------------------------------------ #
    if args.once:
        print("순서대로 1회 출력 후 종료합니다.")
        for i, (path, frame) in enumerate(zip(image_paths, frames)):
            print(f"  [{i+1}/{len(frames)}] {path.name}")
            lcd.show(frame)
            time.sleep(args.interval)
        print("완료.")
        return

    loop_count  = 0
    frame_count = 0

    print("Ctrl+C 로 종료합니다.")
    try:
        while args.loops == 0 or loop_count < args.loops:
            for i, (path, frame) in enumerate(zip(image_paths, frames)):
                lcd.show(frame)
                frame_count += 1
                print(f"  loop {loop_count+1}  [{i+1}/{len(frames)}] {path.name}"
                      f"  (total {frame_count} frames)", end="\r")
                time.sleep(args.interval)
            loop_count += 1
    except KeyboardInterrupt:
        pass

    print(f"\n종료.  총 {frame_count} 프레임 출력.")


if __name__ == "__main__":
    main()
