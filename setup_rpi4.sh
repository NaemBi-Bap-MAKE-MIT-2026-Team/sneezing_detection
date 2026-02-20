#!/usr/bin/env bash
# =============================================================================
# setup_rpi4.sh
# Raspberry Pi 4 (64-bit Raspberry OS / aarch64) 개발환경 초기화 스크립트
#
# 역할:
#   1. 기존 'base' 가상환경 제거
#   2. Python 3.12 설치 확인 (없으면 apt 자동 설치)
#   3. 'base' 이름으로 Python 3.12 가상환경 생성
#   4. src/requirements_rpi4.txt 기반 패키지 설치
#
# 사용법:
#   chmod +x setup_rpi4.sh
#   ./setup_rpi4.sh
# =============================================================================

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_NAME="base"
VENV_PATH="$PROJECT_ROOT/$VENV_NAME"
REQ_FILE="$PROJECT_ROOT/src/requirements_rpi4.txt"
PY="python3.12"

echo ""
echo "============================================="
echo " RPi4 Sneezing Detection 환경 초기화"
echo "============================================="
echo " 프로젝트 경로 : $PROJECT_ROOT"
echo " 가상환경 경로 : $VENV_PATH"
echo " Python 버전   : 3.12.x"
echo "============================================="
echo ""

# ─── Step 1. 기존 가상환경 제거 ───────────────────────────────────────────────
echo "[1/4] 기존 '$VENV_NAME' 가상환경 확인 및 제거..."
if [ -d "$VENV_PATH" ]; then
    rm -rf "$VENV_PATH"
    echo "      삭제 완료: $VENV_PATH"
else
    echo "      기존 가상환경 없음. 스킵."
fi

# ─── Step 2. Python 3.12 확인 ─────────────────────────────────────────────────
echo ""
echo "[2/4] Python 3.12 설치 확인..."
if command -v $PY &>/dev/null; then
    echo "      발견: $($PY --version)"
else
    echo "      Python 3.12 미발견. apt 패키지로 설치합니다..."
    echo "      (sudo 권한이 필요합니다)"
    sudo apt update -y
    sudo apt install -y python3.12 python3.12-venv python3.12-dev
    echo "      Python 3.12 설치 완료: $($PY --version)"
fi

# ─── Step 3. 가상환경 생성 ────────────────────────────────────────────────────
echo ""
echo "[3/4] 가상환경 '$VENV_NAME' 생성 (Python 3.12)..."
$PY -m venv "$VENV_PATH"
echo "      완료: $VENV_PATH"
echo "      Python: $("$VENV_PATH/bin/python" --version)"

# ─── Step 4. 패키지 설치 ──────────────────────────────────────────────────────
echo ""
echo "[4/4] 패키지 설치 중 ($REQ_FILE)..."
"$VENV_PATH/bin/pip" install --upgrade pip --quiet
"$VENV_PATH/bin/pip" install -r "$REQ_FILE"

# ─── 완료 메시지 ──────────────────────────────────────────────────────────────
echo ""
echo "============================================="
echo " 설정 완료!"
echo "============================================="
echo ""
echo " 가상환경 활성화:"
echo "   source $VENV_PATH/bin/activate"
echo ""
echo " 실행 (LCD 없이):"
echo "   cd $PROJECT_ROOT/src"
echo "   python main.py --no-lcd"
echo ""
echo " 실행 (ST7789 LCD 포함):"
echo "   pip install st7789"
echo "   python main.py"
echo "============================================="
