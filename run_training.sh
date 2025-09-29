#!/usr/bin/env bash
set -euo pipefail

# 설정
STUDIO_NAME="fun-tan-2pfe"
TEAMSPACE="TOSS-challenge"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"/toss

# Config 파일 설정 (반드시 지정해야 함)
if [ -z "${CONFIG_FILE:-}" ]; then
    echo "❌ CONFIG_FILE 환경변수가 설정되지 않았습니다!"
    echo ""
    echo "사용법:"
    echo "  CONFIG_FILE=config_fold1.yaml ./run_training.sh"
    echo "  CONFIG_FILE=config_fold2.yaml ./run_training.sh"
    echo ""
    echo "사용 가능한 config 파일들:"
    ls -1 *.yaml 2>/dev/null | sed 's/^/  - /' || echo "  (config 파일이 없습니다)"
    exit 1
fi

echo "🚀 훈련 워크플로우 시작"
echo "=================================="
echo "📅 시작 시간: $(date)"
echo "🏢 Studio: $STUDIO_NAME"
echo "👥 Teamspace: $TEAMSPACE"
echo "📁 작업 디렉토리: $SCRIPT_DIR"
echo "📋 설정 파일: $CONFIG_FILE"
echo ""

# 작업 디렉토리로 이동
cd "$SCRIPT_DIR"

# Python 환경 확인
echo "🐍 Python 환경 확인..."
python --version
echo ""

# 필요한 파일 존재 확인
echo "📋 필수 파일 확인..."
if [ ! -f "train_and_predict.py" ]; then
    echo "❌ train_and_predict.py 파일이 없습니다!"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 설정 파일이 없습니다: $CONFIG_FILE"
    exit 1
fi

echo "✅ 모든 필수 파일 확인 완료"
echo ""

# 훈련 실행
echo "🏋️ 훈련 워크플로우 실행 중..."
echo "=================================="

# 훈련 실행 (오류 발생 시에도 계속 진행)
if python train_and_predict.py --config "$CONFIG_FILE"; then
    echo ""
    echo "✅ 훈련 워크플로우 성공적으로 완료!"
    TRAINING_STATUS="SUCCESS"
else
    echo ""
    echo "❌ 훈련 워크플로우 실행 중 오류 발생!"
    TRAINING_STATUS="FAILED"
fi

echo ""
echo "📊 훈련 결과 요약:"
echo "   • 상태: $TRAINING_STATUS"
echo "   • 완료 시간: $(date)"
echo ""

# 결과 디렉토리 확인
echo "📁 결과 파일 확인..."
if [ -d "results_"* ]; then
    echo "✅ 결과 디렉토리 생성됨:"
    ls -la results_*
else
    echo "⚠️ 결과 디렉토리를 찾을 수 없습니다."
fi

echo ""

# Lightning Studio 정지 (성공/실패 관계없이)
echo "🛑 Lightning Studio 정지 중..."
echo "=================================="

if lightning studio stop --name "$STUDIO_NAME" --teamspace "$TEAMSPACE"; then
    echo "✅ Lightning Studio 정지 완료!"
else
    echo "❌ Lightning Studio 정지 실패!"
    echo "⚠️ 수동으로 정지해주세요:"
    echo "   lightning studio stop --name $STUDIO_NAME --teamspace $TEAMSPACE"
fi

echo ""
echo "🎉 모든 작업 완료!"
echo "📅 종료 시간: $(date)"
echo "=================================="
