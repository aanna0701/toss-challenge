#!/usr/bin/env bash
set -euo pipefail

# 설정
STUDIO_NAME="fun-tan-2pfe"
TEAMSPACE="TOSS-challenge"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"/toss

# Config 파일들 설정 (순차적으로 실행)
CONFIG_FILES=("config_fold1.yaml" "config_fold2.yaml")

echo "📋 실행할 설정 파일들:"
for config in "${CONFIG_FILES[@]}"; do
    echo "  - $config"
done
echo ""

echo "🚀 훈련 워크플로우 시작"
echo "=================================="
echo "📅 시작 시간: $(date)"
echo "🏢 Studio: $STUDIO_NAME"
echo "👥 Teamspace: $TEAMSPACE"
echo "📁 작업 디렉토리: $SCRIPT_DIR"
echo "📋 총 설정 파일: ${#CONFIG_FILES[@]}개"
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

# 모든 설정 파일 존재 확인
for config in "${CONFIG_FILES[@]}"; do
    if [ ! -f "$config" ]; then
        echo "❌ 설정 파일이 없습니다: $config"
        exit 1
    fi
done

echo "✅ 모든 필수 파일 확인 완료"
echo ""

# 훈련 실행 (각 설정 파일별로 순차 실행)
echo "🏋️ 훈련 워크플로우 실행 중..."
echo "=================================="

TOTAL_CONFIGS=${#CONFIG_FILES[@]}
SUCCESS_COUNT=0
FAILED_COUNT=0

for i in "${!CONFIG_FILES[@]}"; do
    CONFIG_FILE="${CONFIG_FILES[$i]}"
    CONFIG_NUM=$((i + 1))
    
    echo ""
    echo "🔄 [$CONFIG_NUM/$TOTAL_CONFIGS] 설정 파일 실행: $CONFIG_FILE"
    echo "=================================="
    
    if python train_and_predict.py --config "$CONFIG_FILE"; then
        echo ""
        echo "✅ [$CONFIG_NUM/$TOTAL_CONFIGS] $CONFIG_FILE 훈련 성공!"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo ""
        echo "❌ [$CONFIG_NUM/$TOTAL_CONFIGS] $CONFIG_FILE 훈련 실패!"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    
    echo "📊 현재 진행 상황: 성공 $SUCCESS_COUNT/$TOTAL_CONFIGS, 실패 $FAILED_COUNT/$TOTAL_CONFIGS"
done

echo ""
echo "📊 전체 훈련 결과 요약:"
echo "   • 총 설정 파일: $TOTAL_CONFIGS개"
echo "   • 성공: $SUCCESS_COUNT개"
echo "   • 실패: $FAILED_COUNT개"
echo "   • 완료 시간: $(date)"
echo ""

# 전체 결과 상태 결정
if [ $FAILED_COUNT -eq 0 ]; then
    TRAINING_STATUS="ALL_SUCCESS"
elif [ $SUCCESS_COUNT -eq 0 ]; then
    TRAINING_STATUS="ALL_FAILED"
else
    TRAINING_STATUS="PARTIAL_SUCCESS"
fi

# 결과 디렉토리 확인
echo "📁 결과 파일 확인..."
if ls fold*_*/ 1> /dev/null 2>&1; then
    echo "✅ 결과 디렉토리 생성됨:"
    ls -la fold*_*/
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
