#!/bin/bash

# Toss Click Prediction 프로젝트 환경 설정 스크립트

echo "🚀 Toss Click Prediction 환경 설정을 시작합니다..."

# GPU 지원 확인
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU가 감지되었습니다."
        nvidia-smi --query-gpu=name --format=csv,noheader
        return 0
    else
        echo "ℹ️  NVIDIA GPU가 감지되지 않았습니다. CPU 버전을 사용합니다."
        return 1
    fi
}

# Conda 설치 확인
check_conda() {
    if ! command -v conda &> /dev/null; then
        echo "❌ Conda가 설치되지 않았습니다."
        echo "   Miniconda 또는 Anaconda를 먼저 설치해주세요:"
        echo "   https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    echo "✅ Conda가 설치되어 있습니다: $(conda --version)"
}

# 환경 생성
create_environment() {
    local env_file=$1
    local env_name=$2
    
    echo "📦 $env_name 환경을 생성합니다..."
    
    if conda env list | grep -q "^$env_name "; then
        echo "⚠️  환경 '$env_name'이 이미 존재합니다."
        read -p "기존 환경을 제거하고 새로 생성하시겠습니까? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🗑️  기존 환경을 제거합니다..."
            conda env remove -n $env_name -y
        else
            echo "❌ 환경 생성을 취소합니다."
            exit 1
        fi
    fi
    
    conda env create -f $env_file
    
    if [ $? -eq 0 ]; then
        echo "✅ 환경 '$env_name'이 성공적으로 생성되었습니다!"
        echo ""
        echo "🎯 다음 명령으로 환경을 활성화하세요:"
        echo "   conda activate $env_name"
        echo ""
        echo "📋 환경 정보:"
        conda env list | grep $env_name
    else
        echo "❌ 환경 생성에 실패했습니다."
        exit 1
    fi
}

# 메인 실행
main() {
    check_conda
    
    echo ""
    if check_gpu; then
        echo ""
        echo "🎮 GPU 지원 환경을 설정합니다..."
        create_environment "environment.yml" "toss-click-prediction"
    else
        echo ""
        echo "💻 CPU 전용 환경을 설정합니다..."
        create_environment "environment-cpu.yml" "toss-click-prediction-cpu"
    fi
    
    echo ""
    echo "🎉 환경 설정이 완료되었습니다!"
    echo ""
    echo "📚 다음 단계:"
    echo "1. 환경 활성화"
    echo "2. 데이터 파일 준비 (train.parquet, test.parquet, sample_submission.csv)"
    echo "3. python train.py 또는 python hyperparam_search.py 실행"
}

# 스크립트 실행
main "$@"
