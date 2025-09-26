@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo 🚀 Toss Click Prediction 환경 설정을 시작합니다...

:: Conda 설치 확인
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ Conda가 설치되지 않았습니다.
    echo    Miniconda 또는 Anaconda를 먼저 설치해주세요:
    echo    https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo ✅ Conda가 설치되어 있습니다.

:: GPU 지원 확인
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    echo ✅ NVIDIA GPU가 감지되었습니다.
    set USE_GPU=1
    set ENV_FILE=environment.yml
    set ENV_NAME=toss-click-prediction
) else (
    echo ℹ️  NVIDIA GPU가 감지되지 않았습니다. CPU 버전을 사용합니다.
    set USE_GPU=0
    set ENV_FILE=environment-cpu.yml
    set ENV_NAME=toss-click-prediction-cpu
)

:: 기존 환경 확인
conda env list | findstr /C:"%ENV_NAME%" >nul
if %errorlevel% equ 0 (
    echo ⚠️  환경 '%ENV_NAME%'이 이미 존재합니다.
    set /p REMOVE="기존 환경을 제거하고 새로 생성하시겠습니까? (y/N): "
    if /i "!REMOVE!"=="y" (
        echo 🗑️  기존 환경을 제거합니다...
        conda env remove -n %ENV_NAME% -y
    ) else (
        echo ❌ 환경 생성을 취소합니다.
        pause
        exit /b 1
    )
)

:: 환경 생성
echo 📦 %ENV_NAME% 환경을 생성합니다...
conda env create -f %ENV_FILE%

if %errorlevel% equ 0 (
    echo ✅ 환경 '%ENV_NAME%'이 성공적으로 생성되었습니다!
    echo.
    echo 🎯 다음 명령으로 환경을 활성화하세요:
    echo    conda activate %ENV_NAME%
    echo.
    echo 🎉 환경 설정이 완료되었습니다!
    echo.
    echo 📚 다음 단계:
    echo 1. 환경 활성화
    echo 2. 데이터 파일 준비 ^(train.parquet, test.parquet, sample_submission.csv^)
    echo 3. python train.py 또는 python hyperparam_search.py 실행
) else (
    echo ❌ 환경 생성에 실패했습니다.
    pause
    exit /b 1
)

pause
