# Toss Click Prediction Project

딥러닝을 이용한 클릭 예측 프로젝트입니다.

## 프로젝트 구조

```
├── main.py                    # 메인 설정 및 초기화
├── utils.py                   # 유틸리티 함수들
├── config.yaml               # 설정 파일
├── data_loader.py            # 데이터 로더
├── model.py                  # 모델 정의
├── train.py                  # 모델 훈련
├── predict.py                # 예측 및 제출
├── hyperparam_search.py      # 하이퍼파라미터 최적화
├── train_with_best_params.py # 최적 파라미터로 훈련
├── analysis/                 # 데이터 분석
│   ├── eda.py               # 상세한 EDA (샘플링)
│   ├── eda_utils.py         # EDA 유틸리티 함수
│   ├── quick_eda.py         # 빠른 데이터 개요
│   ├── chunk_eda.py         # 청크 단위 EDA (전체 데이터)
│   └── chunk_utils.py       # 청크 분석 유틸리티
├── requirements.txt          # Pip 패키지 목록
├── environment.yml           # Conda 환경 (CUDA)
├── environment-cpu.yml       # Conda 환경 (CPU)
├── setup_env.sh              # 환경 설정 스크립트 (Linux/macOS)
├── setup_env.bat             # 환경 설정 스크립트 (Windows)
└── README.md                 # 사용 가이드
```

## 설치

### Option 1: Conda Environment (추천)

#### 자동 설정 (권장)
```bash
# Linux/macOS
./setup_env.sh

# Windows
setup_env.bat
```

#### 수동 설정
```bash
# CUDA 지원 버전 (GPU 사용 가능한 경우)
conda env create -f environment.yml
conda activate toss-click-prediction

# CPU 전용 버전
conda env create -f environment-cpu.yml
conda activate toss-click-prediction-cpu
```

### Option 2: Pip

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 데이터 탐색 (EDA)

```bash
# 빠른 데이터 개요 (샘플링)
python analysis/quick_eda.py

# 상세한 EDA (샘플링, 시각화 포함)
python analysis/eda.py

# 청크 단위 EDA (전체 데이터, 메모리 효율적)
python analysis/chunk_eda.py

# 청크 크기 조정 (기본: 100K)
python analysis/chunk_eda.py --chunk_size 50000
```

### 2. 기본 훈련

```bash
# 기본 설정으로 훈련
python train.py

# 예측 및 제출 파일 생성
python predict.py
```

### 3. 설정 변경

`config.yaml` 파일을 수정하여 설정 변경:

```yaml
BATCH_SIZE: 2048
EPOCHS: 20
LEARNING_RATE: 0.0005
MODEL:
  LSTM_HIDDEN: 128
  HIDDEN_UNITS: [512, 256, 128]
  DROPOUT: 0.3
```

### 4. 하이퍼파라미터 최적화

```bash
# Optuna를 사용한 하이퍼파라미터 서치
python hyperparam_search.py

# 최적 파라미터로 최종 훈련
python train_with_best_params.py
```

## 하이퍼파라미터 서치 범위

- **Learning Rate**: 1e-5 ~ 1e-2 (log scale)
- **Batch Size**: [1024, 2048, 4096, 8192]
- **LSTM Hidden Size**: 32 ~ 256 (step: 32)
- **MLP Layers**: 2 ~ 4 layers
- **Hidden Units**: 64 ~ 1024 (adaptive)
- **Dropout**: 0.1 ~ 0.5 (step: 0.1)

## 출력 파일

### 모델 및 훈련 결과
- `trained_model.pth`: 기본 훈련된 모델
- `optimized_model.pth`: 최적화된 모델
- `best_hyperparams.json`: 최적 하이퍼파라미터
- `training_log.json`: 훈련 기록
- `baseline_submit.csv`: 제출 파일

### 하이퍼파라미터 최적화
- `optuna_study.db`: Optuna 스터디 DB
- `optimization_history.html`: 최적화 히스토리 시각화
- `param_importances.html`: 파라미터 중요도 시각화

### EDA 결과 (analysis/results/)
- `eda_report.json`: 샘플 데이터 EDA 결과 요약
- `chunk_eda_results.json`: 전체 데이터 청크 분석 결과
- `target_distribution.png`: 타겟 변수 분포
- `basic_features_distribution.png`: 기본 피처 분포
- `sequence_analysis.png`: 시퀀스 분석
- `feature_correlations.png`: 피처 상관관계
- `chunk_eda_summary.png`: 청크 분석 요약 시각화

## 설정 검증

YAML 설정 파일에 잘못된 키가 있으면 자동으로 에러가 발생합니다:

```python
# 유효한 설정 키 확인
from main import CFG
from utils import print_valid_keys
print_valid_keys(CFG)
```

## 주요 기능

- ✅ **모듈화된 구조**: 각 기능별로 파일 분리
- ✅ **YAML 설정 관리**: 설정 파일로 쉬운 실험
- ✅ **설정 검증**: 잘못된 키 자동 감지
- ✅ **하이퍼파라미터 최적화**: Optuna 자동 서치
- ✅ **시각화**: 최적화 과정 시각화 (영어 레이블)
- ✅ **메모리 효율적 EDA**: 청크 단위 대용량 데이터 분석
- ✅ **재현성**: 시드 고정으로 일관된 결과

## 시각화 설정

모든 그래프는 영어로 출력되며, 한글 폰트 문제를 방지하기 위해 다음과 같이 설정되어 있습니다:

```python
# Font settings for plots
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
```
