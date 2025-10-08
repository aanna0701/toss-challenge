# Toss Click Prediction Challenge

클릭 예측 대회를 위한 머신러닝 프로젝트입니다. GBDT (XGBoost/CatBoost) 및 DNN (Deep Neural Network) 모델을 지원합니다.

## 🚀 빠른 시작

### 1. 환경 설정 (통합 환경)
```bash
# 단일 통합 환경으로 GBDT와 DNN 모두 사용 가능
conda env create -f environment.yaml
conda activate toss-env
```

### 2. GBDT 모델 실행
```bash
# XGBoost 기본 실행
python train_and_predict_GBDT.py

# CatBoost 사용
python train_and_predict_GBDT.py --config config_GBDT.yaml --preset catboost_deep

# Validation ratio 변경
python train_and_predict_GBDT.py --val-ratio 0.2
```

### 3. DNN 모델 실행
```bash
# 멀티 GPU 훈련 (DDP)
python train_and_predict_dnn_ddp.py
```

## 📁 프로젝트 구조

```
toss-challenge/
├── train_and_predict_GBDT.py    # GBDT 모델 훈련 및 예측
├── train_and_predict_dnn_ddp.py # DNN 멀티 GPU 훈련
├── hpo_xgboost.py               # XGBoost 하이퍼파라미터 최적화
├── hpo_catboost.py              # CatBoost 하이퍼파라미터 최적화
├── hpo_dnn.py                   # DNN 하이퍼파라미터 최적화
├── dataset_split.py             # 데이터셋 10-fold 분할
├── utils.py                     # 공통 유틸리티 함수
├── data_loader.py               # 데이터 로더 (DNN + GBDT)
├── mixup.py                     # MixUp 데이터 증강 함수
├── config_GBDT.yaml            # GBDT 설정 파일
├── environment.yaml            # 통합 conda 환경 설정
├── analysis/                   # 데이터 분석 스크립트 (별도 README 참조)
└── data/                       # 데이터 디렉토리
    ├── train.parquet           # 훈련 데이터 (필수)
    └── test.parquet            # 테스트 데이터 (필수)
```

## 🔧 환경 설정

### 통합 환경 (environment.yaml)
단일 conda 환경으로 GBDT와 DNN 모델을 모두 실행할 수 있습니다.

**주요 라이브러리:**
- Python 3.10
- **RAPIDS 스택**: cudf, nvtabular, cupy (GPU 가속 데이터 처리)
- **GBDT 모델**: XGBoost, CatBoost
- **딥러닝**: PyTorch, Lightning (pip 설치로 GPU 호환성 최적화)
- **데이터 처리**: pandas, numpy, scikit-learn, dask

**설치:**
```bash
conda env create -f environment.yaml
conda activate toss-env
```

**기존 환경 업데이트:**
```bash
conda activate toss-env
conda env update -f environment.yaml --prune
```

## 🛠️ GBDT 모델 사용법

### 기본 실행
```bash
# XGBoost (기본, 10% validation)
python train_and_predict_GBDT.py

# CatBoost 사용
python train_and_predict_GBDT.py --config config_GBDT.yaml

# Validation ratio 변경
python train_and_predict_GBDT.py --val-ratio 0.2

# 데이터 재처리 강제
python train_and_predict_GBDT.py --force-reprocess
```

### MixUp Data Augmentation
MixUp은 두 샘플을 선형 보간하여 새로운 학습 샘플을 생성하는 데이터 증강 기법입니다. 특히 불균형 데이터셋에서 효과적입니다.

```bash
# config_GBDT.yaml에서 설정
training:
  use_mixup: true      # MixUp 활성화
  mixup_alpha: 0.3     # Beta 분포 파라미터 (0.3 권장)
  mixup_ratio: 0.5     # 추가할 MixUp 샘플 비율 (0.5 = 50% 증가)
```

**MixUp 파라미터:**
- `alpha`: Beta(α, α) 분포에서 mixing coefficient λ를 샘플링
  - α=1.0: 균등한 mixing
  - α<1.0: 원본 샘플 선호 (0.3 권장)
  - α>1.0: 균형잡힌 mixing 선호
- `ratio`: 추가할 MixUp 샘플의 비율
  - 0.5: 원본의 50% 추가 (1.5배 데이터)
  - 1.0: 원본과 동일 개수 추가 (2배 데이터)

### 설정 파일 (config_GBDT.yaml)
```yaml
# 모델 선택
model:
  name: "xgboost"  # "xgboost" 또는 "catboost"

# Training 설정
training:
  val_ratio: 0.1
  force_reprocess: false

# XGBoost 설정
xgboost:
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 8
  subsample: 0.8
  colsample_bytree: 0.8
  tree_method: "gpu_hist"
  gpu_id: 0
  early_stopping_rounds: 20

# CatBoost 설정
catboost:
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 8
  task_type: "GPU"
  devices: "0"
  early_stopping_rounds: 20
```

### 출력 파일
```
result_GBDT_{model}/
├── workflow/             # NVTabular 전처리 파이프라인
├── *.parquet            # 전처리된 데이터
└── submission.csv       # 제출 파일
```

## 🧠 DNN 모델 사용법

### 멀티 GPU 훈련
```bash
# 4개 GPU 사용 (DDP)
python train_and_predict_dnn_ddp.py

# 특정 GPU 선택
CUDA_VISIBLE_DEVICES=0,1 python train_and_predict_dnn_ddp.py
```

### 주요 설정 (코드 내부)
```python
CFG = {
    'BATCH_SIZE': 1024,
    'EPOCHS': 5,
    'LEARNING_RATE': 1e-3,
    'NUM_DEVICES': 4,    # GPU 개수
    'STRATEGY': 'ddp',   # 분산 전략
    'VAL_RATIO': 0.1,
    'USE_MIXUP': True,   # MixUp 활성화
    'MIXUP_ALPHA': 0.3,  # Beta 분포 파라미터
    'MIXUP_PROB': 0.5    # 배치별 MixUp 적용 확률
}
```

### MixUp for DNN
DNN 모델은 온라인 MixUp을 사용하여 각 배치마다 확률적으로 데이터 증강을 수행합니다.

**DNN MixUp 특징:**
- **온라인 증강**: 학습 중 각 배치마다 실시간으로 MixUp 적용
- **확률적 적용**: `MIXUP_PROB`로 배치별 적용 확률 조절
- **수치형 피처 전용**: 현재 구현은 연속형 피처에만 MixUp 적용 (범주형/시퀀스는 원본 유지)
- **메모리 효율적**: 원본 데이터 크기 유지하면서 증강 효과

**권장 설정:**
- `MIXUP_ALPHA`: 0.2~0.4 (0.3 권장)
- `MIXUP_PROB`: 0.3~0.7 (0.5 권장)

## 🎯 하이퍼파라미터 최적화 (HPO)

Optuna를 사용하여 모델의 하이퍼파라미터를 자동으로 최적화합니다.

### XGBoost HPO
```bash
# 기본 실행
python hpo_xgboost.py --data-path data/train.parquet --n-trials 100 --val-ratio 0.2

# 빠른 테스트 (데이터 서브샘플링)
python hpo_xgboost.py --data-path data/train.parquet --n-trials 30 --subsample-ratio 0.1

# 시간 제한 설정
python hpo_xgboost.py --data-path data/train.parquet --n-trials 200 --timeout 28800  # 8시간
```

### CatBoost HPO
```bash
# GPU 사용
python hpo_catboost.py --data-path data/train.parquet --n-trials 100 --task-type GPU

# CPU 사용 (colsample_bylevel 포함)
python hpo_catboost.py --data-path data/train.parquet --n-trials 100 --task-type CPU

# NVTabular 처리된 데이터 사용
python hpo_catboost.py --data-path result_GBDT_catboost --n-trials 100
```

### DNN HPO
```bash
# 기본 실행
python hpo_dnn.py --train-path data/train.parquet --n-trials 50 --val-ratio 0.2

# 빠른 테스트 (서브샘플링)
python hpo_dnn.py --train-path data/train.parquet --n-trials 20 --subsample-ratio 0.1 --max-epochs 5

# 전체 최적화
python hpo_dnn.py --train-path data/train.parquet --n-trials 100 --max-epochs 15 --timeout 14400
```

### HPO 명령줄 인자

| 인자 | 설명 | 기본값 |
|------|------|--------|
| `--data-path` / `--train-path` | 데이터 경로 | 필수 |
| `--n-trials` | Optuna 시도 횟수 | 100 (GBDT), 50 (DNN) |
| `--val-ratio` | Validation 비율 | 0.2 |
| `--subsample-ratio` | 사용할 데이터 비율 | 1.0 |
| `--timeout` | 최대 실행 시간 (초) | None |
| `--max-epochs` | Trial당 최대 에포크 (DNN) | 10 |
| `--patience` | Early stopping patience (DNN) | 3 |

### 최적화되는 하이퍼파라미터

**XGBoost:**
- `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`, `max_bin`

**CatBoost:**
- `iterations`, `depth`, `learning_rate`, `subsample`, `l2_leaf_reg`
- `bootstrap_type`, `bagging_temperature`, `colsample_bylevel` (CPU), `border_count`

**DNN:**
- `batch_size`, `learning_rate`, `weight_decay`
- `emb_dim`, `lstm_hidden`, `cross_layers`, `n_layers`, `hidden_size`, `dropout`

### 최적화 후 사용

최적화가 완료되면 `config_*_optimized.yaml` 파일이 생성됩니다.

```bash
# GBDT 최적화 파라미터로 전체 학습
python train_and_predict_GBDT.py --config config_GBDT_optimized.yaml

# DNN은 생성된 config 파일을 코드에 반영하여 사용
```

### 예상 실행 시간

**XGBoost/CatBoost:**
- 빠른 테스트 (10% 데이터, 30 trials): 5-15분
- 중간 최적화 (30% 데이터, 100 trials): 30분-1시간
- 최종 최적화 (100% 데이터, 200 trials): 2-4시간

**DNN:**
- 빠른 테스트 (10% 데이터, 20 trials, 5 epochs): 10-20분
- 중간 최적화 (30% 데이터, 50 trials, 10 epochs): 1-2시간
- 최종 최적화 (100% 데이터, 100 trials, 15 epochs): 3-6시간

## 📊 데이터 준비

### 필수 데이터
```
data/
├── train.parquet    # 훈련 데이터 (10.7M rows)
└── test.parquet     # 테스트 데이터
```

### 데이터 분할 (Optional)
```bash
# 10-fold 데이터 분할
python dataset_split.py
```

이 스크립트는 `clicked` 값에 따라 데이터를 분할합니다:
- `clicked=1`: 모든 fold에 포함 (positive 샘플 보존)
- `clicked=0`: 10개 fold로 분할

## 🎯 피처 설명

### 범주형 피처 (5개)
- `gender`, `age_group`, `inventory_id`, `day_of_week`, `hour`

### 연속형 피처 (110개)
- `feat_a_*` (18개)
- `feat_b_*` (6개)
- `feat_c_*` (8개)
- `feat_d_*` (6개)
- `feat_e_*` (10개)
- `history_a_*` (7개)
- `history_b_*` (30개)
- `l_feat_*` (25개, `l_feat_20`, `l_feat_23` 제외)

### 시퀀스 피처 (DNN만 사용)
- `seq`: 가변 길이 시퀀스 데이터

### 제외 피처
- `l_feat_20`, `l_feat_23` (상수 값)

## 📈 평가 메트릭

대회 메트릭: `Score = 0.5 * AP + 0.5 * (1 / (1 + WLL))`

- **AP (Average Precision)**: 50% 가중치
- **WLL (Weighted LogLoss)**: 50% 가중치 (클래스 50:50 균형)

## 💾 메모리 및 성능

### GBDT 모델
- **GPU 메모리**: 10-14GB (RTX 3090 기준)
- **System RAM**: 32GB+ 권장
- **처리 속도**: 전체 데이터 단일 학습 약 10-30분

### DNN 모델
- **GPU 메모리**: 24GB per GPU (4 GPUs 권장)
- **System RAM**: 64GB+ 권장
- **처리 속도**: 5 epochs 약 30분-1시간

## 🔍 데이터 분석

`analysis/` 디렉토리에 데이터 분석 스크립트가 포함되어 있습니다. 자세한 내용은 `analysis/README.md`를 참조하세요.

**주요 분석 도구:**
- `chunk_eda.py`: 대용량 데이터 EDA (청크 단위 처리)
- `feature_quality_analysis.py`: 피처 품질 분석
- `compute_normalization_stats.py`: 표준화 통계 계산
- Missing pattern 분석 스크립트

## 📝 주요 업데이트

- **2025-10-08**: MixUp 데이터 증강 기법 추가 (GBDT/DNN 모두 지원)
- **2025-10-08**: 통합 환경 (environment.yaml) 구성 - RAPIDS + PyTorch 단일 환경
- **2025-01-08**: Cross-validation → Train/Val split 변경 (validation ratio 0.1)
- **2025-01-08**: 코드 정리 및 구조화, 공통 함수 통합
- **2024-12-31**: HPO (하이퍼파라미터 최적화) 스크립트 추가
- **2024-12-30**: GBDT 모델 (XGBoost/CatBoost) 추가
- **2024-12-29**: DNN 멀티 GPU (DDP) 지원 추가

## 🐛 문제 해결

### torch 모듈을 찾을 수 없는 경우
```bash
# 통합 환경 재설치
conda env remove -n toss-env
conda env create -f environment.yaml
conda activate toss-env
```

### GPU 메모리 부족
```bash
# GBDT: 데이터 서브샘플링 또는 작은 batch_size 사용
python train_and_predict_GBDT.py --val-ratio 0.1  # validation 비율 줄이기

# DNN: GPU 개수 조정 또는 batch size 감소
CUDA_VISIBLE_DEVICES=0,1 python train_and_predict_dnn_ddp.py
```

### cuDF string limit 에러
Raw parquet 파일 사용 시 자동으로 `seq` 컬럼이 제외됩니다.

## 🤝 참고사항

- 모든 스크립트는 단일 통합 환경 (`toss-env`)에서 실행됩니다
- GBDT와 DNN 모델 모두 동일한 conda 환경 사용
- GPU는 필수이며, CUDA 11.8+ 환경 권장
- 데이터 분석 도구는 `analysis/README.md` 참조
