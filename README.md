# Toss Click Prediction Challenge

클릭 예측 대회를 위한 머신러닝 프로젝트입니다. GBDT (XGBoost/CatBoost) 및 DNN (Deep Neural Network) 모델을 지원합니다.

## 🚀 빠른 시작

### 1. 환경 설정
```bash
conda env create -f environment.yaml
conda activate toss-env
```

### 2. 데이터 분할
```bash
# train.parquet → train_t (80%) / train_v (10%) / train_c (10%)
python dataset_split.py
```

### 3. 모델 학습
```bash
# GBDT (XGBoost)
python train_gbdt.py

# DNN (멀티 GPU)
python train_dnn_ddp.py
```

### 4. 예측
```bash
# GBDT (자동 calibration 포함)
python pred_gbdt.py --model-dir result_GBDT_xgboost/20231201_120000

# DNN (자동 calibration 포함)
python pred_dnn_ddp.py --model-dir result_dnn_ddp/20231201_120000
```

## 📁 프로젝트 구조

```
toss-challenge/
├── train_gbdt.py                # GBDT 모델 훈련
├── pred_gbdt.py                 # GBDT 모델 예측 + 자동 calibration
├── train_dnn_ddp.py             # DNN 멀티 GPU 훈련
├── pred_dnn_ddp.py              # DNN 모델 예측 + 자동 calibration
├── hpo_xgboost.py               # XGBoost HPO
├── hpo_catboost.py              # CatBoost HPO
├── hpo_dnn.py                   # DNN HPO
├── dataset_split.py             # 데이터 분할 (train/val/cal)
├── utils.py                     # 공통 유틸리티 함수
├── data_loader.py               # 데이터 로더
├── mixup.py                     # MixUp 데이터 증강
├── config_GBDT.yaml             # GBDT 설정
├── config_dnn_example.yaml      # DNN 설정 예시
├── environment.yaml             # Conda 환경
└── data/
    ├── train.parquet            # 원본 (10.7M rows)
    ├── train_t.parquet          # 훈련 (80%)
    ├── train_v.parquet          # 검증 (10%)
    ├── train_c.parquet          # 캘리브레이션 (10%)
    └── test.parquet             # 테스트
```

## 🛠️ GBDT 모델

### 학습
```bash
# XGBoost 기본 실행
python train_gbdt.py

# CatBoost 사용
python train_gbdt.py --preset catboost_deep

# 데이터 재처리
python train_gbdt.py --force-reprocess
```

### 예측
```bash
# 자동 calibration (기본값 - 권장)
python pred_gbdt.py --model-dir result_GBDT_xgboost/20231201_120000

# Calibration 비활성화
python pred_gbdt.py --model-dir result_GBDT_xgboost/20231201_120000 --no-calibration
```

### 설정 파일 (config_GBDT.yaml)
```yaml
model:
  name: "xgboost"  # "xgboost" or "catboost"

training:
  use_mixup: false
  mixup_alpha: 0.3
  mixup_ratio: 0.5

xgboost:
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 8
  tree_method: "gpu_hist"

catboost:
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 8
  task_type: "GPU"
```

## 🧠 DNN 모델

### 학습

**기본 실행:**
```bash
python train_dnn_ddp.py
```

**HPO 결과 활용 (권장):**
```bash
# 1. HPO 실행
python hpo_dnn.py --train-path data/train_t.parquet --n-trials 50

# 2. 최적 파라미터로 학습
python train_dnn_ddp.py --config config_dnn_optimized_best_params.yaml --epochs 20
```

**커맨드 라인 옵션:**
```bash
# 설정 override
python train_dnn_ddp.py --epochs 10 --learning-rate 0.0005 --no-mixup
```

### 예측
```bash
# 자동 calibration (기본값 - 권장)
python pred_dnn_ddp.py --model-dir result_dnn_ddp/20231201_120000

# Calibration 비활성화
python pred_dnn_ddp.py --model-dir result_dnn_ddp/20231201_120000 --no-calibration
```

### 커맨드 라인 옵션

**Training:**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--config` | YAML config 파일 (HPO 결과) | None |
| `--epochs` | 학습 epochs | 5 |
| `--batch-size` | Batch size | 1024 |
| `--learning-rate` | Learning rate | 0.001 |
| `--no-mixup` | MixUp 비활성화 | False |

**Prediction:**

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--model-dir` | 모델 디렉토리 | 필수 |
| `--test-path` | 테스트 데이터 경로 | data/test.parquet |
| `--no-calibration` | Calibration 비활성화 | False |
| `--batch-size` | 배치 사이즈 | 2048 |

### Config 파일 형식

**HPO 결과 (자동 생성):**
```yaml
best_score: 0.875432
best_params:
  learning_rate: 0.001234
  weight_decay: 0.000056
  emb_dim: 32
  lstm_hidden: 64
  n_layers: 3
  mixup_alpha: 0.3
```

**수동 작성:**
```yaml
LEARNING_RATE: 0.002
WEIGHT_DECAY: 0.0001
USE_MIXUP: true
MODEL:
  EMB_DIM: 64
  LSTM_HIDDEN: 128
  HIDDEN_UNITS: [1024, 512, 256]
  CROSS_LAYERS: 3
```

**Config 우선순위:** CLI args > YAML config > Default config

## 🎯 하이퍼파라미터 최적화 (HPO)

Optuna를 사용하여 자동으로 최적 하이퍼파라미터를 찾습니다.

### 사용법

```bash
# XGBoost
python hpo_xgboost.py --data-path data/train_fold1.parquet --n-trials 1000

# CatBoost
python hpo_catboost.py --data-path data/train_fold1.parquet --n-trials 1000 --task-type GPU

# DNN
python hpo_dnn.py --train-path data/train_fold1.parquet --n-trials 100
```

### HPO 후 사용

```bash
# GBDT
python train_gbdt.py --config config_GBDT_optimized.yaml

# DNN
python train_dnn_ddp.py --config config_dnn_optimized_best_params.yaml --epochs 30
```

### 최적화되는 파라미터

**XGBoost/CatBoost:**
- `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`
- `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`

**DNN:**
- `learning_rate`, `weight_decay`, `batch_size`
- `emb_dim`, `lstm_hidden`, `cross_layers`, `hidden_units`, `dropout`
- `mixup_alpha`, `mixup_prob`

## 📊 데이터 준비

### 데이터 분할 (필수)
```bash
python dataset_split.py
```

**생성 파일:**
- `data/train_t.parquet`: 훈련 (80%) - 모델 학습
- `data/train_v.parquet`: 검증 (10%) - Early stopping
- `data/train_c.parquet`: 캘리브레이션 (10%) - Prediction 시 calibration

**특징:**
- Stratified split (클래스 비율 유지)
- 데이터 누수 방지
- 재현성 보장

## 🎯 Calibration (자동 확률 보정)

**Prediction 시점에 자동으로 최적 calibration 방법 선택**

### 작동 방식

1. **train_c 분할**: Balanced set (50:50) + Test set (imbalanced)
2. **4가지 방법 테스트**: none, isotonic, sigmoid, temperature
3. **자동 선택**: Test set에서 가장 높은 score를 가진 방법 사용
4. **안전장치**: 성능 떨어지면 원본 사용 (none 선택)

```
train_c → Balanced fit (50:50) + Imbalanced test (~1% pos)
           ↓
  Test 4 methods: none, isotonic, sigmoid, temperature
           ↓
  Select best → Apply to test.parquet
```

### 사용 예시

```bash
# 자동 calibration (기본값 - 권장)
python pred_dnn_ddp.py --model-dir result_dnn_ddp/20231201_120000

# Calibration 비활성화
python pred_dnn_ddp.py --model-dir result_dnn_ddp/20231201_120000 --no-calibration
```

**출력 예시:**
```
🎯 Finding Best Calibration Method
...
   [NONE]       Score: 0.850000
   [ISOTONIC]   Score: 0.865000
   [SIGMOID]    Score: 0.868000
   [TEMPERATURE] Score: 0.872000 ← Optimal temperature: 1.2345

🏆 Best Method: TEMPERATURE (Improvement: +0.022000)
```

**장점:**
- ✅ 자동 최적화
- ✅ 학습 시간 단축
- ✅ 성능 떨어지면 원본 사용
- ✅ Balanced fitting으로 robust

## 💡 MixUp 데이터 증강

### GBDT MixUp
- 오프라인 증강: 학습 전 MixUp 샘플 생성
- Config에서 설정: `use_mixup: true`, `mixup_ratio: 0.5`

### DNN MixUp
- 온라인 증강: 배치마다 실시간 MixUp
- 확률적 적용: `MIXUP_PROB`로 배치별 적용 확률 조절
- 수치형 피처만 적용 (범주형/시퀀스는 원본 유지)

**권장 설정:**
- `MIXUP_ALPHA`: 0.3
- `MIXUP_PROB` (DNN): 0.5
- `mixup_ratio` (GBDT): 0.5

## 📈 피처 및 평가

### 피처
- **범주형** (5개): gender, age_group, inventory_id, day_of_week, hour
- **연속형** (110개): feat_a_*, feat_b_*, feat_c_*, history_*, l_feat_*
- **시퀀스** (DNN only): seq
- **제외**: l_feat_20, l_feat_23 (상수)

### 평가 메트릭
```
Score = 0.5 * AP + 0.5 * (1 / (1 + WLL))
```
- **AP**: Average Precision
- **WLL**: Weighted LogLoss (50:50 class balance)

## 💾 메모리 및 성능

### GBDT
- GPU 메모리: 10-14GB
- System RAM: 32GB+
- 학습 시간: 10-30분

### DNN
- GPU 메모리: 24GB per GPU (4 GPUs 권장)
- System RAM: 64GB+
- 학습 시간: 5 epochs → 30분-1시간

## 📝 주요 업데이트

- **2025-10-08**: **Calibration 자동 선택 시스템**
  - Prediction 시 4가지 방법 자동 테스트 및 best 선택
  - Training 시간 단축 (calibration 제거)
  - 성능 떨어지면 원본 사용
  
- **2025-10-08**: **DNN HPO 결과 활용**
  - YAML config로 HPO 결과 저장/로드
  - CLI args > YAML > Default 우선순위
  
- **2025-10-08**: **학습/예측 분리**
  - Training: 모델만 저장
  - Prediction: 자동 calibration + submission 생성
  
- **2025-10-08**: **데이터 분할 변경**
  - `dataset_split.py`로 train_t/train_v/train_c 생성
  - 재현성 향상 및 데이터 누수 방지
  
- **2025-10-08**: **MixUp 증강 추가**
  - GBDT: 오프라인 증강
  - DNN: 온라인 확률적 증강

## 🎯 고급 사용법

### HPO 워크플로우

```bash
# 1. 빠른 HPO (작은 데이터)
python hpo_dnn.py \
    --train-path data/train_t.parquet \
    --n-trials 50 \
    --subsample-ratio 0.3 \
    --max-epochs 5

# 2. 전체 데이터로 학습
python train_dnn_ddp.py \
    --config config_dnn_optimized_best_params.yaml \
    --epochs 30

# 3. 예측 (자동 calibration)
python pred_dnn_ddp.py --model-dir result_dnn_ddp/20231201_120000
```

### MixUp 설정

**GBDT (config_GBDT.yaml):**
```yaml
training:
  use_mixup: true
  mixup_alpha: 0.3
  mixup_ratio: 0.5  # 50% 샘플 추가
```

**DNN (config or CLI):**
```bash
python train_dnn_ddp.py --config hpo_results.yaml
# Config에 mixup_alpha, mixup_prob 포함
```

### Calibration 상세

**자동 선택 프로세스:**
1. train_c를 balanced fit set + imbalanced test set으로 분할
2. 4가지 방법(none, isotonic, sigmoid, temperature) 모두 fitting
3. Test set에서 평가하여 best 선택
4. Best method로 test.parquet 예측

**비활성화:**
```bash
python pred_dnn_ddp.py --model-dir result_dnn_ddp/20231201_120000 --no-calibration
```

## 🔍 데이터 분석

`analysis/` 디렉토리에 EDA 및 피처 분석 도구가 포함되어 있습니다. 자세한 내용은 `analysis/README.md`를 참조하세요.

**주요 도구:**
- `chunk_eda.py`: 대용량 데이터 EDA
- `feature_quality_analysis.py`: 피처 품질 분석
- `compute_normalization_stats.py`: 표준화 통계 계산

## 🐛 문제 해결

### 환경 문제
```bash
# 환경 재설치
conda env remove -n toss-env
conda env create -f environment.yaml
```

### GPU 메모리 부족
```bash
# GBDT: 자동 메모리 관리 (코드 내부)
python train_gbdt.py

# DNN: GPU 개수 또는 batch size 조정
CUDA_VISIBLE_DEVICES=0,1 python train_dnn_ddp.py
python train_dnn_ddp.py --batch-size 512
```

### cuDF string limit
자동으로 `seq` 컬럼이 제외됩니다 (NVTabular 처리 시).

## 🤝 참고사항

- **통합 환경**: GBDT와 DNN 모두 `toss-env` 사용
- **GPU 필수**: CUDA 11.8+ 권장
- **데이터 분할 필수**: `dataset_split.py` 먼저 실행
- **Calibration**: Prediction 시 자동 수행 (--no-calibration으로 비활성화 가능)
- **HPO**: 작은 데이터로 빠르게 실행 후 전체 데이터로 학습 권장

## 📚 상세 문서

- **데이터 분석**: `analysis/README.md`
- **환경 설정**: `environment.yaml` 주석 참조
- **설정 예시**: `config_GBDT.yaml`, `config_dnn_example.yaml`
