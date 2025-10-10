# Toss Click Prediction Challenge

클릭 예측 대회를 위한 머신러닝 프로젝트. GBDT (XGBoost/CatBoost) 및 DNN (WideDeepCTR) 모델 지원.

## 🚀 빠른 시작

### 1. 환경 설정
```bash
conda env create -f environment.yaml
conda activate toss-env
```

### 2. 데이터 전처리 (필수)
```bash
python dataset_split_and_preprocess.py
```

**생성 결과:**
```
data/proc_train_t/     # Training (80%)
data/proc_train_v/     # Validation (10%)
data/proc_train_c/     # Calibration (10%)
data/proc_train_hpo/   # HPO subset (~10% of train_t)
data/proc_test/        # Test data
```

**전처리 내용:**
- ✅ l_feat_20, l_feat_23 제거 (상수 피처)
- ✅ Continuous features standardization (mean=0, std=1)
- ✅ seq 결측치 처리 ('0.0')
- ✅ 공통 데이터로 GBDT/DNN 모두 지원

### 3. 모델 학습

**GBDT (XGBoost):**
```bash
python train_gbdt.py
```

**DNN (Multi-GPU):**
```bash
python train_dnn_ddp.py
```

### 4. 예측
```bash
# GBDT
python pred_gbdt.py --model-dir result_GBDT_xgboost/20231201_120000

# DNN
python pred_dnn_ddp.py --model-dir result_dnn_ddp/20231201_120000
```

## 📊 데이터 파이프라인

```
dataset_split_and_preprocess.py 실행
↓
data/proc_train_t/    (공통 전처리 데이터)
data/proc_train_v/    - l_feat_20, l_feat_23 제거
data/proc_train_c/    - continuous standardized
data/proc_train_hpo/  - seq 결측치 처리 ('0.0')
data/proc_test/       - seq 포함
↓
├─ GBDT (train_gbdt.py, hpo_xgboost.py)
│  └─ load_processed_data_gbdt(path)
│     → seq 자동 제거됨, categorical encoded, continuous standardized
│
└─ DNN (train_dnn_ddp.py, hpo_dnn.py)
   └─ load_processed_dnn_data(path)
      → seq 포함, continuous standardized, categorical raw
```

**장점:**
- ✅ 1회 전처리로 모든 모델 지원
- ✅ 통계 일관성 보장 (전체 train으로 fit)
- ✅ 디스크 절약 (중복 데이터 없음)
- ✅ 빠른 로딩 (pre-processed)

## 🔧 Hyperparameter Optimization

### XGBoost HPO
```bash
python hpo_xgboost.py \
  --n-trials 100 \
  --use-hpo-subset  # proc_train_hpo 사용 (빠름)
```

### DNN HPO
```bash
python hpo_dnn.py \
  --n-trials 50 \
  --use-mixup
```

**결과:**
- `config_optimized.yaml` (GBDT)
- `config_widedeep_optimized_best_params.yaml` (DNN)

## 📁 주요 파일

```
toss-challenge/
├── dataset_split_and_preprocess.py  # 데이터 전처리 (필수)
├── train_gbdt.py                    # GBDT 학습
├── train_dnn_ddp.py                 # DNN 학습 (Multi-GPU)
├── pred_gbdt.py                     # GBDT 예측 + calibration
├── pred_dnn_ddp.py                  # DNN 예측 + calibration
├── hpo_xgboost.py                   # XGBoost HPO
├── hpo_dnn.py                       # DNN HPO
├── data_loader.py                   # 데이터 로더
├── utils.py                         # 공통 함수
├── mixup.py                         # MixUp 증강
├── config_GBDT.yaml                 # GBDT 설정
└── data/
    ├── train.parquet                # 원본 (10.7M rows)
    ├── test.parquet                 # 테스트
    ├── proc_train_t/                # 전처리 완료
    ├── proc_train_v/
    ├── proc_train_c/
    ├── proc_train_hpo/
    └── proc_test/
```

## ⚙️ 설정 파일

### GBDT (config_GBDT.yaml)
```yaml
data:
  train_t_path: "data/proc_train_t"
  train_v_path: "data/proc_train_v"
  train_c_path: "data/proc_train_c"

model:
  name: "xgboost"  # or "catboost"

xgboost:
  n_estimators: 200
  learning_rate: 0.1
  max_depth: 8
  use_mixup: true
  mixup_alpha: 0.6
  mixup_ratio: 0.3
```

### DNN
커맨드 라인 또는 HPO 결과 YAML 사용:
```bash
python train_dnn_ddp.py --config hpo_results.yaml --epochs 5
```

## 📈 피처 및 평가

### 피처
- **Categorical** (4): gender, age_group, inventory_id, l_feat_14
- **Continuous** (110): feat_a_*, feat_b_*, feat_c_*, history_*, l_feat_*
- **Sequence** (DNN only): seq
- **제외**: l_feat_20, l_feat_23 (상수 피처)

### 평가 메트릭
```
Score = 0.5 × AP + 0.5 × (1 / (1 + WLL))
```
- AP: Average Precision
- WLL: Weighted LogLoss (50:50 class balance)

## 🎯 고급 기능

### Calibration (자동)
Prediction 시 4가지 방법 자동 테스트하여 best 선택:
- none, isotonic, sigmoid, temperature
- 성능 개선 없으면 원본 사용

```bash
# 자동 (기본)
python pred_gbdt.py --model-dir result_GBDT_xgboost/xxx

# 비활성화
python pred_gbdt.py --model-dir result_GBDT_xgboost/xxx --no-calibration
```

### MixUp 증강
- **GBDT**: 오프라인 (학습 전)
- **DNN**: 온라인 (배치마다)

## 💾 시스템 요구사항

### GBDT
- GPU: 10-14GB
- RAM: 32GB+
- 학습: 10-30분

### DNN
- GPU: 24GB × 4개 권장
- RAM: 64GB+
- 학습: 30분-1시간 (5 epochs)

## 🐛 문제 해결

### 전처리 중 GPU 메모리 부족
```bash
# 1. GPU 선택 (가장 큰 메모리를 가진 GPU 사용)
CUDA_VISIBLE_DEVICES=0 python dataset_split_and_preprocess.py

# 2. 스크립트는 자동으로 다음과 같이 최적화됨:
#    - RMM pool: 8GB
#    - Part size: 128MB (메모리 효율)
#    - Out files: 4 (파일 수 감소)
#    - 각 단계마다 GPU 메모리 정리
```

### 학습 중 GPU 메모리 부족
```bash
# DNN: GPU 개수 조정
CUDA_VISIBLE_DEVICES=0,1 python train_dnn_ddp.py

# DNN: Batch size 감소
python train_dnn_ddp.py --batch-size 256
```

### 환경 재설치
```bash
conda env remove -n toss-env
conda env create -f environment.yaml
```

## 📚 상세 문서

- **데이터 분석**: `analysis/README.md`
- **환경 설정**: `environment.yaml`
- **설정 예시**: `config_GBDT.yaml`

## 📝 주요 특징

- ✅ 통합 전처리 파이프라인 (1회 실행, 모든 모델 지원)
- ✅ GPU 가속 (GBDT/DNN 모두)
- ✅ 자동 calibration (prediction 시)
- ✅ HPO 지원 (Optuna)
- ✅ MixUp 데이터 증강
- ✅ Multi-GPU DNN 학습 (DDP)
- ✅ 메모리 최적화
