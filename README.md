# Toss Click Prediction Project

딥러닝을 이용한 클릭 예측 프로젝트입니다. **TabularSeq 모델**과 **TabularTransformer 모델**을 지원합니다.

## 🚀 주요 특징

- ✅ **두 가지 모델 지원**: TabularSeq (기존) + TabularTransformer (신규)
- ✅ **고급 피처 처리**: 범주형/수치형/시퀀스 피처 분리 처리
- ✅ **Transformer 아키텍처**: FT-Transformer 기반 테이블 데이터 모델
- ✅ **누락값 처리**: NaN 토큰을 통한 학습 가능한 누락값 처리
- ✅ **메모리 효율적**: 대용량 데이터 샘플링 및 청크 처리
- ✅ **완전 자동화**: 훈련 → 예측 → 결과 저장 원클릭 실행

## 📁 프로젝트 구조

```
├── main.py                    # 메인 설정 및 초기화
├── utils.py                   # 유틸리티 함수들
├── config.yaml               # 설정 파일 (TabularSeq + Transformer)
├── data_loader.py            # 데이터 로더 (모델별 분기)
├── model.py                  # 모델 정의 (TabularSeq + Transformer)
├── train.py                  # 모델 훈련 (모델별 분기)
├── predict.py                # 예측 및 제출 (모델별 분기)
├── train_and_predict.py      # 🆕 원클릭 훈련+예측 워크플로우
├── metrics.py                # 평가 메트릭 (AP, WLL, Score)
├── early_stopping.py         # 조기 종료 기능
├── gradient_norm.py          # 그래디언트 모니터링
├── analysis/                 # 데이터 분석
│   ├── chunk_eda.py         # 청크 단위 EDA (전체 데이터)
│   ├── chunk_utils.py       # 청크 분석 유틸리티
│   ├── compute_normalization_stats.py  # 정규화 통계 계산
│   ├── feature_quality_analysis.py     # 피처 품질 분석
│   ├── eda_utils.py         # EDA 유틸리티 함수
│   └── results/             # 분석 결과
│       ├── chunk_eda_results.json      # 청크 EDA 결과
│       ├── normalization_stats.json    # 정규화 통계
│       └── feature_quality_analysis.json  # 피처 품질 분석
├── requirements.txt          # Pip 패키지 목록
├── environment-cpu.yml       # Conda 환경 (CPU)
├── setup_env.sh              # 환경 설정 스크립트 (Linux/macOS)
├── setup_env.bat             # 환경 설정 스크립트 (Windows)
└── README.md                 # 사용 가이드
```

## 🛠️ 설치

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
# CPU 전용 버전
conda env create -f environment-cpu.yml
conda activate toss-click-prediction-cpu
```

### Option 2: Pip
```bash
pip install -r requirements.txt
```

## 🚀 사용 방법

### 1. 원클릭 실행 (추천)

```bash
# 훈련 → 예측 → 결과 저장을 한 번에 실행
python train_and_predict.py
```

### 2. 단계별 실행

#### 데이터 분석
```bash
# 전체 데이터 청크 분석
python analysis/chunk_eda.py

# 피처 품질 분석
python analysis/feature_quality_analysis.py

# 정규화 통계 계산
python analysis/compute_normalization_stats.py
```

#### 모델 훈련
```bash
# 기본 설정으로 훈련
python train.py

# 예측 및 제출 파일 생성
python predict.py
```

### 3. 모델 선택

`config.yaml`에서 모델 타입을 선택할 수 있습니다:

```yaml
MODEL:
  TYPE: "tabular_transformer"  # 또는 "tabular_seq"
  
  # TabularSeq 모델 설정
  LSTM_HIDDEN: 64
  HIDDEN_UNITS: [256, 128]
  DROPOUT: 0.2
  
  # TabularTransformer 모델 설정
  TRANSFORMER:
    HIDDEN_DIM: 192
    N_HEADS: 8
    N_LAYERS: 3
    FFN_SIZE_FACTOR: 1.333
    ATTENTION_DROPOUT: 0.2
    FFN_DROPOUT: 0.1
    RESIDUAL_DROPOUT: 0.0
    LSTM_HIDDEN: 32
```

## 🧠 모델 아키텍처

### TabularSeq 모델 (기존)
- **구조**: LSTM + MLP
- **입력**: 수치형 피처 + 시퀀스 피처
- **특징**: 간단하고 빠른 훈련

### TabularTransformer 모델 (신규)
- **구조**: Transformer Encoder + LSTM + MLP
- **입력**: 범주형 + 수치형 + 시퀀스 피처 (분리 처리)
- **특징**: 
  - 범주형 피처: Embedding (0 ~ cardinality-1)
  - 수치형 피처: Linear Projection + 표준화
  - 시퀀스 피처: LSTM + Projection
  - 누락값: 학습 가능한 NaN 토큰
  - Column Embeddings + Class Token
  - 3-layer Transformer (192 dim, 8 heads)

## 📊 데이터 처리

### 피처 분류 (자동)
- **범주형**: `gender`, `age_group`, `inventory_id`, `day_of_week`, `hour`
- **수치형**: 나머지 모든 피처 (제외: 범주형, 시퀀스, ID, target)
- **시퀀스**: `seq` (문자열 파싱)
- **제외**: `l_feat_20`, `l_feat_23` (상수 피처)

### 전처리 파이프라인
1. **범주형**: 고유값 정렬 → 0부터 연속 정수 매핑
2. **수치형**: Z-score 표준화 (미리 계산된 통계 사용)
3. **시퀀스**: 문자열 파싱 → 패딩 → LSTM 처리
4. **누락값**: NaN 마스크 생성 → NaN 토큰으로 대체

## 📈 평가 메트릭

- **AP (Average Precision)**: 50% 가중치
- **WLL (Weighted LogLoss)**: 50% 가중치  
- **Score**: `0.5 * AP + 0.5 * (1 / (1 + WLL))`

## 📁 출력 파일

### 훈련 결과
- `trained_model_{datetime}.pth`: 훈련된 모델
- `baseline_submit_{datetime}.csv`: 제출 파일
- `metadata_{datetime}.json`: 메타데이터

### 로그 파일
- `train_logs.csv`: 훈련 로그
- `gradient_norms.csv`: 그래디언트 모니터링

### 분석 결과 (`analysis/results/`)
- `chunk_eda_results.json`: 전체 데이터 분석
- `normalization_stats.json`: 정규화 통계
- `feature_quality_analysis.json`: 피처 품질 분석

## ⚙️ 설정 옵션

### 데이터 샘플링
```yaml
DATA:
  USE_SAMPLING: true
  SAMPLE_SIZE: 1000000  # 샘플 크기
```

### 조기 종료
```yaml
EARLY_STOPPING:
  ENABLED: true
  PATIENCE: 5
  MONITOR: "val_score"
```

### 그래디언트 모니터링
```yaml
GRADIENT_NORM:
  ENABLED: true
  COMPONENTS: ["lstm", "mlp", "total"]
```

## 🔧 주요 기능

- ✅ **모듈화된 구조**: 각 기능별로 파일 분리
- ✅ **YAML 설정 관리**: 설정 파일로 쉬운 실험
- ✅ **설정 검증**: 잘못된 키 자동 감지
- ✅ **메모리 효율적**: 대용량 데이터 청크 처리
- ✅ **재현성**: 시드 고정으로 일관된 결과
- ✅ **자동화**: 원클릭 훈련+예측 워크플로우
- ✅ **모니터링**: 그래디언트 및 훈련 상태 추적

## 🚨 주의사항

1. **메모리 사용량**: 대용량 데이터 처리 시 충분한 RAM 필요
2. **GPU 사용**: Transformer 모델은 GPU에서 더 효율적
3. **피처 분류**: `chunk_eda_results.json` 기반으로 자동 분류
4. **정규화 통계**: `normalization_stats.json` 필요 (자동 생성)

## 📚 참고 논문

- **FT-Transformer**: "Revisiting Deep Learning Models for Tabular Data" (NeurIPS 2021)
- **Tabular Data**: Transformer 아키텍처를 테이블 데이터에 적용한 연구

## 🤝 기여

이슈나 개선사항이 있으면 언제든지 알려주세요!