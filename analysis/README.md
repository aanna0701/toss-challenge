# Toss Click Prediction - Data Analysis Scripts

이 디렉토리는 Toss Click Prediction 프로젝트의 데이터 분석 스크립트들을 포함합니다. 대용량 데이터(1천만+ 행)를 효율적으로 분석하기 위한 청크 단위 처리와 Missing Pattern 분석을 제공합니다.

## 📁 디렉토리 구조

```
analysis/
├── README.md                              # 이 파일
├── chunk_eda.py                           # 메인 EDA 스크립트 (청크 단위)
├── chunk_utils.py                         # 청크 처리 유틸리티
├── eda_utils.py                           # EDA 유틸리티 함수들
├── feature_quality_analysis.py            # 피처 품질 분석 스크립트
├── compute_normalization_stats.py         # 표준화 통계 계산 스크립트
├── quick_eda.py                           # 빠른 EDA (샘플링 기반)
├── missing_pattern_analysis.py           # Missing pattern 분석
├── missing_overlap_analysis.py           # Missing pattern 중복 분석
├── validate_missing_patterns_with_data.py # 실제 데이터로 missing pattern 검증
├── pattern1_detailed_analysis.py         # Pattern 1 상세 분석
├── validate_all_patterns.py              # 모든 pattern 검증
├── extreme_features_analysis.py          # 극단적 features 분석
├── validate_extreme_features_with_data.py # 실제 데이터로 극단적 features 검증
└── results/                               # 분석 결과 저장 디렉토리
    ├── chunk_eda_results.json
    ├── chunk_eda_summary.png
    ├── feature_quality_analysis.json
    ├── feature_quality_analysis.png
    ├── normalization_stats.json
    ├── missing_pattern_*.json
    ├── all_patterns_validation.json
    └── extreme_features_*.json
```

## 🚀 주요 스크립트 및 사용법

### 1. 메인 EDA (Exploratory Data Analysis)

#### `chunk_eda.py` - 대용량 데이터 EDA
```bash
python analysis/chunk_eda.py
```

**목적:** 
- 1천만+ 행의 대용량 데이터를 메모리 효율적으로 분석
- 청크 단위(10만 행씩) 처리로 메모리 부족 문제 해결

**결과물:**
- `results/chunk_eda_results.json`: 전체 통계 결과
- `results/chunk_eda_summary.png`: 시각화 요약
- 기본 통계, 결측값 통계, 카테고리별/수치형 변수 분포

**주요 기능:**
- 온라인 통계 계산 (Welford's algorithm)
- 카테고리별 빈도 계산
- 결측값 패턴 분석
- 자동 시각화 생성

#### `quick_eda.py` - 빠른 샘플링 기반 EDA
```bash
python analysis/quick_eda.py
```

**목적:** 
- 빠른 데이터 탐색을 위한 샘플링 기반 EDA
- 초기 데이터 이해용

**결과물:**
- 샘플 기반 기본 통계
- 빠른 분포 확인

### 2. 표준화 통계 계산

#### `compute_normalization_stats.py` - 표준화 통계 계산
```bash
python analysis/compute_normalization_stats.py
```

**목적:**
- 전체 train set에서 각 피처의 mean/std를 계산
- 표준화(normalization)에 사용할 통계값 제공
- 메모리 효율적인 청크 단위 처리

**주요 기능:**
- **메모리 효율적**: 1천만 행 이상의 대용량 데이터를 청크 단위로 처리
- **자동 제외**: ID, target, sequence 컬럼은 자동으로 제외
- **완전한 통계**: mean, std, min, max, count 정보 제공
- **진행률 표시**: 처리 진행 상황을 실시간으로 확인 가능

**결과물:**
- `results/normalization_stats.json`: 계산된 통계 결과
- 112개 피처의 표준화 통계 (ID, clicked, seq 제외)

**사용법:**
```python
import json

# 통계 로드
with open('analysis/results/normalization_stats.json', 'r') as f:
    stats = json.load(f)

# 특정 피처의 mean, std 가져오기
feature_name = 'l_feat_1'
mean = stats['statistics'][feature_name]['mean']
std = stats['statistics'][feature_name]['std']

# 데이터 표준화
normalized_data = (raw_data - mean) / std

# 역표준화 (예측 결과를 원래 스케일로)
denormalized_data = normalized_data * std + mean
```

**PyTorch에서 사용:**
```python
import torch

# 통계를 텐서로 변환
mean_tensor = torch.tensor(mean, dtype=torch.float32)
std_tensor = torch.tensor(std, dtype=torch.float32)

# 배치 데이터 표준화
normalized_batch = (batch_data - mean_tensor) / std_tensor
```

### 3. 피처 품질 분석

#### `feature_quality_analysis.py` - 피처 품질 분석
```bash
python analysis/feature_quality_analysis.py [--chunk_size 100000] [--data_path ./train.parquet]
```

**목적:**
- 피처별 값 분포 분석으로 분별력이 낮은 피처 식별
- 피처별 클릭률과의 상관관계 분석으로 연관성이 낮은 피처 식별

**주요 기능:**

**수치형 피처 분포 품질 분석:**
- **상수 피처**: 모든 값이 동일한 피처 (표준편차 = 0)
- **낮은 분산 피처**: 변동계수(CV) < 0.01인 피처
- **극단적 Sparse 피처**: 99% 이상이 0값인 피처

**수치형 피처 상관관계 품질 분석:**
- **거의 0인 상관관계**: |correlation| < 0.0001인 피처
- **매우 낮은 상관관계**: |correlation| < 0.001인 피처
- **음의 상관관계**: correlation < -0.001인 피처

**범주형 피처 품질 분석 (ANOVA 기반):**
- **낮은 CTR 편차**: 카테고리별 클릭률 편차가 낮은 피처
- **지배적 카테고리**: 한 카테고리가 95% 이상인 피처
- **적은 카테고리**: 카테고리가 2개 이하인 피처
- **낮은 ANOVA F-statistic**: 카테고리 간 유의미한 차이가 없는 피처

**결과물:**
- `results/feature_quality_analysis.json`: 상세 분석 결과
- `results/feature_quality_analysis.png`: 시각화 결과
- 문제 피처 목록 및 통계 요약

**분석 결과 활용:**
- 수치형/범주형 피처 모두에 대한 종합적인 품질 평가
- 모델링에서 제거할 피처 후보 식별
- 피처 선택 전략 수립 (수치형: 상관관계, 범주형: ANOVA 기반)
- 데이터 품질 개선 방향 제시

### 4. Missing Pattern 분석

#### `missing_pattern_analysis.py` - Missing 패턴 식별
```bash
python analysis/missing_pattern_analysis.py
```

**목적:**
- 동일한 missing count를 가진 feature 그룹 식별
- Missing pattern의 규칙성 분석

**결과물:**
- `results/missing_pattern_analysis.json`: 패턴별 feature 그룹
- 3가지 주요 패턴 식별:
  - Pattern 1: 17,208 missing (77개 features)
  - Pattern 2: 18,598 missing (18개 features - feat_a_*)
  - Pattern 3: 1,085,557 missing (1개 feature - feat_e_3)

#### `missing_overlap_analysis.py` - Missing 패턴 중복 분석
```bash
python analysis/missing_overlap_analysis.py
```

**목적:**
- Missing pattern들 간의 중복 관계 분석
- 데이터 분할 전략 수립

**결과물:**
- `results/missing_overlap_*.json`: 중복 관계 분석
- 계층적 포함 관계 확인: Pattern 3 ⊃ Pattern 2 ⊃ Pattern 1

#### `validate_all_patterns.py` - 실제 데이터로 패턴 검증
```bash
conda activate toss-click-prediction-cpu
python analysis/validate_all_patterns.py
```

**목적:**
- 실제 데이터로 missing pattern 가설 검증
- 동일한 행에서 해당 features들이 모두 missing인지 확인

**결과물:**
- `results/all_patterns_validation.json`: 검증 결과
- **확인된 사실**: 각 패턴별로 동일한 행에서 모든 features가 동시에 missing

### 5. 극단적 Features 분석

#### `extreme_features_analysis.py` - 치우친 features 식별
```bash
python analysis/extreme_features_analysis.py
```

**목적:**
- 극단적으로 치우친 features 식별
- 모델링에 불필요한 features 제거 후보 선정

**결과물:**
- `results/extreme_features_analysis.json`: 극단적 features 목록
- 심각도별 분류 (EXTREME, HIGH, MODERATE)

#### `validate_extreme_features_with_data.py` - 실제 데이터로 검증
```bash
conda activate toss-click-prediction-cpu
python analysis/validate_extreme_features_with_data.py
```

**목적:**
- 실제 데이터로 극단적 features 검증
- 정확한 분포 확인

**결과물:**
- **즉시 제거 권장**: `l_feat_20`, `l_feat_23` (상수)
- **조건부 제거**: `l_feat_8` (99.4% 치우침)

## 📊 핵심 발견사항

### Missing Pattern 분석 결과
1. **체계적인 Missing Pattern**: 3가지 명확한 패턴 존재
2. **동시 Missing**: 각 패턴별로 동일한 행에서 모든 features가 동시에 missing
3. **계층적 구조**: Pattern 3이 가장 큰 영향, Pattern 1, 2는 부분적 중복
4. **데이터 활용률**: 전체 데이터의 89.86%에서 모든 features 사용 가능

### Feature Quality 분석 결과
1. **상수 Features**: 2개 features (`l_feat_20`, `l_feat_23`) - 즉시 제거
2. **극단적 치우침**: 1개 feature (`l_feat_8`) - 제거 고려
3. **정상 분포**: 대부분의 features는 모델링에 적합

## 🛠️ 유틸리티 모듈

### `chunk_utils.py`
- **OnlineStats**: 온라인 통계 계산 클래스
- **ChunkProcessor**: 청크 단위 데이터 처리
- **MemoryEfficientCounter**: 메모리 효율적 카운터

### `eda_utils.py`
- **시각화 함수들**: 분포 플롯, 히스토그램 등
- **통계 요약 함수들**: 기술통계, 상관관계 등

## 📈 모델링 권장사항

### 1. Feature Selection 전략
```python
# 즉시 제거
remove_features = ['l_feat_20', 'l_feat_23']

# 조건부 제거 (성능 테스트 후)
consider_remove = ['l_feat_8', 'feat_e_3']
```

### 2. Missing Pattern 기반 모델링 전략

**전략 1: 단일 모델 + Feature Selection**
- 가장 간단하고 효과적
- feat_e_3 제거 시 99.83% 데이터 활용 가능

**전략 2: 계층적 모델링**
- Level 1: 모든 features (89.86% 데이터)
- Level 2: feat_e_3 제외 (99.83% 데이터)
- Level 3: feat_a_* 제외 (99.84% 데이터)

**전략 3: 앙상블 모델링**
- 각 missing pattern별 모델 구축
- 복잡하지만 최고 성능 기대

## 🔧 환경 요구사항

```bash
# Conda 환경 활성화 (실제 데이터 검증시)
conda activate toss-click-prediction-cpu

# 필요 패키지
pandas>=1.3.0
numpy>=1.21.0
pyarrow>=5.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## ⚠️ 주의사항

1. **메모리 사용량**: 대용량 데이터 처리시 메모리 부족 가능
2. **처리 시간**: 전체 EDA는 수십 분 소요 가능
3. **환경 의존성**: 실제 데이터 검증은 conda 환경 필요

## 📝 사용 예시

```bash
# 1. 기본 EDA 실행
python analysis/chunk_eda.py

# 2. 표준화 통계 계산
python analysis/compute_normalization_stats.py

# 3. 피처 품질 분석 실행
python analysis/feature_quality_analysis.py

# 4. Missing pattern 분석
python analysis/missing_pattern_analysis.py

# 5. 실제 데이터로 검증 (conda 환경에서)
conda activate toss-click-prediction-cpu
python analysis/validate_all_patterns.py
python analysis/validate_extreme_features_with_data.py

# 6. 결과 확인
ls analysis/results/
```

## 🎯 결과 활용

1. **Feature Engineering**: Missing pattern 정보로 새로운 features 생성
2. **Data Preprocessing**: 불필요한 features 제거
3. **Model Architecture**: Missing pattern 기반 모델 설계
4. **Performance Optimization**: 데이터 분할 전략 수립

### 피처 품질 분석 결과 활용 예시
```python
import json

# 분석 결과 로드
with open('analysis/results/feature_quality_analysis.json', 'r') as f:
    results = json.load(f)

# 제거할 피처 목록 추출
remove_features = []
remove_features.extend([f['feature'] for f in results['distribution_issues']['constant']])
remove_features.extend([f['feature'] for f in results['correlation_issues']['zero_correlation']])

# 범주형 피처도 포함 (있는 경우)
if 'categorical_issues' in results:
    remove_features.extend([f['feature'] for f in results['categorical_issues']['low_anova_f']])

print(f"제거 권장 피처: {len(remove_features)}개")
print(f"수치형 - 상수 피처: {results['summary']['constant_features']}개")
print(f"수치형 - 상관관계 거의 0인 피처: {results['summary']['zero_correlation_features']}개")
if 'low_anova_f_categorical' in results['summary']:
    print(f"범주형 - 낮은 ANOVA F 피처: {results['summary']['low_anova_f_categorical']}개")
```

---

**작성자**: AI Assistant  
**최종 업데이트**: 2024년  
**프로젝트**: Toss Click Prediction Challenge
