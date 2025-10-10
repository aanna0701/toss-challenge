#!/usr/bin/env python3
"""
train.parquet 분석: 각 feature가 categorical인지 numerical인지 자동 판별

판별 기준:
1. 정수만 있음 + 고유값 개수 적음 (< 50) → categorical
2. 소수 있음 → numerical
3. 정수만 있음 + 고유값 많음 (>= 50) → numerical (ID 같은 경우)

출력: JSON 파일로 저장
"""
import cudf
import pandas as pd
import numpy as np
import json
from pathlib import Path

print("=" * 80)
print("🔍 Feature Type Analysis: Categorical vs Numerical")
print("=" * 80)

# Load train.parquet (first 100K rows for type analysis)
print("\n📖 Loading train.parquet (first 100K rows for analysis)...")
print("   Note: 100K rows is sufficient for determining data types")

# Use cudf with row_groups to avoid OOM
try:
    gdf = cudf.read_parquet('data/train.parquet', num_rows=100000)
    df = gdf.to_pandas()
    print(f"✅ Loaded: {len(df):,} rows x {len(df.columns)} columns\n")
except Exception as e:
    print(f"❌ cudf failed: {e}")
    print("   Trying dask-cudf...")
    import dask_cudf
    ddf = dask_cudf.read_parquet('data/train.parquet', split_row_groups=True)
    # Take first 100K rows
    gdf = ddf.head(100000, npartitions=-1)
    df = gdf.to_pandas()
    print(f"✅ Loaded via dask-cudf: {len(df):,} rows x {len(df.columns)} columns\n")

# 분석 결과 저장
results = {
    'metadata': {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'analysis_date': pd.Timestamp.now().isoformat()
    },
    'categorical': [],
    'numerical': [],
    'special': [],  # clicked, seq, ID 등
    'details': {}
}

# 특수 컬럼 (분석에서 제외)
SPECIAL_COLS = ['clicked', 'seq', 'ID']

print("🔬 Analyzing each feature...\n")
print(f"{'Column':<20} {'Type':<12} {'Unique':<10} {'Has Float':<12} {'Classification':<15}")
print("-" * 80)

for col in df.columns:
    if col in SPECIAL_COLS:
        results['special'].append(col)
        results['details'][col] = {
            'classification': 'special',
            'reason': 'Special column (target, sequence, or ID)'
        }
        print(f"{col:<20} {'special':<12} {'-':<10} {'-':<12} {'special':<15}")
        continue
    
    # 기본 정보
    dtype = str(df[col].dtype)
    n_unique = df[col].nunique()
    n_missing = df[col].isna().sum()
    
    # NaN 제거한 데이터로 분석
    non_null_data = df[col].dropna()
    
    if len(non_null_data) == 0:
        # 모두 NaN인 경우
        results['numerical'].append(col)
        results['details'][col] = {
            'classification': 'numerical',
            'reason': 'All NaN - default to numerical',
            'dtype': dtype,
            'n_unique': n_unique,
            'n_missing': n_missing,
            'missing_ratio': n_missing / len(df)
        }
        print(f"{col:<20} {'all_nan':<12} {n_unique:<10} {'-':<12} {'numerical':<15}")
        continue
    
    # Float 값이 있는지 확인
    has_float = False
    try:
        # 숫자형으로 변환 시도
        numeric_data = pd.to_numeric(non_null_data, errors='coerce')
        non_null_numeric = numeric_data.dropna()
        
        if len(non_null_numeric) > 0:
            # 정수와 비교해서 차이가 있으면 float
            has_float = not (non_null_numeric == non_null_numeric.astype(int)).all()
    except:
        has_float = True  # 변환 실패하면 float로 간주
    
    # 분류 기준
    classification = None
    reason = None
    
    if has_float:
        # 소수점 있음 → numerical
        classification = 'numerical'
        reason = 'Has floating point values'
    elif n_unique < 50:
        # 정수만 있고 고유값 적음 → categorical
        classification = 'categorical'
        reason = f'Integer only with {n_unique} unique values'
    else:
        # 정수만 있지만 고유값 많음 → numerical (ID 같은 경우)
        classification = 'numerical'
        reason = f'Integer only but {n_unique} unique values (likely ID or continuous)'
    
    # 결과 저장
    if classification == 'categorical':
        results['categorical'].append(col)
    else:
        results['numerical'].append(col)
    
    results['details'][col] = {
        'classification': classification,
        'reason': reason,
        'dtype': dtype,
        'n_unique': n_unique,
        'n_missing': n_missing,
        'missing_ratio': n_missing / len(df),
        'has_float': has_float
    }
    
    print(f"{col:<20} {dtype:<12} {n_unique:<10} {str(has_float):<12} {classification:<15}")

# 결과 요약
print("\n" + "=" * 80)
print("📊 Analysis Summary")
print("=" * 80)

print(f"\n📌 Categorical Features ({len(results['categorical'])}개):")
for col in sorted(results['categorical']):
    detail = results['details'][col]
    print(f"  - {col:<20} (unique: {detail['n_unique']}, {detail['reason']})")

print(f"\n📈 Numerical Features ({len(results['numerical'])}개):")
# 너무 많으면 처음 20개만
if len(results['numerical']) > 20:
    print(f"  (showing first 20 out of {len(results['numerical'])})")
    for col in sorted(results['numerical'])[:20]:
        detail = results['details'][col]
        has_float_str = "float" if detail['has_float'] else "int"
        print(f"  - {col:<20} ({has_float_str}, unique: {detail['n_unique']})")
    print(f"  ... and {len(results['numerical']) - 20} more")
else:
    for col in sorted(results['numerical']):
        detail = results['details'][col]
        has_float_str = "float" if detail['has_float'] else "int"
        print(f"  - {col:<20} ({has_float_str}, unique: {detail['n_unique']})")

print(f"\n🔖 Special Columns ({len(results['special'])}개):")
for col in results['special']:
    print(f"  - {col}")

# JSON 저장
output_path = Path('analysis/results/feature_type_analysis.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_path}")

# Python 코드로 출력 (dataset_split_and_preprocess.py에 바로 사용 가능)
print("\n" + "=" * 80)
print("🐍 Python Code (copy to dataset_split_and_preprocess.py)")
print("=" * 80)

print("\n# CATEGORICAL COLUMNS")
print(f"categorical_cols = {results['categorical']}")

print("\n# NUMERICAL COLUMNS (for continuous feature list)")
numerical_cols = sorted([c for c in results['numerical'] if c not in SPECIAL_COLS])
print(f"# Total: {len(numerical_cols)} columns")
print(f"all_continuous = {numerical_cols}")

print("\n✅ Analysis complete!")
print("\n💡 Next steps:")
print("  1. Review the classification above")
print("  2. Update dataset_split_and_preprocess.py with the correct lists")
print("  3. Re-run preprocessing: python dataset_split_and_preprocess.py")

