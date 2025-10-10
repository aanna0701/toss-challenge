#!/usr/bin/env python3
"""
간단한 Feature 분류 - 기존 분석 결과 기반

기준:
1. 명백한 categorical: gender, age_group, inventory_id (도메인 지식)
2. l_feat_14: 검증 결과 float 값 존재 → numerical
3. 나머지 모든 l_feat, feat_*, history_*: numerical
"""
import json
from pathlib import Path

print("=" * 80)
print("🔍 Simple Feature Classification")
print("=" * 80)

# 명백한 categorical features (도메인 지식)
CATEGORICAL_FEATURES = ['gender', 'age_group', 'inventory_id']

# 특수 컬럼
SPECIAL_COLS = ['clicked', 'seq', 'ID']

# 제외할 컬럼 (상수 피처)
EXCLUDE_COLS = ['l_feat_20', 'l_feat_23']

# 전체 feature 리스트 생성
all_features = []

# feat_a_* (18개)
all_features.extend([f'feat_a_{i}' for i in range(1, 19)])

# feat_b_* (6개)
all_features.extend([f'feat_b_{i}' for i in range(1, 7)])

# feat_c_* (8개)
all_features.extend([f'feat_c_{i}' for i in range(1, 9)])

# feat_d_* (6개)
all_features.extend([f'feat_d_{i}' for i in range(1, 7)])

# feat_e_* (10개)
all_features.extend([f'feat_e_{i}' for i in range(1, 11)])

# history_a_* (7개)
all_features.extend([f'history_a_{i}' for i in range(1, 8)])

# history_b_* (30개)
all_features.extend([f'history_b_{i}' for i in range(1, 31)])

# l_feat_* (27개, 14/20/23 제외)
all_features.extend([f'l_feat_{i}' for i in range(1, 28) if i not in EXCLUDE_COLS])

# Categorical features 추가
all_features.extend(CATEGORICAL_FEATURES)

print(f"\n📊 Total features: {len(all_features)}")
print(f"   - Categorical: {len(CATEGORICAL_FEATURES)}")
print(f"   - Numerical: {len(all_features) - len(CATEGORICAL_FEATURES)}")
print(f"   - Excluded (constant): {len(EXCLUDE_COLS)}")

# Numerical features (categorical 제외)
numerical_features = [f for f in all_features if f not in CATEGORICAL_FEATURES and f not in SPECIAL_COLS]

print(f"\n✅ Classification:")
print(f"\n📌 Categorical ({len(CATEGORICAL_FEATURES)}):")
for col in sorted(CATEGORICAL_FEATURES):
    print(f"  - {col}")

print(f"\n📈 Numerical ({len(numerical_features)}):")
print(f"  (showing first 20)")
for col in sorted(numerical_features)[:20]:
    print(f"  - {col}")
print(f"  ... and {len(numerical_features) - 20} more")

print(f"\n🗑️  Excluded (constant): {EXCLUDE_COLS}")
print(f"🔖 Special: {SPECIAL_COLS}")

# 결과 저장
results = {
    'metadata': {
        'method': 'domain_knowledge_based',
        'note': 'Classification based on domain knowledge and prior analysis'
    },
    'categorical': sorted(CATEGORICAL_FEATURES),
    'numerical': sorted(numerical_features),
    'excluded': sorted(EXCLUDE_COLS),
    'special': sorted(SPECIAL_COLS),
    'counts': {
        'categorical': len(CATEGORICAL_FEATURES),
        'numerical': len(numerical_features),
        'excluded': len(EXCLUDE_COLS),
        'special': len(SPECIAL_COLS),
        'total_features': len(all_features)
    }
}

# JSON 저장
output_path = Path('analysis/results/feature_classification.json')
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {output_path}")

# Python 코드 생성
print("\n" + "=" * 80)
print("🐍 Python Code for dataset_split_and_preprocess.py")
print("=" * 80)

print("\n# CATEGORICAL COLUMNS")
print(f"categorical_cols = {CATEGORICAL_FEATURES}")

print("\n# CONTINUOUS COLUMNS")
print(f"# Total: {len(numerical_features)} columns")
print("all_continuous = (")
print(f"    [f'feat_a_{{i}}' for i in range(1, 19)] +   # 18")
print(f"    [f'feat_b_{{i}}' for i in range(1, 7)] +    # 6")
print(f"    [f'feat_c_{{i}}' for i in range(1, 9)] +    # 8")
print(f"    [f'feat_d_{{i}}' for i in range(1, 7)] +    # 6")
print(f"    [f'feat_e_{{i}}' for i in range(1, 11)] +   # 10")
print(f"    [f'history_a_{{i}}' for i in range(1, 8)] +   # 7")
print(f"    [f'history_b_{{i}}' for i in range(1, 31)] +  # 30")
print(f"    [f'l_feat_{{i}}' for i in range(1, 28) if i not in [20, 23]]  # 25")
print(")")

print(f"\n# Note: l_feat_14 is included in continuous (not categorical)")
print(f"# Total continuous: {len(numerical_features)}")
print(f"# Total categorical: {len(CATEGORICAL_FEATURES)}")

print("\n✅ Classification complete!")

