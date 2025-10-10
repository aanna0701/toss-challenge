#!/usr/bin/env python3
"""
수정사항 확인: l_feat_14가 categorical로 제대로 처리되는지 확인
"""

print("=" * 80)
print("🔍 dataset_split_and_preprocess.py 수정사항 확인")
print("=" * 80)

# Import the function to check
import sys
sys.path.insert(0, '/home/seunghoon.lee/dev/toss-challenge')

# Read the file to verify
with open('dataset_split_and_preprocess.py', 'r') as f:
    content = f.read()

print("\n✅ Checking continuous features definition...")
if 'if i not in [14, 20, 23]' in content:
    print("   ✅ l_feat_14 excluded from continuous features")
else:
    print("   ❌ l_feat_14 still in continuous features!")

print("\n✅ Checking categorical features definition...")
if "categorical_cols = ['gender', 'age_group', 'inventory_id', 'l_feat_14']" in content:
    print("   ✅ l_feat_14 included in categorical features")
else:
    print("   ❌ l_feat_14 not in categorical features!")

print("\n✅ Checking feature counts in docstring...")
if '109 continuous features' in content:
    print("   ✅ Continuous count updated to 109")
else:
    print("   ⚠️  Continuous count might need update")

if '4 categorical features' in content:
    print("   ✅ Categorical count is 4")
else:
    print("   ⚠️  Categorical count might need update")

print("\n" + "=" * 80)
print("✅ 수정사항 확인 완료!")
print("=" * 80)

print("\n📋 다음 단계:")
print("  1. 전처리 재실행:")
print("     $ conda activate toss-env")
print("     $ python dataset_split_and_preprocess.py")
print()
print("  2. 재실행 후 l_feat_14 확인:")
print("     $ conda activate toss-env")
print("     $ python verify_categorify_result.py")
print()
print("  3. 예상 결과:")
print("     - l_feat_14: dtype=int64, NaN=0개")
print("     - 값: 0-based integer (예: 0, 1, 2, 3, ...)")
print("     - Categorify 적용 ✅")

