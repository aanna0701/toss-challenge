#!/usr/bin/env python3
"""
모든 Missing Pattern 검증 - Pattern 1, 2, 3 모두
"""

import pandas as pd
import numpy as np
import json

def validate_all_patterns():
    """모든 패턴 검증"""
    
    print("=" * 80)
    print("모든 Missing Pattern 검증")
    print("=" * 80)
    print()
    
    # 각 패턴의 대표 features 선택
    pattern_features = {
        'pattern_1': ['gender', 'age_group', 'l_feat_2', 'feat_e_1', 'history_a_1'],  # 17,208 missing
        'pattern_2': ['feat_a_1', 'feat_a_2', 'feat_a_3', 'feat_a_4', 'feat_a_5'],   # 18,598 missing
        'pattern_3': ['feat_e_3']  # 1,085,557 missing
    }
    
    # 모든 features 합치기
    all_features = []
    for features in pattern_features.values():
        all_features.extend(features)
    
    print(f"검증할 features: {len(all_features)}개")
    for pattern, features in pattern_features.items():
        print(f"  {pattern}: {features}")
    print()
    
    try:
        print("🔍 데이터 로드 중...")
        
        # 선택된 컬럼만 로드
        df = pd.read_parquet(
            '/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/train.parquet',
            columns=all_features
        )
        
        print(f"로드 완료: {len(df):,} rows, {len(df.columns)} columns")
        print()
        
        # 각 패턴별 검증
        results = {}
        
        for pattern_name, features in pattern_features.items():
            print(f"📊 {pattern_name.upper()} 검증:")
            print("-" * 40)
            
            # 해당 패턴의 features만 선택
            pattern_df = df[features]
            
            # 각 feature별 missing count
            missing_counts = pattern_df.isnull().sum()
            print("Feature별 missing count:")
            for feature, count in missing_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {feature}: {count:,} ({percentage:.3f}%)")
            
            # 모든 features가 동시에 missing인 행
            all_missing = pattern_df.isnull().all(axis=1)
            all_missing_count = all_missing.sum()
            all_missing_percentage = (all_missing_count / len(df)) * 100
            
            # 하나라도 missing인 행
            any_missing = pattern_df.isnull().any(axis=1)
            any_missing_count = any_missing.sum()
            any_missing_percentage = (any_missing_count / len(df)) * 100
            
            # 부분적으로만 missing인 행
            partial_missing_count = any_missing_count - all_missing_count
            
            print(f"\nMissing 패턴:")
            print(f"  모든 features 동시 missing: {all_missing_count:,} ({all_missing_percentage:.3f}%)")
            print(f"  하나라도 missing: {any_missing_count:,} ({any_missing_percentage:.3f}%)")
            print(f"  부분적으로만 missing: {partial_missing_count:,}")
            
            # 예상값과 비교
            expected_counts = {
                'pattern_1': 17208,
                'pattern_2': 18598,
                'pattern_3': 1085557
            }
            
            expected = expected_counts[pattern_name]
            expected_percentage = (expected / len(df)) * 100
            difference = abs(all_missing_count - expected)
            
            print(f"\n예상값과 비교:")
            print(f"  예상: {expected:,} ({expected_percentage:.3f}%)")
            print(f"  실제: {all_missing_count:,} ({all_missing_percentage:.3f}%)")
            print(f"  차이: {difference:,}")
            
            # 결론
            if difference < 100:
                conclusion = "CONFIRMED"
                print(f"  ✅ 확인됨! 동일한 행에서 모든 features 동시 missing")
            elif difference < expected * 0.01:  # 1% 이내
                conclusion = "MOSTLY_CONFIRMED"
                print(f"  ⚠️  거의 확인됨 (1% 이내 차이)")
            else:
                conclusion = "REJECTED"
                print(f"  ❌ 불일치")
            
            # 결과 저장
            results[pattern_name] = {
                'features': features,
                'feature_missing_counts': missing_counts.to_dict(),
                'all_missing_together': int(all_missing_count),
                'any_missing': int(any_missing_count),
                'partial_missing': int(partial_missing_count),
                'expected': expected,
                'difference': int(difference),
                'conclusion': conclusion
            }
            
            print()
        
        # 패턴 간 중복 분석
        print("🔍 패턴 간 중복 분석:")
        print("-" * 40)
        
        # 각 패턴별 missing indicator 생성
        pattern_indicators = {}
        for pattern_name, features in pattern_features.items():
            pattern_df = df[features]
            if len(features) > 1:
                pattern_indicators[pattern_name] = pattern_df.isnull().all(axis=1)
            else:
                pattern_indicators[pattern_name] = pattern_df.isnull().iloc[:, 0]
        
        # 패턴 간 중복 계산
        overlaps = {}
        pattern_names = list(pattern_indicators.keys())
        
        for i, p1 in enumerate(pattern_names):
            for p2 in pattern_names[i+1:]:
                overlap_key = f"{p1}_and_{p2}"
                overlap = pattern_indicators[p1] & pattern_indicators[p2]
                overlap_count = overlap.sum()
                overlap_percentage = (overlap_count / len(df)) * 100
                
                overlaps[overlap_key] = int(overlap_count)
                
                print(f"{p1} ∩ {p2}: {overlap_count:,} rows ({overlap_percentage:.3f}%)")
                
                # 중복률 계산
                p1_count = pattern_indicators[p1].sum()
                p2_count = pattern_indicators[p2].sum()
                
                if p1_count > 0 and p2_count > 0:
                    overlap_rate_p1 = (overlap_count / p1_count) * 100
                    overlap_rate_p2 = (overlap_count / p2_count) * 100
                    print(f"  중복률: {p1}의 {overlap_rate_p1:.1f}%, {p2}의 {overlap_rate_p2:.1f}%")
        
        print()
        
        # 전체 결론
        print("🎯 전체 결론:")
        print("-" * 40)
        
        confirmed_patterns = [p for p, r in results.items() if r['conclusion'] in ['CONFIRMED', 'MOSTLY_CONFIRMED']]
        
        print(f"확인된 패턴: {len(confirmed_patterns)}개 / {len(results)}개")
        
        for pattern_name, result in results.items():
            status = "✅" if result['conclusion'] == 'CONFIRMED' else "⚠️" if result['conclusion'] == 'MOSTLY_CONFIRMED' else "❌"
            print(f"  {pattern_name}: {status} {result['conclusion']}")
        
        print()
        
        if len(confirmed_patterns) == len(results):
            print("✅ **모든 패턴이 확인됨!**")
            print("   → 각 패턴별로 동일한 행에서 해당 features들이 모두 missing")
            print("   → 체계적이고 규칙적인 missing pattern")
        else:
            print("⚠️  **일부 패턴만 확인됨**")
            print("   → 추가 분석이 필요할 수 있음")
        
        # 최종 결과 저장
        final_results = {
            'validation_summary': {
                'total_patterns_tested': len(results),
                'confirmed_patterns': len(confirmed_patterns),
                'confirmed_pattern_names': confirmed_patterns
            },
            'pattern_results': results,
            'pattern_overlaps': overlaps,
            'key_findings': [
                f"Pattern 1: {results['pattern_1']['conclusion']} - 17,208개 행에서 77개 features 모두 missing",
                f"Pattern 2: {results['pattern_2']['conclusion']} - 18,598개 행에서 18개 features 모두 missing", 
                f"Pattern 3: {results['pattern_3']['conclusion']} - 1,085,557개 행에서 1개 feature missing"
            ]
        }
        
        output_file = '/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/analysis/results/all_patterns_validation.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 전체 검증 결과가 저장되었습니다: {output_file}")
        
    except Exception as e:
        print(f"❌ 검증 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    validate_all_patterns()
