#!/usr/bin/env python3
"""
feat_e_3 실제 상관관계 분석
- 실제 데이터를 로드하여 feat_e_3 missing pattern과 클릭률의 상관관계 계산
- 통계적 유의성 검정
"""

import pandas as pd
import numpy as np
import json
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

def analyze_feat_e_3_correlation():
    """feat_e_3 missing pattern과 클릭률의 실제 상관관계 분석"""
    print("feat_e_3 상관관계 분석 시작...")
    
    # 데이터 로드 (샘플 사용)
    print("데이터 로딩 중...")
    df = pd.read_parquet('/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/train.parquet', engine='pyarrow')
    
    # 메모리 절약을 위해 샘플 사용
    sample_size = min(50000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    print(f"샘플 크기: {len(df_sample):,}")
    
    # feat_e_3 missing indicator 생성
    df_sample['feat_e_3_missing'] = df_sample['feat_e_3'].isna().astype(int)
    
    # 기본 통계
    total_rows = len(df_sample)
    missing_rows = df_sample['feat_e_3_missing'].sum()
    non_missing_rows = total_rows - missing_rows
    
    # 클릭률 계산
    overall_click_rate = df_sample['clicked'].mean()
    click_rate_missing = df_sample[df_sample['feat_e_3_missing'] == 1]['clicked'].mean()
    click_rate_non_missing = df_sample[df_sample['feat_e_3_missing'] == 0]['clicked'].mean()
    
    print(f"\n=== feat_e_3 Missing Pattern 분석 ===")
    print(f"전체 샘플 크기: {total_rows:,}")
    print(f"feat_e_3 missing: {missing_rows:,} ({missing_rows/total_rows*100:.2f}%)")
    print(f"feat_e_3 non-missing: {non_missing_rows:,} ({non_missing_rows/total_rows*100:.2f}%)")
    
    print(f"\n=== 클릭률 분석 ===")
    print(f"전체 클릭률: {overall_click_rate:.6f} ({overall_click_rate*100:.4f}%)")
    print(f"feat_e_3 missing일 때 클릭률: {click_rate_missing:.6f} ({click_rate_missing*100:.4f}%)")
    print(f"feat_e_3 non-missing일 때 클릭률: {click_rate_non_missing:.6f} ({click_rate_non_missing*100:.4f}%)")
    print(f"클릭률 차이: {click_rate_non_missing - click_rate_missing:.6f}")
    
    # 통계적 유의성 검정 (Chi-square test)
    contingency_table = pd.crosstab(df_sample['feat_e_3_missing'], df_sample['clicked'])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\n=== 통계적 유의성 검정 ===")
    print(f"Chi-square 통계량: {chi2:.6f}")
    print(f"p-value: {p_value:.2e}")
    print(f"자유도: {dof}")
    print(f"α=0.05에서 유의함: {'Yes' if p_value < 0.05 else 'No'}")
    
    # Effect size (Cramer's V)
    n = total_rows
    cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
    print(f"Cramer's V (effect size): {cramers_v:.6f}")
    
    # 상대적 위험도 (Relative Risk)
    if click_rate_missing > 0:
        relative_risk = click_rate_non_missing / click_rate_missing
        print(f"상대적 위험도 (RR): {relative_risk:.4f}")
    else:
        relative_risk = np.inf
        print(f"상대적 위험도 (RR): ∞ (missing 그룹에서 클릭률이 0)")
    
    # 오즈비 (Odds Ratio)
    # missing 그룹: 클릭/비클릭
    missing_clicks = df_sample[(df_sample['feat_e_3_missing'] == 1) & (df_sample['clicked'] == 1)].shape[0]
    missing_no_clicks = df_sample[(df_sample['feat_e_3_missing'] == 1) & (df_sample['clicked'] == 0)].shape[0]
    
    # non-missing 그룹: 클릭/비클릭
    non_missing_clicks = df_sample[(df_sample['feat_e_3_missing'] == 0) & (df_sample['clicked'] == 1)].shape[0]
    non_missing_no_clicks = df_sample[(df_sample['feat_e_3_missing'] == 0) & (df_sample['clicked'] == 0)].shape[0]
    
    if missing_clicks > 0 and missing_no_clicks > 0 and non_missing_clicks > 0 and non_missing_no_clicks > 0:
        odds_ratio = (non_missing_clicks * missing_no_clicks) / (missing_clicks * non_missing_no_clicks)
        print(f"오즈비 (OR): {odds_ratio:.4f}")
    else:
        odds_ratio = np.nan
        print(f"오즈비 (OR): 계산 불가 (0 값 존재)")
    
    # 결과 저장
    results = {
        'sample_size': total_rows,
        'feat_e_3_missing_count': missing_rows,
        'feat_e_3_missing_percentage': missing_rows/total_rows*100,
        'overall_click_rate': overall_click_rate,
        'click_rate_when_missing': click_rate_missing,
        'click_rate_when_non_missing': click_rate_non_missing,
        'click_rate_difference': click_rate_non_missing - click_rate_missing,
        'relative_risk': relative_risk,
        'odds_ratio': odds_ratio,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cramers_v': cramers_v,
        'contingency_table': contingency_table.to_dict()
    }
    
    return results

def analyze_other_features_correlation():
    """다른 주요 피쳐들의 상관관계도 간단히 분석"""
    print("\n=== 다른 주요 피쳐들 상관관계 분석 ===")
    
    # 데이터 로드
    df = pd.read_parquet('/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/train.parquet', engine='pyarrow')
    sample_size = min(30000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    print(f"샘플 크기: {len(df_sample):,}")
    
    # 주요 피쳐들 분석
    features_to_analyze = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
    
    results = {}
    
    for feature in features_to_analyze:
        if feature in df_sample.columns:
            # Missing indicator 생성
            df_sample[f'{feature}_missing'] = df_sample[feature].isna().astype(int)
            
            # 클릭률 계산
            click_rate_missing = df_sample[df_sample[f'{feature}_missing'] == 1]['clicked'].mean()
            click_rate_non_missing = df_sample[df_sample[f'{feature}_missing'] == 0]['clicked'].mean()
            
            # Chi-square test
            contingency_table = pd.crosstab(df_sample[f'{feature}_missing'], df_sample['clicked'])
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Cramer's V
            n = len(df_sample)
            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
            
            results[feature] = {
                'missing_count': df_sample[f'{feature}_missing'].sum(),
                'missing_percentage': df_sample[f'{feature}_missing'].sum() / len(df_sample) * 100,
                'click_rate_missing': click_rate_missing,
                'click_rate_non_missing': click_rate_non_missing,
                'click_rate_difference': click_rate_non_missing - click_rate_missing,
                'chi2_statistic': chi2,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cramers_v': cramers_v
            }
            
            print(f"\n{feature}:")
            print(f"  Missing: {results[feature]['missing_count']:,} ({results[feature]['missing_percentage']:.2f}%)")
            print(f"  클릭률 차이: {results[feature]['click_rate_difference']:.6f}")
            print(f"  Cramer's V: {results[feature]['cramers_v']:.6f}")
            print(f"  유의함: {'Yes' if results[feature]['significant'] else 'No'}")
    
    return results

def main():
    """메인 분석 함수"""
    print("feat_e_3 상관관계 분석 시작...")
    
    # feat_e_3 분석
    feat_e_3_results = analyze_feat_e_3_correlation()
    
    # 다른 피쳐들 분석
    other_features_results = analyze_other_features_correlation()
    
    # 전체 결과 저장
    all_results = {
        'feat_e_3_analysis': feat_e_3_results,
        'other_features_analysis': other_features_results,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    output_file = '/Users/seunghoon/Library/Mobile Documents/com~apple~CloudDocs/Research/toss/analysis/results/feat_e_3_correlation_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n결과가 저장되었습니다: {output_file}")
    print("분석 완료!")

if __name__ == "__main__":
    main()
