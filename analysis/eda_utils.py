#!/usr/bin/env python3
"""
EDA 유틸리티 함수들
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

# Set English locale and font settings for plots
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_categorical_feature(df, feature, target='clicked', max_categories=20):
    """
    카테고리형 피처 분석
    
    Args:
        df: 데이터프레임
        feature: 분석할 피처명
        target: 타겟 변수명
        max_categories: 표시할 최대 카테고리 수
    
    Returns:
        dict: 분석 결과
    """
    print(f"\n📊 {feature} 분석:")
    
    # 기본 정보
    unique_count = df[feature].nunique()
    missing_count = df[feature].isnull().sum()
    missing_pct = missing_count / len(df) * 100
    
    print(f"   - 고유값 개수: {unique_count:,}")
    print(f"   - 결측값: {missing_count:,}개 ({missing_pct:.2f}%)")
    
    # 값별 분포
    value_counts = df[feature].value_counts().head(max_categories)
    print(f"   - 상위 {min(max_categories, len(value_counts))}개 값:")
    
    results = {'feature': feature, 'categories': {}}
    
    for val, count in value_counts.items():
        pct = count / len(df) * 100
        
        # 해당 카테고리의 클릭률
        category_data = df[df[feature] == val]
        ctr = category_data[target].mean()
        
        print(f"     * {val}: {count:,}개 ({pct:.2f}%) - CTR: {ctr:.6f}")
        
        results['categories'][str(val)] = {
            'count': int(count),
            'percentage': float(pct),
            'ctr': float(ctr)
        }
    
    # 전체 클릭률과 비교
    overall_ctr = df[target].mean()
    print(f"   - 전체 CTR: {overall_ctr:.6f}")
    
    # 카테고리별 CTR 편차
    ctr_by_category = df.groupby(feature)[target].mean().sort_values(ascending=False)
    max_ctr = ctr_by_category.max()
    min_ctr = ctr_by_category.min()
    
    print(f"   - CTR 범위: {min_ctr:.6f} ~ {max_ctr:.6f}")
    print(f"   - CTR 편차: {(max_ctr - min_ctr):.6f}")
    
    results.update({
        'unique_count': int(unique_count),
        'missing_count': int(missing_count),
        'missing_percentage': float(missing_pct),
        'overall_ctr': float(overall_ctr),
        'ctr_range': [float(min_ctr), float(max_ctr)],
        'ctr_deviation': float(max_ctr - min_ctr)
    })
    
    return results

def analyze_numerical_feature(df, feature, target='clicked'):
    """
    수치형 피처 분석
    
    Args:
        df: 데이터프레임
        feature: 분석할 피처명
        target: 타겟 변수명
    
    Returns:
        dict: 분석 결과
    """
    print(f"\n📈 {feature} 분석:")
    
    # 기본 통계
    desc = df[feature].describe()
    missing_count = df[feature].isnull().sum()
    missing_pct = missing_count / len(df) * 100
    
    print(f"   - 결측값: {missing_count:,}개 ({missing_pct:.2f}%)")
    print(f"   - 기본 통계:")
    for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        print(f"     * {stat}: {desc[stat]:.4f}")
    
    # 0값 비율 (sparse feature 체크)
    zero_count = (df[feature] == 0).sum()
    zero_pct = zero_count / len(df) * 100
    print(f"   - 0값: {zero_count:,}개 ({zero_pct:.2f}%)")
    
    # 타겟과의 상관관계
    correlation = df[feature].corr(df[target])
    print(f"   - {target}와 상관관계: {correlation:.6f}")
    
    # 구간별 클릭률 분석
    try:
        # 10분위로 나누어 분석
        df_temp = df[[feature, target]].copy()
        df_temp = df_temp.dropna()
        
        if len(df_temp) > 0:
            df_temp['decile'] = pd.qcut(df_temp[feature], q=10, duplicates='drop')
            ctr_by_decile = df_temp.groupby('decile')[target].agg(['count', 'mean'])
            
            print(f"   - 구간별 CTR (10분위):")
            for decile, (count, ctr) in ctr_by_decile.iterrows():
                print(f"     * {decile}: {ctr:.6f} ({count:,}개)")
    
    except Exception as e:
        print(f"   - 구간별 분석 실패: {e}")
    
    results = {
        'feature': feature,
        'missing_count': int(missing_count),
        'missing_percentage': float(missing_pct),
        'statistics': desc.to_dict(),
        'zero_count': int(zero_count),
        'zero_percentage': float(zero_pct),
        'correlation': float(correlation) if not pd.isna(correlation) else None
    }
    
    return results

def plot_feature_target_relationship(df, feature, target='clicked', figsize=(12, 5)):
    """
    Feature-target relationship visualization
    
    Args:
        df: DataFrame
        feature: Feature name to analyze
        target: Target variable name
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    if df[feature].dtype == 'object' or df[feature].nunique() < 20:
        # Categorical feature
        # 1. Distribution
        value_counts = df[feature].value_counts().head(15)
        axes[0].bar(range(len(value_counts)), value_counts.values, color='lightblue')
        axes[0].set_title(f'{feature} Distribution')
        axes[0].set_xticks(range(len(value_counts)))
        axes[0].set_xticklabels(value_counts.index, rotation=45)
        axes[0].set_ylabel('Count')
        
        # 2. Click rate by category
        ctr_by_category = df.groupby(feature)[target].mean().sort_values(ascending=False).head(15)
        axes[1].bar(range(len(ctr_by_category)), ctr_by_category.values, color='orange')
        axes[1].set_title(f'Click Rate by {feature}')
        axes[1].set_xticks(range(len(ctr_by_category)))
        axes[1].set_xticklabels(ctr_by_category.index, rotation=45)
        axes[1].set_ylabel('Click Rate')
        
    else:
        # Numerical feature
        # 1. Histogram
        axes[0].hist(df[feature].dropna(), bins=50, alpha=0.7, color='lightgreen')
        axes[0].set_title(f'{feature} Distribution')
        axes[0].set_xlabel(feature)
        axes[0].set_ylabel('Frequency')
        
        # 2. Click rate by bins
        try:
            df_temp = df[[feature, target]].dropna()
            df_temp['bins'] = pd.cut(df_temp[feature], bins=20)
            ctr_by_bin = df_temp.groupby('bins')[target].mean()
            
            x_pos = range(len(ctr_by_bin))
            axes[1].plot(x_pos, ctr_by_bin.values, marker='o', color='red')
            axes[1].set_title(f'Click Rate by {feature} Bins')
            axes[1].set_xlabel('Bins')
            axes[1].set_ylabel('Click Rate')
            axes[1].set_xticks(x_pos[::max(1, len(x_pos)//5)])
            
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Binning analysis failed\n{str(e)}', 
                        ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.show()

def compare_feature_groups(df, groups_dict, target='clicked'):
    """
    피처 그룹 간 비교 분석
    
    Args:
        df: 데이터프레임
        groups_dict: 그룹별 피처 딕셔너리
        target: 타겟 변수명
    
    Returns:
        dict: 그룹 비교 결과
    """
    print("\n📊 피처 그룹 비교 분석")
    print("="*50)
    
    group_stats = {}
    
    for group_name, features in groups_dict.items():
        if not features or group_name == 'target':
            continue
        
        # 해당 그룹의 수치형 피처들만
        numeric_features = [f for f in features if f in df.columns and df[f].dtype in ['int64', 'float64']]
        
        if not numeric_features:
            continue
        
        group_data = df[numeric_features]
        
        # 그룹 통계
        stats = {
            'feature_count': len(numeric_features),
            'mean_correlation': group_data.corrwith(df[target]).abs().mean(),
            'max_correlation': group_data.corrwith(df[target]).abs().max(),
            'sparse_ratio': (group_data == 0).mean().mean(),  # 평균 0값 비율
            'missing_ratio': group_data.isnull().mean().mean()  # 평균 결측값 비율
        }
        
        print(f"\n🏷️ {group_name}:")
        print(f"   - 피처 수: {stats['feature_count']}")
        print(f"   - 평균 상관관계: {stats['mean_correlation']:.6f}")
        print(f"   - 최대 상관관계: {stats['max_correlation']:.6f}")
        print(f"   - 평균 Sparse 비율: {stats['sparse_ratio']:.3f}")
        print(f"   - 평균 결측값 비율: {stats['missing_ratio']:.3f}")
        
        # 상위 상관관계 피처들
        correlations = group_data.corrwith(df[target]).abs().sort_values(ascending=False)
        top_features = correlations.head(3)
        print(f"   - 상위 상관관계 피처:")
        for feat, corr in top_features.items():
            print(f"     * {feat}: {corr:.6f}")
        
        group_stats[group_name] = stats
        group_stats[group_name]['top_features'] = top_features.to_dict()
    
    return group_stats

def detect_outliers(df, feature, method='iqr'):
    """
    이상치 탐지
    
    Args:
        df: 데이터프레임
        feature: 분석할 피처명
        method: 탐지 방법 ('iqr', 'zscore')
    
    Returns:
        dict: 이상치 정보
    """
    if df[feature].dtype not in ['int64', 'float64']:
        return {'error': 'Only numeric features supported'}
    
    data = df[feature].dropna()
    
    if method == 'iqr':
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        outliers = data[z_scores > 3]
        lower_bound = data.mean() - 3 * data.std()
        upper_bound = data.mean() + 3 * data.std()
    
    outlier_count = len(outliers)
    outlier_pct = outlier_count / len(data) * 100
    
    result = {
        'method': method,
        'outlier_count': outlier_count,
        'outlier_percentage': outlier_pct,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'total_samples': len(data)
    }
    
    print(f"\n🔍 {feature} 이상치 탐지 ({method}):")
    print(f"   - 이상치 개수: {outlier_count:,}개 ({outlier_pct:.2f}%)")
    print(f"   - 정상 범위: {lower_bound:.4f} ~ {upper_bound:.4f}")
    
    return result

def create_feature_importance_plot(correlations, title="Feature Importance", top_n=20):
    """
    Feature importance visualization
    
    Args:
        correlations: Correlation Series
        title: Plot title
        top_n: Number of top features to display
    """
    top_corr = correlations.abs().sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(top_corr))
    
    plt.barh(y_pos, top_corr.values, color='lightcoral')
    plt.yticks(y_pos, top_corr.index)
    plt.xlabel('Absolute Correlation with Target')
    plt.title(title)
    plt.gca().invert_yaxis()
    
    # Display values
    for i, v in enumerate(top_corr.values):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plt.show()
