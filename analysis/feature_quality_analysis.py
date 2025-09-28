#!/usr/bin/env python3
"""
피처 품질 분석 스크립트
- 피처별 값 분포 분석 (분별력 낮은 피처 식별)
- 피처별 클릭률과의 상관관계 분석 (연관성 낮은 피처 식별)
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import sys
import os
import json
from collections import defaultdict
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# Set English locale and font settings for plots
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import seed_everything
from analysis.chunk_utils import OnlineStats, ChunkCorrelationCalculator, ChunkCategoricalAnalyzer

# 결과 저장 폴더
RESULTS_DIR = Path("analysis/results")
RESULTS_DIR.mkdir(exist_ok=True)

class FeatureQualityAnalyzer:
    """피처 품질 분석 클래스"""
    
    def __init__(self, data_path="./train.parquet", chunk_size=100000):
        """
        초기화
        Args:
            data_path: train.parquet 파일 경로
            chunk_size: 청크 크기
        """
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.parquet_file = None
        self.total_rows = 0
        self.total_chunks = 0
        
        # 분석 결과 저장
        self.feature_stats = defaultdict(lambda: OnlineStats())
        self.correlation_calculator = ChunkCorrelationCalculator()
        self.categorical_analyzer = ChunkCategoricalAnalyzer()
        self.categorical_features = []
        self.numerical_features = []
        
        self.initialize()
    
    def initialize(self):
        """초기화"""
        print("🔍 피처 품질 분석 초기화...")
        
        try:
            self.parquet_file = pq.ParquetFile(self.data_path)
            self.total_rows = self.parquet_file.metadata.num_rows
            self.total_chunks = (self.total_rows + self.chunk_size - 1) // self.chunk_size
            
            print(f"✅ 파일 정보:")
            print(f"   - 전체 행 수: {self.total_rows:,}")
            print(f"   - 청크 크기: {self.chunk_size:,}")
            print(f"   - 총 청크 수: {self.total_chunks}")
            
            # 첫 번째 청크로 피처 정보 파악
            first_batch = next(self.parquet_file.iter_batches(batch_size=1000))
            first_df = first_batch.to_pandas()
            self.categorize_features(first_df.columns.tolist())
            
        except Exception as e:
            print(f"❌ 초기화 실패: {e}")
            raise
    
    def categorize_features(self, columns):
        """피처를 카테고리별로 분류"""
        # 타겟과 시퀀스 제외
        exclude_cols = ['clicked', 'seq']
        
        for col in columns:
            if col not in exclude_cols:
                if col.startswith(('l_feat_', 'feat_', 'history_')):
                    self.numerical_features.append(col)
                else:
                    # 데이터 타입으로 판단
                    first_batch = next(self.parquet_file.iter_batches(batch_size=100))
                    first_df = first_batch.to_pandas()
                    if pd.api.types.is_numeric_dtype(first_df[col]):
                        self.numerical_features.append(col)
                    else:
                        self.categorical_features.append(col)
        
        print(f"\n📋 피처 분류:")
        print(f"   - 수치형 피처: {len(self.numerical_features)}개")
        print(f"   - 카테고리형 피처: {len(self.categorical_features)}개")
        if self.categorical_features:
            print(f"   - 카테고리형 피처 예시: {self.categorical_features[:5]}")
    
    def analyze_chunks(self):
        """청크별 분석 수행"""
        print("\n" + "="*60)
        print("📊 청크별 피처 품질 분석 시작")
        print("="*60)
        
        # 청크별 처리
        for chunk_idx, batch in enumerate(self.parquet_file.iter_batches(batch_size=self.chunk_size)):
            print(f"\r📊 처리 중: {chunk_idx+1}/{self.total_chunks} 청크 "
                  f"({(chunk_idx+1)/self.total_chunks*100:.1f}%)", end="", flush=True)
            
            chunk_df = batch.to_pandas()
            self.process_chunk(chunk_df)
        
        print(f"\n✅ 모든 청크 처리 완료!")
    
    def process_chunk(self, chunk_df):
        """개별 청크 처리"""
        # 수치형 피처 통계 업데이트
        for col in self.numerical_features:
            if col in chunk_df.columns:
                self.feature_stats[col].update_batch(chunk_df[col])
        
        # 범주형 피처 분석 업데이트
        self.categorical_analyzer.update_chunk(chunk_df, self.categorical_features)
        
        # 상관관계 계산기 업데이트
        self.correlation_calculator.update_chunk(chunk_df)
    
    def analyze_distribution_quality(self):
        """피처 분포 품질 분석"""
        print("\n" + "="*60)
        print("📈 피처 분포 품질 분석")
        print("="*60)
        
        distribution_issues = {
            'low_variance': [],      # 분산이 매우 낮은 피처
            'constant': [],          # 상수 피처
            'binary_extreme': [],    # 0 또는 1만 있는 피처 (극단적)
            'sparse_extreme': []     # 99% 이상이 0인 피처
        }
        
        for col in self.numerical_features:
            if col not in self.feature_stats:
                continue
            
            stats = self.feature_stats[col].get_stats()
            
            if stats['count'] == 0:
                continue
            
            # 1. 상수 피처 체크
            if stats['std'] == 0:
                distribution_issues['constant'].append({
                    'feature': col,
                    'value': stats['mean'],
                    'count': stats['count']
                })
                continue
            
            # 2. 분산이 매우 낮은 피처 (CV < 0.01)
            cv = stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else float('inf')
            if cv < 0.01 and stats['mean'] != 0:
                distribution_issues['low_variance'].append({
                    'feature': col,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'cv': cv
                })
            
            # 3. 극단적으로 sparse한 피처 체크 (0값 비율이 99% 이상)
            # 0값 개수를 추정 (정확한 계산은 별도 필요)
            if stats['min'] == 0 and stats['mean'] < 0.01:
                distribution_issues['sparse_extreme'].append({
                    'feature': col,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'estimated_zero_ratio': 1 - stats['mean'] if stats['mean'] < 1 else 0
                })
        
        # 결과 출력
        self.print_distribution_issues(distribution_issues)
        
        return distribution_issues
    
    def print_distribution_issues(self, distribution_issues):
        """분포 문제점 출력"""
        print(f"\n🚨 분포 품질 문제 피처들:")
        
        # 상수 피처
        if distribution_issues['constant']:
            print(f"\n❌ 상수 피처 ({len(distribution_issues['constant'])}개):")
            for item in distribution_issues['constant'][:10]:  # 상위 10개만
                print(f"   - {item['feature']}: 값={item['value']:.6f}, 개수={item['count']:,}")
            if len(distribution_issues['constant']) > 10:
                print(f"   ... 총 {len(distribution_issues['constant'])}개")
        
        # 낮은 분산 피처
        if distribution_issues['low_variance']:
            print(f"\n⚠️ 낮은 분산 피처 ({len(distribution_issues['low_variance'])}개):")
            for item in sorted(distribution_issues['low_variance'], key=lambda x: x['cv'])[:10]:
                print(f"   - {item['feature']}: CV={item['cv']:.6f}, 평균={item['mean']:.4f}, 표준편차={item['std']:.6f}")
            if len(distribution_issues['low_variance']) > 10:
                print(f"   ... 총 {len(distribution_issues['low_variance'])}개")
        
        # 극단적 sparse 피처
        if distribution_issues['sparse_extreme']:
            print(f"\n⚠️ 극단적 Sparse 피처 ({len(distribution_issues['sparse_extreme'])}개):")
            for item in sorted(distribution_issues['sparse_extreme'], key=lambda x: x['estimated_zero_ratio'], reverse=True)[:10]:
                print(f"   - {item['feature']}: 0값 비율≈{item['estimated_zero_ratio']:.3f}, 평균={item['mean']:.6f}")
            if len(distribution_issues['sparse_extreme']) > 10:
                print(f"   ... 총 {len(distribution_issues['sparse_extreme'])}개")
    
    def analyze_correlation_quality(self):
        """피처-클릭률 상관관계 품질 분석"""
        print("\n" + "="*60)
        print("🔗 피처-클릭률 상관관계 품질 분석")
        print("="*60)
        
        # 상관관계 계산
        correlations = self.correlation_calculator.get_correlations(top_n=1000)
        
        # 상관관계가 매우 낮은 피처들 식별
        correlation_issues = {
            'very_low_correlation': [],  # |correlation| < 0.001
            'zero_correlation': [],      # |correlation| < 0.0001
            'negative_correlation': []   # correlation < -0.001
        }
        
        for feature, corr in correlations.items():
            abs_corr = abs(corr)
            
            if abs_corr < 0.0001:
                correlation_issues['zero_correlation'].append({
                    'feature': feature,
                    'correlation': corr
                })
            elif abs_corr < 0.001:
                correlation_issues['very_low_correlation'].append({
                    'feature': feature,
                    'correlation': corr
                })
            
            if corr < -0.001:
                correlation_issues['negative_correlation'].append({
                    'feature': feature,
                    'correlation': corr
                })
        
        # 결과 출력
        self.print_correlation_issues(correlation_issues, correlations)
        
        return correlation_issues, correlations
    
    def analyze_categorical_quality(self):
        """범주형 피처 품질 분석 (ANOVA 기반)"""
        print("\n" + "="*60)
        print("📊 범주형 피처 품질 분석 (ANOVA)")
        print("="*60)
        
        categorical_analysis = self.categorical_analyzer.get_category_analysis()
        
        categorical_issues = {
            'low_variance': [],      # 카테고리별 CTR 편차가 낮은 피처
            'dominant_category': [], # 한 카테고리가 95% 이상인 피처
            'few_categories': [],    # 카테고리가 2개 이하인 피처
            'low_anova_f': []        # ANOVA F-statistic이 낮은 피처
        }
        
        for feature, categories in categorical_analysis.items():
            if not categories:
                continue
            
            # 카테고리 개수
            num_categories = len(categories)
            
            # 카테고리별 통계
            ctr_values = [stats['target_rate'] for stats in categories.values()]
            counts = [stats['count'] for stats in categories.values()]
            total_count = sum(counts)
            
            # CTR 편차 계산
            ctr_std = np.std(ctr_values)
            ctr_mean = np.mean(ctr_values)
            ctr_cv = ctr_std / ctr_mean if ctr_mean > 0 else 0
            
            # 가장 큰 카테고리 비율
            max_category_ratio = max(counts) / total_count
            
            # ANOVA F-statistic 계산 (간단한 버전)
            # Between-group variance / Within-group variance
            if len(ctr_values) > 1 and ctr_std > 0:
                # 간단한 F-statistic 근사치
                anova_f = (ctr_std ** 2) / (ctr_mean * (1 - ctr_mean)) if ctr_mean > 0 and ctr_mean < 1 else 0
            else:
                anova_f = 0
            
            # 문제점 식별
            if ctr_cv < 0.01:  # CTR 편차가 매우 낮음
                categorical_issues['low_variance'].append({
                    'feature': feature,
                    'ctr_cv': ctr_cv,
                    'ctr_std': ctr_std,
                    'num_categories': num_categories
                })
            
            if max_category_ratio > 0.95:  # 한 카테고리가 95% 이상
                categorical_issues['dominant_category'].append({
                    'feature': feature,
                    'max_ratio': max_category_ratio,
                    'num_categories': num_categories
                })
            
            if num_categories <= 2:  # 카테고리가 2개 이하
                categorical_issues['few_categories'].append({
                    'feature': feature,
                    'num_categories': num_categories,
                    'categories': list(categories.keys())
                })
            
            if anova_f < 0.001:  # ANOVA F-statistic이 매우 낮음
                categorical_issues['low_anova_f'].append({
                    'feature': feature,
                    'anova_f': anova_f,
                    'ctr_cv': ctr_cv,
                    'num_categories': num_categories
                })
        
        # 결과 출력
        self.print_categorical_issues(categorical_issues)
        
        return categorical_issues, categorical_analysis
    
    def print_categorical_issues(self, categorical_issues):
        """범주형 피처 문제점 출력"""
        print(f"\n🚨 범주형 피처 품질 문제들:")
        
        # 낮은 CTR 편차
        if categorical_issues['low_variance']:
            print(f"\n⚠️ 낮은 CTR 편차 피처 ({len(categorical_issues['low_variance'])}개):")
            for item in sorted(categorical_issues['low_variance'], key=lambda x: x['ctr_cv'])[:10]:
                print(f"   - {item['feature']}: CTR CV={item['ctr_cv']:.6f}, 카테고리 수={item['num_categories']}")
            if len(categorical_issues['low_variance']) > 10:
                print(f"   ... 총 {len(categorical_issues['low_variance'])}개")
        
        # 지배적 카테고리
        if categorical_issues['dominant_category']:
            print(f"\n⚠️ 지배적 카테고리 피처 ({len(categorical_issues['dominant_category'])}개):")
            for item in sorted(categorical_issues['dominant_category'], key=lambda x: x['max_ratio'], reverse=True)[:10]:
                print(f"   - {item['feature']}: 최대 비율={item['max_ratio']:.3f}, 카테고리 수={item['num_categories']}")
            if len(categorical_issues['dominant_category']) > 10:
                print(f"   ... 총 {len(categorical_issues['dominant_category'])}개")
        
        # 적은 카테고리
        if categorical_issues['few_categories']:
            print(f"\n⚠️ 적은 카테고리 피처 ({len(categorical_issues['few_categories'])}개):")
            for item in categorical_issues['few_categories']:
                print(f"   - {item['feature']}: 카테고리 수={item['num_categories']}, 카테고리={item['categories']}")
        
        # 낮은 ANOVA F-statistic
        if categorical_issues['low_anova_f']:
            print(f"\n❌ 낮은 ANOVA F-statistic 피처 ({len(categorical_issues['low_anova_f'])}개):")
            for item in sorted(categorical_issues['low_anova_f'], key=lambda x: x['anova_f'])[:10]:
                print(f"   - {item['feature']}: F-stat={item['anova_f']:.6f}, CTR CV={item['ctr_cv']:.6f}")
            if len(categorical_issues['low_anova_f']) > 10:
                print(f"   ... 총 {len(categorical_issues['low_anova_f'])}개")
    
    def print_correlation_issues(self, correlation_issues, correlations):
        """상관관계 문제점 출력"""
        print(f"\n🚨 상관관계 품질 문제 피처들:")
        
        # 거의 0에 가까운 상관관계
        if correlation_issues['zero_correlation']:
            print(f"\n❌ 거의 0인 상관관계 피처 ({len(correlation_issues['zero_correlation'])}개):")
            for item in sorted(correlation_issues['zero_correlation'], key=lambda x: abs(x['correlation']))[:10]:
                print(f"   - {item['feature']}: {item['correlation']:.8f}")
            if len(correlation_issues['zero_correlation']) > 10:
                print(f"   ... 총 {len(correlation_issues['zero_correlation'])}개")
        
        # 매우 낮은 상관관계
        if correlation_issues['very_low_correlation']:
            print(f"\n⚠️ 매우 낮은 상관관계 피처 ({len(correlation_issues['very_low_correlation'])}개):")
            for item in sorted(correlation_issues['very_low_correlation'], key=lambda x: abs(x['correlation']))[:10]:
                print(f"   - {item['feature']}: {item['correlation']:.6f}")
            if len(correlation_issues['very_low_correlation']) > 10:
                print(f"   ... 총 {len(correlation_issues['very_low_correlation'])}개")
        
        # 음의 상관관계
        if correlation_issues['negative_correlation']:
            print(f"\n⚠️ 음의 상관관계 피처 ({len(correlation_issues['negative_correlation'])}개):")
            for item in sorted(correlation_issues['negative_correlation'], key=lambda x: x['correlation'])[:10]:
                print(f"   - {item['feature']}: {item['correlation']:.6f}")
            if len(correlation_issues['negative_correlation']) > 10:
                print(f"   ... 총 {len(correlation_issues['negative_correlation'])}개")
        
        # 상위 상관관계 피처들
        top_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        print(f"\n✅ 상위 상관관계 피처들:")
        for feature, corr in top_correlations:
            print(f"   - {feature}: {corr:.6f}")
    
    def create_visualizations(self, distribution_issues, correlation_issues, correlations, categorical_issues=None):
        """시각화 생성"""
        print(f"\n📊 시각화 생성 중...")
        
        try:
            if categorical_issues:
                fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            else:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 1. 상관관계 분포 히스토그램
            corr_values = list(correlations.values())
            axes[0, 0].hist(corr_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].axvline(x=0.001, color='red', linestyle='--', label='|corr| = 0.001')
            axes[0, 0].axvline(x=-0.001, color='red', linestyle='--')
            axes[0, 0].axvline(x=0.0001, color='orange', linestyle='--', label='|corr| = 0.0001')
            axes[0, 0].axvline(x=-0.0001, color='orange', linestyle='--')
            axes[0, 0].set_title('Feature-Target Correlation Distribution')
            axes[0, 0].set_xlabel('Correlation')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 2. 문제 피처 카운트
            issue_counts = [
                len(distribution_issues['constant']),
                len(distribution_issues['low_variance']),
                len(distribution_issues['sparse_extreme']),
                len(correlation_issues['zero_correlation']),
                len(correlation_issues['very_low_correlation'])
            ]
            issue_labels = ['Constant', 'Low Variance', 'Extreme Sparse', 'Zero Corr', 'Very Low Corr']
            
            axes[0, 1].bar(range(len(issue_counts)), issue_counts, color=['red', 'orange', 'yellow', 'purple', 'pink'])
            axes[0, 1].set_title('Feature Quality Issues Count')
            axes[0, 1].set_xticks(range(len(issue_labels)))
            axes[0, 1].set_xticklabels(issue_labels, rotation=45)
            axes[0, 1].set_ylabel('Count')
            
            # 3. 상위 상관관계 피처들
            top_20 = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
            top_features = [item[0] for item in top_20]
            top_corrs = [item[1] for item in top_20]
            
            y_pos = np.arange(len(top_features))
            colors = ['red' if c < 0 else 'blue' for c in top_corrs]
            axes[1, 0].barh(y_pos, top_corrs, color=colors, alpha=0.7)
            axes[1, 0].set_yticks(y_pos)
            axes[1, 0].set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in top_features])
            axes[1, 0].set_title('Top 20 Feature Correlations')
            axes[1, 0].set_xlabel('Correlation')
            axes[1, 0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # 4. 상관관계 vs 분산 산점도 (샘플)
            sample_features = list(correlations.keys())[:100]  # 상위 100개만
            sample_corrs = [correlations[f] for f in sample_features]
            sample_stds = [self.feature_stats[f].get_stats()['std'] for f in sample_features]
            
            axes[1, 1].scatter(sample_stds, [abs(c) for c in sample_corrs], alpha=0.6, color='green')
            axes[1, 1].set_xlabel('Standard Deviation')
            axes[1, 1].set_ylabel('Absolute Correlation')
            axes[1, 1].set_title('Correlation vs Feature Variance')
            axes[1, 1].grid(True, alpha=0.3)
            
            # 범주형 분석 시각화 (있는 경우)
            if categorical_issues:
                # 5. 범주형 피처 문제 카운트
                cat_issue_counts = [
                    len(categorical_issues['low_variance']),
                    len(categorical_issues['dominant_category']),
                    len(categorical_issues['few_categories']),
                    len(categorical_issues['low_anova_f'])
                ]
                cat_issue_labels = ['Low CTR Var', 'Dominant Cat', 'Few Cats', 'Low ANOVA F']
                
                axes[2, 0].bar(range(len(cat_issue_counts)), cat_issue_counts, color=['orange', 'red', 'purple', 'brown'])
                axes[2, 0].set_title('Categorical Feature Issues Count')
                axes[2, 0].set_xticks(range(len(cat_issue_labels)))
                axes[2, 0].set_xticklabels(cat_issue_labels, rotation=45)
                axes[2, 0].set_ylabel('Count')
                
                # 6. 범주형 피처별 카테고리 수 분포
                cat_features = list(categorical_issues.keys())
                cat_counts = [len(categorical_issues[key]) for key in cat_features]
                axes[2, 1].pie(cat_counts, labels=cat_issue_labels, autopct='%1.1f%%', startangle=90)
                axes[2, 1].set_title('Categorical Issues Distribution')
            
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / 'feature_quality_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✅ 시각화가 {RESULTS_DIR}/feature_quality_analysis.png에 저장되었습니다.")
            
        except Exception as e:
            print(f"⚠️ 시각화 생성 실패: {e}")
    
    def save_results(self, distribution_issues, correlation_issues, correlations, categorical_issues=None, categorical_analysis=None):
        """결과 저장"""
        print(f"\n💾 결과 저장 중...")
        
        # 전체 결과 구성
        results = {
            'analysis_info': {
                'total_numerical_features': len(self.numerical_features),
                'total_categorical_features': len(self.categorical_features),
                'total_chunks_processed': self.total_chunks,
                'total_rows_processed': self.total_rows
            },
            'distribution_issues': dict(distribution_issues),
            'correlation_issues': dict(correlation_issues),
            'all_correlations': correlations,
            'summary': {
                'constant_features': len(distribution_issues['constant']),
                'low_variance_features': len(distribution_issues['low_variance']),
                'sparse_extreme_features': len(distribution_issues['sparse_extreme']),
                'zero_correlation_features': len(correlation_issues['zero_correlation']),
                'very_low_correlation_features': len(correlation_issues['very_low_correlation']),
                'negative_correlation_features': len(correlation_issues['negative_correlation'])
            }
        }
        
        # 범주형 분석 결과 추가
        if categorical_issues:
            results['categorical_issues'] = dict(categorical_issues)
            results['categorical_analysis'] = categorical_analysis
            results['summary'].update({
                'low_ctr_variance_categorical': len(categorical_issues['low_variance']),
                'dominant_category_categorical': len(categorical_issues['dominant_category']),
                'few_categories_categorical': len(categorical_issues['few_categories']),
                'low_anova_f_categorical': len(categorical_issues['low_anova_f'])
            })
        
        # JSON으로 저장
        with open(RESULTS_DIR / 'feature_quality_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 결과가 {RESULTS_DIR}/feature_quality_analysis.json에 저장되었습니다.")
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("🚀 피처 품질 분석 시작")
        print("="*60)
        
        # 청크별 분석
        self.analyze_chunks()
        
        # 분포 품질 분석
        distribution_issues = self.analyze_distribution_quality()
        
        # 상관관계 품질 분석
        correlation_issues, correlations = self.analyze_correlation_quality()
        
        # 범주형 피처 품질 분석
        categorical_issues = None
        categorical_analysis = None
        if self.categorical_features:
            categorical_issues, categorical_analysis = self.analyze_categorical_quality()
        
        # 시각화 생성
        self.create_visualizations(distribution_issues, correlation_issues, correlations, categorical_issues)
        
        # 결과 저장
        self.save_results(distribution_issues, correlation_issues, correlations, categorical_issues, categorical_analysis)
        
        print(f"\n🎉 피처 품질 분석 완료!")
        print(f"📁 결과 파일:")
        print(f"   - {RESULTS_DIR}/feature_quality_analysis.json")
        print(f"   - {RESULTS_DIR}/feature_quality_analysis.png")

def main():
    """메인 실행 함수"""
    seed_everything(42)
    
    # 인자 파싱
    import argparse
    parser = argparse.ArgumentParser(description='피처 품질 분석')
    parser.add_argument('--chunk_size', type=int, default=100000, help='청크 크기')
    parser.add_argument('--data_path', type=str, default='./train.parquet', help='데이터 파일 경로')
    args = parser.parse_args()
    
    # 분석 실행
    analyzer = FeatureQualityAnalyzer(data_path=args.data_path, chunk_size=args.chunk_size)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
