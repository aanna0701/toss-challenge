#!/usr/bin/env python3
"""
청크 단위 분석을 위한 유틸리티 함수들
"""

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

class OnlineStats:
    """온라인 통계 계산 클래스"""
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
        self.sum_val = 0.0
    
    def update(self, value):
        """새로운 값으로 통계 업데이트 (Welford's algorithm)"""
        if pd.isna(value):
            return
            
        self.n += 1
        self.sum_val += value
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
        
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
    
    def update_batch(self, values):
        """배치로 통계 업데이트"""
        values = pd.Series(values).dropna()
        for val in values:
            self.update(val)
    
    def get_stats(self):
        """현재 통계 반환"""
        if self.n < 1:
            return {
                'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'sum': 0
            }
        
        variance = self.M2 / self.n if self.n > 0 else 0
        std = np.sqrt(variance)
        
        return {
            'count': self.n,
            'mean': self.mean,
            'std': std,
            'min': self.min_val if self.min_val != float('inf') else 0,
            'max': self.max_val if self.max_val != float('-inf') else 0,
            'sum': self.sum_val
        }

class ChunkCorrelationCalculator:
    """청크 단위 상관관계 계산기"""
    
    def __init__(self, target_col='clicked'):
        self.target_col = target_col
        self.feature_stats = defaultdict(OnlineStats)
        self.target_stats = OnlineStats()
        self.covariance_stats = defaultdict(lambda: {'sum_xy': 0, 'n': 0})
    
    def update_chunk(self, chunk_df):
        """청크 데이터로 상관관계 통계 업데이트"""
        if self.target_col not in chunk_df.columns:
            return
        
        target_data = chunk_df[self.target_col].dropna()
        
        # 수치형 컬럼들만 선택
        numeric_cols = chunk_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != self.target_col]
        
        for col in numeric_cols:
            feature_data = chunk_df[col].dropna()
            
            # 공통 인덱스 찾기
            common_idx = target_data.index.intersection(feature_data.index)
            if len(common_idx) == 0:
                continue
            
            target_common = target_data[common_idx]
            feature_common = feature_data[common_idx]
            
            # 개별 통계 업데이트
            self.feature_stats[col].update_batch(feature_common)
            
            # 공분산 통계 업데이트
            cov_stats = self.covariance_stats[col]
            cov_stats['sum_xy'] += (target_common * feature_common).sum()
            cov_stats['n'] += len(common_idx)
        
        # 타겟 통계 업데이트
        self.target_stats.update_batch(target_data)
    
    def get_correlations(self, top_n=20):
        """상관관계 계산 및 상위 N개 반환"""
        correlations = {}
        target_stats = self.target_stats.get_stats()
        
        if target_stats['count'] == 0:
            return {}
        
        for col, feature_stat in self.feature_stats.items():
            feature_stats = feature_stat.get_stats()
            cov_stats = self.covariance_stats[col]
            
            if feature_stats['count'] == 0 or cov_stats['n'] == 0:
                continue
            
            # 상관관계 계산
            n = min(target_stats['count'], feature_stats['count'], cov_stats['n'])
            if n <= 1:
                continue
            
            # E[XY] - E[X]E[Y]
            covariance = (cov_stats['sum_xy'] / cov_stats['n']) - (target_stats['mean'] * feature_stats['mean'])
            
            # 표준편차의 곱
            std_product = target_stats['std'] * feature_stats['std']
            
            if std_product > 0:
                correlation = covariance / std_product
                correlations[col] = correlation
        
        # 절댓값 기준으로 정렬하여 상위 N개 반환
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        return dict(sorted_corr[:top_n])

class ChunkCategoricalAnalyzer:
    """청크 단위 카테고리형 변수 분석기"""
    
    def __init__(self, target_col='clicked'):
        self.target_col = target_col
        self.category_stats = defaultdict(lambda: defaultdict(lambda: {'count': 0, 'target_sum': 0}))
    
    def update_chunk(self, chunk_df, categorical_cols):
        """청크 데이터로 카테고리 통계 업데이트"""
        if self.target_col not in chunk_df.columns:
            return
        
        for col in categorical_cols:
            if col not in chunk_df.columns:
                continue
            
            # 카테고리별 타겟 통계
            grouped = chunk_df.groupby(col)[self.target_col].agg(['count', 'sum'])
            
            for category, (count, target_sum) in grouped.iterrows():
                self.category_stats[col][str(category)]['count'] += count
                self.category_stats[col][str(category)]['target_sum'] += target_sum
    
    def get_category_analysis(self):
        """카테고리 분석 결과 반환"""
        results = {}
        
        for col, categories in self.category_stats.items():
            col_results = {}
            total_count = sum(stats['count'] for stats in categories.values())
            
            for category, stats in categories.items():
                count = stats['count']
                target_sum = stats['target_sum']
                
                col_results[category] = {
                    'count': count,
                    'percentage': count / total_count * 100 if total_count > 0 else 0,
                    'target_rate': target_sum / count if count > 0 else 0,
                    'target_count': target_sum
                }
            
            # 카운트 기준으로 정렬
            col_results = dict(sorted(col_results.items(), 
                                    key=lambda x: x[1]['count'], reverse=True))
            results[col] = col_results
        
        return results

def analyze_parquet_schema(file_path):
    """Parquet 파일의 스키마 정보 분석"""
    try:
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema
        
        print("📋 Parquet 파일 스키마 정보:")
        print(f"   - 총 컬럼 수: {len(schema)}")
        print(f"   - 총 행 수: {parquet_file.metadata.num_rows:,}")
        print(f"   - 파일 크기: {parquet_file.metadata.serialized_size / 1024**2:.2f} MB")
        print(f"   - Row Group 수: {parquet_file.metadata.num_row_groups}")
        
        # 컬럼 타입별 분포
        type_counts = defaultdict(int)
        for i in range(len(schema)):
            col_type = str(schema.field(i).type)
            type_counts[col_type] += 1
        
        print("\n🏷️ 컬럼 타입 분포:")
        for col_type, count in sorted(type_counts.items()):
            print(f"   - {col_type}: {count}개")
        
        # Row Group 정보
        print(f"\n📦 Row Group 정보:")
        for i in range(min(3, parquet_file.metadata.num_row_groups)):  # 상위 3개만
            rg = parquet_file.metadata.row_group(i)
            print(f"   - Group {i}: {rg.num_rows:,} 행, {rg.total_byte_size / 1024**2:.2f} MB")
        
        return {
            'total_rows': parquet_file.metadata.num_rows,
            'total_columns': len(schema),
            'file_size_mb': parquet_file.metadata.serialized_size / 1024**2,
            'num_row_groups': parquet_file.metadata.num_row_groups,
            'column_types': dict(type_counts)
        }
        
    except Exception as e:
        print(f"❌ 스키마 분석 실패: {e}")
        return None

def estimate_memory_usage(file_path, chunk_size=100000):
    """메모리 사용량 추정"""
    try:
        parquet_file = pq.ParquetFile(file_path)
        
        # 작은 샘플로 메모리 사용량 측정
        sample_batch = next(parquet_file.iter_batches(batch_size=min(1000, chunk_size)))
        sample_df = sample_batch.to_pandas()
        
        sample_memory = sample_df.memory_usage(deep=True).sum()
        estimated_chunk_memory = sample_memory * (chunk_size / len(sample_df))
        
        print(f"💾 메모리 사용량 추정:")
        print(f"   - 샘플 크기: {len(sample_df):,} 행")
        print(f"   - 샘플 메모리: {sample_memory / 1024**2:.2f} MB")
        print(f"   - 청크 메모리 (예상): {estimated_chunk_memory / 1024**2:.2f} MB")
        
        return estimated_chunk_memory
        
    except Exception as e:
        print(f"❌ 메모리 추정 실패: {e}")
        return None

def recommend_chunk_size(file_path, target_memory_mb=500):
    """권장 청크 크기 계산"""
    try:
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        
        # 샘플로 행당 메모리 사용량 계산
        sample_batch = next(parquet_file.iter_batches(batch_size=1000))
        sample_df = sample_batch.to_pandas()
        memory_per_row = sample_df.memory_usage(deep=True).sum() / len(sample_df)
        
        # 목표 메모리에 맞는 청크 크기 계산
        target_memory_bytes = target_memory_mb * 1024 * 1024
        recommended_chunk_size = int(target_memory_bytes / memory_per_row)
        
        # 합리적인 범위로 제한
        recommended_chunk_size = max(10000, min(1000000, recommended_chunk_size))
        
        print(f"🎯 권장 청크 크기:")
        print(f"   - 목표 메모리: {target_memory_mb} MB")
        print(f"   - 행당 메모리: {memory_per_row:.2f} bytes")
        print(f"   - 권장 청크 크기: {recommended_chunk_size:,} 행")
        print(f"   - 예상 총 청크 수: {(total_rows + recommended_chunk_size - 1) // recommended_chunk_size}")
        
        return recommended_chunk_size
        
    except Exception as e:
        print(f"❌ 청크 크기 계산 실패: {e}")
        return 100000  # 기본값
