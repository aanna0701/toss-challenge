#!/usr/bin/env python3
"""
청크 단위 EDA - 대용량 데이터를 위한 메모리 효율적 분석
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

warnings.filterwarnings('ignore')

# Set English locale and font settings for plots
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
import locale
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_ALL, 'C')
    except:
        pass

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import seed_everything

# 결과 저장 폴더
RESULTS_DIR = Path("analysis/results")
RESULTS_DIR.mkdir(exist_ok=True)

class ChunkEDA:
    """청크 단위 EDA 클래스"""
    
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
        self.feature_groups = {}
        self.stats = defaultdict(dict)
        
        self.initialize()
    
    def initialize(self):
        """초기화"""
        print("📊 청크 단위 EDA 초기화...")
        
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
        self.feature_groups = {
            'basic': ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour'],
            'sequence': ['seq'],
            'l_feat': [col for col in columns if col.startswith('l_feat_')],
            'feat_e': [col for col in columns if col.startswith('feat_e_')],
            'feat_d': [col for col in columns if col.startswith('feat_d_')],
            'feat_c': [col for col in columns if col.startswith('feat_c_')],
            'feat_b': [col for col in columns if col.startswith('feat_b_')],
            'feat_a': [col for col in columns if col.startswith('feat_a_')],
            'history_a': [col for col in columns if col.startswith('history_a_')],
            'target': ['clicked'],
            'other': []
        }
        
        # 분류되지 않은 컬럼들을 'other'에 추가
        all_categorized = []
        for group_cols in self.feature_groups.values():
            all_categorized.extend(group_cols)
        
        self.feature_groups['other'] = [col for col in columns if col not in all_categorized]
        
        print("\n📋 피처 그룹별 개수:")
        for group, cols in self.feature_groups.items():
            if cols:
                print(f"   - {group}: {len(cols)}개")
    
    def analyze_chunks(self):
        """청크별 분석 수행"""
        print("\n" + "="*60)
        print("📊 청크별 데이터 분석 시작")
        print("="*60)
        
        # 통계 초기화
        self.stats = {
            'basic_info': {
                'total_rows': self.total_rows,
                'total_chunks': self.total_chunks,
                'chunk_size': self.chunk_size
            },
            'target_stats': {'clicked_0': 0, 'clicked_1': 0},
            'missing_stats': defaultdict(int),
            'categorical_stats': defaultdict(lambda: defaultdict(int)),
            'numerical_stats': defaultdict(lambda: {
                'count': 0, 'sum': 0, 'sum_sq': 0, 'min': float('inf'), 'max': float('-inf')
            }),
            'sequence_stats': {'total_length': 0, 'total_count': 0, 'empty_count': 0},
            'dtypes': {}
        }
        
        # 청크별 처리
        for chunk_idx, batch in enumerate(self.parquet_file.iter_batches(batch_size=self.chunk_size)):
            print(f"\r📊 처리 중: {chunk_idx+1}/{self.total_chunks} 청크 "
                  f"({(chunk_idx+1)/self.total_chunks*100:.1f}%)", end="", flush=True)
            
            chunk_df = batch.to_pandas()
            self.process_chunk(chunk_df, chunk_idx)
        
        print(f"\n✅ 모든 청크 처리 완료!")
        self.finalize_stats()
    
    def process_chunk(self, chunk_df, chunk_idx):
        """개별 청크 처리"""
        # 첫 번째 청크에서 데이터 타입 저장
        if chunk_idx == 0:
            self.stats['dtypes'] = chunk_df.dtypes.to_dict()
        
        # 타겟 변수 통계
        if 'clicked' in chunk_df.columns:
            target_counts = chunk_df['clicked'].value_counts()
            self.stats['target_stats']['clicked_0'] += target_counts.get(0, 0)
            self.stats['target_stats']['clicked_1'] += target_counts.get(1, 0)
        
        # 결측값 통계
        missing = chunk_df.isnull().sum()
        for col, count in missing.items():
            self.stats['missing_stats'][col] += count
        
        # 카테고리형 변수 통계
        categorical_cols = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']
        for col in categorical_cols:
            if col in chunk_df.columns:
                value_counts = chunk_df[col].value_counts()
                for val, count in value_counts.items():
                    self.stats['categorical_stats'][col][str(val)] += count
        
        # 수치형 변수 통계
        numeric_cols = chunk_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['clicked']:  # 타겟 제외
                data = chunk_df[col].dropna()
                if len(data) > 0:
                    stats = self.stats['numerical_stats'][col]
                    stats['count'] += len(data)
                    stats['sum'] += data.sum()
                    stats['sum_sq'] += (data ** 2).sum()
                    stats['min'] = min(stats['min'], data.min())
                    stats['max'] = max(stats['max'], data.max())
        
        # 시퀀스 통계
        if 'seq' in chunk_df.columns:
            for seq_str in chunk_df['seq'].astype(str):
                if seq_str and seq_str != 'nan':
                    try:
                        seq_length = len(seq_str.split(','))
                        self.stats['sequence_stats']['total_length'] += seq_length
                        self.stats['sequence_stats']['total_count'] += 1
                    except:
                        self.stats['sequence_stats']['empty_count'] += 1
                else:
                    self.stats['sequence_stats']['empty_count'] += 1
    
    def finalize_stats(self):
        """통계 최종 계산"""
        print("\n📊 통계 최종 계산 중...")
        
        # 수치형 변수의 평균, 표준편차 계산
        for col, stats in self.stats['numerical_stats'].items():
            if stats['count'] > 0:
                mean = stats['sum'] / stats['count']
                variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
                std = np.sqrt(max(0, variance))
                
                stats['mean'] = mean
                stats['std'] = std
        
        # 시퀀스 평균 길이 계산
        seq_stats = self.stats['sequence_stats']
        if seq_stats['total_count'] > 0:
            seq_stats['avg_length'] = seq_stats['total_length'] / seq_stats['total_count']
        else:
            seq_stats['avg_length'] = 0
    
    def print_results(self):
        """결과 출력"""
        print("\n" + "="*60)
        print("📈 청크 단위 EDA 결과")
        print("="*60)
        
        # 기본 정보
        print(f"\n📋 기본 정보:")
        print(f"   - 전체 행 수: {self.stats['basic_info']['total_rows']:,}")
        print(f"   - 처리된 청크 수: {self.stats['basic_info']['total_chunks']}")
        print(f"   - 청크 크기: {self.stats['basic_info']['chunk_size']:,}")
        
        # 데이터 타입
        print(f"\n🏷️ 데이터 타입:")
        dtype_counts = pd.Series(self.stats['dtypes']).value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   - {dtype}: {count}개")
        
        # 타겟 변수
        print(f"\n🎯 타겟 변수 (전체 데이터):")
        total_samples = self.stats['target_stats']['clicked_0'] + self.stats['target_stats']['clicked_1']
        for label in [0, 1]:
            count = self.stats['target_stats'][f'clicked_{label}']
            pct = count / total_samples * 100 if total_samples > 0 else 0
            print(f"   - clicked={label}: {count:,}개 ({pct:.2f}%)")
        
        ctr = self.stats['target_stats']['clicked_1'] / total_samples if total_samples > 0 else 0
        print(f"   - 전체 클릭률 (CTR): {ctr:.6f}")
        
        # 결측값
        print(f"\n🕳️ 결측값 (전체 데이터):")
        missing_cols = [(col, count) for col, count in self.stats['missing_stats'].items() if count > 0]
        missing_cols.sort(key=lambda x: x[1], reverse=True)
        
        if missing_cols:
            print(f"   - 결측값 있는 컬럼: {len(missing_cols)}개")
            for col, count in missing_cols[:10]:
                pct = count / self.total_rows * 100
                print(f"     * {col}: {count:,}개 ({pct:.2f}%)")
            if len(missing_cols) > 10:
                print(f"     ... 총 {len(missing_cols)}개 컬럼")
        else:
            print("   - 결측값 없음! ✅")
        
        # 카테고리형 변수
        print(f"\n🔍 주요 카테고리형 변수:")
        for col, value_counts in self.stats['categorical_stats'].items():
            if value_counts:
                total_count = sum(value_counts.values())
                sorted_values = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
                top_value, top_count = sorted_values[0]
                top_pct = top_count / total_count * 100
                
                print(f"   - {col}: {len(sorted_values)}개 고유값, 최빈값 '{top_value}' ({top_pct:.1f}%)")
        
        # 시퀀스 분석
        print(f"\n🔢 시퀀스 분석 (전체 데이터):")
        seq_stats = self.stats['sequence_stats']
        print(f"   - 평균 길이: {seq_stats['avg_length']:.2f}")
        print(f"   - 총 시퀀스 수: {seq_stats['total_count']:,}")
        print(f"   - 빈 시퀀스: {seq_stats['empty_count']:,}개")
        if seq_stats['total_count'] > 0:
            empty_pct = seq_stats['empty_count'] / (seq_stats['total_count'] + seq_stats['empty_count']) * 100
            print(f"   - 빈 시퀀스 비율: {empty_pct:.2f}%")
        
        # 수치형 변수 요약
        print(f"\n📊 주요 수치형 변수 통계:")
        for group_name, columns in self.feature_groups.items():
            if group_name in ['target', 'basic', 'sequence', 'other']:
                continue
            
            numeric_cols = [col for col in columns if col in self.stats['numerical_stats']]
            if numeric_cols:
                print(f"\n   🏷️ {group_name} 그룹 ({len(numeric_cols)}개 피처):")
                
                # 그룹 통계 요약
                means = [self.stats['numerical_stats'][col]['mean'] for col in numeric_cols[:5]]  # 상위 5개만
                stds = [self.stats['numerical_stats'][col]['std'] for col in numeric_cols[:5]]
                
                if means:
                    print(f"     - 평균값 범위: {min(means):.4f} ~ {max(means):.4f}")
                    print(f"     - 표준편차 범위: {min(stds):.4f} ~ {max(stds):.4f}")
    
    def save_results(self):
        """결과 저장"""
        print(f"\n💾 결과 저장 중...")
        
        # JSON으로 통계 저장
        stats_to_save = {}
        for key, value in self.stats.items():
            if key == 'numerical_stats':
                # numerical_stats는 너무 크므로 요약만 저장
                stats_to_save[key] = {
                    col: {k: v for k, v in stats.items() if k in ['count', 'mean', 'std', 'min', 'max']}
                    for col, stats in value.items()
                }
            else:
                stats_to_save[key] = dict(value) if hasattr(value, 'items') else value
        
        with open(RESULTS_DIR / 'chunk_eda_results.json', 'w', encoding='utf-8') as f:
            json.dump(stats_to_save, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 결과가 {RESULTS_DIR}/chunk_eda_results.json에 저장되었습니다.")
    
    def create_summary_plots(self):
        """Summary visualization generation"""
        print(f"\n📊 Creating summary visualizations...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 1. Target distribution
            target_data = [self.stats['target_stats']['clicked_0'], self.stats['target_stats']['clicked_1']]
            axes[0, 0].pie(target_data, labels=['Not Clicked', 'Clicked'], 
                          autopct='%1.2f%%', colors=['skyblue', 'salmon'])
            axes[0, 0].set_title('Overall Click Distribution')
            
            # 2. Data type distribution
            dtype_counts = pd.Series(self.stats['dtypes']).value_counts()
            axes[0, 1].bar(range(len(dtype_counts)), dtype_counts.values, color='lightblue')
            axes[0, 1].set_title('Data Type Distribution')
            axes[0, 1].set_xticks(range(len(dtype_counts)))
            axes[0, 1].set_xticklabels(dtype_counts.index, rotation=45)
            axes[0, 1].set_ylabel('Count')
            
            # 3. Top 10 missing value features
            missing_data = [(col, count) for col, count in self.stats['missing_stats'].items() if count > 0]
            missing_data.sort(key=lambda x: x[1], reverse=True)
            
            if missing_data:
                top_missing = missing_data[:10]
                cols, counts = zip(*top_missing)
                axes[1, 0].barh(range(len(cols)), counts, color='orange')
                axes[1, 0].set_title('Top 10 Features with Missing Values')
                axes[1, 0].set_yticks(range(len(cols)))
                axes[1, 0].set_yticklabels(cols)
                axes[1, 0].invert_yaxis()
                axes[1, 0].set_xlabel('Missing Count')
            else:
                axes[1, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Missing Values Status')
            
            # 4. Feature group counts
            group_counts = {k: len(v) for k, v in self.feature_groups.items() if v}
            axes[1, 1].bar(range(len(group_counts)), list(group_counts.values()), color='lightgreen')
            axes[1, 1].set_title('Feature Count by Group')
            axes[1, 1].set_xticks(range(len(group_counts)))
            axes[1, 1].set_xticklabels(list(group_counts.keys()), rotation=45)
            axes[1, 1].set_ylabel('Feature Count')
            
            plt.tight_layout()
            plt.savefig(RESULTS_DIR / 'chunk_eda_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"✅ Visualization saved to {RESULTS_DIR}/chunk_eda_summary.png")
            
        except Exception as e:
            print(f"⚠️  Error creating visualization: {e}")
    
    def run_analysis(self):
        """전체 분석 실행"""
        print("🚀 청크 단위 EDA 시작")
        print("="*60)
        
        # 청크별 분석
        self.analyze_chunks()
        
        # 결과 출력
        self.print_results()
        
        # 결과 저장
        self.save_results()
        
        # 시각화 생성
        self.create_summary_plots()
        
        print(f"\n🎉 청크 단위 EDA 완료!")
        print(f"📁 결과 파일:")
        print(f"   - {RESULTS_DIR}/chunk_eda_results.json")
        print(f"   - {RESULTS_DIR}/chunk_eda_summary.png")

def main():
    """메인 실행 함수"""
    seed_everything(42)
    
    # 청크 크기를 인자로 받을 수 있도록
    import argparse
    parser = argparse.ArgumentParser(description='청크 단위 EDA')
    parser.add_argument('--chunk_size', type=int, default=100000, help='청크 크기')
    parser.add_argument('--data_path', type=str, default='./train.parquet', help='데이터 파일 경로')
    args = parser.parse_args()
    
    # EDA 실행
    eda = ChunkEDA(data_path=args.data_path, chunk_size=args.chunk_size)
    eda.run_analysis()

if __name__ == "__main__":
    main()
