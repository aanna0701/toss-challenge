#!/usr/bin/env python3
"""
전체 train set에서 mean/std를 계산하는 스크립트
표준화(normalization)에 사용할 통계값을 계산하고 JSON 형태로 저장합니다.
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import yaml
from datetime import datetime
import argparse


def load_config(config_path="../config.yaml"):
    """설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def safe_load_parquet(file_path, sample_size=None, use_sampling=False):
    """안전한 parquet 로드 함수 (메모리 효율적)"""
    print(f"⚠️  {file_path} 대용량 데이터 - 청크 단위로 처리...")
    
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        print(f"📊 총 {total_rows:,} 행 처리 예정")
        
        # 청크 단위로 통계 계산
        stats_accumulator = {}
        processed_rows = 0
        
        for batch in parquet_file.iter_batches(batch_size=10000):  # 배치 크기 줄임
            chunk_df = batch.to_pandas()
            processed_rows += len(chunk_df)
            
            # 각 컬럼별로 누적 통계 계산
            for col in chunk_df.select_dtypes(include=[np.number]).columns:
                if col not in stats_accumulator:
                    stats_accumulator[col] = {
                        'sum': 0.0,
                        'sum_sq': 0.0,
                        'count': 0,
                        'min': np.inf,
                        'max': -np.inf
                    }
                
                col_data = chunk_df[col].dropna()
                if len(col_data) > 0:
                    stats_accumulator[col]['sum'] += col_data.sum()
                    stats_accumulator[col]['sum_sq'] += (col_data ** 2).sum()
                    stats_accumulator[col]['count'] += len(col_data)
                    stats_accumulator[col]['min'] = min(stats_accumulator[col]['min'], col_data.min())
                    stats_accumulator[col]['max'] = max(stats_accumulator[col]['max'], col_data.max())
            
            if processed_rows % 100000 == 0:  # 10만 행마다 진행률 출력
                print(f"📈 진행률: {processed_rows:,}/{total_rows:,} ({processed_rows/total_rows*100:.1f}%)")
        
        print(f"📈 최종 진행률: {processed_rows:,}/{total_rows:,} (100.0%)")
        
        # 최종 통계 계산
        final_stats = {}
        for col, stats in stats_accumulator.items():
            if stats['count'] > 0:
                mean = stats['sum'] / stats['count']
                variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
                std = np.sqrt(max(0, variance))  # 음수 방지
                
                final_stats[col] = {
                    'mean': float(mean),
                    'std': float(std),
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'count': int(stats['count'])
                }
        
        return final_stats
        
    except Exception as e:
        print(f"❌ {file_path} 처리 실패: {e}")
        raise


def compute_normalization_stats(train_data_path, output_path, feature_cols=None, exclude_cols=None):
    """
    전체 train set에서 normalization 통계를 계산합니다.
    
    Args:
        train_data_path: train.parquet 파일 경로
        output_path: 결과를 저장할 JSON 파일 경로
        feature_cols: 계산할 특정 컬럼들 (None이면 모든 숫자 컬럼)
        exclude_cols: 제외할 컬럼들 (ID, target 등)
    """
    
    print("🚀 Normalization 통계 계산 시작")
    print(f"📁 데이터 파일: {train_data_path}")
    print(f"💾 결과 저장: {output_path}")
    
    # 제외할 컬럼들 기본 설정
    if exclude_cols is None:
        exclude_cols = {'ID', 'clicked', 'seq'}  # ID, target, sequence 컬럼 제외
    
    try:
        # 데이터 로드 (메모리 효율적)
        print("\n📊 데이터 로드 중...")
        stats = safe_load_parquet(train_data_path)
        
        # 제외할 컬럼들 필터링
        if exclude_cols:
            stats = {col: stat for col, stat in stats.items() if col not in exclude_cols}
        
        # 특정 컬럼만 계산하는 경우
        if feature_cols:
            stats = {col: stat for col, stat in stats.items() if col in feature_cols}
        
        print(f"✅ 청크 단위 처리 완료")
        print(f"📈 계산된 컬럼 수: {len(stats)}")
        print(f"📋 컬럼 목록: {list(stats.keys())[:10]}{'...' if len(stats) > 10 else ''}")
        
        # 결과 저장
        result = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'data_file': train_data_path,
                'total_columns': len(stats),
                'excluded_columns': list(exclude_cols) if exclude_cols else [],
                'description': 'Normalization statistics for feature scaling'
            },
            'statistics': stats
        }
        
        # JSON 파일로 저장
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 통계 계산 완료!")
        print(f"📊 총 {len(stats)}개 컬럼 처리")
        print(f"💾 결과 저장: {output_path}")
        
        # 요약 정보 출력
        print(f"\n📋 요약 정보:")
        for i, (col, stat) in enumerate(list(stats.items())[:5]):
            print(f"  {col}: mean={stat['mean']:.4f}, std={stat['std']:.4f}")
        if len(stats) > 5:
            print(f"  ... 및 {len(stats) - 5}개 컬럼 더")
        
        return result
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        raise


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='Train set normalization 통계 계산')
    parser.add_argument('--config', default='../config.yaml', help='설정 파일 경로')
    parser.add_argument('--output', default='results/normalization_stats.json', help='출력 파일 경로')
    parser.add_argument('--exclude', nargs='*', default=['ID', 'clicked', 'seq'], help='제외할 컬럼들')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = load_config(args.config)
    train_data_path = config['PATHS']['TRAIN_DATA']
    
    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(train_data_path):
        train_data_path = os.path.join(os.path.dirname(args.config), train_data_path)
    
    # 결과 파일 경로 설정
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(args.config), 'analysis', output_path)
    
    # 통계 계산 실행
    compute_normalization_stats(
        train_data_path=train_data_path,
        output_path=output_path,
        exclude_cols=set(args.exclude)
    )


if __name__ == "__main__":
    main()
