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
from datetime import datetime
import argparse


def load_parquet_chunked_precise(file_path, chunk_size=100000):
    """청크 단위로 로드하여 double precision으로 정확한 통계 계산"""
    print(f"📊 {file_path} 청크 단위로 로드 중...")
    
    try:
        import pyarrow.parquet as pq
        parquet_file = pq.ParquetFile(file_path)
        total_rows = parquet_file.metadata.num_rows
        total_chunks = (total_rows + chunk_size - 1) // chunk_size
        
        print(f"📈 총 {total_rows:,} 행, {total_chunks}개 청크로 처리")
        
        # 통계 누적을 위한 딕셔너리 (double precision 사용)
        stats_accumulator = {}
        processed_rows = 0
        
        # 첫 번째 청크로 컬럼 정보 파악
        first_batch = next(parquet_file.iter_batches(batch_size=1000))
        first_df = first_batch.to_pandas()
        numeric_cols = first_df.select_dtypes(include=[np.number]).columns
        print(f"📋 수치형 컬럼 수: {len(numeric_cols)}")
        
        # 각 컬럼별 누적 통계 초기화 (double precision)
        for col in numeric_cols:
            stats_accumulator[col] = {
                'sum': 0.0,           # double precision
                'sum_sq': 0.0,        # double precision  
                'count': 0,
                'min': np.inf,
                'max': -np.inf
            }
        
        # 청크별 처리
        for chunk_idx, batch in enumerate(parquet_file.iter_batches(batch_size=chunk_size)):
            chunk_df = batch.to_pandas()
            processed_rows += len(chunk_df)
            
            print(f"\r📊 처리 중: {chunk_idx+1}/{total_chunks} 청크 "
                  f"({(chunk_idx+1)/total_chunks*100:.1f}%)", end="", flush=True)
            
            # 각 컬럼별로 누적 통계 계산 (double precision)
            for col in numeric_cols:
                if col in chunk_df.columns:
                    col_data = chunk_df[col].dropna()
                    if len(col_data) > 0:
                        # double precision으로 변환하여 계산
                        col_data_double = col_data.astype(np.float64)
                        
                        stats = stats_accumulator[col]
                        stats['sum'] += col_data_double.sum()
                        stats['sum_sq'] += (col_data_double ** 2).sum()
                        stats['count'] += len(col_data_double)
                        stats['min'] = min(stats['min'], col_data_double.min())
                        stats['max'] = max(stats['max'], col_data_double.max())
        
        print(f"\n✅ 모든 청크 처리 완료!")
        
        # 최종 통계 계산 (double precision)
        final_stats = {}
        for col, stats in stats_accumulator.items():
            if stats['count'] > 0:
                # double precision으로 정확한 mean/std 계산
                mean = stats['sum'] / stats['count']
                variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
                std = np.sqrt(max(0, variance))  # 음수 방지
                
                final_stats[col] = {
                    'mean': float(mean),      # double precision 유지
                    'std': float(std),        # double precision 유지
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'count': int(stats['count'])
                }
        
        print(f"✅ 정확한 통계 계산 완료: {len(final_stats)}개 컬럼")
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
        # 데이터 로드 (청크 단위, double precision)
        print("\n📊 데이터 로드 중...")
        stats = load_parquet_chunked_precise(train_data_path)
        
        # 제외할 컬럼들 필터링
        if exclude_cols:
            stats = {col: stat for col, stat in stats.items() if col not in exclude_cols}
        
        # 특정 컬럼만 계산하는 경우
        if feature_cols:
            stats = {col: stat for col, stat in stats.items() if col in feature_cols}
        
        print(f"✅ 청크 단위 처리 완료 (double precision)")
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
    parser.add_argument('--train_data', required=True, help='Train 데이터 파일 경로 (예: data/train.parquet)')
    parser.add_argument('--output', default='results/normalization_stats.json', help='출력 파일 경로')
    parser.add_argument('--exclude', nargs='*', default=['ID', 'clicked', 'seq'], help='제외할 컬럼들')
    
    args = parser.parse_args()
    
    # 결과 파일 경로 설정
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.getcwd(), output_path)
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 통계 계산 실행
    compute_normalization_stats(
        train_data_path=args.train_data,
        output_path=output_path,
        exclude_cols=set(args.exclude)
    )


if __name__ == "__main__":
    main()
