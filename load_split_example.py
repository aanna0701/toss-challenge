#!/usr/bin/env python3
"""
분할된 데이터 로딩 사용 예제
"""

import pandas as pd
from datetime import datetime

def load_split_data(split_type='train'):
    """분할된 데이터 로드
    
    Args:
        split_type: 'train', 'val', 'cal' 중 하나
    """
    filename_map = {
        'train': 'data/train_t.parquet',
        'val': 'data/train_v.parquet',
        'cal': 'data/train_c.parquet'
    }
    
    if split_type not in filename_map:
        raise ValueError(f"split_type은 {list(filename_map.keys())} 중 하나여야 합니다")
    
    filename = filename_map[split_type]
    print(f"📊 {split_type.upper()} 데이터 로딩: {filename}")
    
    start_time = datetime.now()
    data = pd.read_parquet(filename, engine='pyarrow')
    load_time = (datetime.now() - start_time).total_seconds()
    
    print(f"  ✅ 로딩 완료: {len(data):,}개 행, {load_time:.2f}초")
    print(f"  📊 데이터 분포:")
    print(f"    - clicked=0: {(data['clicked']==0).sum():,}개")
    print(f"    - clicked=1: {(data['clicked']==1).sum():,}개")
    
    return data

def main():
    """메인 함수"""
    print("🚀 분할된 데이터 로딩 테스트")
    print("=" * 60)
    
    # Train 데이터 로딩
    print("\n[TRAIN 데이터]")
    train_data = load_split_data('train')
    
    # Validation 데이터 로딩
    print("\n[VALIDATION 데이터]")
    val_data = load_split_data('val')
    
    # Calibration 데이터 로딩
    print("\n[CALIBRATION 데이터]")
    cal_data = load_split_data('cal')
    
    print("\n✅ 모든 데이터 로딩 완료!")

if __name__ == "__main__":
    main()
