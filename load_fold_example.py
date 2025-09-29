#!/usr/bin/env python3
"""
Fold별 데이터 로딩 사용 예제
"""

import pandas as pd
from datetime import datetime

def load_fold_data(fold_number):
    """특정 fold 데이터 로드"""
    filename = f'data/train_fold{fold_number}.parquet'
    print(f"📊 Fold {fold_number} 데이터 로딩: {filename}")
    
    start_time = datetime.now()
    data = pd.read_parquet(filename, engine='pyarrow')
    load_time = (datetime.now() - start_time).total_seconds()
    
    print(f"  ✅ 로딩 완료: {len(data):,}개 행, {load_time:.2f}초")
    print(f"  📊 데이터 분포:")
    print(f"    - feat_e_3 missing: {data['feat_e_3'].isna().sum():,}개")
    print(f"    - clicked=0: {(data['clicked']==0).sum():,}개")
    print(f"    - clicked=1: {(data['clicked']==1).sum():,}개")
    
    return data

def main():
    """메인 함수"""
    print("🚀 Fold별 데이터 로딩 테스트")
    print("=" * 40)
    
    # 모든 fold 데이터 로딩 테스트
    for fold in range(1, 11):
        data = load_fold_data(fold)
        print()
        
        # 메모리 정리
        del data

if __name__ == "__main__":
    main()
