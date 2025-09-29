#!/usr/bin/env python3
"""
데이터셋을 10-fold로 분할하는 스크립트
- feat_e_3 missing 데이터: 모든 fold에 할당
- feat_e_3 available + clicked=1: 모든 fold에 할당  
- feat_e_3 available + clicked=0: 10-fold로 분할
- 각 fold별로 별도 parquet 파일 생성: train_fold1.parquet, ..., train_fold10.parquet
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import gc
import pyarrow.parquet as pq

def create_classified_parquet_files():
    """분류된 데이터를 별도 parquet 파일로 저장 (즉시 저장 방식)"""
    print("📊 분류된 데이터 파일 생성/확인 중...")
    
    missing_file = 'data/missing_data.parquet'
    clicked_1_file = 'data/clicked_1_data.parquet'
    clicked_0_file = 'data/clicked_0_data.parquet'
    
    # 기존 파일이 있으면 로드
    if os.path.exists(missing_file) and os.path.exists(clicked_1_file) and os.path.exists(clicked_0_file):
        print("  ✅ 기존 분류 파일 발견, 로드 중...")
        missing_data = pd.read_parquet(missing_file, engine='pyarrow')
        clicked_1_data = pd.read_parquet(clicked_1_file, engine='pyarrow')
        clicked_0_data = pd.read_parquet(clicked_0_file, engine='pyarrow')
        
        print(f"  데이터 분포:")
        print(f"    - feat_e_3 missing: {len(missing_data):,}개")
        print(f"    - feat_e_3 available + clicked=1: {len(clicked_1_data):,}개")
        print(f"    - feat_e_3 available + clicked=0: {len(clicked_0_data):,}개")
        
        return missing_data, clicked_1_data, clicked_0_data
    
    # 기존 파일이 없으면 생성
    print("  📝 새로운 분류 파일 생성 중...")
    
    # 청크 단위로 처리하여 즉시 저장
    chunk_size = 10000  # 1만개씩 처리
    parquet_file = pq.ParquetFile('./data/train.parquet')
    total_rows = parquet_file.metadata.num_rows
    print(f"  전체 데이터: {total_rows:,}개 행")
    
    # 각 분류별 카운터 초기화
    missing_count = 0
    clicked_1_count = 0
    clicked_0_count = 0
    
    # 첫 번째 청크로 파일 초기화
    first_batch = True
    
    print("  🔄 청크 단위로 데이터 분류 및 즉시 저장 중...")
    batch_count = 0
    
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        chunk = batch.to_pandas()
        
        # 1. feat_e_3 missing 데이터 처리
        missing_chunk = chunk[chunk['feat_e_3'].isna()]
        if len(missing_chunk) > 0:
            if first_batch:
                missing_chunk.to_parquet(missing_file, engine='pyarrow', compression='snappy', index=False)
            else:
                # 기존 파일에 추가
                existing = pd.read_parquet(missing_file, engine='pyarrow')
                combined = pd.concat([existing, missing_chunk], ignore_index=True)
                combined.to_parquet(missing_file, engine='pyarrow', compression='snappy', index=False)
                del existing, combined
            missing_count += len(missing_chunk)
        
        # 2. feat_e_3 available + clicked=1 데이터 처리
        clicked_1_chunk = chunk[(chunk['feat_e_3'].notna()) & (chunk['clicked'] == 1)]
        if len(clicked_1_chunk) > 0:
            if first_batch:
                clicked_1_chunk.to_parquet(clicked_1_file, engine='pyarrow', compression='snappy', index=False)
            else:
                # 기존 파일에 추가
                existing = pd.read_parquet(clicked_1_file, engine='pyarrow')
                combined = pd.concat([existing, clicked_1_chunk], ignore_index=True)
                combined.to_parquet(clicked_1_file, engine='pyarrow', compression='snappy', index=False)
                del existing, combined
            clicked_1_count += len(clicked_1_chunk)
        
        # 3. feat_e_3 available + clicked=0 데이터 처리
        clicked_0_chunk = chunk[(chunk['feat_e_3'].notna()) & (chunk['clicked'] == 0)]
        if len(clicked_0_chunk) > 0:
            if first_batch:
                clicked_0_chunk.to_parquet(clicked_0_file, engine='pyarrow', compression='snappy', index=False)
            else:
                # 기존 파일에 추가
                existing = pd.read_parquet(clicked_0_file, engine='pyarrow')
                combined = pd.concat([existing, clicked_0_chunk], ignore_index=True)
                combined.to_parquet(clicked_0_file, engine='pyarrow', compression='snappy', index=False)
                del existing, combined
            clicked_0_count += len(clicked_0_chunk)
        
        # 메모리 정리
        del chunk, missing_chunk, clicked_1_chunk, clicked_0_chunk
        gc.collect()
        
        first_batch = False
        batch_count += 1
        
        # 진행상황 출력 (매우 자주)
        if batch_count % 10 == 0:
            processed = batch_count * chunk_size
            print(f"    처리 진행: {processed:,}개 / {total_rows:,}개 ({processed/total_rows*100:.1f}%)")
    
    print(f"  데이터 분포:")
    print(f"    - feat_e_3 missing: {missing_count:,}개")
    print(f"    - feat_e_3 available + clicked=1: {clicked_1_count:,}개")
    print(f"    - feat_e_3 available + clicked=0: {clicked_0_count:,}개")
    
    print(f"  ✅ 분류 파일 저장 완료:")
    print(f"    - {missing_file}: {os.path.getsize(missing_file)/(1024*1024):.1f} MB")
    print(f"    - {clicked_1_file}: {os.path.getsize(clicked_1_file)/(1024*1024):.1f} MB")
    print(f"    - {clicked_0_file}: {os.path.getsize(clicked_0_file)/(1024*1024):.1f} MB")
    
    # 최종 로드하여 반환
    print("  📂 최종 데이터 로드 중...")
    missing_data = pd.read_parquet(missing_file, engine='pyarrow')
    clicked_1_data = pd.read_parquet(clicked_1_file, engine='pyarrow')
    clicked_0_data = pd.read_parquet(clicked_0_file, engine='pyarrow')
    
    print(f"  ✅ 분류 파일 생성 완료!")
    print(f"  데이터 분포:")
    print(f"    - feat_e_3 missing: {len(missing_data):,}개")
    print(f"    - feat_e_3 available + clicked=1: {len(clicked_1_data):,}개")
    print(f"    - feat_e_3 available + clicked=0: {len(clicked_0_data):,}개")
    
    return missing_data, clicked_1_data, clicked_0_data

def create_fold_parquet_files():
    """각 fold별로 별도 parquet 파일 생성 (메모리 효율적)"""
    print("🚀 Fold별 parquet 파일 생성 시작")
    print("=" * 60)
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 분류된 데이터 로드/생성
    missing_data, clicked_1_data, clicked_0_data = create_classified_parquet_files()
    
    # 2. clicked=0 데이터를 10-fold로 분할
    print("🔄 clicked=0 데이터를 10-fold로 분할 중...")
    if len(clicked_0_data) > 0:
        n_clicked_0 = len(clicked_0_data)
        base_size = n_clicked_0 // 9
        remainder = n_clicked_0 % 9
        
        # 데이터를 섞기
        clicked_0_data = clicked_0_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # fold별로 분할
        fold_clicked_0_data = {}
        start_idx = 0
        
        for fold in range(1, 11):
            if fold <= 9:
                fold_size = base_size
                if fold == 9:
                    fold_size += remainder
                end_idx = start_idx + fold_size
                fold_clicked_0_data[fold] = clicked_0_data.iloc[start_idx:end_idx].copy()
                start_idx = end_idx
            else:
                # 10번째 fold: 모든 clicked=0 데이터
                fold_clicked_0_data[fold] = clicked_0_data.copy()
    
    # 3. 각 fold별 parquet 파일 생성
    print("💾 Fold별 parquet 파일 생성 중...")
    fold_counts = {}
    
    for fold in range(1, 11):
        print(f"  📁 Fold {fold} 생성 중...")
        
        # 현재 fold의 데이터 구성
        fold_data_list = []
        
        # 1. feat_e_3 missing 데이터 추가 (모든 fold에 공유)
        fold_data_list.append(missing_data.copy())
        
        # 2. feat_e_3 available + clicked=1 데이터 추가 (모든 fold에 공유)
        fold_data_list.append(clicked_1_data.copy())
        
        # 3. feat_e_3 available + clicked=0 데이터 추가 (fold별 분할)
        if len(clicked_0_data) > 0:
            fold_data_list.append(fold_clicked_0_data[fold].copy())
        
        # 4. 모든 데이터 결합 및 섞기
        fold_df = pd.concat(fold_data_list, ignore_index=True)
        fold_df = fold_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # 5. parquet 파일로 저장
        filename = f'data/train_fold{fold}.parquet'
        fold_df.to_parquet(filename, engine='pyarrow', compression='snappy', index=False)
        
        fold_counts[fold] = len(fold_df)
        print(f"    ✅ {filename}: {len(fold_df):,}개 행, {os.path.getsize(filename)/(1024*1024):.1f} MB")
        
        # 메모리 정리
        del fold_df, fold_data_list
        gc.collect()
    
    # 4. 최종 요약
    print(f"\n✅ Fold별 parquet 파일 생성 완료!")
    print(f"📊 각 fold 데이터 개수:")
    for fold in range(1, 11):
        print(f"  Fold {fold}: {fold_counts[fold]:,}개")
    
    print(f"\n📁 생성된 파일들:")
    for fold in range(1, 11):
        filename = f'train_fold{fold}.parquet'
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024 * 1024)
            print(f"  {filename}: {file_size:.1f} MB")
    
    # 5. 사용 예제 생성
    create_usage_example()
    
    return fold_counts

def create_usage_example():
    """사용 예제 생성"""
    example_code = '''#!/usr/bin/env python3
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
'''
    
    with open('load_fold_example.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print(f"📝 사용 예제 생성: load_fold_example.py")

def main():
    """메인 함수"""
    print("🚀 데이터셋 10-fold 분할 시작")
    print("=" * 60)
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Fold별 parquet 파일 생성
        fold_counts = create_fold_parquet_files()
        
        print(f"\n🎉 모든 작업 완료!")
        print(f"📅 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 메모리 정리
        gc.collect()
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()