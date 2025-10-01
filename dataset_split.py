#!/usr/bin/env python3
"""
데이터셋을 10-fold로 분할하는 스크립트
- clicked=1 데이터: 모든 fold에 할당
- clicked=0 데이터: 10-fold로 분할
- 각 fold별로 별도 parquet 파일 생성: train_fold1.parquet, ..., train_fold10.parquet
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import gc
import pyarrow.parquet as pq
import argparse
import psutil

def create_classified_parquet_files():
    """분류된 데이터를 별도 parquet 파일로 저장 (전체 데이터 한번에 로드 방식)"""
    print("📊 분류된 데이터 파일 생성/확인 중...")
    
    clicked_1_file = 'data/clicked_1_data.parquet'
    clicked_0_file = 'data/clicked_0_data.parquet'
    
    # 기존 파일 존재 여부 확인
    clicked_1_exists = os.path.exists(clicked_1_file)
    clicked_0_exists = os.path.exists(clicked_0_file)
    
    if clicked_1_exists and clicked_0_exists:
        print("  📂 기존 분류 파일들이 존재합니다. 로드 중...")
        
        # 기존 파일들 로드
        clicked_1_data = pd.read_parquet(clicked_1_file, engine='pyarrow')
        clicked_0_data = pd.read_parquet(clicked_0_file, engine='pyarrow')
        
        clicked_1_count = len(clicked_1_data)
        clicked_0_count = len(clicked_0_data)
        
        print(f"    ✅ 기존 파일 로드 완료:")
        print(f"      - clicked=1: {clicked_1_count:,}개")
        print(f"      - clicked=0: {clicked_0_count:,}개")
        
    else:
        print("  📝 새로운 분류 파일 생성 중...")
        
        # 전체 데이터 한 번에 로드
        print("  📂 전체 데이터 로드 중...")
        memory_info = psutil.virtual_memory()
        print(f"    💾 로드 전 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB / {memory_info.total/(1024**3):.1f}GB)")
        
        all_data = pd.read_parquet('./data/train.parquet', engine='pyarrow')
        total_rows = len(all_data)
        print(f"  전체 데이터: {total_rows:,}개 행 로드 완료")
        
        memory_info = psutil.virtual_memory()
        print(f"    💾 로드 후 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB / {memory_info.total/(1024**3):.1f}GB)")
        
        # 1. clicked=1 데이터 처리
        print("  🔍 clicked=1 데이터 분류 및 저장 중...")
        clicked_1_data = all_data[all_data['clicked'] == 1]
        clicked_1_count = len(clicked_1_data)
        
        if clicked_1_count > 0:
            clicked_1_data.to_parquet(clicked_1_file, engine='pyarrow', compression='snappy', index=False)
            print(f"    ✅ clicked=1 데이터 저장: {clicked_1_count:,}개")
            
            memory_info = psutil.virtual_memory()
            print(f"    💾 clicked=1 처리 후 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB)")
        
        # 2. clicked=0 데이터 처리
        print("  🔍 clicked=0 데이터 분류 및 저장 중...")
        clicked_0_data = all_data[all_data['clicked'] == 0]
        clicked_0_count = len(clicked_0_data)
        
        if clicked_0_count > 0:
            clicked_0_data.to_parquet(clicked_0_file, engine='pyarrow', compression='snappy', index=False)
            print(f"    ✅ clicked=0 데이터 저장: {clicked_0_count:,}개")
            
            memory_info = psutil.virtual_memory()
            print(f"    💾 clicked=0 처리 후 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB)")
        
        # 전체 데이터 메모리 해제
        del all_data
        gc.collect()
        
        memory_info = psutil.virtual_memory()
        print(f"    💾 전체 데이터 해제 후 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB)")
    

        print(f"  ✅ 분류 파일 저장 완료:")
        print(f"    - {clicked_1_file}: {os.path.getsize(clicked_1_file)/(1024*1024):.1f} MB")
        print(f"    - {clicked_0_file}: {os.path.getsize(clicked_0_file)/(1024*1024):.1f} MB")


    print(f"  데이터 분포:")
    print(f"    - clicked=1: {clicked_1_count:,}개")
    print(f"    - clicked=0: {clicked_0_count:,}개")    
    return clicked_1_data, clicked_0_data

def create_fold_parquet_files():
    """각 fold별로 별도 parquet 파일 생성 (메모리 효율적)"""
    print("🚀 Fold별 parquet 파일 생성 시작")
    print("=" * 60)
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 분류된 데이터 로드/생성
    clicked_1_data, clicked_0_data = create_classified_parquet_files()
    
    # 2. clicked=0 데이터를 10-fold로 분할
    print("🔄 clicked=0 데이터를 10-fold로 분할 중...")
    if len(clicked_0_data) > 0:
        n_clicked_0 = len(clicked_0_data)
        base_size = n_clicked_0 // 10
        remainder = n_clicked_0 % 10
        
        # 데이터를 섞기
        clicked_0_data = clicked_0_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # fold별로 분할
        fold_clicked_0_data = {}
        start_idx = 0
        
        for fold in range(1, 11):  # 1부터 10까지
            fold_size = base_size
            if fold == 10:
                fold_size += remainder  # 10번째 fold에 나머지 데이터 포함
            end_idx = start_idx + fold_size
            fold_clicked_0_data[fold] = clicked_0_data.iloc[start_idx:end_idx].copy()
            start_idx = end_idx
    
    # 3. 각 fold별 parquet 파일 생성
    print("💾 Fold별 parquet 파일 생성 중...")
    fold_counts = {}
    
    for fold in range(1, 11):  # 1부터 10까지
        print(f"  📁 Fold {fold} 생성 중...")
        
        memory_info = psutil.virtual_memory()
        print(f"    💾 Fold {fold} 시작 전 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB)")
        
        # 현재 fold의 데이터 구성
        fold_data_list = []
        
        # 1. clicked=1 데이터 추가 (모든 fold에 공유)
        print(f"    🔄 clicked=1 데이터 복사 중...")
        fold_data_list.append(clicked_1_data.copy())
        memory_info = psutil.virtual_memory()
        print(f"      💾 clicked=1 복사 후 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB)")
        
        # 2. clicked=0 데이터 추가 (fold별 분할)
        if len(clicked_0_data) > 0:
            print(f"    🔄 clicked=0 데이터 복사 중...")
            fold_data_list.append(fold_clicked_0_data[fold].copy())
            memory_info = psutil.virtual_memory()
            print(f"      💾 clicked=0 복사 후 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB)")
        
        # 3. 모든 데이터 결합 및 섞기
        print(f"    🔄 데이터 결합 및 섞기 중...")
        fold_df = pd.concat(fold_data_list, ignore_index=True)
        fold_df = fold_df.sample(frac=1, random_state=42).reset_index(drop=True)
        memory_info = psutil.virtual_memory()
        print(f"      💾 데이터 결합 후 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB)")
        
        # 4. parquet 파일로 저장
        filename = f'data/train_fold{fold}.parquet'
        print(f"    💾 파일 저장 중: {filename}")
        fold_df.to_parquet(filename, engine='pyarrow', compression='snappy', index=False)
        
        fold_counts[fold] = len(fold_df)
        print(f"    ✅ {filename}: {len(fold_df):,}개 행, {os.path.getsize(filename)/(1024*1024):.1f} MB")
        
        # 메모리 정리
        print(f"    🗑️ 메모리 정리 중...")
        del fold_df, fold_data_list
        gc.collect()
        
        memory_info = psutil.virtual_memory()
        print(f"    💾 Fold {fold} 완료 후 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB)")
        print("    " + "="*50)
    
    # 4. 최종 요약
    print(f"\n✅ Fold별 parquet 파일 생성 완료!")
    print(f"📊 각 fold 데이터 개수:")
    for fold in range(1, 11):  # 1부터 10까지
        print(f"  Fold {fold}: {fold_counts[fold]:,}개")
    
    print(f"\n📁 생성된 파일들:")
    for fold in range(1, 11):  # 1부터 10까지
        filename = f'data/train_fold{fold}.parquet'
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
    parser = argparse.ArgumentParser(description='10-fold 데이터셋 분할')
    args = parser.parse_args()
    
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