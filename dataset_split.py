#!/usr/bin/env python3
"""
데이터셋을 train/val/cal로 분할하는 스크립트
- 전체 데이터를 0.8 / 0.1 / 0.1 비율로 분할
- train_t.parquet (80%), train_v.parquet (10%), train_c.parquet (10%)로 저장
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import gc
import pyarrow.parquet as pq
import argparse
import psutil

def split_dataset(train_ratio=0.8, val_ratio=0.1, cal_ratio=0.1, random_state=42):
    """데이터셋을 train/val/cal로 분할"""
    print("🚀 데이터셋 분할 시작")
    print("=" * 60)
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 분할 비율: train={train_ratio}, val={val_ratio}, cal={cal_ratio}")
    
    # 비율 검증
    assert abs(train_ratio + val_ratio + cal_ratio - 1.0) < 1e-6, "비율의 합이 1이 되어야 합니다"
    
    # 출력 파일명
    train_file = 'data/train_t.parquet'
    val_file = 'data/train_v.parquet'
    cal_file = 'data/train_c.parquet'
    
    # 기존 파일 존재 여부 확인
    if os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(cal_file):
        print("\n⚠️  기존 분할 파일들이 존재합니다.")
        print(f"  - {train_file}: {os.path.getsize(train_file)/(1024*1024):.1f} MB")
        print(f"  - {val_file}: {os.path.getsize(val_file)/(1024*1024):.1f} MB")
        print(f"  - {cal_file}: {os.path.getsize(cal_file)/(1024*1024):.1f} MB")
        
        response = input("\n🔄 기존 파일을 덮어쓰시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("❌ 작업을 취소했습니다.")
            return
        print("\n📝 기존 파일을 덮어씁니다...")
    
    # 1. 전체 데이터 로드
    print("\n📂 전체 데이터 로드 중...")
    memory_info = psutil.virtual_memory()
    print(f"  💾 로드 전 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB / {memory_info.total/(1024**3):.1f}GB)")
    
    start_time = datetime.now()
    all_data = pd.read_parquet('data/train.parquet', engine='pyarrow')
    load_time = (datetime.now() - start_time).total_seconds()
    
    total_rows = len(all_data)
    print(f"  ✅ 전체 데이터: {total_rows:,}개 행 로드 완료 ({load_time:.2f}초)")
    
    memory_info = psutil.virtual_memory()
    print(f"  💾 로드 후 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB / {memory_info.total/(1024**3):.1f}GB)")
    
    # 데이터 분포 확인
    clicked_1_count = (all_data['clicked'] == 1).sum()
    clicked_0_count = (all_data['clicked'] == 0).sum()
    print(f"\n  📊 원본 데이터 분포:")
    print(f"    - clicked=1: {clicked_1_count:,}개 ({clicked_1_count/total_rows*100:.2f}%)")
    print(f"    - clicked=0: {clicked_0_count:,}개 ({clicked_0_count/total_rows*100:.2f}%)")
    
    # 2. 데이터 섞기
    print("\n🔀 데이터 섞기 중...")
    all_data = all_data.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print("  ✅ 데이터 섞기 완료")
    
    # 3. 데이터 분할
    print("\n✂️  데이터 분할 중...")
    train_size = int(total_rows * train_ratio)
    val_size = int(total_rows * val_ratio)
    cal_size = total_rows - train_size - val_size  # 나머지를 cal에 할당
    
    print(f"  📊 분할 크기:")
    print(f"    - train: {train_size:,}개 ({train_size/total_rows*100:.2f}%)")
    print(f"    - val:   {val_size:,}개 ({val_size/total_rows*100:.2f}%)")
    print(f"    - cal:   {cal_size:,}개 ({cal_size/total_rows*100:.2f}%)")
    
    # 데이터 분할
    train_data = all_data.iloc[:train_size].copy()
    val_data = all_data.iloc[train_size:train_size+val_size].copy()
    cal_data = all_data.iloc[train_size+val_size:].copy()
    
    # 원본 데이터 메모리 해제
    del all_data
    gc.collect()
    
    memory_info = psutil.virtual_memory()
    print(f"  💾 분할 후 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB / {memory_info.total/(1024**3):.1f}GB)")
    
    # 4. 각 split 데이터 저장
    print("\n💾 분할된 데이터 저장 중...")
    
    datasets = [
        ('train', train_data, train_file),
        ('val', val_data, val_file),
        ('cal', cal_data, cal_file)
    ]
    
    for split_name, data, filename in datasets:
        print(f"\n  📁 {split_name.upper()} 데이터 저장 중...")
        
        # 데이터 분포 확인
        clicked_1 = (data['clicked'] == 1).sum()
        clicked_0 = (data['clicked'] == 0).sum()
        total = len(data)
        
        print(f"    📊 데이터 분포:")
        print(f"      - clicked=1: {clicked_1:,}개 ({clicked_1/total*100:.2f}%)")
        print(f"      - clicked=0: {clicked_0:,}개 ({clicked_0/total*100:.2f}%)")
        
        # 파일 저장
        start_time = datetime.now()
        data.to_parquet(filename, engine='pyarrow', compression='snappy', index=False)
        save_time = (datetime.now() - start_time).total_seconds()
        
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"    ✅ 저장 완료: {filename}")
        print(f"      - 크기: {file_size:.1f} MB")
        print(f"      - 시간: {save_time:.2f}초")
        
        # 메모리 정리
        del data
        gc.collect()
        
        memory_info = psutil.virtual_memory()
        print(f"    💾 저장 후 메모리: {memory_info.percent:.1f}% ({memory_info.used/(1024**3):.1f}GB / {memory_info.total/(1024**3):.1f}GB)")
    
    # 5. 최종 요약
    print("\n" + "=" * 60)
    print("✅ 데이터셋 분할 완료!")
    print(f"📅 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n📁 생성된 파일들:")
    for _, _, filename in datasets:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024 * 1024)
            print(f"  {filename}: {file_size:.1f} MB")
    
    # 6. 사용 예제 생성
    create_usage_example()

def create_usage_example():
    """사용 예제 생성"""
    example_code = '''#!/usr/bin/env python3
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
    print("\\n[TRAIN 데이터]")
    train_data = load_split_data('train')
    
    # Validation 데이터 로딩
    print("\\n[VALIDATION 데이터]")
    val_data = load_split_data('val')
    
    # Calibration 데이터 로딩
    print("\\n[CALIBRATION 데이터]")
    cal_data = load_split_data('cal')
    
    print("\\n✅ 모든 데이터 로딩 완료!")

if __name__ == "__main__":
    main()
'''
    
    with open('load_split_example.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print(f"\n📝 사용 예제 생성: load_split_example.py")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='데이터셋을 train/val/cal로 분할')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train 데이터 비율 (기본값: 0.8)')
    parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation 데이터 비율 (기본값: 0.1)')
    parser.add_argument('--cal-ratio', type=float, default=0.1, help='Calibration 데이터 비율 (기본값: 0.1)')
    parser.add_argument('--random-state', type=int, default=42, help='랜덤 시드 (기본값: 42)')
    args = parser.parse_args()
    
    try:
        split_dataset(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            cal_ratio=args.cal_ratio,
            random_state=args.random_state
        )
        
        print("\n🎉 모든 작업 완료!")
        
        # 메모리 정리
        gc.collect()
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()

