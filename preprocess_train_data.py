#!/usr/bin/env python3
"""
훈련 데이터 전처리 및 텐서 저장 스크립트
train.parquet 파일을 읽어서 FeatureProcessor로 변환한 후 torch 텐서로 저장
"""

import pandas as pd
import torch
import numpy as np
import os
from data_loader import FeatureProcessor
import yaml
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time

def process_batch_save_files(args):
    """단일 배치를 처리하고 파일로 저장하는 함수 (멀티프로세싱용)"""
    batch_data, batch_indices, processor, folders = args
    
    # FeatureProcessor로 변환
    x_categorical, x_numerical, sequences, nan_mask = processor.transform(batch_data)
    
    # 각 샘플별로 개별 파일로 저장
    for j, (idx, row) in enumerate(batch_data.iterrows()):
        sample_id = batch_indices[j]
        
        # 개별 텐서 저장
        torch.save(x_categorical[j], os.path.join(folders['categorical'], f'{sample_id}.pt'))
        torch.save(x_numerical[j], os.path.join(folders['numerical'], f'{sample_id}.pt'))
        torch.save(sequences[j], os.path.join(folders['sequences'], f'{sample_id}.pt'))
        torch.save(nan_mask[j], os.path.join(folders['nan_masks'], f'{sample_id}.pt'))
        torch.save(torch.tensor(row['clicked'], dtype=torch.float32), 
                  os.path.join(folders['targets'], f'{sample_id}.pt'))
    
    # 첫 번째 샘플의 형태 정보 반환
    return {
        'categorical_shape': x_categorical[0].shape,
        'numerical_shape': x_numerical[0].shape,
        'sequence_shape': sequences[0].shape,
        'nan_mask_shape': nan_mask[0].shape
    }

def preprocess_train_data_and_save_tensors():
    """train.parquet를 전처리하고 개별 텐서 파일로 저장 (메모리 효율적)"""
    
    # 현재 스크립트의 디렉토리 기준으로 상대경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Loading train.parquet...")
    # train.parquet 파일 로드
    train_path = os.path.join(script_dir, 'train.parquet')
    df = pd.read_parquet(train_path)
    print(f"Loaded {len(df)} samples")
    
    # ID를 0부터 시작하도록 재할당
    print("Reassigning IDs from 0...")
    df['ID'] = range(len(df))
    print(f"ID range: {df['ID'].min()} to {df['ID'].max()}")
    
    # FeatureProcessor 초기화 및 학습
    print("Initializing FeatureProcessor...")
    processor = FeatureProcessor()
    
    print("Fitting FeatureProcessor...")
    processor.fit(df)
    
    # 데이터 저장 디렉토리 생성 (새로운 구조)
    data_dir = os.path.join(script_dir, 'data')
    categorical_dir = os.path.join(data_dir, 'categorical')
    numerical_dir = os.path.join(data_dir, 'numerical')
    sequences_dir = os.path.join(data_dir, 'sequences')
    nan_masks_dir = os.path.join(data_dir, 'nan_masks')
    targets_dir = os.path.join(data_dir, 'targets')
    
    os.makedirs(categorical_dir, exist_ok=True)
    os.makedirs(numerical_dir, exist_ok=True)
    os.makedirs(sequences_dir, exist_ok=True)
    os.makedirs(nan_masks_dir, exist_ok=True)
    os.makedirs(targets_dir, exist_ok=True)
    
    # 배치 단위로 처리하여 메모리 효율성 확보
    # 멀티프로세싱을 고려하여 배치 크기 동적 조정
    num_workers = min(cpu_count(), 64)  # 최대 8개 프로세스 사용
    batch_size = max(2000, len(df) // (num_workers * 4))  # 프로세스당 4개 배치 정도
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    print(f"🚀 멀티프로세싱 처리: {total_batches}개 배치를 {num_workers}개 프로세스로 병렬 처리")
    print(f"💾 메모리 효율적 저장: 배치별로 즉시 저장하여 메모리 누적 방지")
    print(f"📊 배치 크기: {batch_size}개 샘플/배치")
    
    # 성능 측정 시작
    start_time = time.time()
    
    # 메타데이터용 변수들
    total_samples = len(df)
    sample_shapes = None
    
    # 폴더 정보 딕셔너리
    folders = {
        'categorical': categorical_dir,
        'numerical': numerical_dir,
        'sequences': sequences_dir,
        'nan_masks': nan_masks_dir,
        'targets': targets_dir
    }
    
    # 배치 데이터 준비
    batch_args = []
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx].copy()
        batch_indices = batch_df['ID'].values
        batch_args.append((batch_df, batch_indices, processor, folders))
    
    # 멀티프로세싱으로 병렬 처리
    with Pool(processes=num_workers) as pool:
        # tqdm과 함께 진행상황 표시
        results = list(tqdm(
            pool.imap(process_batch_save_files, batch_args),
            total=total_batches,
            desc="🚀 병렬 처리 및 저장"
        ))
    
    # 첫 번째 배치 결과에서 샘플 형태 정보 추출
    if results:
        sample_shapes = results[0]
    
    # 성능 측정 완료
    end_time = time.time()
    processing_time = end_time - start_time
    samples_per_second = total_samples / processing_time
    
    print(f"⚡ 처리 성능: {processing_time:.2f}초, {samples_per_second:.0f} 샘플/초")
    
    # FeatureProcessor 저장 (나중에 test 데이터 처리용)
    torch.save(processor, os.path.join(data_dir, 'feature_processor.pt'))
    
    # 메타데이터 저장
    metadata = {
        'num_samples': total_samples,
        'categorical_features': processor.categorical_features,
        'numerical_features': processor.numerical_features,
        'sequential_feature': processor.sequential_feature,
        'categorical_cardinalities': processor.categorical_cardinalities,
        'sample_shapes': sample_shapes,
        'storage_structure': 'individual_files',
        'processing_info': {
            'batch_size': batch_size,
            'total_batches': total_batches,
            'num_workers': num_workers,
            'processing_time_seconds': processing_time,
            'samples_per_second': samples_per_second
        },
        'folders': {
            'categorical': categorical_dir,
            'numerical': numerical_dir,
            'sequences': sequences_dir,
            'nan_masks': nan_masks_dir,
            'targets': targets_dir
        }
    }
    
    with open(os.path.join(data_dir, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 데이터 저장 완료: {data_dir}/")
    print(f"📁 저장 구조:")
    print(f"  - categorical/{'{ID}.pt'}")
    print(f"  - numerical/{'{ID}.pt'}")
    print(f"  - sequences/{'{ID}.pt'}")
    print(f"  - nan_masks/{'{ID}.pt'}")
    print(f"  - targets/{'{ID}.pt'}")
    print(f"📊 총 {total_samples}개 샘플 저장")
    print(f"🚀 멀티프로세싱으로 {num_workers}개 프로세스 사용하여 빠른 처리 완료")
    
    return metadata

if __name__ == "__main__":
    # 멀티프로세싱 환경에서 안전한 실행을 위한 설정
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    metadata = preprocess_train_data_and_save_tensors()
    print("Train data preprocessing completed successfully!")
