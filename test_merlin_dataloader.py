#!/usr/bin/env python3
"""
NVIDIA Merlin 데이터로더 테스트 스크립트
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(n_samples=100000, n_categorical=5, n_numerical=20):
    """대용량 샘플 데이터 생성"""
    logger.info(f"📊 샘플 데이터 생성 중: {n_samples:,}개 행")
    
    np.random.seed(42)
    
    # 범주형 피처 생성
    categorical_data = {}
    for i in range(n_categorical):
        categories = [f'cat_{i}_val_{j}' for j in range(np.random.randint(5, 20))]
        categorical_data[f'cat_{i}'] = np.random.choice(categories, n_samples)
    
    # 수치형 피처 생성
    numerical_data = {}
    for i in range(n_numerical):
        numerical_data[f'num_{i}'] = np.random.normal(0, 1, n_samples)
    
    # 시퀀스 피처 생성
    sequence_data = {}
    for i in range(n_samples):
        seq_length = np.random.randint(1, 10)
        seq_values = np.random.uniform(-1, 1, seq_length)
        sequence_data[f'seq_{i}'] = ','.join(map(str, seq_values))
    
    # 타겟 생성
    target = np.random.binomial(1, 0.1, n_samples)  # 10% positive rate
    
    # DataFrame 생성
    data = {
        **categorical_data, 
        **numerical_data, 
        'seq': [sequence_data[f'seq_{i}'] for i in range(n_samples)],
        'clicked': target,
        'ID': range(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    logger.info(f"✅ 샘플 데이터 생성 완료: {df.shape}")
    logger.info(f"   • 범주형 피처: {n_categorical}개")
    logger.info(f"   • 수치형 피처: {n_numerical}개")
    logger.info(f"   • 타겟 분포: {df['clicked'].value_counts().to_dict()}")
    
    return df

def test_merlin_dataloader():
    """Merlin 데이터로더 테스트"""
    logger.info("🧪 NVIDIA Merlin 데이터로더 테스트 시작")
    logger.info("=" * 60)
    
    try:
        # Merlin 데이터로더 임포트
        from data_loader_merlin import (
            MerlinFeatureProcessor, 
            MerlinClickDataset, 
            create_merlin_dataloader,
            MERLIN_AVAILABLE
        )
        
        if not MERLIN_AVAILABLE:
            logger.warning("⚠️  NVIDIA Merlin이 설치되지 않았습니다. 표준 데이터로더로 테스트합니다.")
        
        # 샘플 데이터 생성
        df = create_sample_data(n_samples=50000, n_categorical=3, n_numerical=10)
        
        # 설정 생성
        config = {
            'MODEL': {
                'FEATURES': {
                    'CATEGORICAL': ['cat_0', 'cat_1', 'cat_2'],
                    'NUMERICAL': [],
                    'EXCLUDED': [],
                    'SEQUENTIAL': 'seq'
                }
            }
        }
        
        # 피처 프로세서 생성 및 학습
        logger.info("🔧 피처 프로세서 학습 중...")
        processor = MerlinFeatureProcessor(
            config=config,
            normalization_stats_path="analysis/results/normalization_stats.json",
            use_merlin=MERLIN_AVAILABLE
        )
        
        # 더미 정규화 통계 생성 (실제 파일이 없는 경우)
        if not os.path.exists("analysis/results/normalization_stats.json"):
            logger.info("📊 더미 정규화 통계 생성 중...")
            os.makedirs("analysis/results", exist_ok=True)
            
            dummy_stats = {
                'statistics': {}
            }
            
            # 수치형 피처에 대한 더미 통계 생성
            numerical_features = [col for col in df.columns 
                                if col.startswith('num_') and col not in ['clicked', 'ID', 'seq']]
            
            for feat in numerical_features:
                dummy_stats['statistics'][feat] = {
                    'mean': float(df[feat].mean()),
                    'std': float(df[feat].std())
                }
            
            import json
            with open("analysis/results/normalization_stats.json", 'w') as f:
                json.dump(dummy_stats, f, indent=2)
            
            logger.info("✅ 더미 정규화 통계 생성 완료")
        
        # 피처 프로세서 학습
        processor.fit(df)
        
        # 데이터셋 생성
        logger.info("📊 데이터셋 생성 중...")
        dataset = MerlinClickDataset(
            df=df,
            feature_processor=processor,
            target_col='clicked',
            has_target=True,
            has_id=True,
            use_merlin=MERLIN_AVAILABLE
        )
        
        logger.info(f"✅ 데이터셋 생성 완료: {len(dataset)}개 샘플")
        
        # DataLoader 생성
        logger.info("🚀 DataLoader 생성 중...")
        dataloader = create_merlin_dataloader(
            dataset=dataset,
            batch_size=1024,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        logger.info(f"✅ DataLoader 생성 완료: {len(dataloader)}개 배치")
        
        # 데이터로딩 성능 테스트
        logger.info("⚡ 데이터로딩 성능 테스트 중...")
        
        start_time = time.time()
        batch_count = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            batch_count += 1
            total_samples += len(batch['ys'])
            
            # 배치 정보 확인
            if batch_idx == 0:
                logger.info(f"📊 첫 번째 배치 정보:")
                logger.info(f"   • 범주형 피처 shape: {batch['x_categorical'].shape}")
                logger.info(f"   • 수치형 피처 shape: {batch['x_numerical'].shape}")
                logger.info(f"   • 시퀀스 피처 shape: {batch['seqs'].shape}")
                logger.info(f"   • 타겟 shape: {batch['ys'].shape}")
                logger.info(f"   • 시퀀스 길이 shape: {batch['seq_lengths'].shape}")
            
            # 처음 5개 배치만 테스트
            if batch_idx >= 4:
                break
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"⚡ 성능 테스트 결과:")
        logger.info(f"   • 처리된 배치: {batch_count}개")
        logger.info(f"   • 처리된 샘플: {total_samples:,}개")
        logger.info(f"   • 총 소요 시간: {elapsed_time:.2f}초")
        logger.info(f"   • 배치당 평균 시간: {elapsed_time/batch_count:.3f}초")
        logger.info(f"   • 샘플당 평균 시간: {elapsed_time/total_samples*1000:.3f}ms")
        logger.info(f"   • 처리 속도: {total_samples/elapsed_time:.0f} 샘플/초")
        
        # 메모리 사용량 확인
        if torch.cuda.is_available():
            logger.info(f"💾 GPU 메모리 사용량: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        
        # Merlin 사용 여부 확인
        if MERLIN_AVAILABLE:
            logger.info("✅ NVIDIA Merlin이 성공적으로 사용되었습니다!")
        else:
            logger.info("⚠️  표준 PyTorch 데이터로더가 사용되었습니다.")
        
        logger.info("🎉 Merlin 데이터로더 테스트 완료!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """표준 데이터로더와 Merlin 데이터로더 성능 비교"""
    logger.info("📊 성능 비교 테스트 시작")
    logger.info("=" * 60)
    
    try:
        # 표준 데이터로더 임포트
        from data_loader import FeatureProcessor, ClickDataset, collate_fn_transformer_train
        
        # Merlin 데이터로더 임포트
        from data_loader_merlin import (
            MerlinFeatureProcessor, 
            MerlinClickDataset, 
            create_merlin_dataloader,
            MERLIN_AVAILABLE
        )
        
        # 샘플 데이터 생성
        df = create_sample_data(n_samples=20000, n_categorical=3, n_numerical=10)
        
        # 설정
        config = {
            'MODEL': {
                'FEATURES': {
                    'CATEGORICAL': ['cat_0', 'cat_1', 'cat_2'],
                    'NUMERICAL': [],
                    'EXCLUDED': [],
                    'SEQUENTIAL': 'seq'
                }
            }
        }
        
        # 더미 정규화 통계 생성
        if not os.path.exists("analysis/results/normalization_stats.json"):
            os.makedirs("analysis/results", exist_ok=True)
            
            dummy_stats = {'statistics': {}}
            numerical_features = [col for col in df.columns 
                                if col.startswith('num_') and col not in ['clicked', 'ID', 'seq']]
            
            for feat in numerical_features:
                dummy_stats['statistics'][feat] = {
                    'mean': float(df[feat].mean()),
                    'std': float(df[feat].std())
                }
            
            import json
            with open("analysis/results/normalization_stats.json", 'w') as f:
                json.dump(dummy_stats, f, indent=2)
        
        # 표준 데이터로더 테스트
        logger.info("🔧 표준 데이터로더 테스트 중...")
        
        standard_processor = FeatureProcessor(config, "analysis/results/normalization_stats.json")
        standard_processor.fit(df)
        
        standard_dataset = ClickDataset(df, standard_processor, 'clicked', has_target=True, has_id=True)
        standard_dataloader = DataLoader(
            standard_dataset, 
            batch_size=1024, 
            shuffle=True, 
            num_workers=2,
            collate_fn=collate_fn_transformer_train
        )
        
        start_time = time.time()
        standard_batches = 0
        for batch_idx, batch in enumerate(standard_dataloader):
            standard_batches += 1
            if batch_idx >= 4:  # 5개 배치만 테스트
                break
        standard_time = time.time() - start_time
        
        logger.info(f"📊 표준 데이터로더 결과:")
        logger.info(f"   • 처리 시간: {standard_time:.2f}초")
        logger.info(f"   • 처리된 배치: {standard_batches}개")
        
        # Merlin 데이터로더 테스트
        if MERLIN_AVAILABLE:
            logger.info("🚀 Merlin 데이터로더 테스트 중...")
            
            merlin_processor = MerlinFeatureProcessor(
                config=config,
                normalization_stats_path="analysis/results/normalization_stats.json",
                use_merlin=True
            )
            merlin_processor.fit(df)
            
            merlin_dataset = MerlinClickDataset(
                df=df,
                feature_processor=merlin_processor,
                target_col='clicked',
                has_target=True,
                has_id=True,
                use_merlin=True
            )
            
            merlin_dataloader = create_merlin_dataloader(
                dataset=merlin_dataset,
                batch_size=1024,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            
            start_time = time.time()
            merlin_batches = 0
            for batch_idx, batch in enumerate(merlin_dataloader):
                merlin_batches += 1
                if batch_idx >= 4:  # 5개 배치만 테스트
                    break
            merlin_time = time.time() - start_time
            
            logger.info(f"📊 Merlin 데이터로더 결과:")
            logger.info(f"   • 처리 시간: {merlin_time:.2f}초")
            logger.info(f"   • 처리된 배치: {merlin_batches}개")
            
            # 성능 비교
            speedup = standard_time / merlin_time
            logger.info(f"⚡ 성능 비교 결과:")
            logger.info(f"   • Merlin 속도 향상: {speedup:.2f}x")
            logger.info(f"   • 시간 절약: {(standard_time - merlin_time):.2f}초")
        else:
            logger.info("⚠️  NVIDIA Merlin이 설치되지 않아 성능 비교를 건너뜁니다.")
        
        logger.info("🎉 성능 비교 테스트 완료!")
        
    except Exception as e:
        logger.error(f"❌ 성능 비교 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logger.info("🚀 NVIDIA Merlin 데이터로더 테스트 시작")
    
    # 기본 테스트
    success = test_merlin_dataloader()
    
    if success:
        # 성능 비교 테스트
        test_performance_comparison()
    
    logger.info("🏁 모든 테스트 완료!")
