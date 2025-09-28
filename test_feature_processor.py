#!/usr/bin/env python3
"""
실제 데이터로 FeatureProcessor 테스트
"""

import pandas as pd
from data_loader import FeatureProcessor, create_data_loaders

def test_with_real_data():
    """실제 데이터로 FeatureProcessor 테스트"""
    
    print("📊 실제 데이터 로드 중...")
    
    # 실제 데이터 로드 (샘플링)
    try:
        train_df = pd.read_parquet("train.parquet")
        print(f"✅ 전체 데이터 로드: {train_df.shape}")
        
        # 샘플링 (테스트용)
        sample_size = 10000
        train_sample = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        print(f"✅ 샘플 데이터: {train_sample.shape}")
        
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    
    # FeatureProcessor 테스트
    print("\n🔧 FeatureProcessor 테스트...")
    processor = FeatureProcessor()
    processor.fit(train_sample)
    
    print(f"범주형 피처: {processor.categorical_features}")
    print(f"범주형 피처 카디널리티: {processor.categorical_cardinalities}")
    print(f"수치형 피처 개수: {len(processor.numerical_features)}")
    print(f"제외된 피처: {processor.excluded_features}")
    
    # 데이터 변환 테스트
    print("\n🔄 데이터 변환 테스트...")
    x_cat, x_num, seqs, nan_mask = processor.transform(train_sample.head(5))
    
    print(f"변환 결과:")
    print(f"  범주형: {x_cat.shape}, dtype: {x_cat.dtype}")
    print(f"  수치형: {x_num.shape}, dtype: {x_num.dtype}")
    print(f"  시퀀스: {len(seqs)}개, 첫 번째 길이: {len(seqs[0])}")
    print(f"  NaN 마스크: {nan_mask.shape}, dtype: {nan_mask.dtype}")
    print(f"  NaN 마스크 샘플: {nan_mask[0]}")
    
    # 데이터 로더 테스트
    print("\n📦 데이터 로더 테스트...")
    
    # 훈련/검증 분할
    train_size = int(0.8 * len(train_sample))
    train_data = train_sample[:train_size]
    val_data = train_sample[train_size:]
    
    # Transformer 모델용 데이터 로더
    print("Transformer 모델용 데이터 로더 생성...")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, feature_processor = create_data_loaders(
        train_data, val_data, None, list(train_sample.columns), 'seq', 'clicked', 
        batch_size=16, model_type="tabular_transformer"
    )
    
    # 배치 테스트
    print("\n배치 테스트:")
    for i, batch in enumerate(train_loader):
        print(f"배치 {i+1}:")
        print(f"  범주형: {batch['x_categorical'].shape}")
        print(f"  수치형: {batch['x_numerical'].shape}")
        print(f"  시퀀스: {batch['seqs'].shape}")
        print(f"  시퀀스 길이: {batch['seq_lengths'].shape}")
        print(f"  NaN 마스크: {batch['nan_mask'].shape}")
        print(f"  타겟: {batch['ys'].shape}")
        print(f"  NaN 마스크 샘플: {batch['nan_mask'][0]}")
        break
    
    print("\n✅ 모든 테스트 완료!")

if __name__ == "__main__":
    test_with_real_data()
