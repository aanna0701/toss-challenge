#!/usr/bin/env python3
"""
디버깅용 테스트 스크립트 - 샘플링 10개로 Transformer 모델 테스트
"""

import os
import sys
import yaml
import pandas as pd
import torch
from datetime import datetime

# config_debug.yaml을 사용하도록 설정
def load_debug_config():
    """디버깅용 config 로드"""
    with open('config_debug.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # main.py의 CFG를 덮어쓰기
    import main
    main.CFG = config
    return config

def test_data_loading():
    """데이터 로딩 테스트"""
    print("🔍 1. 데이터 로딩 테스트...")
    
    try:
        from data_loader import load_and_preprocess_data
        train_data, test_data, feature_cols, seq_col, target_col = load_and_preprocess_data()
        
        print(f"✅ 데이터 로딩 성공:")
        print(f"   - Train shape: {train_data.shape}")
        print(f"   - Test shape: {test_data.shape}")
        print(f"   - Feature cols: {len(feature_cols)}")
        print(f"   - Seq col: {seq_col}")
        print(f"   - Target col: {target_col}")
        
        # 샘플 데이터 확인
        print(f"\n📊 샘플 데이터:")
        print(f"   - Train columns: {list(train_data.columns)[:10]}...")
        print(f"   - Train dtypes: {train_data.dtypes.value_counts().to_dict()}")
        print(f"   - Target distribution: {train_data[target_col].value_counts().to_dict()}")
        
        return train_data, test_data, feature_cols, seq_col, target_col
        
    except Exception as e:
        print(f"❌ 데이터 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None

def test_feature_processor():
    """FeatureProcessor 테스트"""
    print("\n🔍 2. FeatureProcessor 테스트...")
    
    try:
        from data_loader import FeatureProcessor
        
        # 더미 데이터로 테스트
        dummy_data = {
            'gender': ['1.0', '2.0', '1.0', '2.0', '1.0'],
            'age_group': ['7.0', '8.0', '6.0', '5.0', '4.0'],
            'inventory_id': ['2', '36', '37', '29', '42'],
            'day_of_week': ['5', '4', '1', '3', '6'],
            'hour': ['08', '07', '12', '20', '09'],
            'l_feat_1': [1.0, 2.0, 1.0, 2.0, 1.0],
            'feat_a_1': [1.5, 2.3, 0.8, 3.1, 1.9],
            'seq': ['1.0,2.0,3.0', '2.0,3.0', '1.0', '1.0,2.0,3.0,4.0', '2.0,3.0,4.0'],
            'clicked': [1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(dummy_data)
        
        processor = FeatureProcessor()
        processor.fit(df)
        
        print(f"✅ FeatureProcessor 생성 성공:")
        print(f"   - Categorical features: {processor.categorical_features}")
        print(f"   - Categorical cardinalities: {processor.categorical_cardinalities}")
        print(f"   - Numerical features: {len(processor.numerical_features)}")
        print(f"   - Excluded features: {processor.excluded_features}")
        
        # 변환 테스트
        x_cat, x_num, seqs, nan_mask = processor.transform(df)
        print(f"   - Categorical shape: {x_cat.shape}")
        print(f"   - Numerical shape: {x_num.shape}")
        print(f"   - Sequences count: {len(seqs)}")
        print(f"   - NaN mask shape: {nan_mask.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ FeatureProcessor 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loaders():
    """데이터 로더 테스트"""
    print("\n🔍 3. 데이터 로더 테스트...")
    
    try:
        from data_loader import create_data_loaders
        
        # 더미 데이터 생성
        dummy_data = {
            'gender': ['1.0', '2.0', '1.0', '2.0', '1.0'],
            'age_group': ['7.0', '8.0', '6.0', '5.0', '4.0'],
            'inventory_id': ['2', '36', '37', '29', '42'],
            'day_of_week': ['5', '4', '1', '3', '6'],
            'hour': ['08', '07', '12', '20', '09'],
            'l_feat_1': [1.0, 2.0, 1.0, 2.0, 1.0],
            'feat_a_1': [1.5, 2.3, 0.8, 3.1, 1.9],
            'seq': ['1.0,2.0,3.0', '2.0,3.0', '1.0', '1.0,2.0,3.0,4.0', '2.0,3.0,4.0'],
            'clicked': [1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(dummy_data)
        
        # Transformer 모델용 데이터 로더 생성
        train_loader, val_loader, test_loader, train_dataset, val_dataset, feature_processor = create_data_loaders(
            df, df, df, list(df.columns), 'seq', 'clicked', batch_size=2, model_type="tabular_transformer"
        )
        
        print(f"✅ 데이터 로더 생성 성공:")
        print(f"   - Train loader batches: {len(train_loader)}")
        print(f"   - Val loader batches: {len(val_loader)}")
        print(f"   - Test loader batches: {len(test_loader)}")
        
        # 배치 테스트
        for i, batch in enumerate(train_loader):
            print(f"   - Batch {i+1}:")
            print(f"     * Categorical: {batch['x_categorical'].shape}")
            print(f"     * Numerical: {batch['x_numerical'].shape}")
            print(f"     * Sequences: {batch['seqs'].shape}")
            print(f"     * Seq lengths: {batch['seq_lengths'].shape}")
            print(f"     * NaN mask: {batch['nan_mask'].shape}")
            print(f"     * Targets: {batch['ys'].shape}")
            if i >= 1:  # 첫 2개 배치만 확인
                break
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터 로더 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_creation():
    """모델 생성 테스트"""
    print("\n🔍 4. 모델 생성 테스트...")
    
    try:
        from model import create_tabular_transformer_model
        
        # Transformer 모델 생성
        print(f"   - 모델 생성 파라미터:")
        print(f"     * lstm_hidden: 32")
        print(f"     * hidden_dim: 192")
        print(f"     * n_heads: 8")
        print(f"     * n_layers: 3")
        
        model = create_tabular_transformer_model(
            num_categorical_features=5,
            categorical_cardinalities=[2, 8, 20, 7, 24],
            num_numerical_features=2,
            lstm_hidden=32,
            hidden_dim=192,
            n_heads=8,
            n_layers=3,
            device="cpu"  # 디버깅용으로 CPU 사용
        )
        
        print(f"   - 실제 LSTM 설정:")
        print(f"     * LSTM hidden_size: {model.lstm.hidden_size}")
        print(f"     * LSTM num_layers: {model.lstm.num_layers}")
        print(f"     * LSTM input_size: {model.lstm.input_size}")
        
        print(f"✅ Transformer 모델 생성 성공:")
        print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Forward pass 테스트 (LSTM 없이)
        batch_size = 2
        x_categorical = torch.randint(0, 2, (batch_size, 5))
        x_numerical = torch.randn(batch_size, 2)
        x_seq = None  # LSTM 없이 테스트
        seq_lengths = None
        nan_mask = torch.zeros(batch_size, 7)  # 5 cat + 2 num (seq 제외)
        
        with torch.no_grad():
            try:
                output = model(
                    x_categorical=x_categorical,
                    x_numerical=x_numerical,
                    x_seq=x_seq,
                    seq_lengths=seq_lengths,
                    nan_mask=nan_mask
                )
            except Exception as e:
                print(f"   - Forward pass 오류: {e}")
                # LSTM 디버깅
                print(f"   - LSTM hidden_size: {model.lstm.hidden_size}")
                print(f"   - LSTM num_layers: {model.lstm.num_layers}")
                print(f"   - x_seq shape: {x_seq.shape}")
                print(f"   - seq_lengths: {seq_lengths}")
                
                # LSTM 직접 테스트
                x_seq_test = x_seq.unsqueeze(-1)  # (B, L, 1)
                packed_test = torch.nn.utils.rnn.pack_padded_sequence(
                    x_seq_test, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                try:
                    _, (h_n_test, _) = model.lstm(packed_test)
                    print(f"   - h_n shape: {h_n_test.shape}")
                    print(f"   - h_n[-1] shape: {h_n_test[-1].shape}")
                except Exception as lstm_e:
                    print(f"   - LSTM 직접 테스트 오류: {lstm_e}")
                raise
        
        print(f"   - Forward pass 성공: {output.shape}")
        print(f"   - Output range: [{output.min():.4f}, {output.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 모델 생성 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_loop():
    """훈련 루프 테스트 (1 에포크만)"""
    print("\n🔍 5. 훈련 루프 테스트...")
    
    try:
        from train import train_model
        
        # 더미 데이터로 훈련 테스트
        dummy_data = {
            'gender': ['1.0', '2.0', '1.0', '2.0', '1.0', '2.0', '1.0', '2.0', '1.0', '2.0'],
            'age_group': ['7.0', '8.0', '6.0', '5.0', '4.0', '7.0', '8.0', '6.0', '5.0', '4.0'],
            'inventory_id': ['2', '36', '37', '29', '42', '2', '36', '37', '29', '42'],
            'day_of_week': ['5', '4', '1', '3', '6', '5', '4', '1', '3', '6'],
            'hour': ['08', '07', '12', '20', '09', '08', '07', '12', '20', '09'],
            'l_feat_1': [1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0],
            'feat_a_1': [1.5, 2.3, 0.8, 3.1, 1.9, 1.5, 2.3, 0.8, 3.1, 1.9],
            'seq': ['1.0,2.0,3.0', '2.0,3.0', '1.0', '1.0,2.0,3.0,4.0', '2.0,3.0,4.0', 
                   '1.0,2.0,3.0', '2.0,3.0', '1.0', '1.0,2.0,3.0,4.0', '2.0,3.0,4.0'],
            'clicked': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]
        }
        df = pd.DataFrame(dummy_data)
        
        # 1 에포크만 훈련
        import main
        original_epochs = main.CFG['EPOCHS']
        main.CFG['EPOCHS'] = 1
        
        model = train_model(
            train_df=df,
            feature_cols=list(df.columns),
            seq_col='seq',
            target_col='clicked',
            device="cpu"
        )
        
        # 원래 설정 복원
        main.CFG['EPOCHS'] = original_epochs
        
        print(f"✅ 훈련 루프 성공:")
        print(f"   - 모델 타입: {type(model).__name__}")
        print(f"   - 모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ 훈련 루프 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """메인 디버깅 함수"""
    print("🚀 Transformer 모델 디버깅 시작")
    print("=" * 60)
    print(f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 디버깅용 config 로드
    config = load_debug_config()
    print(f"✅ 디버깅용 config 로드 완료")
    print(f"   - 샘플 크기: {config['DATA']['SAMPLE_SIZE']}")
    print(f"   - 모델 타입: {config['MODEL']['TYPE']}")
    print(f"   - 배치 크기: {config['BATCH_SIZE']}")
    print(f"   - 에포크: {config['EPOCHS']}")
    
    # 테스트 실행
    tests = [
        ("데이터 로딩", test_data_loading),
        ("FeatureProcessor", test_feature_processor),
        ("데이터 로더", test_data_loaders),
        ("모델 생성", test_model_creation),
        ("훈련 루프", test_training_loop)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 테스트 중 예외 발생: {e}")
            results.append((test_name, False))
    
    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 테스트 결과 요약:")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ 통과" if result else "❌ 실패"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 총 {len(results)}개 테스트 중 {passed}개 통과")
    
    if passed == len(results):
        print("🎉 모든 테스트 통과! Transformer 모델이 정상적으로 동작합니다.")
    else:
        print("⚠️  일부 테스트 실패. 위의 오류 메시지를 확인해주세요.")
    
    print(f"📅 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
