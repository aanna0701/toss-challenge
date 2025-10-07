import pandas as pd
import torch
from tqdm import tqdm

from data_loader import (
    FeatureProcessor,
    ClickDataset,
    collate_fn_transformer_infer,
)
from torch.utils.data import DataLoader
from model import create_tabular_transformer_model, create_widedeep_ctr_model


def predict(model, test_loader, device, fabric=None):
    """예측 함수 - 딕셔너리 배치에서 ID와 예측값을 함께 반환 (Lightning Fabric 지원)"""
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch_ids = batch.get('ids', [])  # collate_fn에서 이미 검증됨
            ids.extend(batch_ids)
            
            # 배치 데이터 처리 (Fabric 지원)
            # Fabric을 사용하는 경우 배치가 자동으로 올바른 디바이스로 이동됨
            x_categorical = batch.get('x_categorical')
            x_numerical = batch.get('x_numerical')
            seqs = batch.get('seqs')
            seq_lengths = batch.get('seq_lengths')
            
            # Fabric을 사용하지 않는 경우에만 수동으로 디바이스 이동
            if not fabric:
                x_categorical = x_categorical.to(device)
                x_numerical = x_numerical.to(device)
                seqs = seqs.to(device)
                seq_lengths = seq_lengths.to(device)
            
            # 모델 타입에 따라 forward 호출 방식 결정
            if hasattr(model, 'forward') and 'num_x' in model.forward.__code__.co_varnames:
                # WideDeepCTR 모델
                logits = model(
                    num_x=x_numerical,
                    cat_x=x_categorical,
                    seqs=seqs,
                    seq_lengths=seq_lengths
                )
            else:
                # TabularTransformer 모델
                logits = model(
                    x_categorical=x_categorical,
                    x_numerical=x_numerical,
                    x_seq=seqs,
                    seq_lengths=seq_lengths
                )
            
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu())
    
    predictions_array = torch.cat(predictions).numpy()
    
    # ID와 예측값을 딕셔너리 형태로 반환
    result = {
        'ids': ids,
        'predictions': predictions_array
    }
    
    return result

def create_submission(prediction_result, CFG, output_path):
    """제출 파일 생성 함수 - 딕셔너리 형태의 예측 결과를 받음"""
    
    # 예측 결과에서 ID와 predictions 추출
    ids = prediction_result['ids']
    predictions = prediction_result['predictions']
    
    # 검증
    if not ids:
        raise ValueError("❌ ID 정보가 없습니다!")
    
    if len(ids) != len(predictions):
        raise ValueError(f"❌ ID 개수({len(ids)})와 예측 결과 개수({len(predictions)})가 일치하지 않습니다!")
    
    submit = pd.DataFrame({
        'ID': ids,
        'clicked': predictions
    })
    
    print("✅ ID와 예측값 매칭 완료")
    submit.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
    print(f"Submission shape: {submit.shape}")
    return submit

def load_trained_model(feature_processor, CFG, model_path, device):
    """훈련된 모델 로드 함수"""
    print(f"🔧 모델 로딩 시작...")
    print(f"   • 모델 경로: {model_path}")
    print(f"   • 디바이스: {device}")
    
    # FeatureProcessor에서 피처 정보 추출
    categorical_cardinalities = list(feature_processor.categorical_cardinalities.values())
    num_categorical_features = len(feature_processor.categorical_features)
    num_numerical_features = len(feature_processor.numerical_features)
    
    print(f"✅ 피처 정보:")
    print(f"   • 범주형 피처: {num_categorical_features}개")
    print(f"   • 수치형 피처: {num_numerical_features}개")
    print(f"   • 범주형 카디널리티: {categorical_cardinalities}")
    
    # 모델 타입 결정
    model_type = CFG.get('MODEL_TYPE', 'transformer')  # 기본값: transformer
    
    if model_type == 'widedeep':
        # WideDeepCTR 모델 생성
        model = create_widedeep_ctr_model(
            num_features=num_numerical_features,
            cat_cardinalities=categorical_cardinalities,
            emb_dim=CFG['MODEL']['WIDEDEEP']['EMB_DIM'],
            lstm_hidden=CFG['MODEL']['WIDEDEEP']['LSTM_HIDDEN'],
            hidden_units=CFG['MODEL']['WIDEDEEP']['HIDDEN_UNITS'],
            dropout=CFG['MODEL']['WIDEDEEP']['DROPOUT'],
            device=device
        )
        model_type_name = "WideDeepCTR"
    else:
        # TabularTransformer 모델 생성 (기본값)
        model = create_tabular_transformer_model(
            num_categorical_features=num_categorical_features,
            categorical_cardinalities=categorical_cardinalities,
            num_numerical_features=num_numerical_features,
            lstm_hidden=CFG['MODEL']['TRANSFORMER']['HIDDEN_DIM'],
            hidden_dim=CFG['MODEL']['TRANSFORMER']['HIDDEN_DIM'],
            n_heads=CFG['MODEL']['TRANSFORMER']['N_HEADS'],
            n_layers=CFG['MODEL']['TRANSFORMER']['N_LAYERS'],
            ffn_size_factor=CFG['MODEL']['TRANSFORMER']['FFN_SIZE_FACTOR'],
            attention_dropout=CFG['MODEL']['TRANSFORMER']['ATTENTION_DROPOUT'],
            ffn_dropout=CFG['MODEL']['TRANSFORMER']['FFN_DROPOUT'],
            residual_dropout=CFG['MODEL']['TRANSFORMER']['RESIDUAL_DROPOUT'],
            device=device
        )
        model_type_name = "TabularTransformer"
    
    print(f"✅ 모델 타입: {model_type_name}")
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"✅ 모델 로딩 완료: {model_path}")
    
    # Best checkpoint인지 확인
    if "best.pth" in model_path:
        print(f"🏆 Best checkpoint 사용 중 (최고 성능 모델)")
    else:
        print(f"📝 일반 checkpoint 사용 중")
    
    return model

def predict_test_data(test_data, feature_processor, CFG, model_path, device, fabric=None):
    # FeatureProcessor 직접 생성 및 테스트 로더 생성
    print("🔧 FeatureProcessor 및 테스트 로더 생성...")
    # 테스트 데이터셋 생성
    test_dataset = ClickDataset(test_data, feature_processor, has_target=False, has_id=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn_transformer_infer)
    
    # Fabric을 사용하는 경우 DataLoader를 래핑
    if fabric:
        print(f"🔧 테스트 DataLoader를 Fabric으로 래핑 중...")
        test_loader = fabric.setup_dataloaders(test_loader)
        print(f"✅ 테스트 DataLoader Fabric 래핑 완료")
    
    # 모델 로드
    model = load_trained_model(feature_processor, CFG, model_path, device)
    
    # 예측 수행 (test_loader 직접 사용)
    prediction_result = predict(model, test_loader, device, fabric)
    
    # 제출 파일 생성 (임시 파일로 저장)
    import tempfile
    import os
    temp_submission_path = os.path.join(tempfile.gettempdir(), "temp_submission.csv")
    submission = create_submission(prediction_result, CFG=CFG, output_path=temp_submission_path)
    
    predictions = prediction_result['predictions']
    print(f"✅ 예측 완료!")
    print(f"📊 예측 결과 통계:")
    print(f"   - Shape: {predictions.shape}")
    print(f"   - Min: {predictions.min():.4f}")
    print(f"   - Max: {predictions.max():.4f}")
    print(f"   - Mean: {predictions.mean():.4f}")
    print(f"   - Std: {predictions.std():.4f}")
    
    return submission
