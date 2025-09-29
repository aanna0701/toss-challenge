import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from data_loader import load_and_preprocess_data, ClickDataset, collate_fn_transformer_infer, FeatureProcessor
from model import *


def predict(model, test_loader, device="cuda"):
    """예측 함수 - 딕셔너리 배치에서 ID와 예측값을 함께 반환"""
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch_ids = batch.get('ids', [])  # collate_fn에서 이미 검증됨
            ids.extend(batch_ids)
            
            # TabularTransformer 모델용 배치 처리
            x_categorical = batch.get('x_categorical').to(device)
            x_numerical = batch.get('x_numerical').to(device)
            seqs = batch.get('seqs').to(device)
            seq_lengths = batch.get('seq_lengths').to(device)
            nan_mask = batch.get('nan_mask').to(device)
            logits = model(
                x_categorical=x_categorical,
                x_numerical=x_numerical,
                x_seq=seqs,
                seq_lengths=seq_lengths,
                nan_mask=nan_mask
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

def create_submission(prediction_result, CFG, output_path=None):
    """제출 파일 생성 함수 - 딕셔너리 형태의 예측 결과를 받음"""
    if output_path is None:
        output_path = CFG['PATHS']['SUBMISSION']
    
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

def load_trained_model(feature_cols, CFG, model_path=None, device="cuda"):
    """훈련된 모델 로드 함수"""
    try:
        if model_path is None:
            model_path = CFG['PATHS']['MODEL_SAVE']
        
        print(f"🔧 모델 로딩 시작...")
        print(f"   • 모델 경로: {model_path}")
        print(f"   • 피처 개수: {len(feature_cols)}")
        print(f"   • 디바이스: {device}")
        
        # TabularTransformer 모델용 피처 정보 (훈련 시와 동일한 FeatureProcessor 필요)
        # 실제로는 훈련 시 저장된 feature_processor 정보를 로드해야 함
        # 여기서는 간단히 config에서 가져옴
        categorical_cardinalities = [2, 8, 20, 7, 24]  # gender, age_group, inventory_id, day_of_week, hour
        num_categorical_features = len(CFG['MODEL']['FEATURES']['CATEGORICAL'])
        num_numerical_features = len(feature_cols) - num_categorical_features - 1  # seq 제외
        
        print(f"   • 범주형 피처: {num_categorical_features}개")
        print(f"   • 수치형 피처: {num_numerical_features}개")
        print(f"   • 범주형 카디널리티: {categorical_cardinalities}")
        
    except KeyError as e:
        raise KeyError(f"❌ 설정 파일에서 필요한 키를 찾을 수 없습니다: {e}")
    except Exception as e:
        raise Exception(f"❌ 모델 로딩 준비 중 오류 발생: {e}")
    
    model = create_tabular_transformer_model(
        num_categorical_features=num_categorical_features,
        categorical_cardinalities=categorical_cardinalities,
        num_numerical_features=num_numerical_features,
        lstm_hidden=CFG['MODEL']['TRANSFORMER']['LSTM_HIDDEN'],
        hidden_dim=CFG['MODEL']['TRANSFORMER']['HIDDEN_DIM'],
        n_heads=CFG['MODEL']['TRANSFORMER']['N_HEADS'],
        n_layers=CFG['MODEL']['TRANSFORMER']['N_LAYERS'],
        ffn_size_factor=CFG['MODEL']['TRANSFORMER']['FFN_SIZE_FACTOR'],
        attention_dropout=CFG['MODEL']['TRANSFORMER']['ATTENTION_DROPOUT'],
        ffn_dropout=CFG['MODEL']['TRANSFORMER']['FFN_DROPOUT'],
        residual_dropout=CFG['MODEL']['TRANSFORMER']['RESIDUAL_DROPOUT'],
        device=device
    )
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"✅ 모델 로딩 완료: {model_path}")
    
    # Best checkpoint인지 확인
    if "best.pth" in model_path:
        print(f"🏆 Best checkpoint 사용 중 (최고 성능 모델)")
    else:
        print(f"📝 일반 checkpoint 사용 중")
    
    return model

def run_inference(model, test_data, feature_cols, seq_col, batch_size, device="cuda"):
    """추론 실행 함수"""
    # TabularTransformer 모델용 데이터셋
    feature_processor = FeatureProcessor()
    feature_processor.fit(test_data)  # 테스트 데이터로 fit (실제로는 훈련 데이터로 fit해야 함)
    test_dataset = ClickDataset(test_data, feature_processor, has_target=False, has_id=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_transformer_infer)
    
    # 예측 수행
    prediction_result = predict(model, test_loader, device)
    
    return prediction_result


def predict_test_data(test_data, feature_cols, seq_col, CFG, model_path=None, device="cuda"):
    # 모델 로드
    model = load_trained_model(feature_cols, CFG, model_path, device)
    
    # 예측 수행
    prediction_result = run_inference(
        model=model,
        test_data=test_data,
        feature_cols=feature_cols,
        seq_col=seq_col,
        batch_size=CFG['BATCH_SIZE'],
        device=device
    )
    
    # 제출 파일 생성
    submission = create_submission(prediction_result, CFG=CFG)
    
    predictions = prediction_result['predictions']
    print(f"✅ 예측 완료!")
    print(f"📊 예측 결과 통계:")
    print(f"   - Shape: {predictions.shape}")
    print(f"   - Min: {predictions.min():.4f}")
    print(f"   - Max: {predictions.max():.4f}")
    print(f"   - Mean: {predictions.mean():.4f}")
    print(f"   - Std: {predictions.std():.4f}")
    
    return submission
