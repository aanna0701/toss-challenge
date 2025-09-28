import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from main import CFG, device, initialize
from data_loader import load_and_preprocess_data, ClickDataset, TabularSeqDataset, collate_fn_seq_infer, collate_fn_transformer_infer, FeatureProcessor
from model import *


def predict(model, test_loader, device="cuda", model_type="tabular_seq"):
    """예측 함수 - 딕셔너리 배치에서 ID와 예측값을 함께 반환"""
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch_ids = batch.get('ids', [])  # collate_fn에서 이미 검증됨
            ids.extend(batch_ids)
            
            if model_type == 'tabular_seq':
                # TabularSeq 모델용 배치 처리
                xs = batch.get('xs').to(device)
                seqs = batch.get('seqs').to(device)
                seq_lengths = batch.get('seq_lengths').to(device)
                logits = model(xs, seqs, seq_lengths)
            elif model_type == 'tabular_transformer':
                # Transformer 모델용 배치 처리
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

def create_submission(prediction_result, output_path=None):
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

def load_trained_model(feature_cols, model_path=None, device="cuda", model_type=None):
    """훈련된 모델 로드 함수"""
    if model_path is None:
        model_path = CFG['PATHS']['MODEL_SAVE']
    
    if model_type is None:
        model_type = CFG['MODEL']['TYPE']
    
    if model_type == 'tabular_seq':
        d_features = len(feature_cols)
        model = create_tabular_seq_model(
            d_features=d_features, 
            lstm_hidden=CFG['MODEL']['LSTM_HIDDEN'], 
            hidden_units=CFG['MODEL']['HIDDEN_UNITS'], 
            dropout=CFG['MODEL']['DROPOUT'], 
            device=device
        )
    elif model_type == 'tabular_transformer':
        # Transformer 모델용 피처 정보 (훈련 시와 동일한 FeatureProcessor 필요)
        # 실제로는 훈련 시 저장된 feature_processor 정보를 로드해야 함
        # 여기서는 간단히 config에서 가져옴
        categorical_cardinalities = [2, 8, 20, 7, 24]  # gender, age_group, inventory_id, day_of_week, hour
        num_categorical_features = len(CFG['MODEL']['FEATURES']['CATEGORICAL'])
        num_numerical_features = len(feature_cols) - num_categorical_features - 1  # seq 제외
        
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
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path, weights_only=True))
    print(f"Model loaded from {model_path}")
    return model

def run_inference(model, test_data, feature_cols, seq_col, batch_size, device="cuda", model_type="tabular_seq"):
    """추론 실행 함수"""
    if model_type == 'tabular_seq':
        # TabularSeq 모델용 데이터셋
        test_dataset = TabularSeqDataset(test_data, feature_cols, seq_col, has_target=False, has_id=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_seq_infer)
    elif model_type == 'tabular_transformer':
        # Transformer 모델용 데이터셋
        feature_processor = FeatureProcessor()
        feature_processor.fit(test_data)  # 테스트 데이터로 fit (실제로는 훈련 데이터로 fit해야 함)
        test_dataset = ClickDataset(test_data, feature_processor, has_target=False, has_id=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_transformer_infer)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 예측 수행
    prediction_result = predict(model, test_loader, device, model_type)
    
    return prediction_result

if __name__ == "__main__":
    # 초기화
    initialize()
    
    # 데이터 로드
    _, test_data, feature_cols, seq_col, target_col = load_and_preprocess_data()
    
    # 훈련된 모델 로드 (train.py에서 저장된 모델)
    try:
        model = load_trained_model(feature_cols, device=device)
    except FileNotFoundError:
        print("Trained model not found. Please run train.py first!")
        exit(1)
    
    # 추론 실행
    prediction_result = run_inference(
        model=model,
        test_data=test_data,
        feature_cols=feature_cols,
        seq_col=seq_col,
        batch_size=CFG['BATCH_SIZE'],
        device=device
    )
    
    # 제출 파일 생성
    submission = create_submission(prediction_result)
    predictions = prediction_result['predictions']
    print("Prediction completed!")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction stats: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")


def predict_test_data(test_data, feature_cols, seq_col, model_path=None, device="cuda"):
    model_type = CFG['MODEL']['TYPE']
    
    # 모델 로드
    model = load_trained_model(feature_cols, model_path, device, model_type)
    
    # 예측 수행
    prediction_result = run_inference(
        model=model,
        test_data=test_data,
        feature_cols=feature_cols,
        seq_col=seq_col,
        batch_size=CFG['BATCH_SIZE'],
        device=device,
        model_type=model_type
    )
    
    # 제출 파일 생성
    submission = create_submission(prediction_result)
    
    predictions = prediction_result['predictions']
    print(f"✅ 예측 완료!")
    print(f"📊 예측 결과 통계:")
    print(f"   - Shape: {predictions.shape}")
    print(f"   - Min: {predictions.min():.4f}")
    print(f"   - Max: {predictions.max():.4f}")
    print(f"   - Mean: {predictions.mean():.4f}")
    print(f"   - Std: {predictions.std():.4f}")
    
    return submission
