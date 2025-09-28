import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from main import CFG, device, initialize
from data_loader import load_and_preprocess_data, ClickDataset, collate_fn_infer
from model import *


def predict(model, test_loader, device="cuda"):
    """예측 함수 - 딕셔너리 배치에서 ID와 예측값을 함께 반환"""
    model.eval()
    predictions = []
    ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            # 딕셔너리 배치에서 필요한 값들 추출 (collate_fn_infer에서 이미 ID 검증 완료)
            xs = batch.get('xs').to(device)
            seqs = batch.get('seqs').to(device)
            seq_lengths = batch.get('seq_lengths').to(device)
            batch_ids = batch.get('ids', [])  # collate_fn_infer에서 이미 검증됨
            
            ids.extend(batch_ids)
            
            logits = model(xs, seqs, seq_lengths)
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

def load_trained_model(feature_cols, model_path=None, device="cuda"):
    """훈련된 모델 로드 함수"""
    if model_path is None:
        model_path = CFG['PATHS']['MODEL_SAVE']
    
    d_features = len(feature_cols)
    model = create_tabular_seq_model(
        d_features=d_features, 
        lstm_hidden=CFG['MODEL']['LSTM_HIDDEN'], 
        hidden_units=CFG['MODEL']['HIDDEN_UNITS'], 
        dropout=CFG['MODEL']['DROPOUT'], 
        device=device
    )
    model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    return model

def run_inference(model, test_data, feature_cols, seq_col, batch_size, device="cuda"):
    """추론 실행 함수"""
    # Test dataset 생성 (예측 시에는 ID가 반드시 필요)
    test_dataset = ClickDataset(test_data, feature_cols, seq_col, has_target=False, has_id=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_infer)
    
    # 예측 수행
    prediction_result = predict(model, test_loader, device)
    
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
    
    # 모델 로드
    model = load_trained_model(feature_cols, model_path, device)
    
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
