import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from main import CFG, device, initialize
from data_loader import load_and_preprocess_data, ClickDataset, collate_fn_infer
from model import *


def predict(model, test_loader, device="cuda"):
    """예측 함수"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for xs, seqs, lens in tqdm(test_loader, desc="Inference"):
            xs, seqs, lens = xs.to(device), seqs.to(device), lens.to(device)
            logits = model(xs, seqs, lens)
            probs = torch.sigmoid(logits)
            predictions.append(probs.cpu())
    
    return torch.cat(predictions).numpy()

def create_submission(predictions, output_path=None):
    """제출 파일 생성 함수"""
    if output_path is None:
        output_path = CFG['PATHS']['SUBMISSION']
    
    submit = pd.read_csv(CFG['PATHS']['SAMPLE_SUBMISSION'])
    submit['clicked'] = predictions
    submit.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")
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
    # Test dataset 생성
    test_dataset = ClickDataset(test_data, feature_cols, seq_col, has_target=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_infer)
    
    # 예측 수행
    predictions = predict(model, test_loader, device)
    
    return predictions

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
    predictions = run_inference(
        model=model,
        test_data=test_data,
        feature_cols=feature_cols,
        seq_col=seq_col,
        batch_size=CFG['BATCH_SIZE'],
        device=device
    )
    
    # 제출 파일 생성
    submission = create_submission(predictions)
    print("Prediction completed!")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Prediction stats: min={predictions.min():.4f}, max={predictions.max():.4f}, mean={predictions.mean():.4f}")
