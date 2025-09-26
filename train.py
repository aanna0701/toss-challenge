import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from main import CFG, device, initialize
from data_loader import load_and_preprocess_data, create_data_loaders
from model import *

def train_model(train_df, feature_cols, seq_col, target_col, device="cuda"):
    """모델 훈련 함수"""
    
    # 1) split
    tr_df, va_df = train_test_split(train_df, test_size=CFG['VAL_SPLIT'], random_state=42, shuffle=True)

    # 2) Dataset / Loader
    train_loader, val_loader, _, train_dataset, val_dataset = create_data_loaders(
        tr_df, va_df, None, feature_cols, seq_col, target_col, CFG['BATCH_SIZE']
    )

    # 3) 모델 선택 및 생성
    d_features = len(feature_cols)
    
    if CFG['MODEL']['TYPE'] == 'tabular_seq':
        model = create_tabular_seq_model(
            d_features=d_features, 
            lstm_hidden=CFG['MODEL']['LSTM_HIDDEN'], 
            hidden_units=CFG['MODEL']['HIDDEN_UNITS'], 
            dropout=CFG['MODEL']['DROPOUT'], 
            device=device
        )
    else:
        raise ValueError(f"Unknown model type: {CFG['MODEL']['TYPE']}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['LEARNING_RATE'])

    # 4) Training Loop
    for epoch in range(1, CFG['EPOCHS']+1):
        model.train()
        train_loss = 0.0
        for xs, seqs, seq_lens, ys in tqdm(train_loader, desc=f"Train Epoch {epoch}"):
            xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
            optimizer.zero_grad()
            logits = model(xs, seqs, seq_lens)
            loss = criterion(logits, ys)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * ys.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xs, seqs, seq_lens, ys in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
                logits = model(xs, seqs, seq_lens)
                loss = criterion(logits, ys)
                val_loss += loss.item() * len(ys)
        val_loss /= len(val_dataset)

        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model

def save_model(model, path="model.pth"):
    """모델 저장 함수"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path="model.pth"):
    """모델 로드 함수"""
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

if __name__ == "__main__":
    # 초기화 (YAML 설정 파일 로드 포함)
    initialize()
    
    # 데이터 로드 및 전처리
    train_data, test_data, feature_cols, seq_col, target_col = load_and_preprocess_data()
    
    # 모델 훈련
    model = train_model(
        train_df=train_data,
        feature_cols=feature_cols,
        seq_col=seq_col,
        target_col=target_col,
        device=device
    )
    
    # 모델 저장
    save_model(model, CFG['PATHS']['MODEL_SAVE'])
    print("Training completed!")
