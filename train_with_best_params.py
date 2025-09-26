#!/usr/bin/env python3
"""
최적 하이퍼파라미터로 모델 훈련하는 스크립트
"""

import json
import os
from main import CFG, device, initialize
from data_loader import load_and_preprocess_data, create_data_loaders
from model import create_tabular_seq_model
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_best_hyperparams(json_path="best_hyperparams.json"):
    """최적 하이퍼파라미터 로드"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Best hyperparameters file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        result = json.load(f)
    
    return result['best_params']

def train_with_best_params(best_params, epochs=None, save_path="best_model.pth"):
    """최적 하이퍼파라미터로 모델 훈련"""
    
    # 에포크 수 설정 (기본값 또는 CFG 사용)
    if epochs is None:
        epochs = CFG['EPOCHS']
    
    print("Training with best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"  epochs: {epochs}")
    print()
    
    # 데이터 로드
    train_data, test_data, feature_cols, seq_col, target_col = load_and_preprocess_data()
    
    # Train/Validation 분할
    tr_df, va_df = train_test_split(
        train_data, 
        test_size=CFG['VAL_SPLIT'], 
        random_state=CFG['SEED'], 
        shuffle=True
    )
    
    # 데이터로더 생성
    train_loader, val_loader, _, train_dataset, val_dataset = create_data_loaders(
        tr_df, va_df, None, feature_cols, seq_col, target_col, best_params['batch_size']
    )
    
    # 모델 생성
    d_features = len(feature_cols)
    model = create_tabular_seq_model(
        d_features=d_features,
        lstm_hidden=best_params['lstm_hidden'],
        hidden_units=best_params['hidden_units'],
        dropout=best_params['dropout'],
        device=device
    )
    
    # 손실 함수와 옵티마이저
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    # 훈련 루프
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        # 훈련
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
        train_losses.append(train_loss)
        
        # 검증
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xs, seqs, seq_lens, ys in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
                xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
                logits = model(xs, seqs, seq_lens)
                loss = criterion(logits, ys)
                val_loss += loss.item() * ys.size(0)
        
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)
        
        # 최고 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[Epoch {epoch}] New best model saved! Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        else:
            print(f"[Epoch {epoch}] Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best model saved to: {save_path}")
    
    return model, train_losses, val_losses

def main():
    """메인 실행 함수"""
    # 초기화
    initialize()
    
    # 최적 하이퍼파라미터 로드
    try:
        best_params = load_best_hyperparams()
        print("Loaded best hyperparameters from hyperparam search")
    except FileNotFoundError:
        print("Best hyperparameters not found. Please run hyperparam_search.py first!")
        return
    
    # 최적 하이퍼파라미터로 훈련
    model, train_losses, val_losses = train_with_best_params(
        best_params, 
        epochs=20,  # 더 많은 에포크로 최종 훈련
        save_path="optimized_model.pth"
    )
    
    # 훈련 기록 저장
    training_log = {
        'best_params': best_params,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': min(val_losses)
    }
    
    with open('training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    
    print("Training log saved to training_log.json")

if __name__ == "__main__":
    main()
