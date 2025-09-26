#!/usr/bin/env python3
"""
Optuna를 이용한 하이퍼파라미터 최적화 스크립트
"""

import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import copy
import json
from datetime import datetime

from utils import seed_everything, get_device
from data_loader import load_and_preprocess_data, create_data_loaders
from model import create_tabular_seq_model

# 하이퍼파라미터 서치용 기본 설정
CFG = {
    'BATCH_SIZE': 4096,
    'EPOCHS': 5,  # 빠른 서치를 위해 에포크 수 줄임
    'SEED': 42,
    'VAL_SPLIT': 0.2,
    'MODEL': {
        'TYPE': 'tabular_seq'
    },
    'PATHS': {
        'TRAIN_DATA': './train.parquet',
        'TEST_DATA': './test.parquet'
    },
    'OPTUNA': {
        'N_TRIALS': 300,  # 시행 횟수
        'STUDY_NAME': 'toss_hyperparam_search',
        'STORAGE': 'sqlite:///optuna_study.db',  # SQLite DB로 결과 저장
        'DIRECTION': 'minimize'  # validation loss 최소화
    }
}

device = get_device()

def define_search_space(trial):
    """
    Optuna trial을 사용하여 하이퍼파라미터 검색 공간 정의
    """
    # 학습률 범위
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # 배치 크기 (2의 거듭제곱)
    batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048, 4096])
    
    # 모델 아키텍처 파라미터
    lstm_hidden = trial.suggest_int('lstm_hidden', 32, 256, step=32)
    
    # MLP 히든 레이어 구조
    n_layers = trial.suggest_int('n_layers', 2, 4)
    hidden_units = []
    for i in range(n_layers):
        if i == 0:
            # 첫 번째 레이어는 더 큰 범위
            hidden_size = trial.suggest_int(f'hidden_{i}', 128, 1024, step=128)
        else:
            # 이후 레이어들은 이전 레이어보다 작거나 같게
            prev_size = hidden_units[i-1] if i > 0 else 1024
            hidden_size = trial.suggest_int(f'hidden_{i}', 64, prev_size, step=64)
        hidden_units.append(hidden_size)
    
    # 드롭아웃 비율
    dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
    
    return {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'lstm_hidden': lstm_hidden,
        'hidden_units': hidden_units,
        'dropout': dropout
    }

def objective(trial):
    """
    Optuna objective 함수 - 최소화할 목표 함수
    """
    try:
        # 하이퍼파라미터 샘플링
        hyperparams = define_search_space(trial)
        
        # 시드 고정
        seed_everything(CFG['SEED'])
        
        # 데이터 로드
        train_data, _, feature_cols, seq_col, target_col = load_and_preprocess_data()
        
        # Train/Validation 분할
        tr_df, va_df = train_test_split(
            train_data, 
            test_size=CFG['VAL_SPLIT'], 
            random_state=CFG['SEED'], 
            shuffle=True
        )
        
        # 데이터로더 생성
        train_loader, val_loader, _, train_dataset, val_dataset = create_data_loaders(
            tr_df, va_df, None, feature_cols, seq_col, target_col, hyperparams['batch_size']
        )
        
        # 모델 생성
        d_features = len(feature_cols)
        model = create_tabular_seq_model(
            d_features=d_features,
            lstm_hidden=hyperparams['lstm_hidden'],
            hidden_units=hyperparams['hidden_units'],
            dropout=hyperparams['dropout'],
            device=device
        )
        
        # 손실 함수와 옵티마이저
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        
        # 훈련 루프
        best_val_loss = float('inf')
        patience = 3  # Early stopping
        patience_counter = 0
        
        for epoch in range(1, CFG['EPOCHS'] + 1):
            # 훈련
            model.train()
            train_loss = 0.0
            for xs, seqs, seq_lens, ys in train_loader:
                xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
                
                optimizer.zero_grad()
                logits = model(xs, seqs, seq_lens)
                loss = criterion(logits, ys)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * ys.size(0)
            
            train_loss /= len(train_dataset)
            
            # 검증
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xs, seqs, seq_lens, ys in val_loader:
                    xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
                    logits = model(xs, seqs, seq_lens)
                    loss = criterion(logits, ys)
                    val_loss += loss.item() * ys.size(0)
            
            val_loss /= len(val_dataset)
            
            # Early stopping 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Optuna pruning (중간 결과가 좋지 않으면 조기 종료)
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            if patience_counter >= patience:
                break
        
        return best_val_loss
        
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('inf')  # 실패한 경우 최악의 값 반환

def save_best_params(study, output_file="best_hyperparams.json"):
    """최적 하이퍼파라미터를 JSON 파일로 저장"""
    best_params = study.best_params
    best_value = study.best_value
    
    result = {
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Best hyperparameters saved to {output_file}")
    return result

def print_study_results(study):
    """스터디 결과 출력"""
    print("\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION RESULTS")
    print("="*50)
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial value (validation loss): {study.best_value:.6f}")
    
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print("\nTop 5 trials:")
    trials_df = study.trials_dataframe()
    top_trials = trials_df.nsmallest(5, 'value')[['number', 'value', 'params_learning_rate', 
                                                   'params_batch_size', 'params_lstm_hidden', 
                                                   'params_dropout']]
    print(top_trials.to_string(index=False))

def main():
    """메인 실행 함수"""
    print("Starting Hyperparameter Optimization with Optuna")
    print(f"Device: {device}")
    print(f"Search configuration:")
    print(f"  - Number of trials: {CFG['OPTUNA']['N_TRIALS']}")
    print(f"  - Epochs per trial: {CFG['EPOCHS']}")
    print(f"  - Study name: {CFG['OPTUNA']['STUDY_NAME']}")
    print()
    
    # Optuna 스터디 생성
    study = optuna.create_study(
        direction=CFG['OPTUNA']['DIRECTION'],
        study_name=CFG['OPTUNA']['STUDY_NAME'],
        storage=CFG['OPTUNA']['STORAGE'],
        load_if_exists=True  # 기존 스터디가 있으면 로드
    )
    
    # 하이퍼파라미터 최적화 실행
    study.optimize(
        objective, 
        n_trials=CFG['OPTUNA']['N_TRIALS'],
        show_progress_bar=True
    )
    
    # 결과 출력 및 저장
    print_study_results(study)
    save_best_params(study)
    
    # Optuna 시각화 (선택사항)
    try:
        import optuna.visualization as vis
        
        # 최적화 히스토리 플롯
        fig1 = vis.plot_optimization_history(study)
        fig1.write_html("optimization_history.html")
        
        # 파라미터 중요도 플롯
        fig2 = vis.plot_param_importances(study)
        fig2.write_html("param_importances.html")
        
        print("\nVisualization files saved:")
        print("  - optimization_history.html")
        print("  - param_importances.html")
        
    except ImportError:
        print("\nOptuna visualization not available. Install with: pip install plotly")

if __name__ == "__main__":
    main()
