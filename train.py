import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from main import CFG, device, initialize
from data_loader import load_and_preprocess_data, create_data_loaders
from model import *
from early_stopping import create_early_stopping_from_config
from metrics import evaluate_model, print_metrics, save_training_logs, get_best_checkpoint_info
from gradient_norm import (
    calculate_gradient_norms, print_gradient_norms, save_gradient_norm_logs,
    analyze_gradient_behavior, print_gradient_analysis, check_gradient_issues, print_gradient_issues
)

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

    # Early Stopping 설정
    early_stopping = create_early_stopping_from_config(CFG)
    if early_stopping:
        print(f"🛑 Early Stopping 활성화:")
        print(f"   • Monitor: {CFG['EARLY_STOPPING']['MONITOR']}")
        print(f"   • Patience: {CFG['EARLY_STOPPING']['PATIENCE']}")
        print(f"   • Min Delta: {CFG['EARLY_STOPPING']['MIN_DELTA']}")
        print(f"   • Mode: {CFG['EARLY_STOPPING']['MODE']}")
    else:
        print("🚀 Early Stopping 비활성화 - 전체 에포크 훈련")

    # 훈련 로그 초기화
    training_logs = []
    gradient_norm_logs = []

    # Gradient norm 설정 확인
    gradient_norm_enabled = CFG['GRADIENT_NORM']['ENABLED']
    gradient_components = CFG['GRADIENT_NORM']['COMPONENTS']
    
    if gradient_norm_enabled:
        print(f"📊 Gradient Norm 측정 활성화:")
        print(f"   • 측정 구성 요소: {gradient_components}")
        print(f"   • 로그 저장: {CFG['GRADIENT_NORM']['SAVE_LOGS']}")

    # 4) Training Loop
    for epoch in range(1, CFG['EPOCHS']+1):
        # 훈련 단계
        model.train()
        train_loss = 0.0
        epoch_gradient_norms = []
        
        for batch_idx, (xs, seqs, seq_lens, ys) in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
            xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
            optimizer.zero_grad()
            logits = model(xs, seqs, seq_lens)
            loss = criterion(logits, ys)
            loss.backward()
            
            # Gradient norm 측정 (backward 후, step 전)
            if gradient_norm_enabled:
                gradient_norms = calculate_gradient_norms(model, gradient_components)
                epoch_gradient_norms.append(gradient_norms)
                
                # 첫 번째 배치에서만 상세 출력
                if batch_idx == 0:
                    print_gradient_norms(gradient_norms, f"[Epoch {epoch}] ")
                    
                    # Gradient 문제 체크
                    issues = check_gradient_issues(gradient_norms)
                    print_gradient_issues(issues)
            
            optimizer.step()
            train_loss += loss.item() * ys.size(0)
        
        train_loss /= len(train_dataset)
        
        # 에포크별 평균 gradient norm 계산
        if gradient_norm_enabled and epoch_gradient_norms:
            avg_gradient_norms = {}
            for component in gradient_components:
                component_norms = [gn[component] for gn in epoch_gradient_norms if component in gn]
                avg_gradient_norms[f'{component}_grad_norm'] = np.mean(component_norms) if component_norms else 0.0
            
            # Gradient norm 로그 저장
            gradient_log_entry = {
                'epoch': epoch,
                **avg_gradient_norms
            }
            gradient_norm_logs.append(gradient_log_entry)

        # 검증 단계 및 메트릭 계산
        val_metrics = evaluate_model(model, val_loader, device)
        
        # 로그 출력
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}")
        print_metrics(val_metrics, "Val ")
        
        # 훈련 로그 저장
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_ap': val_metrics['ap'],
            'val_wll': val_metrics['wll'],
            'val_score': val_metrics['score']
        }
        training_logs.append(log_entry)
        
        # Early Stopping 체크 (Score 기준)
        monitor_value = val_metrics[CFG['EARLY_STOPPING']['MONITOR'].replace('val_', '')]
        if early_stopping:
            if early_stopping(monitor_value, model):
                print(f"🏁 훈련 조기 종료 (Epoch {epoch}/{CFG['EPOCHS']})")
                break

    # 최종 결과 출력
    if early_stopping:
        best_score = early_stopping.get_best_score()
        print(f"🏆 최고 성능: {CFG['EARLY_STOPPING']['MONITOR']} = {best_score:.6f}")

    # 훈련 로그 저장
    if CFG['METRICS']['SAVE_LOGS']:
        log_filepath = CFG['PATHS']['RESULTS_DIR'] + "/" + CFG['METRICS']['LOG_FILE']
        save_training_logs(training_logs, log_filepath)
        
        # 최고 성능 정보 출력
        best_info = get_best_checkpoint_info(training_logs)
        if best_info:
            print(f"🏆 최고 성능 체크포인트:")
            print(f"   • Epoch: {best_info['epoch']}")
            print(f"   • Val Score: {best_info['val_score']:.6f}")
            print(f"   • Val AP: {best_info['val_ap']:.6f}")
            print(f"   • Val WLL: {best_info['val_wll']:.6f}")

    # Gradient norm 로그 저장 및 분석
    if gradient_norm_enabled and CFG['GRADIENT_NORM']['SAVE_LOGS'] and gradient_norm_logs:
        gradient_log_filepath = CFG['PATHS']['RESULTS_DIR'] + "/" + CFG['GRADIENT_NORM']['LOG_FILE']
        save_gradient_norm_logs(gradient_norm_logs, gradient_log_filepath)
        
        # Gradient 행동 분석
        gradient_analysis = analyze_gradient_behavior(gradient_norm_logs)
        print_gradient_analysis(gradient_analysis)

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
