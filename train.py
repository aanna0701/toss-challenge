import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# from main import CFG, device, initialize
from data_loader import create_data_loaders
from model import create_tabular_transformer_model
from early_stopping import create_early_stopping_from_config
from metrics import evaluate_model, print_metrics, save_training_logs, get_best_checkpoint_info
from gradient_norm import (
    calculate_gradient_norms, print_gradient_norms, save_gradient_norm_logs,
    analyze_gradient_behavior, print_gradient_analysis, check_gradient_issues, print_gradient_issues
)


def train_model(train_df, feature_cols, seq_col, target_col, CFG, device="cuda", results_dir=None):
    """모델 훈련 함수"""
    
    # 1) split
    tr_df, va_df = train_test_split(train_df, test_size=CFG['VAL_SPLIT'], random_state=42, shuffle=True)

    # 2) Dataset / Loader
    train_loader, val_loader, _, train_dataset, val_dataset, feature_processor = create_data_loaders(
        tr_df, va_df, None, feature_cols, seq_col, target_col, CFG['BATCH_SIZE'], CFG
    )

    # 3) TabularTransformer 모델 생성
    categorical_cardinalities = list(feature_processor.categorical_cardinalities.values())
    num_categorical_features = len(feature_processor.categorical_features)
    num_numerical_features = len(feature_processor.numerical_features)
    
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

    criterion = nn.BCEWithLogitsLoss()

    # Weight decay 적용 (특정 파라미터 제외)
    weight_decay_params = []
    no_decay_params = []

    no_decay_keys = ['class_token', 'column_embeddings', 'nan_token', 'bias', 'norm', 'ln', 'embedding']

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        if any(k in lname for k in no_decay_keys):
            no_decay_params.append(param)
        else:
            weight_decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': weight_decay_params, 'weight_decay': CFG['WEIGHT_DECAY']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=CFG['LEARNING_RATE'])
    
    print(f"🔧 Optimizer 설정:")
    print(f"   • Learning Rate: {CFG['LEARNING_RATE']}")
    print(f"   • Weight Decay: {CFG['WEIGHT_DECAY']}")
    print(f"   • Weight Decay 적용 파라미터: {len(weight_decay_params)}개")
    print(f"   • Weight Decay 제외 파라미터: {len(no_decay_params)}개")

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
    
    # Checkpoint 저장 디렉토리 생성
    if results_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = CFG['PATHS']['RESULTS_DIR'].replace('{datetime}', timestamp)
        os.makedirs(checkpoint_dir, exist_ok=True)
    else:
        checkpoint_dir = results_dir
    print(f"📁 Checkpoint 디렉토리: {checkpoint_dir}")

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
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
            optimizer.zero_grad()
            
            # TabularTransformer 모델용 배치 처리
            x_categorical = batch.get('x_categorical').to(device)
            x_numerical = batch.get('x_numerical').to(device)
            seqs = batch.get('seqs').to(device)
            seq_lens = batch.get('seq_lengths').to(device)
            nan_mask = batch.get('nan_mask').to(device)
            ys = batch.get('ys').to(device)
            logits = model(
                x_categorical=x_categorical,
                x_numerical=x_numerical,
                x_seq=seqs,
                seq_lengths=seq_lens,
                nan_mask=nan_mask
            )
            
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
        val_metrics = evaluate_model(model, val_loader, device, "tabular_transformer")
        
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
        
        # 5 epoch마다 checkpoint 저장
        if epoch % 5 == 0:
            save_checkpoint(model, epoch, optimizer, train_loss, val_metrics, checkpoint_dir, CFG=CFG)
        
        # Early Stopping 체크 (Score 기준)
        monitor_value = val_metrics[CFG['EARLY_STOPPING']['MONITOR'].replace('val_', '')]
        if early_stopping:
            if early_stopping(monitor_value, model):
                print(f"🏁 훈련 조기 종료 (Epoch {epoch}/{CFG['EPOCHS']})")
                # 조기 종료 시에도 checkpoint 저장
                save_checkpoint(model, epoch, optimizer, train_loss, val_metrics, checkpoint_dir, CFG=CFG)
                break

    # 최종 결과 출력
    if early_stopping:
        best_score = early_stopping.get_best_score()
        print(f"🏆 최고 성능: {CFG['EARLY_STOPPING']['MONITOR']} = {best_score:.6f}")
    
    # 최종 checkpoint 저장 (훈련 완료 시)
    print(f"💾 최종 checkpoint 저장 중...")
    save_checkpoint(model, epoch, optimizer, train_loss, val_metrics, checkpoint_dir, CFG=CFG)
    
    # Best checkpoint 저장 (최고 성능 가중치)
    if early_stopping and early_stopping.get_best_weights() is not None:
        print(f"🏆 Best checkpoint 저장 중...")
        best_checkpoint_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(early_stopping.get_best_weights(), best_checkpoint_path)
        print(f"✅ Best checkpoint 저장 완료: {best_checkpoint_path}")
        print(f"   • Best {CFG['EARLY_STOPPING']['MONITOR']}: {early_stopping.get_best_score():.6f}")
    else:
        print(f"⚠️  Best checkpoint 저장 건너뜀 (Early Stopping 비활성화 또는 가중치 없음)")

    # 훈련 로그 저장
    if CFG['METRICS']['SAVE_LOGS']:
        # results_dir가 제공되지 않으면 기본 경로 사용
        if results_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = CFG['PATHS']['RESULTS_DIR'].replace('{datetime}', timestamp)
            os.makedirs(results_dir, exist_ok=True)
        log_filepath = results_dir + "/" + CFG['METRICS']['LOG_FILE']
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
        # RESULTS_DIR에서 {datetime}을 실제 타임스탬프로 치환 (이미 위에서 생성됨)
        gradient_log_filepath = results_dir + "/" + CFG['GRADIENT_NORM']['LOG_FILE']
        save_gradient_norm_logs(gradient_norm_logs, gradient_log_filepath)
        
        # Gradient 행동 분석
        gradient_analysis = analyze_gradient_behavior(gradient_norm_logs)
        print_gradient_analysis(gradient_analysis)

    return model

def save_model(model, path="model.pth"):
    """모델 저장 함수"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def save_checkpoint(model, epoch, optimizer, train_loss, val_metrics, checkpoint_dir, CFG):
    """Checkpoint 저장 함수"""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_metrics': val_metrics,
        'config': CFG
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Checkpoint 로드 함수"""
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"📂 Checkpoint loaded: {checkpoint_path}")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"   Val Score: {checkpoint['val_metrics']['score']:.4f}")
    
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_metrics']

def load_model(model, path="model.pth"):
    """모델 저장 함수"""
    model.load_state_dict(torch.load(path, weights_only=True))
    print(f"Model loaded from {path}")
    return model
