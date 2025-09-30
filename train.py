import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

# from main import CFG, device, initialize
from data_loader import create_data_loaders
from model import create_tabular_transformer_model
from early_stopping import create_early_stopping_from_config
from metrics import evaluate_model, print_metrics, save_training_logs, get_best_checkpoint_info
from gradient_norm import (
    calculate_gradient_norms, print_gradient_norms, save_gradient_norm_logs,
    analyze_gradient_behavior, print_gradient_analysis, check_gradient_issues, print_gradient_issues
)

def print_model_summary(model, log_file_path=None):
    """모델의 상세 구조를 출력하고 로그 파일에 저장"""
    print("🔍 모델 구조 분석 중...")
    
    # 기본 summary 생성
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 레이어별 상세 정보 수집
    layer_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf module
            param_count = sum(p.numel() for p in module.parameters())
            trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # 모듈 타입과 이름
            module_type = type(module).__name__
            if name:
                layer_name = f"{name} ({module_type})"
            else:
                layer_name = f"({module_type})"
            
            # 출력 형태 추정
            if hasattr(module, 'out_features'):
                output_shape = f"(*, {module.out_features})"
            elif hasattr(module, 'num_embeddings'):
                output_shape = f"(*, {module.embedding_dim})"
            elif hasattr(module, 'hidden_size'):
                output_shape = f"(*, {module.hidden_size})"
            else:
                output_shape = "(*, *)"
            
            layer_info.append({
                'name': layer_name,
                'output_shape': output_shape,
                'param_count': param_count,
                'trainable': trainable_count > 0
            })
    
    # Summary 텍스트 생성
    summary_lines = []
    summary_lines.append(f"모델 구조 Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary_lines.append("=" * 80)
    summary_lines.append(f"{'Layer (type)':<50} {'Output Shape':<20} {'Param #':<15} {'Trainable':<10}")
    summary_lines.append("=" * 80)
    
    for layer in layer_info:
        line = f"{layer['name']:<50} {layer['output_shape']:<20} {layer['param_count']:<15,} {'Yes' if layer['trainable'] else 'No':<10}"
        summary_lines.append(line)
    
    summary_lines.append("=" * 80)
    summary_lines.append(f"Total params: {total_params:,}")
    summary_lines.append(f"Trainable params: {trainable_params:,}")
    summary_lines.append(f"Non-trainable params: {total_params - trainable_params:,}")
    summary_lines.append("=" * 80)
    
    summary_text = "\n".join(summary_lines)
    
    
    # 로그 파일에 저장
    if log_file_path:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        print(f"📋 모델 구조가 로그 파일에 저장되었습니다: {log_file_path}")


def train_model(train_df, feature_cols, seq_col, target_col, CFG, device, results_dir):
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
    
    # 모델 생성 직후 summary 출력
    print(f"\n📊 모델 Summary:")
    print(f"   • 모델 타입: TabularTransformer")
    print(f"   • 총 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   • 학습 가능한 파라미터: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"   • Hidden Dimension: {CFG['MODEL']['TRANSFORMER']['HIDDEN_DIM']}")
    print(f"   • Attention Heads: {CFG['MODEL']['TRANSFORMER']['N_HEADS']}")
    print(f"   • Transformer Layers: {CFG['MODEL']['TRANSFORMER']['N_LAYERS']}")
    print(f"   • LSTM Hidden Size: {CFG['MODEL']['TRANSFORMER']['HIDDEN_DIM']}")
    print(f"   • 범주형 피처 수: {num_categorical_features}")
    print(f"   • 수치형 피처 수: {num_numerical_features}")
    print(f"   • Device: {device}")
    
    # 상세 모델 구조 출력
    print(f"\n🔍 상세 모델 구조:")
    if results_dir:
        from datetime import datetime
        model_summary_log_path = os.path.join(results_dir, f"model_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        print_model_summary(model, model_summary_log_path)
    else:
        print_model_summary(model)

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
    
    # Warmup 스케줄러 설정
    warmup_enabled = CFG.get('WARMUP', {}).get('ENABLED', False)
    warmup_epochs = CFG.get('WARMUP', {}).get('WARMUP_EPOCHS', 2)
    
    if warmup_enabled:
        print(f"🔥 Warmup 스케줄러 활성화:")
        print(f"   • Warmup Epochs: {warmup_epochs}")
        print(f"   • Base Learning Rate: {CFG['LEARNING_RATE']}")
    else:
        print("🚀 Warmup 스케줄러 비활성화")
    
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
    
    # 상세 로그 파일 설정 (CSV 형태)
    detailed_log_path = None
    if results_dir:
        detailed_log_path = os.path.join(results_dir, "training_detailed.csv")
        
        # CSV 파일 헤더 작성
        with open(detailed_log_path, 'w', encoding='utf-8') as f:
            f.write("step,epoch,batch_idx,train_loss,learning_rate,warmup_factor,val_loss,val_ap,val_wll,val_score\n")
        
        print(f"📊 상세 로그 파일 생성 (CSV): {detailed_log_path}")
    
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

    # 4) Training Loop with Step-based Logging
    global_step = 0
    steps_per_epoch = len(train_loader)
    total_steps = CFG['EPOCHS'] * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch if warmup_enabled else 0
    
    print(f"📊 훈련 설정:")
    print(f"   • 총 스텝: {total_steps}")
    print(f"   • 에포크당 스텝: {steps_per_epoch}")
    print(f"   • Warmup 스텝: {warmup_steps}")
    
    for epoch in range(1, CFG['EPOCHS']+1):
        # 훈련 단계
        model.train()
        epoch_train_loss = 0.0
        epoch_gradient_norms = []
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch}")):
            global_step += 1
            
            # Warmup 스케줄링
            if warmup_enabled and global_step <= warmup_steps:
                # Linear warmup: 0에서 base_lr까지 선형 증가
                warmup_factor = global_step / warmup_steps
                current_lr = CFG['LEARNING_RATE'] * warmup_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            elif warmup_enabled and global_step == warmup_steps + 1:
                # Warmup 완료 후 base learning rate로 설정
                for param_group in optimizer.param_groups:
                    param_group['lr'] = CFG['LEARNING_RATE']
            
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
            
            optimizer.step()
            epoch_train_loss += loss.item() * ys.size(0)
            
            # 스텝별 로깅 (10 에폭 단위로만 저장)
            current_lr = optimizer.param_groups[0]['lr']
            warmup_factor = global_step / warmup_steps if warmup_enabled and global_step <= warmup_steps else 1.0
            
            # 10 에폭 단위일 때만 로그 저장
            if epoch % 10 == 0:
                log_entry = {
                    'step': global_step,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'train_loss': loss.item(),
                    'learning_rate': current_lr,
                    'warmup_factor': warmup_factor
                }
                training_logs.append(log_entry)
            
            # 스텝별 출력 (매 100 스텝마다)
            if global_step % 100 == 0:
                print(f"[Step {global_step}] Epoch {epoch}/{CFG['EPOCHS']}, Batch {batch_idx+1}/{steps_per_epoch}")
                print(f"   • Train Loss: {loss.item():.4f}")
                print(f"   • Learning Rate: {current_lr:.6f}")
                if warmup_enabled and global_step <= warmup_steps:
                    print(f"   • Warmup Progress: {global_step}/{warmup_steps} ({global_step/warmup_steps*100:.1f}%)")
        
        epoch_train_loss /= len(train_dataset)
        
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
        
        # 에포크별 로그 출력
        print(f"\n[Epoch {epoch}/{CFG['EPOCHS']}] Summary:")
        print(f"   • Avg Train Loss: {epoch_train_loss:.4f}")
        print(f"   • Current LR: {optimizer.param_groups[0]['lr']:.6f}")
        print_metrics(val_metrics, "Val ")
        
        # 에포크별 검증 로그 저장 (10 에폭 단위로만)
        if epoch % 10 == 0:
            epoch_log_entry = {
                'step': global_step,
                'epoch': epoch,
                'epoch_train_loss': epoch_train_loss,
                'val_loss': val_metrics['loss'],
                'val_ap': val_metrics['ap'],
                'val_wll': val_metrics['wll'],
                'val_score': val_metrics['score'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            training_logs.append(epoch_log_entry)
        
        # 상세 로그 파일에 에포크별 검증 결과 기록 (10 에폭마다)
        if detailed_log_path and epoch % 10 == 0:
            # 해당 에폭의 모든 스텝 로그를 파일에 저장
            epoch_logs = [log for log in training_logs if log['epoch'] == epoch and 'val_score' not in log]
            with open(detailed_log_path, 'a', encoding='utf-8') as f:
                for log in epoch_logs:
                    f.write(f"{log['step']},{log['epoch']},{log['batch_idx']},{log['train_loss']:.6f},{log['learning_rate']:.8f},{log['warmup_factor']:.3f},,,,\n")
                # 검증 결과 추가
                f.write(f"{global_step},{epoch},-1,{epoch_train_loss:.6f},{optimizer.param_groups[0]['lr']:.8f},1.0,{val_metrics['loss']:.6f},{val_metrics['ap']:.6f},{val_metrics['wll']:.6f},{val_metrics['score']:.6f}\n")  # 구분선
    
        # 5 epoch마다 checkpoint 저장
        if epoch % 5 == 0:
            save_checkpoint(model, epoch, optimizer, epoch_train_loss, val_metrics, checkpoint_dir, CFG=CFG)
        
        # Early Stopping 체크 (Score 기준)
        monitor_value = val_metrics[CFG['EARLY_STOPPING']['MONITOR'].replace('val_', '')]
        if early_stopping:
            if early_stopping(monitor_value, model):
                print(f"🏁 훈련 조기 종료 (Epoch {epoch}/{CFG['EPOCHS']})")
                # 조기 종료 시에도 checkpoint 저장
                save_checkpoint(model, epoch, optimizer, epoch_train_loss, val_metrics, checkpoint_dir, CFG=CFG)
                break

    # 최종 결과 출력
    if early_stopping:
        best_score = early_stopping.get_best_score()
        print(f"🏆 최고 성능: {CFG['EARLY_STOPPING']['MONITOR']} = {best_score:.6f}")
    
    # 최종 checkpoint 저장 (훈련 완료 시)
    print(f"💾 최종 checkpoint 저장 중...")
    save_checkpoint(model, epoch, optimizer, epoch_train_loss, val_metrics, checkpoint_dir, CFG=CFG)
    
    # Best checkpoint 저장 (최고 성능 가중치)
    if early_stopping and early_stopping.get_best_weights() is not None:
        print(f"🏆 Best checkpoint 저장 중...")
        best_checkpoint_path = os.path.join(checkpoint_dir, "best.pth")
        torch.save(early_stopping.get_best_weights(), best_checkpoint_path)
        print(f"✅ Best checkpoint 저장 완료: {best_checkpoint_path}")
        print(f"   • Best {CFG['EARLY_STOPPING']['MONITOR']}: {early_stopping.get_best_score():.6f}")
    else:
        print(f"⚠️  Best checkpoint 저장 건너뜀 (Early Stopping 비활성화 또는 가중치 없음)")

    # 훈련 로그 저장 (스텝 기반) - CSV는 이미 실시간으로 저장됨
    if CFG['METRICS']['SAVE_LOGS']:
        # results_dir가 제공되지 않으면 기본 경로 사용
        if results_dir is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = CFG['PATHS']['RESULTS_DIR'].replace('{datetime}', timestamp)
            os.makedirs(results_dir, exist_ok=True)
        
        # 스텝별 로그 저장 (JSON 형태로 백업)
        log_filepath = os.path.join(results_dir, "training_logs.json")
        save_training_logs(training_logs, log_filepath)
        
        # 상세 로그 파일에 훈련 완료 요약 추가
        if detailed_log_path:
            with open(detailed_log_path, 'a', encoding='utf-8') as f:
                # 훈련 완료 요약
                f.write(f"# Training completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Total steps: {global_step}\n")
                f.write(f"# Total epochs: {CFG['EPOCHS']}\n")
                if warmup_enabled:
                    f.write(f"# Warmup steps: {warmup_steps}\n")
                    f.write(f"# Final LR: {CFG['LEARNING_RATE']:.8f}\n")
                
                # 최고 성능 정보
                epoch_logs = [log for log in training_logs if 'val_score' in log]
                if epoch_logs:
                    best_epoch_log = max(epoch_logs, key=lambda x: x['val_score'])
                    f.write(f"# Best performance: Step {best_epoch_log['step']}, Epoch {best_epoch_log['epoch']}, Val Score {best_epoch_log['val_score']:.6f}\n")
        
        print(f"📊 CSV 로그 저장 완료: {detailed_log_path}")
        
        # Warmup 정보 출력
        if warmup_enabled:
            print(f"🔥 Warmup 완료:")
            print(f"   • Warmup Steps: {warmup_steps}")
            print(f"   • Total Steps: {global_step}")
            print(f"   • Final LR: {CFG['LEARNING_RATE']:.6f}")
        
        # 최고 성능 정보 출력 (에포크별 검증 로그에서)
        epoch_logs = [log for log in training_logs if 'val_score' in log]
        if epoch_logs:
            best_epoch_log = max(epoch_logs, key=lambda x: x['val_score'])
            print(f"🏆 최고 성능 체크포인트:")
            print(f"   • Step: {best_epoch_log['step']}")
            print(f"   • Epoch: {best_epoch_log['epoch']}")
            print(f"   • Val Score: {best_epoch_log['val_score']:.6f}")
            print(f"   • Val AP: {best_epoch_log['val_ap']:.6f}")
            print(f"   • Val WLL: {best_epoch_log['val_wll']:.6f}")
            print(f"   • Learning Rate: {best_epoch_log['learning_rate']:.6f}")

    # Gradient norm 로그 저장 및 분석
    if gradient_norm_enabled and CFG['GRADIENT_NORM']['SAVE_LOGS'] and gradient_norm_logs:
        # RESULTS_DIR에서 {datetime}을 실제 타임스탬프로 치환 (이미 위에서 생성됨)
        gradient_log_filepath = results_dir + "/" + CFG['GRADIENT_NORM']['LOG_FILE']
        save_gradient_norm_logs(gradient_norm_logs, gradient_log_filepath)
        
        # Gradient 행동 분석
        gradient_analysis = analyze_gradient_behavior(gradient_norm_logs)
        print_gradient_analysis(gradient_analysis)

    return model, feature_processor

def save_model(model, path, feature_processor):
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
