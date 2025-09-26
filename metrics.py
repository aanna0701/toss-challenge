"""
평가 메트릭 구현
- AP (Average Precision, 50%): 예측 확률에 대해 계산된 평균 정밀도 점수
- WLL (Weighted LogLoss, 50%): 'clicked'의 0과 1의 클래스 기여를 50:50로 맞춘 가중 LogLoss
- Score = 0.5*AP + 0.5*(1/1+WLL)
"""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss
import torch
import torch.nn.functional as F


def calculate_ap(y_true, y_pred):
    """
    Average Precision 계산
    
    Args:
        y_true: 실제 레이블 (0 또는 1)
        y_pred: 예측 확률 (0~1)
    
    Returns:
        float: Average Precision 점수
    """
    if len(np.unique(y_true)) < 2:
        # 클래스가 하나만 있는 경우
        return 0.0
    
    return average_precision_score(y_true, y_pred)


def calculate_weighted_logloss(y_true, y_pred):
    """
    Weighted LogLoss 계산 (50:50 가중치)
    
    Args:
        y_true: 실제 레이블 (0 또는 1)
        y_pred: 예측 확률 (0~1)
    
    Returns:
        float: Weighted LogLoss 점수
    """
    # 클래스별 샘플 수 계산
    class_counts = np.bincount(y_true.astype(int))
    
    if len(class_counts) < 2:
        # 클래스가 하나만 있는 경우
        return float('inf')
    
    # 클래스별 가중치 계산 (50:50로 맞추기 위해)
    total_samples = len(y_true)
    class_0_weight = total_samples / (2 * class_counts[0]) if class_counts[0] > 0 else 0
    class_1_weight = total_samples / (2 * class_counts[1]) if class_counts[1] > 0 else 0
    
    # 가중치 적용
    sample_weights = np.where(y_true == 0, class_0_weight, class_1_weight)
    
    # LogLoss 계산
    try:
        loss = log_loss(y_true, y_pred, sample_weight=sample_weights)
        return loss
    except ValueError:
        return float('inf')


def calculate_score(y_true, y_pred):
    """
    최종 Score 계산: Score = 0.5*AP + 0.5*(1/1+WLL)
    
    Args:
        y_true: 실제 레이블 (0 또는 1)
        y_pred: 예측 확률 (0~1)
    
    Returns:
        dict: AP, WLL, Score를 포함한 딕셔너리
    """
    # AP 계산
    ap = calculate_ap(y_true, y_pred)
    
    # WLL 계산
    wll = calculate_weighted_logloss(y_true, y_pred)
    
    # Score 계산
    if wll == float('inf'):
        score = ap * 0.5  # WLL이 무한대인 경우 AP만 사용
    else:
        score = 0.5 * ap + 0.5 * (1 / (1 + wll))
    
    return {
        'ap': ap,
        'wll': wll,
        'score': score
    }


def evaluate_model(model, data_loader, device="cuda"):
    """
    모델 평가 함수
    
    Args:
        model: 평가할 모델
        data_loader: 평가 데이터 로더
        device: 사용할 디바이스
    
    Returns:
        dict: 평가 결과 (loss, ap, wll, score)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    with torch.no_grad():
        for xs, seqs, seq_lens, ys in data_loader:
            xs, seqs, seq_lens, ys = xs.to(device), seqs.to(device), seq_lens.to(device), ys.to(device)
            
            # 예측
            logits = model(xs, seqs, seq_lens)
            probs = torch.sigmoid(logits)
            
            # Loss 계산
            loss = criterion(logits, ys)
            total_loss += loss.item() * ys.size(0)
            
            # 예측 결과 수집
            all_predictions.extend(probs.cpu().numpy().flatten())
            all_targets.extend(ys.cpu().numpy().flatten())
    
    # 평균 Loss 계산
    avg_loss = total_loss / len(data_loader.dataset)
    
    # 메트릭 계산
    metrics = calculate_score(np.array(all_targets), np.array(all_predictions))
    metrics['loss'] = avg_loss
    
    return metrics


def print_metrics(metrics, prefix=""):
    """
    메트릭 결과 출력
    
    Args:
        metrics: evaluate_model에서 반환된 메트릭 딕셔너리
        prefix: 출력 접두사
    """
    print(f"{prefix}Loss: {metrics['loss']:.6f}")
    print(f"{prefix}AP: {metrics['ap']:.6f}")
    print(f"{prefix}WLL: {metrics['wll']:.6f}")
    print(f"{prefix}Score: {metrics['score']:.6f}")


def save_training_logs(logs, filepath):
    """
    훈련 로그를 CSV 파일로 저장
    
    Args:
        logs: 훈련 로그 리스트 (각 요소는 딕셔너리)
        filepath: 저장할 파일 경로
    """
    df = pd.DataFrame(logs)
    df.to_csv(filepath, index=False)
    print(f"📊 훈련 로그 저장: {filepath}")
    print(f"   • 총 {len(logs)}개 에포크 기록")
    print(f"   • 최고 Score: {df['val_score'].max():.6f} (Epoch {df['val_score'].idxmax() + 1})")


def get_best_checkpoint_info(logs):
    """
    최고 성능 체크포인트 정보 반환
    
    Args:
        logs: 훈련 로그 리스트
    
    Returns:
        dict: 최고 성능 정보
    """
    if not logs:
        return None
    
    df = pd.DataFrame(logs)
    best_idx = df['val_score'].idxmax()
    best_epoch = best_idx + 1
    
    return {
        'epoch': best_epoch,
        'val_score': df.loc[best_idx, 'val_score'],
        'val_ap': df.loc[best_idx, 'val_ap'],
        'val_wll': df.loc[best_idx, 'val_wll'],
        'val_loss': df.loc[best_idx, 'val_loss'],
        'train_loss': df.loc[best_idx, 'train_loss']
    }
