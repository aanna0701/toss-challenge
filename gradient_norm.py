"""
Gradient Norm 측정 및 로깅 기능
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def calculate_gradient_norms(model, components: List[str]) -> Dict[str, float]:
    """
    모델 구성 요소별 gradient norm 계산
    
    Args:
        model: PyTorch 모델
        components: 측정할 구성 요소 리스트 ['lstm', 'mlp', 'total']
    
    Returns:
        Dict[str, float]: 구성 요소별 gradient norm
    """
    gradient_norms = {}
    
    # 전체 모델 gradient norm
    if 'total' in components:
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        gradient_norms['total'] = total_norm ** 0.5
    
    # LSTM 부분 gradient norm
    if 'lstm' in components:
        lstm_norm = 0.0
        lstm_params = 0
        for name, param in model.named_parameters():
            if 'lstm' in name.lower() and param.grad is not None:
                param_norm = param.grad.data.norm(2)
                lstm_norm += param_norm.item() ** 2
                lstm_params += 1
        gradient_norms['lstm'] = lstm_norm ** 0.5 if lstm_params > 0 else 0.0
    
    # 모델 부분 gradient norm (MLP/Transformer 등)
    if 'model' in components:
        model_norm = 0.0
        model_params = 0
        for name, param in model.named_parameters():
            if ('mlp' in name.lower() or 
                'transformer' in name.lower() or 
                ('linear' in name.lower() and 'lstm' not in name.lower()) or
                'attention' in name.lower() or
                'feedforward' in name.lower()) and param.grad is not None:
                param_norm = param.grad.data.norm(2)
                model_norm += param_norm.item() ** 2
                model_params += 1
        gradient_norms['model'] = model_norm ** 0.5 if model_params > 0 else 0.0
    
    return gradient_norms


def print_gradient_norms(gradient_norms: Dict[str, float], prefix: str = ""):
    """
    Gradient norm 결과 출력
    
    Args:
        gradient_norms: gradient norm 딕셔너리
        prefix: 출력 접두사
    """
    print(f"{prefix}Gradient Norms:")
    for component, norm in gradient_norms.items():
        print(f"   • {component.upper()}: {norm:.6f}")


def save_gradient_norm_logs(logs: List[Dict], filepath: str):
    """
    Gradient norm 로그를 CSV 파일로 저장
    
    Args:
        logs: gradient norm 로그 리스트
        filepath: 저장할 파일 경로
    """
    if not logs:
        print("⚠️ 저장할 gradient norm 로그가 없습니다.")
        return
    
    df = pd.DataFrame(logs)
    df.to_csv(filepath, index=False)
    print(f"📊 Gradient norm 로그 저장: {filepath}")
    print(f"   • 총 {len(logs)}개 기록")
    
    # 각 구성 요소별 통계 출력
    for component in ['total', 'lstm', 'mlp']:
        if component in df.columns:
            col_name = f'{component}_grad_norm'
            if col_name in df.columns:
                print(f"   • {component.upper()} 평균: {df[col_name].mean():.6f}")
                print(f"   • {component.upper()} 최대: {df[col_name].max():.6f}")
                print(f"   • {component.upper()} 최소: {df[col_name].min():.6f}")


def analyze_gradient_behavior(logs: List[Dict]) -> Dict[str, any]:
    """
    Gradient norm 행동 분석
    
    Args:
        logs: gradient norm 로그 리스트
    
    Returns:
        Dict: 분석 결과
    """
    if not logs:
        return {}
    
    df = pd.DataFrame(logs)
    analysis = {}
    
    for component in ['total', 'lstm', 'mlp']:
        col_name = f'{component}_grad_norm'
        if col_name in df.columns:
            norms = df[col_name]
            analysis[component] = {
                'mean': float(norms.mean()),
                'std': float(norms.std()),
                'max': float(norms.max()),
                'min': float(norms.min()),
                'trend': 'increasing' if norms.iloc[-1] > norms.iloc[0] else 'decreasing',
                'stability': float(norms.std() / norms.mean()) if norms.mean() > 0 else 0.0
            }
    
    return analysis


def print_gradient_analysis(analysis: Dict[str, any]):
    """
    Gradient 분석 결과 출력
    
    Args:
        analysis: analyze_gradient_behavior 결과
    """
    print("🔍 Gradient Norm 분석:")
    for component, stats in analysis.items():
        print(f"   • {component.upper()}:")
        print(f"     - 평균: {stats['mean']:.6f}")
        print(f"     - 표준편차: {stats['std']:.6f}")
        print(f"     - 최대: {stats['max']:.6f}")
        print(f"     - 최소: {stats['min']:.6f}")
        print(f"     - 트렌드: {stats['trend']}")
        print(f"     - 안정성: {stats['stability']:.6f}")


def check_gradient_issues(gradient_norms: Dict[str, float], 
                         max_norm: float = 10.0, 
                         min_norm: float = 1e-6) -> List[str]:
    """
    Gradient 문제 체크
    
    Args:
        gradient_norms: gradient norm 딕셔너리
        max_norm: 최대 허용 norm (gradient explosion 체크)
        min_norm: 최소 허용 norm (gradient vanishing 체크)
    
    Returns:
        List[str]: 발견된 문제들
    """
    issues = []
    
    for component, norm in gradient_norms.items():
        if norm > max_norm:
            issues.append(f"{component.upper()} gradient explosion (norm: {norm:.6f})")
        elif norm < min_norm:
            issues.append(f"{component.upper()} gradient vanishing (norm: {norm:.6f})")
    
    return issues


def print_gradient_issues(issues: List[str]):
    """
    Gradient 문제 출력
    
    Args:
        issues: check_gradient_issues 결과
    """
    if issues:
        print("⚠️ Gradient 문제 발견:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ Gradient 상태 정상")
