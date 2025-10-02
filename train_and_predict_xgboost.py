#!/usr/bin/env python3
"""
XGBoost 모델 전용 훈련 및 예측 워크플로우
"""

import argparse
import gc
import json
import logging
import os
import psutil
import traceback
from datetime import datetime

import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score

def parse_args():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(description='XGBoost 모델 훈련 및 예측 워크플로우 실행')
    parser.add_argument('--config', type=str, default='config_xgboost.yaml',
                       help='설정 파일 경로 (기본값: config_xgboost.yaml)')
    return parser.parse_args()

def load_config(config_path):
    """설정 파일 로드"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 설정 파일을 찾을 수 없습니다: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"📋 설정 파일 로드: {config_path}")
    return config

# 명령행 인수 파싱 및 설정 로드
args = parse_args()
CFG = load_config(args.config)

from utils import seed_everything, get_device
from data_loader import load_train_data, load_test_data
from xgboost_model import create_xgboost_model

DEVICE = get_device()

def create_results_directory():
    """결과 저장 디렉토리 생성"""
    # datetime 플레이스홀더를 실제 datetime으로 교체
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = CFG['PATHS']['RESULTS_DIR'].replace('{datetime}', timestamp)
    
    # 결과 디렉토리 생성
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 결과 디렉토리 생성: {results_dir}")
    
    return results_dir

def train_xgboost_model(train_df, CFG, results_dir):
    """XGBoost 모델 훈련"""
    print(f"\n🚀 XGBoost 모델 훈련 시작...")
    
    # 데이터 분할
    tr_df, va_df = train_test_split(train_df, test_size=CFG['VAL_SPLIT'], random_state=42, shuffle=True, stratify=train_df['clicked'])
    
    print(f"📊 Stratified Split 결과:")
    print(f"   • 전체 데이터: {len(train_df):,}개 (clicked=0: {len(train_df[train_df['clicked']==0]):,}개, clicked=1: {len(train_df[train_df['clicked']==1]):,}개)")
    print(f"   • 훈련 데이터: {len(tr_df):,}개 (clicked=0: {len(tr_df[tr_df['clicked']==0]):,}개, clicked=1: {len(tr_df[tr_df['clicked']==1]):,}개)")
    print(f"   • 검증 데이터: {len(va_df):,}개 (clicked=0: {len(va_df[va_df['clicked']==0]):,}개, clicked=1: {len(va_df[va_df['clicked']==1]):,}개)")
    
    # 클래스 비율 확인
    train_ratio_0 = len(tr_df[tr_df['clicked']==0]) / len(tr_df)
    train_ratio_1 = len(tr_df[tr_df['clicked']==1]) / len(tr_df)
    val_ratio_0 = len(va_df[va_df['clicked']==0]) / len(va_df)
    val_ratio_1 = len(va_df[va_df['clicked']==1]) / len(va_df)
    print(f"   • 훈련 데이터 클래스 비율: clicked=0 ({train_ratio_0:.3f}), clicked=1 ({train_ratio_1:.3f})")
    print(f"   • 검증 데이터 클래스 비율: clicked=0 ({val_ratio_0:.3f}), clicked=1 ({val_ratio_1:.3f})")
    
    # 피처 분리 (시퀀스 피처 제외)
    categorical_features = CFG['MODEL']['FEATURES']['CATEGORICAL']
    excluded_features = CFG['MODEL']['FEATURES']['EXCLUDED']
    
    # 수치형 피처는 범주형과 제외 피처를 제외한 나머지
    numerical_features = [col for col in train_df.columns 
                         if col not in categorical_features 
                         and col not in excluded_features
                         and col not in ['clicked', 'id']]
    
    print(f"📊 피처 정보:")
    print(f"   • 범주형 피처: {len(categorical_features)}개")
    print(f"   • 수치형 피처: {len(numerical_features)}개")
    print(f"   • 시퀀스 피처: 사용하지 않음 (XGBoost)")
    print(f"   • 제외 피처: {excluded_features}")
    
    # 훈련 데이터 준비
    train_categorical = tr_df[categorical_features] if categorical_features else None
    train_numerical = tr_df[numerical_features] if numerical_features else None
    train_y = tr_df['clicked'].values
    
    # 검증 데이터 준비
    val_categorical = va_df[categorical_features] if categorical_features else None
    val_numerical = va_df[numerical_features] if numerical_features else None
    val_y = va_df['clicked'].values
    
    # XGBoost 모델 생성
    xgb_config = CFG['MODEL']['XGBOOST']
    xgb_params = {
        'n_estimators': xgb_config['N_ESTIMATORS'],
        'max_depth': xgb_config['MAX_DEPTH'],
        'learning_rate': xgb_config['LEARNING_RATE'],
        'subsample': xgb_config['SUBSAMPLE'],
        'colsample_bytree': xgb_config['COLSAMPLE_BYTREE'],
        'reg_alpha': xgb_config['REG_ALPHA'],
        'reg_lambda': xgb_config['REG_LAMBDA'],
        'random_state': xgb_config['RANDOM_STATE'],
        'n_jobs': xgb_config['N_JOBS'],
        'early_stopping_rounds': xgb_config['EARLY_STOPPING_ROUNDS'],
        'eval_metric': xgb_config['EVAL_METRIC']
    }
    
    model = create_xgboost_model(xgb_params)
    
    print(f"🔧 XGBoost 모델 설정:")
    print(f"   • N Estimators: {xgb_params['n_estimators']}")
    print(f"   • Max Depth: {xgb_params['max_depth']}")
    print(f"   • Learning Rate: {xgb_params['learning_rate']}")
    print(f"   • Subsample: {xgb_params['subsample']}")
    print(f"   • Colsample By Tree: {xgb_params['colsample_bytree']}")
    print(f"   • Reg Alpha: {xgb_params['reg_alpha']}")
    print(f"   • Reg Lambda: {xgb_params['reg_lambda']}")
    print(f"   • Early Stopping Rounds: {xgb_params['early_stopping_rounds']}")
    print(f"   • Random State: {xgb_params['random_state']}")
    print(f"   • N Jobs: {xgb_params['n_jobs']}")
    
    # 모델 훈련
    print(f"🚀 모델 훈련 중...")
    model.fit(
        X_categorical=train_categorical,
        X_numerical=train_numerical,
        y=train_y,
        X_val_categorical=val_categorical,
        X_val_numerical=val_numerical,
        y_val=val_y
    )
    
    # 검증 데이터로 성능 평가
    print(f"📊 성능 평가 중...")
    val_predictions = model.predict(
        X_categorical=val_categorical,
        X_numerical=val_numerical
    )
    
    # 메트릭 계산
    val_rmse = np.sqrt(mean_squared_error(val_y, val_predictions))
    
    # AUC 계산을 위해 sigmoid 적용
    val_probs = 1 / (1 + np.exp(-val_predictions))  # sigmoid
    val_auc = roc_auc_score(val_y, val_probs)
    
    print(f"✅ XGBoost 훈련 완료!")
    print(f"   • 검증 RMSE: {val_rmse:.6f}")
    print(f"   • 검증 AUC: {val_auc:.6f}")
    
    # 모델 저장
    model_path = os.path.join(results_dir, "xgboost_model.pkl")
    model.save(model_path)
    print(f"💾 XGBoost 모델 저장 완료: {model_path}")
    
    return model, categorical_features, numerical_features

def predict_xgboost_model(test_df, model, categorical_features, numerical_features, CFG, results_dir):
    """XGBoost 모델 예측"""
    print(f"\n🔮 XGBoost 모델 예측 시작...")
    
    # 테스트 데이터 준비
    test_categorical = test_df[categorical_features] if categorical_features else None
    test_numerical = test_df[numerical_features] if numerical_features else None
    
    print(f"📊 테스트 데이터:")
    print(f"   • 테스트 샘플: {len(test_df):,}개")
    print(f"   • 범주형 피처: {len(categorical_features)}개")
    print(f"   • 수치형 피처: {len(numerical_features)}개")
    
    # 예측 수행
    print(f"🔮 예측 수행 중...")
    predictions = model.predict(
        X_categorical=test_categorical,
        X_numerical=test_numerical
    )
    
    # sigmoid 적용하여 확률로 변환
    probabilities = 1 / (1 + np.exp(-predictions))
    
    print(f"✅ 예측 완료!")
    print(f"📊 예측 결과 통계:")
    print(f"   • Shape: {probabilities.shape}")
    print(f"   • Min: {probabilities.min():.4f}")
    print(f"   • Max: {probabilities.max():.4f}")
    print(f"   • Mean: {probabilities.mean():.4f}")
    print(f"   • Std: {probabilities.std():.4f}")
    
    # 제출 파일 생성
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'clicked': probabilities
    })
    
    submission_path = os.path.join(results_dir, f"xgboost_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"💾 제출 파일 저장 완료: {submission_path}")
    
    return submission_df

def main():
    """메인 실행 함수"""
    try:
        print("🚀 XGBoost 모델 훈련 및 예측 워크플로우 시작")
        print("=" * 60)
        
        # 시드 설정
        seed_everything(CFG['SEED'])
        print(f"🎲 시드 설정: {CFG['SEED']}")
        
        # 결과 디렉토리 생성
        results_dir = create_results_directory()
        
        # 메모리 사용량 확인
        memory_info = psutil.virtual_memory()
        print(f"💾 메모리 사용량: {memory_info.percent:.1f}% ({memory_info.used / 1024**3:.1f}GB / {memory_info.total / 1024**3:.1f}GB)")
        
        # 훈련 데이터 로드
        print(f"\n📊 훈련 데이터 로드 중...")
        train_df = load_train_data(CFG['PATHS']['TRAIN_DATA'])
        print(f"   • 훈련 데이터 크기: {train_df.shape}")
        print(f"   • 클래스 분포: {train_df['clicked'].value_counts().to_dict()}")
        
        # XGBoost 모델 훈련
        model, categorical_features, numerical_features = train_xgboost_model(train_df, CFG, results_dir)
        
        # 메모리 정리
        del train_df
        gc.collect()
        
        # 테스트 데이터 로드
        print(f"\n📊 테스트 데이터 로드 중...")
        test_df = load_test_data(CFG['PATHS']['TEST_DATA'])
        print(f"   • 테스트 데이터 크기: {test_df.shape}")
        
        # 예측 수행
        submission_df = predict_xgboost_model(test_df, model, categorical_features, numerical_features, CFG, results_dir)
        
        # 메타데이터 저장
        metadata = {
            'model_type': 'xgboost',
            'config': CFG,
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'timestamp': datetime.now().isoformat(),
            'submission_shape': submission_df.shape
        }
        
        metadata_path = os.path.join(results_dir, f"metadata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        print(f"💾 메타데이터 저장 완료: {metadata_path}")
        
        print("\n🎉 XGBoost 워크플로우 완료!")
        print(f"📁 결과 저장 위치: {results_dir}")
        
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        print(f"📋 상세 오류 정보:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
