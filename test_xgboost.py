#!/usr/bin/env python3
"""
XGBoost 모델 테스트 스크립트
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from xgboost_model import create_xgboost_model

def create_sample_data(n_samples=10000, n_categorical=5, n_numerical=20):
    """샘플 데이터 생성 (시퀀스 피처 제외)"""
    np.random.seed(42)
    
    # 범주형 피처 생성
    categorical_data = {}
    for i in range(n_categorical):
        categories = [f'cat_{i}_val_{j}' for j in range(np.random.randint(3, 10))]
        categorical_data[f'cat_{i}'] = np.random.choice(categories, n_samples)
    
    # 수치형 피처 생성
    numerical_data = {}
    for i in range(n_numerical):
        numerical_data[f'num_{i}'] = np.random.normal(0, 1, n_samples)
    
    # 타겟 생성 (범주형과 수치형 피처의 조합으로)
    target = np.zeros(n_samples)
    for i in range(n_categorical):
        target += pd.Categorical(categorical_data[f'cat_{i}']).codes * 0.1
    for i in range(n_numerical):
        target += numerical_data[f'num_{i}'] * 0.05
    
    # sigmoid 적용하여 확률로 변환
    target_probs = 1 / (1 + np.exp(-target))
    target_binary = (target_probs > 0.5).astype(int)
    
    # DataFrame 생성
    data = {**categorical_data, **numerical_data, 'target': target_binary}
    df = pd.DataFrame(data)
    
    return df

def test_xgboost_model():
    """XGBoost 모델 테스트"""
    print("🧪 XGBoost 모델 테스트 시작")
    print("=" * 50)
    
    # 샘플 데이터 생성
    print("📊 샘플 데이터 생성 중...")
    df = create_sample_data(n_samples=10000, n_categorical=5, n_numerical=20)
    print(f"   • 데이터 크기: {df.shape}")
    print(f"   • 타겟 분포: {df['target'].value_counts().to_dict()}")
    
    # 범주형과 수치형 피처 분리
    categorical_features = [col for col in df.columns if col.startswith('cat_')]
    numerical_features = [col for col in df.columns if col.startswith('num_')]
    
    print(f"   • 범주형 피처: {len(categorical_features)}개")
    print(f"   • 수치형 피처: {len(numerical_features)}개")
    
    # 훈련/검증 분할
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    print(f"   • 훈련 데이터: {len(train_df)}개")
    print(f"   • 검증 데이터: {len(val_df)}개")
    
    # 피처 준비
    train_categorical = train_df[categorical_features]
    train_numerical = train_df[numerical_features]
    train_y = train_df['target'].values
    
    val_categorical = val_df[categorical_features]
    val_numerical = val_df[numerical_features]
    val_y = val_df['target'].values
    
    # XGBoost 모델 생성
    print("\n🔧 XGBoost 모델 생성 중...")
    xgb_params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'early_stopping_rounds': 20,
        'eval_metric': 'rmse'
    }
    
    model = create_xgboost_model(xgb_params)
    print(f"   • 모델 파라미터: {xgb_params}")
    
    # 모델 훈련
    print("\n🚀 모델 훈련 중...")
    model.fit(
        X_categorical=train_categorical,
        X_numerical=train_numerical,
        y=train_y,
        X_val_categorical=val_categorical,
        X_val_numerical=val_numerical,
        y_val=val_y
    )
    print("   ✅ 훈련 완료!")
    
    # 예측
    print("\n🔮 예측 수행 중...")
    val_predictions = model.predict(
        X_categorical=val_categorical,
        X_numerical=val_numerical
    )
    print(f"   • 예측 결과 shape: {val_predictions.shape}")
    
    # 성능 평가
    val_rmse = np.sqrt(mean_squared_error(val_y, val_predictions))
    val_probs = 1 / (1 + np.exp(-val_predictions))  # sigmoid
    val_auc = roc_auc_score(val_y, val_probs)
    
    print(f"\n📊 성능 평가 결과:")
    print(f"   • 검증 RMSE: {val_rmse:.6f}")
    print(f"   • 검증 AUC: {val_auc:.6f}")
    
    # 모델 저장/로드 테스트
    print("\n💾 모델 저장/로드 테스트 중...")
    model.save("test_xgboost_model.pkl")
    print("   ✅ 모델 저장 완료!")
    
    # 새 모델 인스턴스로 로드
    new_model = create_xgboost_model()
    new_model.load("test_xgboost_model.pkl")
    print("   ✅ 모델 로드 완료!")
    
    # 로드된 모델로 예측
    loaded_predictions = new_model.predict(
        X_categorical=val_categorical,
        X_numerical=val_numerical
    )
    
    # 예측 결과 비교
    predictions_match = np.allclose(val_predictions, loaded_predictions, rtol=1e-10)
    print(f"   • 예측 결과 일치: {predictions_match}")
    
    # 정리
    import os
    if os.path.exists("test_xgboost_model.pkl"):
        os.remove("test_xgboost_model.pkl")
        print("   🗑️ 테스트 파일 정리 완료!")
    
    print("\n🎉 XGBoost 모델 테스트 완료!")
    return val_auc

if __name__ == "__main__":
    test_xgboost_model()
