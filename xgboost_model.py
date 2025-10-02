import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class XGBoostModel:
    """XGBoost 모델 클래스 (시퀀스 피처 제외)"""
    
    def __init__(self, 
                 n_estimators=100,
                 max_depth=6,
                 learning_rate=0.1,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 reg_alpha=0.0,
                 reg_lambda=1.0,
                 random_state=42,
                 n_jobs=-1,
                 early_stopping_rounds=None,
                 eval_metric='rmse'):
        """
        XGBoost 모델 초기화
        
        Args:
            n_estimators: 부스팅 라운드 수
            max_depth: 트리의 최대 깊이
            learning_rate: 학습률
            subsample: 샘플링 비율
            colsample_bytree: 피처 샘플링 비율
            reg_alpha: L1 정규화
            reg_lambda: L2 정규화
            random_state: 랜덤 시드
            n_jobs: 병렬 처리 스레드 수
            early_stopping_rounds: 조기 종료 라운드
            eval_metric: 평가 메트릭
        """
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'random_state': random_state,
            'n_jobs': n_jobs,
            'early_stopping_rounds': early_stopping_rounds,
            'eval_metric': eval_metric
        }
        
        self.model = None
        self.label_encoders = {}
        self.feature_names = None
        self.is_fitted = False
        
    def _preprocess_categorical(self, X_categorical, fit=True):
        """범주형 변수 전처리"""
        if X_categorical is None or X_categorical.shape[1] == 0:
            return None
            
        X_cat_processed = X_categorical.copy()
        
        for col_idx in range(X_categorical.shape[1]):
            col_name = f'cat_{col_idx}'
            
            if fit:
                # 훈련 시: 새로운 LabelEncoder 생성
                le = LabelEncoder()
                # NaN 값을 특별한 값으로 처리
                col_data = X_categorical.iloc[:, col_idx].fillna('__nan__')
                X_cat_processed.iloc[:, col_idx] = le.fit_transform(col_data)
                self.label_encoders[col_name] = le
            else:
                # 예측 시: 기존 LabelEncoder 사용
                if col_name in self.label_encoders:
                    le = self.label_encoders[col_name]
                    col_data = X_categorical.iloc[:, col_idx].fillna('__nan__')
                    # 새로운 값이 있으면 알려진 값으로 대체
                    col_data = col_data.apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    X_cat_processed.iloc[:, col_idx] = le.transform(col_data)
                else:
                    raise ValueError(f"LabelEncoder for {col_name} not found")
                    
        return X_cat_processed
    
    def _prepare_features(self, X_categorical=None, X_numerical=None, fit=True):
        """피처 준비 (시퀀스 피처 제외)"""
        features = []
        
        # 범주형 피처 처리
        if X_categorical is not None and X_categorical.shape[1] > 0:
            X_cat_processed = self._preprocess_categorical(X_categorical, fit=fit)
            features.append(X_cat_processed)
        
        # 수치형 피처 처리
        if X_numerical is not None and X_numerical.shape[1] > 0:
            features.append(X_numerical)
        
        if not features:
            raise ValueError("No features provided")
        
        # 모든 피처 결합
        X_combined = pd.concat(features, axis=1)
        
        if fit:
            self.feature_names = X_combined.columns.tolist()
        
        return X_combined
    
    def fit(self, X_categorical=None, X_numerical=None, y=None, 
            X_val_categorical=None, X_val_numerical=None, y_val=None):
        """모델 훈련 (시퀀스 피처 제외)"""
        # 훈련 데이터 준비
        X_train = self._prepare_features(X_categorical, X_numerical, fit=True)
        
        # 검증 데이터 준비 (있는 경우)
        eval_set = None
        if (X_val_categorical is not None or X_val_numerical is not None) and y_val is not None:
            X_val = self._prepare_features(X_val_categorical, X_val_numerical, fit=False)
            eval_set = [(X_val, y_val)]
        
        # XGBoost 모델 생성 및 훈련
        self.model = xgb.XGBRegressor(**self.params)
        
        if eval_set is not None and self.params['early_stopping_rounds'] is not None:
            self.model.fit(
                X_train, y,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X_categorical=None, X_numerical=None):
        """예측 (시퀀스 피처 제외)"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_test = self._prepare_features(X_categorical, X_numerical, fit=False)
        return self.model.predict(X_test)
    
    def save(self, filepath):
        """모델 저장"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # 모델과 전처리 정보를 함께 저장
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'params': self.params
        }
        
        joblib.dump(model_data, filepath)
    
    def load(self, filepath):
        """모델 로드"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.params = model_data['params']
        self.is_fitted = True
        
        return self


def create_xgboost_model(xgb_params=None):
    """XGBoost 모델 생성 함수"""
    if xgb_params is None:
        xgb_params = {}
    
    model = XGBoostModel(**xgb_params)
    return model
