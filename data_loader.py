import json
import os
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class FeatureProcessor:
    """피처 전처리 클래스 (LabelEncoder 사용)"""
    
    def __init__(self, config, normalization_stats_path):
        self.config = config
        
        # Normalization stats 로드
        script_dir = os.path.dirname(os.path.abspath(__file__))
        norm_stats_full_path = os.path.join(script_dir, normalization_stats_path)
        with open(norm_stats_full_path, 'r', encoding='utf-8') as f:
            self.norm_stats = json.load(f)['statistics']
        
        # 피처 분류
        self.categorical_features = self.config['MODEL']['FEATURES']['CATEGORICAL']
        self.sequential_feature = self.config['MODEL']['FEATURES']['SEQUENTIAL']
        self.excluded_features = self.config['MODEL']['FEATURES']['EXCLUDED']
        
        # numerical_features는 fit 시점에 데이터를 보고 결정
        self.numerical_features = []
        
        # LabelEncoder 사용
        self.label_encoders = {}  # {feat: LabelEncoder()}
        self.categorical_cardinalities = {}
        
    def fit(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None):
        """
        LabelEncoder를 사용하여 범주형 피처 인코딩 학습
        train_df와 test_df를 모두 받아서 전체 범주 파악
        """
        # 수치형 피처 목록 생성 (범주형, 시퀀스, ID, target, 제외 피처 제외)
        exclude_cols = set(self.categorical_features + [self.sequential_feature, 'ID', 'clicked'] + self.excluded_features)
        self.numerical_features = [col for col in train_df.columns if col not in exclude_cols]
        
        print("🔧 범주형 피처 인코딩 학습 중...")
        # 범주형 피처 인코딩 설정 (LabelEncoder 사용)
        for feat in self.categorical_features:
            if feat not in train_df.columns:
                continue
            
            # train과 test의 모든 값을 합쳐서 fit
            if test_df is not None and feat in test_df.columns:
                all_values = pd.concat([
                    train_df[feat].astype(str).fillna("UNK"),
                    test_df[feat].astype(str).fillna("UNK")
                ], axis=0)
            else:
                all_values = train_df[feat].astype(str).fillna("UNK")
            
            le = LabelEncoder()
            le.fit(all_values)
            
            self.label_encoders[feat] = le
            self.categorical_cardinalities[feat] = len(le.classes_)
            
            print(f"   • {feat}: {len(le.classes_)} unique categories")
        
        print("✅ 범주형 피처 인코딩 학습 완료")
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """데이터 변환 (LabelEncoder 사용)"""
        batch_size = len(df)
        
        # 범주형 피처 처리 (LabelEncoder 사용)
        categorical_data = []
        for feat in self.categorical_features:
            if feat in df.columns:
                # LabelEncoder로 변환
                encoded = self.label_encoders[feat].transform(
                    df[feat].astype(str).fillna("UNK")
                )
                categorical_data.append(encoded)
            else:
                raise ValueError(f"❌ 범주형 피처 '{feat}'가 데이터에 없습니다!")
        
        if categorical_data:
            x_categorical = torch.tensor(np.column_stack(categorical_data), dtype=torch.long)
        else:
            x_categorical = torch.empty(batch_size, 0, dtype=torch.long)
        
        # 수치형 피처 처리 (표준화) - 무조건 적용
        numerical_data = []
        for feat in self.numerical_features:
            if feat in df.columns:
                if feat in self.norm_stats:
                    # 표준화: (x - mean) / std
                    mean = self.norm_stats[feat]['mean']
                    std = self.norm_stats[feat]['std']
                    # 데이터를 float로 변환
                    feat_data = pd.to_numeric(df[feat], errors='coerce')
                    # 결측치를 제외하고 표준화한 후, 결측치를 0으로 채움
                    standardized = (feat_data - mean) / std
                    standardized = standardized.fillna(0)
                    numerical_data.append(standardized.values.astype(np.float32))
                else:
                    # norm_stats에 없는 경우 에러 발생
                    raise ValueError(f"❌ {feat} 피처의 normalization stats가 없습니다! config.yaml과 normalization_stats.json을 확인해주세요.")
            else:
                raise ValueError(f"❌ {feat} 피처가 데이터에 없습니다!")
        
        if numerical_data:
            x_numerical = torch.tensor(np.column_stack(numerical_data), dtype=torch.float32)
        else:
            x_numerical = torch.empty(batch_size, 0, dtype=torch.float32)
        
        # 시퀀스 피처 처리
        if self.sequential_feature in df.columns:
            seq_strings = df[self.sequential_feature].astype(str).values
            sequences = []
            for s in seq_strings:
                if s and s != 'nan':
                    try:
                        arr = np.fromstring(s, sep=",", dtype=np.float32)
                        if arr.size == 0:
                            arr = np.array([0.0], dtype=np.float32)
                    except:
                        arr = np.array([0.0], dtype=np.float32)
                else:
                    arr = np.array([0.0], dtype=np.float32)
                sequences.append(torch.from_numpy(arr))
        else:
            raise ValueError(f"❌ 시퀀스 피처 '{self.sequential_feature}'가 데이터에 없습니다!")
        
        return x_categorical, x_numerical, sequences


class ClickDataset(Dataset):
    def __init__(self, df, feature_processor: FeatureProcessor, target_col=None, has_target=True, has_id=False):
        self.df = df.reset_index(drop=True)
        self.feature_processor = feature_processor
        self.target_col = target_col
        self.has_target = has_target
        self.has_id = has_id

        # 피처 처리
        self.x_categorical, self.x_numerical, self.sequences = feature_processor.transform(df)

        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 딕셔너리 형태로 반환
        item = {
            'x_categorical': self.x_categorical[idx],
            'x_numerical': self.x_numerical[idx],
            'seq': self.sequences[idx]
        }
        
        # ID가 필요한 경우에만 처리
        if self.has_id:
            if 'ID' not in self.df.columns:
                raise ValueError("❌ ID가 필요한데 데이터에 'ID' 컬럼이 없습니다!")
            item['id'] = self.df.iloc[idx]['ID']
        
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            item['y'] = y
        
        return item



def collate_fn_transformer_train(batch):
    """Transformer 모델용 훈련 collate 함수"""
    # 딕셔너리 배치에서 값들 추출
    x_categorical = [item['x_categorical'] for item in batch]
    x_numerical = [item['x_numerical'] for item in batch]
    seqs = [item['seq'] for item in batch]
    ys = [item['y'] for item in batch]  # has_target=True인 경우만
    
    # 스택으로 변환
    x_categorical = torch.stack(x_categorical)
    x_numerical = torch.stack(x_numerical)
    ys = torch.stack(ys)
    
    # 시퀀스 패딩
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)  # 빈 시퀀스 방지
    
    # 딕셔너리 형태로 배치 반환
    return {
        'x_categorical': x_categorical,
        'x_numerical': x_numerical,
        'seqs': seqs_padded,
        'seq_lengths': seq_lengths,
        'ys': ys
    }

def collate_fn_transformer_infer(batch):
    """Transformer 모델용 추론 collate 함수"""
    # 딕셔너리 배치에서 값들 추출
    x_categorical = [item['x_categorical'] for item in batch]
    x_numerical = [item['x_numerical'] for item in batch]
    seqs = [item['seq'] for item in batch]
    
    # 예측 시에는 ID가 반드시 필요
    if 'id' not in batch[0]:
        raise ValueError("❌ 예측 시에는 ID가 반드시 필요합니다! 테스트 데이터에 'ID' 컬럼이 있는지 확인해주세요.")
    
    ids = [item['id'] for item in batch]
    
    # ID에 None이 포함되어 있는지 확인
    if any(id_val is None for id_val in ids):
        raise ValueError("❌ ID 값에 None이 포함되어 있습니다! 테스트 데이터의 'ID' 컬럼을 확인해주세요.")
    
    # 스택으로 변환
    x_categorical = torch.stack(x_categorical)
    x_numerical = torch.stack(x_numerical)
    
    # 시퀀스 패딩
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    
    # 딕셔너리 형태로 배치 반환
    result = {
        'x_categorical': x_categorical,
        'x_numerical': x_numerical,
        'seqs': seqs_padded,
        'seq_lengths': seq_lengths,
        'ids': ids
    }
    
    return result


def safe_load_parquet(file_path):
    """안전한 parquet 로드 함수 - 항상 전체 데이터 로드"""
    print(f"📊 전체 데이터 로드 - {file_path}")
    try:
        return pd.read_parquet(file_path, engine="pyarrow")
    except Exception as e:
        print(f"⚠️  데이터 로드 실패: {e}")
        raise


def load_train_data(config):
    """훈련 데이터만 로드 및 전처리 함수"""
    print("📊 훈련 데이터 로드 중...")
    train = safe_load_parquet(config['PATHS']['TRAIN_DATA'])
    
    print("Train shape:", train.shape)
    
    # Target / Sequence
    target_col = "clicked"
    seq_col = "seq"

    # 학습에 사용할 피처: ID/seq/target/제외 피처 제외, 나머지 전부
    excluded_features = config['MODEL']['FEATURES']['EXCLUDED']
    FEATURE_EXCLUDE = {target_col, seq_col, "ID"} | set(excluded_features)
    feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

    # 훈련 데이터에서 ID 컬럼 제거 (훈련 시에는 ID가 필요하지 않음)
    if 'ID' in train.columns:
        train = train.drop(columns=['ID'])
        print("✅ 훈련 데이터에서 ID 컬럼 제거 완료")

    print("Num features:", len(feature_cols))
    print("Sequence:", seq_col)
    print("Target:", target_col)
    print("Excluded features:", excluded_features)

    return train, feature_cols, seq_col, target_col


def load_test_data(config):
    """테스트 데이터만 로드 함수"""
    print("📊 테스트 데이터 로드 중...")
    test = safe_load_parquet(config['PATHS']['TEST_DATA'])
    
    # 테스트 데이터에는 ID 컬럼이 반드시 있어야 함 (예측 시 필요)
    if 'ID' not in test.columns:
        raise ValueError("❌ 테스트 데이터에 'ID' 컬럼이 없습니다! 예측을 위해서는 ID가 필요합니다.")
    
    print(f"✅ 테스트 데이터 로드 완료: {test.shape[0]}개 행, ID 컬럼 포함")
    print("Test shape:", test.shape)
    
    return test


def save_feature_processor(feature_processor: FeatureProcessor, save_path: str):
    """FeatureProcessor를 파일로 저장"""
    print(f"💾 FeatureProcessor 저장 중...")
    print(f"   • 경로: {save_path}")
    
    # 디렉토리가 없으면 생성
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    # FeatureProcessor 저장
    with open(save_path, 'wb') as f:
        pickle.dump(feature_processor, f)
    
    print(f"✅ FeatureProcessor 저장 완료: {save_path}")
    print(f"   • 범주형 피처: {len(feature_processor.categorical_features)}개")
    print(f"   • 수치형 피처: {len(feature_processor.numerical_features)}개")
    print(f"   • 시퀀스 컬럼: {feature_processor.sequential_feature}")


def load_feature_processor(load_path: str) -> FeatureProcessor:
    """저장된 FeatureProcessor를 로드"""
    print(f"📂 FeatureProcessor 로드 중...")
    print(f"   • 경로: {load_path}")
    
    # 파일 존재 확인
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"❌ FeatureProcessor 파일을 찾을 수 없습니다: {load_path}")
    
    # FeatureProcessor 로드
    with open(load_path, 'rb') as f:
        feature_processor = pickle.load(f)
    
    print(f"✅ FeatureProcessor 로드 완료")
    print(f"   • 범주형 피처: {len(feature_processor.categorical_features)}개")
    print(f"   • 수치형 피처: {len(feature_processor.numerical_features)}개")
    print(f"   • 시퀀스 컬럼: {feature_processor.sequential_feature}")
    
    return feature_processor
