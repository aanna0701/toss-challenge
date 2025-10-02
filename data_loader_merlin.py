#!/usr/bin/env python3
"""
NVIDIA Merlin을 활용한 고성능 데이터로더
대용량 테이블 데이터 처리를 위한 최적화된 데이터로딩 시스템
"""

import json
import os
import warnings
from typing import Tuple, Dict, List, Optional, Union
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# NVIDIA Merlin imports
try:
    import nvtabular as nvt
    from nvtabular import ops
    from nvtabular.ops import Normalize, FillMissing, Categorify, AddMetadata
    from nvtabular.workflow import Workflow
    from nvtabular.io import ParquetDataset
    from nvtabular.utils import device_mem_size, get_rmm_size
    import cudf
    import cupy as cp
    MERLIN_AVAILABLE = True
except ImportError:
    MERLIN_AVAILABLE = False
    warnings.warn("NVIDIA Merlin not available. Falling back to standard PyTorch DataLoader.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MerlinFeatureProcessor:
    """NVIDIA Merlin 기반 피처 전처리 클래스"""
    
    def __init__(self, config: Dict, normalization_stats_path: str, use_merlin: bool = True):
        self.config = config
        self.use_merlin = use_merlin and MERLIN_AVAILABLE
        self.normalization_stats_path = normalization_stats_path
        
        # 피처 분류
        self.categorical_features = self.config['MODEL']['FEATURES']['CATEGORICAL']
        self.sequential_feature = self.config['MODEL']['FEATURES']['SEQUENTIAL']
        self.excluded_features = self.config['MODEL']['FEATURES']['EXCLUDED']
        
        # Merlin 워크플로우 초기화
        self.workflow = None
        self.categorical_cardinalities = {}
        self.numerical_features = []
        
        if self.use_merlin:
            logger.info("🚀 NVIDIA Merlin 피처 프로세서 초기화")
            self._setup_merlin_workflow()
        else:
            logger.info("⚠️  NVIDIA Merlin 사용 불가. 표준 피처 프로세서 사용")
            self._load_normalization_stats()
    
    def _setup_merlin_workflow(self):
        """Merlin 워크플로우 설정"""
        try:
            # GPU 메모리 설정
            device_mem = device_mem_size()
            logger.info(f"💾 GPU 메모리: {device_mem / 1024**3:.2f} GB")
            
            # 피처 정의
            categorical_cols = self.categorical_features
            numerical_cols = self._get_numerical_features()
            
            logger.info(f"📊 범주형 피처: {len(categorical_cols)}개")
            logger.info(f"📊 수치형 피처: {len(numerical_cols)}개")
            
            # Merlin 워크플로우 구성
            categorical_ops = [
                ops.FillMissing(fill_val=0),
                ops.Categorify()
            ]
            
            numerical_ops = [
                ops.FillMissing(fill_val=0.0),
                ops.Normalize()
            ]
            
            # 워크플로우 생성
            self.workflow = Workflow(
                categorical_cols >> categorical_ops,
                numerical_cols >> numerical_ops
            )
            
            logger.info("✅ Merlin 워크플로우 설정 완료")
            
        except Exception as e:
            logger.error(f"❌ Merlin 워크플로우 설정 실패: {e}")
            self.use_merlin = False
            self._load_normalization_stats()
    
    def _get_numerical_features(self) -> List[str]:
        """수치형 피처 목록 생성"""
        # 전체 피처에서 범주형, 시퀀스, 제외 피처를 제외
        exclude_cols = set(
            self.categorical_features + 
            [self.sequential_feature, 'ID', 'clicked'] + 
            self.excluded_features
        )
        
        # 실제 데이터에서 피처 확인 (임시로 빈 리스트 반환, fit에서 실제 설정)
        return []
    
    def _load_normalization_stats(self):
        """표준화 통계 로드 (Merlin 사용 불가 시)"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            norm_stats_full_path = os.path.join(script_dir, self.normalization_stats_path)
            
            with open(norm_stats_full_path, 'r', encoding='utf-8') as f:
                self.norm_stats = json.load(f)['statistics']
            
            logger.info("✅ 표준화 통계 로드 완료")
        except Exception as e:
            logger.error(f"❌ 표준화 통계 로드 실패: {e}")
            self.norm_stats = {}
    
    def fit(self, df: pd.DataFrame) -> 'MerlinFeatureProcessor':
        """데이터에 맞춰 전처리 파이프라인 학습"""
        logger.info(f"🔧 피처 프로세서 학습 시작: {df.shape}")
        
        if self.use_merlin:
            self._fit_merlin(df)
        else:
            self._fit_standard(df)
        
        logger.info("✅ 피처 프로세서 학습 완료")
        return self
    
    def _fit_merlin(self, df: pd.DataFrame):
        """Merlin 기반 학습"""
        try:
            # cuDF로 변환
            if isinstance(df, pd.DataFrame):
                df_cudf = cudf.from_pandas(df)
            else:
                df_cudf = df
            
            # 수치형 피처 업데이트
            self.numerical_features = self._get_numerical_features_from_df(df)
            
            # 워크플로우 재설정
            categorical_cols = self.categorical_features
            numerical_cols = self.numerical_features
            
            categorical_ops = [
                ops.FillMissing(fill_val=0),
                ops.Categorify()
            ]
            
            numerical_ops = [
                ops.FillMissing(fill_val=0.0),
                ops.Normalize()
            ]
            
            self.workflow = Workflow(
                categorical_cols >> categorical_ops,
                numerical_cols >> numerical_ops
            )
            
            # 워크플로우 학습
            self.workflow.fit(df_cudf)
            
            # 카디널리티 정보 추출
            for col in categorical_cols:
                if col in df.columns:
                    unique_vals = df[col].dropna().nunique()
                    self.categorical_cardinalities[col] = unique_vals + 1  # +1 for missing
            
            logger.info(f"📊 Merlin 학습 완료 - 범주형: {len(categorical_cols)}, 수치형: {len(numerical_cols)}")
            
        except Exception as e:
            logger.error(f"❌ Merlin 학습 실패: {e}")
            self.use_merlin = False
            self._fit_standard(df)
    
    def _fit_standard(self, df: pd.DataFrame):
        """표준 PyTorch 기반 학습"""
        # 범주형 피처 카디널리티 계산
        for feat in self.categorical_features:
            if feat in df.columns:
                unique_vals = df[feat].dropna().nunique()
                self.categorical_cardinalities[feat] = unique_vals + 1  # +1 for missing
        
        # 수치형 피처 목록 생성
        exclude_cols = set(
            self.categorical_features + 
            [self.sequential_feature, 'ID', 'clicked'] + 
            self.excluded_features
        )
        self.numerical_features = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"📊 표준 학습 완료 - 범주형: {len(self.categorical_features)}, 수치형: {len(self.numerical_features)}")
    
    def _get_numerical_features_from_df(self, df: pd.DataFrame) -> List[str]:
        """데이터프레임에서 수치형 피처 추출"""
        exclude_cols = set(
            self.categorical_features + 
            [self.sequential_feature, 'ID', 'clicked'] + 
            self.excluded_features
        )
        return [col for col in df.columns if col not in exclude_cols]
    
    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """데이터 변환"""
        if self.use_merlin:
            return self._transform_merlin(df)
        else:
            return self._transform_standard(df)
    
    def _transform_merlin(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Merlin 기반 변환"""
        try:
            # cuDF로 변환
            if isinstance(df, pd.DataFrame):
                df_cudf = cudf.from_pandas(df)
            else:
                df_cudf = df
            
            # Merlin 워크플로우 적용
            transformed = self.workflow.transform(df_cudf)
            
            # 범주형 피처 추출
            categorical_data = []
            for feat in self.categorical_features:
                if feat in transformed.columns:
                    cat_data = transformed[feat].values.get()  # GPU에서 CPU로
                    categorical_data.append(cat_data)
            
            if categorical_data:
                x_categorical = torch.tensor(np.column_stack(categorical_data), dtype=torch.long)
            else:
                x_categorical = torch.empty(len(df), 0, dtype=torch.long)
            
            # 수치형 피처 추출
            numerical_data = []
            for feat in self.numerical_features:
                if feat in transformed.columns:
                    num_data = transformed[feat].values.get()  # GPU에서 CPU로
                    numerical_data.append(num_data)
            
            if numerical_data:
                x_numerical = torch.tensor(np.column_stack(numerical_data), dtype=torch.float32)
            else:
                x_numerical = torch.empty(len(df), 0, dtype=torch.float32)
            
            # 시퀀스 피처 처리 (Merlin에서는 별도 처리)
            sequences = self._process_sequences(df)
            
            return x_categorical, x_numerical, sequences
            
        except Exception as e:
            logger.error(f"❌ Merlin 변환 실패: {e}")
            return self._transform_standard(df)
    
    def _transform_standard(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """표준 PyTorch 기반 변환"""
        batch_size = len(df)
        
        # 범주형 피처 처리
        categorical_data = []
        for feat in self.categorical_features:
            if feat in df.columns:
                # 간단한 라벨 인코딩 (0부터 시작)
                unique_vals = df[feat].dropna().unique()
                val_to_idx = {val: idx + 1 for idx, val in enumerate(sorted(unique_vals))}
                encoded = df[feat].map(val_to_idx).fillna(0).astype(int)
                categorical_data.append(encoded.values)
        
        if categorical_data:
            x_categorical = torch.tensor(np.column_stack(categorical_data), dtype=torch.long)
        else:
            x_categorical = torch.empty(batch_size, 0, dtype=torch.long)
        
        # 수치형 피처 처리 (표준화)
        numerical_data = []
        for feat in self.numerical_features:
            if feat in df.columns:
                if feat in self.norm_stats:
                    mean = self.norm_stats[feat]['mean']
                    std = self.norm_stats[feat]['std']
                    feat_data = pd.to_numeric(df[feat], errors='coerce')
                    standardized = (feat_data - mean) / std
                    standardized = standardized.fillna(0)
                    numerical_data.append(standardized.values.astype(np.float32))
                else:
                    # 표준화 통계가 없으면 원본 값 사용
                    feat_data = pd.to_numeric(df[feat], errors='coerce').fillna(0)
                    numerical_data.append(feat_data.values.astype(np.float32))
        
        if numerical_data:
            x_numerical = torch.tensor(np.column_stack(numerical_data), dtype=torch.float32)
        else:
            x_numerical = torch.empty(batch_size, 0, dtype=torch.float32)
        
        # 시퀀스 피처 처리
        sequences = self._process_sequences(df)
        
        return x_categorical, x_numerical, sequences
    
    def _process_sequences(self, df: pd.DataFrame) -> List[torch.Tensor]:
        """시퀀스 피처 처리"""
        if self.sequential_feature not in df.columns:
            return [torch.tensor([0.0], dtype=torch.float32) for _ in range(len(df))]
        
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
        
        return sequences
    
    def save(self, filepath: str):
        """전처리 파이프라인 저장"""
        save_data = {
            'config': self.config,
            'categorical_features': self.categorical_features,
            'sequential_feature': self.sequential_feature,
            'excluded_features': self.excluded_features,
            'categorical_cardinalities': self.categorical_cardinalities,
            'numerical_features': self.numerical_features,
            'use_merlin': self.use_merlin,
            'norm_stats': getattr(self, 'norm_stats', {})
        }
        
        torch.save(save_data, filepath)
        logger.info(f"💾 피처 프로세서 저장 완료: {filepath}")
    
    def load(self, filepath: str):
        """전처리 파이프라인 로드"""
        save_data = torch.load(filepath, map_location='cpu')
        
        self.config = save_data['config']
        self.categorical_features = save_data['categorical_features']
        self.sequential_feature = save_data['sequential_feature']
        self.excluded_features = save_data['excluded_features']
        self.categorical_cardinalities = save_data['categorical_cardinalities']
        self.numerical_features = save_data['numerical_features']
        self.use_merlin = save_data.get('use_merlin', False)
        self.norm_stats = save_data.get('norm_stats', {})
        
        logger.info(f"📂 피처 프로세서 로드 완료: {filepath}")


class MerlinClickDataset(Dataset):
    """NVIDIA Merlin 기반 고성능 데이터셋"""
    
    def __init__(self, 
                 df: pd.DataFrame, 
                 feature_processor: MerlinFeatureProcessor,
                 target_col: str = 'clicked',
                 has_target: bool = True,
                 has_id: bool = False,
                 use_merlin: bool = True):
        
        self.df = df.reset_index(drop=True)
        self.feature_processor = feature_processor
        self.target_col = target_col
        self.has_target = has_target
        self.has_id = has_id
        self.use_merlin = use_merlin and MERLIN_AVAILABLE
        
        logger.info(f"📊 Merlin 데이터셋 초기화: {self.df.shape}")
        
        # 피처 처리
        if self.use_merlin:
            self._process_with_merlin()
        else:
            self._process_with_standard()
    
    def _process_with_merlin(self):
        """Merlin 기반 처리"""
        try:
            # cuDF로 변환
            df_cudf = cudf.from_pandas(self.df)
            
            # Merlin 워크플로우 적용
            transformed = self.feature_processor.workflow.transform(df_cudf)
            
            # 범주형 피처 처리
            self.x_categorical = self._extract_categorical_features(transformed)
            
            # 수치형 피처 처리
            self.x_numerical = self._extract_numerical_features(transformed)
            
            # 시퀀스 피처 처리
            self.sequences = self.feature_processor._process_sequences(self.df)
            
            logger.info("✅ Merlin 기반 데이터 처리 완료")
            
        except Exception as e:
            logger.error(f"❌ Merlin 처리 실패: {e}")
            self.use_merlin = False
            self._process_with_standard()
    
    def _process_with_standard(self):
        """표준 PyTorch 기반 처리"""
        self.x_categorical, self.x_numerical, self.sequences = self.feature_processor.transform(self.df)
        
        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values
        
        logger.info("✅ 표준 기반 데이터 처리 완료")
    
    def _extract_categorical_features(self, transformed_df) -> torch.Tensor:
        """범주형 피처 추출"""
        categorical_data = []
        for feat in self.feature_processor.categorical_features:
            if feat in transformed_df.columns:
                cat_data = transformed_df[feat].values.get()  # GPU에서 CPU로
                categorical_data.append(cat_data)
        
        if categorical_data:
            return torch.tensor(np.column_stack(categorical_data), dtype=torch.long)
        else:
            return torch.empty(len(self.df), 0, dtype=torch.long)
    
    def _extract_numerical_features(self, transformed_df) -> torch.Tensor:
        """수치형 피처 추출"""
        numerical_data = []
        for feat in self.feature_processor.numerical_features:
            if feat in transformed_df.columns:
                num_data = transformed_df[feat].values.get()  # GPU에서 CPU로
                numerical_data.append(num_data)
        
        if numerical_data:
            return torch.tensor(np.column_stack(numerical_data), dtype=torch.float32)
        else:
            return torch.empty(len(self.df), 0, dtype=torch.float32)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        item = {
            'x_categorical': self.x_categorical[idx],
            'x_numerical': self.x_numerical[idx],
            'seq': self.sequences[idx]
        }
        
        if self.has_id:
            if 'ID' not in self.df.columns:
                raise ValueError("❌ ID가 필요한데 데이터에 'ID' 컬럼이 없습니다!")
            item['id'] = self.df.iloc[idx]['ID']
        
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            item['y'] = y
        
        return item


def create_merlin_dataloader(dataset: MerlinClickDataset, 
                           batch_size: int = 2048,
                           shuffle: bool = True,
                           num_workers: int = 4,
                           pin_memory: bool = True) -> DataLoader:
    """Merlin 기반 고성능 DataLoader 생성"""
    
    collate_fn = collate_fn_merlin_train if dataset.has_target else collate_fn_merlin_infer
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=True if num_workers > 0 else False
    )


def collate_fn_merlin_train(batch):
    """Merlin 기반 훈련 collate 함수"""
    x_categorical = [item['x_categorical'] for item in batch]
    x_numerical = [item['x_numerical'] for item in batch]
    seqs = [item['seq'] for item in batch]
    ys = [item['y'] for item in batch]
    
    # 스택으로 변환
    x_categorical = torch.stack(x_categorical)
    x_numerical = torch.stack(x_numerical)
    ys = torch.stack(ys)
    
    # 시퀀스 패딩
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    
    return {
        'x_categorical': x_categorical,
        'x_numerical': x_numerical,
        'seqs': seqs_padded,
        'seq_lengths': seq_lengths,
        'ys': ys
    }


def collate_fn_merlin_infer(batch):
    """Merlin 기반 추론 collate 함수"""
    x_categorical = [item['x_categorical'] for item in batch]
    x_numerical = [item['x_numerical'] for item in batch]
    seqs = [item['seq'] for item in batch]
    
    if 'id' not in batch[0]:
        raise ValueError("❌ 예측 시에는 ID가 반드시 필요합니다!")
    
    ids = [item['id'] for item in batch]
    
    # 스택으로 변환
    x_categorical = torch.stack(x_categorical)
    x_numerical = torch.stack(x_numerical)
    
    # 시퀀스 패딩
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    
    return {
        'x_categorical': x_categorical,
        'x_numerical': x_numerical,
        'seqs': seqs_padded,
        'seq_lengths': seq_lengths,
        'ids': ids
    }


def safe_load_parquet(file_path: str) -> pd.DataFrame:
    """안전한 parquet 로드 함수"""
    logger.info(f"📊 데이터 로드: {file_path}")
    try:
        return pd.read_parquet(file_path, engine="pyarrow")
    except Exception as e:
        logger.error(f"❌ 데이터 로드 실패: {e}")
        raise


def load_train_data_merlin(config: Dict) -> Tuple[pd.DataFrame, List[str], str, str]:
    """Merlin 기반 훈련 데이터 로드"""
    logger.info("📊 Merlin 기반 훈련 데이터 로드 중...")
    
    train = safe_load_parquet(config['PATHS']['TRAIN_DATA'])
    logger.info(f"✅ 훈련 데이터 로드 완료: {train.shape}")
    
    # 피처 정보
    target_col = "clicked"
    seq_col = "seq"
    
    FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
    feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]
    
    # 훈련 데이터에서 ID 컬럼 제거
    if 'ID' in train.columns:
        train = train.drop(columns=['ID'])
        logger.info("✅ 훈련 데이터에서 ID 컬럼 제거 완료")
    
    logger.info(f"📊 피처 정보 - 전체: {len(feature_cols)}, 시퀀스: {seq_col}, 타겟: {target_col}")
    
    return train, feature_cols, seq_col, target_col


def load_test_data_merlin(config: Dict) -> pd.DataFrame:
    """Merlin 기반 테스트 데이터 로드"""
    logger.info("📊 Merlin 기반 테스트 데이터 로드 중...")
    
    test = safe_load_parquet(config['PATHS']['TEST_DATA'])
    
    if 'ID' not in test.columns:
        raise ValueError("❌ 테스트 데이터에 'ID' 컬럼이 없습니다!")
    
    logger.info(f"✅ 테스트 데이터 로드 완료: {test.shape[0]}개 행")
    
    return test


# 기존 함수들과의 호환성을 위한 래퍼
def load_train_data(config):
    """기존 호환성을 위한 래퍼 함수"""
    return load_train_data_merlin(config)


def load_test_data(config):
    """기존 호환성을 위한 래퍼 함수"""
    return load_test_data_merlin(config)


# 기존 클래스들과의 호환성을 위한 별칭
FeatureProcessor = MerlinFeatureProcessor
ClickDataset = MerlinClickDataset
