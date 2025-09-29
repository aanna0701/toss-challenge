import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
import os
from typing import Tuple

EPS_STD = 1e-6  # 표준화 분모 안정화


class FeatureProcessor:
    """피처 전처리 클래스 (+1 인덱싱, seq 결측 정합, 표준화 eps)"""

    def __init__(self, CFG: dict, normalization_stats_path: str = "analysis/results/normalization_stats.json"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config = CFG

        # Normalization stats 로드
        norm_stats_full_path = os.path.join(script_dir, normalization_stats_path)
        with open(norm_stats_full_path, 'r', encoding='utf-8') as f:
            self.norm_stats = json.load(f)['statistics']

        # 피처 분류
        self.categorical_features = self.config['MODEL']['FEATURES']['CATEGORICAL']
        self.sequential_feature = self.config['MODEL']['FEATURES']['SEQUENTIAL']
        self.excluded_features = self.config['MODEL']['FEATURES']['EXCLUDED']

        # 범주형 피처의 카디널리티 및 인코더
        self.categorical_cardinalities = {}
        self.categorical_encoders = {}

    def fit(self, df: pd.DataFrame):
        """데이터에 맞춰 인코더 학습 (+1 인덱싱: 0은 OOV/미싱 예약)"""
        for feat in self.categorical_features:
            if feat in df.columns:
                unique_vals = df[feat].dropna().unique()
                unique_vals_sorted = sorted(unique_vals)
                # 1..N 매핑, 0은 OOV/미싱
                self.categorical_encoders[feat] = {val: idx + 1 for idx, val in enumerate(unique_vals_sorted)}
                self.categorical_cardinalities[feat] = len(unique_vals_sorted) + 1  # +1 for OOV/미싱
            else:
                # 존재하지 않으면 최소 카디널리티 1 (OOV만)로 설정
                self.categorical_encoders[feat] = {}
                self.categorical_cardinalities[feat] = 1

        # 수치형 피처 목록 생성 (범주형, 시퀀스, ID, target, 제외 피처 제외)
        exclude_cols = set(self.categorical_features + [self.sequential_feature, 'ID', 'clicked'] + self.excluded_features)
        self.numerical_features = [col for col in df.columns if col not in exclude_cols]
        return self

    def _encode_categorical_column(self, s: pd.Series, feat: str) -> np.ndarray:
        enc = self.categorical_encoders.get(feat, {})
        # 미등록/OOV → 0, 결측도 0
        return s.map(enc).fillna(0).astype(int).values

    def transform(self, df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor, list, torch.Tensor]:
        """데이터 변환: (x_categorical, x_numerical, sequences(list[tensor]), nan_mask)"""

        batch_size = len(df)

        # 범주형 처리 (+1 인덱싱)
        categorical_data = []
        for feat in self.categorical_features:
            if feat in df.columns:
                encoded = self._encode_categorical_column(df[feat], feat)
                categorical_data.append(encoded)
            else:
                # 없는 컬럼은 전부 0(OOV/미싱)
                categorical_data.append(np.zeros(batch_size, dtype=np.int64))
        x_categorical = torch.tensor(np.column_stack(categorical_data), dtype=torch.long) if categorical_data \
            else torch.empty(batch_size, 0, dtype=torch.long)

        # 수치형 처리(표준화 + eps)
        numerical_data = []
        for feat in self.numerical_features:
            if feat in df.columns:
                if feat not in self.norm_stats:
                    raise ValueError(f"❌ {feat} 피처의 normalization stats가 없습니다! config.yaml과 normalization_stats.json을 확인해주세요.")
                mean = float(self.norm_stats[feat]['mean'])
                std = float(self.norm_stats[feat]['std'])
                std = max(std, EPS_STD)
                feat_data = pd.to_numeric(df[feat], errors='coerce')  # NaN 유지
                standardized = ((feat_data - mean) / std).astype(np.float32)
                # NaN은 그대로 두고 nan_mask에서 처리 → 모델에서 nan_token으로 바꿈
                numerical_data.append(standardized.values)
            else:
                raise ValueError(f"❌ {feat} 피처가 데이터에 없습니다!")
        x_numerical = torch.tensor(np.column_stack(numerical_data), dtype=torch.float32) if numerical_data \
            else torch.empty(batch_size, 0, dtype=torch.float32)

        # 시퀀스 처리: 파싱 실패/빈 문자열/문자열 'nan' → 결측으로 간주
        seq_is_missing = np.zeros(batch_size, dtype=np.int64)
        sequences = []
        if self.sequential_feature in df.columns:
            seq_strings = df[self.sequential_feature].astype(str).values
            for i, s in enumerate(seq_strings):
                val = s.strip()
                if (val == "") or (val.lower() == "nan"):
                    seq_is_missing[i] = 1
                    arr = np.array([0.0], dtype=np.float32)
                else:
                    try:
                        arr = np.fromstring(val, sep=",", dtype=np.float32)
                        if arr.size == 0:
                            seq_is_missing[i] = 1
                            arr = np.array([0.0], dtype=np.float32)
                    except Exception:
                        seq_is_missing[i] = 1
                        arr = np.array([0.0], dtype=np.float32)
                sequences.append(torch.from_numpy(arr))
        else:
            raise ValueError(f"❌ 시퀀스 피처 '{self.sequential_feature}'가 데이터에 없습니다!")

        # NaN 마스크 생성 (범주형: df.isna OR OOV? → OOV는 인덱스 0이므로 모델에서 처리,
        # 여기서는 '원본 NaN'만 표시. 수치형: 원본 NaN. 시퀀스: df.isna OR 파싱 결측)
        nan_mask_parts = []

        # 범주형 NaN 마스크 (원본 NaN 기준)
        for feat in self.categorical_features:
            if feat in df.columns:
                nan_mask_parts.append(df[feat].isna().astype(int).values)
            else:
                nan_mask_parts.append(np.ones(batch_size, dtype=int))

        # 수치형 NaN 마스크
        for feat in self.numerical_features:
            if feat in df.columns:
                nan_mask_parts.append(df[feat].isna().astype(int).values)
            else:
                nan_mask_parts.append(np.ones(batch_size, dtype=int))

        # 시퀀스 NaN 마스크 (원본 NaN OR 파싱결측)
        if self.sequential_feature in df.columns:
            seq_nan = df[self.sequential_feature].isna().astype(int).values
            nan_mask_parts.append(np.maximum(seq_nan, seq_is_missing))
        else:
            nan_mask_parts.append(np.ones(batch_size, dtype=int))

        nan_mask = torch.tensor(np.column_stack(nan_mask_parts), dtype=torch.float32)

        return x_categorical, x_numerical, sequences, nan_mask


class ClickDataset(Dataset):
    def __init__(self, df, feature_processor: FeatureProcessor, target_col=None, has_target=True, has_id=False):
        self.df = df.reset_index(drop=True)
        self.feature_processor = feature_processor
        self.target_col = target_col
        self.has_target = has_target
        self.has_id = has_id

        self.x_categorical, self.x_numerical, self.sequences, self.nan_mask = feature_processor.transform(df)

        if self.has_target:
            if self.target_col is None or self.target_col not in self.df.columns:
                raise ValueError("❌ 타깃 컬럼이 누락되었습니다.")
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = {
            'x_categorical': self.x_categorical[idx],
            'x_numerical': self.x_numerical[idx],
            'seq': self.sequences[idx],
            'nan_mask': self.nan_mask[idx],
        }
        if self.has_id:
            if 'ID' not in self.df.columns:
                raise ValueError("❌ ID가 필요한데 데이터에 'ID' 컬럼이 없습니다!")
            item['id'] = self.df.iloc[idx]['ID']
        if self.has_target:
            item['y'] = torch.tensor(self.y[idx], dtype=torch.float32)
        return item


def collate_fn_transformer_train(batch):
    x_categorical = torch.stack([b['x_categorical'] for b in batch])
    x_numerical = torch.stack([b['x_numerical'] for b in batch])
    nan_masks = torch.stack([b['nan_mask'] for b in batch])
    ys = torch.stack([b['y'] for b in batch])

    seqs = [b['seq'] for b in batch]
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long).clamp_min(1)

    return {
        'x_categorical': x_categorical,
        'x_numerical': x_numerical,
        'seqs': seqs_padded,
        'seq_lengths': seq_lengths,
        'nan_mask': nan_masks,
        'ys': ys,
    }


def collate_fn_transformer_infer(batch):
    if 'id' not in batch[0]:
        raise ValueError("❌ 예측 시에는 ID가 반드시 필요합니다! 테스트 데이터에 'ID' 컬럼이 있는지 확인해주세요.")

    x_categorical = torch.stack([b['x_categorical'] for b in batch])
    x_numerical = torch.stack([b['x_numerical'] for b in batch])
    nan_masks = torch.stack([b['nan_mask'] for b in batch])
    ids = [b['id'] for b in batch]
    if any(id_val is None for id_val in ids):
        raise ValueError("❌ ID 값에 None이 포함되어 있습니다! 테스트 데이터의 'ID' 컬럼을 확인해주세요.")

    seqs = [b['seq'] for b in batch]
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long).clamp_min(1)

    return {
        'x_categorical': x_categorical,
        'x_numerical': x_numerical,
        'seqs': seqs_padded,
        'seq_lengths': seq_lengths,
        'nan_mask': nan_masks,
        'ids': ids,
    }


def create_data_loaders(train_df, val_df, test_df, feature_cols, seq_col, target_col, batch_size, CFG):
    feature_processor = FeatureProcessor(CFG)
    if train_df is not None and len(train_df) > 0:
        feature_processor.fit(train_df)
    else:
        # 더미 데이터로 fit
        dummy = {col: [0.0] for col in feature_cols}
        dummy[seq_col] = ["0.0"]
        dummy[target_col] = [0.0]
        feature_processor.fit(pd.DataFrame(dummy))

    # Train
    if train_df is not None and len(train_df) > 0:
        train_dataset = ClickDataset(train_df, feature_processor, target_col, has_target=True, has_id=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=collate_fn_transformer_train)
    else:
        train_loader = None
        train_dataset = None

    # Val
    if val_df is not None and len(val_df) > 0:
        val_dataset = ClickDataset(val_df, feature_processor, target_col, has_target=True, has_id=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=collate_fn_transformer_train)
    else:
        val_loader = None
        val_dataset = None

    # Test
    if test_df is not None and len(test_df) > 0:
        test_dataset = ClickDataset(test_df, feature_processor, has_target=False, has_id=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=collate_fn_transformer_infer)
    else:
        test_loader = None
        test_dataset = None

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, feature_processor


def load_and_preprocess_data(CFG):
    """데이터 로드 및 전처리 함수"""
    def safe_load_parquet(file_path):
        """안전한 parquet 로드 함수 - 항상 전체 데이터 로드"""
        print(f"📊 전체 데이터 로드 - {file_path}")
        try:
            return pd.read_parquet(file_path, engine="pyarrow")
        except Exception as e:
            print(f"⚠️  데이터 로드 실패: {e}")
            raise
    
    # 데이터 로드
    print("📊 훈련 데이터 로드 중...")
    all_train = safe_load_parquet(CFG['PATHS']['TRAIN_DATA'])
    
    print("📊 테스트 데이터 로드 중...")
    test = safe_load_parquet(CFG['PATHS']['TEST_DATA'])
    # 테스트 데이터에는 ID 컬럼이 반드시 있어야 함 (예측 시 필요)
    if 'ID' not in test.columns:
        raise ValueError("❌ 테스트 데이터에 'ID' 컬럼이 없습니다! 예측을 위해서는 ID가 필요합니다.")
    
    print(f"✅ 테스트 데이터 로드 완료: {test.shape[0]}개 행, ID 컬럼 포함")

    print("Train shape:", all_train.shape)
    print("Test shape:", test.shape)

    # feat_e_3 missing 기준으로 샘플링
    # 1. feat_e_3이 missing인 데이터는 모두 포함
    missing_feat_e_3 = all_train[all_train['feat_e_3'].isna()]
    
    # 2. feat_e_3이 있는 데이터 중에서 clicked == 1인 데이터
    available_feat_e_3_clicked_1 = all_train[(all_train['feat_e_3'].notna()) & (all_train['clicked'] == 1)]
    
    # 3. feat_e_3이 있는 데이터 중에서 clicked == 0인 데이터
    available_feat_e_3_clicked_0 = all_train[(all_train['feat_e_3'].notna()) & (all_train['clicked'] == 0)]
    
    # 4. missing 데이터에서 clicked=1 데이터를 뺀 만큼만 clicked=0 데이터에서 샘플링
    target_size = len(missing_feat_e_3) - len(available_feat_e_3_clicked_1)
    if target_size > 0 and len(available_feat_e_3_clicked_0) >= target_size:
        sampled_clicked_0 = available_feat_e_3_clicked_0.sample(n=target_size, random_state=42)
    else:
        # target_size가 0 이하이거나 available 데이터가 부족하면 모두 사용
        sampled_clicked_0 = available_feat_e_3_clicked_0
    
    # 5. 최종 훈련 데이터 구성
    train = pd.concat([missing_feat_e_3, available_feat_e_3_clicked_1, sampled_clicked_0], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    print("📊 샘플링 결과:")
    print(f"  1. feat_e_3 missing 데이터: {len(missing_feat_e_3):,}개")
    print(f"  2. feat_e_3 available + clicked=1: {len(available_feat_e_3_clicked_1):,}개")
    print(f"  3. feat_e_3 available + clicked=0 (샘플링): {len(sampled_clicked_0):,}개")
    print(f"  - 총 훈련 데이터: {len(train):,}개")
    print(f"  - 최종 clicked=0: {len(train[train['clicked']==0]):,}개")
    print(f"  - 최종 clicked=1: {len(train[train['clicked']==1]):,}개")

    # Target / Sequence
    target_col = "clicked"
    seq_col = "seq"

    # 학습에 사용할 피처: ID/seq/target 제외, 나머지 전부
    FEATURE_EXCLUDE = {target_col, seq_col, "ID"}
    feature_cols = [c for c in train.columns if c not in FEATURE_EXCLUDE]

    # 훈련 데이터에서 ID 컬럼 제거 (훈련 시에는 ID가 필요하지 않음)
    if 'ID' in train.columns:
        train = train.drop(columns=['ID'])
        print("✅ 훈련 데이터에서 ID 컬럼 제거 완료")

    print("Num features:", len(feature_cols))
    print("Sequence:", seq_col)
    print("Target:", target_col)

    return train, test, feature_cols, seq_col, target_col
