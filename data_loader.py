# Standard library
import gc
import os
import time

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
from merlin.io import Dataset as MerlinDataset
from torch.utils.data import Dataset

# Custom modules
from utils import clear_gpu_memory


# ============================================================================
# GBDT 모델용 데이터 로더 (Pre-processed data from dataset_split_and_preprocess.py)
# ============================================================================

def load_processed_data_gbdt(data_path, drop_seq=True):
    """
    Load pre-processed data from dataset_split_and_preprocess.py
    
    Args:
        data_path: Path to processed directory (e.g., 'data/proc_train_t')
        drop_seq: If True, drop 'seq' column (for GBDT models)
    
    Returns:
        X_np: numpy array of features (float32)
        y: numpy array of labels
    
    Note:
        - 반드시 dataset_split_and_preprocess.py를 먼저 실행하여 전처리 데이터 생성 필요
        - Categorical: 이미 Categorify로 인코딩됨 (0-based indices)
        - Continuous: 이미 Standardization 적용됨 (mean=0, std=1)
    """
    print(f"\n📦 Loading data from {data_path}...")
    if drop_seq:
        print("   🗑️  seq column will be dropped (GBDT mode)")
    start_load = time.time()
    
    # Pre-processed directory만 지원
    if not os.path.isdir(data_path):
        raise ValueError(
            f"❌ {data_path}는 디렉토리가 아닙니다!\n"
            "   dataset_split_and_preprocess.py를 먼저 실행하여 전처리 데이터를 생성하세요.\n"
            "   예: python dataset_split_and_preprocess.py"
        )
    
    # Pre-processed directory - load directly (FAST!)
    try:
        dataset = MerlinDataset(data_path, engine='parquet', part_size='128MB')
        print("   Converting to GPU DataFrame...")
        gdf = dataset.to_ddf().compute()
        print(f"   ✅ Loaded {len(gdf):,} rows x {len(gdf.columns)} columns")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("   Trying with smaller partitions...")
        try:
            dataset = MerlinDataset(data_path, engine='parquet', part_size='64MB')
            gdf = dataset.to_ddf().compute()
            print(f"   ✅ Loaded with 64MB partitions: {len(gdf):,} rows")
        except Exception as e2:
            print(f"❌ Failed even with 64MB partitions: {e2}")
            raise
    
    # Prepare X and y (matching train_and_predict_GBDT.py exactly)
    print("\n📊 Preparing data for GBDT...")
    if 'clicked' not in gdf.columns:
        raise ValueError("'clicked' column not found in data")
    
    y = gdf['clicked'].to_numpy()
    X = gdf.drop('clicked', axis=1)
    
    # Drop seq column if requested (for GBDT)
    if drop_seq and 'seq' in X.columns:
        X = X.drop('seq', axis=1)
        print("   ✅ Dropped 'seq' column")
    
    # Convert all features to float32 (single pass) - matching train_and_predict_GBDT.py
    print("   Converting all features to float32 (single pass)...")
    try:
        X = X.astype('float32', copy=False)
    except Exception as e:
        print(f"   ⚠️ astype(float32) failed with copy=False: {e}")
        X = X.astype('float32')
    
    # Convert to numpy
    print("   Converting to numpy...")
    X_np = X.to_numpy()
    print(f"   Shape: {X_np.shape}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]:,}")
    print(f"   Time: {time.time() - start_load:.1f}s")
    
    # Cleanup
    del X, gdf
    gc.collect()
    clear_gpu_memory()
    
    return X_np, y


# ============================================================================
# DNN 모델용 데이터 로더 (WideDeepCTR 등)
# ============================================================================

def load_processed_dnn_data(data_path):
    """Load pre-processed data from dataset_split_and_preprocess.py"""
    dataset = MerlinDataset(data_path, engine='parquet', part_size='128MB')
    gdf = dataset.to_ddf().compute()
    df = gdf.to_pandas()
    return df

class ClickDatasetDNN(Dataset):
    """DNN 모델용 데이터셋 (WideDeepCTR 등)
    
    Note:
        - Data is already preprocessed by dataset_split_and_preprocess.py
        - Categorical: already Categorify-encoded (0-based indices)
        - Numerical: already normalized (mean=0, std=1)
        - Missing values: already filled with 0
    """
    def __init__(self, df, num_cols, cat_cols, seq_col, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target
        
        # Data is already preprocessed - just convert to numpy arrays
        self.num_X = self.df[self.num_cols].fillna(0).astype(float).values
        self.cat_X = self.df[self.cat_cols].astype(int).values
        self.seq_strings = self.df[self.seq_col].astype(str).values
        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        num_x = torch.tensor(self.num_X[idx], dtype=torch.float)
        cat_x = torch.tensor(self.cat_X[idx], dtype=torch.long)
        s = self.seq_strings[idx]
        if s:
            arr = np.fromstring(s, sep=",", dtype=np.float32)
        else:
            arr = np.array([0.0], dtype=np.float32)
        seq = torch.from_numpy(arr)
        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float)
            return num_x, cat_x, seq, y
        else:
            return num_x, cat_x, seq


def collate_fn_dnn_train(batch):
    """DNN 훈련용 collate 함수"""
    num_x, cat_x, seqs, ys = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths, ys


def collate_fn_dnn_infer(batch):
    """DNN 추론용 collate 함수"""
    num_x, cat_x, seqs = zip(*batch)
    num_x = torch.stack(num_x)
    cat_x = torch.stack(cat_x)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return num_x, cat_x, seqs_padded, seq_lengths
