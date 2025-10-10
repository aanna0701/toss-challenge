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

def load_processed_data_gbdt(data_path):
    """
    Load pre-processed data from dataset_split_and_preprocess.py
    
    GBDT models don't use the 'seq' column, so it's excluded during loading.
    
    Args:
        data_path: Path to processed directory (e.g., 'data/proc_train_t')
    
    Returns:
        X_np: numpy array of features (float32) - WITHOUT 'seq' column
        y: numpy array of labels
    
    Note:
        - 반드시 dataset_split_and_preprocess.py를 먼저 실행하여 전처리 데이터 생성 필요
        - Categorical: 이미 Categorify로 인코딩됨 (0-based indices)
        - Continuous: 이미 Standardization 적용됨 (mean=0, std=1)
        - 'seq' column is ALWAYS excluded (GBDT doesn't use sequential features)
    """
    print(f"\n📦 Loading data from {data_path} (GBDT mode - seq column will be excluded)...")
    start_load = time.time()
    
    # Pre-processed directory만 지원
    if not os.path.isdir(data_path):
        raise ValueError(
            f"❌ {data_path}는 디렉토리가 아닙니다!\n"
            "   dataset_split_and_preprocess.py를 먼저 실행하여 전처리 데이터를 생성하세요.\n"
            "   예: python dataset_split_and_preprocess.py"
        )
    
    # Pre-processed directory - Use Pandas to avoid CUDF String column size limit
    # CUDF has a 2GB limit for string columns, but 'seq' column exceeds this
    # GBDT doesn't use 'seq' column anyway, so we exclude it from the start
    try:
        import glob
        import pandas as pd
        
        # Find all parquet files in the directory
        parquet_files = glob.glob(os.path.join(data_path, "*.parquet"))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {data_path}")
        
        print(f"   Found {len(parquet_files)} parquet file(s)")
        print("   Reading with Pandas (excluding 'seq' column)...")
        
        # Read first file to get column names (excluding 'seq')
        sample_df = pd.read_parquet(parquet_files[0], engine='pyarrow')
        cols_to_read = [col for col in sample_df.columns if col != 'seq']
        del sample_df
        
        print(f"   Columns to read: {len(cols_to_read)} (seq excluded)")
        
        # Read all files with selected columns
        dfs = []
        for pf in parquet_files:
            dfs.append(pd.read_parquet(pf, columns=cols_to_read, engine='pyarrow'))
        df = pd.concat(dfs, ignore_index=True)
        
        # Convert to cudf for GPU acceleration in subsequent operations
        import cudf
        gdf = cudf.from_pandas(df)
        del df, dfs
        
        print(f"   ✅ Loaded {len(gdf):,} rows x {len(gdf.columns)} columns (seq excluded)")
    except Exception as e:
        print(f"❌ Error loading data with Pandas: {e}")
        print("   Trying original MerlinDataset method (will drop seq after loading)...")
        try:
            dataset = MerlinDataset(data_path, engine='parquet', part_size='128MB')
            print("   Converting to GPU DataFrame...")
            gdf = dataset.to_ddf().compute()
            # Drop seq if it exists
            if 'seq' in gdf.columns:
                gdf = gdf.drop('seq', axis=1)
                print("   ✅ Dropped 'seq' column")
            print(f"   ✅ Loaded {len(gdf):,} rows x {len(gdf.columns)} columns")
        except Exception as e2:
            print(f"❌ Failed with MerlinDataset: {e2}")
            raise
    
    # Prepare X and y
    print("\n📊 Preparing data for GBDT...")
    if 'clicked' not in gdf.columns:
        raise ValueError("'clicked' column not found in data")
    
    y = gdf['clicked'].to_numpy()
    X = gdf.drop('clicked', axis=1)
    
    # Sanity check: seq should already be excluded
    if 'seq' in X.columns:
        print("   ⚠️  WARNING: 'seq' column still present, dropping now...")
        X = X.drop('seq', axis=1)
    
    print(f"   ✅ Features ready: {X.shape[1]} columns (seq excluded)")
    
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
