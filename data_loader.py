import os
import gc
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset


# ============================================================================
# GBDT 모델용 데이터 로더 (NVTabular 사용)
# ============================================================================

def create_workflow_gbdt():
    """Create NVTabular workflow optimized for GBDT models"""
    import nvtabular as nvt
    from nvtabular import ops
    
    print("\n🔧 Creating GBDT-optimized workflow...")

    # TRUE CATEGORICAL COLUMNS (only 5)
    true_categorical = ['gender', 'age_group', 'inventory_id', 'day_of_week', 'hour']

    # CONTINUOUS COLUMNS (110 total, l_feat_20, l_feat_23 제외)
    all_continuous = (
        [f'feat_a_{i}' for i in range(1, 19)] +   # 18
        [f'feat_b_{i}' for i in range(1, 7)] +    # 6
        [f'feat_c_{i}' for i in range(1, 9)] +    # 8
        [f'feat_d_{i}' for i in range(1, 7)] +    # 6
        [f'feat_e_{i}' for i in range(1, 11)] +   # 10
        [f'history_a_{i}' for i in range(1, 8)] +   # 7
        [f'history_b_{i}' for i in range(1, 31)] +  # 30
        [f'l_feat_{i}' for i in range(1, 28) if i not in [20, 23]]  # 25 (l_feat_20, l_feat_23 제외)
    )

    print(f"   Categorical: {len(true_categorical)} columns")
    print(f"   Continuous: {len(all_continuous)} columns")
    print(f"   Total features: {len(true_categorical) + len(all_continuous)}")

    # Minimal preprocessing for GBDT models
    cat_features = true_categorical >> ops.Categorify(
        freq_threshold=0,
        max_size=50000
    )
    cont_features = all_continuous >> ops.FillMissing(fill_val=0)

    workflow = nvt.Workflow(cat_features + cont_features + ['clicked'])

    print("   ✅ Workflow created (no normalization for tree models)")
    return workflow


def process_data_with_nvtabular(data_path, temp_dir='tmp'):
    """Process data with NVTabular (matching train_and_predict_GBDT.py)"""
    import shutil
    import pandas as pd
    import pyarrow.parquet as pq
    import nvtabular as nvt
    from merlin.io import Dataset
    from utils import clear_gpu_memory
    
    print("\n" + "="*70)
    print("🚀 NVTabular Data Processing")
    print("="*70)

    # Create temp directory if it doesn't exist
    os.makedirs(temp_dir, exist_ok=True)

    # Prepare data without 'seq' column
    temp_path = f'{temp_dir}/train_no_seq.parquet'
    if not os.path.exists(temp_path):
        print("\n📋 Creating temp file without 'seq' column...")
        pf = pq.ParquetFile(data_path)
        cols = [c for c in pf.schema.names if c not in ['seq', '']]
        print(f"   Total columns: {len(pf.schema.names)}")
        print(f"   Using columns: {len(cols)} (excluded 'seq')")

        df = pd.read_parquet(data_path, columns=cols)
        print(f"   Loaded {len(df):,} rows")
        df.to_parquet(temp_path, index=False)
        del df
        gc.collect()
        print("   ✅ Temp file created")
    else:
        print(f"✅ Using existing temp file: {temp_path}")

    # Create dataset with balanced partitions
    print("\n📦 Creating NVTabular Dataset...")
    print("   Using 64MB partitions for better throughput vs memory")
    clear_gpu_memory()

    dataset = Dataset(
        temp_path,
        engine='parquet',
        part_size='64MB'  # 32~64MB 권장
    )
    print("   ✅ Dataset created")

    # Create and fit workflow
    print("\n📊 Fitting workflow...")
    workflow = create_workflow_gbdt()
    workflow.fit(dataset)
    print("   ✅ Workflow fitted")

    # Transform and return processed data
    print(f"\n💾 Transforming data...")
    clear_gpu_memory()

    try:
        gdf = workflow.transform(dataset).to_ddf().compute()
        print(f"   ✅ Data processed: {len(gdf):,} rows x {len(gdf.columns)} columns")
        return gdf
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        raise


def load_processed_data_gbdt(data_path):
    """Load processed data (matching train_and_predict_GBDT.py exactly)"""
    from merlin.io import Dataset
    from utils import clear_gpu_memory
    
    print(f"\n📦 Loading data from {data_path}...")
    start_load = time.time()
    
    # Check if it's NVTabular processed data or raw parquet
    if os.path.isdir(data_path):
        # NVTabular processed directory - load directly
        try:
            dataset = Dataset(data_path, engine='parquet', part_size='128MB')
            print("   Converting to GPU DataFrame...")
            gdf = dataset.to_ddf().compute()
            print(f"   ✅ Loaded {len(gdf):,} rows x {len(gdf.columns)} columns")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("   Trying with even smaller partitions...")
            try:
                dataset = Dataset(data_path, engine='parquet', part_size='64MB')
                gdf = dataset.to_ddf().compute()
                print(f"   ✅ Loaded with 64MB partitions: {len(gdf):,} rows")
            except Exception as e2:
                print(f"❌ Failed even with 64MB partitions: {e2}")
                raise
    else:
        # Raw parquet file - process with NVTabular
        gdf = process_data_with_nvtabular(data_path)
    
    # Prepare X and y (matching train_and_predict_GBDT.py exactly)
    print("\n📊 Preparing data for GBDT...")
    if 'clicked' not in gdf.columns:
        raise ValueError("'clicked' column not found in data")
    
    y = gdf['clicked'].to_numpy()
    X = gdf.drop('clicked', axis=1)
    
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

class ClickDatasetDNN(Dataset):
    """DNN 모델용 데이터셋 (WideDeepCTR 등)"""
    def __init__(self, df, num_cols, cat_cols, seq_col, norm_stats, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target
        
        # Standardize numerical features using normalization_stats.json
        num_data = self.df[self.num_cols].astype(float)
        for col in self.num_cols:
            if col in norm_stats:
                mean = norm_stats[col]['mean']
                std = norm_stats[col]['std']
                # Avoid division by zero
                if std > 0:
                    num_data[col] = (num_data[col] - mean) / std
        self.num_X = num_data.fillna(0).values
        
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


def encode_categoricals_dnn(train_df, test_df, cat_cols):
    """범주형 피처 인코딩 (DNN용)"""
    from sklearn.preprocessing import LabelEncoder
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        all_values = pd.concat([train_df[col], test_df[col]], axis=0).astype(str).fillna("UNK")
        le.fit(all_values)
        train_df[col] = le.transform(train_df[col].astype(str).fillna("UNK"))
        test_df[col]  = le.transform(test_df[col].astype(str).fillna("UNK"))
        encoders[col] = le
        print(f"{col} unique categories: {len(le.classes_)}")
    return train_df, test_df, encoders
